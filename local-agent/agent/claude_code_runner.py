"""
Claude Code Runner — Execute Claude Code tasks from Discord.

Spawns headless Claude Code sessions via the CLI binary to run tasks
on the local machine. Uses the Pro subscription (not API key).

Supports two modes:
  - One-shot (claudeCode): Single prompt, single response, session ends.
  - Conversational (claudeChat): Multi-turn session with --resume.
    Each reply continues the same Claude Code conversation with full
    context, tools, and file access preserved across turns.

Usage from Discord:
    claudeCode <prompt>           — one-shot task
    claudeChat <prompt>           — start a conversation
    (reply to claudeChat message) — continue the conversation
    claudeChat end                — end the active session
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root for working directory
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# Timeout for a single Claude Code turn (15 minutes)
TASK_TIMEOUT: int = 900


def _find_claude_binary() -> Path | None:
    """Find the Claude Code binary from the VS Code extension.

    Searches for the most recent version of the Claude Code extension
    installed in VS Code's extensions directory.

    Returns:
        Path to claude.exe, or None if not found
    """
    # Auto-discover VS Code extensions directory
    extensions_dir = Path.home() / ".vscode" / "extensions"
    if not extensions_dir.exists():
        return None

    candidates = sorted(
        extensions_dir.glob("anthropic.claude-code-*/resources/native-binary/claude.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else None


# ============================================================================
# CHAT SESSION MANAGEMENT
# ============================================================================

@dataclass
class ChatSession:
    """Tracks an active Claude Code conversational session.

    Attributes:
        session_id: Claude Code's internal session UUID (from --resume)
        user: Discord username who owns this session
        channel_id: Discord channel ID where the session is active
        last_message_id: Discord message ID of the last bot reply (for threading)
        started_at: Timestamp when the session was created
        turn_count: Number of turns completed in this session
        total_cost_usd: Cumulative cost across all turns
    """

    session_id: str
    user: str
    channel_id: int
    last_message_id: int = 0
    started_at: float = field(default_factory=time.time)
    turn_count: int = 0
    total_cost_usd: float = 0.0


# Active sessions: channel_id -> ChatSession
_active_sessions: dict[int, ChatSession] = {}


def get_active_session(channel_id: int) -> ChatSession | None:
    """Get the active chat session for a Discord channel.

    Args:
        channel_id: The Discord channel ID

    Returns:
        The active ChatSession, or None if no session is active
    """
    return _active_sessions.get(channel_id)


def end_session(channel_id: int) -> ChatSession | None:
    """End and remove the active session for a channel.

    Args:
        channel_id: The Discord channel ID

    Returns:
        The ended ChatSession, or None if no session was active
    """
    return _active_sessions.pop(channel_id, None)


async def run_claude_code(
    prompt: str, cwd: str | None = None, image_paths: list[str] | None = None
) -> tuple[bool, str, float]:
    """Run a Claude Code task headlessly via the CLI.

    Args:
        prompt: The task for Claude Code to execute
        cwd: Working directory (defaults to technomancer project root)
        image_paths: Optional list of local image file paths to include.
                     Claude Code's Read tool can view images, so we inject
                     instructions to read them into the prompt.

    Returns:
        Tuple of (success, output, duration_seconds)
    """
    start = time.time()

    binary = _find_claude_binary()
    if not binary:
        return False, "Error: Claude Code binary not found in VS Code extensions.", 0.0

    work_dir = cwd or str(PROJECT_ROOT)

    # If images were attached, append instructions to read them
    full_prompt = prompt
    if image_paths:
        image_instructions = "\n\nThe user attached the following image(s). Read each one:\n"
        for path in image_paths:
            image_instructions += f"- {path}\n"
        image_instructions += "\nAnalyze the image(s) as part of your task."
        full_prompt = prompt + image_instructions

    # Build environment — remove CLAUDECODE to avoid "nested session" error
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("ANTHROPIC_API_KEY", None)  # Force Pro subscription, not API credits

    try:
        proc = await asyncio.create_subprocess_exec(
            str(binary),
            "-p", full_prompt,
            "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
            "--max-turns", "50",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=TASK_TIMEOUT
        )

        duration = time.time() - start
        output = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            return True, output, duration
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return False, f"Claude Code exited with code {proc.returncode}:\n{error or output}", duration

    except asyncio.TimeoutError:
        duration = time.time() - start
        try:
            proc.kill()
        except Exception:
            pass
        return False, f"Claude Code task timed out after {TASK_TIMEOUT}s", duration
    except Exception as e:
        duration = time.time() - start
        return False, f"Error running Claude Code: {e}", duration


# ============================================================================
# CONVERSATIONAL CHAT (multi-turn via --resume)
# ============================================================================

@dataclass
class ChatResult:
    """Result from a single turn of a Claude Code chat session.

    Attributes:
        success: Whether the turn completed without errors
        response: Claude's text response
        session_id: The session UUID (use with --resume for next turn)
        duration: Wall-clock seconds for this turn
        cost_usd: API cost for this turn
        is_new_session: True if this was the first turn (new session)
    """

    success: bool
    response: str
    session_id: str
    duration: float
    cost_usd: float
    is_new_session: bool


async def run_claude_chat(
    prompt: str,
    session_id: str | None = None,
    cwd: str | None = None,
    image_paths: list[str] | None = None,
) -> ChatResult:
    """Run one turn of a Claude Code conversational session.

    If session_id is None, starts a new session. If provided, continues
    the existing session via --resume. The returned session_id should be
    stored and passed to subsequent calls for multi-turn conversation.

    Args:
        prompt: The user's message for this turn
        session_id: Previous session ID to continue (None = new session)
        cwd: Working directory (defaults to technomancer project root)
        image_paths: Optional image files to include in the prompt

    Returns:
        ChatResult with response, session_id, cost, and duration
    """
    start = time.time()

    binary = _find_claude_binary()
    if not binary:
        return ChatResult(
            success=False, response="Error: Claude Code binary not found.",
            session_id="", duration=0, cost_usd=0, is_new_session=True,
        )

    work_dir = cwd or str(PROJECT_ROOT)

    # Append image instructions if images were attached
    full_prompt = prompt
    if image_paths:
        image_instructions = "\n\nThe user attached the following image(s). Read each one:\n"
        for path in image_paths:
            image_instructions += f"- {path}\n"
        image_instructions += "\nAnalyze the image(s) as part of your task."
        full_prompt = prompt + image_instructions

    # Build environment
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("ANTHROPIC_API_KEY", None)  # Force Pro subscription, not API credits

    # Build command args
    args = [
        str(binary),
        "-p", full_prompt,
        "--output-format", "json",
        "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
        "--max-turns", "50",
    ]
    if session_id:
        args.extend(["--resume", session_id])

    is_new = session_id is None

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=TASK_TIMEOUT
        )

        duration = time.time() - start
        raw_output = stdout.decode("utf-8", errors="replace").strip()

        # Parse JSON result
        try:
            result_json = json.loads(raw_output)
            response_text = result_json.get("result", raw_output)
            new_session_id = result_json.get("session_id", session_id or "")
            cost = result_json.get("total_cost_usd", 0)
        except (json.JSONDecodeError, TypeError):
            # Fallback if JSON parsing fails — use raw text
            response_text = raw_output
            new_session_id = session_id or ""
            cost = 0

        if proc.returncode == 0:
            return ChatResult(
                success=True, response=response_text,
                session_id=new_session_id, duration=duration,
                cost_usd=cost, is_new_session=is_new,
            )
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return ChatResult(
                success=False, response=f"Error: {error or response_text}",
                session_id=new_session_id, duration=duration,
                cost_usd=cost, is_new_session=is_new,
            )

    except asyncio.TimeoutError:
        duration = time.time() - start
        try:
            proc.kill()
        except Exception:
            pass
        return ChatResult(
            success=False, response=f"Timed out after {TASK_TIMEOUT}s",
            session_id=session_id or "", duration=duration,
            cost_usd=0, is_new_session=is_new,
        )
    except Exception as e:
        duration = time.time() - start
        return ChatResult(
            success=False, response=f"Error: {e}",
            session_id=session_id or "", duration=duration,
            cost_usd=0, is_new_session=is_new,
        )
