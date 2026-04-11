"""
Idea Executor — Manages Claude Code subprocess lifecycle for idea execution.

Provides:
- Live stdout streaming to the idea's execution_log
- Discord notifications as execution progresses
- Process PID tracking for cancel/health checks
- Auto-timeout recovery (no stuck "executing" states)
- Cancel support via PID kill

The execution log is updated in real-time as Claude Code works, so
the dashboard can poll and display progress line by line.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import settings
from .models import add_comment, get_idea, mark_done, mark_executing, mark_failed

logger = logging.getLogger(__name__)

# Timeout for Claude Code execution (15 minutes)
EXECUTION_TIMEOUT: int = 900

# Bridge token file for Discord notifications
BRIDGE_TOKEN_FILE: Path = Path(__file__).parent.parent / ".bridge_token"


@dataclass
class ExecutionState:
    """Tracks a running Claude Code execution.

    Attributes:
        idea_id: The idea being executed
        pid: Claude Code subprocess PID
        started_at: Unix timestamp when execution started
        log_lines: Live buffer of stdout lines
        thread: The background thread running the execution
        cancelled: Whether cancellation was requested
    """

    idea_id: str
    pid: int | None = None
    started_at: float = field(default_factory=time.time)
    log_lines: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    cancelled: bool = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def is_alive(self) -> bool:
        """Check if the Claude Code process is still running."""
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @property
    def log_text(self) -> str:
        return "\n".join(self.log_lines)


# Active executions: idea_id -> ExecutionState
_active: dict[str, ExecutionState] = {}


def get_execution(idea_id: str) -> ExecutionState | None:
    """Get the active execution state for an idea.

    Args:
        idea_id: The idea ID

    Returns:
        ExecutionState if executing, None otherwise
    """
    return _active.get(idea_id)


def _notify_discord(message: str) -> None:
    """Send a notification to the #claude-code-updates channel via webhook.

    Falls back to the bridge API (main chat channel) if no webhook is configured.
    """
    try:
        import requests

        webhook_url = settings.discord_claude_code_webhook
        if webhook_url:
            requests.post(
                webhook_url,
                json={"content": message},
                timeout=5,
            )
            return

        # Fallback: bridge API to main channel
        if not BRIDGE_TOKEN_FILE.exists():
            return
        token = BRIDGE_TOKEN_FILE.read_text(encoding="utf-8").strip()
        requests.post(
            "http://127.0.0.1:8321/api/send",
            headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
            json={"message": message},
            timeout=5,
        )
    except Exception:
        pass


def _find_claude_binary() -> Path | None:
    """Find the Claude Code binary."""
    extensions_dir = Path.home() / ".vscode" / "extensions"
    if not extensions_dir.exists():
        return None
    candidates = sorted(
        extensions_dir.glob("anthropic.claude-code-*/resources/native-binary/claude.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def execute_idea(idea_id: str) -> ExecutionState | None:
    """Start executing an idea with Claude Code.

    Spawns a background thread that runs claude.exe and streams
    stdout line-by-line into the execution state. The dashboard
    polls this state for live updates.

    Args:
        idea_id: The idea to execute

    Returns:
        ExecutionState for tracking, or None if idea not found
    """
    idea = get_idea(idea_id)
    if not idea:
        return None

    # Don't start if already executing
    if idea_id in _active and _active[idea_id].is_alive:
        return _active[idea_id]

    mark_executing(idea_id)

    state = ExecutionState(idea_id=idea_id)
    _active[idea_id] = state

    # Build the discussion context
    discussion = ""
    if idea.comments:
        discussion = "\n\nDiscussion (context from the team):\n"
        for c in idea.comments:
            label = settings.owner_name if c.author == "owner" else "Engineer (LLM)"
            discussion += f"- {label}: {c.text}\n"

    prompt = (
        f"Implement this improvement for the Technomancer project.\n\n"
        f"Title: {idea.title}\n"
        f"Description: {idea.description}\n"
        f"{discussion}\n"
        f"MANDATORY WORKFLOW — follow these steps exactly:\n"
        f"1. cd local-agent\n"
        f"2. python safe_update.py {idea.id}\n"
        f"3. Make your code changes\n"
        f"4. python validate.py startup (MUST pass before committing)\n"
        f"5. git add <files> && git commit -m 'description'\n"
        f"6. python safe_update.py continue\n"
        f"7. python bot_service.py status (MUST show Bot running: True)\n"
        f"\nDo NOT skip any steps. Do NOT commit without validate.py passing."
    )

    def _run() -> None:
        """Background thread: spawn claude.exe and stream stdout."""
        import subprocess

        binary = _find_claude_binary()
        if not binary:
            state.log_lines.append("ERROR: Claude Code binary not found")
            mark_failed(idea_id, "Claude Code binary not found")
            _active.pop(idea_id, None)
            return

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        project_root = Path(__file__).parent.parent.parent

        _notify_discord(f"Starting execution of {idea_id}: {idea.title}")

        try:
            proc = subprocess.Popen(
                [
                    str(binary), "-p", prompt,
                    "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
                    "--max-turns", "50",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(project_root),
                env=env,
            )
            state.pid = proc.pid
            state.log_lines.append(f"Claude Code started (PID: {proc.pid})")
            state.log_lines.append(f"Working on: {idea.title}")
            state.log_lines.append("Waiting for Claude Code to complete...")
            logger.info(f"[Executor] {idea_id} started, PID {proc.pid}")

            # Monitor the process with periodic status updates.
            # claude -p buffers all stdout until exit, so we poll process
            # status instead of trying to read line-by-line.
            last_update_time = time.time()
            while proc.poll() is None:
                time.sleep(3)

                if state.cancelled:
                    proc.kill()
                    state.log_lines.append("CANCELLED by user")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(f"Execution of {idea_id} was cancelled.")
                    _active.pop(idea_id, None)
                    return

                if state.elapsed > EXECUTION_TIMEOUT:
                    proc.kill()
                    state.log_lines.append(f"TIMEOUT after {EXECUTION_TIMEOUT}s")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(f"Execution of {idea_id} timed out after {EXECUTION_TIMEOUT // 60} minutes.")
                    _active.pop(idea_id, None)
                    return

                if time.time() - last_update_time > 30:
                    state.log_lines.append(f"Still running... ({state.elapsed:.0f}s elapsed)")
                    _notify_discord(f"[{idea_id}] Still working... ({state.elapsed:.0f}s)")
                    last_update_time = time.time()

            # Process finished — read all output
            stdout_bytes = proc.stdout.read() if proc.stdout else b""
            output = stdout_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode

            if output:
                for line in output.split("\n"):
                    state.log_lines.append(line)

            if exit_code == 0:
                state.log_lines.append(f"Completed successfully (exit code 0, {state.elapsed:.0f}s)")
                mark_done(idea_id, state.log_text[-5000:])
                _notify_discord(f"Idea {idea_id} executed successfully ({state.elapsed:.0f}s): {idea.title}")
            else:
                state.log_lines.append(f"Failed (exit code {exit_code}, {state.elapsed:.0f}s)")
                mark_failed(idea_id, state.log_text[-5000:])
                _notify_discord(f"Idea {idea_id} execution failed (exit code {exit_code}): {idea.title}")

        except Exception as e:
            state.log_lines.append(f"ERROR: {e}")
            mark_failed(idea_id, state.log_text)
            _notify_discord(f"Idea {idea_id} execution error: {e}")

        finally:
            _active.pop(idea_id, None)

    thread = threading.Thread(target=_run, daemon=True, name=f"executor-{idea_id}")
    thread.start()
    state.thread = thread
    return state


def cancel_execution(idea_id: str) -> bool:
    """Cancel a running execution.

    Args:
        idea_id: The idea to cancel

    Returns:
        True if cancellation was initiated, False if not executing
    """
    state = _active.get(idea_id)
    if not state:
        return False

    state.cancelled = True

    # Also try to kill the process directly
    if state.pid and state.is_alive:
        try:
            os.kill(state.pid, signal.SIGTERM)
            logger.info(f"[Executor] Sent SIGTERM to PID {state.pid} for {idea_id}")
        except (OSError, ProcessLookupError):
            pass

    return True
