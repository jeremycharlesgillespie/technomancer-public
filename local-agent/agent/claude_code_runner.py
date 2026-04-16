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
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import executor_runs_db
from .config import settings

logger = logging.getLogger(__name__)

# Project root for working directory
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# Legacy default. Actual executor timeout is read from
# ``settings.executor_max_runtime_seconds`` at call time so ops can tune it
# without a code change.
TASK_TIMEOUT: int = 900


async def _drain_stream(stream: Any) -> str:
    """Read any remaining bytes from a StreamReader after termination.

    Best-effort: returns "" on any error or timeout. Used to capture partial
    stdout/stderr from a subprocess we just SIGKILL'd.
    """
    if stream is None:
        return ""
    try:
        data = await asyncio.wait_for(stream.read(), timeout=1)
    except (asyncio.TimeoutError, Exception):
        return ""
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")
    return ""


async def _terminate_and_capture(
    proc: Any,
    correlation_id: str | None = None,
    grace_seconds: int | None = None,
) -> tuple[str, str]:
    """Send SIGTERM, wait for graceful exit, escalate to SIGKILL if needed.

    Captures any partial stdout/stderr from the subprocess pipes after it
    exits so operators can see what the run was doing when it hung. Never
    raises — all errors are swallowed so the calling timeout path stays
    simple.

    Args:
        proc: An ``asyncio.subprocess.Process`` (or compatible mock).
        correlation_id: Opaque ID (e.g. artifact run_id) included in log
            entries so operators can trace a timeout back to its run.
        grace_seconds: Seconds to wait after SIGTERM before escalating.
            Defaults to ``settings.executor_sigterm_grace_seconds``.

    Returns:
        ``(stdout_partial, stderr_partial)`` as decoded strings — either may
        be empty if no bytes were buffered or the read failed.
    """
    grace = (
        grace_seconds if grace_seconds is not None
        else settings.executor_sigterm_grace_seconds
    )

    try:
        proc.terminate()
    except (ProcessLookupError, OSError) as exc:
        logger.debug(
            "terminate() failed (correlation=%s): %s", correlation_id, exc
        )

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:
        logger.warning(
            "Subprocess did not exit %ds after SIGTERM — escalating to SIGKILL",
            grace,
            extra={
                "correlation_id": correlation_id,
                "event": "executor_sigkill",
            },
        )
        try:
            proc.kill()
        except (ProcessLookupError, OSError) as exc:
            logger.debug(
                "kill() failed (correlation=%s): %s", correlation_id, exc
            )
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning(
                "Subprocess still alive 5s after SIGKILL (correlation=%s)",
                correlation_id,
            )
        except Exception:  # pragma: no cover — defensive
            pass
    except Exception:  # pragma: no cover — defensive
        pass

    stdout_partial = await _drain_stream(getattr(proc, "stdout", None))
    stderr_partial = await _drain_stream(getattr(proc, "stderr", None))
    return stdout_partial, stderr_partial


def _safe_record(**fields: Any) -> int | None:
    """Record an executor-run row, swallowing all DB errors.

    Instrumentation must never crash the caller — if the SQLite layer has
    any issue (locked file, corrupt db, missing dir), we log at debug and
    return None so the run continues normally.
    """
    try:
        return executor_runs_db.record_run(**fields)
    except Exception:
        logger.debug("executor_runs_db.record_run failed", exc_info=True)
        return None


def _safe_record_tool_call(**fields: Any) -> int | None:
    """Record an executor_tool_calls row, swallowing all DB errors."""
    try:
        return executor_runs_db.record_tool_call(**fields)
    except Exception:
        logger.debug("executor_runs_db.record_tool_call failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Per-tool telemetry from stream-json events
# ---------------------------------------------------------------------------
#
# Claude Code's ``--output-format stream-json`` emits events for every tool
# call the assistant makes. Two events bracket each invocation:
#
#   {"type": "tool_use",    "tool_use_id": "toolu_abc", "name": "Bash", ...}
#   {"type": "tool_result", "tool_use_id": "toolu_abc", "content": ..., "is_error": false, ...}
#
# Real streams also wrap these inside an "assistant" / "user" message with a
# ``content`` array of blocks. The extractor below handles both shapes.
#
# Usage — pass a fresh ``pending`` dict and a known ``db_run_id``:
#
#     pending: dict[str, dict[str, Any]] = {}
#     for line in stream:
#         event = json.loads(line)
#         record_tool_events(event, db_run_id, pending)


def _iter_tool_blocks(event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(block_type, block)`` tuples from any event shape.

    Accepts the three common shapes seen in real Claude Code streams:

    * flat: ``{"type": "tool_use", ...}``
    * nested message: ``{"type": "assistant", "message": {"content": [blocks]}}``
    * result bundle: ``{"type": "result", "messages": [{"content": [blocks]}, ...]}``

    Unknown shapes return an empty list so the caller can safely ignore them.
    """
    if not isinstance(event, dict):
        return []

    evt_type = event.get("type", "")
    out: list[tuple[str, dict[str, Any]]] = []

    # Flat tool_use / tool_result — treat the event itself as the block.
    if evt_type in ("tool_use", "tool_result"):
        out.append((evt_type, event))
        return out

    # Nested: event.message.content = [blocks]
    message = event.get("message")
    if isinstance(message, dict):
        for block in message.get("content") or []:
            if isinstance(block, dict):
                bt = block.get("type", "")
                if bt in ("tool_use", "tool_result"):
                    out.append((bt, block))

    # Nested: event.content = [blocks]  (some stream variants)
    for block in event.get("content") or []:
        if isinstance(block, dict):
            bt = block.get("type", "")
            if bt in ("tool_use", "tool_result"):
                out.append((bt, block))

    return out


def _extract_tool_use_id(block: dict[str, Any]) -> str | None:
    """Find the id for a tool_use or tool_result block.

    Claude Code has used several field names across versions:
    ``tool_use_id`` (tool_result), ``id`` (tool_use), and nested
    ``tool.id`` / ``tool.use_id``.
    """
    for key in ("tool_use_id", "id"):
        val = block.get(key)
        if isinstance(val, str) and val:
            return val
    tool = block.get("tool")
    if isinstance(tool, dict):
        for key in ("id", "use_id", "tool_use_id"):
            val = tool.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _extract_tool_name(block: dict[str, Any]) -> str:
    """Best-effort extract of the tool name from a tool_use block."""
    name = block.get("name")
    if isinstance(name, str) and name:
        return name
    tool = block.get("tool")
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str) and name:
            return name
    return "unknown"


def _extract_token_counts(event: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull input/output token counts from an event's ``usage`` block.

    Returns ``(None, None)`` if no usage info is present. Callers attribute
    those to the most recently completed tool call so we surface the cost of
    the Claude turn that issued each tool invocation.
    """
    usage = None
    if isinstance(event, dict):
        usage = event.get("usage")
        if usage is None:
            message = event.get("message")
            if isinstance(message, dict):
                usage = message.get("usage")
    if not isinstance(usage, dict):
        return (None, None)
    in_tokens = usage.get("input_tokens")
    out_tokens = usage.get("output_tokens")
    return (
        int(in_tokens) if isinstance(in_tokens, int) else None,
        int(out_tokens) if isinstance(out_tokens, int) else None,
    )


def _extract_error(block: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(ok, error_message)`` for a tool_result block.

    A tool call is ``ok`` when ``is_error`` is absent or false. When an
    error is present, pull a short string from the content for the DB.
    """
    is_error = bool(block.get("is_error"))
    if not is_error:
        return (True, None)

    content = block.get("content")
    msg: str | None = None
    if isinstance(content, str):
        msg = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        msg = "\n".join(parts) if parts else None
    if msg is not None:
        msg = msg.strip()[:2000] or None
    return (False, msg)


def record_tool_events(
    event: dict[str, Any],
    db_run_id: int | None,
    pending: dict[str, dict[str, Any]],
) -> list[int]:
    """Process one Claude Code stream-json event, emit DB rows for tool calls.

    Maintains a ``pending`` dict mapping ``tool_use_id`` -> start metadata so
    a later ``tool_result`` event can be paired up and the full
    ``executor_tool_calls`` row inserted in one go. The caller owns
    ``pending`` and should reuse it across a single run.

    Usage-token blocks are attributed to the most recently *completed* tool
    call when they arrive, so operators can see both the latency and the
    Claude API spend that each tool invocation caused.

    Args:
        event: One parsed JSON event from Claude Code's stream-json output.
        db_run_id: Parent ``executor_runs.id`` — tool_call rows FK here.
            ``None`` short-circuits the function (nothing gets recorded).
        pending: State carried between events. Initialise to ``{}``.

    Returns:
        List of row ids that were inserted into ``executor_tool_calls`` by
        this event. Empty when the event didn't complete any tool calls.
    """
    if db_run_id is None or not isinstance(event, dict):
        return []

    inserted: list[int] = []
    for block_type, block in _iter_tool_blocks(event):
        if block_type == "tool_use":
            use_id = _extract_tool_use_id(block)
            if not use_id:
                continue
            pending[use_id] = {
                "tool_name": _extract_tool_name(block),
                "started_at_iso": datetime.now().isoformat(),
                "started_monotonic": time.monotonic(),
            }
        elif block_type == "tool_result":
            use_id = _extract_tool_use_id(block)
            if not use_id or use_id not in pending:
                continue
            start = pending.pop(use_id)
            duration_ms = int((time.monotonic() - start["started_monotonic"]) * 1000)
            ok, err_msg = _extract_error(block)
            row_id = _safe_record_tool_call(
                run_id=db_run_id,
                tool_name=start["tool_name"],
                started_at=start["started_at_iso"],
                duration_ms=max(duration_ms, 0),
                ok=ok,
                error_message=err_msg,
            )
            if row_id is not None:
                inserted.append(row_id)
                pending["__last_row_id__"] = {"row_id": row_id}

    # Attribute usage tokens to the most recently completed row. Usage blocks
    # typically arrive on the same event that contains the tool_result, so
    # pairing them is straightforward here.
    in_tokens, out_tokens = _extract_token_counts(event)
    last = pending.get("__last_row_id__")
    if last and (in_tokens is not None or out_tokens is not None):
        updates: dict[str, Any] = {"id": last["row_id"]}
        if in_tokens is not None:
            updates["input_tokens"] = in_tokens
        if out_tokens is not None:
            updates["output_tokens"] = out_tokens
        _safe_record_tool_call(**updates)
    return inserted


def _make_run_id(jira_key: str | None) -> str:
    """Build the sortable, greppable run_id used for artifact archive dirs.

    Format: ``YYYYMMDD-HHMMSS-<jira_key|unknown>``. Lexicographic sort
    matches chronological order, so prune_old_artifacts can keep newest N
    by sorting directory names alone.
    """
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{jira_key or 'unknown'}"


def _detect_branch(cwd: str) -> str | None:
    """Return the current git branch in ``cwd``, or None if detection fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _safe_archive(
    run_id: str, stdout: str, stderr: str, branch_name: str | None
) -> None:
    """Archive run artifacts, swallowing all errors.

    Same contract as :func:`_safe_record` — disk / git failures must never
    propagate up into the runner's main success path.
    """
    try:
        executor_runs_db.archive_run(run_id, stdout, stderr, branch_name)
    except Exception:
        logger.debug("executor_runs_db.archive_run failed", exc_info=True)


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
    prompt: str,
    cwd: str | None = None,
    image_paths: list[str] | None = None,
    jira_key: str | None = None,
) -> tuple[bool, str, float]:
    """Run a Claude Code task headlessly via the CLI.

    Args:
        prompt: The task for Claude Code to execute
        cwd: Working directory (defaults to technomancer project root)
        image_paths: Optional list of local image file paths to include.
                     Claude Code's Read tool can view images, so we inject
                     instructions to read them into the prompt.
        jira_key: Optional Jira key (e.g. ``"TK-452"``) used to build the
                  artifact run_id. ``None`` produces ``...-unknown``.

    Returns:
        Tuple of (success, output, duration_seconds)
    """
    start = time.time()
    artifact_run_id = _make_run_id(jira_key)
    db_id = _safe_record(
        run_id=artifact_run_id,
        jira_key=jira_key,
        started_at=datetime.now().isoformat(),
        status="running",
    )

    binary = _find_claude_binary()
    if not binary:
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=0,
            status="binary_not_found",
        )
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

    effective_timeout = settings.executor_max_runtime_seconds

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
            proc.communicate(), timeout=effective_timeout
        )

        duration = time.time() - start
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        output = stdout_text.strip()

        if proc.returncode == 0:
            _safe_record(
                id=db_id,
                ended_at=datetime.now().isoformat(),
                duration_ms=int(duration * 1000),
                status="success",
                exit_code=0,
            )
            _safe_archive(
                artifact_run_id, stdout_text, stderr_text,
                _detect_branch(work_dir),
            )
            return True, output, duration
        else:
            error = stderr_text.strip()
            _safe_record(
                id=db_id,
                ended_at=datetime.now().isoformat(),
                duration_ms=int(duration * 1000),
                status="failure",
                exit_code=proc.returncode,
            )
            _safe_archive(
                artifact_run_id, stdout_text, stderr_text,
                _detect_branch(work_dir),
            )
            return False, f"Claude Code exited with code {proc.returncode}:\n{error or output}", duration

    except asyncio.TimeoutError:
        duration = time.time() - start
        stdout_partial, stderr_partial = await _terminate_and_capture(
            proc, correlation_id=artifact_run_id,
        )
        logger.warning(
            "Claude Code run timed out after %ds — subprocess terminated",
            effective_timeout,
            extra={
                "correlation_id": artifact_run_id,
                "jira_key": jira_key,
                "timeout_seconds": effective_timeout,
                "event": "executor_timeout",
            },
        )
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=int(duration * 1000),
            status="timeout",
        )
        _safe_archive(
            artifact_run_id, stdout_partial, stderr_partial,
            _detect_branch(work_dir),
        )
        msg = f"Claude Code task timed out after {effective_timeout}s"
        if stdout_partial.strip():
            msg += f"\nPartial output ({len(stdout_partial)} chars captured)"
        return False, msg, duration
    except Exception as e:
        duration = time.time() - start
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=int(duration * 1000),
            status="error",
        )
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
    jira_key: str | None = None,
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
        jira_key: Optional Jira key used to build the artifact run_id

    Returns:
        ChatResult with response, session_id, cost, and duration
    """
    start = time.time()
    artifact_run_id = _make_run_id(jira_key)
    db_id = _safe_record(
        run_id=artifact_run_id,
        jira_key=jira_key,
        started_at=datetime.now().isoformat(),
        status="running",
    )

    binary = _find_claude_binary()
    if not binary:
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=0,
            status="binary_not_found",
        )
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
    effective_timeout = settings.executor_max_runtime_seconds

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=effective_timeout
        )

        duration = time.time() - start
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        raw_output = stdout_text.strip()

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
            _safe_record(
                id=db_id,
                ended_at=datetime.now().isoformat(),
                duration_ms=int(duration * 1000),
                cost_usd=float(cost) if cost else 0.0,
                status="success",
                exit_code=0,
            )
            _safe_archive(
                artifact_run_id, stdout_text, stderr_text,
                _detect_branch(work_dir),
            )
            return ChatResult(
                success=True, response=response_text,
                session_id=new_session_id, duration=duration,
                cost_usd=cost, is_new_session=is_new,
            )
        else:
            error = stderr_text.strip()
            _safe_record(
                id=db_id,
                ended_at=datetime.now().isoformat(),
                duration_ms=int(duration * 1000),
                cost_usd=float(cost) if cost else 0.0,
                status="failure",
                exit_code=proc.returncode,
            )
            _safe_archive(
                artifact_run_id, stdout_text, stderr_text,
                _detect_branch(work_dir),
            )
            return ChatResult(
                success=False, response=f"Error: {error or response_text}",
                session_id=new_session_id, duration=duration,
                cost_usd=cost, is_new_session=is_new,
            )

    except asyncio.TimeoutError:
        duration = time.time() - start
        stdout_partial, stderr_partial = await _terminate_and_capture(
            proc, correlation_id=artifact_run_id,
        )
        logger.warning(
            "Claude Code chat timed out after %ds — subprocess terminated",
            effective_timeout,
            extra={
                "correlation_id": artifact_run_id,
                "jira_key": jira_key,
                "timeout_seconds": effective_timeout,
                "event": "executor_timeout",
            },
        )
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=int(duration * 1000),
            status="timeout",
        )
        _safe_archive(
            artifact_run_id, stdout_partial, stderr_partial,
            _detect_branch(work_dir),
        )
        msg = f"Timed out after {effective_timeout}s"
        if stdout_partial.strip():
            msg += f"\nPartial output ({len(stdout_partial)} chars captured)"
        return ChatResult(
            success=False, response=msg,
            session_id=session_id or "", duration=duration,
            cost_usd=0, is_new_session=is_new,
        )
    except Exception as e:
        duration = time.time() - start
        _safe_record(
            id=db_id,
            ended_at=datetime.now().isoformat(),
            duration_ms=int(duration * 1000),
            status="error",
        )
        return ChatResult(
            success=False, response=f"Error: {e}",
            session_id=session_id or "", duration=duration,
            cost_usd=0, is_new_session=is_new,
        )


# ============================================================================
# SHARED UTILITY: Simple claude -p prompt (Pro subscription, no API credits)
# ============================================================================


def run_claude_prompt(
    prompt: str,
    timeout: int = 60,
    max_turns: int = 1,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run a simple claude -p prompt synchronously using Pro subscription.

    This is the shared utility for replacing Anthropic API calls with
    claude -p subprocess calls. Uses --output-format json and strips
    ANTHROPIC_API_KEY so Claude Code uses OAuth/Pro subscription.

    Args:
        prompt: The prompt text to send
        timeout: Max seconds to wait (default 60)
        max_turns: Max conversation turns (default 1 for simple queries)
        cwd: Working directory (defaults to project root)

    Returns:
        Dict with keys: success (bool), result (str), cost_usd (float),
        session_id (str), duration (float), error (str or None)
    """
    binary = _find_claude_binary()
    if not binary:
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": 0,
            "error": "Claude Code binary not found",
        }

    work_dir = cwd or str(PROJECT_ROOT)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("ANTHROPIC_API_KEY", None)

    start = time.time()

    try:
        result = subprocess.run(
            [
                str(binary),
                "-p", prompt,
                "--output-format", "json",
                "--max-turns", str(max_turns),
            ],
            capture_output=True,
            text=True,
            cwd=work_dir,
            env=env,
            timeout=timeout,
        )

        duration = time.time() - start
        raw = result.stdout.strip()

        if result.returncode != 0:
            error_text = result.stderr.strip() or raw or f"Exit code {result.returncode}"
            return {
                "success": False,
                "result": error_text,
                "cost_usd": 0,
                "session_id": "",
                "duration": duration,
                "error": error_text,
            }

        # Parse JSON result
        try:
            data = json.loads(raw)
            return {
                "success": True,
                "result": data.get("result", raw),
                "cost_usd": data.get("total_cost_usd", 0),
                "session_id": data.get("session_id", ""),
                "duration": duration,
                "error": None,
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "success": True,
                "result": raw,
                "cost_usd": 0,
                "session_id": "",
                "duration": duration,
                "error": None,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": time.time() - start,
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": time.time() - start,
            "error": str(e),
        }


async def run_claude_prompt_async(
    prompt: str,
    timeout: int = 60,
    max_turns: int = 1,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Async version of run_claude_prompt for use in async contexts.

    Same interface as run_claude_prompt but uses asyncio subprocess.

    Args:
        prompt: The prompt text to send
        timeout: Max seconds to wait (default 60)
        max_turns: Max conversation turns (default 1)
        cwd: Working directory (defaults to project root)

    Returns:
        Same dict as run_claude_prompt.
    """
    binary = _find_claude_binary()
    if not binary:
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": 0,
            "error": "Claude Code binary not found",
        }

    work_dir = cwd or str(PROJECT_ROOT)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("ANTHROPIC_API_KEY", None)

    start = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            str(binary),
            "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(max_turns),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        duration = time.time() - start
        raw = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace").strip() or raw
            return {
                "success": False,
                "result": error_text,
                "cost_usd": 0,
                "session_id": "",
                "duration": duration,
                "error": error_text,
            }

        try:
            data = json.loads(raw)
            return {
                "success": True,
                "result": data.get("result", raw),
                "cost_usd": data.get("total_cost_usd", 0),
                "session_id": data.get("session_id", ""),
                "duration": duration,
                "error": None,
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "success": True,
                "result": raw,
                "cost_usd": 0,
                "session_id": "",
                "duration": duration,
                "error": None,
            }

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": time.time() - start,
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "success": False,
            "result": "",
            "cost_usd": 0,
            "session_id": "",
            "duration": time.time() - start,
            "error": str(e),
        }
