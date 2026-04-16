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

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import settings

from board import get_provider as _get_board_provider


def get_idea(idea_id):
    return _get_board_provider().get(idea_id)


def load_ideas():
    return _get_board_provider().load_all()


def mark_executing(idea_id):
    return _get_board_provider().mark_executing(idea_id)


def mark_done(idea_id, execution_log):
    result = _get_board_provider().mark_done(idea_id, execution_log)
    _write_done_sentinel(idea_id, "done")
    return result


def mark_failed(idea_id, error):
    result = _get_board_provider().mark_failed(idea_id, error)
    _write_done_sentinel(idea_id, "failed")
    return result


def _write_done_sentinel(idea_id: str, final_state: str) -> None:
    """Write ``execution_logs/<idea_id>.done`` containing the final idea state.

    The cross-process SSE streamer in web.py watches for this file to know
    when to emit the terminal ``done`` event and close the connection.
    Best-effort — IO errors are swallowed so they can't crash the deploy.
    """
    try:
        EXECUTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = EXECUTION_LOGS_DIR / f"{idea_id}.done"
        path.write_text(final_state.strip() + "\n", encoding="utf-8")
    except Exception as exc:
        logger.debug(
            "[Executor] Failed to write done sentinel for %s: %s", idea_id, exc,
        )


def get_execution_order(idea_id):
    return _get_board_provider().get_execution_order(idea_id)


def _post_deploy_comment(idea_id: str, short_sha: str) -> None:
    """Post a ``[Deployed] <sha> at <timestamp>`` comment on the board item.

    Called after a successful merge lands on main so the Jira issue records
    which commit shipped and when — separate from the Jira status transition
    time (which can lag). Sits alongside the existing ``[Execution Log]``
    comment that ``mark_done`` already posts.

    Failure-safe: a Jira hiccup (401, timeout, etc.) is logged as a warning
    and swallowed so the deploy still reports success.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = f"[Deployed] {short_sha} at {timestamp}"
        _get_board_provider().add_comment(idea_id, author="executor", text=text)
    except Exception as exc:
        logger.warning(
            "[Executor] Failed to post deploy comment for %s: %s", idea_id, exc
        )


def _sync_progress_comment(idea_id, state) -> None:
    """Push the tail of an execution's log into a single Jira progress comment.

    The provider's ``append_progress_comment`` edits the same comment in
    place after the first call, so a 30-minute run does not generate 30
    separate comments. Silently no-op when the provider doesn't support it
    (e.g. LocalProvider).

    The body leads with a ``Live log: <url>`` line pointing at the hub's
    ``/live/<idea_id>`` SSE viewer so anyone reading the Jira issue can jump
    straight to the running stream. Jira ADF auto-links bare URLs.
    """
    try:
        provider = _get_board_provider()
        appender = getattr(provider, "append_progress_comment", None)
        if appender is None:
            return
        recent = "\n".join(state.log_lines[-30:])
        if not recent.strip():
            return
        live_url = f"http://{settings.server_host}:8322/live/{idea_id}"
        body = f"Live log: {live_url}\n\n{recent}"
        appender(idea_id, body)
    except Exception as exc:
        logger.debug("[Executor] progress comment sync failed for %s: %s", idea_id, exc)

logger = logging.getLogger(__name__)

# Timeout for Claude Code execution (30 minutes — includes safe_update workflow)
EXECUTION_TIMEOUT: int = 1800


# Timeout for pytest (10 minutes)
PYTEST_TIMEOUT: int = 600

# Max retries when tests/validation fail — Claude gets to fix its own bugs
MAX_FIX_RETRIES: int = 5

# File to persist known failures between executor runs (replaces baseline)
KNOWN_FAILURES_FILE: Path = Path(__file__).parent / ".known_test_failures.json"

# Per-execution streaming log files. One file per idea — the cross-process
# live log viewer tails these instead of poking at the in-memory ExecutionState.
EXECUTION_LOGS_DIR: Path = Path(__file__).parent / "execution_logs"

# Minimum seconds between Discord webhook sends (rate limiting)
DISCORD_RATE_LIMIT: float = 10.0

# Interval between Jira progress-comment updates during a running execution.
# The provider edits a single per-issue comment in place, so Jira doesn't
# get spammed with one comment per minute over a 30-minute run.
JIRA_PROGRESS_INTERVAL: float = 60.0

# Category-specific implementation guidance
CATEGORY_GUIDANCE: dict[str, str] = {
    "performance": (
        "- Measure baseline metrics BEFORE making changes (use profiler.py or time commands)\n"
        "- Profile with agent/profiler.py to identify bottlenecks\n"
        "- Include before/after numbers in the commit message\n"
        "- Avoid premature optimization — measure first, optimize what matters"
    ),
    "feature": (
        "- Add unit tests in tests/unit/test_<module>.py\n"
        "- Follow existing test patterns from tests/conftest.py fixtures "
        "(mock_ollama_client, temp_vault, etc.)\n"
        "- Register new tools in discord_memory_bot.py on_ready() if applicable\n"
        "- Use create_tool() from agent/core.py for new tools"
    ),
    "quality": (
        "- Focus on readability and maintainability\n"
        "- Keep files under 1000 lines — extract utilities if needed\n"
        "- Run the FULL test suite, not just new tests\n"
        "- Don't add features — stick to the quality improvement scope"
    ),
    "security": (
        "- Check OWASP top 10 vulnerabilities\n"
        "- Validate all external input at system boundaries\n"
        "- Never hardcode secrets — use settings from agent/config.py\n"
        "- Check for command injection in any shell/subprocess calls"
    ),
    "ux": (
        "- Test the change from the Discord user's perspective\n"
        "- Ensure error messages are helpful and actionable\n"
        "- Keep Discord messages under 2000 chars\n"
        "- Use validate_discord_message() from message_validators.py"
    ),
}

# Bridge token file for Discord notifications
BRIDGE_TOKEN_FILE: Path = Path(__file__).parent.parent / ".bridge_token"

# Substrings we treat as evidence of a Claude usage/rate-limit outage.
# Matched case-insensitively against the full execution log. Kept as module-
# level so tests (and callers that want to extend the set) can import it.
RATE_LIMIT_KEYWORDS: tuple[str, ...] = (
    "rate_limit",
    "rate limit",
    "usage limit",
    "usage has exceeded",
    "429",
    "overloaded",
    "billing",
    "credit limit",
    "credit balance",
    "quota",
)

# Claude exits this quickly when it never got past the auth/preflight stage
# (rate-limited, credits exhausted). Real failures produce far more log lines
# because the stream-json protocol emits an event per tool call / assistant
# message.
RATE_LIMIT_MAX_LOG_LINES: int = 20


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
        rate_limited: Set when the Claude subprocess was classified as hitting
            a usage/rate limit. The worker inspects this after the execution
            thread exits and triggers its retry loop instead of calling
            mark_failed.
    """

    idea_id: str
    pid: int | None = None
    started_at: float = field(default_factory=time.time)
    log_lines: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    cancelled: bool = False
    baseline_failures: set[str] = field(default_factory=set)
    rate_limited: bool = False

    def log(self, msg: str) -> None:
        """Append a timestamped message to the execution log."""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        _append_execution_log_line(self.idea_id, line)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def is_alive(self) -> bool:
        """Check if the executor thread or Claude process is still running."""
        if self.thread and self.thread.is_alive():
            return True
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


def _append_execution_log_line(idea_id: str, line: str) -> None:
    """Append ``line + "\\n"`` to ``execution_logs/<idea_id>.log``.

    The viewer in the next story tails these files across processes, so we
    open-write-close per line — every line is flushed before the call
    returns. Best-effort: IO errors are swallowed because losing a log line
    should never crash an execution.
    """
    try:
        EXECUTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = EXECUTION_LOGS_DIR / f"{idea_id}.log"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.debug(
            "[Executor] Failed to write execution log for %s: %s", idea_id, exc,
        )


def _prune_stale_execution_logs() -> None:
    """Delete ``execution_logs/*.log`` and ``*.done`` files whose stem is not
    in ``_active``.

    Called once at module import so the directory doesn't grow unboundedly
    across executor restarts. At import time ``_active`` is empty, so this
    effectively clears every leftover file — correct, because reaching module
    load means no execution from a prior process can still be running here.
    """
    try:
        if not EXECUTION_LOGS_DIR.exists():
            return
        for artifact in EXECUTION_LOGS_DIR.iterdir():
            if artifact.suffix not in (".log", ".done"):
                continue
            if artifact.stem not in _active:
                try:
                    artifact.unlink()
                except OSError as exc:
                    logger.debug(
                        "[Executor] Could not remove stale artifact %s: %s",
                        artifact, exc,
                    )
    except Exception as exc:
        logger.debug("[Executor] Failed to prune execution logs: %s", exc)


def _clear_execution_artifacts(idea_id: str) -> None:
    """Remove any leftover ``.log`` / ``.done`` files for ``idea_id``.

    Called when a fresh execution starts for an idea so a stale sentinel
    from a prior run can't trick the SSE streamer into emitting ``done``
    immediately.
    """
    for suffix in (".log", ".done"):
        path = EXECUTION_LOGS_DIR / f"{idea_id}{suffix}"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug(
                "[Executor] Could not remove %s: %s", path, exc,
            )


_prune_stale_execution_logs()


def get_execution(idea_id: str) -> ExecutionState | None:
    """Get the active execution state for an idea.

    Args:
        idea_id: The idea ID

    Returns:
        ExecutionState if executing, None otherwise
    """
    return _active.get(idea_id)


def is_any_executing() -> bool:
    """Check if any idea is currently being executed."""
    return any(state.is_alive for state in _active.values())


def get_active_execution_ids() -> list[str]:
    """Get IDs of all currently executing ideas."""
    return [idea_id for idea_id, state in _active.items() if state.is_alive]


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


def _post_execution_log_to_jira(idea_id: str, state: ExecutionState) -> None:
    """Post the full execution log as a Jira comment for traceability."""
    try:
        from .jira_sync import _api, find_jira_issue, is_jira_configured

        if not is_jira_configured():
            return

        jira_key = find_jira_issue(idea_id)
        if not jira_key:
            return

        idea = get_idea(idea_id)
        idea_state = idea.state if idea else "unknown"
        log_text = state.log_text[-15000:]  # Cap at 15K chars

        # Jira code blocks have a limit — truncate if needed
        comment_adf = {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": f"Execution {idea_state}",
                         "marks": [{"type": "strong"}]},
                        {"type": "text",
                         "text": f" ({state.elapsed:.0f}s)" if state.elapsed else ""},
                    ],
                },
                {
                    "type": "codeBlock",
                    "attrs": {"language": "text"},
                    "content": [
                        {"type": "text", "text": log_text[-10000:]},
                    ],
                },
            ],
        }

        _api("post", f"/issue/{jira_key}/comment", json={"body": comment_adf})
        logger.info("[Executor] Posted execution log to %s", jira_key)
    except Exception as e:
        logger.warning("[Executor] Failed to post log to Jira: %s", e)


def _snapshot_system_load() -> str:
    """Capture a one-line summary of system load for diagnostics."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        procs = {
            "python": 0,
            "claude": 0,
            "ollama": 0,
            "total": len(list(psutil.process_iter())),
        }
        for p in psutil.process_iter(["name"]):
            name = (p.info["name"] or "").lower()
            if "python" in name:
                procs["python"] += 1
            elif "claude" in name:
                procs["claude"] += 1
            elif "ollama" in name:
                procs["ollama"] += 1
        return (
            f"cpu={cpu}% mem={mem.percent}% "
            f"py={procs['python']} claude={procs['claude']} "
            f"ollama={procs['ollama']} total={procs['total']}"
        )
    except Exception as e:
        return f"(load snapshot failed: {e})"


def _run_pytest_with_progress(
    cmd: list[str],
    cwd: str,
    state: ExecutionState,
    label: str,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run pytest as a subprocess, streaming progress lines to the execution log.

    Reads stdout line by line so the execute page shows live updates
    instead of a blank screen for minutes.

    Returns a CompletedProcess-like result with stdout captured.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd,
    )
    start = time.time()
    stdout_lines: list[str] = []

    while True:
        if time.time() - start > timeout:
            proc.kill()
            raise subprocess.TimeoutExpired(cmd, timeout)

        raw = proc.stdout.readline() if proc.stdout else b""
        if not raw:
            if proc.poll() is not None:
                break
            continue

        line = raw.decode("utf-8", errors="replace").rstrip()
        stdout_lines.append(line)

        # Stream every non-empty line — the user wants to see activity
        if line.strip():
            state.log(f"[{label}] {line.strip()}")

    # Drain remaining
    rest = proc.stdout.read() if proc.stdout else b""
    if rest:
        stdout_lines.extend(rest.decode("utf-8", errors="replace").split("\n"))

    proc.wait()
    stdout = "\n".join(stdout_lines)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr="")


def _find_related_tests(project_root: str | Path) -> list[str]:
    """Find test files related to changed source files on the current branch.

    Compares HEAD against main to get changed .py files, then maps each
    to its corresponding test file(s) using naming convention:
        agent/foo.py -> tests/unit/test_foo.py, tests/unit/test_foo_extended.py
        idea_board/bar.py -> tests/unit/test_bar.py

    Returns:
        List of existing test file paths (relative to local-agent/)
    """
    local_agent = Path(project_root) / "local-agent"
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "main...HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(project_root),
        )
        changed = [
            f for f in diff.stdout.strip().split("\n")
            if f.startswith("local-agent/") and f.endswith(".py")
        ]
    except Exception:
        return []

    test_files: list[str] = []
    for filepath in changed:
        # Strip prefix: local-agent/agent/foo.py -> agent/foo.py
        rel = filepath.replace("local-agent/", "", 1)
        parts = Path(rel)
        module_name = parts.stem  # foo

        # Look for test_foo.py and test_foo_extended.py
        for pattern in [f"test_{module_name}.py", f"test_{module_name}_extended.py"]:
            test_path = local_agent / "tests" / "unit" / pattern
            if test_path.exists():
                test_files.append(str(test_path.relative_to(local_agent)))

        # If the changed file IS a test file, include it directly
        if "tests/" in rel and rel.endswith(".py"):
            full = local_agent / rel
            if full.exists() and str(full.relative_to(local_agent)) not in test_files:
                test_files.append(str(full.relative_to(local_agent)))

    return sorted(set(test_files))


def _parse_pytest_failures(output: str) -> set[str]:
    """Parse pytest output for FAILED test node IDs.

    Looks for lines like:
        FAILED tests/unit/test_core.py::test_something - AssertionError: ...
        FAILED tests/unit/test_foo.py::TestBar::test_baz

    Returns:
        Set of test node IDs (e.g. "tests/unit/test_foo.py::TestBar::test_baz")
    """
    failures: set[str] = set()
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("FAILED "):
            # Format: "FAILED test_id" or "FAILED test_id - error description"
            rest = line[7:]  # Remove "FAILED "
            test_id = rest.split(" - ")[0].strip()
            if test_id:
                failures.add(test_id)
    return failures


def _load_known_failures() -> set[str]:
    """Load known test failures from the last successful full suite run."""
    if KNOWN_FAILURES_FILE.exists():
        try:
            data = json.loads(KNOWN_FAILURES_FILE.read_text())
            return set(data.get("failures", []))
        except Exception:
            pass
    return set()


def _save_known_failures(failures: set[str]) -> None:
    """Save test failures from the full suite for future comparison."""
    KNOWN_FAILURES_FILE.write_text(json.dumps({
        "failures": sorted(failures),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))


def _has_branch_commits(project_root: Path, base: str = "main") -> bool:
    """Return True if the current branch has commits not already on ``base``.

    Used as a rate-limit signal: when ``claude -p`` bails out without writing
    any code, the branch will have zero commits past main, so we can tell
    "Claude hit its limit" apart from "Claude actually failed".
    """
    try:
        result = subprocess.run(
            ["git", "log", f"{base}..HEAD", "--oneline"],
            capture_output=True, text=True, timeout=10,
            cwd=str(project_root),
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _classify_rate_limit(
    state: ExecutionState,
    claude_succeeded: bool,
    project_root: Path,
    branch_base: str = "main",
) -> bool:
    """Return True when the Claude run matches a usage/rate-limit outage.

    All three signals must fire, per the TK-410 design:

    1. Claude did not finish cleanly (no ``result`` stream event) AND the
       total log line count is small (≲ :data:`RATE_LIMIT_MAX_LOG_LINES`).
    2. The combined log text contains at least one keyword from
       :data:`RATE_LIMIT_KEYWORDS`.
    3. The branch has no commits past ``branch_base`` — i.e. Claude wrote
       nothing, so the "success" wasn't masking real work.

    Returning False here sends the caller down the normal failure path
    (``mark_failed``), so being conservative is correct: we only claim
    rate-limited when we are highly confident.
    """
    if claude_succeeded:
        return False
    if len(state.log_lines) >= RATE_LIMIT_MAX_LOG_LINES:
        return False
    text = state.log_text.lower()
    if not any(kw in text for kw in RATE_LIMIT_KEYWORDS):
        return False
    if _has_branch_commits(project_root, branch_base):
        return False
    return True


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


# ---------------------------------------------------------------------------
# Prompt builders — assemble rich context for Claude Code
# ---------------------------------------------------------------------------


def _build_discussion(idea: Any) -> str:
    """Format discussion thread from idea comments."""
    if not idea.comments:
        return ""
    lines = ["\n## Discussion (what was decided)"]
    for c in idea.comments:
        label = f"{settings.owner_name} (manager)" if c.author == "owner" else "LLM (engineer)"
        lines.append(f"- {label}: {c.text}")
    return "\n".join(lines)


def _build_epic_context(idea: Any) -> str:
    """Build parent epic and sibling context for a story."""
    if not idea.parent_id:
        return ""
    parent = get_idea(idea.parent_id)
    if not parent:
        return ""

    all_ideas = load_ideas()
    siblings = [i for i in all_ideas if i.parent_id == idea.parent_id]

    lines = [
        f"\n## Parent Epic: {parent.title}",
        f"**Epic Description:** {parent.description}",
        "",
        "**Stories in this epic:**",
    ]
    for s in siblings:
        if s.id == idea.id:
            lines.append(f"  - **[THIS] {s.id}: {s.title}** <-- you are implementing this one")
        elif s.state == "done":
            lines.append(f"  - [DONE] {s.id}: {s.title}")
        else:
            lines.append(f"  - {s.id}: {s.title}")
    lines.append(
        "\nBuild on what the completed stories created. "
        "Ensure your implementation integrates with the epic's full lifecycle goal."
    )
    return "\n".join(lines)


def _build_children_context(idea: Any) -> str:
    """Build child story list for an epic."""
    all_ideas = load_ideas()
    kids = [i for i in all_ideas if i.parent_id == idea.id]
    if not kids:
        return ""
    lines = ["\n**Stories in this epic:**"]
    for k in kids:
        done_marker = " [DONE]" if k.state == "done" else ""
        lines.append(f"  - {k.id}: {k.title}{done_marker}")
    return "\n".join(lines)


def _build_epic_execution_context(
    epic: Any, previous_results: list[dict[str, str]]
) -> str:
    """Build context from epic narrative and previous story results.

    Used by execute_epic() to give each story awareness of the epic's
    big-picture goal and what prior stories accomplished.

    Args:
        epic: The parent epic Idea object
        previous_results: List of dicts with id, title, state, summary keys

    Returns:
        Context string, or empty string if nothing to inject
    """
    return _format_injected_epic_context(
        epic.epic_context or "", previous_results
    )


def _format_injected_epic_context(
    epic_context: str,
    previous_results: list[dict[str, str]] | None,
) -> str:
    """Format epic context and previous story results for prompt injection.

    Called by _build_story_prompt() when executing stories within an epic.
    Provides the executing Claude session with the epic's big-picture goal
    and a summary of what previous stories accomplished.

    Args:
        epic_context: Free-text narrative for the epic's overall goal
        previous_results: List of dicts with id, title, state, summary keys

    Returns:
        Formatted context string, or empty string if nothing to inject
    """
    lines: list[str] = []

    if epic_context:
        lines.append("## Epic Context")
        lines.append(epic_context)
        lines.append("")

    if previous_results:
        done = [r for r in previous_results if r.get("state") == "done"]
        if done:
            lines.append("## Previous Stories (already completed)")
            for r in done:
                entry = f"- **{r['id']}**: {r['title']} [DONE]"
                if r.get("summary"):
                    entry += f"\n  Result: {r['summary'][:500]}"
                lines.append(entry)
            lines.append("")
            lines.append(
                "Build on what these stories created. Do not duplicate their work."
            )

    return "\n".join(lines)


def _load_codebase_summary() -> str:
    """List all Python files in agent/ and idea_board/ with their first docstring line."""
    agent_dir = Path(__file__).parent.parent / "agent"
    lines = []
    for f in sorted(agent_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        desc = ""
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if '"""' in content:
                doc_start = content.index('"""') + 3
                doc_end = content.index('"""', doc_start)
                first_line = content[doc_start:doc_end].strip().split("\n")[0]
                desc = f" — {first_line}"
        except (ValueError, OSError):
            pass
        lines.append(f"- {f.name}{desc}")

    board_dir = Path(__file__).parent
    for f in sorted(board_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        lines.append(f"- idea_board/{f.name}")
    return "\n".join(lines)


def _find_relevant_test_file(idea: Any) -> str:
    """Detect which module the idea targets and include its test file as reference.

    Scans the idea description for module names (e.g. "agent/foo.py", "foo.py",
    or bare names matching files in agent/). If a corresponding test file exists,
    includes its first 80 lines. Falls back to conftest.py fixtures if no match.
    """
    import re

    tests_dir = Path(__file__).parent.parent / "tests" / "unit"
    agent_dir = Path(__file__).parent.parent / "agent"
    conftest = Path(__file__).parent.parent / "tests" / "conftest.py"

    description = f"{idea.title} {idea.description}"

    # Strategy 1: Look for explicit file references like "agent/foo.py" or "foo.py"
    file_refs = re.findall(r"(?:agent/)?(\w+)\.py", description)

    # Strategy 2: Look for module-like words that match actual agent/*.py files
    agent_modules = {f.stem for f in agent_dir.glob("*.py") if not f.name.startswith("_")}

    # Score candidates by how likely they are the target module
    candidates: list[str] = []
    for ref in file_refs:
        if ref in agent_modules and ref not in ("__init__", "config", "core"):
            candidates.append(ref)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # Try to find a matching test file
    for module_name in unique[:3]:
        test_file = tests_dir / f"test_{module_name}.py"
        if test_file.exists():
            try:
                lines = test_file.read_text(encoding="utf-8", errors="replace").split("\n")
                snippet = "\n".join(lines[:80])
                if len(snippet) > 2000:
                    snippet = snippet[:2000] + "\n... (truncated)"
                return (
                    f"\n## Test Pattern Reference (from test_{module_name}.py)\n"
                    f"```python\n{snippet}\n```"
                )
            except OSError:
                continue

    # Fallback: show conftest.py fixtures
    if conftest.exists():
        try:
            lines = conftest.read_text(encoding="utf-8", errors="replace").split("\n")
            snippet = "\n".join(lines[:50])
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "\n... (truncated)"
            return (
                f"\n## Test Pattern Reference (from conftest.py — available fixtures)\n"
                f"```python\n{snippet}\n```"
            )
        except OSError:
            pass

    return ""


def _get_category_guidance(category: str) -> str:
    """Get category-specific implementation guidance."""
    guidance = CATEGORY_GUIDANCE.get(category, "")
    if not guidance:
        return ""
    return f"\n## Category Guidance ({category})\n{guidance}"


def _build_workflow_section(idea: Any) -> str:
    """Build the mandatory workflow and completion instructions."""
    return (
        "\n## MANDATORY WORKFLOW\n"
        "The branch has ALREADY been created for you. You are already on it.\n\n"
        "Follow these steps EXACTLY:\n"
        "1. Read CLAUDE.md for project conventions\n"
        "2. `cd local-agent`\n"
        "3. Make your code changes (with tests if adding new functionality)\n"
        f"4. `git add <files>` && `git commit -m '[{idea.id}] description'`\n"
        f"\n**Every commit message MUST start with `[{idea.id}]`.**\n"
        "\n**YOUR JOB IS DONE AFTER COMMITTING.**\n"
        "\nDo NOT run `safe_update.py` — it is blocked in this environment.\n"
        "Do NOT run `validate.py` — the executor runs it after you finish.\n"
        "Do NOT run `pytest` — the executor runs it after you finish.\n"
        "Do NOT try to deploy, merge, or restart anything.\n"
        "Do NOT call add_idea() or create ideas in production code — only in tests.\n"
        "\nJust write code, write tests, and commit. The executor handles the rest.\n"
        "\n## TESTING RULES\n"
        "Tests use `@patch('agent.module_name.thing')` to mock dependencies.\n"
        "This ONLY works if `thing` is imported at module level.\n"
        "**Any dependency your code uses that tests will need to mock MUST be\n"
        "imported at the top of the file, not inside a function body.**\n"
        "Example — WRONG: `def foo(): from .config import settings`\n"
        "Example — RIGHT: top of file has `from .config import settings`\n"
        "\n## COMMON TEST PITFALLS IN THIS CODEBASE\n"
        "These are real failures from previous executor runs. Check each one:\n\n"
        "1. **Mock patch targets must exist at module level.** If you patch\n"
        "   `agent.web_search.settings` but web_search.py imports settings\n"
        "   inside a function, the patch fails with AttributeError.\n\n"
        "2. **Flask content_type includes charset.** Flask responses have\n"
        "   `content_type = 'text/event-stream; charset=utf-8'`, not bare\n"
        "   `'text/event-stream'`. Use `'text/event-stream' in resp.content_type`\n"
        "   instead of `==`.\n\n"
        "3. **aiohttp mocks need async context managers.** `async with session.get()`\n"
        "   requires the mock to have `__aenter__`/`__aexit__`. Wrap mock responses:\n"
        "   `cm = AsyncMock(); cm.__aenter__ = AsyncMock(return_value=resp)`\n\n"
        "4. **Windows path separators.** `Path.relative_to()` returns backslashes\n"
        "   on Windows. Use `result.replace('\\\\', '/')` for cross-platform assertions.\n\n"
        "5. **os.kill(pid, 0) doesn't work on Windows.** Use ctypes OpenProcess\n"
        "   or subprocess-based checks instead.\n\n"
        "6. **StopIteration in async (Python 3.14).** Raising StopIteration inside\n"
        "   a coroutine becomes RuntimeError. Use KeyboardInterrupt or a custom\n"
        "   exception to break async loops in tests.\n"
    )


def _build_prior_failure_context(idea: Any) -> str:
    """Inject the most-recent [Execution Log - Failed] comment into the prompt.

    Lets a retried story learn from its previous attempt. Pulls the comment
    stream from the active BoardProvider, filters to the ``Execution Log -
    Failed`` marker, and renders the newest one as a ``Prior Failure
    Context`` section. Returns an empty string when no such comment exists
    so the caller can omit the section entirely.

    The failure log is truncated to ~4000 chars from the front — the tail
    typically contains the actual traceback, which is the useful part.
    """
    max_failure_chars = 4000
    try:
        provider = _get_board_provider()
        getter = getattr(provider, "get_comments", None)
        if getter is None:
            return ""
        comments = getter(idea.id) or []
    except Exception as exc:
        logger.debug(
            "[Executor] get_comments for %s failed: %s", getattr(idea, "id", "?"), exc,
        )
        return ""

    failures = [c for c in comments if getattr(c, "marker", None) == "[Execution Log - Failed]"]
    if not failures:
        return ""

    latest = failures[-1]
    body = (latest.text or "").strip()
    if len(body) > max_failure_chars:
        body = "... (truncated)\n" + body[-max_failure_chars:]

    return (
        "\n## Prior Failure Context\n"
        "The previous attempt failed with the log below. Read it carefully "
        "and avoid repeating the same mistakes. Address the root cause "
        "before continuing.\n\n"
        f"```\n{body}\n```"
    )


def _enrich_stub_description(idea: Any) -> str:
    """If a story has only a stub description, pull the parent epic's full description.

    Auto-generated stories from idea_generator get placeholder descriptions like
    "Story under epic: <title>". These are useless for implementation. When detected,
    we pull the parent epic's full WHAT/WHY/HOW description and prepend it.
    """
    desc = idea.description or ""
    is_stub = (
        desc.startswith("Story under epic:")
        or len(desc.strip()) < 80
    )
    if not is_stub or not idea.parent_id:
        return desc

    parent = get_idea(idea.parent_id)
    if not parent or not parent.description:
        return desc

    return (
        f"**This story is part of:** {parent.title}\n\n"
        f"**Epic context (use this to guide your implementation):**\n"
        f"{parent.description}\n\n"
        f"**Your specific task:** {idea.title}\n"
        f"{desc}"
    )


def _build_story_prompt(
    idea: Any,
    epic_context: str = "",
    previous_results: list[dict[str, str]] | None = None,
) -> str:
    """Build a rich prompt for executing a single story/task.

    Args:
        idea: The Idea object to build a prompt for
        epic_context: Optional epic narrative injected during epic execution
        previous_results: Optional list of prior story results (id, title, state, summary)
    """
    type_label = f"[{idea.idea_type.upper()}] " if idea.idea_type != "story" else ""
    description = _enrich_stub_description(idea)

    sections = [
        f"# Task: Implement {idea.title}\n",
        f"You are implementing a {idea.idea_type} for the Technomancer project.\n",
        f"## {type_label}Idea Details",
        f"- **ID:** {idea.id}",
        f"- **Category:** {idea.category}",
        f"- **Description:** {description}",
        _build_epic_context(idea),
        _format_injected_epic_context(epic_context, previous_results),
        _build_discussion(idea),
        _build_prior_failure_context(idea),
        f"\n## Codebase (what already exists — don't duplicate)\n{_load_codebase_summary()}",
        _get_category_guidance(idea.category),
        _find_relevant_test_file(idea),
        _build_workflow_section(idea),
    ]
    return "\n".join(s for s in sections if s)


def _build_epic_prompt(idea: Any) -> str:
    """Build a rich prompt for executing an entire epic sequentially."""
    all_ideas = load_ideas()
    stories = [i for i in all_ideas if i.parent_id == idea.id and i.state != "done"]
    done_stories = [i for i in all_ideas if i.parent_id == idea.id and i.state == "done"]

    if not stories and not done_stories:
        return _build_story_prompt(idea)

    # Done context
    done_context = ""
    if done_stories:
        done_lines = ["\n## Already Completed Stories"]
        for d in done_stories:
            done_lines.append(f"- {d.id}: {d.title} [DONE]")
        done_lines.append("\nThese are already implemented. Build on them, don't duplicate them.")
        done_context = "\n".join(done_lines)

    # Story sections
    story_sections = ""
    for idx, story in enumerate(stories, 1):
        discussion = ""
        if story.comments:
            discussion = "**Discussion:**\n"
            for c in story.comments:
                label = settings.owner_name if c.author == "owner" else "LLM"
                discussion += f"  - {label}: {c.text}\n"

        story_sections += (
            f"\n{'=' * 70}\n"
            f"## Story {idx}/{len(stories)}: {story.title}\n"
            f"**ID:** {story.id}\n"
            f"**Category:** {story.category}\n\n"
            f"**Description:** {story.description}\n\n"
            f"{discussion}"
            f"**After completing this story**, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/done\n"
            f"```\n"
            f"If this story fails, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/comment "
            f'-H "Content-Type: application/json" '
            f"-d '{{\"author\": \"claude\", \"text\": \"Execution failed: <describe what went wrong>\"}}'\n"
            f"```\n"
            f"Then move to the next story.\n"
        )

    sections = [
        f"# EPIC: {idea.title}\n",
        "You are implementing an entire epic for the Technomancer project.",
        f"This epic has **{len(stories)} stories** to implement sequentially.\n",
        f"## Epic Description\n{idea.description}",
        done_context,
        f"\n## Codebase (what already exists — don't duplicate)\n{_load_codebase_summary()}",
        "\n## Implementation Process\n"
        "The branch has ALREADY been created for you. You are already on it.\n\n"
        "For EACH story below:\n"
        "1. Read CLAUDE.md for project conventions\n"
        "2. Implement the story (code, tests)\n"
        f"3. `git add <files>` && `git commit -m '[{idea.id}] description'`\n\n"
        f"**Every commit message MUST start with `[{idea.id}]`.**\n\n"
        "**YOUR JOB IS DONE AFTER COMMITTING.**\n\n"
        "Do NOT run safe_update.py, validate.py, pytest, bot_service.py, or "
        "any deploy/merge/restart commands. They are blocked in this environment. "
        "The executor handles all testing, validation, and deployment after you finish.\n\n"
        "Just write code, write tests, and commit.\n",
        f"\n# Stories to Implement\n{story_sections}",
    ]
    return "\n".join(s for s in sections if s)


def _build_diagnostic_fix_prompt(
    idea: Any,
    idea_id: str,
    failure_output: str,
    failed_tests: set[str],
    codebase_context: str,
    project_root: Path,
) -> str:
    """Build a diagnostic fix prompt that mimics how a human debugs.

    Instead of "here's the error, fix it", this gives Claude:
    1. Full content of failing test files
    2. Full content of the production code files being tested
    3. The complete error output
    4. System environment info (Python version, OS)
    5. Known pitfalls from previous failures
    6. Instruction to DIAGNOSE first, then fix
    """
    local_agent = project_root / "local-agent"
    sections = []

    # 1. System environment
    sections.append(
        f"## Environment\n"
        f"- Python: {sys.version.split()[0]}\n"
        f"- OS: Windows 11 (win32)\n"
        f"- Test runner: pytest with xdist (parallel) + rerunfailures\n"
    )

    # 2. The error output
    sections.append(
        f"## Test Failures\n```\n{failure_output}\n```\n"
    )

    # 3. Read the actual failing test files and their corresponding source files
    files_read = set()
    for test_id in sorted(failed_tests):
        # test_id format: tests/unit/test_foo.py::TestClass::test_method
        # or: tests\unit\test_foo.py::TestClass::test_method (Windows)
        test_path_str = test_id.split("::")[0].replace("\\", "/")
        test_file = local_agent / test_path_str

        if test_file.exists() and str(test_file) not in files_read:
            files_read.add(str(test_file))
            try:
                content = test_file.read_text(encoding="utf-8")
                sections.append(
                    f"## Failing test file: {test_path_str}\n"
                    f"```python\n{content}\n```\n"
                )
            except Exception:
                pass

            # Find the corresponding source file
            test_name = test_file.stem  # test_foo
            source_name = test_name.replace("test_", "", 1) + ".py"
            for search_dir in [local_agent / "agent", local_agent / "idea_board"]:
                source_file = search_dir / source_name
                if source_file.exists() and str(source_file) not in files_read:
                    files_read.add(str(source_file))
                    try:
                        content = source_file.read_text(encoding="utf-8")
                        # Cap at 5000 chars to avoid prompt bloat
                        if len(content) > 5000:
                            content = content[:5000] + "\n... (truncated)"
                        sections.append(
                            f"## Source file: {source_file.relative_to(local_agent)}\n"
                            f"```python\n{content}\n```\n"
                        )
                    except Exception:
                        pass

    # 4. Known pitfalls
    sections.append(
        "## KNOWN PITFALLS IN THIS CODEBASE\n"
        "These are real bugs from previous executor runs. Check each:\n\n"
        "1. **Mock patch targets must be module-level imports.** "
        "patch('agent.module.thing') fails if 'thing' is imported inside a function.\n"
        "2. **Flask content_type includes '; charset=utf-8'.** Use 'in' not '=='.\n"
        "3. **aiohttp mocks need async context managers.** "
        "Wrap with __aenter__/__aexit__.\n"
        "4. **Windows paths use backslashes.** Normalize in assertions.\n"
        "5. **Mutable list aliasing.** Pass list(x) not x if the list grows.\n"
        "6. **StopIteration in async (Python 3.14).** Becomes RuntimeError. "
        "Use KeyboardInterrupt to break async loops in tests.\n"
    )

    # 5. Diagnostic instruction
    sections.append(
        "## YOUR TASK\n\n"
        "**Step 1: Diagnose.** Read the error, the test code, and the source code. "
        "Identify the ROOT CAUSE — not just what failed, but WHY.\n\n"
        "**Step 2: Fix.** Make the minimum change to fix the root cause. "
        "This might be in the test (wrong assertion, bad mock) or in the "
        "production code (wrong behavior, missing import).\n\n"
        "**Step 3: Commit.** `git add <files>` && "
        f"`git commit -m '[{idea_id}] Fix: <one-line description of root cause>'`\n\n"
        "Do NOT run safe_update.py, validate.py, or pytest. Just fix and commit.\n"
    )

    return "\n".join(sections)


def _build_codebase_context(idea: Any, project_root: Path) -> str:
    """Build codebase context with code, not an LLM.

    Replaces the 5-minute LLM exploration pass with a <1 second script
    that reads CLAUDE.md, identifies relevant files, and extracts test
    patterns — everything the LLM was slowly discovering on its own.
    """
    local_agent = project_root / "local-agent"
    agent_dir = local_agent / "agent"
    test_dir = local_agent / "tests" / "unit"
    sections = []

    # 1. CLAUDE.md (full — it's the project bible)
    claude_md = project_root / "CLAUDE.md"
    if claude_md.exists():
        sections.append(f"## CLAUDE.md\n```\n{claude_md.read_text(encoding='utf-8')}\n```")

    # 2. Find relevant files from the idea description
    desc = (idea.description or "") + " " + (idea.title or "")
    desc_lower = desc.lower()

    # Extract explicit file references: "agent/foo.py", "idea_board/bar.py"
    import re as _re
    explicit_files = _re.findall(
        r'(?:agent|idea_board|tests/unit)/[\w/]+\.py', desc
    )

    relevant_modules = []
    # Add explicitly mentioned files
    for ref in explicit_files:
        full = local_agent / ref
        if full.exists():
            relevant_modules.append(full)

    # Also match module names by partial word overlap
    all_dirs = [agent_dir, local_agent / "idea_board"]
    for search_dir in all_dirs:
        if not search_dir.exists():
            continue
        for py_file in sorted(search_dir.glob("*.py")):
            name = py_file.stem
            if name == "__init__" or py_file in relevant_modules:
                continue
            # Match: full name, underscored name, or any word fragment
            name_words = name.split("_")
            if (name in desc_lower
                    or name.replace("_", " ") in desc_lower
                    or any(w in desc_lower for w in name_words if len(w) > 3)):
                relevant_modules.append(py_file)

    # 3. For each relevant module, include first 50 lines (imports + class/function signatures)
    if relevant_modules:
        sections.append("## Relevant modules (first 50 lines each)")
        for mod in relevant_modules[:8]:  # Cap at 8 to avoid prompt bloat
            try:
                lines = mod.read_text(encoding="utf-8").split("\n")[:50]
                rel_path = mod.relative_to(project_root)
                sections.append(f"### {rel_path}\n```python\n{chr(10).join(lines)}\n```")
            except Exception:
                pass

    # 4. Matching test files
    sections.append("## Existing test patterns")
    for mod in relevant_modules[:4]:
        test_name = f"test_{mod.stem}.py"
        test_file = test_dir / test_name
        if test_file.exists():
            try:
                lines = test_file.read_text(encoding="utf-8").split("\n")[:30]
                sections.append(f"### {test_name} (first 30 lines)\n```python\n{chr(10).join(lines)}\n```")
            except Exception:
                pass

    # 5. conftest.py fixtures
    conftest = test_dir / "conftest.py"
    if conftest.exists():
        try:
            lines = conftest.read_text(encoding="utf-8").split("\n")[:40]
            sections.append(f"### conftest.py (fixtures)\n```python\n{chr(10).join(lines)}\n```")
        except Exception:
            pass

    # 6. Module inventory (so Claude knows what exists)
    all_modules = sorted(f.stem for f in agent_dir.glob("*.py") if f.stem != "__init__")
    sections.append(f"## All modules in agent/\n{', '.join(all_modules)}")

    return "\n\n".join(sections)



# _run_exploration_pass removed — replaced by _build_codebase_context()
# which builds context with code in <1s instead of an LLM in 5+ minutes.


def _parse_stream_event(line: str) -> tuple[str, str]:
    """Parse a stream-json line into (event_type, display_text).

    Claude Code --output-format stream-json emits one JSON object per line.
    Key event types:
      - {"type": "assistant", "message": {"content": [{"text": "..."}]}}
      - {"type": "tool_use", "tool": {"name": "Edit"}, ...}
      - {"type": "tool_result", ...}
      - {"type": "result", "result": "...", "session_id": "..."}

    Returns:
        (event_type, display_text) — display_text is empty if not worth showing.
    """
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ("unknown", "")

    event_type = event.get("type", "")

    if event_type == "assistant":
        # Extract text from content blocks
        message = event.get("message", {})
        content = message.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return ("assistant", "\n".join(texts))

    if event_type == "tool_use":
        tool_name = event.get("tool", {}).get("name", event.get("name", "tool"))
        return ("tool_use", f"Using tool: {tool_name}")

    if event_type == "tool_result":
        return ("tool_result", "")

    if event_type == "result":
        return ("result", "Final result")

    return (event_type, "")


def execute_idea(
    idea_id: str,
    extra_context: str = "",
    epic_context: str = "",
    previous_results: list[dict[str, str]] | None = None,
) -> ExecutionState | None:
    """Start executing an idea with Claude Code.

    Spawns a background thread that runs claude.exe and streams
    stdout line-by-line into the execution state. The dashboard
    polls this state for live updates.

    Args:
        idea_id: The idea to execute
        extra_context: Optional flat context string (legacy, still supported)
        epic_context: Optional epic narrative for story-in-epic execution
        previous_results: Optional list of prior story results (id, title, state, summary)

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
    _clear_execution_artifacts(idea_id)

    state = ExecutionState(idea_id=idea_id)
    _active[idea_id] = state

    # Build the rich prompt
    if idea.idea_type == "epic":
        prompt = _build_epic_prompt(idea)
    else:
        prompt = _build_story_prompt(
            idea,
            epic_context=epic_context,
            previous_results=previous_results,
        )

    # Log prompt size for debugging context window issues
    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4
    logger.info(f"[Executor] {idea_id} prompt: {prompt_chars} chars (~{prompt_tokens_est} tokens)")

    def _run() -> None:
        """Background thread: two-pass Claude Code execution.

        Pass 1 (exploration): Read-only exploration of the codebase to build
        understanding of architecture, patterns, and test structure.

        Pass 2 (implementation): Full implementation with --resume to carry
        forward all context from the exploration pass.

        Falls back to single-pass if exploration fails.
        """
        binary = _find_claude_binary()
        if not binary:
            state.log("ERROR: Claude Code binary not found")
            mark_failed(idea_id, "Claude Code binary not found")
            _active.pop(idea_id, None)
            return

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("ANTHROPIC_API_KEY", None)  # Force Pro subscription, not API credits
        env["EXECUTOR_MODE"] = "1"  # Blocks safe_update.py continue
        project_root = Path(__file__).parent.parent.parent
        local_agent_dir = str(Path(__file__).parent.parent)

        _notify_discord(f"Starting execution of {idea_id}: {idea.title}")

        try:
            # Load known failures from previous full suite (replaces baseline run)
            state.baseline_failures = _load_known_failures()
            if state.baseline_failures:
                state.log(
                    f"Known failures from last run: {len(state.baseline_failures)}"
                )
            else:
                state.log("No known failures cached")

            # --- Phase 0b: Create branch (pure git, no safe_update.py) ---
            # safe_update.py takes ~15s just to import (loads entire bot stack).
            # Branch creation is simple git operations — do it directly.
            short_name = idea.id.replace("idea-", "")
            timestamp = time.strftime("%Y-%m-%d-%H%M%S")
            branch_name = f"{timestamp}-{short_name}"

            def _git(args: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["git"] + args,
                    capture_output=True, text=True, timeout=timeout,
                    cwd=str(project_root),
                )

            try:
                state.log("--- Setting up fresh branch ---")

                # Step 1: Force switch to main
                current = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
                if current != "main":
                    state.log(f"Resetting from {current} to main...")
                    _git(["checkout", "--force", "main"])
                    _git(["branch", "-D", current])
                    state.log(f"Deleted old branch {current}")

                # Step 2: Clean working directory
                _git(["checkout", "--force", "main"])
                _git(["clean", "-fd"], timeout=30)
                state.log("Working directory clean")

                # Step 3: Clean stale safe_update state
                state_file = Path(local_agent_dir) / ".safe_update_state"
                if state_file.exists():
                    state_file.unlink(missing_ok=True)

                # Step 4: Pull latest
                state.log("Pulling latest main...")
                _git(["pull", "origin", "main"], timeout=30)

                # Step 5: Create branch
                state.log(f"Creating branch {branch_name}...")
                result = _git(["checkout", "-b", branch_name])
                if result.returncode != 0:
                    raise RuntimeError(result.stderr or result.stdout)

                # Step 6: Write safe_update state file (so safe_update.py continue works)
                state_file.write_text(branch_name)

                state.log(f"Branch created: {branch_name}")

            except Exception as e:
                msg = f"Branch creation failed: {e}"
                state.log(msg)
                _notify_discord(f"[{idea_id}] {msg}")
                mark_failed(idea_id, state.log_text)
                return

            # branch_name already set above in Phase 0b
            # --- Phase 1: Build codebase context (code, not LLM) ---
            state.log("--- Building codebase context ---")
            codebase_context = _build_codebase_context(idea, project_root)
            context_chars = len(codebase_context)
            state.log(
                f"Context built: {context_chars} chars "
                f"(~{context_chars // 4} tokens)"
            )

            # Inject context into the prompt
            epic_ctx_section = ""
            if extra_context:
                epic_ctx_section = (
                    f"## Epic Execution Context\n\n"
                    f"{extra_context}\n\n"
                )

            full_prompt = (
                f"## Codebase Context (pre-built)\n\n"
                f"{codebase_context}\n\n"
                f"{epic_ctx_section}"
                f"---\n\n"
                f"{prompt}"
            )
            full_prompt_chars = len(full_prompt)
            full_prompt_tokens = full_prompt_chars // 4

            if state.cancelled:
                state.log("CANCELLED by user")
                mark_failed(idea_id, state.log_text)
                _notify_discord(f"Execution of {idea_id} was cancelled.")
                _active.pop(idea_id, None)
                return

            # --- Phase 2: Implementation (streaming) ---
            state.log("")
            state.log("--- Phase 2: Implementation ---")
            state.log(
                f"Prompt: {full_prompt_chars} chars (~{full_prompt_tokens} tokens)"
            )

            # Write prompt to temp file — Windows has 32K command-line limit
            import tempfile
            prompt_file = Path(tempfile.mktemp(suffix=".txt", prefix="executor_"))
            prompt_file.write_text(full_prompt, encoding="utf-8")

            cmd = [
                str(binary), "-p", "-",
                "--output-format", "stream-json",
                "--verbose",
                "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
                "--max-turns", "50",
            ]
            state.log("Starting Claude Code with pre-built context...")

            prompt_input = open(prompt_file, "r", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdin=prompt_input,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(project_root),
                env=env,
            )
            state.pid = proc.pid
            state.log(f"Claude Code started (PID: {proc.pid})")
            state.log(f"Working on: {idea.title}")
            logger.info(f"[Executor] {idea_id} started, PID {proc.pid}")

            # Stream stdout line-by-line, parsing JSON events as they arrive
            last_discord_time = 0.0
            last_jira_progress_time = 0.0
            final_result = ""

            while True:
                # Check cancellation and timeout before blocking on readline
                if state.cancelled:
                    proc.kill()
                    state.log("CANCELLED by user")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(f"Execution of {idea_id} was cancelled.")
                    _active.pop(idea_id, None)
                    return

                if state.elapsed > EXECUTION_TIMEOUT:
                    proc.kill()
                    state.log(f"TIMEOUT after {EXECUTION_TIMEOUT}s")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(
                        f"Execution of {idea_id} timed out after "
                        f"{EXECUTION_TIMEOUT // 60} minutes."
                    )
                    _active.pop(idea_id, None)
                    return

                raw_line = proc.stdout.readline() if proc.stdout else b""
                if not raw_line:
                    if proc.poll() is not None:
                        break  # Process exited and no more output
                    continue

                line_text = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line_text:
                    continue

                # Parse the stream-json event
                event_type, display_text = _parse_stream_event(line_text)

                if event_type == "result":
                    # Capture final result metadata and exit the stream loop
                    try:
                        result_data = json.loads(line_text)
                        final_result = result_data.get("result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    if display_text:
                        state.log(display_text)
                    break  # Result event = Claude is done, stop reading

                if display_text:
                    # Log to dashboard
                    state.log(display_text)

                    # Rate-limited Discord notification for meaningful events
                    now = time.time()
                    if event_type in ("assistant", "tool_use") and now - last_discord_time >= DISCORD_RATE_LIMIT:
                        # Truncate for Discord (keep it concise)
                        discord_msg = display_text[:300]
                        if len(display_text) > 300:
                            discord_msg += "..."
                        _notify_discord(f"[{idea_id}] {discord_msg}")
                        last_discord_time = now

                    # Periodic Jira progress comment (edit-in-place)
                    if now - last_jira_progress_time >= JIRA_PROGRESS_INTERVAL:
                        _sync_progress_comment(idea_id, state)
                        last_jira_progress_time = now

            # Process finished — kill immediately to free resources for Phase 3
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            # Brief pause to let OS fully release resources
            time.sleep(2)

            # Success is authoritatively determined by whether the feature
            # branch has commits ahead of main. Claude's stdout is a weak
            # signal — stream-json can miss events, credit exhaustion can
            # exit non-gracefully, and the substring "result" can appear
            # in unrelated output. Commits on the branch are the ground
            # truth that Claude can't fake. The stream-json final_result
            # stays as a secondary hint that's logged but not load-bearing.
            project_root_str = str(project_root)
            current_branch_check = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=project_root_str,
            )
            current_branch = current_branch_check.stdout.strip()
            commits_ahead = 0
            if current_branch and current_branch != "main":
                ahead = subprocess.run(
                    ["git", "rev-list", "--count", f"main..{current_branch}"],
                    capture_output=True, text=True, cwd=project_root_str,
                )
                commits_ahead = int((ahead.stdout or "0").strip() or "0")
            claude_succeeded = commits_ahead > 0
            state.log(
                f"Success check: {commits_ahead} commit(s) on {current_branch} "
                f"ahead of main (final_result={'set' if final_result else 'empty'})"
            )

            if not claude_succeeded:
                # Distinguish "Claude hit a usage/rate limit and bailed" from
                # a real failure. On rate-limit we do NOT call mark_failed —
                # we return with state.rate_limited set, and the Worker runs
                # its retry loop (see aim/worker.py::execute_assigned_idea).
                if _classify_rate_limit(state, claude_succeeded, project_root):
                    state.rate_limited = True
                    state.log(
                        "Detected Claude usage/rate limit — deferring retry to worker"
                    )
                    _notify_discord(
                        f"[{idea_id}] Rate limited — worker will retry"
                    )
                    return

                state.log(
                    f"Claude failed ({state.elapsed:.0f}s)"
                )
                mark_failed(idea_id, state.log_text[-5000:])
                _notify_discord(
                    f"Idea {idea_id} execution failed: {idea.title}"
                )
                return

            state.log(
                f"Claude finished ({state.elapsed:.0f}s). Validating..."
            )
            _notify_discord(f"[{idea_id}] Code complete. Running validation...")

            # --- Phase 2.5: Validate + targeted test retry loop ---
            # Fast feedback: validate + run only tests related to changed files.
            # If failures, give Claude a chance to fix. Full suite runs once at the end.
            related_tests = _find_related_tests(project_root)
            if related_tests:
                state.log(
                    f"Related tests: {len(related_tests)} file(s) — "
                    + ", ".join(Path(t).name for t in related_tests)
                )
            else:
                state.log("No related test files found — will run full suite only")

            for attempt in range(1, MAX_FIX_RETRIES + 2):  # +2: 1 initial + N retries
                failure_output = ""
                delta: set[str] = set()

                # Validate
                state.log("")
                state.log(
                    f"--- Validation (attempt {attempt}) ---"
                )
                validate_result = subprocess.run(
                    [sys.executable, "validate.py", "import"],
                    capture_output=True, text=True, timeout=60,
                    cwd=local_agent_dir,
                )
                if validate_result.returncode != 0:
                    fail_lines = [
                        vline.strip()
                        for vline in validate_result.stdout.split("\n")
                        if "FAIL" in vline or "BLOCKED" in vline
                        or "ERROR" in vline
                    ]
                    for fl in fail_lines:
                        state.log(fl)
                    failure_output = (
                        "VALIDATION FAILED:\n"
                        + validate_result.stdout[-2000:]
                    )
                else:
                    state.log("Validation passed")

                    # Run targeted tests (fast feedback)
                    if related_tests:
                        state.log(
                            f"--- Targeted tests (attempt {attempt}) ---"
                        )
                        _notify_discord(
                            f"[{idea_id}] Running {len(related_tests)} "
                            f"related test file(s) (attempt {attempt})..."
                        )
                        test_start = time.time()
                        test_result = subprocess.run(
                            [sys.executable, "-m", "pytest", "-q",
                             "--tb=short"] + related_tests,
                            capture_output=True, text=True, timeout=120,
                            cwd=local_agent_dir,
                        )
                        test_duration = time.time() - test_start
                        test_summary = [
                            ln.strip()
                            for ln in test_result.stdout.split("\n")
                            if "passed" in ln or "failed" in ln
                            or "error" in ln.lower()
                        ]
                        for line in test_summary:
                            state.log(line)
                        state.log(
                            f"Targeted tests: {test_duration:.0f}s"
                        )

                        if test_result.returncode != 0:
                            # Check baseline diff
                            new_failures = _parse_pytest_failures(
                                test_result.stdout
                            )
                            delta = new_failures - state.baseline_failures
                            if not delta:
                                state.log(
                                    "All failure(s) are pre-existing — OK"
                                )
                            else:
                                state.log(
                                    f"New failures: {len(delta)}"
                                )
                                for f in sorted(delta):
                                    state.log(f"  - {f}")
                                failure_output = (
                                    "TESTS FAILED:\n"
                                    + test_result.stdout[-3000:]
                                )
                    # If no related tests or targeted tests passed, continue
                    if not failure_output:
                        break  # Targeted tests OK — proceed to full suite

                # If we have a failure and retries remain, launch Claude to fix
                if failure_output and attempt <= MAX_FIX_RETRIES:
                    state.log(
                        f"Launching Claude to fix (retry {attempt}/{MAX_FIX_RETRIES})..."
                    )
                    _sync_progress_comment(idea_id, state)
                    _notify_discord(
                        f"[{idea_id}] Tests/validation failed. "
                        f"Retry {attempt}/{MAX_FIX_RETRIES}..."
                    )

                    fix_prompt = _build_diagnostic_fix_prompt(
                        idea, idea_id, failure_output, delta,
                        codebase_context, project_root,
                    )

                    fix_file = Path(tempfile.mktemp(suffix=".txt", prefix="fix_"))
                    fix_file.write_text(fix_prompt, encoding="utf-8")
                    fix_input = open(fix_file, "r", encoding="utf-8")

                    fix_cmd = [
                        str(binary), "-p", "-",
                        "--output-format", "stream-json",
                        "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
                        "--max-turns", "30",
                    ]

                    fix_proc = subprocess.Popen(
                        fix_cmd,
                        stdin=fix_input,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=str(project_root),
                        env=env,
                    )
                    state.pid = fix_proc.pid
                    state.log(
                        f"Fix Claude started (PID: {fix_proc.pid})"
                    )

                    # Stream fix output (same as Phase 2)
                    while True:
                        if state.cancelled:
                            fix_proc.kill()
                            state.log("CANCELLED by user")
                            mark_failed(idea_id, state.log_text)
                            _notify_discord(
                                f"Execution of {idea_id} was cancelled."
                            )
                            _active.pop(idea_id, None)
                            return

                        if state.elapsed > EXECUTION_TIMEOUT:
                            fix_proc.kill()
                            state.log(
                                f"TIMEOUT after {EXECUTION_TIMEOUT}s"
                            )
                            mark_failed(idea_id, state.log_text)
                            _active.pop(idea_id, None)
                            return

                        raw = (
                            fix_proc.stdout.readline()
                            if fix_proc.stdout
                            else b""
                        )
                        if not raw:
                            if fix_proc.poll() is not None:
                                break
                            continue

                        line_text = raw.decode(
                            "utf-8", errors="replace"
                        ).rstrip()
                        if not line_text:
                            continue

                        evt, dtxt = _parse_stream_event(line_text)
                        if evt == "result":
                            if dtxt:
                                state.log(dtxt)
                            break
                        if dtxt:
                            state.log(dtxt)

                    # Kill fix process
                    if fix_proc.poll() is None:
                        fix_proc.kill()
                        try:
                            fix_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                    time.sleep(2)

                    state.log(
                        f"Fix attempt {attempt} complete. Re-validating..."
                    )
                    _sync_progress_comment(idea_id, state)
                    continue  # Back to top of retry loop

                elif failure_output:
                    # No retries left
                    state.log(
                        f"Failed after {attempt} attempt(s) — aborting deploy"
                    )
                    mark_failed(idea_id, state.log_text[-5000:])
                    _notify_discord(
                        f"Idea {idea_id} failed after {attempt} attempts: "
                        f"{idea.title}"
                    )
                    return

            # --- Full test suite (final gate before deploy) ---
            state.log("")
            state.log("--- Full test suite (parallel) ---")
            _notify_discord(f"[{idea_id}] Running full test suite...")
            load_before = _snapshot_system_load()
            state.log(f"Pre-test: {load_before}")
            full_start = time.time()

            full_result = _run_pytest_with_progress(
                [sys.executable, "-m", "pytest", "-q", "--tb=short",
                 "--reruns", "2", "--reruns-delay", "1"],
                cwd=local_agent_dir,
                state=state,
                label="tests",
                timeout=PYTEST_TIMEOUT,
            )

            full_duration = time.time() - full_start
            load_after = _snapshot_system_load()
            state.log(
                f"Full suite: {full_duration:.0f}s | {load_after}"
            )

            if full_result.returncode != 0:
                # Check baseline diff
                full_failures = _parse_pytest_failures(full_result.stdout)
                delta = full_failures - state.baseline_failures
                if delta:
                    state.log(
                        f"Full suite: {len(delta)} new failure(s):"
                    )
                    for f in sorted(delta):
                        state.log(f"  - {f}")
                    state.log(
                        "Full suite FAILED — aborting deploy"
                    )
                    mark_failed(idea_id, state.log_text[-5000:])
                    _notify_discord(
                        f"Idea {idea_id} full suite failed: {idea.title}"
                    )
                    return
                else:
                    state.log(
                        f"All {len(full_failures)} failure(s) are "
                        f"pre-existing — proceeding to deploy"
                    )

            # Save failures from this run as the baseline for the next execution
            full_failures = _parse_pytest_failures(full_result.stdout)
            _save_known_failures(full_failures)

            # --- Phase 3: Deploy (merge + push) ---
            state.log("")
            state.log("--- Phase 3: Deploy ---")

            try:
                # Merge to main
                state.log("Merging to main...")
                project_root = str(Path(__file__).parent.parent.parent)
                branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, cwd=project_root,
                ).stdout.strip()

                # Defense-in-depth: verify the branch still has commits
                # ahead of main. Success detection already checked this,
                # but state can drift between that check and here (another
                # process committed, branch got rebased, etc.). A zero-
                # commit merge would silently produce "Already up to date"
                # and the auto-README commit would falsely claim success.
                # Fail loudly instead of silently.
                ahead_check = subprocess.run(
                    ["git", "rev-list", "--count", f"main..{branch}"],
                    capture_output=True, text=True, cwd=project_root,
                )
                if int((ahead_check.stdout or "0").strip() or "0") == 0:
                    state.log(
                        f"Merge aborted: feature branch '{branch}' has 0 "
                        f"commits ahead of main. Nothing to merge."
                    )
                    mark_failed(
                        idea_id,
                        "Merge phase saw 0 commits ahead of main. "
                        "Story cannot be marked done without real work.",
                    )
                    _notify_discord(
                        f"Idea {idea_id} merge aborted — no commits to merge.",
                    )
                    return

                # Abort if the feature branch has uncommitted changes —
                # git checkout would silently carry them to main and the
                # auto-README commit would claim credit for uncommitted
                # work. That's the exact failure mode that lost TK-410's
                # rate-limit code. Prefer a loud failure.
                status_check = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, cwd=project_root,
                )
                if status_check.stdout.strip():
                    state.log(
                        "Merge aborted: feature branch has uncommitted changes "
                        f"({len(status_check.stdout.splitlines())} paths). "
                        "Claude Code was supposed to commit. Refusing to "
                        "checkout main and carry dirt onto the shipping branch."
                    )
                    mark_failed(
                        idea_id,
                        "Uncommitted working tree at merge time. "
                        "The branch still has its changes — recover manually "
                        "or let the story re-run.",
                    )
                    _notify_discord(
                        f"Idea {idea_id} merge aborted — uncommitted changes.",
                    )
                    return

                subprocess.run(
                    ["git", "checkout", "main"],
                    capture_output=True, text=True, cwd=project_root,
                )
                merge_result = subprocess.run(
                    ["git", "merge", "--no-ff", branch,
                     "-m", f"[{idea_id}] Merge branch '{branch}' - executor auto-deploy"],
                    capture_output=True, text=True, cwd=project_root,
                )
                if merge_result.returncode != 0:
                    state.log(f"Merge failed: {merge_result.stderr[:200]}")
                    mark_failed(idea_id, state.log_text[-5000:])
                    _notify_discord(f"Idea {idea_id} merge failed: {idea.title}")
                    return

                # Capture the merge commit SHA for the [Deployed] comment
                sha_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, cwd=project_root,
                )
                deploy_sha = ""
                if sha_result.returncode == 0:
                    deploy_sha = sha_result.stdout.strip()[:7]

                # Step 3c: Delete branch
                subprocess.run(
                    ["git", "branch", "-d", branch],
                    capture_output=True, text=True, cwd=project_root,
                )

                # Step 3d: Push to origin
                state.log("Pushing to origin...")
                subprocess.run(
                    ["git", "push", "origin", "main"],
                    capture_output=True, text=True, timeout=30,
                    cwd=project_root,
                )

                # Step 3e: Clean up safe_update state
                state_file = Path(local_agent_dir) / ".safe_update_state"
                state_file.unlink(missing_ok=True)

                # Step 3f: Regenerate README (pass test count, skip rerunning pytest)
                import re as _re
                test_count = 0
                for ln in (full_result.stdout or "").split("\n"):
                    m = _re.search(r"(\d+) passed", ln)
                    if m:
                        test_count = int(m.group(1))

                try:
                    readme_script = Path(local_agent_dir) / "generate_readme.py"
                    if readme_script.exists():
                        readme_cmd = [sys.executable, str(readme_script)]
                        if test_count:
                            readme_cmd += ["--test-count", str(test_count)]
                        subprocess.run(
                            readme_cmd,
                            capture_output=True, text=True, timeout=60,
                            cwd=local_agent_dir,
                        )
                        # Check if README files actually changed (idempotency)
                        local_readme = "local-agent/README.md"
                        root_readme = "README.md"
                        local_diff = subprocess.run(
                            ["git", "diff", "--quiet", "--", local_readme],
                            capture_output=True, cwd=project_root,
                        )
                        root_diff = subprocess.run(
                            ["git", "diff", "--quiet", "--", root_readme],
                            capture_output=True, cwd=project_root,
                        )
                        if local_diff.returncode == 0 and root_diff.returncode == 0:
                            state.log("README unchanged — skipping auto-commit")
                        else:
                            if local_diff.returncode != 0:
                                subprocess.run(
                                    ["git", "add", local_readme],
                                    capture_output=True, timeout=10,
                                    cwd=project_root,
                                )
                            if root_diff.returncode != 0:
                                subprocess.run(
                                    ["git", "add", root_readme],
                                    capture_output=True, timeout=10,
                                    cwd=project_root,
                                )
                            subprocess.run(
                                ["git", "commit", "-m",
                                 f"[{idea_id}] Deploy + update stats"],
                                capture_output=True, timeout=10,
                                cwd=project_root,
                            )
                            subprocess.run(
                                ["git", "push", "origin", "main"],
                                capture_output=True, timeout=30,
                                cwd=project_root,
                            )
                            state.log("README updated")
                except Exception as e:
                    state.log(f"README error (non-blocking): {e}")

                # Step 3g: Publish to public repo (with retry)
                publish_script = Path(local_agent_dir) / "publish.py"
                if publish_script.exists():
                    for pub_attempt in range(3):
                        try:
                            pub = subprocess.run(
                                [sys.executable, str(publish_script),
                                 "--push", "--force"],
                                capture_output=True, text=True, timeout=120,
                                cwd=local_agent_dir,
                            )
                            if pub.returncode == 0:
                                state.log("Published to technomancer-public")
                                break
                            else:
                                err = (pub.stderr or pub.stdout)[:2000]
                                state.log(
                                    f"Publish attempt {pub_attempt + 1}/3 failed:\n{err}"
                                )
                        except Exception as e:
                            state.log(
                                f"Publish attempt {pub_attempt + 1}/3 error: {e}"
                            )
                        if pub_attempt < 2:
                            state.log("Retrying publish in 30s...")
                            time.sleep(30)

                state.log(
                    f"Deploy complete ({state.elapsed:.0f}s total). "
                    f"Bot restart needed — run: python bot_service.py start"
                )
                if deploy_sha:
                    _post_deploy_comment(idea_id, deploy_sha)
                mark_done(idea_id, state.log_text[-5000:])
                _notify_discord(
                    f"Idea {idea_id} deployed ({state.elapsed:.0f}s): "
                    f"{idea.title}. Bot restart needed."
                )

            except subprocess.TimeoutExpired:
                load_at_timeout = _snapshot_system_load()
                state.log(
                    f"Deploy timed out after {PYTEST_TIMEOUT}s | {load_at_timeout}"
                )
                mark_failed(idea_id, state.log_text[-5000:])
                _notify_discord(
                    f"Idea {idea_id} deploy timed out ({PYTEST_TIMEOUT}s). "
                    f"System: {load_at_timeout}"
                )
            except Exception as deploy_err:
                state.log(f"Deploy error: {deploy_err}")
                mark_failed(idea_id, state.log_text[-5000:])
                _notify_discord(f"Idea {idea_id} deploy error: {deploy_err}")

        except Exception as e:
            tb = traceback.format_exc()
            state.log(f"ERROR: {e}")
            state.log(tb)
            logger.error(f"[Executor] {idea_id} error: {tb}")
            mark_failed(idea_id, state.log_text)
            _notify_discord(f"Idea {idea_id} execution error: {e}")

        finally:
            _active.pop(idea_id, None)
            # Dump execution log to Jira comment for traceability
            _post_execution_log_to_jira(idea_id, state)

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

    # Kill the process tree (taskkill /F /T on Windows for reliable tree kill)
    if state.pid and state.is_alive:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(state.pid)],
                    capture_output=True,
                    timeout=10,
                )
                logger.info(
                    f"[Executor] Killed process tree for PID {state.pid} ({idea_id})"
                )
            else:
                os.kill(state.pid, signal.SIGTERM)
                logger.info(
                    f"[Executor] Sent SIGTERM to PID {state.pid} for {idea_id}"
                )
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            pass

    return True


def execute_epic(epic_id: str) -> ExecutionState | None:
    """Execute an epic by running each story in execution_order sequentially.

    Spawns a background thread that iterates through the epic's child stories,
    executing each via execute_idea() and waiting for completion before
    starting the next.

    Each story receives the epic's big-picture context and a summary of
    what previous stories accomplished.

    Args:
        epic_id: The epic idea ID

    Returns:
        ExecutionState for tracking, or None if epic not found or not an epic
    """
    epic = get_idea(epic_id)
    if not epic or epic.idea_type != "epic":
        return None

    # Don't start if already executing
    if epic_id in _active and _active[epic_id].is_alive:
        return _active[epic_id]

    execution_order = get_execution_order(epic_id)

    mark_executing(epic_id)
    _clear_execution_artifacts(epic_id)
    state = ExecutionState(idea_id=epic_id)
    _active[epic_id] = state

    if not execution_order:
        state.log("No stories in execution order")
        mark_failed(epic_id, "No stories in execution order")
        _active.pop(epic_id, None)
        return state

    def _run_epic() -> None:
        try:
            story_results: list[dict[str, str]] = []

            state.log(f"=== Epic Executor: {epic.title} ===")
            state.log(
                f"Stories to execute: {len(execution_order)}"
            )
            for i, sid in enumerate(execution_order):
                story = get_idea(sid)
                name = story.title if story else sid
                state.log(f"  {i + 1}. {sid}: {name}")
            state.log("")

            _notify_discord(
                f"Starting epic {epic_id}: {epic.title} "
                f"({len(execution_order)} stories)"
            )

            for idx, story_id in enumerate(execution_order):
                story = get_idea(story_id)
                if not story:
                    state.log(
                        f"Story {story_id} not found -- skipping"
                    )
                    continue

                # Skip already-done stories
                if story.state == "done":
                    state.log(
                        f"[{idx + 1}/{len(execution_order)}] "
                        f"{story_id}: {story.title} -- already done, skipping"
                    )
                    story_results.append({
                        "id": story_id,
                        "title": story.title,
                        "state": "done",
                        "summary": "(completed before this epic run)",
                    })
                    continue

                state.log("=" * 60)
                state.log(
                    f"[{idx + 1}/{len(execution_order)}] "
                    f"Starting: {story_id} -- {story.title}"
                )
                state.log("=" * 60)

                _notify_discord(
                    f"[{epic_id}] Story {idx + 1}/{len(execution_order)}: "
                    f"{story.title}"
                )

                # Execute the story (full lifecycle: branch, Claude, tests, deploy)
                # Pass epic context and previous results as structured data
                # so _build_story_prompt() can embed them in the prompt.
                story_state = execute_idea(
                    story_id,
                    epic_context=epic.epic_context or "",
                    previous_results=list(story_results),  # Copy — list grows after each story
                )
                if not story_state:
                    state.log(
                        f"Failed to start {story_id}"
                    )
                    mark_failed(
                        epic_id,
                        f"Could not start story {story_id}\n\n"
                        + state.log_text[-5000:],
                    )
                    _notify_discord(
                        f"[{epic_id}] Epic FAILED: "
                        f"could not start {story_id}"
                    )
                    return

                # Wait for story execution to complete
                if story_state.thread:
                    story_state.thread.join()

                # Check final state
                completed_story = get_idea(story_id)
                final_state = (
                    completed_story.state
                    if completed_story
                    else "unknown"
                )

                # Capture summary from execution log
                summary_lines = [
                    ln
                    for ln in story_state.log_lines[-10:]
                    if ln.strip()
                ]
                summary = "\n".join(summary_lines[-5:])

                story_results.append({
                    "id": story_id,
                    "title": story.title,
                    "state": final_state,
                    "summary": summary,
                })

                if final_state == "done":
                    state.log(
                        f"[DONE] {story_id} completed successfully"
                    )
                elif final_state == "failed":
                    state.log(f"[FAILED] {story_id}")
                    state.log(
                        f"Stopping epic -- story {story_id} failed"
                    )
                    mark_failed(
                        epic_id,
                        f"Failed at story {story_id}: {story.title}"
                        f"\n\n{state.log_text[-5000:]}",
                    )
                    _notify_discord(
                        f"[{epic_id}] Epic FAILED at story "
                        f"{story_id}: {story.title}"
                    )
                    return
                else:
                    state.log(
                        f"[?] {story_id} ended in unexpected state: "
                        f"{final_state}"
                    )
                    mark_failed(
                        epic_id,
                        f"Story {story_id} ended in state "
                        f"'{final_state}'\n\n{state.log_text[-5000:]}",
                    )
                    _notify_discord(
                        f"[{epic_id}] Epic stopped -- {story_id} "
                        f"in state '{final_state}'"
                    )
                    return

                state.log("")

            # All stories completed
            state.log("=" * 60)
            state.log(
                f"All {len(execution_order)} stories completed!"
            )
            state.log(
                f"Epic execution time: {state.elapsed:.0f}s"
            )

            mark_done(epic_id, state.log_text[-5000:])
            _notify_discord(
                f"Epic {epic_id} complete ({state.elapsed:.0f}s): "
                f"{epic.title}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            state.log(f"Epic execution error: {e}")
            state.log(tb)
            logger.error(f"[EpicExecutor] {epic_id} error: {tb}")
            mark_failed(epic_id, state.log_text[-5000:])
            _notify_discord(f"[{epic_id}] Epic error: {e}")

        finally:
            _active.pop(epic_id, None)

    thread = threading.Thread(
        target=_run_epic, daemon=True, name=f"epic-executor-{epic_id}"
    )
    thread.start()
    state.thread = thread
    return state
