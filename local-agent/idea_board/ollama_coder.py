"""OllamaCoder — local Ollama agent that implements Jira stories.

Architecture
------------
Outer fix-round loop (max_rounds, default 20)
  Inner tool-call loop (max_turns per round, default 40)
    → call /api/chat with 7 tools
    → execute tool calls
    → if finish(): break inner loop
  → run pytest on changed files
  → if tests pass: SUCCESS
  → if round < max_rounds: build fix prompt, continue outer loop
→ exhausted: return failure

GPU exclusivity: acquires coder priority in run(), releases in finally.
This makes chat() callers in the Discord bot and AIM brain yield the
GPU while coding work is in progress.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from agent.ollama_client import (
    OLLAMA_HOST,
    acquire_coder_priority,
    release_coder_priority,
)
from agent.accountability import GitDirtyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class EnvironmentReadyError(Exception):
    """Raised when the environment is not ready for code generation."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASH_ALLOWLIST = ("git", "pytest", "python", "python3", "py")
BASH_BLOCKLIST = ("safe_update", "push origin", "push --force", "merge", "checkout main",
                   "checkout master", "rm -rf", "rmdir /s")

# Map of common bash shapes the model tries → structured-tool guidance.
# When a command is rejected by the allowlist, we look up the first token
# here and append a "Better tool:" hint to the error so the model can
# self-correct on its next turn instead of retrying the same shape. This
# is intentionally a hand-curated dict — the goal is teaching, not parsing.
BASH_SHAPE_GUIDANCE: dict[str, str] = {
    "cat":   "Use the `read_file` tool — give it the path you want to inspect.",
    "head":  "Use the `read_file` tool — pass `offset` and `length` for a slice.",
    "tail":  "Use the `read_file` tool — pass `offset` past the end of the file.",
    "ls":    "Use the `list_files` tool — give it a directory path (or omit for cwd).",
    "find":  "Use the `list_files` tool, or `search_code` for content search.",
    "grep":  "Use the `search_code` tool — give it a regex pattern and an optional path.",
    "rg":    "Use the `search_code` tool — same idea, regex-based.",
    "cp":    "Don't copy files for backup — git already tracks history. Use `git stash` if you need a quick rollback point.",
    "mv":    "Use the `edit_file` tool to rewrite the file at its new path, then delete the old one in a second `edit_file` call.",
    "wc":    "Use the `read_file` tool and count lines yourself, or `grep -c` via `search_code`.",
    "tree":  "Use `list_files` recursively, or pass a glob pattern to enumerate the tree.",
    "file":  "Inspect the first ~100 bytes via `read_file` — file-type guessing is rarely necessary in this codebase.",
    "echo":  "If you're trying to write a file, use `edit_file`. Don't pipe `echo` to a file.",
    "touch": "Use the `edit_file` tool with empty `new_string` to create an empty file.",
    "rm":    "Don't delete files manually. If you need to remove a file from a commit, use `git rm` (the `git` allowlist permits it).",
}

# Drift detection: number of rounds we look back for "no progress" check.
# 4 = one bad round + one wrong-turn round + two confirmation rounds. Less
# aggressive than 3 — user explicitly asked not to fail out the LLMs too
# quickly because some stories ARE genuinely hard.
DRIFT_WINDOW = 4
# For the "empty shop" condition, require at least this many rounds in the
# window to have hit the inner-loop max-turn exhaustion. A model that's
# slowly thinking through a hard problem won't usually exhaust the inner
# loop — flailing models will.
DRIFT_EMPTY_EXHAUSTION_THRESHOLD = 2

READ_FILE_MAX_CHARS = 20_000
LIST_FILES_MAX = 200

# Per-round dedup thresholds. The inner loop tracks each (tool_name, args)
# signature within a round; once the count *exceeds* this number, the call
# is short-circuited with a synthetic "[duplicate suppressed]" tool result
# so the model is forced to try a different argument. Read-only tools get
# a small budget (a second look at the same file/pattern is occasionally
# legitimate). Mutating tools must not repeat the same exact write — if
# the model is asking to write byte-identical content twice, it's stuck.
_DEDUP_THRESHOLDS: dict[str, int] = {
    "search_code": 2,
    "list_files": 2,
    "read_file": 2,
    "write_file": 1,
    "edit_file": 1,
}


def _coerce_int(value: Any) -> int | None:
    """Coerce a value (often a JSON-string from the model) to int.

    Returns None for None / empty / unparseable.  The model habitually
    sends ``offset='100'`` even when the schema declares an integer; the
    Ollama JSON path doesn't always coerce.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Error → hint mapping for re-injection
_ERROR_HINTS: list[tuple[str, str]] = [
    ("ImportError", "Check your import statements and package structure"),
    ("ModuleNotFoundError", "Check your import statements and package structure"),
    ("SyntaxError", "There's a Python syntax error — run `python -m py_compile <file>` to locate it"),
    ("AssertionError", "Check expected vs actual values carefully"),
    ("AttributeError", "Check that the attribute/method you're calling exists on the object"),
    ("TypeError", "Check function signatures and argument types"),
    ("NameError", "Check that variables and functions are defined before use"),
]

# ---------------------------------------------------------------------------
# Tool definitions (Ollama /api/chat format)
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. By default returns the whole "
                "file (truncated if very large). Pass ``offset`` (1-indexed "
                "line number) and/or ``length`` (number of lines) to page "
                "through large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed starting line (optional)",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of lines to read from offset (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "old_string": {"type": "string", "description": "Exact string to find"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path"},
                    "pattern": {"type": "string", "description": "Glob pattern (optional)", "default": "*"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern in source files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or string to search for"},
                    "path": {"type": "string", "description": "Absolute directory to search (default: project root)"},
                    "file_pattern": {"type": "string", "description": "File glob (e.g. '*.py')", "default": "*.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that the implementation is complete. The harness will run "
                "pytest and commit the changes — you do not need to do either yourself. "
                "Pass a brief summary describing what you changed; the harness will use "
                "it in the commit message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was done"}
                },
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# OllamaCoder
# ---------------------------------------------------------------------------

class OllamaCoder:
    """Implements a Jira story using a local Ollama model with tool-calling."""

    def __init__(
        self,
        prompt: str,
        project_root: str | Path,
        idea_id: str,
        state: Any,
        model: str = "qwen3.5:27b",
        max_turns: int = 40,
        max_rounds: int = 20,
        num_ctx: int = 16384,
        host: str = "",
        story_title: str = "",
    ) -> None:
        self.prompt = prompt
        self.project_root = Path(project_root)
        self.idea_id = idea_id
        self.state = state
        self.model = model
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.num_ctx = num_ctx
        # Ollama base URL for *this* coder instance. Empty string falls
        # back to the module-level OLLAMA_HOST (localhost) so existing
        # callers are unaffected. The A/B harness sets this to point
        # model B at a remote Ollama (e.g. http://192.168.1.150:11434).
        self.host = host or OLLAMA_HOST
        # Story title — used by _commit_changes to build the deterministic
        # commit message ``[<idea_id>] <title> <model> r<N>: <summary>``.
        # Empty string is acceptable; the message just omits the title slot.
        self.story_title = story_title
        self._log = state.log if hasattr(state, "log") else lambda m: None
        # Counts edit_file / write_file calls during the *current* round.
        # _run_rounds resets it before each inner loop and reads it after,
        # so _build_fix_prompt can flag a model that finished a round
        # without making any edits — the failure mode that wasted 2 of
        # TK-1045's rounds (model called finish() after only read_file).
        self._round_edit_count = 0
        # Snapshot of self._round_edit_count from the *previous* round.
        # Read by _build_fix_prompt to inject a strong nudge when the
        # model called finish() without editing anything.
        self.prev_round_edit_count: int | None = None
        # Per-round flag: set True once the inner loop has injected the
        # "you called finish() with zero edits" nudge so we don't loop
        # on it. Reset alongside _round_edit_count in _run_rounds.
        self._nudge_sent_this_round: bool = False
        # MCP bridge — spawned in run(), torn down in the finally clause.
        # When None, the coder runs with the core six tools only. The
        # bridge is best-effort: a failed start logs and we continue.
        self._mcp_bridge: Any = None
        # Per-round dedup state. Keyed by (tool_name, sorted_args_repr);
        # value is the call count for that signature within the current
        # round. Reset in _run_rounds alongside _round_edit_count. The
        # inner loop short-circuits a call with a synthetic result when
        # the count exceeds a tool-specific threshold — direct kill for
        # the search-loop pathology where the model repeats the same
        # zero-result query for many turns. See _DEDUP_THRESHOLDS below.
        self._tool_call_signatures: dict[tuple[str, str], int] = {}
        # Per-round flag for the 75%-budget zero-edit warning. Set True
        # the turn after the warning is injected so it fires at most
        # once per round.
        self._budget_warning_sent_this_round: bool = False

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _is_cancelled(self) -> bool:
        return bool(getattr(self.state, "cancelled", False))

    def _is_graceful_stop_requested(self) -> bool:
        """Check the AIM graceful-stop flag.

        Imported lazily so tests that don't have the aim package on the path
        (or that want to mock the file system) don't pay an import cost.
        Returns False on any error — better to keep working than to bail
        on a transient filesystem hiccup.
        """
        try:
            from aim.graceful_stop import is_stop_requested
            return is_stop_requested()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full outer fix-round loop. Acquires GPU priority for duration."""
        acquire_coder_priority()
        self._start_mcp_bridge()
        try:
            self._log(f"[OllamaCoder] Starting with model={self.model}, "
                      f"host={self.host}, "
                      f"max_rounds={self.max_rounds}, max_turns={self.max_turns}")
            self._run_rounds()
        finally:
            self._stop_mcp_bridge()
            release_coder_priority()

    def _start_mcp_bridge(self) -> None:
        """Spawn the MCP context server for this coder session.

        Best-effort: a failed start is logged and the coder runs
        with the core six tools only. We never let an MCP failure
        kill a story.
        """
        try:
            from .mcp_bridge import MCPBridge
            bridge = MCPBridge(log=self._log)
            if bridge.start():
                self._mcp_bridge = bridge
            else:
                self._log("[OllamaCoder] MCP bridge unavailable — "
                          "running with core tools only")
                self._mcp_bridge = None
        except Exception as exc:  # noqa: BLE001
            self._log(f"[OllamaCoder] MCP bridge import/start raised: {exc} "
                      f"— running with core tools only")
            self._mcp_bridge = None

    def _stop_mcp_bridge(self) -> None:
        """Tear down the MCP bridge if we spawned one."""
        bridge = self._mcp_bridge
        self._mcp_bridge = None
        if bridge is None:
            return
        try:
            bridge.stop()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[OllamaCoder] MCP bridge stop raised: {exc}")

    # ------------------------------------------------------------------
    # Outer round loop
    # ------------------------------------------------------------------

    def _run_rounds(self) -> None:
        test_output = ""
        failing_tests: list[str] = []
        changed_files: list[str] = []

        # Drift detection: track the last DRIFT_WINDOW rounds. We compare
        # had_commit / failing_set / changed_set across rounds to decide
        # whether the model is making forward progress. See _check_drift.
        drift_history: list[dict[str, Any]] = []
        prev_commit_count = self._count_branch_commits()

        for round_num in range(self.max_rounds):
            if self._is_cancelled():
                self._log("[OllamaCoder] Cancelled — stopping")
                return
            # Graceful stop check between rounds — bail out at a clean
            # boundary before kicking off the next inner loop. Story
            # stays in "executing" so the worker's post-handler resets it.
            if self._is_graceful_stop_requested():
                self._log(
                    f"[OllamaCoder] Graceful stop detected before round {round_num} — "
                    f"breaking out of round loop"
                )
                return
            self._log(f"[OllamaCoder] --- Round {round_num} ---")

            # Snapshot which files existed at the start of the round. Used by
            # _commit_changes to label each file as edit/create in the commit
            # message. This must happen BEFORE the inner loop runs, otherwise
            # we'd label every file as "edit" (since the model just wrote it).
            pre_round_existing: set[str] = {
                str(self.project_root / f)
                for f in self._git_branch_files_or_empty()
            }

            # Build prompt for this round
            if round_num == 0:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_initial_prompt()
            else:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_fix_prompt(round_num, test_output, changed_files, failing_tests)

            # Reset the per-round edit counter. Incremented inside
            # _execute_tool when the model successfully calls write_file
            # or edit_file. Snapshot to prev_round_edit_count after the
            # inner loop runs so the *next* round's _build_fix_prompt
            # can see whether the model actually edited anything.
            self._round_edit_count = 0
            self._nudge_sent_this_round = False
            self._tool_call_signatures = {}
            self._budget_warning_sent_this_round = False

            # Run inner tool-calling loop
            messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
            finished = self._run_inner_loop(system_prompt, messages, round_num)

            self.prev_round_edit_count = self._round_edit_count

            inner_exhausted = not finished
            if inner_exhausted:
                self._log(f"[OllamaCoder] Round {round_num}: inner loop exhausted max turns")

            # Checkpoint commit tag
            self._tag_round_commits(round_num)

            # Run tests
            self._log(f"[OllamaCoder] Round {round_num}: running tests...")
            test_result = self._run_pytest()
            test_output = test_result["output"]
            failing_tests = test_result["failing"]
            changed_files = self._get_changed_files()

            if test_result["passed"]:
                self._log(f"[OllamaCoder] Round {round_num}: tests PASSED ✓")
                # Only commit if tests pass. The gate MUST check the
                # working-tree state (tracked-modified + untracked) via
                # _files_to_stage(), NOT ``main...HEAD`` (changed_files) —
                # on round 0 nothing is committed yet, so ``_get_changed_files``
                # returns []. The first deploy of this code (commit 38bb8a0)
                # gated on changed_files and silently fell through to the
                # post-coder.run() safety net (_auto_commit_uncommitted),
                # which used the OLD ``[<id>] <description>`` message format
                # instead of the deterministic
                # ``[<id>] <title> <model> r<N>: <verb> <file>`` format.
                files_to_stage = self._files_to_stage()
                if files_to_stage:
                    self._commit_changes(
                        round_num,
                        files=files_to_stage,
                        pre_round_existing=pre_round_existing,
                    )
                return

            self._log(
                f"[OllamaCoder] Round {round_num}: {len(failing_tests)} test(s) failing"
            )
            if round_num < self.max_rounds - 1:
                hint = _classify_error_hint(test_output)
                if hint:
                    self._log(f"[OllamaCoder] Hint: {hint}")

            # Drift detection: record this round's signals and check whether
            # the last DRIFT_WINDOW rounds together meet an abort condition.
            new_commit_count = self._count_branch_commits()
            had_commit = new_commit_count > prev_commit_count
            prev_commit_count = new_commit_count

            drift_history.append({
                "round_num": round_num,
                "had_commit": had_commit,
                "failing_set": frozenset(failing_tests),
                "changed_set": frozenset(changed_files),
                "changed_count": len(changed_files),
                "inner_exhausted": inner_exhausted,
            })
            if len(drift_history) > DRIFT_WINDOW:
                drift_history = drift_history[-DRIFT_WINDOW:]

            drift_reason = self._check_drift(drift_history)
            if drift_reason is not None:
                self._log(
                    f"[OllamaCoder] Drift detected — aborting after round "
                    f"{round_num} (4-round window, no progress)"
                )
                self._log(f"[OllamaCoder] Drift reason: {drift_reason}")
                self._log(
                    f"[OllamaCoder] Stopped at round {round_num + 1}/"
                    f"{self.max_rounds} (drift) — story will be marked failed"
                )
                break
        else:
            # ``for/else`` runs only when the loop completes without ``break``.
            # The previous version logged "Exhausted N rounds" unconditionally
            # at the end, including on the drift-break path — confusing
            # because the model only ran a handful of rounds before drift
            # killed it. Now exhaustion and drift each get their own message.
            self._log(
                f"[OllamaCoder] Exhausted {self.max_rounds} rounds — "
                f"story will be marked failed"
            )

    def _count_branch_commits(self) -> int:
        """Count commits on the current branch ahead of main.

        Returns 0 on any error (no git, detached HEAD, etc.) — drift detection
        will treat that as "no commit" which is the safe default.
        """
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", "main..HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return int((result.stdout or "0").strip() or "0")
        except Exception:
            return 0

    def _check_drift(self, history: list[dict[str, Any]]) -> str | None:
        """Return a short reason string if drift fires, else None.

        Only fires once we have a full DRIFT_WINDOW of rounds. Two conditions:

        A) No-progress drift: every round in the window had no commit AND the
           failing-test set stayed identical AND the changed-file set stayed
           identical. The model is grinding the same broken state.

        B) Empty-shop drift: every round in the window had no commit AND zero
           changed files AND at least DRIFT_EMPTY_EXHAUSTION_THRESHOLD rounds
           also exhausted the inner loop. Model is flailing without producing
           anything.
        """
        if len(history) < DRIFT_WINDOW:
            return None

        all_no_commit = all(not r["had_commit"] for r in history)
        if not all_no_commit:
            return None

        # Condition A
        first_failing = history[0]["failing_set"]
        first_changed = history[0]["changed_set"]
        same_failing = all(r["failing_set"] == first_failing for r in history)
        same_changed = all(r["changed_set"] == first_changed for r in history)
        if same_failing and same_changed:
            return (
                f"no_progress: {DRIFT_WINDOW} rounds with no commit, constant "
                f"failing-set ({len(first_failing)} tests), constant changed-set "
                f"({len(first_changed)} files)"
            )

        # Condition B
        all_empty = all(r["changed_count"] == 0 for r in history)
        exhaustion_count = sum(1 for r in history if r["inner_exhausted"])
        if all_empty and exhaustion_count >= DRIFT_EMPTY_EXHAUSTION_THRESHOLD:
            return (
                f"empty_shop: {DRIFT_WINDOW} rounds with no commit, zero changed "
                f"files, {exhaustion_count} inner-loop exhaustions"
            )

        return None

    # ------------------------------------------------------------------
    # Inner tool-calling loop
    # ------------------------------------------------------------------

    def _run_inner_loop(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        round_num: int,
    ) -> bool:
        """Run up to max_turns turns of tool-calling. Returns True if finish() called."""
        self_reviewed = False

        # 75%-budget warning fires once per round when we've burned 3/4 of
        # the turn budget and still made zero edits. Pre-compute the turn
        # index at which to inject — `int(max_turns * 0.75)` matches the
        # plan; for max_turns < 4 it lands at the start which is fine
        # (those budgets are too small for the warning to be useful but
        # we don't want a div-by-zero).
        budget_warning_turn = max(1, int(self.max_turns * 0.75))

        for turn in range(self.max_turns):
            if self._is_cancelled():
                self._log("[OllamaCoder] Cancelled — stopping inner loop")
                return False

            # 75%-budget zero-edit warning. Inject before the next chat
            # call so the model sees it as the most recent user-role
            # message and has a chance to actually edit before the wall.
            if (
                turn == budget_warning_turn
                and self._round_edit_count == 0
                and not self._budget_warning_sent_this_round
            ):
                self._budget_warning_sent_this_round = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You have used 75% of your turns this round and "
                        "have not edited any file. Stop exploring and "
                        "start editing — pick the most likely file, read "
                        "it once if you haven't, then make your change "
                        "with edit_file or write_file. Reading more files "
                        "will not finish the task."
                    ),
                })
                self._log("[OllamaCoder] 75%-budget zero-edit warning injected")

            messages = self._trim_context(messages)
            response = self._chat_with_tools(system_prompt, messages)
            if self._is_cancelled():
                self._log("[OllamaCoder] Cancelled after Ollama call — stopping")
                return False
            if response is None:
                self._log(f"[OllamaCoder] Round {round_num} turn {turn}: Ollama returned None, aborting")
                return False

            # Extract think content and strip
            content = response.get("message", {}).get("content", "") or ""
            think_match = _THINK_RE.search(content)
            if think_match:
                logger.debug("[OllamaCoder][think] %s", think_match.group()[:200])
            content = _THINK_RE.sub("", content).strip()

            # Extract tool calls — native format first, fallback JSON parse
            tool_calls = response.get("message", {}).get("tool_calls") or []
            if not tool_calls and content:
                tool_calls = _parse_tool_calls_from_content(content)

            if not tool_calls:
                # No tool calls — model is narrating; append as assistant message and nudge
                messages.append({"role": "assistant", "content": content})
                if content:
                    self._log(f"[OllamaCoder] {content[:200]}")
                messages.append({
                    "role": "user",
                    "content": "Please use the available tools to make progress. Call finish() when done.",
                })
                continue

            # Append assistant message with tool_calls
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            # Execute each tool call
            finished = False
            for tc in tool_calls:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else _safe_json(raw_args)

                self._log(f"[OllamaCoder] → {name}({_fmt_args(args)})")
                # Per-round duplicate-call suppression. If the same
                # (name, args) signature has already been seen too many
                # times this round, return a synthetic "[duplicate
                # suppressed]" result instead of running the tool. Direct
                # kill for the search-loop pathology where the model
                # repeats a zero-result query for many turns.
                dup_result = self._check_dedup(name, args)
                if dup_result is not None:
                    result: str = dup_result
                    self._log(f"[OllamaCoder] ← (dup-suppressed) {result[:200]}")
                else:
                    result = self._execute_tool(name, args)
                    self._log(f"[OllamaCoder] ← {str(result)[:300]}")
                    # On a successful edit/write, drop dedup signatures
                    # that mention the same path so the model can re-read
                    # the now-changed file without being suppressed.
                    if (
                        name in ("edit_file", "write_file")
                        and not str(result).startswith("ERROR")
                    ):
                        self._clear_dedup_for_path(args.get("path", ""))

                # Append tool result
                messages.append({
                    "role": "tool",
                    "content": str(result),
                })

                if name == "finish":
                    # Mid-round zero-edit nudge: model called finish() but
                    # never invoked edit_file or write_file this round.
                    # Reading and searching are not work; the round-2
                    # fallback nudge in _build_fix_prompt only fires AFTER
                    # an entire wasted round. Catch it now (turn 0+) so
                    # the model gets one chance to actually edit before
                    # we waste the round. Fire at most once per round.
                    if (
                        self._round_edit_count == 0
                        and not self._nudge_sent_this_round
                        and turn < self.max_turns - 2
                    ):
                        self._nudge_sent_this_round = True
                        messages.append({
                            "role": "user",
                            "content": (
                                "You called finish() but you have not made any "
                                "edits this round. Reading and searching are not "
                                "work. Use edit_file or write_file to change the "
                                "source, then call finish() again. If after "
                                "careful reading you genuinely believe no code "
                                "change is needed, say so explicitly in "
                                "finish(summary=...) on your next call — but "
                                "think twice; the tests are not currently passing."
                            ),
                        })
                        self._log("[OllamaCoder] Mid-round no-edit nudge injected")
                        break  # restart the turn loop after the nudge
                    # Self-review pass on round 0 if model finished very quickly
                    if round_num == 0 and not self_reviewed and turn < 3:
                        self_reviewed = True
                        messages.append({
                            "role": "user",
                            "content": (
                                "Before we finalize: review your changes for syntax errors, "
                                "missing imports, and edge cases the tests might hit. "
                                "Fix any issues you find, then call finish() again."
                            ),
                        })
                        self._log("[OllamaCoder] Self-review pass injected")
                        break  # restart the turn loop after self-review nudge
                    finished = True

            if finished:
                return True

        return False

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _effective_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the tool definitions to send to Ollama on each chat call.

        Always includes the six core tools (read_file, write_file,
        edit_file, list_files, search_code, finish). Appends MCP-
        discovered tools when the bridge is up. Built fresh on every
        call so a bridge that dies mid-run gracefully degrades to
        core-tools-only on the next chat.

        Defensive against bypassed-__init__ test fixtures: tests use
        ``OllamaCoder.__new__(OllamaCoder)`` to skip __init__ when
        exercising the chat layer in isolation, so ``_mcp_bridge``
        may not exist on the instance — getattr fallback covers it.
        """
        tools: list[dict[str, Any]] = list(_TOOLS)
        bridge = getattr(self, "_mcp_bridge", None)
        if bridge is not None:
            try:
                tools.extend(bridge.tool_definitions)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"[OllamaCoder] MCP bridge tool_definitions raised: {exc}"
                )
        return tools

    def _tool_signature(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Build a stable signature key for a tool call.

        Sorts arg keys so ``{"a":1,"b":2}`` and ``{"b":2,"a":1}`` collide.
        Uses ``repr`` on the sorted-items tuple to handle unhashable values
        (lists, dicts) gracefully without raising. The args dict can come
        from the model in any shape; we never want signature-building to
        crash dispatch.
        """
        try:
            sig = repr(sorted(args.items()))
        except Exception:  # noqa: BLE001
            sig = repr(args)
        return (name, sig)

    def _check_dedup(self, name: str, args: dict[str, Any]) -> str | None:
        """Return a synthetic tool result if this call is a duplicate.

        Increments the per-round counter for ``(name, args)`` and, when the
        count exceeds the threshold for that tool, returns a synthetic
        "[duplicate suppressed]" message so the model is forced to change
        approach. Returns ``None`` (let the call proceed) when the tool is
        not in the threshold table or the count is still under the limit.
        """
        threshold = _DEDUP_THRESHOLDS.get(name)
        if threshold is None:
            return None
        sig = self._tool_signature(name, args)
        count = self._tool_call_signatures.get(sig, 0) + 1
        self._tool_call_signatures[sig] = count
        if count <= threshold:
            return None
        # Render the args compactly for the synthetic message — the model
        # should see exactly what it just repeated.
        return (
            f"[duplicate suppressed] You already called "
            f"{name}({_fmt_args(args)}) {count - 1} time(s) this round. "
            f"Repeating the same call will not return new results. Try a "
            f"different argument, a different file, or read the file "
            f"directly instead of searching."
        )

    def _clear_dedup_for_path(self, path: str) -> None:
        """After a successful edit/write to ``path``, drop any signatures
        whose args reference that path so subsequent re-reads of the now-
        changed file are allowed. Without this, the model can't sensibly
        re-read a file it just edited.
        """
        if not path:
            return
        marker = repr(path)
        keys_to_drop = [
            key for key in self._tool_call_signatures
            if marker in key[1]
        ]
        for key in keys_to_drop:
            del self._tool_call_signatures[key]

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        # MCP-discovered tools (e.g. aiw_purpose, codebase_index) route
        # through the bridge before falling through to the core six.
        # Bridge dispatch comes first so a future MCP server can shadow
        # core tools intentionally if it ever needs to.
        bridge = self._mcp_bridge
        if bridge is not None:
            try:
                if bridge.has_tool(name):
                    return bridge.call(name, args)
            except Exception as exc:  # noqa: BLE001
                # Don't let a bridge wobble kill the dispatch — fall
                # through to the core-tool path which will return
                # "unknown tool" if the name isn't core either.
                self._log(f"[OllamaCoder] MCP dispatch raised for {name}: {exc}")

        try:
            # Note: there is no per-tool-call git/branch readiness gate here.
            # The AIW worker runs in an isolated A/B worktree (see
            # idea_board/ab_worktree.py and idea_board/ab_executor.py); the
            # branch is created once by execute_idea(branch_suffix=...) and
            # the worktree is torn down at the end of the run. A previous
            # gate (TK-1095) re-checked git-clean and created a fresh branch
            # on EVERY tool call, which (a) targeted the main repo instead
            # of the worktree, (b) doubled the TK- prefix when idea_id was
            # already "TK-NNNN", and (c) created hundreds of stale branches
            # per run when the model invoked tools in a tight loop. Removed.
            if name == "read_file":
                # The model frequently passes ``offset`` and ``length`` even
                # when the schema doesn't declare them, expecting Read-tool-
                # like pagination.  Honor those args (line-based) instead of
                # silently dropping them — pre-fix the model wasted entire
                # rounds re-reading the same head of a file.
                return self._tool_read_file(
                    args.get("path", ""),
                    offset=args.get("offset"),
                    length=args.get("length") or args.get("limit"),
                )
            elif name == "write_file":
                result = self._tool_write_file(args.get("path", ""), args.get("content", ""))
                if not str(result).startswith("ERROR"):
                    self._round_edit_count += 1
                return result
            elif name == "edit_file":
                result = self._tool_edit_file(
                    args.get("path", ""),
                    args.get("old_string", ""),
                    args.get("new_string", ""),
                )
                if not str(result).startswith("ERROR"):
                    self._round_edit_count += 1
                return result
            elif name == "list_files":
                return self._tool_list_files(args.get("path", "."), args.get("pattern", "*"))
            elif name == "search_code":
                return self._tool_search_code(
                    args.get("pattern", ""),
                    args.get("path", "."),
                    args.get("file_pattern", "*.py"),
                )
            elif name == "finish":
                return f"Finished: {args.get('summary', '')}"
            elif name in ("run_bash", "verify_git_clean", "create_branch"):
                # These were available in older versions of the coder but the
                # harness now owns all mechanical work (branching, testing,
                # committing). Tell the model explicitly so it stops retrying.
                return (
                    f"ERROR: unknown tool '{name}'. The harness owns "
                    f"branches/tests/commits — you have no shell. Use "
                    f"read_file / write_file / edit_file / list_files / "
                    f"search_code, and call finish(summary=...) when done."
                )
            else:
                return f"ERROR: unknown tool '{name}'"
        except EnvironmentReadyError as e:
            # Re-raise readiness errors
            raise e
        except Exception as exc:
            return f"ERROR: {exc}"

    # Project subdirectories that we recognize as legitimate re-anchor points
    # when the model emits an absolute path with a typo'd or wrong project prefix.
    # Order matters: deeper / more specific anchors first.
    _REANCHOR_SEGMENTS = (
        "local-agent",
        "docs",
        "tests",
        "agent",
        "idea_board",
        "aim",
        "aiv",
    )

    def _resolve_path(self, path: str) -> Path:
        """Return absolute Path within project_root.

        Path repair strategy (in order):
        1. Relative path → join with project_root (normal case).
        2. Absolute path inside project_root → use as-is.
        3. Absolute path *outside* project_root but containing a recognizable
           project segment (e.g. ``/Users/gman/code/typo-path/local-agent/...``)
           → re-anchor at that segment under project_root. This handles the case
           where the model typo's the worktree path but the suffix is correct.
        4. Otherwise → reject by clamping to project_root/<basename>.

        Step 3 exists because UUID-suffixed worktree paths (``technomancer-aiw-<uuid>``)
        are too long for the model to retype reliably across many turns, and the
        sandbox would otherwise reject every tool call after the first typo.
        """
        p = Path(path)
        project_root_resolved = self.project_root.resolve()

        if not p.is_absolute():
            # Repair the common "double-prefix" bug:  ``project_root`` is
            # already ``.../local-agent``, but the model habitually writes
            # paths like ``local-agent/agent/foo.py`` because that's how the
            # files appear in the repo and in the prompt context.  Joining
            # those naively produces ``local-agent/local-agent/agent/foo.py``
            # which never exists.  Detect that case and strip the redundant
            # leading ``local-agent`` segment.
            if (
                project_root_resolved.name == "local-agent"
                and p.parts
                and p.parts[0] == "local-agent"
            ):
                p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
            return (self.project_root / p).resolve()

        resolved = p.resolve()
        try:
            resolved.relative_to(project_root_resolved)
            return resolved
        except ValueError:
            pass

        # Try to re-anchor: find a known project segment in the path and rebuild
        # the suffix under project_root.
        parts = p.parts
        for anchor in self._REANCHOR_SEGMENTS:
            if anchor in parts:
                idx = parts.index(anchor)
                tail_parts = parts[idx:]
                # Avoid the double-prefix bug: if project_root already ends in
                # the anchor segment (e.g. project_root=.../local-agent and
                # anchor='local-agent'), strip it from tail so we don't produce
                # .../local-agent/local-agent/...
                if (
                    project_root_resolved.name == anchor
                    and len(tail_parts) > 1
                ):
                    tail_parts = tail_parts[1:]
                tail = Path(*tail_parts) if tail_parts else Path(".")
                repaired = (project_root_resolved / tail).resolve()
                # Defense-in-depth: ensure the repaired path is still inside project_root.
                try:
                    repaired.relative_to(project_root_resolved)
                except ValueError:
                    continue
                logger.info(
                    "[OllamaCoder] Path %s re-anchored to %s (anchor=%s)",
                    path, repaired, anchor,
                )
                return repaired

        # Could not repair — clamp to project_root/<basename> as a last resort.
        logger.warning("[OllamaCoder] Path %s escapes project_root, rejected", path)
        return project_root_resolved / Path(path).name

    def _tool_read_file(
        self,
        path: str,
        offset: Any = None,
        length: Any = None,
    ) -> str:
        """Read a file, with optional line-based pagination.

        ``offset`` is a 1-indexed line number (matching the Read tool the
        model is familiar with from Claude Code).  ``length`` is the number
        of lines to return.  Both may arrive as strings — coerce safely.
        When neither is provided, behavior matches the original
        whole-file-with-truncation read.
        """
        full = self._resolve_path(path)
        if not full.exists():
            return f"ERROR: file not found: {path}"
        if full.is_dir():
            return f"ERROR: file not found: {path} (is a directory)"
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: {exc}"

        offset_int = _coerce_int(offset)
        length_int = _coerce_int(length)

        if offset_int is not None or length_int is not None:
            lines = content.splitlines(keepends=True)
            start = max((offset_int or 1) - 1, 0)
            if start >= len(lines):
                return (
                    f"(offset {offset_int} past end of file; "
                    f"file has {len(lines)} lines)"
                )
            end = start + length_int if length_int is not None else len(lines)
            sliced = "".join(lines[start:end])
            if len(sliced) > READ_FILE_MAX_CHARS:
                sliced = sliced[:READ_FILE_MAX_CHARS] + f"\n... [truncated at {READ_FILE_MAX_CHARS} chars]"
            header = f"(lines {start + 1}-{min(end, len(lines))} of {len(lines)})\n"
            return header + sliced

        if len(content) > READ_FILE_MAX_CHARS:
            content = content[:READ_FILE_MAX_CHARS] + f"\n... [truncated at {READ_FILE_MAX_CHARS} chars]"
        return content

    def _tool_write_file(self, path: str, content: str) -> str:
        full = self._resolve_path(path)
        # Defense-in-depth against the "double-prefix" path bug: refuse to
        # write into ``.../local-agent/local-agent/...``.  ``_resolve_path``
        # now strips a leading ``local-agent/`` from relative inputs when
        # ``project_root`` already ends in ``local-agent``, but earlier
        # versions of this code created phantom nested trees that were then
        # committed.  Keep the guard so future regressions surface loudly.
        path_str = str(full).replace("\\", "/")
        if "/local-agent/local-agent/" in path_str:
            return (
                f"ERROR: refusing to write nested phantom path '{path}' — "
                f"resolved to {full}.  Drop the leading 'local-agent/' prefix "
                f"(project_root is already inside local-agent)."
            )
        # Guard: pytest only collects files starting with test_ or ending _test.py.
        # If the file is under a tests/ dir or contains def test_* functions,
        # require it to follow that naming so the work isn't silently invisible.
        is_under_tests = "/tests/" in path_str
        looks_like_tests = bool(re.search(r"^def test_\w+", content, re.MULTILINE))
        if (is_under_tests or looks_like_tests) and full.name.endswith(".py"):
            allowed_test_names = {"conftest.py", "__init__.py"}
            is_pytest_named = full.name.startswith("test_") or full.name.endswith("_test.py")
            is_helper = full.name in allowed_test_names
            if not is_pytest_named and not is_helper:
                return (
                    f"ERROR: refusing to write test-shaped file '{full.name}' — "
                    f"pytest will not collect it. Rename to test_{full.name} "
                    f"(or {full.stem}_test.py) and try again."
                )
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except Exception as exc:
            return f"ERROR: {exc}"

    def _tool_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        if not old_string:
            return "ERROR: old_string must not be empty"
        full = self._resolve_path(path)
        if not full.exists():
            return f"ERROR: file not found: {path}"
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: {exc}"
        if old_string not in content:
            return f"ERROR: old_string not found in {path}"
        new_content = content.replace(old_string, new_string, 1)
        try:
            full.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            return f"ERROR: {exc}"
        return f"Edited {path}"

    def _tool_run_bash(self, command: str) -> str:
        # The model habitually prefixes commands with `cd <project_root> && `
        # even though the subprocess already runs there. Strip that prefix
        # before allowlist checks so we don't reject otherwise-valid commands.
        # Only handle the simple `cd <path> && rest` shape — anything more
        # exotic still gets rejected by the allowlist.
        cd_prefix_match = re.match(r"\s*cd\s+\S+\s*&&\s*(.+)", command, re.DOTALL)
        if cd_prefix_match:
            command = cd_prefix_match.group(1).strip()
        cmd_lower = command.lower().strip()
        # Blocklist check
        for blocked in BASH_BLOCKLIST:
            if blocked in cmd_lower:
                return f"BLOCKED: command contains '{blocked}'"
        # Allowlist check — require word boundary so e.g. `pythonista` doesn't
        # match `python`. A bare command (`git`) and a command with arguments
        # (`git status`) both qualify; nothing else does.
        allowed = any(
            cmd_lower == prefix or cmd_lower.startswith(prefix + " ")
            for prefix in BASH_ALLOWLIST
        )
        if not allowed:
            first_token = cmd_lower.split(maxsplit=1)[0] if cmd_lower else ""
            guidance = BASH_SHAPE_GUIDANCE.get(first_token)
            msg_lines = [
                f"BLOCKED: '{command[:80]}'",
                f"  Reason: bash commands must start with one of {BASH_ALLOWLIST}.",
            ]
            if guidance:
                msg_lines.append(f"  Better tool: {guidance}")
            else:
                msg_lines.append(
                    "  Better tool: read/write file ops belong to the structured "
                    "tools (`read_file`, `edit_file`, `list_files`, `search_code`). "
                    "Bash is for `git`, `pytest`, and `python` only."
                )
            return "\n".join(msg_lines)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=120,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if len(output) > 4000:
                output = output[-4000:]
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 120s"
        except Exception as exc:
            return f"ERROR: {exc}"

    def _tool_list_files(self, path: str, pattern: str = "*") -> str:
        full = self._resolve_path(path)
        if not full.is_dir():
            return f"ERROR: not a directory: {path}"
        try:
            files = list(full.glob(pattern))[:LIST_FILES_MAX]
            return "\n".join(f.as_posix() for f in files) or "(empty)"
        except Exception as exc:
            return f"ERROR: {exc}"

    def _tool_search_code(self, pattern: str, path: str = ".", file_pattern: str = "*.py") -> str:
        search_dir = self._resolve_path(path)
        if not search_dir.exists():
            return f"ERROR: search path does not exist: {path}"
        # Prefer ripgrep if available (fast, multi-file). Fall back to native
        # rglob+regex when not present. Previous implementation invoked
        # ``python -m grep`` which always fails (no such module), then short-
        # circuited at returncode==1 with empty stdout — meaning every search
        # returned "(no matches)" regardless of whether the symbol existed.
        try:
            result = subprocess.run(
                ["rg", "-n", pattern, "--glob", file_pattern, str(search_dir)],
                capture_output=True, text=True, cwd=str(self.project_root), timeout=30,
            )
            if result.returncode in (0, 1):
                out = result.stdout[:4000]
                return out or "(no matches)"
        except FileNotFoundError:
            pass
        except Exception:
            pass
        # Manual fallback — used when rg is not installed.
        # ``search_dir`` may be a file (when the model passes a specific file
        # path) or a directory.  ``Path.rglob`` returns nothing on a file, so
        # we have to iterate explicitly.
        matches: list[str] = []
        if search_dir.is_file():
            files_iter: Iterable[Path] = [search_dir]
        else:
            files_iter = search_dir.rglob(file_pattern)
        try:
            for f in files_iter:
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if re.search(pattern, line):
                            matches.append(f"{f.as_posix()}:{i}: {line.strip()}")
                            if len(matches) >= 50:
                                break
                except Exception:
                    pass
                if len(matches) >= 50:
                    break
        except Exception as exc:
            return f"ERROR: {exc}"
        return "\n".join(matches) or "(no matches)"

    # ------------------------------------------------------------------
    # Ollama /api/chat
    # ------------------------------------------------------------------

    def _chat_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """POST to Ollama's /api/chat with bounded cancel latency.

        Streams the response so we can poll ``self._is_cancelled()``
        between chunks and bail out within a few seconds when the
        worker flips the cancel flag. The previous non-streaming
        implementation used ``timeout=900`` as a single read timeout —
        which meant a wedged generation could ignore cancellation for
        up to 15 minutes, leaving the worker locked in "refusing to
        stack" mode and forcing a full-stack restart.

        Behaviour:
        - Streams chunks with a short per-chunk read timeout
          (CHAT_CHUNK_TIMEOUT). Between chunks we check the cancel
          flag and abort if set.
        - Reassembles the streamed chunks into the same dict shape
          ``stream=False`` would have returned, so the caller never
          knows whether streaming was used.
        - HTTP 500 retry backoff is preserved (Ollama's transient
          tool-format bug). Network errors still return None.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "tools": self._effective_tool_definitions(),
            "stream": True,
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.2},
            # keep_alive=-1 pins the model resident in Ollama's scheduler
            # forever (no idle eviction). Prevents the 6+ second reload
            # penalty + full KV cache rebuild between coder rounds. Other
            # callers (brain, splitter, web_search, quality_test) use
            # shorter keep_alive values and may briefly evict this runner
            # if they load a different model, but during a coder's active
            # run the runner will not idle-unload.
            "keep_alive": -1,
        }
        # Pre-compute prompt size once so every retry attempt logs the
        # same number. ``json.dumps`` is overkill here — we only need a
        # rough char count to spot prompts that have ballooned past what
        # the model can chew through inside CHAT_CHUNK_TIMEOUT.
        prompt_chars = sum(
            len(str(m.get("content") or "")) for m in body["messages"]
        )
        msg_count = len(body["messages"])
        for attempt in range(4):  # 1 initial + 3 retries
            response = self._stream_chat_once(
                body,
                prompt_chars=prompt_chars,
                msg_count=msg_count,
                attempt=attempt,
            )
            if response is None:
                return None
            status, payload = response
            if status == "ok":
                return payload
            if status == "cancelled":
                logger.info("[OllamaCoder] cancelled during streaming chat")
                return None
            if status == "retry_500" and attempt < 3:
                delay = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "[OllamaCoder] HTTP 500 (attempt %d/4), retrying in %ds: %s",
                    attempt + 1, delay, str(payload)[:120],
                )
                # Sleep with cancel polling so a cancel during back-off
                # also responds quickly.
                if self._sleep_with_cancel(delay):
                    return None
                continue
            # Any other status is a non-retryable failure.
            return None
        return None

    # Per-chunk read timeout for the streaming chat. Short enough that
    # a stuck connection releases promptly, long enough that the model's
    # natural inter-chunk gap (typically <2 s for token-by-token
    # streaming) doesn't trigger spurious timeouts.
    CHAT_CHUNK_TIMEOUT = 30

    # Hard cap on total streaming wall-clock. Defends against the
    # pathological case where Ollama keeps emitting heartbeat-shaped
    # chunks but never returns a "done" event. 15 minutes matches the
    # old non-streaming timeout=900 so we don't change the upper bound,
    # only the cancel-response latency.
    CHAT_TOTAL_TIMEOUT = 900

    def _stream_chat_once(
        self,
        body: dict[str, Any],
        *,
        prompt_chars: int = 0,
        msg_count: int = 0,
        attempt: int = 0,
    ) -> tuple[str, Any] | None:
        """Run one streaming /api/chat attempt.

        Returns ``None`` on a network error (caller maps to None).
        Otherwise returns ``(status, payload)`` where ``status`` is one of:

        - ``"ok"``: payload is the assembled response dict.
        - ``"cancelled"``: payload is None; cancel flag fired mid-stream.
        - ``"retry_500"``: payload is the response text; caller may retry.
        - ``"http_error"``: payload is the response text; non-retryable.
        - ``"non_json"``: payload is the offending line; non-retryable.

        Diagnostic logging: every attempt logs prompt size + model + host
        at INFO before the request fires; on success, on network error,
        and on mid-stream timeout we log how long it took to fail and
        how far we got. This is the data we need to decide whether the
        timeout fix is "longer first-chunk timeout" vs "retry on read
        timeout" vs "smaller prompt".
        """
        url = f"{self.host}/api/chat"
        # Use a (connect, read) timeout tuple so the connect handshake
        # has its own short window. The read timeout governs how long
        # we wait between streamed chunks.
        request_timeout = (10, self.CHAT_CHUNK_TIMEOUT)
        t_request_start = time.time()
        logger.info(
            "[OllamaCoder] chat request: model=%s host=%s attempt=%d "
            "messages=%d prompt_chars=%d num_ctx=%d "
            "chunk_timeout=%ds total_timeout=%ds",
            self.model, self.host, attempt, msg_count, prompt_chars,
            self.num_ctx, self.CHAT_CHUNK_TIMEOUT, self.CHAT_TOTAL_TIMEOUT,
        )
        try:
            # ``with`` ensures the underlying socket is closed even if
            # we bail out of the iter_lines loop on cancellation.
            with requests.post(
                url, json=body, stream=True, timeout=request_timeout,
            ) as r:
                if r.status_code == 500:
                    return ("retry_500", r.text)
                if r.status_code != 200:
                    logger.warning(
                        "[OllamaCoder] HTTP %d: %s", r.status_code, r.text[:200],
                    )
                    return ("http_error", r.text)
                return self._consume_chat_stream(
                    r,
                    t_request_start=t_request_start,
                    prompt_chars=prompt_chars,
                )
        except requests.RequestException as exc:
            elapsed = time.time() - t_request_start
            logger.warning(
                "[OllamaCoder] Network error after %.1fs: model=%s host=%s "
                "prompt_chars=%d attempt=%d exc=%s",
                elapsed, self.model, self.host, prompt_chars, attempt, exc,
            )
            return None

    def _consume_chat_stream(
        self,
        response: "requests.Response",
        *,
        t_request_start: float | None = None,
        prompt_chars: int = 0,
    ) -> tuple[str, Any]:
        """Read NDJSON chunks from a streaming chat response.

        Ollama emits one JSON object per line. Each chunk has a
        ``message`` field (with incremental ``content`` and possibly
        ``tool_calls``) and a final chunk has ``done: true``. We
        accumulate everything and reassemble a non-streaming-shaped
        dict for the caller.

        Cancel polling: we check :meth:`_is_cancelled` between chunks
        plus once per ``iter_lines`` iteration. ``iter_lines`` itself
        blocks until either a line arrives or the per-chunk read
        timeout fires; on timeout it raises a ``RequestException``
        which the caller maps to None — so cancel latency is bounded
        by ``CHAT_CHUNK_TIMEOUT`` in the worst case.

        Diagnostic timing: ``t_request_start`` (when caller fired
        ``requests.post``) lets us record first-chunk latency, which
        is dominated by Ollama's prompt-eval pass. On a 30b model with
        a 30k-token prompt, prompt-eval can easily exceed
        ``CHAT_CHUNK_TIMEOUT`` even though the model is healthy —
        that's the hypothesis driving these logs.
        """
        if t_request_start is None:
            t_request_start = time.time()
        accumulated_content: list[str] = []
        accumulated_tool_calls: list[Any] = []
        final_message: dict[str, Any] = {}
        final_metadata: dict[str, Any] = {}
        deadline = time.time() + self.CHAT_TOTAL_TIMEOUT

        # Stream-progress counters used both for INFO logging on success
        # and for forensic logging on timeout.
        first_chunk_time: float | None = None
        chunks_received = 0
        last_chunk_time = t_request_start

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                # First check: cancellation. Bail before doing any
                # parsing work so the worker's join window is short.
                if self._is_cancelled():
                    response.close()
                    return ("cancelled", None)

                # Hard wall-clock cap. Prevents a stream that keeps
                # producing keep-alive lines from running forever.
                if time.time() > deadline:
                    logger.warning(
                        "[OllamaCoder] streaming chat exceeded "
                        "CHAT_TOTAL_TIMEOUT=%ds (model=%s host=%s "
                        "prompt_chars=%d chunks_received=%d "
                        "first_chunk_latency=%s)",
                        self.CHAT_TOTAL_TIMEOUT, self.model, self.host,
                        prompt_chars, chunks_received,
                        f"{first_chunk_time - t_request_start:.1f}s"
                        if first_chunk_time else "never",
                    )
                    response.close()
                    return ("http_error", "stream timeout")

                if not raw_line:
                    # Heartbeat / keep-alive — just loop and re-check
                    # cancel.
                    continue

                # First non-empty chunk: this is when prompt-eval
                # finishes and the model starts generating tokens.
                # The biggest knob in the timeout problem.
                now = time.time()
                if first_chunk_time is None:
                    first_chunk_time = now
                    logger.info(
                        "[OllamaCoder] first chunk arrived after %.1fs "
                        "(model=%s host=%s prompt_chars=%d)",
                        now - t_request_start, self.model, self.host,
                        prompt_chars,
                    )
                chunks_received += 1
                last_chunk_time = now

                try:
                    chunk = json.loads(raw_line)
                except (ValueError, TypeError):
                    logger.warning(
                        "[OllamaCoder] non-JSON stream line: %s",
                        str(raw_line)[:200],
                    )
                    return ("non_json", raw_line)

                msg = chunk.get("message") or {}
                if isinstance(msg, dict):
                    # Token streaming — append to running buffer.
                    if isinstance(msg.get("content"), str):
                        accumulated_content.append(msg["content"])
                    if isinstance(msg.get("tool_calls"), list):
                        accumulated_tool_calls.extend(msg["tool_calls"])
                    # Hold onto every field except content/tool_calls
                    # so role/etc. reach the caller intact.
                    for k, v in msg.items():
                        if k not in ("content", "tool_calls"):
                            final_message[k] = v

                if chunk.get("done"):
                    # Final chunk — capture the metadata fields
                    # (eval_count, prompt_eval_count, etc.) the caller
                    # may want.
                    for k, v in chunk.items():
                        if k != "message":
                            final_metadata[k] = v
                    break
        except requests.RequestException as exc:
            # Per-chunk read timeout or socket error mid-stream. The
            # forensic log: how far we got, and where the stall sits
            # (waiting on first chunk = prompt-eval, waiting on later
            # chunk = generation hiccup or remote socket issue).
            elapsed = time.time() - t_request_start
            time_since_last = time.time() - last_chunk_time
            logger.warning(
                "[OllamaCoder] stream read error after %.1fs: model=%s "
                "host=%s prompt_chars=%d chunks_received=%d "
                "first_chunk_latency=%s time_since_last_chunk=%.1fs "
                "stalled_on=%s exc=%s",
                elapsed, self.model, self.host, prompt_chars,
                chunks_received,
                f"{first_chunk_time - t_request_start:.1f}s"
                if first_chunk_time else "never",
                time_since_last,
                "first_chunk" if first_chunk_time is None else "subsequent_chunk",
                exc,
            )
            return ("http_error", str(exc))

        # Reassemble a dict that looks like the old non-streaming
        # response shape so callers don't need to know about streaming.
        final_message["content"] = "".join(accumulated_content)
        if accumulated_tool_calls:
            final_message["tool_calls"] = accumulated_tool_calls
        final_message.setdefault("role", "assistant")

        result: dict[str, Any] = dict(final_metadata)
        result["message"] = final_message

        # Success log: surfaces healthy first-chunk latency so we know
        # what "normal" looks like and can spot drift over time.
        total_elapsed = time.time() - t_request_start
        first_chunk_latency = (
            first_chunk_time - t_request_start
            if first_chunk_time is not None
            else 0.0
        )
        logger.info(
            "[OllamaCoder] chat done: model=%s host=%s prompt_chars=%d "
            "first_chunk_latency=%.1fs total_elapsed=%.1fs chunks=%d "
            "prompt_eval_count=%s eval_count=%s",
            self.model, self.host, prompt_chars,
            first_chunk_latency, total_elapsed, chunks_received,
            final_metadata.get("prompt_eval_count"),
            final_metadata.get("eval_count"),
        )
        return ("ok", result)

    def _sleep_with_cancel(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, returning True if cancelled meanwhile.

        Used for the HTTP 500 retry backoff so a cancel during a
        retry-wait responds within ~0.5 s instead of waiting out the
        full delay.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._is_cancelled():
                return True
            time.sleep(min(0.5, deadline - time.time()))
        return False

    # ------------------------------------------------------------------
    # Context trimming
    # ------------------------------------------------------------------

    def _trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop oldest tool messages when context is getting large."""
        # Rough token estimate: 4 chars ≈ 1 token
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        threshold = self.num_ctx * 4 * 0.75  # 75% of context in chars
        if total_chars <= threshold:
            return messages

        # Keep first user message + last 6 non-tool messages
        trimmed: list[dict[str, Any]] = []
        if messages:
            trimmed.append(messages[0])  # original story/fix prompt

        tool_messages = [m for m in messages[1:] if m.get("role") == "tool"]
        non_tool = [m for m in messages[1:] if m.get("role") != "tool"]

        # Drop oldest tool messages, keep last 6 non-tool
        trimmed.extend(non_tool[-6:])
        # Keep only most recent tool messages to stay under threshold
        trimmed.extend(tool_messages[-4:])

        logger.debug(
            "[OllamaCoder] Trimmed context: %d → %d messages",
            len(messages),
            len(trimmed),
        )
        return trimmed

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """System prompt for the coder.

        Pure context handoff. The harness owns all mechanical work
        (branching, running pytest, committing, merging). The model owns
        thinking: read files, decide what to change, write the patch.
        Crucially, we do NOT instruct the model to run git or pytest —
        it has no shell. Past versions of this prompt told the model to
        run those commands itself, which caused work duplication and one
        842-stale-branch runaway when a per-tool readiness gate misfired
        (commit 22f4cbe).
        """
        return (
            "You are an expert Python software engineer implementing Jira stories.\n"
            "\n"
            "The harness has already set up an isolated worktree on a clean branch "
            "for this story. After you call finish(), the harness will run the test "
            "suite and commit your changes. You do NOT run git, you do NOT run "
            "tests, and you have no shell.\n"
            "\n"
            "Your job is to:\n"
            "  1. Read the relevant files to understand the code.\n"
            "  2. Decide what change to make.\n"
            "  3. Write the change with read_file / write_file / edit_file / "
            "list_files / search_code.\n"
            "  4. Call finish(summary=...) when you believe the implementation "
            "is complete. Your summary will be used in the commit message.\n"
            "\n"
            "Tools (these are ALL the tools you have):\n"
            "  - read_file(path, offset?, length?)\n"
            "  - write_file(path, content)\n"
            "  - edit_file(path, old_string, new_string)\n"
            "  - list_files(path, pattern?)\n"
            "  - search_code(pattern, path?, file_pattern?)\n"
            "  - finish(summary)\n"
            "\n"
            f"Project root: {self.project_root.as_posix()}\n"
            f"Story ID: {self.idea_id}\n"
            "\n"
            "Path conventions:\n"
            "  - All tool paths accept relative OR absolute. Prefer relative paths "
            "anchored at the project root (e.g. `agent/foo.py`).\n"
            "  - Test files live under `tests/unit/` and must start with `test_` "
            "(e.g. `tests/unit/test_foo.py`). pytest will not collect any other filename.\n"
            "\n"
            "Rules:\n"
            "  - Always read a file before editing it.\n"
            "  - Make the smallest change that satisfies the story. Do not refactor "
            "unrelated code, rename imports, or reformat files outside the story's "
            "stated WHAT/WHERE.\n"
            "  - Writing tests is your call — if the change is non-trivial, add or "
            "update tests under `tests/unit/` so the harness can verify your work. "
            "Skip tests only for purely cosmetic changes or when the story "
            "explicitly says no tests.\n"
            "  - Don't gut existing tests just to make them pass — fix the code, "
            "not the assertion (unless the test was actually wrong).\n"
            "  - Call finish(summary=...) when you're done. The harness takes it "
            "from there.\n"
            "\n"
            "Workflow per round:\n"
            "  1. Locate ONCE: list_files / search_code to find the source and "
            "test file. Do not repeat the same search — it will not return new results.\n"
            "  2. Read in full: the failing test file, then the source file.\n"
            "  3. Plan briefly (one sentence) what you will change.\n"
            "  4. Edit with edit_file or write_file. If you do not edit anything, "
            "you have not done the work.\n"
            "  5. finish(summary=...) ONLY after at least one edit_file/write_file "
            "succeeded this round. Calling finish() with zero edits is wrong.\n"
            "\n"
            "Project context:\n"
            "  For project conventions, the AIW workflow, allowed commands, "
            "the codebase index, or the AIV scoring rubric, call the "
            "corresponding MCP tool: aiw_purpose, conventions, "
            "allowed_commands, codebase_index(area=...), scoring_rubric, "
            "overview. These are authoritative — prefer them over guessing. "
            "If an MCP tool is not available in this session, fall back to "
            "reading the source directly.\n"
        )

    def _build_initial_prompt(self) -> str:
        # Find relevant test file for test-first context
        test_context = self._find_test_context()
        prompt = self.prompt
        if test_context:
            prompt += f"\n\n## Relevant Test File (read to understand what 'passing' looks like)\n{test_context}"
        return prompt

    def _build_fix_prompt(
        self,
        round_num: int,
        test_output: str,
        changed_files: list[str],
        failing_tests: list[str],
    ) -> str:
        hint = _classify_error_hint(test_output)
        hint_section = f"\n\n## Hint\n{hint}" if hint else ""
        committed_summary = self._git_diff_stat()
        uncommitted_summary = self._git_diff_stat_uncommitted()

        # Progressive context: full file content from round 1+
        file_context = ""
        if changed_files:
            file_context = "\n\n## Changed Files (read these and fix the failures)\n"
            for f in changed_files[:3]:
                file_context += f"\n### {f}\n"
                content = self._tool_read_file(f)
                file_context += content[:3000] + "\n"

        # The harness only commits at the end of a passing round. So if the
        # previous round's tests failed, the model's edits are still sitting
        # in the working tree as unstaged diff — show that explicitly so the
        # model doesn't think its work was lost.
        change_summary = (
            f"### Committed (git diff --stat main...HEAD)\n{committed_summary}\n"
            f"\n### Uncommitted (git diff --stat HEAD — your edits from the previous round)\n"
            f"{uncommitted_summary}"
        )

        # If the previous round called finish() without making any edits
        # (and tests are still failing — which they are, since we're in
        # _build_fix_prompt), inject a strong nudge at the top of the
        # prompt. Empirically observed during TK-1045: the model read
        # the test file, decided "this looks fine", and called finish()
        # — burning 2 rounds before drift detection caught it. The
        # nudge tells the model explicitly that "I read it" is not a
        # valid response when tests are red, and lists the only tools
        # that count as taking action.
        no_edit_nudge = ""
        if self.prev_round_edit_count == 0:
            no_edit_nudge = (
                "## ⚠️ You did not edit any files last round\n"
                "You called `finish()` but invoked zero `edit_file` or "
                "`write_file` calls — and the tests below are still failing. "
                "Reading files alone does not change anything. This round you "
                "MUST use `edit_file` or `write_file` to modify the source "
                "before calling `finish()` again. If after careful reading "
                "you genuinely believe no code change is required (e.g., the "
                "failing tests are bugs in unrelated code), explain that "
                "concretely in your `finish()` summary instead of restating "
                "what the test does.\n\n"
            )

        return (
            f"{no_edit_nudge}"
            f"## Original Task (fix attempt round {round_num}/{self.max_rounds})\n"
            f"{self.prompt}\n\n"
            f"## What Was Changed\n{change_summary}\n\n"
            f"## Test Failures (fix these)\n"
            f"Failing tests: {', '.join(failing_tests) if failing_tests else '(see output below)'}\n\n"
            f"## Test Output (last 4000 chars)\n{test_output[-4000:]}"
            f"{hint_section}"
            f"{file_context}\n\n"
            f"## Your Job\n"
            f"Fix the failing tests without breaking passing tests. "
            f"Read the failing test file and the source file you changed. "
            f"Edit the source to make the failing tests pass, then call "
            f"`finish(summary=...)`. The harness will run pytest and commit "
            f"your changes — do not try to run git or pytest yourself."
        )

    def _find_test_context(self) -> str:
        """Return first 80 lines of a relevant test file, if one exists."""
        test_dir = self.project_root / "tests" / "unit"
        if not test_dir.is_dir():
            return ""
        # Look for test files that mention the idea_id or common keywords from prompt
        words = set(re.findall(r'\b\w{4,}\b', self.prompt.lower()))
        best: tuple[int, Path] | None = None
        for tf in test_dir.glob("test_*.py"):
            try:
                name_words = set(tf.stem.replace("test_", "").split("_"))
                overlap = len(name_words & words)
                if overlap > 0:
                    if best is None or overlap > best[0]:
                        best = (overlap, tf)
            except Exception:
                pass
        if best:
            try:
                lines = best[1].read_text(encoding="utf-8", errors="replace").splitlines()[:80]
                return "\n".join(lines)
            except Exception:
                pass
        return ""

    # ------------------------------------------------------------------
    # Test running and git helpers
    # ------------------------------------------------------------------

    def _run_pytest(self) -> dict[str, Any]:
        """Run pytest and return {passed, failing, output}."""
        # Tests live in local-agent/ — run from there so pytest.ini is found
        test_cwd = self.project_root / "local-agent"
        if not test_cwd.is_dir():
            test_cwd = self.project_root

        changed = self._get_changed_files()
        test_files = _find_related_tests_for_files(changed, self.project_root)

        if test_files:
            cmd = [sys.executable, "-m", "pytest"] + test_files + ["-x", "--tb=short", "-q"]
        else:
            # When no test files are found, run a basic pytest to check if it's working
            cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]

        # Write output to a temp file to avoid Windows pipe-deadlock when pytest
        # spawns child processes and capture_output=True fills the pipe buffer.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            out_path = tf.name

        try:
            with open(out_path, "w") as out_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=out_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(test_cwd),
                )
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return {"passed": False, "failing": [], "output": "ERROR: pytest timed out"}
            with open(out_path, encoding="utf-8", errors="replace") as f:
                output = f.read()
            failing = _parse_failing_tests(output)
            passed = proc.returncode == 0
            return {"passed": passed, "failing": failing, "output": output}
        except Exception as exc:
            return {"passed": False, "failing": [], "output": f"ERROR: {exc}"}
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    def _get_changed_files(self) -> list[str]:
        """Return absolute paths of files changed on current branch vs main."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "main...HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return [
                str(self.project_root / f)
                for f in result.stdout.strip().splitlines() if f
            ]
        except Exception:
            return []

    def _git_branch_files_or_empty(self) -> list[str]:
        """Return relative paths of files committed on the branch so far.

        Used as the "what existed at round start" snapshot so the commit
        message can label each file as edit/create. Returns relative paths
        (so the caller can join with project_root). Returns [] on any error
        (no git, fresh branch, etc.) — the commit message will just label
        every file as ``create`` in that case, which is acceptable.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "main...HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return [f for f in result.stdout.strip().splitlines() if f]
        except Exception:
            return []

    def _files_to_stage(self) -> list[str]:
        """Return absolute paths of files to stage for this round's commit.

        Combines:
          - Tracked-file modifications and deletions: ``git diff --name-only HEAD``
          - Untracked files: ``git ls-files --others --exclude-standard``

        We deliberately DON'T use ``git add -A`` — operators have been bitten
        by that staging scratch files outside scope. The combined output here
        is the legitimate "everything the model touched in this round" set.
        """
        files: list[str] = []
        try:
            modified = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            files.extend(f for f in modified.stdout.strip().splitlines() if f)
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            files.extend(f for f in untracked.stdout.strip().splitlines() if f)
        except Exception as exc:
            self._log(f"[OllamaCoder] Warning: _files_to_stage git query failed: {exc}")
            return []
        # De-dupe while preserving order, then absolutize.
        seen: set[str] = set()
        unique: list[str] = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return [str(self.project_root / f) for f in unique]

    def _git_diff_stat(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "main...HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return result.stdout.strip() or "(no changes yet)"
        except Exception:
            return "(could not get diff)"

    def _git_diff_stat_uncommitted(self) -> str:
        """Return ``git diff --stat HEAD`` — i.e. unstaged + uncommitted edits.

        Used in fix prompts to show the model what edits from the previous
        round are still sitting in the working tree (the harness only commits
        on a passing round, so a failing round's edits stay uncommitted).
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return result.stdout.strip() or "(no uncommitted edits)"
        except Exception:
            return "(could not get diff)"

    def _commit_changes(
        self,
        round_num: int,
        files: list[str] | None = None,
        pre_round_existing: set[str] | None = None,
    ) -> None:
        """Commit the given files with a deterministic message.

        ``files`` is the explicit set of files to stage — captured at the start
        of the round by the caller so that the staged set is deterministic
        (recomputing inside this function via ``_get_changed_files`` was
        racy, since the model could touch a file mid-round that we'd then
        also stage). When ``files`` is None, fall back to the legacy
        whole-branch behavior so existing call sites and tests still work.

        ``pre_round_existing`` is the set of files (absolute paths) that
        existed on disk at the start of the round. Used to label each file
        as ``edit`` (existed) or ``create`` (didn't) in the commit message.

        Commit message format:
            ``[<idea_id>] <title> <model> r<N>: <verb> <first> (+M more)``

        Title and model are pulled from the instance attributes set at
        construction time. The (+M more) suffix is omitted when only one
        file changed.
        """
        try:
            target_files = files if files is not None else self._get_changed_files()
            if not target_files:
                return

            existing_set = pre_round_existing or set()

            # Stage exactly the files the caller passed. Never use ``git add -A``
            # or ``.`` — operators have been bitten by that before when the model
            # touched scratch files outside its scope.
            subprocess.run(
                ["git", "add"] + list(target_files),
                capture_output=True, cwd=str(self.project_root),
            )

            commit_msg = self._build_commit_message(
                round_num, list(target_files), existing_set
            )
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, cwd=str(self.project_root),
            )
            self._log(f"[OllamaCoder] Round {round_num}: committed changes — {commit_msg}")

        except Exception as exc:
            self._log(f"[OllamaCoder] Warning: Failed to commit changes: {exc}")

    def _build_commit_message(
        self,
        round_num: int,
        files: list[str],
        pre_round_existing: set[str],
    ) -> str:
        """Build the deterministic commit message for ``_commit_changes``.

        Extracted so unit tests can pin the format without going through git.
        """
        # File path for the auto-summary: prefer the relative path from
        # project_root so the message stays short and stable across machines.
        first = files[0]
        try:
            first_rel = str(Path(first).relative_to(self.project_root))
        except (ValueError, TypeError):
            first_rel = Path(first).name

        verb = "create" if first not in pre_round_existing else "edit"
        extra = len(files) - 1
        suffix = f" (+{extra} more)" if extra > 0 else ""
        auto_summary = f"{verb} {first_rel}{suffix}"

        # Title and model are optional decorations — keep the message readable
        # if they're empty (older callers / tests).
        parts = [f"[{self.idea_id}]"]
        if self.story_title:
            parts.append(self.story_title)
        if self.model:
            parts.append(self.model)
        header = " ".join(parts)
        return f"{header} r{round_num}: {auto_summary}"

    def _tag_round_commits(self, round_num: int) -> None:
        """Amend the latest commit message to include round tag if on a story branch."""
        if round_num == 0:
            return  # Round 0 commits are tagged by the coder's own git commit calls
        try:
            # Check if there are commits on branch
            result = subprocess.run(
                ["git", "rev-list", "--count", "main..HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            count = int((result.stdout or "0").strip() or "0")
            if count == 0:
                return
            # Check if latest commit message already has round tag
            msg_result = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            msg = msg_result.stdout.strip()
            if f"[r{round_num}]" not in msg and "[r" not in msg:
                # Amend silently to add round tag
                subprocess.run(
                    ["git", "commit", "--amend", "--no-edit", "-m",
                     f"{msg} [r{round_num}]"],
                    capture_output=True, cwd=str(self.project_root),
                )
        except Exception as exc:
            # Check if this is a Git-related exception that should stop processing
            from agent.accountability import GitNotInstalledError, GitDirtyError
            if isinstance(exc, (GitNotInstalledError, GitDirtyError)):
                logger.error(f"[OllamaCoder] Aborted due to git exception: {exc}")
                # Re-raise to stop the execution loop
                raise
            logger.warning("git status failed or dirty repo detected: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _classify_error_hint(test_output: str) -> str:
    for error_type, hint in _ERROR_HINTS:
        if error_type in test_output:
            return hint
    return ""


def _parse_failing_tests(output: str) -> list[str]:
    """Extract failing test IDs from pytest output."""
    failing: list[str] = []
    # Match lines like: FAILED tests/unit/test_foo.py::TestBar::test_baz
    for match in re.finditer(r"FAILED\s+(tests/\S+)", output):
        failing.append(match.group(1))
    return failing


def _find_related_tests_for_files(changed_files: list[str], project_root: Path) -> list[str]:
    """Find test files relevant to the current branch's changes.

    Includes:
    1. Conventionally named tests for changed source files
       (``agent/foo.py`` -> ``tests/unit/test_foo.py``).
    2. Test files the branch itself changed or added — but only when their
       filename matches pytest's default discovery patterns (``test_*.py``
       or ``*_test.py``). Test files written under non-conforming names
       are skipped here intentionally so the executor's pre-suite check
       (see ``_detect_uncollected_test_files`` in executor.py) catches and
       fails them instead of silently no-op'ing.
    """
    # Tests live in local-agent/tests/unit/
    test_dir = project_root / "local-agent" / "tests" / "unit"
    if not test_dir.is_dir():
        test_dir = project_root / "tests" / "unit"
    if not test_dir.is_dir():
        return []
    related: set[str] = set()
    for src in changed_files:
        src_path = Path(src)
        # 1. Conventional test file for the changed source.
        stem = src_path.stem
        test_file = test_dir / f"test_{stem}.py"
        if test_file.exists():
            related.add(str(test_file))
        # 2. The changed file is itself a test under tests/.
        try:
            parts = src_path.parts
        except Exception:
            continue
        if "tests" not in parts:
            continue
        name = src_path.name
        if name in {"conftest.py", "__init__.py"}:
            continue
        if not (name.startswith("test_") or name.endswith("_test.py")):
            # Pytest will skip this file — don't include it.
            continue
        if src_path.is_absolute() and src_path.exists():
            related.add(str(src_path))
        else:
            candidate = project_root / src_path
            if candidate.exists():
                related.add(str(candidate))
    return sorted(related)


def _parse_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Fallback: parse tool calls from model content when tool_calls field is empty."""
    tool_calls: list[dict[str, Any]] = []
    # Match {"name": "...", "arguments": {...}} or {"name": "...", "parameters": {...}}
    for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', content, re.DOTALL):
        try:
            data = json.loads(match.group())
            name = data.get("name", "")
            args = data.get("arguments") or data.get("parameters") or {}
            if name and name in {t["function"]["name"] for t in _TOOLS}:
                tool_calls.append({"function": {"name": name, "arguments": args}})
        except (json.JSONDecodeError, KeyError):
            pass
    return tool_calls


def _safe_json(raw: Any) -> dict[str, Any]:
    """Safely parse tool call arguments."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool args for logging — truncate long values."""
    parts: list[str] = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:60] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
