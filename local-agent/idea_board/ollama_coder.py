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
from agent.accountability import verify_git_clean, create_branch, GitDirtyError

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

READ_FILE_MAX_CHARS = 20_000
LIST_FILES_MAX = 200


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
            "name": "run_bash",
            "description": (
                "Run a shell command. Allowed prefixes: git, pytest, python, python3. "
                "Blocked: safe_update, push origin, push --force, merge, checkout main, rm -rf."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"],
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
            "name": "verify_git_clean",
            "description": "Verify that the git working directory is clean (no uncommitted changes).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_branch",
            "description": "Create a new git branch with the given name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch_name": {
                        "type": "string",
                        "description": "Name of the branch to create"
                    }
                },
                "required": ["branch_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that implementation is complete. Call only after committing all changes.",
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
        self._log = state.log if hasattr(state, "log") else lambda m: None

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _is_cancelled(self) -> bool:
        return bool(getattr(self.state, "cancelled", False))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full outer fix-round loop. Acquires GPU priority for duration."""
        acquire_coder_priority()
        try:
            self._log(f"[OllamaCoder] Starting with model={self.model}, "
                      f"host={self.host}, "
                      f"max_rounds={self.max_rounds}, max_turns={self.max_turns}")
            self._run_rounds()
        finally:
            release_coder_priority()

    # ------------------------------------------------------------------
    # Outer round loop
    # ------------------------------------------------------------------

    def _run_rounds(self) -> None:
        test_output = ""
        failing_tests: list[str] = []
        changed_files: list[str] = []

        for round_num in range(self.max_rounds):
            if self._is_cancelled():
                self._log("[OllamaCoder] Cancelled — stopping")
                return
            self._log(f"[OllamaCoder] --- Round {round_num} ---")

            # Build prompt for this round
            if round_num == 0:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_initial_prompt()
            else:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_fix_prompt(round_num, test_output, changed_files, failing_tests)

            # Run inner tool-calling loop
            messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
            finished = self._run_inner_loop(system_prompt, messages, round_num)

            if not finished:
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
                # Create a commit if tests pass
                self._create_commit_if_needed(round_num)
                return

            self._log(
                f"[OllamaCoder] Round {round_num}: {len(failing_tests)} test(s) failing"
            )
            if round_num < self.max_rounds - 1:
                hint = _classify_error_hint(test_output)
                if hint:
                    self._log(f"[OllamaCoder] Hint: {hint}")

        self._log(f"[OllamaCoder] Exhausted {self.max_rounds} rounds — story will be marked failed")

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

        for turn in range(self.max_turns):
            if self._is_cancelled():
                self._log("[OllamaCoder] Cancelled — stopping inner loop")
                return False
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
                result = self._execute_tool(name, args)
                self._log(f"[OllamaCoder] ← {str(result)[:300]}")

                # Append tool result
                messages.append({
                    "role": "tool",
                    "content": str(result),
                })

                if name == "finish":
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

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            # Enforce readiness gate for code generation tools
            if name in ("read_file", "write_file", "run_bash"):
                self._check_readiness_gate()
            
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
                return self._tool_write_file(args.get("path", ""), args.get("content", ""))
            elif name == "edit_file":
                return self._tool_edit_file(
                    args.get("path", ""),
                    args.get("old_string", ""),
                    args.get("new_string", ""),
                )
            elif name == "run_bash":
                return self._tool_run_bash(args.get("command", ""))
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
            else:
                return f"ERROR: unknown tool '{name}'"
        except EnvironmentReadyError as e:
            # Re-raise readiness errors
            raise e
        except Exception as exc:
            return f"ERROR: {exc}"

    def _check_readiness_gate(self) -> None:
        """Check that the environment is ready for code generation.

        This method ensures that:
        1. The git repository is clean (no uncommitted changes)
        2. A new branch can be created for this work

        Raises:
            EnvironmentReadyError: If either check fails.
        """
        # Check 1: Verify git is clean
        git_clean_result = verify_git_clean()
        if "VERIFIED" not in git_clean_result:
            # If git is not clean, raise EnvironmentReadyError
            raise EnvironmentReadyError("Git repository is not clean")
        
        # Check 2: Try to create a branch for this work
        # We'll use a branch name based on the idea ID
        branch_name = f"TK-{self.idea_id}"
        branch_result = create_branch(branch_name)
        if "ERROR" in branch_result:
            # If branch creation fails, raise EnvironmentReadyError
            raise EnvironmentReadyError("Failed to create branch")

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
        # Allowlist check
        allowed = any(cmd_lower.startswith(prefix) for prefix in BASH_ALLOWLIST)
        if not allowed:
            return f"BLOCKED: command must start with one of {BASH_ALLOWLIST}"
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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "tools": _TOOLS,
            "stream": False,
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
        for attempt in range(4):  # 1 initial + 3 retries
            try:
                r = requests.post(
                    f"{self.host}/api/chat",
                    json=body,
                    timeout=900,
                )
            except requests.RequestException as exc:
                logger.warning("[OllamaCoder] Network error: %s", exc)
                return None
            if r.status_code == 200:
                break
            # HTTP 500 with Ollama's XML parse bug is transient — retry with backoff
            if r.status_code == 500 and attempt < 3:
                delay = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "[OllamaCoder] HTTP 500 (attempt %d/4), retrying in %ds: %s",
                    attempt + 1, delay, r.text[:120],
                )
                time.sleep(delay)
                continue
            logger.warning("[OllamaCoder] HTTP %d: %s", r.status_code, r.text[:200])
            return None
        try:
            return r.json()
        except ValueError:
            logger.warning("[OllamaCoder] Non-JSON response")
            return None

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
        return (
            "/no_think\n"
            "You are an expert Python software engineer implementing Jira stories.\n"
            "You have tools to read, write, and edit files, run git/pytest/python commands, "
            "list files, and search code.\n\n"
            f"Project root: {self.project_root.as_posix()}\n"
            "All tool paths (read_file, write_file, edit_file, list_files, search_code) "
            "accept either relative or absolute paths. Prefer RELATIVE paths "
            "anchored at the project root (e.g. `local-agent/agent/foo.py`) — "
            "they are shorter, less error-prone, and unaffected by the project "
            "root's exact location on disk.\n\n"
            "Rules:\n"
            "- Always read relevant files before editing them\n"
            "- Prefer relative paths (e.g. `local-agent/tests/unit/test_foo.py`) "
            "over absolute paths when referencing files\n"
            "- Run `git add <file1> <file2> ... && git commit -m '[<idea_id>] <description>'` after each meaningful change. Only add files you explicitly modified — never use `git add -A` or `git add .`\n"
            "- Run tests with pytest to verify your implementation\n"
            "- Call finish() only after committing all changes\n"
            "- Do not modify test files unless the story explicitly asks you to\n"
            "- Test files MUST live under local-agent/tests/unit/ and start "
            "with `test_` (e.g. tests/unit/test_jira_retry.py). pytest will "
            "not collect any other filename.\n"
            "- run_bash already runs in the project root. Do NOT prefix commands "
            "with `cd <path> && ...`. Just call `pytest tests/unit/test_foo.py`, "
            "`git status`, etc. directly.\n"
            "- Make minimal, focused changes that solve the task\n"
            f"- Story ID: {self.idea_id}\n"
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
        change_summary = self._git_diff_stat()

        # Progressive context: full file content from round 1+
        file_context = ""
        if changed_files:
            file_context = "\n\n## Changed Files (read these and fix the failures)\n"
            for f in changed_files[:3]:
                file_context += f"\n### {f}\n"
                content = self._tool_read_file(f)
                file_context += content[:3000] + "\n"

        return (
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
            f"Edit the source to make tests pass. Commit when done, then call finish()."
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

    def _git_diff_stat(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "main...HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            return result.stdout.strip() or "(no changes yet)"
        except Exception:
            return "(could not get diff)"

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

    def _create_commit_if_needed(self, round_num: int) -> None:
        """Create a commit if there are changes and tests pass."""
        try:
            # Check if there are any changes
            result = subprocess.run(
                ["git", "diff", "--name-only", "main...HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root),
            )
            changed_files = result.stdout.strip().splitlines()
            
            # Only create commit if there are actual changes
            if changed_files and any(changed_files):
                # There are changes, create a commit
                commit_msg = f"TK-{self.idea_id}: implemented story (round {round_num})"
                subprocess.run(
                    ["git", "add", "."],
                    capture_output=True, cwd=str(self.project_root),
                )
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    capture_output=True, cwd=str(self.project_root),
                )
                self._log(f"[OllamaCoder] Created commit for round {round_num}")
        except Exception as exc:
            logger.warning("Failed to create commit: %s", exc)


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


def _check_environment_ready() -> None:
    """Check that the environment is ready for code generation.

    Raises:
        EnvironmentReadyError: If git repository is not clean.
    """
    # Check that git repository is clean
    try:
        result = verify_git_clean()
        if not result.startswith("VERIFIED:"):
            raise EnvironmentReadyError("Git repository is not clean")
    except GitDirtyError:
        raise EnvironmentReadyError("Git repository is not clean")
    except Exception:
        # If we can't verify git status, we can't proceed safely
        raise EnvironmentReadyError("Failed to verify git status")


def create_branch(short_name: str) -> str:
    """Create and checkout new branch with timestamp.
    
    This is a helper function that mimics the behavior of safe_update.create_branch
    but is available for use in the OllamaCoder context.
    
    Args:
        short_name: Short name for the branch
        
    Returns:
        The created branch name
        
    Raises:
        EnvironmentReadyError: If branch creation fails
    """
    try:
        from safe_update import create_branch as safe_create_branch
        return safe_create_branch(short_name)
    except Exception as e:
        raise EnvironmentReadyError(f"Failed to create branch: {str(e)}")


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
