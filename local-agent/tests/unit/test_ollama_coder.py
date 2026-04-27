"""Tests for idea_board/ollama_coder.py — OllamaCoder local Ollama agent."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import (
    OllamaCoder,
    _classify_error_hint,
    _find_related_tests_for_files,
    _fmt_args,
    _parse_failing_tests,
    _parse_tool_calls_from_content,
    _safe_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coder(tmp_path: Path, prompt: str = "Implement feature X") -> OllamaCoder:
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    return OllamaCoder(
        prompt=prompt,
        project_root=tmp_path,
        idea_id="TK-999",
        state=state,
        model="qwen3.5:27b",
        max_turns=5,
        max_rounds=3,
        num_ctx=4096,
    )


def _tool_call(name: str, **kwargs: object) -> dict:
    return {"function": {"name": name, "arguments": kwargs}}


def _response(tool_calls: list | None = None, content: str = "") -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }
    }


# ---------------------------------------------------------------------------
# TestToolExecution
# ---------------------------------------------------------------------------

class TestToolExecution:
    def test_read_file_returns_content(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hello')", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("hello.py")
        assert "print('hello')" in result

    def test_read_file_truncates_large_files(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_text("x" * 25_000, encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("big.py")
        assert "truncated" in result
        assert len(result) < 22_000

    def test_read_file_missing(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("nonexistent.py")
        assert "ERROR" in result

    def test_read_file_with_offset_and_length(self, tmp_path: Path) -> None:
        """When the model passes offset+length, paginate by lines."""
        f = tmp_path / "many.py"
        f.write_text(
            "\n".join(f"line_{i}" for i in range(1, 21)) + "\n",
            encoding="utf-8",
        )
        coder = _make_coder(tmp_path)
        # Lines 5..7 (1-indexed offset=5, length=3)
        result = coder._tool_read_file("many.py", offset=5, length=3)
        assert "line_5" in result
        assert "line_6" in result
        assert "line_7" in result
        assert "line_4" not in result
        assert "line_8" not in result
        # Header should reflect the slice
        assert "lines 5-7" in result

    def test_read_file_offset_as_string(self, tmp_path: Path) -> None:
        """The model habitually sends offset as a JSON string ('5');
        the tool must coerce instead of dropping the arg silently."""
        f = tmp_path / "many.py"
        f.write_text(
            "\n".join(f"line_{i}" for i in range(1, 11)) + "\n",
            encoding="utf-8",
        )
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("many.py", offset="3", length="2")
        assert "line_3" in result
        assert "line_4" in result
        assert "line_2" not in result
        assert "line_5" not in result

    def test_read_file_offset_past_eof(self, tmp_path: Path) -> None:
        """Offset past EOF gets a clear "past end" message instead of
        silently returning the start of the file."""
        f = tmp_path / "short.py"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("short.py", offset=100)
        assert "past end" in result
        assert "3 lines" in result

    def test_read_file_no_offset_unchanged_behavior(self, tmp_path: Path) -> None:
        """Backward compatibility: no offset/length → whole-file read."""
        f = tmp_path / "small.py"
        f.write_text("hello\nworld\n", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_read_file("small.py")
        assert "hello" in result
        assert "world" in result
        # No paginated header when no slicing.
        assert "lines 1-" not in result

    def test_write_file_creates_file(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("new.py", "x = 1")
        assert "Wrote" in result
        assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1"

    def test_write_file_rejects_test_file_without_test_prefix(self, tmp_path: Path) -> None:
        """Files under tests/ that don't start with test_ won't be collected by pytest."""
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("tests/jira_retry_backoff.py", "def test_foo(): pass")
        assert "ERROR" in result
        assert "test_jira_retry_backoff.py" in result
        assert not (tmp_path / "tests" / "jira_retry_backoff.py").exists()

    def test_write_file_rejects_test_shaped_content_outside_tests(self, tmp_path: Path) -> None:
        """A file containing def test_* anywhere — even outside tests/ — should be flagged."""
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("foo.py", "def test_something():\n    assert True")
        assert "ERROR" in result
        assert "test_foo.py" in result

    def test_write_file_allows_test_prefixed_file(self, tmp_path: Path) -> None:
        """test_*.py files under tests/ should be allowed."""
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("tests/unit/test_thing.py", "def test_x(): pass")
        assert "Wrote" in result
        assert (tmp_path / "tests" / "unit" / "test_thing.py").exists()

    def test_write_file_allows_underscore_test_suffix(self, tmp_path: Path) -> None:
        """foo_test.py is a valid pytest filename too."""
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("tests/foo_test.py", "def test_x(): pass")
        assert "Wrote" in result

    def test_write_file_allows_conftest_under_tests(self, tmp_path: Path) -> None:
        """conftest.py under tests/ is fine — pytest treats it specially."""
        coder = _make_coder(tmp_path)
        result = coder._tool_write_file("tests/conftest.py", "import pytest")
        assert "Wrote" in result

    def test_edit_file_replaces_string(self, tmp_path: Path) -> None:
        f = tmp_path / "src.py"
        f.write_text("def old(): pass", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_edit_file("src.py", "def old():", "def new():")
        assert result == "Edited src.py"
        assert "def new():" in f.read_text(encoding="utf-8")

    def test_edit_file_missing_old_string(self, tmp_path: Path) -> None:
        f = tmp_path / "src.py"
        f.write_text("def foo(): pass", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_edit_file("src.py", "DOES NOT EXIST", "new")
        assert "ERROR" in result

    def test_edit_file_empty_old_string(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_edit_file("any.py", "", "new")
        assert "ERROR" in result

    def test_run_bash_allowed(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("python --version")
        assert result  # some output

    def test_run_bash_blocklist(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("git push origin main")
        assert "BLOCKED" in result

    def test_run_bash_not_in_allowlist(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("curl https://evil.com")
        assert "BLOCKED" in result

    def test_run_bash_rm_rf_blocked(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("rm -rf /")
        assert "BLOCKED" in result

    def test_run_bash_strips_leading_cd(self, tmp_path: Path) -> None:
        """Model habitually prefixes commands with `cd <root> && X` — strip it."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash(f"cd {tmp_path} && python --version")
        # Should NOT be blocked — the `cd ... && ` should be stripped
        assert "BLOCKED" not in result
        # Should have run python --version
        assert "Python" in result or "python" in result.lower()

    def test_run_bash_strips_cd_preserves_blocklist(self, tmp_path: Path) -> None:
        """cd-stripping must not bypass the blocklist (e.g. git push)."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash(f"cd {tmp_path} && git push origin main")
        assert "BLOCKED" in result

    def test_list_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_list_files(".")
        assert "a.py" in result
        assert "b.py" in result

    def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._execute_tool("nonexistent", {})
        assert "ERROR" in result or "unknown" in result.lower()

    def test_search_code_finds_pattern(self, tmp_path: Path) -> None:
        """search_code must actually find existing patterns.

        Regression: previous implementation invoked ``python -m grep`` (non-
        existent module) which returned rc=1 with empty stdout, causing every
        search to return "(no matches)" regardless of file contents. This
        cascaded into OllamaCoder concluding that work was already complete
        and producing 0 commits.
        """
        (tmp_path / "src.py").write_text(
            "class JiraRetryExhausted(Exception):\n    pass\n", encoding="utf-8"
        )
        coder = _make_coder(tmp_path)
        result = coder._tool_search_code("JiraRetryExhausted", path=".")
        assert "JiraRetryExhausted" in result
        assert "(no matches)" not in result

    def test_search_code_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "src.py").write_text("x = 1\n", encoding="utf-8")
        coder = _make_coder(tmp_path)
        result = coder._tool_search_code("DefinitelyNotInThisFile", path=".")
        assert result == "(no matches)"

    def test_search_code_missing_path(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_search_code("anything", path="does/not/exist")
        assert "ERROR" in result
        assert "does not exist" in result

    def test_search_code_recurses_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        (sub / "buried.py").write_text(
            "def needle(): pass\n", encoding="utf-8"
        )
        coder = _make_coder(tmp_path)
        result = coder._tool_search_code("needle", path=".")
        assert "needle" in result
        assert "buried.py" in result

    def test_search_code_in_specific_file(self, tmp_path: Path) -> None:
        """Regression: when ``path`` points at a single file (not a dir), the
        manual fallback used ``rglob`` which returns nothing on a file —
        every search returned ``(no matches)`` even when the pattern
        clearly existed.  The fix special-cases ``is_file()``."""
        f = tmp_path / "target.py"
        f.write_text(
            "import subprocess\n"
            "subprocess.run(['ls'])\n"
            "x = 1\n",
            encoding="utf-8",
        )
        coder = _make_coder(tmp_path)
        result = coder._tool_search_code("subprocess", path="target.py")
        assert "subprocess" in result
        assert "target.py" in result
        # Both lines must be reported.
        assert "import subprocess" in result
        assert "subprocess.run" in result


# ---------------------------------------------------------------------------
# TestOllamaCoderRun
# ---------------------------------------------------------------------------

class TestOllamaCoderRun:
    """Test the outer run() loop with mocked _chat_with_tools."""

    def _mock_responses(self, coder: OllamaCoder, response_sequence: list) -> None:
        """Replace _chat_with_tools with a sequence of responses."""
        iter_resp = iter(response_sequence)
        coder._chat_with_tools = lambda sys, msgs: next(iter_resp, None)  # type: ignore[method-assign]

    def _mock_pytest_pass(self, coder: OllamaCoder) -> None:
        coder._run_pytest = lambda: {"passed": True, "failing": [], "output": "1 passed"}  # type: ignore[method-assign]

    def _mock_pytest_fail(self, coder: OllamaCoder, failing: list | None = None) -> None:
        coder._run_pytest = lambda: {  # type: ignore[method-assign]
            "passed": False,
            "failing": failing or ["tests/unit/test_foo.py::test_bar"],
            "output": "FAILED tests/unit/test_foo.py::test_bar\nAssertionError: 0 != 1",
        }

    def test_single_round_success(self, tmp_path: Path) -> None:
        """Model finishes in one round with passing tests."""
        coder = _make_coder(tmp_path)
        finish_response = _response([_tool_call("finish", summary="done")])
        self._mock_responses(coder, [finish_response])
        self._mock_pytest_pass(coder)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()
        # No exception = success

    def test_two_round_fix(self, tmp_path: Path) -> None:
        """Round 0 fails tests, round 1 fixes them."""
        coder = _make_coder(tmp_path, prompt="Fix the bug")
        finish_response = _response([_tool_call("finish", summary="fixed")])

        call_count = [0]
        def mock_pytest():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"passed": False, "failing": ["test_foo::bar"], "output": "FAILED\nAssertionError"}
            return {"passed": True, "failing": [], "output": "1 passed"}

        coder._chat_with_tools = lambda sys, msgs: finish_response  # type: ignore[method-assign]
        coder._run_pytest = mock_pytest  # type: ignore[method-assign]
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()

        assert call_count[0] == 2

    def test_max_rounds_exhausted(self, tmp_path: Path) -> None:
        """All rounds fail tests — run() returns without exception."""
        coder = _make_coder(tmp_path)
        finish_response = _response([_tool_call("finish", summary="done")])
        coder._chat_with_tools = lambda sys, msgs: finish_response  # type: ignore[method-assign]
        self._mock_pytest_fail(coder)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()  # should NOT raise

    def test_ollama_none_response_aborts_inner_loop(self, tmp_path: Path) -> None:
        """If Ollama returns None, inner loop aborts gracefully."""
        coder = _make_coder(tmp_path)
        coder._chat_with_tools = lambda sys, msgs: None  # type: ignore[method-assign]
        self._mock_pytest_pass(coder)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()  # should not raise


# ---------------------------------------------------------------------------
# TestCancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_cancelled_before_round_stops_immediately(self, tmp_path: Path) -> None:
        """If state.cancelled is True before round 0, run() exits without calling Ollama."""
        coder = _make_coder(tmp_path)
        coder.state.cancelled = True
        call_count = [0]
        coder._chat_with_tools = lambda s, m: (call_count.__setitem__(0, call_count[0] + 1) or {})  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()

        assert call_count[0] == 0

    def test_cancelled_after_http_call_stops_inner_loop(self, tmp_path: Path) -> None:
        """If cancelled is set while Ollama call is in flight, stop after it returns."""
        coder = _make_coder(tmp_path)
        finish_response = _response([_tool_call("finish", summary="done")])

        call_count = [0]
        def mock_chat(sys, msgs):
            call_count[0] += 1
            coder.state.cancelled = True  # simulate worker setting cancel mid-call
            return finish_response

        coder._chat_with_tools = mock_chat  # type: ignore[method-assign]
        coder._run_pytest = lambda: {"passed": True, "failing": [], "output": ""}  # type: ignore[method-assign]
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()

        # Stopped after exactly 1 Ollama call (the cancellation check fires before finish executes)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TestFixPromptBuilding
# ---------------------------------------------------------------------------

class TestFixPromptBuilding:
    def test_fix_prompt_contains_original_task(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path, prompt="Add retry logic")
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._git_diff_stat = lambda: "1 file changed"  # type: ignore[method-assign]
        prompt = coder._build_fix_prompt(1, "AssertionError: 0 != 1", [], ["tests/test_foo.py::bar"])
        assert "Add retry logic" in prompt
        assert "test_foo" in prompt

    def test_fix_prompt_includes_test_output(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._git_diff_stat = lambda: "(stat)"  # type: ignore[method-assign]
        prompt = coder._build_fix_prompt(1, "FAILED tests/foo.py AssertionError", [], [])
        assert "FAILED" in prompt or "AssertionError" in prompt

    def test_error_classification_import_error(self) -> None:
        hint = _classify_error_hint("ImportError: cannot import name 'foo'")
        assert "import" in hint.lower()

    def test_error_classification_assertion_error(self) -> None:
        hint = _classify_error_hint("AssertionError: expected True but got False")
        assert "expected" in hint.lower() or "actual" in hint.lower() or "value" in hint.lower()

    def test_error_classification_attribute_error(self) -> None:
        hint = _classify_error_hint("AttributeError: 'NoneType' has no attribute 'foo'")
        assert "attribute" in hint.lower() or "method" in hint.lower()

    def test_error_classification_syntax_error(self) -> None:
        hint = _classify_error_hint("SyntaxError: invalid syntax at line 5")
        assert "syntax" in hint.lower()

    def test_error_classification_no_match(self) -> None:
        hint = _classify_error_hint("some random output with no known error")
        assert hint == ""

    def test_no_edit_nudge_when_prev_round_had_zero_edits(self, tmp_path: Path) -> None:
        """prev_round_edit_count == 0 → fix prompt leads with the no-edit nudge.

        TK-1045 burned 2 rounds because the model called finish() after only
        read_file calls. The fix prompt now leads with a strong nudge when
        the previous round made zero edits. Without this, the model has no
        signal that "I read the file" wasn't enough.
        """
        coder = _make_coder(tmp_path)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._git_diff_stat = lambda: ""  # type: ignore[method-assign]
        coder._git_diff_stat_uncommitted = lambda: ""  # type: ignore[method-assign]
        coder.prev_round_edit_count = 0

        prompt = coder._build_fix_prompt(1, "FAILED tests/foo.py", [], ["tests/foo.py::bar"])

        assert "did not edit any files last round" in prompt
        assert "edit_file" in prompt and "write_file" in prompt
        # Nudge must come BEFORE the original-task header so the model sees
        # it before deciding what to do.
        nudge_idx = prompt.find("did not edit any files last round")
        task_idx = prompt.find("Original Task")
        assert nudge_idx < task_idx, (
            "no-edit nudge must precede the task header so the model sees "
            "it before re-engaging with the task"
        )

    def test_no_nudge_when_prev_round_made_edits(self, tmp_path: Path) -> None:
        """prev_round_edit_count > 0 → no nudge (model is doing the right thing).

        We don't want to scold a model that's actively editing — the nudge
        is precisely scoped to the failure mode where finish() arrived
        without any tool-driven changes.
        """
        coder = _make_coder(tmp_path)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._git_diff_stat = lambda: ""  # type: ignore[method-assign]
        coder._git_diff_stat_uncommitted = lambda: ""  # type: ignore[method-assign]
        coder.prev_round_edit_count = 3

        prompt = coder._build_fix_prompt(1, "FAILED tests/foo.py", [], [])

        assert "did not edit any files last round" not in prompt

    def test_no_nudge_when_prev_round_count_is_none(self, tmp_path: Path) -> None:
        """First fix round (None sentinel) doesn't emit the nudge.

        prev_round_edit_count is None until the first inner loop completes.
        We still go through _build_fix_prompt on round 1, but should NOT
        scold the model for "doing nothing last round" when there *was*
        no last round in the relevant sense (round 0 ran the initial
        prompt, not the fix prompt).
        """
        coder = _make_coder(tmp_path)
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._git_diff_stat = lambda: ""  # type: ignore[method-assign]
        coder._git_diff_stat_uncommitted = lambda: ""  # type: ignore[method-assign]
        # Default state from __init__
        assert coder.prev_round_edit_count is None

        prompt = coder._build_fix_prompt(1, "FAILED tests/foo.py", [], [])

        assert "did not edit any files last round" not in prompt

    def test_edit_counter_increments_on_successful_edit(self, tmp_path: Path) -> None:
        """_execute_tool('edit_file', ...) increments _round_edit_count on success."""
        # Create a real file in the worktree so edit_file has something to edit.
        local_agent = tmp_path / "local-agent"
        local_agent.mkdir()
        target = local_agent / "foo.py"
        target.write_text("hello world\n")

        coder = _make_coder(tmp_path)
        coder._round_edit_count = 0

        result = coder._execute_tool(
            "edit_file",
            {"path": "local-agent/foo.py", "old_string": "hello", "new_string": "goodbye"},
        )

        assert not str(result).startswith("ERROR"), f"edit failed: {result}"
        assert coder._round_edit_count == 1
        assert target.read_text() == "goodbye world\n"

    def test_edit_counter_does_not_increment_on_error(self, tmp_path: Path) -> None:
        """edit_file that returns ERROR (e.g., file not found) does NOT bump
        _round_edit_count — otherwise an attempted-but-failed edit would
        suppress the next round's nudge despite no real progress.
        """
        coder = _make_coder(tmp_path)
        coder._round_edit_count = 0

        result = coder._execute_tool(
            "edit_file",
            {"path": "nonexistent/path.py", "old_string": "x", "new_string": "y"},
        )

        assert str(result).startswith("ERROR"), (
            f"expected ERROR for nonexistent path, got: {result}"
        )
        assert coder._round_edit_count == 0


# ---------------------------------------------------------------------------
# TestStripThink
# ---------------------------------------------------------------------------

class TestStripThink:
    def test_think_content_stripped_from_response(self, tmp_path: Path) -> None:
        """think content is stripped before processing tool calls."""
        coder = _make_coder(tmp_path)
        logs: list[str] = []
        coder._log = logs.append

        response_with_think = {
            "message": {
                "role": "assistant",
                "content": "<think>I should call finish</think>",
                "tool_calls": [_tool_call("finish", summary="done")],
            }
        }
        self._mock_pytest_pass(coder)
        coder._run_pytest = lambda: {"passed": True, "failing": [], "output": "1 passed"}  # type: ignore[method-assign]
        coder._get_changed_files = lambda: []  # type: ignore[method-assign]
        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]
        coder._chat_with_tools = lambda sys, msgs: response_with_think  # type: ignore[method-assign]

        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()
        # Should complete without error — think content stripped, tool_calls processed

    def _mock_pytest_pass(self, coder: OllamaCoder) -> None:
        coder._run_pytest = lambda: {"passed": True, "failing": [], "output": "1 passed"}  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# TestContextTrimming
# ---------------------------------------------------------------------------

class TestContextTrimming:
    def test_trims_when_over_threshold(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        # Build messages that exceed 75% of num_ctx (4096) * 4 chars = 12288 threshold
        messages: list[dict] = [
            {"role": "user", "content": "Original story"},
        ]
        # Add enough tool messages to exceed the threshold: 30 × 500 = 15000 > 12288
        for i in range(30):
            messages.append({"role": "tool", "content": "x" * 500})

        result = coder._trim_context(messages)
        assert len(result) < len(messages)

    def test_preserves_first_user_message(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        messages: list[dict] = [
            {"role": "user", "content": "Original story prompt"},
        ]
        for i in range(20):
            messages.append({"role": "tool", "content": "x" * 500})

        result = coder._trim_context(messages)
        assert result[0]["content"] == "Original story prompt"

    def test_no_trim_when_under_threshold(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        messages = [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "ok"},
        ]
        result = coder._trim_context(messages)
        assert result == messages


# ---------------------------------------------------------------------------
# TestGPUGate
# ---------------------------------------------------------------------------

class TestGPUGate:
    def test_acquire_sets_event(self) -> None:
        from agent.ollama_client import _coder_active, acquire_coder_priority, release_coder_priority
        release_coder_priority()  # ensure clean state
        assert not _coder_active.is_set()
        acquire_coder_priority()
        assert _coder_active.is_set()
        release_coder_priority()
        assert not _coder_active.is_set()

    def test_chat_waits_when_coder_active(self) -> None:
        """chat() waits up to 60s when coder is active, but proceeds after release."""
        from agent import ollama_client

        ollama_client.acquire_coder_priority()
        wait_called = threading.Event()
        proceed_event = threading.Event()

        # Release after short delay
        def release_after():
            import time
            time.sleep(0.1)
            ollama_client.release_coder_priority()
            proceed_event.set()

        t = threading.Thread(target=release_after)
        t.start()

        with patch.object(ollama_client, "requests") as mock_requests:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "hello"}
            mock_requests.post.return_value = mock_resp

            result = ollama_client.chat("test", "qwen3.5:27b", timeout=5)

        t.join()
        assert proceed_event.is_set()
        # chat() should have proceeded (result from mock) after coder released


# ---------------------------------------------------------------------------
# Module-level helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_failing_tests(self) -> None:
        output = "FAILED tests/unit/test_foo.py::TestBar::test_baz - AssertionError"
        failing = _parse_failing_tests(output)
        assert "tests/unit/test_foo.py::TestBar::test_baz" in failing

    def test_parse_failing_tests_multiple(self) -> None:
        output = (
            "FAILED tests/unit/test_a.py::test_one\n"
            "FAILED tests/unit/test_b.py::test_two\n"
        )
        failing = _parse_failing_tests(output)
        assert len(failing) == 2

    def test_parse_tool_calls_from_content(self) -> None:
        # Regex matches flat JSON objects (no nested braces)
        content = '{"name": "finish", "summary": "done"}'
        calls = _parse_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "finish"

    def test_parse_tool_calls_unknown_name_ignored(self) -> None:
        content = '{"name": "launch_nukes", "arguments": {}}'
        calls = _parse_tool_calls_from_content(content)
        assert calls == []

    def test_safe_json_dict_passthrough(self) -> None:
        assert _safe_json({"a": 1}) == {"a": 1}

    def test_safe_json_string(self) -> None:
        assert _safe_json('{"a": 1}') == {"a": 1}

    def test_safe_json_invalid(self) -> None:
        assert _safe_json("not json") == {}

    def test_fmt_args_truncates(self) -> None:
        result = _fmt_args({"content": "x" * 100})
        assert len(result) < 200
        assert "..." in result


# ---------------------------------------------------------------------------
# _find_related_tests_for_files
# ---------------------------------------------------------------------------


class TestFindRelatedTestsForFiles:
    """Verify the helper that decides which test files OllamaCoder runs each
    round. Must include changed test files directly so the Worker validates
    its OWN test additions, but only when the filename matches pytest's
    default discovery patterns. Files like ``tests/jira_retry_permanent.py``
    are intentionally NOT returned here — the executor has a separate
    pre-suite check that fails them loudly. See: TK-1184 incident.
    """

    def test_includes_conventional_test_for_changed_source(self, tmp_path: Path) -> None:
        unit_dir = tmp_path / "local-agent" / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        target = unit_dir / "test_foo.py"
        target.write_text("# test")

        result = _find_related_tests_for_files(
            [str(tmp_path / "local-agent" / "agent" / "foo.py")],
            tmp_path,
        )

        assert str(target) in result

    def test_includes_changed_test_file_directly(self, tmp_path: Path) -> None:
        """A branch that adds tests/unit/test_new.py should run that file."""
        unit_dir = tmp_path / "local-agent" / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        new_test = unit_dir / "test_new.py"
        new_test.write_text("# test")

        result = _find_related_tests_for_files([str(new_test)], tmp_path)

        assert str(new_test) in result

    def test_skips_uncollectable_test_filename(self, tmp_path: Path) -> None:
        """tests/jira_retry_permanent.py is NOT collected by pytest's
        default discovery, so don't include it. The executor's pre-suite
        check is responsible for failing this case."""
        tests_dir = tmp_path / "local-agent" / "tests"
        tests_dir.mkdir(parents=True)
        unit_dir = tests_dir / "unit"
        unit_dir.mkdir()
        bad = tests_dir / "jira_retry_permanent.py"
        bad.write_text("# test")

        result = _find_related_tests_for_files([str(bad)], tmp_path)

        assert str(bad) not in result

    def test_skips_conftest(self, tmp_path: Path) -> None:
        """conftest.py is fixtures, not tests."""
        unit_dir = tmp_path / "local-agent" / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        conftest = unit_dir / "conftest.py"
        conftest.write_text("# fixtures")

        result = _find_related_tests_for_files([str(conftest)], tmp_path)

        assert str(conftest) not in result

    def test_underscore_test_suffix_collected(self, tmp_path: Path) -> None:
        """foo_test.py also matches default discovery."""
        unit_dir = tmp_path / "local-agent" / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        target = unit_dir / "foo_test.py"
        target.write_text("# test")

        result = _find_related_tests_for_files([str(target)], tmp_path)

        assert str(target) in result

    def test_returns_empty_when_no_matches(self, tmp_path: Path) -> None:
        unit_dir = tmp_path / "local-agent" / "tests" / "unit"
        unit_dir.mkdir(parents=True)

        result = _find_related_tests_for_files(
            [str(tmp_path / "local-agent" / "agent" / "nonexistent.py")],
            tmp_path,
        )

        assert result == []


# ---------------------------------------------------------------------------
# TestResolvePath
# ---------------------------------------------------------------------------
# These tests cover the path-repair behavior that lets OllamaCoder survive
# the model emitting a typo'd absolute path. This matters in particular for
# A/B runs where each run gets a UUID-suffixed worktree
# (e.g. /Users/gman/code/technomancer-aiw-cde0c3cb), which the model has been
# observed dropping a character from across long tool-calling sessions.

class TestResolvePath:
    def test_relative_path_joins_project_root(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        resolved = coder._resolve_path("local-agent/agent/foo.py")
        assert resolved == (tmp_path / "local-agent" / "agent" / "foo.py").resolve()

    def test_absolute_path_inside_project_root_unchanged(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        target = tmp_path / "local-agent" / "agent" / "foo.py"
        resolved = coder._resolve_path(str(target))
        assert resolved == target.resolve()

    def test_typoed_worktree_prefix_reanchors_via_local_agent(self, tmp_path: Path) -> None:
        """Regression test: when the model emits an absolute path with a typo'd
        worktree prefix (e.g. cde03cb instead of cde0c3cb), we should re-anchor
        at `local-agent/` and rebuild the path under the real project_root.
        """
        coder = _make_coder(tmp_path)
        typoed = "/Users/gman/code/technomancer-aiw-cde03cb/local-agent/agent/foo.py"
        resolved = coder._resolve_path(typoed)
        expected = (tmp_path / "local-agent" / "agent" / "foo.py").resolve()
        assert resolved == expected

    def test_completely_wrong_prefix_reanchors_via_local_agent(self, tmp_path: Path) -> None:
        """Even an absolute path under /tmp or anywhere else gets re-anchored
        if it contains a recognizable project segment."""
        coder = _make_coder(tmp_path)
        weird = "/tmp/somewhere/local-agent/tests/unit/test_x.py"
        resolved = coder._resolve_path(weird)
        expected = (tmp_path / "local-agent" / "tests" / "unit" / "test_x.py").resolve()
        assert resolved == expected

    def test_reanchor_via_docs_segment(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        resolved = coder._resolve_path("/wrong/prefix/docs/architecture.md")
        expected = (tmp_path / "docs" / "architecture.md").resolve()
        assert resolved == expected

    def test_no_recognizable_segment_clamps_to_basename(self, tmp_path: Path) -> None:
        """When the path can't be repaired, fall back to clamping to project_root/<basename>
        so we never silently write outside the sandbox."""
        coder = _make_coder(tmp_path)
        resolved = coder._resolve_path("/etc/passwd")
        assert resolved == (tmp_path / "passwd").resolve()

    def test_reanchor_prefers_first_known_segment(self, tmp_path: Path) -> None:
        """If multiple known segments appear, the first match in the path wins
        (giving us the deepest legitimate root)."""
        coder = _make_coder(tmp_path)
        # local-agent appears before tests, so re-anchor at local-agent
        path = "/wrong/local-agent/tests/unit/test_foo.py"
        resolved = coder._resolve_path(path)
        expected = (tmp_path / "local-agent" / "tests" / "unit" / "test_foo.py").resolve()
        assert resolved == expected

    def test_reanchor_does_not_escape_project_root(self, tmp_path: Path) -> None:
        """Defense-in-depth: even after re-anchoring, the result must be inside project_root.
        A path like /wrong/local-agent/../../../etc/passwd would re-anchor to
        local-agent/../../../etc/passwd which resolves outside — should be rejected.
        """
        coder = _make_coder(tmp_path)
        evil = "/wrong/local-agent/../../etc/passwd"
        resolved = coder._resolve_path(evil)
        # Should NOT resolve to /etc/passwd; should clamp to basename inside project_root.
        assert str(resolved).startswith(str(tmp_path.resolve()))

    def test_reanchor_inside_existing_project_root_path_is_passthrough(
        self, tmp_path: Path
    ) -> None:
        """If the absolute path is already inside project_root, no re-anchoring needed
        even though it contains a known segment."""
        coder = _make_coder(tmp_path)
        target = tmp_path / "local-agent" / "agent" / "bar.py"
        resolved = coder._resolve_path(str(target))
        assert resolved == target.resolve()


# ---------------------------------------------------------------------------
# TestDoublePrefixRepair
# ---------------------------------------------------------------------------
# Regression: in production, ``project_root`` for ``OllamaCoder`` is the
# worktree's ``local-agent/`` subdirectory.  The model habitually emits
# relative paths like ``local-agent/agent/foo.py`` (the way they appear in
# the repo and in the prompt context).  Naive joining produces
# ``.../local-agent/local-agent/agent/foo.py`` which never exists and led
# to phantom directories being created on every A/B run for hours.

class TestDoublePrefixRepair:
    def _make_coder_in_local_agent(self, tmp_path: Path) -> OllamaCoder:
        """Build a coder whose project_root ends in ``local-agent`` —
        exactly the production layout."""
        la = tmp_path / "local-agent"
        la.mkdir()
        state = MagicMock()
        state.log = lambda m: None
        state.cancelled = False
        return OllamaCoder(
            prompt="x",
            project_root=la,
            idea_id="TK-999",
            state=state,
            model="qwen3.5:27b",
            max_turns=5,
            max_rounds=3,
            num_ctx=4096,
        )

    def test_strips_redundant_local_agent_prefix(self, tmp_path: Path) -> None:
        coder = self._make_coder_in_local_agent(tmp_path)
        resolved = coder._resolve_path("local-agent/agent/foo.py")
        expected = (tmp_path / "local-agent" / "agent" / "foo.py").resolve()
        assert resolved == expected, f"expected {expected}, got {resolved}"

    def test_does_not_strip_when_project_root_basename_differs(
        self, tmp_path: Path
    ) -> None:
        """Backward compatibility: when project_root is NOT named
        ``local-agent``, treat ``local-agent/...`` as a literal subdir."""
        # tmp_path.name is some random pytest-* name, not "local-agent"
        coder = _make_coder(tmp_path)
        resolved = coder._resolve_path("local-agent/agent/foo.py")
        expected = (tmp_path / "local-agent" / "agent" / "foo.py").resolve()
        assert resolved == expected

    def test_path_without_local_agent_prefix_unchanged(self, tmp_path: Path) -> None:
        coder = self._make_coder_in_local_agent(tmp_path)
        resolved = coder._resolve_path("agent/foo.py")
        expected = (tmp_path / "local-agent" / "agent" / "foo.py").resolve()
        assert resolved == expected

    def test_bare_local_agent_resolves_to_project_root(self, tmp_path: Path) -> None:
        """``list_files(path='local-agent')`` should resolve to project_root
        itself, not to ``project_root/local-agent``."""
        coder = self._make_coder_in_local_agent(tmp_path)
        resolved = coder._resolve_path("local-agent")
        expected = (tmp_path / "local-agent").resolve()
        assert resolved == expected

    def test_reanchor_absolute_path_strips_double_local_agent(
        self, tmp_path: Path
    ) -> None:
        """Regression: in production the model emitted absolute paths like
        ``/Users/gman/code/technomancer-ai030/local-agent/tests/unit/x.py``
        (truncated worktree prefix). The re-anchor logic found ``local-agent``
        as the anchor, took the tail ``local-agent/tests/unit/x.py`` and
        joined it under project_root which itself ends in ``local-agent``,
        producing ``.../local-agent/local-agent/tests/unit/x.py``.
        Instead it must strip the leading anchor segment from the tail
        when project_root.name already equals the anchor.
        """
        coder = self._make_coder_in_local_agent(tmp_path)
        truncated_abs = "/Users/gman/code/technomancer-ai030/local-agent/tests/unit/x.py"
        resolved = coder._resolve_path(truncated_abs)
        expected = (tmp_path / "local-agent" / "tests" / "unit" / "x.py").resolve()
        assert resolved == expected, f"expected {expected}, got {resolved}"
        # Negative assertion: the doubled phantom path must NOT appear.
        assert "local-agent/local-agent" not in str(resolved).replace("\\", "/")

    def test_reanchor_absolute_path_with_correct_worktree_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Sanity: when the absolute path is already inside project_root,
        no re-anchoring runs at all even though the path contains
        ``local-agent``."""
        coder = self._make_coder_in_local_agent(tmp_path)
        target = tmp_path / "local-agent" / "agent" / "ok.py"
        resolved = coder._resolve_path(str(target))
        assert resolved == target.resolve()

    def test_write_file_refuses_nested_phantom_path(self, tmp_path: Path) -> None:
        """Defense-in-depth: even if path resolution were ever bypassed,
        ``write_file`` must refuse to write into the phantom nested tree."""
        # Build a coder whose project_root is the OUTER tmp_path so
        # ``_resolve_path`` does NOT strip; the resolved path will end up
        # in the nested-phantom shape and the guard must trigger.
        coder = _make_coder(tmp_path)
        # Force-create a path that resolves to .../local-agent/local-agent/...
        # by passing an absolute path directly inside project_root.
        nested = tmp_path / "local-agent" / "local-agent" / "agent" / "x.py"
        result = coder._tool_write_file(str(nested), "print('phantom')\n")
        assert "ERROR" in result
        assert "nested phantom" in result.lower()
        assert not nested.exists()


# ---------------------------------------------------------------------------
# TestDriftDetection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """The outer round loop should abort early when the model is making zero
    forward progress for DRIFT_WINDOW (=4) rounds straight. See
    ``OllamaCoder._check_drift`` for the rules."""

    def _wire_drift_test(
        self,
        coder: OllamaCoder,
        *,
        failing_per_round: list[list[str]],
        changed_per_round: list[list[str]],
        commit_count_per_call: list[int],
        inner_finished_per_round: list[bool],
    ) -> None:
        """Helper: wire deterministic per-round signals onto the coder.

        ``commit_count_per_call`` is consumed by ``_count_branch_commits`` —
        which is called once before the loop (baseline) and once per round
        (after that round's work), so the list must be at least
        len(rounds)+1.
        """
        finish_response = _response([_tool_call("finish", summary="done")])
        coder._chat_with_tools = lambda sys, msgs: finish_response  # type: ignore[method-assign]

        round_idx = [0]
        def mock_pytest():
            i = min(round_idx[0], len(failing_per_round) - 1)
            failing = failing_per_round[i]
            return {
                "passed": False,
                "failing": list(failing),
                "output": "FAILED stuff\n",
            }
        coder._run_pytest = mock_pytest  # type: ignore[method-assign]

        def mock_changed():
            i = min(round_idx[0], len(changed_per_round) - 1)
            return list(changed_per_round[i])
        coder._get_changed_files = mock_changed  # type: ignore[method-assign]

        commit_iter = iter(commit_count_per_call)
        coder._count_branch_commits = lambda: next(  # type: ignore[method-assign]
            commit_iter, commit_count_per_call[-1] if commit_count_per_call else 0
        )

        # _run_inner_loop returns True when finish was called, False on
        # max-turn exhaustion. Drive that directly via per-round list.
        inner_iter = iter(inner_finished_per_round)
        def mock_inner(sys, msgs, rn):
            round_idx[0] = rn
            return next(inner_iter, True)
        coder._run_inner_loop = mock_inner  # type: ignore[method-assign]

        coder._tag_round_commits = lambda r: None  # type: ignore[method-assign]

    def _run_with_logger(self, coder: OllamaCoder) -> list[str]:
        log_lines: list[str] = []
        coder._log = lambda m: log_lines.append(m)  # type: ignore[method-assign]
        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            coder.run()
        return log_lines

    def test_four_rounds_no_progress_aborts(self, tmp_path: Path) -> None:
        """Condition A: 4 rounds, same failing-set, same changed-set, no commits → abort."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 10
        same_failing = ["test_a", "test_b"]
        same_changed = ["a.py", "b.py"]
        self._wire_drift_test(
            coder,
            failing_per_round=[same_failing] * 10,
            changed_per_round=[same_changed] * 10,
            commit_count_per_call=[0] * 11,
            inner_finished_per_round=[True] * 10,
        )
        log_lines = self._run_with_logger(coder)

        round_log_lines = [m for m in log_lines if "--- Round " in m]
        # Started rounds 0,1,2,3 — drift fires at end of round 3 (index 3).
        assert len(round_log_lines) == 4, (
            f"Expected exactly 4 rounds before drift abort, got {len(round_log_lines)}"
        )
        assert any("Drift detected" in m for m in log_lines)
        assert any("no_progress" in m for m in log_lines)

    def test_four_rounds_with_commits_does_not_abort(self, tmp_path: Path) -> None:
        """Commits in every round → no drift, run all 10 rounds."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 10
        self._wire_drift_test(
            coder,
            failing_per_round=[["t1"]] * 10,
            changed_per_round=[["a.py"]] * 10,
            # Baseline 0, then 1,2,3,4,5,6,7,8,9,10 — increases each round.
            commit_count_per_call=list(range(11)),
            inner_finished_per_round=[True] * 10,
        )
        log_lines = self._run_with_logger(coder)

        round_log_lines = [m for m in log_lines if "--- Round " in m]
        assert len(round_log_lines) == 10
        assert not any("Drift detected" in m for m in log_lines)

    def test_three_rounds_no_progress_does_not_abort(self, tmp_path: Path) -> None:
        """History not yet full (only 3 rounds collected) → no drift."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 3  # forces only 3 rounds to run
        self._wire_drift_test(
            coder,
            failing_per_round=[["t1"]] * 3,
            changed_per_round=[[]] * 3,
            commit_count_per_call=[0] * 4,
            inner_finished_per_round=[True] * 3,
        )
        log_lines = self._run_with_logger(coder)
        # 3 rounds < DRIFT_WINDOW=4, so drift should NOT fire.
        assert not any("Drift detected" in m for m in log_lines)

    def test_failing_set_changes_does_not_abort(self, tmp_path: Path) -> None:
        """Failing-set shifts each round → not drift (model making progress)."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 5
        self._wire_drift_test(
            coder,
            # Different failing set each round — looks like progress, even
            # without commits, until condition B kicks in. We deliberately
            # provide non-empty changed_files so condition B can't fire.
            failing_per_round=[["t1"], ["t2"], ["t3"], ["t4"], ["t5"]],
            changed_per_round=[["a.py"]] * 5,
            commit_count_per_call=[0] * 6,
            inner_finished_per_round=[True] * 5,
        )
        log_lines = self._run_with_logger(coder)
        assert not any("Drift detected" in m for m in log_lines), (
            f"Drift should NOT fire when failing-set varies. Logs:\n"
            + "\n".join(log_lines[-15:])
        )

    def test_empty_shop_requires_two_exhaustions(self, tmp_path: Path) -> None:
        """Condition B requires >= 2 inner-loop exhaustions in the window.
        Only 1 exhaustion → no drift, even with all-empty changed-files."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 4  # exactly the window
        self._wire_drift_test(
            coder,
            # Failing-set varies → condition A blocked.
            failing_per_round=[["t1"], ["t2"], ["t3"], ["t4"]],
            changed_per_round=[[]] * 4,
            commit_count_per_call=[0] * 5,
            # Only 1 round exhausted (round 0). Threshold is 2.
            inner_finished_per_round=[False, True, True, True],
        )
        log_lines = self._run_with_logger(coder)
        assert not any("Drift detected" in m for m in log_lines)

    def test_empty_shop_with_two_exhaustions_aborts(self, tmp_path: Path) -> None:
        """Condition B: 4 rounds, all empty changed-files, 2 inner-loop exhaustions → abort."""
        coder = _make_coder(tmp_path)
        coder.max_rounds = 10
        self._wire_drift_test(
            coder,
            # Vary failing-set so condition A is blocked — only B can fire.
            failing_per_round=[["t1"], ["t2"], ["t3"], ["t4"]],
            changed_per_round=[[]] * 4,
            commit_count_per_call=[0] * 5,
            inner_finished_per_round=[False, False, True, True],
        )
        log_lines = self._run_with_logger(coder)
        round_log_lines = [m for m in log_lines if "--- Round " in m]
        assert len(round_log_lines) == 4
        assert any("Drift detected" in m for m in log_lines)
        assert any("empty_shop" in m for m in log_lines)

    def test_drift_abort_does_not_log_exhausted_rounds(self, tmp_path: Path) -> None:
        """Drift-break path must NOT log "Exhausted N rounds".

        The original code put a single ``self._log("Exhausted ...")`` after
        the for-loop. That message fires on EVERY exit, including
        drift-break — which produced a confusing log where a story died
        at round 3 of 20 but the log claimed "Exhausted 20 rounds".
        After the fix, drift uses a "Stopped at round X/Y (drift)"
        message and the for/else only logs "Exhausted" when the loop
        truly runs to completion.
        """
        coder = _make_coder(tmp_path)
        coder.max_rounds = 20
        same_failing = ["t1"]
        same_changed = ["a.py"]
        # 4 rounds same failing-set + same changed-set + zero commits → drift.
        self._wire_drift_test(
            coder,
            failing_per_round=[same_failing] * 10,
            changed_per_round=[same_changed] * 10,
            commit_count_per_call=[0] * 11,
            inner_finished_per_round=[True] * 10,
        )
        log_lines = self._run_with_logger(coder)

        assert any("Drift detected" in m for m in log_lines)
        assert any("Stopped at round 4/20 (drift)" in m for m in log_lines), (
            "Drift abort must log a 'Stopped at round X/Y (drift)' message "
            "so operators can see at a glance whether the run hit drift "
            "early or genuinely exhausted all rounds"
        )
        assert not any("Exhausted 20 rounds" in m for m in log_lines), (
            "Drift abort must NOT claim 'Exhausted 20 rounds' — the loop "
            "broke at round 3, the model never got 20 attempts"
        )

    def test_genuine_exhaustion_logs_exhausted(self, tmp_path: Path) -> None:
        """When the for-loop completes naturally, log "Exhausted N rounds".

        This is the path where every round had progress signals (commits or
        varying failing-set) so drift never fired, but tests still didn't
        pass at the end. The "Exhausted N rounds" message is the right
        terminal log here.
        """
        coder = _make_coder(tmp_path)
        coder.max_rounds = 5
        # Vary failing-set + changed-set + commits → drift never fires.
        self._wire_drift_test(
            coder,
            failing_per_round=[["t1"], ["t2"], ["t3"], ["t4"], ["t5"]],
            changed_per_round=[["a.py"], ["b.py"], ["c.py"], ["d.py"], ["e.py"]],
            commit_count_per_call=list(range(6)),
            inner_finished_per_round=[True] * 5,
        )
        log_lines = self._run_with_logger(coder)

        assert not any("Drift detected" in m for m in log_lines)
        assert any("Exhausted 5 rounds" in m for m in log_lines), (
            "Genuine round exhaustion must log 'Exhausted N rounds' "
            "(the for/else branch). Without this message operators can't "
            "distinguish 'model couldn't fix it in N tries' from a crash."
        )


# ---------------------------------------------------------------------------
# TestBashSandbox
# ---------------------------------------------------------------------------

class TestBashSandbox:
    """The bash sandbox should tutor the model when it picks the wrong tool —
    not just say BLOCKED. See BASH_SHAPE_GUIDANCE in ollama_coder.py."""

    def test_blocked_cat_suggests_read_file(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("cat /etc/hosts")
        assert "BLOCKED" in result
        assert "read_file" in result

    def test_blocked_find_suggests_list_files_or_search_code(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("find . -name '*.py'")
        assert "BLOCKED" in result
        assert ("list_files" in result) or ("search_code" in result)

    def test_blocked_ls_suggests_list_files(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("ls -la")
        assert "BLOCKED" in result
        assert "list_files" in result

    def test_blocked_grep_suggests_search_code(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("grep -r 'foo' .")
        assert "BLOCKED" in result
        assert "search_code" in result

    def test_blocked_unknown_command_falls_back_to_generic_message(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("xyzzy --help")
        assert "BLOCKED" in result
        # Generic guidance lists the structured tool family.
        assert "structured" in result.lower() or "read_file" in result

    def test_blocklist_still_overrides(self, tmp_path: Path) -> None:
        """Blocklist runs before the new guidance layer — git push must still block."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("git push origin main")
        assert "BLOCKED" in result
        assert "push origin" in result

    def test_pythonista_does_not_match_python(self, tmp_path: Path) -> None:
        """Word-boundary tightening: a command that *starts with* 'python' but
        is a different word entirely must be rejected."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("pythonista --version")
        assert "BLOCKED" in result

    def test_existing_allowlist_still_passes(self, tmp_path: Path) -> None:
        """Regression: `git status` (well-formed allowed command) still runs.
        We can't easily run subprocess in the test env, so just verify it
        DOESN'T return a BLOCKED message — the subprocess.run call may
        fail, which is fine."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("git status")
        assert "BLOCKED" not in result

    def test_bare_allowed_command_passes(self, tmp_path: Path) -> None:
        """Word-boundary check: bare `git` (no args) should also be allowed."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("git")
        assert "BLOCKED" not in result

    def test_blocked_message_includes_truncated_command(self, tmp_path: Path) -> None:
        """The error should echo back what the model tried (so it can see
        what was rejected and self-correct)."""
        coder = _make_coder(tmp_path)
        result = coder._tool_run_bash("cat some_file.py")
        # The first 80 chars of the original command should appear.
        assert "cat some_file.py" in result


# ---------------------------------------------------------------------------
# TestToolCallDedup
# ---------------------------------------------------------------------------

class TestToolCallDedup:
    """Per-round dedup short-circuits repeated tool calls.

    The plan's Tier 3 fix targets glm's search-loop pathology
    (16/51 exhausted_max_turns runs caused by repeated zero-result
    search_code calls). The threshold table is exercised here.
    """

    def test_signature_collides_regardless_of_arg_order(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        sig_a = coder._tool_signature("search_code", {"pattern": "x", "path": "."})
        sig_b = coder._tool_signature("search_code", {"path": ".", "pattern": "x"})
        assert sig_a == sig_b

    def test_signature_differs_on_different_args(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        sig_a = coder._tool_signature("search_code", {"pattern": "x"})
        sig_b = coder._tool_signature("search_code", {"pattern": "y"})
        assert sig_a != sig_b

    def test_dedup_returns_none_under_threshold(self, tmp_path: Path) -> None:
        """search_code threshold is 2 — the first two calls should pass."""
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {}
        args = {"pattern": "foo"}
        assert coder._check_dedup("search_code", args) is None
        assert coder._check_dedup("search_code", args) is None

    def test_dedup_suppresses_third_search_code(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {}
        args = {"pattern": "foo"}
        coder._check_dedup("search_code", args)
        coder._check_dedup("search_code", args)
        result = coder._check_dedup("search_code", args)
        assert result is not None
        assert result.startswith("[duplicate suppressed]")
        assert "search_code" in result

    def test_dedup_suppresses_second_edit_file(self, tmp_path: Path) -> None:
        """edit_file threshold is 1 — second identical call is suppressed."""
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {}
        args = {"path": "a.py", "old_string": "x", "new_string": "y"}
        assert coder._check_dedup("edit_file", args) is None
        result = coder._check_dedup("edit_file", args)
        assert result is not None
        assert result.startswith("[duplicate suppressed]")

    def test_dedup_ignores_finish_and_unknown_tools(self, tmp_path: Path) -> None:
        """Tools without a threshold entry are never deduped."""
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {}
        for _ in range(10):
            assert coder._check_dedup("finish", {"summary": "x"}) is None
            assert coder._check_dedup("not_a_tool", {}) is None

    def test_clear_dedup_for_path_drops_matching_signatures(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {}
        coder._check_dedup("read_file", {"path": "a.py"})
        coder._check_dedup("read_file", {"path": "b.py"})
        coder._check_dedup("search_code", {"pattern": "x", "path": "a.py"})
        coder._clear_dedup_for_path("a.py")
        # Anything mentioning 'a.py' is gone, b.py is preserved.
        keys = list(coder._tool_call_signatures.keys())
        assert all("'a.py'" not in key[1] for key in keys)
        assert any("'b.py'" in key[1] for key in keys)

    def test_clear_dedup_for_path_no_op_on_empty_path(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._tool_call_signatures = {("read_file", "[('path', 'a.py')]"): 1}
        coder._clear_dedup_for_path("")
        assert len(coder._tool_call_signatures) == 1

    def test_signature_handles_unhashable_args(self, tmp_path: Path) -> None:
        """args containing a list (e.g., 'paths': [...]) shouldn't crash."""
        coder = _make_coder(tmp_path)
        sig = coder._tool_signature("read_file", {"paths": ["a.py", "b.py"]})
        assert isinstance(sig, tuple)
        assert sig[0] == "read_file"


# ---------------------------------------------------------------------------
# TestBudgetWarning
# ---------------------------------------------------------------------------

class TestBudgetWarning:
    """75%-budget zero-edit warning fires once per round when the model
    has exhausted 3/4 of its turn budget without making any edits."""

    def test_warning_injected_when_zero_edits_at_75_percent(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)  # max_turns=5 → warning at turn 3
        coder._round_edit_count = 0
        coder._budget_warning_sent_this_round = False
        coder._tool_call_signatures = {}
        coder._nudge_sent_this_round = False

        # Plan a sequence of model responses: the first three turns the
        # model just narrates with no tool calls, the fourth turn it
        # finally calls finish so we can exit.
        responses = [
            _response(content=""),
            _response(content=""),
            _response(content=""),
            _response(tool_calls=[_tool_call("finish", summary="ok")]),
        ]
        coder._chat_with_tools = lambda sys, msgs: responses.pop(0)  # type: ignore[method-assign]

        # Capture all messages mutated by the inner loop.
        messages: list[dict] = [{"role": "user", "content": "go"}]
        coder._run_inner_loop("sys", messages, round_num=0)

        # The warning should appear at least once in the user-role messages.
        injected = [
            m for m in messages
            if m.get("role") == "user" and "75%" in m.get("content", "")
        ]
        assert len(injected) == 1, f"expected exactly one budget warning, got {len(injected)}"
        assert "have not edited any file" in injected[0]["content"]

    def test_warning_not_injected_if_edits_already_made(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._round_edit_count = 1  # already edited something
        coder._budget_warning_sent_this_round = False
        coder._tool_call_signatures = {}
        coder._nudge_sent_this_round = False

        responses = [
            _response(content=""),
            _response(content=""),
            _response(content=""),
            _response(tool_calls=[_tool_call("finish", summary="ok")]),
        ]
        coder._chat_with_tools = lambda sys, msgs: responses.pop(0)  # type: ignore[method-assign]

        messages: list[dict] = [{"role": "user", "content": "go"}]
        coder._run_inner_loop("sys", messages, round_num=0)

        injected = [
            m for m in messages
            if m.get("role") == "user" and "75%" in m.get("content", "")
        ]
        assert injected == []

    def test_warning_fires_at_most_once_per_round(self, tmp_path: Path) -> None:
        coder = _make_coder(tmp_path)
        coder._round_edit_count = 0
        coder._budget_warning_sent_this_round = False
        coder._tool_call_signatures = {}
        coder._nudge_sent_this_round = False

        # Even if the model never finishes, we should see exactly one warning.
        coder._chat_with_tools = lambda sys, msgs: _response(content="")  # type: ignore[method-assign]

        messages: list[dict] = [{"role": "user", "content": "go"}]
        coder._run_inner_loop("sys", messages, round_num=0)

        injected = [
            m for m in messages
            if m.get("role") == "user" and "75%" in m.get("content", "")
        ]
        assert len(injected) == 1
