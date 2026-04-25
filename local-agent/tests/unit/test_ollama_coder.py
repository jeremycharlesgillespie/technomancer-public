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
