"""Tests for idea_board.executor — pytest baseline, failure diffing, test targeting, and epic execution."""

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.executor import (
    MAX_FIX_RETRIES,
    PYTEST_TIMEOUT,
    RATE_LIMIT_KEYWORDS,
    RATE_LIMIT_MAX_LOG_LINES,
    ExecutionState,
    _append_execution_log_line,
    _build_epic_execution_context,
    _build_prior_failure_context,
    _build_story_prompt,
    _classify_rate_limit,
    _clear_execution_artifacts,
    _find_related_tests,
    _format_injected_epic_context,
    _has_branch_commits,
    _parse_pytest_failures,
    _post_deploy_comment,
    _prune_stale_execution_logs,
    _sync_progress_comment,
    _write_done_sentinel,
    execute_epic,
    mark_done,
    mark_failed,
)


# ---------------------------------------------------------------------------
# _parse_pytest_failures
# ---------------------------------------------------------------------------


class TestParsePytestFailures:
    """Test parsing of FAILED lines from pytest output."""

    def test_empty_output(self):
        assert _parse_pytest_failures("") == set()

    def test_all_passing(self):
        output = "1580 passed, 186 warnings in 407.27s (0:06:47)\n"
        assert _parse_pytest_failures(output) == set()

    def test_single_failure(self):
        output = (
            "FAILED tests/unit/test_core.py::test_something - AssertionError: bad\n"
            "1 failed, 1579 passed\n"
        )
        result = _parse_pytest_failures(output)
        assert result == {"tests/unit/test_core.py::test_something"}

    def test_multiple_failures(self):
        output = (
            "FAILED tests/unit/test_core.py::test_thread_safety - TimeoutError\n"
            "FAILED tests/unit/test_pdf_tools_extended.py"
            "::TestExtractTextFromPdf::test_no_parameters_returns_error - assert\n"
            "2 failed, 1578 passed\n"
        )
        result = _parse_pytest_failures(output)
        assert result == {
            "tests/unit/test_core.py::test_thread_safety",
            "tests/unit/test_pdf_tools_extended.py"
            "::TestExtractTextFromPdf::test_no_parameters_returns_error",
        }

    def test_failure_without_error_description(self):
        """pytest --tb=no -q produces FAILED lines without ' - ...' suffix."""
        output = (
            "FAILED tests/unit/test_core.py::test_thread_safety\n"
            "1 failed, 1579 passed\n"
        )
        result = _parse_pytest_failures(output)
        assert result == {"tests/unit/test_core.py::test_thread_safety"}

    def test_ignores_non_failed_lines(self):
        output = (
            "tests/unit/test_core.py ..F..\n"
            "FAILED tests/unit/test_core.py::test_x - err\n"
            "=============== short test summary ================\n"
            "1 failed, 4 passed\n"
        )
        result = _parse_pytest_failures(output)
        assert result == {"tests/unit/test_core.py::test_x"}

    def test_strips_whitespace(self):
        output = "  FAILED tests/unit/test_a.py::test_b - err  \n"
        result = _parse_pytest_failures(output)
        assert result == {"tests/unit/test_a.py::test_b"}

    def test_class_method_format(self):
        output = "FAILED tests/unit/test_foo.py::TestBar::test_baz - TypeError\n"
        result = _parse_pytest_failures(output)
        assert result == {"tests/unit/test_foo.py::TestBar::test_baz"}


# ---------------------------------------------------------------------------
# Baseline diff logic
# ---------------------------------------------------------------------------


class TestBaselineDiff:
    """Test that the set-subtraction logic correctly identifies new failures."""

    def test_all_failures_in_baseline(self):
        """When all failures are pre-existing, delta is empty."""
        baseline = {
            "tests/unit/test_core.py::test_thread_safety",
            "tests/unit/test_pdf.py::TestPdf::test_no_params",
        }
        current = {
            "tests/unit/test_core.py::test_thread_safety",
            "tests/unit/test_pdf.py::TestPdf::test_no_params",
        }
        delta = current - baseline
        assert delta == set()

    def test_new_failure_detected(self):
        """New failures not in baseline should be flagged."""
        baseline = {"tests/unit/test_core.py::test_thread_safety"}
        current = {
            "tests/unit/test_core.py::test_thread_safety",
            "tests/unit/test_new.py::test_broken",
        }
        delta = current - baseline
        assert delta == {"tests/unit/test_new.py::test_broken"}

    def test_empty_baseline(self):
        """With no baseline failures, all failures are new."""
        baseline: set[str] = set()
        current = {"tests/unit/test_new.py::test_broken"}
        delta = current - baseline
        assert delta == {"tests/unit/test_new.py::test_broken"}

    def test_baseline_failure_fixed(self):
        """A baseline failure that got fixed shouldn't appear in delta."""
        baseline = {
            "tests/unit/test_core.py::test_thread_safety",
            "tests/unit/test_old.py::test_flaky",
        }
        current = {"tests/unit/test_core.py::test_thread_safety"}
        delta = current - baseline
        assert delta == set()

    def test_empty_current(self):
        """All tests passing — delta is empty regardless of baseline."""
        baseline = {"tests/unit/test_core.py::test_thread_safety"}
        current: set[str] = set()
        delta = current - baseline
        assert delta == set()


# ---------------------------------------------------------------------------
# ExecutionState.baseline_failures field
# ---------------------------------------------------------------------------


class TestExecutionState:
    """Test that ExecutionState includes baseline_failures."""

    def test_baseline_failures_default_empty(self):
        state = ExecutionState(idea_id="test-1")
        assert state.baseline_failures == set()
        assert isinstance(state.baseline_failures, set)

    def test_baseline_failures_stored(self):
        state = ExecutionState(idea_id="test-1")
        state.baseline_failures = {"tests/unit/test_a.py::test_b"}
        assert len(state.baseline_failures) == 1

    def test_baseline_failures_independent_per_instance(self):
        """Each ExecutionState should have its own set (not shared)."""
        s1 = ExecutionState(idea_id="test-1")
        s2 = ExecutionState(idea_id="test-2")
        s1.baseline_failures.add("tests/unit/test_a.py::test_b")
        assert s2.baseline_failures == set()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify executor constants exist and are reasonable."""

    def test_pytest_timeout_is_10_min(self):
        assert PYTEST_TIMEOUT == 600

    def test_max_fix_retries(self):
        assert MAX_FIX_RETRIES == 5


# ---------------------------------------------------------------------------
# _find_related_tests
# ---------------------------------------------------------------------------


class TestFindRelatedTests:
    """Test mapping changed source files to their test files."""

    def test_finds_matching_test_file(self, tmp_path):
        """agent/core.py -> tests/unit/test_core.py"""
        la = tmp_path / "local-agent"
        (la / "agent").mkdir(parents=True)
        (la / "tests" / "unit").mkdir(parents=True)
        (la / "agent" / "core.py").write_text("x")
        test_file = la / "tests" / "unit" / "test_core.py"
        test_file.write_text("x")

        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="local-agent/agent/core.py\n"
            )
            result = _find_related_tests(tmp_path)

        # Normalize separators for cross-platform
        normalized = [r.replace("\\", "/") for r in result]
        assert "tests/unit/test_core.py" in normalized

    def test_finds_extended_test_file(self, tmp_path):
        """agent/foo.py -> tests/unit/test_foo_extended.py"""
        la = tmp_path / "local-agent"
        (la / "agent").mkdir(parents=True)
        (la / "tests" / "unit").mkdir(parents=True)
        (la / "agent" / "foo.py").write_text("x")
        ext_file = la / "tests" / "unit" / "test_foo_extended.py"
        ext_file.write_text("x")

        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="local-agent/agent/foo.py\n"
            )
            result = _find_related_tests(tmp_path)

        normalized = [r.replace("\\", "/") for r in result]
        assert "tests/unit/test_foo_extended.py" in normalized

    def test_no_matching_test(self, tmp_path):
        """Changed file with no corresponding test returns empty."""
        la = tmp_path / "local-agent"
        (la / "agent").mkdir(parents=True)
        (la / "tests" / "unit").mkdir(parents=True)

        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="local-agent/agent/obscure.py\n"
            )
            result = _find_related_tests(tmp_path)

        assert result == []

    def test_no_changes(self, tmp_path):
        la = tmp_path / "local-agent"
        (la / "tests" / "unit").mkdir(parents=True)

        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = _find_related_tests(tmp_path)

        assert result == []

    def test_deduplicates(self, tmp_path):
        """Same test file not listed twice."""
        la = tmp_path / "local-agent"
        (la / "agent").mkdir(parents=True)
        (la / "tests" / "unit").mkdir(parents=True)
        (la / "agent" / "core.py").write_text("x")
        (la / "tests" / "unit" / "test_core.py").write_text("x")

        with patch("idea_board.executor.subprocess.run") as mock_run:
            # Same module appears twice in diff (shouldn't happen, but be safe)
            mock_run.return_value = MagicMock(
                stdout="local-agent/agent/core.py\nlocal-agent/agent/core.py\n"
            )
            result = _find_related_tests(tmp_path)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# _build_epic_execution_context
# ---------------------------------------------------------------------------


class TestBuildEpicExecutionContext:
    """Test building context for epic story injection."""

    def test_with_context_and_results(self):
        epic = MagicMock()
        epic.epic_context = "Build a complete auth system"
        results = [
            {
                "id": "s-1",
                "title": "Add user model",
                "state": "done",
                "summary": "Created User table with migrations",
            },
        ]
        ctx = _build_epic_execution_context(epic, results)
        assert "Build a complete auth system" in ctx
        assert "Add user model" in ctx
        assert "Created User table" in ctx
        assert "Do not duplicate" in ctx

    def test_no_epic_context(self):
        epic = MagicMock()
        epic.epic_context = ""
        results = [
            {"id": "s-1", "title": "Story 1", "state": "done", "summary": "Done"},
        ]
        ctx = _build_epic_execution_context(epic, results)
        assert "## Epic Context" not in ctx
        assert "Story 1" in ctx

    def test_no_previous_results(self):
        epic = MagicMock()
        epic.epic_context = "Build something great"
        ctx = _build_epic_execution_context(epic, [])
        assert "Build something great" in ctx
        assert "Previous Stories" not in ctx

    def test_empty_context_and_no_results(self):
        epic = MagicMock()
        epic.epic_context = ""
        ctx = _build_epic_execution_context(epic, [])
        assert ctx == ""

    def test_skips_non_done_results(self):
        epic = MagicMock()
        epic.epic_context = ""
        results = [
            {"id": "s-1", "title": "Done story", "state": "done", "summary": "OK"},
            {"id": "s-2", "title": "Failed story", "state": "failed", "summary": "Err"},
        ]
        ctx = _build_epic_execution_context(epic, results)
        assert "Done story" in ctx
        assert "Failed story" not in ctx

    def test_truncates_long_summary(self):
        epic = MagicMock()
        epic.epic_context = ""
        long_summary = "x" * 1000
        results = [
            {"id": "s-1", "title": "Story", "state": "done", "summary": long_summary},
        ]
        ctx = _build_epic_execution_context(epic, results)
        # Summary truncated to 500 chars — total output well under 800
        assert len(ctx) < 800


# ---------------------------------------------------------------------------
# execute_epic
# ---------------------------------------------------------------------------


class TestExecuteEpic:
    """Test epic orchestration — sequential story execution."""

    @staticmethod
    def _make_mock_ideas():
        """Create mock ideas dict for testing."""
        epic = MagicMock()
        epic.id = "epic-1"
        epic.title = "Test Epic"
        epic.idea_type = "epic"
        epic.state = "approved"
        epic.epic_context = "End-to-end feature"

        s1 = MagicMock()
        s1.id = "s-1"
        s1.title = "Story 1"
        s1.state = "approved"
        s1.idea_type = "story"

        s2 = MagicMock()
        s2.id = "s-2"
        s2.title = "Story 2"
        s2.state = "approved"
        s2.idea_type = "story"

        return {"epic-1": epic, "s-1": s1, "s-2": s2}

    @staticmethod
    def _quick_state(idea_id):
        """Return an ExecutionState with an already-finished thread."""
        es = ExecutionState(idea_id=idea_id)
        es.log_lines.append("Done")
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        es.thread = t
        return es

    def test_all_stories_succeed(self):
        """All stories execute and complete — epic marked done."""
        ideas = self._make_mock_ideas()
        executed = []

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            executed.append(idea_id)
            ideas[idea_id].state = "done"
            return self._quick_state(idea_id)

        mock_mark_done = MagicMock()
        mock_mark_failed = MagicMock()

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=mock_mark_done,
            mark_failed=mock_mark_failed,
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        assert executed == ["s-1", "s-2"]
        mock_mark_done.assert_called_once()
        assert mock_mark_done.call_args[0][0] == "epic-1"
        mock_mark_failed.assert_not_called()

    def test_stops_on_failed_story(self):
        """When a story fails, epic stops and remaining stories are skipped."""
        ideas = self._make_mock_ideas()
        executed = []

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            executed.append(idea_id)
            # First story fails
            ideas[idea_id].state = "failed"
            return self._quick_state(idea_id)

        mock_mark_done = MagicMock()
        mock_mark_failed = MagicMock()

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=mock_mark_done,
            mark_failed=mock_mark_failed,
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        # s-2 was never started
        assert executed == ["s-1"]
        mock_mark_failed.assert_called()
        assert mock_mark_failed.call_args[0][0] == "epic-1"
        mock_mark_done.assert_not_called()

    def test_skips_done_stories(self):
        """Already-done stories are skipped."""
        ideas = self._make_mock_ideas()
        ideas["s-1"].state = "done"
        executed = []

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            executed.append(idea_id)
            ideas[idea_id].state = "done"
            return self._quick_state(idea_id)

        mock_mark_done = MagicMock()

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=mock_mark_done,
            mark_failed=MagicMock(),
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        # Only s-2 was executed; s-1 was skipped
        assert executed == ["s-2"]
        mock_mark_done.assert_called_once()

    def test_empty_execution_order(self):
        """Epic with no stories is marked failed immediately."""
        ideas = self._make_mock_ideas()

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        mock_mark_failed = MagicMock()

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            get_execution_order=lambda eid: [],
            mark_executing=MagicMock(),
            mark_done=MagicMock(),
            mark_failed=mock_mark_failed,
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None

        mock_mark_failed.assert_called_once()
        assert mock_mark_failed.call_args[0][0] == "epic-1"

    def test_not_found_returns_none(self):
        """Non-existent epic returns None."""
        from idea_board.executor import _active
        _active.clear()

        with patch("idea_board.executor.get_idea", return_value=None):
            result = execute_epic("nonexistent")
        assert result is None

    def test_non_epic_returns_none(self):
        """Calling execute_epic on a story returns None."""
        story = MagicMock()
        story.idea_type = "story"

        from idea_board.executor import _active
        _active.clear()

        with patch("idea_board.executor.get_idea", return_value=story):
            result = execute_epic("story-1")
        assert result is None

    def test_passes_epic_context_to_stories(self):
        """Structured epic_context and previous_results passed to execute_idea."""
        ideas = self._make_mock_ideas()
        captured_calls = []

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            captured_calls.append(kwargs)
            ideas[idea_id].state = "done"
            return self._quick_state(idea_id)

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=MagicMock(),
            mark_failed=MagicMock(),
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        # First story gets epic context but no previous results
        assert captured_calls[0]["epic_context"] == "End-to-end feature"
        assert captured_calls[0]["previous_results"] == []
        # Second story gets epic context AND first story's result
        assert captured_calls[1]["epic_context"] == "End-to-end feature"
        assert len(captured_calls[1]["previous_results"]) == 1
        assert captured_calls[1]["previous_results"][0]["title"] == "Story 1"

    def test_all_stories_already_done(self):
        """When all stories are done, epic is marked done without executing any."""
        ideas = self._make_mock_ideas()
        ideas["s-1"].state = "done"
        ideas["s-2"].state = "done"
        executed = []

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            executed.append(idea_id)
            return self._quick_state(idea_id)

        mock_mark_done = MagicMock()

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=mock_mark_done,
            mark_failed=MagicMock(),
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        # No stories executed
        assert executed == []
        mock_mark_done.assert_called_once()
        assert mock_mark_done.call_args[0][0] == "epic-1"

    def test_logs_progress(self):
        """Epic execution state captures progress log lines."""
        ideas = self._make_mock_ideas()

        def mock_get_idea(idea_id):
            return ideas.get(idea_id)

        def mock_execute_idea(idea_id, **kwargs):
            ideas[idea_id].state = "done"
            return self._quick_state(idea_id)

        from idea_board.executor import _active
        _active.clear()

        with patch.multiple(
            "idea_board.executor",
            get_idea=mock_get_idea,
            execute_idea=mock_execute_idea,
            get_execution_order=lambda eid: ["s-1", "s-2"],
            mark_executing=MagicMock(),
            mark_done=MagicMock(),
            mark_failed=MagicMock(),
            _notify_discord=MagicMock(),
        ):
            result = execute_epic("epic-1")
            assert result is not None
            result.thread.join(timeout=10)

        log = result.log_text
        assert "Epic Executor: Test Epic" in log
        assert "Stories to execute: 2" in log
        assert "[DONE] s-1" in log
        assert "[DONE] s-2" in log
        assert "All 2 stories completed" in log


# ---------------------------------------------------------------------------
# _format_injected_epic_context
# ---------------------------------------------------------------------------


class TestFormatInjectedEpicContext:
    """Test formatting of epic context and previous results for prompt injection."""

    def test_with_context_and_results(self):
        ctx = _format_injected_epic_context(
            "Build a complete auth system",
            [
                {
                    "id": "s-1",
                    "title": "Add user model",
                    "state": "done",
                    "summary": "Created User table with migrations",
                },
            ],
        )
        assert "## Epic Context" in ctx
        assert "Build a complete auth system" in ctx
        assert "## Previous Stories (already completed)" in ctx
        assert "Add user model" in ctx
        assert "Created User table" in ctx
        assert "Do not duplicate" in ctx

    def test_no_epic_context(self):
        ctx = _format_injected_epic_context(
            "",
            [{"id": "s-1", "title": "Story 1", "state": "done", "summary": "Done"}],
        )
        assert "## Epic Context" not in ctx
        assert "Story 1" in ctx

    def test_no_previous_results(self):
        ctx = _format_injected_epic_context("Build something great", [])
        assert "Build something great" in ctx
        assert "Previous Stories" not in ctx

    def test_none_previous_results(self):
        ctx = _format_injected_epic_context("Big goal", None)
        assert "Big goal" in ctx
        assert "Previous Stories" not in ctx

    def test_empty_context_and_no_results(self):
        assert _format_injected_epic_context("", []) == ""

    def test_empty_context_and_none_results(self):
        assert _format_injected_epic_context("", None) == ""

    def test_skips_non_done_results(self):
        ctx = _format_injected_epic_context(
            "",
            [
                {"id": "s-1", "title": "Done story", "state": "done", "summary": "OK"},
                {"id": "s-2", "title": "Failed story", "state": "failed", "summary": "Err"},
            ],
        )
        assert "Done story" in ctx
        assert "Failed story" not in ctx

    def test_truncates_long_summary(self):
        long_summary = "x" * 1000
        ctx = _format_injected_epic_context(
            "",
            [{"id": "s-1", "title": "Story", "state": "done", "summary": long_summary}],
        )
        # Summary truncated to 500 chars
        assert len(ctx) < 800


# ---------------------------------------------------------------------------
# _build_story_prompt with epic context injection
# ---------------------------------------------------------------------------


class TestBuildStoryPromptEpicContext:
    """Test that _build_story_prompt injects epic context and previous results."""

    @staticmethod
    def _make_story():
        idea = MagicMock()
        idea.id = "s-1"
        idea.title = "Add API endpoint"
        idea.idea_type = "story"
        idea.category = "feature"
        idea.parent_id = "epic-1"
        idea.description = "Create the /api/users endpoint"
        return idea

    def test_includes_epic_context_when_provided(self):
        idea = self._make_story()
        with patch.multiple(
            "idea_board.executor",
            _enrich_stub_description=lambda i: i.description,
            _build_epic_context=lambda i: "",
            _build_discussion=lambda i: "",
            _build_prior_failure_context=lambda i: "",
            _load_codebase_summary=lambda: "",
            _get_category_guidance=lambda c: "",
            _find_relevant_test_file=lambda i: "",
            _build_workflow_section=lambda i: "",
        ):
            prompt = _build_story_prompt(
                idea,
                epic_context="Build a complete auth system",
                previous_results=[
                    {
                        "id": "s-0",
                        "title": "Create data model",
                        "state": "done",
                        "summary": "Added User model with migrations",
                    },
                ],
            )

        assert "## Epic Context" in prompt
        assert "Build a complete auth system" in prompt
        assert "## Previous Stories (already completed)" in prompt
        assert "Create data model" in prompt
        assert "Added User model" in prompt

    def test_no_epic_sections_without_params(self):
        idea = self._make_story()
        with patch.multiple(
            "idea_board.executor",
            _enrich_stub_description=lambda i: i.description,
            _build_epic_context=lambda i: "",
            _build_discussion=lambda i: "",
            _build_prior_failure_context=lambda i: "",
            _load_codebase_summary=lambda: "",
            _get_category_guidance=lambda c: "",
            _find_relevant_test_file=lambda i: "",
            _build_workflow_section=lambda i: "",
        ):
            prompt = _build_story_prompt(idea)

        assert "## Epic Context" not in prompt
        assert "## Previous Stories" not in prompt

    def test_preserves_story_details(self):
        """Epic context injection doesn't clobber the story's own details."""
        idea = self._make_story()
        with patch.multiple(
            "idea_board.executor",
            _enrich_stub_description=lambda i: i.description,
            _build_epic_context=lambda i: "",
            _build_discussion=lambda i: "",
            _build_prior_failure_context=lambda i: "",
            _load_codebase_summary=lambda: "",
            _get_category_guidance=lambda c: "",
            _find_relevant_test_file=lambda i: "",
            _build_workflow_section=lambda i: "",
        ):
            prompt = _build_story_prompt(
                idea,
                epic_context="Big picture goal",
            )

        assert "Add API endpoint" in prompt
        assert "s-1" in prompt
        assert "Create the /api/users endpoint" in prompt


# ---------------------------------------------------------------------------
# _build_prior_failure_context — retry-memory injector
# ---------------------------------------------------------------------------


class _StubComment:
    """Minimal stand-in for board.provider.Comment.

    Tests only touch ``text`` and ``marker`` — using a lightweight object
    avoids pulling the provider layer into these unit tests.
    """

    def __init__(self, text: str, marker: str | None, author: str = "claude"):
        self.text = text
        self.marker = marker
        self.author = author
        self.created = "2026-04-15T10:00:00.000+0000"


class TestBuildPriorFailureContext:
    """Test injection of the most-recent [Execution Log - Failed] comment."""

    @staticmethod
    def _idea(idea_id: str = "TK-396"):
        idea = MagicMock()
        idea.id = idea_id
        return idea

    def _patch_comments(self, comments):
        provider = MagicMock()
        provider.get_comments.return_value = comments
        return patch(
            "idea_board.executor._get_board_provider", return_value=provider
        )

    def test_returns_empty_when_no_comments(self):
        with self._patch_comments([]):
            assert _build_prior_failure_context(self._idea()) == ""

    def test_returns_empty_when_no_failure_markers(self):
        """Progress and free-form comments are ignored — only failure markers inject."""
        comments = [
            _StubComment("just a note", marker=None),
            _StubComment("[AIM Progress]\nstep 1 done", marker="[AIM Progress]"),
            _StubComment("[Execution Log]\nsuccess trace", marker="[Execution Log]"),
        ]
        with self._patch_comments(comments):
            assert _build_prior_failure_context(self._idea()) == ""

    def test_includes_failure_text_and_header(self):
        failure_text = (
            "[Execution Log - Failed]\n"
            "Traceback: NameError: name 'settings' is not defined"
        )
        comments = [_StubComment(failure_text, marker="[Execution Log - Failed]")]
        with self._patch_comments(comments):
            result = _build_prior_failure_context(self._idea())

        assert "## Prior Failure Context" in result
        assert "NameError" in result
        assert "avoid repeating the same mistakes" in result
        assert "root cause" in result

    def test_uses_most_recent_failure_when_multiple(self):
        """Only the newest failure is injected (list is newest-last)."""
        comments = [
            _StubComment(
                "[Execution Log - Failed]\nFIRST failure: old trace",
                marker="[Execution Log - Failed]",
            ),
            _StubComment("[AIM Progress]\nretry started", marker="[AIM Progress]"),
            _StubComment(
                "[Execution Log - Failed]\nSECOND failure: newer trace",
                marker="[Execution Log - Failed]",
            ),
        ]
        with self._patch_comments(comments):
            result = _build_prior_failure_context(self._idea())

        assert "SECOND failure" in result
        assert "FIRST failure" not in result

    def test_truncates_long_failure_from_front(self):
        """Front is trimmed, tail is preserved — the tail has the useful traceback."""
        # 5000 chars of setup noise + a unique sentinel at the end
        noise = "X" * 5000
        tail = "UNIQUE_TAIL_MARKER: this is the traceback"
        body = f"[Execution Log - Failed]\n{noise}\n{tail}"
        comments = [_StubComment(body, marker="[Execution Log - Failed]")]
        with self._patch_comments(comments):
            result = _build_prior_failure_context(self._idea())

        # Tail is preserved, truncation marker is added
        assert "UNIQUE_TAIL_MARKER" in result
        assert "truncated" in result
        # The head of the noise block is dropped
        assert "X" * 5000 not in result

    def test_short_failure_is_not_truncated(self):
        """A failure under the cap is injected verbatim, no truncation marker."""
        body = "[Execution Log - Failed]\nshort and sweet trace"
        comments = [_StubComment(body, marker="[Execution Log - Failed]")]
        with self._patch_comments(comments):
            result = _build_prior_failure_context(self._idea())

        assert "short and sweet trace" in result
        assert "truncated" not in result

    def test_returns_empty_when_provider_raises(self):
        """Provider failure must not break prompt assembly."""
        provider = MagicMock()
        provider.get_comments.side_effect = RuntimeError("Jira down")
        with patch(
            "idea_board.executor._get_board_provider", return_value=provider
        ):
            assert _build_prior_failure_context(self._idea()) == ""

    def test_returns_empty_when_provider_lacks_get_comments(self):
        """Older provider without get_comments degrades gracefully."""
        provider = object()  # No get_comments attribute
        with patch(
            "idea_board.executor._get_board_provider", return_value=provider
        ):
            assert _build_prior_failure_context(self._idea()) == ""


class TestBuildStoryPromptPriorFailure:
    """Integration: _build_story_prompt wires _build_prior_failure_context in."""

    @staticmethod
    def _make_story():
        idea = MagicMock()
        idea.id = "TK-396"
        idea.title = "Inject prior failure"
        idea.idea_type = "story"
        idea.category = "quality"
        idea.parent_id = None
        idea.description = "Wire retry memory into the executor prompt"
        return idea

    def _patch_sections(self, comments):
        provider = MagicMock()
        provider.get_comments.return_value = comments
        return patch.multiple(
            "idea_board.executor",
            _enrich_stub_description=lambda i: i.description,
            _build_epic_context=lambda i: "",
            _build_discussion=lambda i: "",
            _load_codebase_summary=lambda: "CODEBASE_SUMMARY_SENTINEL",
            _get_category_guidance=lambda c: "",
            _find_relevant_test_file=lambda i: "",
            _build_workflow_section=lambda i: "",
            _get_board_provider=MagicMock(return_value=provider),
        )

    def test_injects_section_when_failure_exists(self):
        comments = [
            _StubComment(
                "[Execution Log - Failed]\nSENTINEL_FAIL_TEXT",
                marker="[Execution Log - Failed]",
            ),
        ]
        with self._patch_sections(comments):
            prompt = _build_story_prompt(self._make_story())

        assert "## Prior Failure Context" in prompt
        assert "SENTINEL_FAIL_TEXT" in prompt

    def test_omits_section_when_no_failure(self):
        with self._patch_sections([]):
            prompt = _build_story_prompt(self._make_story())

        assert "## Prior Failure Context" not in prompt

    def test_section_precedes_codebase_summary(self):
        """Failure context is placed before the codebase summary so the LLM reads it early."""
        comments = [
            _StubComment(
                "[Execution Log - Failed]\nFAILURE_MARKER_TEXT",
                marker="[Execution Log - Failed]",
            ),
        ]
        with self._patch_sections(comments):
            prompt = _build_story_prompt(self._make_story())

        assert "## Prior Failure Context" in prompt
        assert "CODEBASE_SUMMARY_SENTINEL" in prompt
        assert prompt.index("## Prior Failure Context") < prompt.index(
            "CODEBASE_SUMMARY_SENTINEL"
        )


# ---------------------------------------------------------------------------
# [TK-421] Trimmed ambient sections — git log, crash log, similar execution logs
# ---------------------------------------------------------------------------


class TestBuildStoryPromptTrimmedSections:
    """_build_story_prompt no longer emits low-signal ambient sections.

    TK-421 dropped three sections that carried little signal for Claude's
    story implementation: recent git commits, crash_log tail, and prior
    execution logs from same-category ideas. These headers must not appear.
    """

    @staticmethod
    def _make_story():
        idea = MagicMock()
        idea.id = "TK-421-test"
        idea.title = "Trim test"
        idea.idea_type = "story"
        idea.category = "quality"
        idea.parent_id = None
        idea.description = "verify low-signal sections are gone"
        idea.comments = []
        return idea

    def test_story_prompt_omits_trimmed_headers(self):
        provider = MagicMock()
        provider.get_comments.return_value = []
        with patch(
            "idea_board.executor._get_board_provider", return_value=provider
        ):
            prompt = _build_story_prompt(self._make_story())

        assert "## Recent Changes (git log)" not in prompt
        assert "## Recent Errors (from crash log)" not in prompt
        assert "## Reference: How similar ideas were implemented" not in prompt

    def test_story_prompt_keeps_task_critical_sections(self):
        """Sanity: the sections preservation-required by the spec remain."""
        provider = MagicMock()
        provider.get_comments.return_value = []
        with patch(
            "idea_board.executor._get_board_provider", return_value=provider
        ):
            prompt = _build_story_prompt(self._make_story())

        assert "verify low-signal sections are gone" in prompt  # description
        assert "## MANDATORY WORKFLOW" in prompt  # workflow section


# ---------------------------------------------------------------------------
# _post_deploy_comment — [Deployed] SHA + timestamp comment on successful deploy
# ---------------------------------------------------------------------------


class TestPostDeployComment:
    """Verify the [Deployed] comment posted after a successful merge."""

    def test_posts_comment_with_expected_format(self):
        """Comment is posted via provider.add_comment with the documented format."""
        provider = MagicMock()
        fake_now = datetime(2026, 4, 15, 20, 30, 45, tzinfo=timezone.utc)
        fake_datetime = MagicMock()
        fake_datetime.now.return_value = fake_now

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.datetime", fake_datetime):
            _post_deploy_comment("TK-400", "abc1234")

        provider.add_comment.assert_called_once()
        args, kwargs = provider.add_comment.call_args
        assert args[0] == "TK-400"
        assert kwargs["author"] == "executor"
        assert kwargs["text"] == "[Deployed] abc1234 at 2026-04-15T20:30:45+00:00"
        fake_datetime.now.assert_called_once_with(timezone.utc)

    def test_timestamp_is_iso8601_utc_seconds(self):
        """Generated timestamp matches ISO-8601 UTC with second precision."""
        provider = MagicMock()
        with patch("idea_board.executor._get_board_provider", return_value=provider):
            _post_deploy_comment("TK-401", "deadbee")

        text = provider.add_comment.call_args.kwargs["text"]
        # Format: [Deployed] <7-char sha> at <YYYY-MM-DDTHH:MM:SS+00:00>
        pattern = r"^\[Deployed\] [0-9a-f]{7} at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
        assert re.match(pattern, text), f"text does not match expected format: {text!r}"

    def test_swallows_provider_errors_and_logs(self, caplog):
        """A failing provider doesn't propagate — it's logged as a warning."""
        provider = MagicMock()
        provider.add_comment.side_effect = RuntimeError("jira 500")

        with patch("idea_board.executor._get_board_provider", return_value=provider):
            with caplog.at_level("WARNING", logger="idea_board.executor"):
                _post_deploy_comment("TK-402", "1234567")  # must not raise

        provider.add_comment.assert_called_once()
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(
            "deploy comment" in m.lower() and "TK-402" in m for m in messages
        )

    def test_swallows_provider_factory_errors(self):
        """An error fetching the provider is swallowed too (belt-and-braces)."""
        with patch(
            "idea_board.executor._get_board_provider",
            side_effect=RuntimeError("no provider"),
        ):
            # Should not raise
            _post_deploy_comment("TK-403", "abcdef1")

    def test_passes_short_sha_through_as_given(self):
        """Shortening SHAs is the caller's responsibility — we pass through."""
        provider = MagicMock()
        with patch("idea_board.executor._get_board_provider", return_value=provider):
            _post_deploy_comment("TK-404", "feedface")  # 8 chars on purpose

        text = provider.add_comment.call_args.kwargs["text"]
        assert "feedface" in text


# ---------------------------------------------------------------------------
# _sync_progress_comment — leads body with clickable Live log URL (TK-429)
# ---------------------------------------------------------------------------


class TestSyncProgressCommentLiveUrl:
    """The Jira progress comment body starts with a Live log: <url> line."""

    def _state(self, lines):
        state = ExecutionState(idea_id="TK-429")
        state.log_lines = list(lines)
        return state

    def test_prepends_live_url_with_blank_line_separator(self):
        provider = MagicMock()
        provider.append_progress_comment = MagicMock()
        fake_settings = MagicMock(server_host="myhost.local")
        state = self._state(["line one", "line two"])

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.settings", fake_settings):
            _sync_progress_comment("TK-429", state)

        provider.append_progress_comment.assert_called_once()
        idea_id, body = provider.append_progress_comment.call_args.args
        assert idea_id == "TK-429"
        assert body.startswith("Live log: http://myhost.local:8322/live/TK-429\n\n")
        assert body.endswith("line one\nline two")

    def test_uses_idea_id_in_url_path(self):
        provider = MagicMock()
        fake_settings = MagicMock(server_host="localhost")
        state = self._state(["x"])

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.settings", fake_settings):
            _sync_progress_comment("TK-9999", state)

        body = provider.append_progress_comment.call_args.args[1]
        first_line = body.splitlines()[0]
        assert first_line == "Live log: http://localhost:8322/live/TK-9999"

    def test_no_call_when_log_is_blank(self):
        """Empty/whitespace logs still skip the call — no header-only comments."""
        provider = MagicMock()
        fake_settings = MagicMock(server_host="localhost")
        state = self._state(["", "   "])

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.settings", fake_settings):
            _sync_progress_comment("TK-429", state)

        provider.append_progress_comment.assert_not_called()

    def test_silent_noop_when_provider_lacks_appender(self):
        """LocalProvider has no append_progress_comment — must not raise."""
        provider = MagicMock(spec=[])  # no append_progress_comment attr
        state = self._state(["line"])

        with patch("idea_board.executor._get_board_provider", return_value=provider):
            _sync_progress_comment("TK-429", state)  # must not raise

    def test_swallows_provider_errors(self):
        provider = MagicMock()
        provider.append_progress_comment.side_effect = RuntimeError("jira down")
        fake_settings = MagicMock(server_host="localhost")
        state = self._state(["line"])

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.settings", fake_settings):
            _sync_progress_comment("TK-429", state)  # must not raise

    def test_only_last_30_lines_included(self):
        """The tail-30 behavior is preserved alongside the new URL line."""
        provider = MagicMock()
        fake_settings = MagicMock(server_host="localhost")
        state = self._state([f"line {i}" for i in range(50)])

        with patch("idea_board.executor._get_board_provider", return_value=provider), \
             patch("idea_board.executor.settings", fake_settings):
            _sync_progress_comment("TK-429", state)

        body = provider.append_progress_comment.call_args.args[1]
        tail = body.split("\n\n", 1)[1]
        assert tail.splitlines() == [f"line {i}" for i in range(20, 50)]


# ---------------------------------------------------------------------------
# Claude rate-limit detection (TK-410)
# ---------------------------------------------------------------------------


class TestExecutionStateRateLimited:
    """ExecutionState gains a rate_limited flag the Worker inspects."""

    def test_default_false(self):
        state = ExecutionState(idea_id="idea-1")
        assert state.rate_limited is False

    def test_independent_per_instance(self):
        s1 = ExecutionState(idea_id="idea-1")
        s2 = ExecutionState(idea_id="idea-2")
        s1.rate_limited = True
        assert s2.rate_limited is False


class TestRateLimitKeywords:
    """Sanity checks on the public keyword tuple."""

    def test_contains_common_signals(self):
        joined = " ".join(RATE_LIMIT_KEYWORDS)
        # Spot-check a few well-known indicators.
        for expected in ("rate_limit", "429", "overloaded", "quota", "billing"):
            assert expected in joined

    def test_is_tuple(self):
        # Tuples are immutable — tests shouldn't accidentally mutate module state.
        assert isinstance(RATE_LIMIT_KEYWORDS, tuple)


class TestHasBranchCommits:
    """_has_branch_commits: delegate to git log, swallow errors."""

    def test_returns_true_when_log_nonempty(self, tmp_path):
        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="abcdef1 [TK-410] Some work\n",
            )
            assert _has_branch_commits(tmp_path) is True

    def test_returns_false_when_log_empty(self, tmp_path):
        with patch("idea_board.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            assert _has_branch_commits(tmp_path) is False

    def test_returns_false_on_git_error(self, tmp_path):
        with patch("idea_board.executor.subprocess.run", side_effect=OSError("git missing")):
            assert _has_branch_commits(tmp_path) is False


class TestClassifyRateLimit:
    """_classify_rate_limit: all three signals must fire."""

    def _make_state(self, lines, idea_id="idea-1"):
        state = ExecutionState(idea_id=idea_id)
        state.log_lines.extend(lines)
        return state

    def test_success_never_rate_limited(self, tmp_path):
        state = self._make_state(["Started", "rate_limit_error seen"])
        # claude_succeeded=True short-circuits immediately.
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, True, tmp_path) is False

    def test_long_log_is_not_rate_limit(self, tmp_path):
        # Too many lines → treat as a real failure even if keyword shows up.
        lines = [f"line {i}" for i in range(RATE_LIMIT_MAX_LOG_LINES + 5)]
        lines.append("rate_limit_error")
        state = self._make_state(lines)
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is False

    def test_no_keyword_is_not_rate_limit(self, tmp_path):
        state = self._make_state(["starting", "ImportError: no module"])
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is False

    def test_commits_present_is_not_rate_limit(self, tmp_path):
        # If Claude actually committed something, this isn't a rate-limit
        # bail-out — treat as a real failure.
        state = self._make_state(["Error: rate_limit_error"])
        with patch("idea_board.executor._has_branch_commits", return_value=True):
            assert _classify_rate_limit(state, False, tmp_path) is False

    def test_all_signals_present_returns_true(self, tmp_path):
        state = self._make_state([
            "Setting up branch",
            "Creating branch 2026-04-15-TK-410",
            "Branch created",
            "Starting Claude Code",
            "Error: rate_limit_error (429)",
        ])
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is True

    def test_detects_credit_balance_keyword(self, tmp_path):
        state = self._make_state([
            "Starting Claude",
            "API error: Credit balance is too low for this request",
        ])
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is True

    def test_detects_overloaded_keyword(self, tmp_path):
        state = self._make_state(["Claude started", "overloaded_error from upstream"])
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is True

    def test_keyword_match_is_case_insensitive(self, tmp_path):
        state = self._make_state(["Rate Limit exceeded — try again later"])
        with patch("idea_board.executor._has_branch_commits", return_value=False):
            assert _classify_rate_limit(state, False, tmp_path) is True


# ---------------------------------------------------------------------------
# Per-execution streaming log files (execution_logs/<idea_id>.log)
# ---------------------------------------------------------------------------


class TestAppendExecutionLogLine:
    """_append_execution_log_line: per-line append to execution_logs/<id>.log."""

    def test_creates_directory_and_file(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _append_execution_log_line("TK-427", "first line")

        log_file = logs_dir / "TK-427.log"
        assert logs_dir.is_dir()
        assert log_file.exists()
        assert log_file.read_text(encoding="utf-8") == "first line\n"

    def test_appends_multiple_lines(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _append_execution_log_line("TK-427", "line one")
        _append_execution_log_line("TK-427", "line two")
        _append_execution_log_line("TK-427", "line three")

        content = (logs_dir / "TK-427.log").read_text(encoding="utf-8")
        assert content == "line one\nline two\nline three\n"

    def test_separate_files_per_idea(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _append_execution_log_line("TK-100", "alpha")
        _append_execution_log_line("TK-200", "beta")

        assert (logs_dir / "TK-100.log").read_text(encoding="utf-8") == "alpha\n"
        assert (logs_dir / "TK-200.log").read_text(encoding="utf-8") == "beta\n"

    def test_writes_utf8(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _append_execution_log_line("TK-427", "emoji: 🚀 accents: é")

        content = (logs_dir / "TK-427.log").read_text(encoding="utf-8")
        assert content == "emoji: 🚀 accents: é\n"

    def test_io_error_is_swallowed(self, tmp_path, monkeypatch):
        # Point at a path whose parent can't be created (collision with file).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setattr(
            "idea_board.executor.EXECUTION_LOGS_DIR", blocker / "execution_logs",
        )

        # Should not raise even though mkdir fails.
        _append_execution_log_line("TK-427", "should not explode")


class TestExecutionStateLogWritesFile:
    """ExecutionState.log() mirrors every line to the per-execution file."""

    def test_log_creates_file_on_first_call(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        state = ExecutionState(idea_id="TK-427")
        state.log("starting execution")

        log_file = logs_dir / "TK-427.log"
        assert log_file.exists()
        assert "starting execution" in log_file.read_text(encoding="utf-8")

    def test_each_state_log_appears_in_file(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        state = ExecutionState(idea_id="TK-427")
        state.log("phase 1")
        state.log("phase 2")
        state.log("phase 3")

        lines = (logs_dir / "TK-427.log").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert "phase 1" in lines[0]
        assert "phase 2" in lines[1]
        assert "phase 3" in lines[2]

    def test_file_mirrors_log_lines_buffer(self, tmp_path, monkeypatch):
        """In-memory log_lines and on-disk file must stay in sync line-for-line."""
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        state = ExecutionState(idea_id="TK-427")
        for i in range(5):
            state.log(f"message {i}")

        file_lines = (logs_dir / "TK-427.log").read_text(encoding="utf-8").splitlines()
        assert file_lines == state.log_lines


class TestPruneStaleExecutionLogs:
    """_prune_stale_execution_logs: delete .log files not in _active."""

    def test_removes_all_logs_when_active_empty(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-100.log").write_text("stale a")
        (logs_dir / "TK-200.log").write_text("stale b")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)
        monkeypatch.setattr("idea_board.executor._active", {})

        _prune_stale_execution_logs()

        assert not (logs_dir / "TK-100.log").exists()
        assert not (logs_dir / "TK-200.log").exists()

    def test_keeps_active_logs(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-100.log").write_text("active")
        (logs_dir / "TK-200.log").write_text("stale")
        (logs_dir / "TK-300.log").write_text("stale")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)
        monkeypatch.setattr(
            "idea_board.executor._active",
            {"TK-100": ExecutionState(idea_id="TK-100")},
        )

        _prune_stale_execution_logs()

        assert (logs_dir / "TK-100.log").exists()
        assert not (logs_dir / "TK-200.log").exists()
        assert not (logs_dir / "TK-300.log").exists()

    def test_no_dir_is_not_an_error(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", missing)
        monkeypatch.setattr("idea_board.executor._active", {})

        _prune_stale_execution_logs()  # Must not raise
        assert not missing.exists()

    def test_ignores_non_log_files(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-100.log").write_text("stale")
        (logs_dir / "README.md").write_text("keep me")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)
        monkeypatch.setattr("idea_board.executor._active", {})

        _prune_stale_execution_logs()

        assert not (logs_dir / "TK-100.log").exists()
        assert (logs_dir / "README.md").exists()

    def test_file_persists_after_execution_completes(self, tmp_path, monkeypatch):
        """AC: on completion, the file remains — the reader uses it.

        Simulates a full execution by writing lines, then removing the idea
        from _active. The log file must still exist until the next prune
        (which happens on module load, not on completion).
        """
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)
        monkeypatch.setattr(
            "idea_board.executor._active",
            {"TK-427": ExecutionState(idea_id="TK-427")},
        )

        state = ExecutionState(idea_id="TK-427")
        state.log("working")
        state.log("done")

        # Execution "finishes" — state removed from _active
        monkeypatch.setattr("idea_board.executor._active", {})

        log_file = logs_dir / "TK-427.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "working" in content
        assert "done" in content

    def test_prunes_stale_done_sentinels(self, tmp_path, monkeypatch):
        """Stale .done files are pruned alongside .log files on module load."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-100.log").write_text("stale")
        (logs_dir / "TK-100.done").write_text("done")
        (logs_dir / "TK-200.done").write_text("failed")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)
        monkeypatch.setattr("idea_board.executor._active", {})

        _prune_stale_execution_logs()

        assert not (logs_dir / "TK-100.log").exists()
        assert not (logs_dir / "TK-100.done").exists()
        assert not (logs_dir / "TK-200.done").exists()


class TestWriteDoneSentinel:
    """_write_done_sentinel writes <idea_id>.done with the final state."""

    def test_writes_state_string(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _write_done_sentinel("TK-428", "done")

        sentinel = logs_dir / "TK-428.done"
        assert sentinel.exists()
        assert sentinel.read_text(encoding="utf-8").strip() == "done"

    def test_overwrites_existing_sentinel(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-428.done").write_text("failed")
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _write_done_sentinel("TK-428", "done")

        assert (logs_dir / "TK-428.done").read_text(encoding="utf-8").strip() == "done"

    def test_io_error_is_swallowed(self, tmp_path, monkeypatch):
        # Parent path collision — mkdir will fail.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setattr(
            "idea_board.executor.EXECUTION_LOGS_DIR", blocker / "execution_logs",
        )

        # Must not raise.
        _write_done_sentinel("TK-428", "done")


class TestMarkDoneWritesSentinel:
    """mark_done / mark_failed wrappers create .done sentinels via provider."""

    def test_mark_done_writes_sentinel(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        provider = MagicMock()
        monkeypatch.setattr(
            "idea_board.executor._get_board_provider", lambda: provider,
        )

        mark_done("TK-428", "final log")

        provider.mark_done.assert_called_once_with("TK-428", "final log")
        sentinel = logs_dir / "TK-428.done"
        assert sentinel.read_text(encoding="utf-8").strip() == "done"

    def test_mark_failed_writes_sentinel(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        provider = MagicMock()
        monkeypatch.setattr(
            "idea_board.executor._get_board_provider", lambda: provider,
        )

        mark_failed("TK-428", "boom")

        provider.mark_failed.assert_called_once_with("TK-428", "boom")
        sentinel = logs_dir / "TK-428.done"
        assert sentinel.read_text(encoding="utf-8").strip() == "failed"


class TestClearExecutionArtifacts:
    """_clear_execution_artifacts removes stale .log and .done files per idea."""

    def test_removes_both_artifacts(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-428.log").write_text("old log")
        (logs_dir / "TK-428.done").write_text("done")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _clear_execution_artifacts("TK-428")

        assert not (logs_dir / "TK-428.log").exists()
        assert not (logs_dir / "TK-428.done").exists()

    def test_missing_artifacts_is_noop(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        # No files present — must not raise.
        _clear_execution_artifacts("TK-428")

    def test_leaves_other_ideas_alone(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-428.log").write_text("target")
        (logs_dir / "TK-428.done").write_text("done")
        (logs_dir / "TK-999.log").write_text("other")
        (logs_dir / "TK-999.done").write_text("done")

        monkeypatch.setattr("idea_board.executor.EXECUTION_LOGS_DIR", logs_dir)

        _clear_execution_artifacts("TK-428")

        assert not (logs_dir / "TK-428.log").exists()
        assert not (logs_dir / "TK-428.done").exists()
        assert (logs_dir / "TK-999.log").exists()
        assert (logs_dir / "TK-999.done").exists()
