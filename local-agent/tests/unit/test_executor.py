"""Tests for idea_board.executor — pytest baseline, failure diffing, test targeting, and epic execution."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.executor import (
    MAX_FIX_RETRIES,
    PYTEST_TIMEOUT,
    ExecutionState,
    _build_epic_execution_context,
    _build_story_prompt,
    _find_related_tests,
    _format_injected_epic_context,
    _parse_pytest_failures,
    execute_epic,
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
            _load_codebase_summary=lambda: "",
            _load_git_history=lambda: "",
            _load_recent_errors=lambda: "",
            _load_similar_execution_logs=lambda i: "",
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
            _load_codebase_summary=lambda: "",
            _load_git_history=lambda: "",
            _load_recent_errors=lambda: "",
            _load_similar_execution_logs=lambda i: "",
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
            _load_codebase_summary=lambda: "",
            _load_git_history=lambda: "",
            _load_recent_errors=lambda: "",
            _load_similar_execution_logs=lambda i: "",
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
