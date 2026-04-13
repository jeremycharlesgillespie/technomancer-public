"""Tests for idea_board.executor — pytest baseline and failure diffing."""

from unittest.mock import MagicMock, patch
from dataclasses import fields

import pytest

from idea_board.executor import (
    BASELINE_TIMEOUT,
    ExecutionState,
    _parse_pytest_failures,
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
# BASELINE_TIMEOUT constant
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify baseline timeout constant exists and is reasonable."""

    def test_baseline_timeout_exists(self):
        assert BASELINE_TIMEOUT == 120

    def test_baseline_timeout_is_int(self):
        assert isinstance(BASELINE_TIMEOUT, int)
