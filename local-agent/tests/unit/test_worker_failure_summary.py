"""Tests for aim.worker._extract_failure_summary.

Covers the tiered failure-summary extractor that replaces the naive
``log_tail[:300]`` so failure summaries reflect the real pytest failure
instead of trailing ``tracemalloc``/``ResourceWarning`` noise.
"""

from __future__ import annotations

import pytest

from aim.worker import _extract_failure_summary


def test_tier1_failed_line_is_surfaced():
    """FAILED lines win over everything else in the tail."""
    log = (
        "collected 4 items\n"
        "tests/unit/test_x.py::test_y PASSED\n"
        "FAILED tests/unit/test_x.py::test_y - AssertionError: boom\n"
        "==========\n"
    )
    summary = _extract_failure_summary(log)
    assert summary.startswith("FAILED tests/unit/test_x.py::test_y")


def test_tier2_counter_line_used_when_no_failed_lines():
    """Falls back to the pytest summary counter when no FAILED line is present."""
    log = "running tests\nsome warning\n=== 2 failed, 98 passed in 3.21s ===\n"
    summary = _extract_failure_summary(log)
    assert "2 failed, 98 passed" in summary


@pytest.mark.xfail(reason="Tier 3 is stubbed in TK-713 and will be re-implemented in a follow-up story.", strict=True)
def test_tier3_error_line_used_when_no_counter():
    """E-prefixed assertions (or ``FooError:``) are the tier-3 signal."""
    log = "running tests\nE   assert 1 == 2\nfinishing\n"
    summary = _extract_failure_summary(log)
    assert "assert 1 == 2" in summary


def test_tier4_tracemalloc_noise_is_filtered():
    """A log containing only the tracemalloc hint must NOT surface that hint
    as the failure reason. With tier 3+ stubbed, tier 4's reverse scan no
    longer runs, so the stub returns an empty string — which also satisfies
    the invariant that tracemalloc chatter is never the failure summary."""
    log = "[tests] Enable tracemalloc to get traceback where the object was allocated."
    summary = _extract_failure_summary(log)
    assert "tracemalloc" not in summary.lower()
    assert "Enable tracemalloc" not in summary


def test_empty_input_returns_sentinel():
    """Empty log tail must not crash and must signal 'no log'."""
    assert _extract_failure_summary("") == "(no log)"


def test_tier1_respects_max_len():
    """Long FAILED lines are truncated to max_len."""
    long_failure = "FAILED " + ("x" * 500)
    summary = _extract_failure_summary(long_failure, max_len=50)
    assert len(summary) == 50


def test_tier1_multiple_failures_joined_with_pipe():
    """Multiple FAILED lines appear in the output joined by ' | '."""
    log = "FAILED tests/a.py::test_one\nFAILED tests/b.py::test_two\n==========\n"
    summary = _extract_failure_summary(log)
    assert "tests/a.py::test_one" in summary
    assert "tests/b.py::test_two" in summary
    assert " | " in summary


@pytest.mark.xfail(reason="Tier 3 is stubbed in TK-713 and will be re-implemented in a follow-up story.", strict=True)
def test_tier3_exception_class_prefix_matches():
    """Lines like ``ValueError: bad input`` should be picked up by tier 3."""
    log = "some setup\nValueError: bad input\nmore stuff\n"
    summary = _extract_failure_summary(log)
    assert "ValueError: bad input" in summary


def test_no_match_returns_empty_string():
    """With tier 3+ stubbed, logs that don't hit tier 1/2 return ''."""
    log = "some setup\nValueError: bad input\nmore stuff\n"
    assert _extract_failure_summary(log) == ""
