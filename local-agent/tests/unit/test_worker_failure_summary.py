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


@pytest.mark.xfail(
    reason="Tier 3 matches ``ClassError: msg`` / ``ClassException: msg`` only; "
           "pytest ``E   `` assertion-prefix lines are not supported by the "
           "current regex and are left for a future enhancement.",
    strict=True,
)
def test_tier3_error_line_used_when_no_counter():
    """E-prefixed assertions (or ``FooError:``) are the tier-3 signal."""
    log = "running tests\nE   assert 1 == 2\nfinishing\n"
    summary = _extract_failure_summary(log)
    assert "assert 1 == 2" in summary


def test_tier4_tracemalloc_noise_is_filtered():
    """A log containing only the tracemalloc hint must NOT surface that hint
    as the failure reason. Tracemalloc lines don't end in ``Error``/
    ``Exception`` so tier 3 doesn't pick them up, and with no other tier
    matching the function returns an empty string — preserving the
    invariant that tracemalloc chatter is never the failure summary."""
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


def test_tier3_exception_class_prefix_matches():
    """Lines like ``ValueError: bad input`` should be picked up by tier 3."""
    log = "some setup\nValueError: bad input\nmore stuff\n"
    summary = _extract_failure_summary(log)
    assert "ValueError: bad input" in summary


def test_tier3_exception_line():
    """Tier 3 strips indentation and ignores raise/import source lines.

    Typical pytest traceback: the indented ``raise ValueError(...)`` source
    line must NOT be surfaced — the un-indented ``ValueError: unexpected
    shape`` line is the real signal.
    """
    log = (
        "Traceback (most recent call last):\n"
        "  File \"foo.py\", line 10, in bar\n"
        "    import numpy\n"
        "    raise ValueError(\"unexpected shape\")\n"
        "ValueError: unexpected shape\n"
    )
    summary = _extract_failure_summary(log)
    assert summary == "ValueError: unexpected shape"


def test_tier3_picks_last_exception():
    """When several exceptions appear, the final one wins.

    Chained tracebacks list earlier causes first; the last exception is
    the one that actually propagated out and is almost always the real
    failure.
    """
    log = (
        "ImportError: cannot import name 'foo'\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "KeyError: 'missing'\n"
        "\n"
        "The above exception was the direct cause of the following exception:\n"
        "\n"
        "RuntimeError: last one wins\n"
    )
    summary = _extract_failure_summary(log)
    assert summary == "RuntimeError: last one wins"


def test_no_match_returns_empty_string():
    """Logs that don't hit any tier return '' (not '(no log)')."""
    log = "some setup\nnothing interesting here\nmore stuff\n"
    assert _extract_failure_summary(log) == ""


def test_whitespace_only_input_returns_sentinel():
    """Logs that are whitespace-only count as empty — they yield ``(no log)``."""
    assert _extract_failure_summary("   \n\t\n") == "(no log)"


def test_truncation_appends_ellipsis():
    """When a match exceeds ``max_len``, the output ends with ``...``."""
    log = "FAILED tests/a.py::test_one - " + ("x" * 400)
    summary = _extract_failure_summary(log, max_len=80)
    assert len(summary) == 80
    assert summary.endswith("...")


def test_tier1_beats_tier2_when_both_present():
    """When a log has both FAILED lines AND a counter, tier 1 wins."""
    log = (
        "FAILED tests/a.py::test_one - AssertionError: boom\n"
        "=== 1 failed, 10 passed in 1.23s ===\n"
    )
    summary = _extract_failure_summary(log)
    assert summary.startswith("FAILED tests/a.py::test_one")
    assert "1 failed, 10 passed" not in summary


def test_tier2_beats_tier3_when_both_present():
    """When a log has a counter AND an exception line, tier 2 wins."""
    log = (
        "ValueError: something bad\n"
        "=== 2 failed, 98 passed in 3.21s ===\n"
    )
    summary = _extract_failure_summary(log)
    assert "2 failed, 98 passed" in summary
    assert "ValueError" not in summary


def test_tier3_strips_leading_whitespace():
    """Indented exception lines are surfaced without their indentation."""
    log = "    RuntimeError: indented failure\n"
    summary = _extract_failure_summary(log)
    assert summary == "RuntimeError: indented failure"
    assert not summary.startswith(" ")
