"""Tests for ``aiv.ab_compare`` — the LLM-judged A/B comparison.

Coverage:

- Happy path: well-formed JSON parses into :class:`ABComparison` with all
  fields populated.
- Failure modes return sentinels (``error`` populated, ``winner`` empty)
  rather than raising:
    - LLM returns ``None`` → ``llm_error``.
    - LLM raises an exception → ``llm_error``.
    - Malformed JSON / no JSON object in the response → ``parse_failure``.
    - Winner string not in :data:`COMPARISON_WINNERS` → ``parse_failure``.
- ``delta_axes`` is always all-seven-keys (zeros if missing).
- Empty diffs / empty story metadata don't crash the prompt builder.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aiv.ab_compare import ABComparison, compare


def _make_run(label: str = "qwen3-coder", status: str = "success") -> dict:
    return {
        "model": f"{label}:30b",
        "status": status,
        "branch_name": "br",
        "commit_sha": "abc1234",
        "diff": "diff --git a/x.py b/x.py\n+pass\n",
        "failure_log": "",
        "meets_requirements": 9,
        "code_quality": 8,
        "test_quality": 7,
        "security_safety": 10,
        "scope_discipline": 9,
        "edge_cases": 6,
        "product_impact": 8,
    }


def _story() -> dict:
    return {
        "key": "TK-1",
        "title": "Add retry logic",
        "description": "Add exponential backoff to webhook delivery.",
    }


def test_compare_happy_path_parses_winner_and_delta() -> None:
    payload = json.dumps({
        "winner": "model_a",
        "reasoning": "A's tests are stronger.",
        "delta_axes": {
            "meets_requirements": 0,
            "code_quality": 1,
            "test_quality": 2,
            "security_safety": 0,
            "scope_discipline": 0,
            "edge_cases": -1,
            "product_impact": 0,
        },
    })
    with patch("agent.llm_router.complete", return_value=payload):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert isinstance(result, ABComparison)
    assert result.winner == "model_a"
    assert result.error == ""
    assert "tests are stronger" in result.reasoning
    assert result.delta_axes["test_quality"] == 2
    assert result.delta_axes["edge_cases"] == -1
    # All seven axes present.
    assert len(result.delta_axes) == 7


def test_compare_handles_llm_returning_none() -> None:
    with patch("agent.llm_router.complete", return_value=None):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.error == "llm_error"
    assert result.winner == ""


def test_compare_handles_llm_raising() -> None:
    with patch("agent.llm_router.complete", side_effect=RuntimeError("timeout")):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.error == "llm_error"
    assert result.winner == ""


def test_compare_handles_malformed_json() -> None:
    with patch("agent.llm_router.complete", return_value="not json at all"):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.error == "parse_failure"


def test_compare_handles_invalid_winner() -> None:
    payload = json.dumps({"winner": "bogus", "reasoning": "x", "delta_axes": {}})
    with patch("agent.llm_router.complete", return_value=payload):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.error == "parse_failure"
    assert result.winner == ""


def test_compare_fills_missing_delta_axes_with_zero() -> None:
    payload = json.dumps({
        "winner": "tie",
        "reasoning": "equivalent",
        "delta_axes": {"code_quality": 0},  # missing the other six
    })
    with patch("agent.llm_router.complete", return_value=payload):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.winner == "tie"
    assert len(result.delta_axes) == 7
    assert result.delta_axes["test_quality"] == 0


def test_compare_clamps_oversize_reasoning() -> None:
    long_reasoning = "x" * 5000
    payload = json.dumps({
        "winner": "model_a",
        "reasoning": long_reasoning,
        "delta_axes": {},
    })
    with patch("agent.llm_router.complete", return_value=payload):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.winner == "model_a"
    assert len(result.reasoning) <= 1500


def test_compare_handles_both_empty_diffs() -> None:
    """Both runs failed before producing any diff — should still call LLM."""
    run_a = _make_run("a", status="failed")
    run_a["diff"] = ""
    run_a["failure_log"] = "tests failed"
    run_b = _make_run("b", status="failed")
    run_b["diff"] = ""
    run_b["failure_log"] = "model timeout"
    payload = json.dumps({
        "winner": "both_failed",
        "reasoning": "Neither produced a runnable diff.",
        "delta_axes": {},
    })
    with patch("agent.llm_router.complete", return_value=payload) as mc:
        result = compare(_story(), run_a, run_b)
    assert result.winner == "both_failed"
    assert mc.called


def test_compare_winner_winners_set_includes_all_documented() -> None:
    """All four documented winner strings must round-trip cleanly."""
    for winner in ("model_a", "model_b", "tie", "both_failed"):
        payload = json.dumps({"winner": winner, "reasoning": "", "delta_axes": {}})
        with patch("agent.llm_router.complete", return_value=payload):
            result = compare(_story(), _make_run("a"), _make_run("b"))
        assert result.winner == winner, f"failed for {winner}"


def test_compare_coerces_string_delta_to_int() -> None:
    payload = json.dumps({
        "winner": "model_a",
        "reasoning": "x",
        "delta_axes": {
            "meets_requirements": "3",
            "code_quality": "not-a-number",
            "test_quality": 100,  # out of range — should clamp to 0
        },
    })
    with patch("agent.llm_router.complete", return_value=payload):
        result = compare(_story(), _make_run("a"), _make_run("b"))
    assert result.delta_axes["meets_requirements"] == 3
    assert result.delta_axes["code_quality"] == 0
    assert result.delta_axes["test_quality"] == 0
