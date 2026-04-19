"""Tests for ``idea_board.dedup_llm`` — Haiku-backed near-exact dedup judge.

Covers the seven acceptance criteria from TK-744:

    1. Valid JSON with ``verdict="SAME"`` → ``(True, reason)``
    2. Valid JSON with ``verdict="DIFFERENT"`` → ``(False, reason)``
    3. ``subprocess.TimeoutExpired`` → ``(False, "llm_timeout")``
    4. Non-zero exit code → ``(False, "llm_exit_<code>")``
    5. No JSON in output → ``(False, "no_json")``
    6. Malformed JSON → ``(False, "parse_failure")``
    7. Missing claude binary → ``(False, "no_binary")`` and subprocess never called

Every test mocks ``subprocess.run`` so the real ``claude -p`` binary is
never invoked. The ``_reset_binary_cache`` autouse fixture forces
binary discovery to a known path so tests are deterministic regardless
of what's installed on the host.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from idea_board import dedup_llm
from idea_board.dedup_llm import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    MAX_DESC_CHARS,
    MAX_TITLE_CHARS,
    _build_prompt,
    _parse_verdict,
    is_near_exact_duplicate,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _fake_completed_process(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> MagicMock:
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _verdict_json(verdict: str, reason: str = "matches scope and files") -> str:
    return json.dumps({"verdict": verdict, "reason": reason})


@pytest.fixture(autouse=True)
def _reset_binary_cache(monkeypatch):
    """Force the binary-discovery cache to return a known path per test.

    Without this, a host with claude installed leaks a real binary path
    into tests, and a host without it would fail every test on the
    no_binary fast path.
    """
    monkeypatch.setattr(dedup_llm, "_claude_binary_cache", None)
    monkeypatch.setattr(dedup_llm, "_find_claude_binary", lambda: "/usr/bin/claude")


@pytest.fixture
def story_a() -> dict:
    return {
        "title": "Cache Ollama responses to improve performance",
        "desc": "WHAT: Cache responses keyed by prompt hash. WHY: Avoid recompute.",
    }


@pytest.fixture
def story_b() -> dict:
    return {
        "title": "Add response cache for Ollama performance gains",
        "desc": "WHAT: Add a hash-keyed Ollama response cache. WHY: GPU savings.",
    }


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_both_titles(self, story_a, story_b):
        prompt = _build_prompt(
            story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
        )
        assert story_a["title"] in prompt
        assert story_b["title"] in prompt

    def test_includes_both_descriptions(self, story_a, story_b):
        prompt = _build_prompt(
            story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
        )
        assert story_a["desc"] in prompt
        assert story_b["desc"] in prompt

    def test_truncates_long_titles(self, story_b):
        long_title = "x" * (MAX_TITLE_CHARS + 200)
        prompt = _build_prompt(long_title, "desc", story_b["title"], story_b["desc"])
        # Truncated form (first MAX_TITLE_CHARS x's) appears; the full
        # blob does not.
        assert "x" * MAX_TITLE_CHARS in prompt
        assert long_title not in prompt

    def test_truncates_long_descriptions(self, story_b):
        long_desc = "y" * (MAX_DESC_CHARS + 500)
        prompt = _build_prompt("title", long_desc, story_b["title"], story_b["desc"])
        assert "y" * MAX_DESC_CHARS in prompt
        assert long_desc not in prompt

    def test_empty_inputs_are_tolerated(self):
        prompt = _build_prompt("", "", "", "")
        # Just confirm no exception and the rule block survives.
        assert "verdict" in prompt
        assert "SAME" in prompt
        assert "DIFFERENT" in prompt

    def test_none_inputs_are_coerced_to_empty(self):
        # The function uses `(value or "")` so None should not crash.
        prompt = _build_prompt(None, None, None, None)  # type: ignore[arg-type]
        assert "verdict" in prompt


# ---------------------------------------------------------------------------
# _parse_verdict — parse-only paths (no subprocess)
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_same_verdict_returns_true_with_reason(self):
        is_dup, reason = _parse_verdict(_verdict_json("SAME", "identical scope"))
        assert is_dup is True
        assert reason == "identical scope"

    def test_different_verdict_returns_false_with_reason(self):
        is_dup, reason = _parse_verdict(_verdict_json("DIFFERENT", "scope diverges"))
        assert is_dup is False
        assert reason == "scope diverges"

    def test_lowercase_same_is_normalized(self):
        is_dup, _ = _parse_verdict(_verdict_json("same"))
        assert is_dup is True

    def test_mixed_case_same_is_normalized(self):
        is_dup, _ = _parse_verdict(_verdict_json("Same"))
        assert is_dup is True

    def test_unknown_verdict_treated_as_different(self):
        is_dup, _ = _parse_verdict(_verdict_json("MAYBE"))
        assert is_dup is False

    def test_empty_string_returns_no_json(self):
        is_dup, reason = _parse_verdict("")
        assert is_dup is False
        assert reason == "no_json"

    def test_whitespace_only_returns_no_json(self):
        is_dup, reason = _parse_verdict("   \n\t  ")
        assert is_dup is False
        assert reason == "no_json"

    def test_no_json_object_returns_no_json(self):
        is_dup, reason = _parse_verdict("just some prose with no braces")
        assert is_dup is False
        assert reason == "no_json"

    def test_malformed_json_returns_parse_failure(self):
        is_dup, reason = _parse_verdict('{"verdict": "SAME", "reason": }')
        assert is_dup is False
        assert reason == "parse_failure"

    def test_extracts_json_from_prose_preamble(self):
        noisy = 'Here is my verdict:\n{"verdict": "SAME", "reason": "match"}\nthanks'
        is_dup, reason = _parse_verdict(noisy)
        assert is_dup is True
        assert reason == "match"

    def test_missing_reason_field_yields_empty_reason(self):
        is_dup, reason = _parse_verdict(json.dumps({"verdict": "SAME"}))
        assert is_dup is True
        assert reason == ""

    def test_long_reason_is_truncated(self):
        is_dup, reason = _parse_verdict(_verdict_json("DIFFERENT", "z" * 500))
        assert is_dup is False
        assert len(reason) == 300


# ---------------------------------------------------------------------------
# is_near_exact_duplicate — subprocess integration
# ---------------------------------------------------------------------------


class TestIsNearExactDuplicate:
    """Acceptance criteria 1-7."""

    def test_same_verdict_returns_true_with_reason(self, story_a, story_b):
        """AC #1 + #6: Valid JSON with verdict=SAME → (True, reason)."""
        with patch(
            "agent.llm_router.complete",
            return_value=_verdict_json("SAME", "near-identical"),
        ) as mock_run:
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is True
        assert reason == "near-identical"
        assert mock_run.call_count == 1

    def test_different_verdict_returns_false_with_reason(self, story_a, story_b):
        """AC #7: Valid JSON with verdict=DIFFERENT → (False, reason)."""
        fake_cp = _fake_completed_process(
            _verdict_json("DIFFERENT", "scopes diverge")
        )
        with patch("agent.llm_router.complete", return_value=fake_cp.stdout):
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is False
        assert reason == "scopes diverge"

    def test_router_failure_returns_llm_unavailable(self, story_a, story_b):
        """All router-side error paths (timeout, non-zero exit, missing binary,
        network failure) collapse to a single code now that the router owns
        subprocess error handling. The router returns None; this function
        falls open with ``(False, "llm_unavailable")``.
        """
        with patch("agent.llm_router.complete", return_value=None):
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is False
        assert reason == "llm_unavailable"

    def test_no_json_in_output_returns_no_json(self, story_a, story_b):
        """AC #4: No JSON in output → (False, 'no_json')."""
        fake_cp = _fake_completed_process("just prose, no braces here at all")
        with patch("agent.llm_router.complete", return_value=fake_cp.stdout):
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is False
        assert reason == "no_json"

    def test_empty_stdout_returns_no_json(self, story_a, story_b):
        """AC #4: empty stdout is the same fall-open path as missing JSON."""
        fake_cp = _fake_completed_process("")
        with patch("agent.llm_router.complete", return_value=fake_cp.stdout):
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is False
        assert reason == "no_json"

    def test_malformed_json_returns_parse_failure(self, story_a, story_b):
        """AC #5: Malformed JSON → (False, 'parse_failure')."""
        fake_cp = _fake_completed_process('{"verdict": "SAME", "reason":}')
        with patch("agent.llm_router.complete", return_value=fake_cp.stdout):
            is_dup, reason = is_near_exact_duplicate(
                story_a["title"], story_a["desc"], story_b["title"], story_b["desc"]
            )

        assert is_dup is False
        assert reason == "parse_failure"


# Subprocess-invocation tests retired: the command shape is now owned by
# ``agent.llm_router._claude_chat`` (and by the ollama primary path in
# the router). See ``tests/unit/test_llm_router.py`` for the router's
# own shape + error-path coverage. This module only tests the verdict-
# parsing and the pass-through-on-failure contract.
