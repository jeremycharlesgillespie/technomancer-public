"""Tests for aiv.classifier_llm — Haiku-backed tiebreak verifier picker.

Verifies:
    * Happy path: mocked ``claude -p`` JSON envelope parses into the
      returned verifier literal.
    * Parse failure paths (empty output, malformed JSON, unknown label)
      fall back to ``"tests-only"``.
    * Subprocess failures (timeout, non-zero exit, missing binary,
      arbitrary OSError) fall back to ``"tests-only"`` and never raise.
    * Prompt mentions all four verifiers, includes the story summary,
      and lists the diff paths.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aiv import classifier_llm
from aiv.classifier_llm import (
    ALLOWED_VERIFIERS,
    DEFAULT_MODEL,
    FALLBACK_VERIFIER,
    VERIFIER_DESCRIPTIONS,
    _build_prompt,
    _parse_response,
    pick_verifier,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _fake_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _claude_envelope(result_text: str) -> str:
    """Wrap a model response in the claude -p --output-format json envelope."""
    return json.dumps(
        {
            "result": result_text,
            "total_cost_usd": 0.0,
            "session_id": "sess-classifier-llm",
        }
    )


@pytest.fixture(autouse=True)
def _reset_binary_cache(monkeypatch):
    """Force the binary-discovery cache to return a known path per test."""
    monkeypatch.setattr(classifier_llm, "_claude_binary_cache", None)
    monkeypatch.setattr(classifier_llm, "_find_claude_binary", lambda: "/usr/bin/claude")


@pytest.fixture
def sample_paths() -> list[str]:
    return [
        "local-agent/idea_board/web.py",
        "src/api/endpoints.py",
    ]


@pytest.fixture
def sample_summary() -> str:
    return "Add retry logic to the /api/deploy endpoint and its HTMX fragment."


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_lists_all_four_verifiers(self, sample_paths, sample_summary):
        prompt = _build_prompt(sample_paths, sample_summary)
        for name in ALLOWED_VERIFIERS:
            assert name in prompt, f"verifier {name!r} missing from prompt"

    def test_prompt_includes_each_verifier_description(self, sample_paths, sample_summary):
        prompt = _build_prompt(sample_paths, sample_summary)
        for description in VERIFIER_DESCRIPTIONS.values():
            assert description in prompt

    def test_prompt_includes_story_summary(self, sample_paths, sample_summary):
        prompt = _build_prompt(sample_paths, sample_summary)
        assert sample_summary in prompt

    def test_prompt_includes_each_path(self, sample_paths, sample_summary):
        prompt = _build_prompt(sample_paths, sample_summary)
        for p in sample_paths:
            assert p in prompt

    def test_empty_summary_is_tolerated(self, sample_paths):
        prompt = _build_prompt(sample_paths, "")
        assert "(no summary provided)" in prompt

    def test_empty_paths_is_tolerated(self, sample_summary):
        prompt = _build_prompt([], sample_summary)
        assert sample_summary in prompt

    def test_truncates_excessive_paths(self, sample_summary):
        many_paths = [f"src/pkg/file_{i}.py" for i in range(classifier_llm.MAX_PATHS_IN_PROMPT + 5)]
        prompt = _build_prompt(many_paths, sample_summary)
        assert "paths omitted" in prompt
        assert many_paths[0] in prompt
        # Paths beyond the cap should NOT appear verbatim.
        assert many_paths[-1] not in prompt

    def test_truncates_long_summary(self, sample_paths):
        long_summary = "x" * (classifier_llm.MAX_SUMMARY_CHARS + 500)
        prompt = _build_prompt(sample_paths, long_summary)
        # Prompt shouldn't carry the full blob verbatim.
        assert long_summary not in prompt
        assert " ..." in prompt


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    @pytest.mark.parametrize("label", list(ALLOWED_VERIFIERS))
    def test_parses_each_allowed_verifier(self, label):
        envelope = _claude_envelope(json.dumps({"verifier": label}))
        assert _parse_response(envelope) == label

    def test_parses_bare_json_without_envelope(self):
        assert _parse_response(json.dumps({"verifier": "web-render"})) == "web-render"

    def test_unknown_label_falls_back_to_tests_only(self):
        envelope = _claude_envelope(json.dumps({"verifier": "screenshot-diff"}))
        assert _parse_response(envelope) == FALLBACK_VERIFIER

    def test_missing_verifier_key_falls_back(self):
        envelope = _claude_envelope(json.dumps({"other": "value"}))
        assert _parse_response(envelope) == FALLBACK_VERIFIER

    def test_empty_string_falls_back(self):
        assert _parse_response("") == FALLBACK_VERIFIER

    def test_whitespace_only_falls_back(self):
        assert _parse_response("   \n ") == FALLBACK_VERIFIER

    def test_malformed_json_falls_back(self):
        assert _parse_response("not json at all") == FALLBACK_VERIFIER

    def test_malformed_inner_json_falls_back(self):
        envelope = _claude_envelope("this is not json")
        assert _parse_response(envelope) == FALLBACK_VERIFIER

    def test_top_level_non_dict_falls_back(self):
        # JSON array wrapped in envelope — search still finds a brace pair
        # somewhere, but the parsed value isn't a dict.
        assert _parse_response("[1, 2, 3]") == FALLBACK_VERIFIER

    def test_non_string_verifier_falls_back(self):
        envelope = _claude_envelope(json.dumps({"verifier": 3}))
        assert _parse_response(envelope) == FALLBACK_VERIFIER

    def test_extracts_verifier_from_response_with_prose(self):
        # Model sometimes adds a preamble. Regex should still find the object.
        noisy = "Here is my pick:\n{\"verifier\": \"api-call\"}\nHope that helps!"
        envelope = _claude_envelope(noisy)
        assert _parse_response(envelope) == "api-call"


# ---------------------------------------------------------------------------
# pick_verifier — subprocess integration
# ---------------------------------------------------------------------------


class TestPickVerifier:
    """Verifier picking now routes through agent.llm_router.complete."""

    def test_happy_path_returns_web_render(self, sample_paths, sample_summary):
        raw = json.dumps({"verifier": "web-render"})
        with patch("agent.llm_router.complete", return_value=raw) as mock_llm:
            verdict = pick_verifier(sample_paths, sample_summary)
        assert verdict == "web-render"
        assert mock_llm.call_count == 1
        args, kwargs = mock_llm.call_args
        assert args[0] == "aiv_classifier"
        assert kwargs.get("timeout") == classifier_llm.DEFAULT_TIMEOUT

    def test_parse_failure_returns_tests_only(self, sample_paths, sample_summary):
        with patch("agent.llm_router.complete", return_value="garbled response"):
            assert pick_verifier(sample_paths, sample_summary) == FALLBACK_VERIFIER

    @pytest.mark.parametrize("label", list(ALLOWED_VERIFIERS))
    def test_each_verifier_roundtrips(self, sample_paths, sample_summary, label):
        raw = json.dumps({"verifier": label})
        with patch("agent.llm_router.complete", return_value=raw):
            assert pick_verifier(sample_paths, sample_summary) == label

    def test_unknown_verifier_label_falls_back(self, sample_paths, sample_summary):
        raw = json.dumps({"verifier": "load-test"})
        with patch("agent.llm_router.complete", return_value=raw):
            assert pick_verifier(sample_paths, sample_summary) == FALLBACK_VERIFIER

    def test_router_none_falls_back(self, sample_paths, sample_summary):
        """When the router returns None (all backends failed) → fallback."""
        with patch("agent.llm_router.complete", return_value=None):
            assert pick_verifier(sample_paths, sample_summary) == FALLBACK_VERIFIER

    def test_custom_timeout_is_forwarded(self, sample_paths, sample_summary):
        raw = json.dumps({"verifier": "db-query"})
        with patch("agent.llm_router.complete", return_value=raw) as mock_llm:
            verdict = pick_verifier(sample_paths, sample_summary, timeout=7)
        assert verdict == "db-query"
        assert mock_llm.call_args.kwargs.get("timeout") == 7

    def test_empty_paths_and_summary_still_return_valid_verifier(self):
        raw = json.dumps({"verifier": "tests-only"})
        with patch("agent.llm_router.complete", return_value=raw):
            assert pick_verifier([], "") == "tests-only"

    def test_never_raises_on_arbitrary_exception(self, sample_paths, sample_summary):
        with patch("agent.llm_router.complete", side_effect=Exception("boom")):
            assert pick_verifier(sample_paths, sample_summary) == FALLBACK_VERIFIER
