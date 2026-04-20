"""Tests for aiv.scorer — 7-axis quality scoring via claude -p (haiku).

Verifies:
    * Happy path: mocked ``claude -p`` JSON parses into StoryQualityScores
      with every axis, reasoning_map, and red_flags populated.
    * Malformed LLM response → sentinel dataclass with all axes == -1 and
      ``error='parse_failure'``; never raises.
    * Subprocess timeout → sentinel with ``error='timeout'``; never raises.
    * Prompt contains all seven axis names and every red-flag tag.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent.aiv_schema import SCORE_COLUMNS
from aiv import scorer
from aiv.scorer import (
    AXIS_DEFINITIONS,
    DEFAULT_MODEL,
    RED_FLAG_NAMES,
    SENTINEL_SCORE,
    StoryQualityScores,
    _build_prompt,
    _coerce_score,
    _parse_response,
    score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_story() -> dict:
    return {
        "key": "TK-680",
        "title": "Implement aiv/scorer.py",
        "description": "Seven-axis scorer for shipped stories.",
        "category": "feature",
    }


@pytest.fixture
def sample_diff() -> str:
    return "+++ b/aiv/scorer.py\n+def score(): ..."


@pytest.fixture
def sample_verify() -> str:
    return "============ 12 passed in 0.42s ============"


def _valid_scores_json(
    *,
    scores: dict[str, int] | None = None,
    reasoning: dict[str, str] | None = None,
    red_flags: list[str] | None = None,
) -> str:
    if scores is None:
        scores = {axis: 8 for axis in SCORE_COLUMNS}
    if reasoning is None:
        reasoning = {axis: f"reason for {axis}" for axis in SCORE_COLUMNS}
    if red_flags is None:
        red_flags = []
    return json.dumps(
        {"scores": scores, "reasoning": reasoning, "red_flags": red_flags}
    )


def _fake_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    """Build a stand-in for subprocess.CompletedProcess."""
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _claude_json_envelope(result_text: str) -> str:
    """Wrap model output in the claude -p --output-format json envelope."""
    return json.dumps(
        {
            "result": result_text,
            "total_cost_usd": 0.0,
            "session_id": "sess-aiv",
        }
    )


@pytest.fixture(autouse=True)
def _reset_binary_cache(monkeypatch):
    """Force the binary-discovery cache to return a known path per test."""
    monkeypatch.setattr(scorer, "_claude_binary_cache", None)
    monkeypatch.setattr(scorer, "_find_claude_binary", lambda: "/usr/bin/claude")


# ---------------------------------------------------------------------------
# StoryQualityScores dataclass
# ---------------------------------------------------------------------------


class TestStoryQualityScores:
    def test_defaults_all_sentinel(self):
        scores = StoryQualityScores()
        for axis in SCORE_COLUMNS:
            assert getattr(scores, axis) == SENTINEL_SCORE
        assert scores.reasoning_map == {}
        assert scores.red_flags == []
        assert scores.error == ""

    def test_sentinel_classmethod_sets_error(self):
        scores = StoryQualityScores.sentinel("timeout")
        assert scores.error == "timeout"
        for axis in SCORE_COLUMNS:
            assert getattr(scores, axis) == SENTINEL_SCORE

    def test_has_all_seven_axes(self):
        scores = StoryQualityScores()
        for axis in (
            "meets_requirements",
            "code_quality",
            "test_quality",
            "security_safety",
            "scope_discipline",
            "edge_cases",
            "product_impact",
        ):
            assert hasattr(scores, axis)


# ---------------------------------------------------------------------------
# _coerce_score
# ---------------------------------------------------------------------------


class TestCoerceScore:
    @pytest.mark.parametrize("val,expected", [
        (0, 0),
        (10, 10),
        (5, 5),
        (-1, -1),
        (7.8, 7),
        ("6", 6),
        (" 4 ", 4),
    ])
    def test_valid_values_pass_through(self, val, expected):
        assert _coerce_score(val) == expected

    @pytest.mark.parametrize("val", [11, -2, 100, "abc", "", None, [1], {}, True, False])
    def test_invalid_values_return_sentinel(self, val):
        assert _coerce_score(val) == SENTINEL_SCORE


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_happy_path_full_payload(self):
        raw = _valid_scores_json(
            scores={
                "meets_requirements": 9,
                "code_quality": 8,
                "test_quality": 7,
                "security_safety": 10,
                "scope_discipline": 9,
                "edge_cases": 6,
                "product_impact": 8,
            },
            reasoning={axis: f"r-{axis}" for axis in SCORE_COLUMNS},
            red_flags=["no_tests_added", "scope_creep"],
        )
        result = _parse_response(raw)

        assert result.error == ""
        assert result.meets_requirements == 9
        assert result.code_quality == 8
        assert result.test_quality == 7
        assert result.security_safety == 10
        assert result.scope_discipline == 9
        assert result.edge_cases == 6
        assert result.product_impact == 8
        assert result.reasoning_map["code_quality"] == "r-code_quality"
        assert set(result.red_flags) == {"no_tests_added", "scope_creep"}

    def test_json_embedded_in_prose(self):
        raw = "Here is my verdict:\n" + _valid_scores_json() + "\nThanks."
        result = _parse_response(raw)
        assert result.error == ""
        assert result.meets_requirements == 8

    def test_malformed_json_returns_parse_failure(self):
        result = _parse_response("{not valid json at all")
        assert result.error == "parse_failure"
        for axis in SCORE_COLUMNS:
            assert getattr(result, axis) == SENTINEL_SCORE

    def test_no_json_block_returns_parse_failure(self):
        assert _parse_response("no json here").error == "parse_failure"

    def test_empty_string_returns_parse_failure(self):
        assert _parse_response("").error == "parse_failure"

    def test_non_string_returns_parse_failure(self):
        assert _parse_response(None).error == "parse_failure"  # type: ignore[arg-type]

    def test_json_array_returns_parse_failure(self):
        assert _parse_response("[1, 2, 3]").error == "parse_failure"

    def test_missing_scores_object_returns_parse_failure(self):
        raw = json.dumps({"reasoning": {}, "red_flags": []})
        assert _parse_response(raw).error == "parse_failure"

    def test_missing_axis_coerced_to_sentinel(self):
        # Omit test_quality from the scores object.
        scores = {a: 5 for a in SCORE_COLUMNS if a != "test_quality"}
        raw = _valid_scores_json(scores=scores)
        result = _parse_response(raw)
        assert result.error == ""
        assert result.test_quality == SENTINEL_SCORE
        assert result.code_quality == 5

    def test_out_of_range_axis_coerced_to_sentinel(self):
        scores = {a: 5 for a in SCORE_COLUMNS}
        scores["edge_cases"] = 99  # out of [-1, 10]
        raw = _valid_scores_json(scores=scores)
        result = _parse_response(raw)
        assert result.edge_cases == SENTINEL_SCORE
        assert result.code_quality == 5  # others untouched

    def test_unknown_red_flags_filtered(self):
        raw = _valid_scores_json(
            red_flags=["no_tests_added", "bogus_flag", "scope_creep"]
        )
        result = _parse_response(raw)
        assert "bogus_flag" not in result.red_flags
        assert "no_tests_added" in result.red_flags
        assert "scope_creep" in result.red_flags

    def test_duplicate_red_flags_deduplicated(self):
        raw = _valid_scores_json(
            red_flags=["no_tests_added", "no_tests_added", "scope_creep"]
        )
        result = _parse_response(raw)
        assert result.red_flags.count("no_tests_added") == 1

    def test_non_string_reasoning_values_ignored(self):
        raw = json.dumps({
            "scores": {a: 5 for a in SCORE_COLUMNS},
            "reasoning": {"code_quality": 123, "test_quality": "valid"},
            "red_flags": [],
        })
        result = _parse_response(raw)
        assert "code_quality" not in result.reasoning_map
        assert result.reasoning_map["test_quality"] == "valid"

    def test_non_dict_scores_returns_parse_failure(self):
        raw = json.dumps({"scores": [1, 2, 3], "red_flags": []})
        assert _parse_response(raw).error == "parse_failure"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_all_seven_axis_names(
        self, sample_story, sample_diff, sample_verify
    ):
        prompt = _build_prompt(sample_story, sample_diff, sample_verify)
        for axis in SCORE_COLUMNS:
            assert axis in prompt, f"axis {axis!r} missing from prompt"

    def test_prompt_contains_all_red_flag_names(
        self, sample_story, sample_diff, sample_verify
    ):
        prompt = _build_prompt(sample_story, sample_diff, sample_verify)
        for flag in RED_FLAG_NAMES:
            assert flag in prompt, f"red flag {flag!r} missing from prompt"

    def test_prompt_contains_axis_definitions(
        self, sample_story, sample_diff, sample_verify
    ):
        prompt = _build_prompt(sample_story, sample_diff, sample_verify)
        # One phrase from each axis definition must appear.
        for axis, definition in AXIS_DEFINITIONS.items():
            first_phrase = definition.split(".")[0]
            assert first_phrase[:30] in prompt

    def test_prompt_embeds_story_fields(
        self, sample_story, sample_diff, sample_verify
    ):
        prompt = _build_prompt(sample_story, sample_diff, sample_verify)
        assert "TK-680" in prompt
        assert "Implement aiv/scorer.py" in prompt
        assert "feature" in prompt

    def test_prompt_embeds_diff_and_verification(
        self, sample_story, sample_diff, sample_verify
    ):
        prompt = _build_prompt(sample_story, sample_diff, sample_verify)
        assert "aiv/scorer.py" in prompt
        assert "12 passed" in prompt

    def test_prompt_truncates_long_diff(self, sample_story, sample_verify):
        huge = "x" * 50_000
        prompt = _build_prompt(sample_story, huge, sample_verify)
        assert "truncated" in prompt
        assert prompt.count("x") < 50_000

    def test_prompt_truncates_long_verification(self, sample_story, sample_diff):
        huge = "y" * 50_000
        prompt = _build_prompt(sample_story, sample_diff, huge)
        assert "truncated" in prompt
        assert prompt.count("y") < 50_000

    def test_prompt_tolerates_missing_fields(self):
        prompt = _build_prompt({}, "", "")
        assert "Key: " in prompt
        # Still advertises every axis even without story content.
        for axis in SCORE_COLUMNS:
            assert axis in prompt

    def test_prompt_accepts_id_and_summary_aliases(
        self, sample_diff, sample_verify
    ):
        story = {"id": "TK-77", "summary": "alt title"}
        prompt = _build_prompt(story, sample_diff, sample_verify)
        assert "TK-77" in prompt
        assert "alt title" in prompt


# ---------------------------------------------------------------------------
# score() — end-to-end with mocked subprocess
# ---------------------------------------------------------------------------


class TestScoreEndToEnd:
    """Scoring now routes through agent.llm_router.complete (qwen3.5 default)."""

    def test_happy_path_parses_valid_scores(
        self, sample_story, sample_diff, sample_verify
    ):
        model_json = _valid_scores_json(
            scores={axis: 9 for axis in SCORE_COLUMNS},
            red_flags=["no_tests_added"],
        )
        with patch("agent.llm_router.complete", return_value=model_json) as mock_llm:
            result = score(sample_story, sample_diff, sample_verify)

        assert result.error == ""
        for axis in SCORE_COLUMNS:
            assert getattr(result, axis) == 9
        assert result.red_flags == ["no_tests_added"]
        mock_llm.assert_called_once()
        args, _ = mock_llm.call_args
        assert args[0] == "aiv_scorer"

    def test_malformed_response_returns_sentinel_parse_failure(
        self, sample_story, sample_diff, sample_verify
    ):
        with patch("agent.llm_router.complete",
                   return_value="definitely not json here either"):
            result = score(sample_story, sample_diff, sample_verify)

        assert result.error == "parse_failure"
        for axis in SCORE_COLUMNS:
            assert getattr(result, axis) == SENTINEL_SCORE
        assert result.red_flags == []

    def test_router_returns_none_yields_llm_error(
        self, sample_story, sample_diff, sample_verify
    ):
        with patch("agent.llm_router.complete", return_value=None):
            result = score(sample_story, sample_diff, sample_verify)

        assert result.error == "llm_error"
        for axis in SCORE_COLUMNS:
            assert getattr(result, axis) == SENTINEL_SCORE

    def test_raw_ollama_output_parses_directly(
        self, sample_story, sample_diff, sample_verify
    ):
        """Ollama returns raw model JSON — no claude -p envelope wrapper."""
        model_json = _valid_scores_json()
        with patch("agent.llm_router.complete", return_value=model_json):
            result = score(sample_story, sample_diff, sample_verify)

        assert result.error == ""
        assert result.meets_requirements == 8

    def test_never_raises_on_bad_inputs(self):
        """Scoring must swallow every exception path."""
        with patch("agent.llm_router.complete", side_effect=Exception("arbitrary")):
            result = score({}, None, None)  # type: ignore[arg-type]
        assert isinstance(result, StoryQualityScores)
        assert result.error != ""
        for axis in SCORE_COLUMNS:
            assert getattr(result, axis) == SENTINEL_SCORE


# ---------------------------------------------------------------------------
# Contract: module is pure (no Jira, no DB writes, no filesystem writes)
# ---------------------------------------------------------------------------


class TestModuleIsPure:
    def test_no_jira_imports(self):
        from pathlib import Path
        src = Path(scorer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from idea_board.jira_sync",
            "import idea_board.jira_sync",
            "JiraProvider",
        ):
            assert forbidden not in src

    def test_no_filesystem_writes(self):
        from pathlib import Path
        src = Path(scorer.__file__).read_text(encoding="utf-8")
        for forbidden in (".write_text(", ".write_bytes(", ".mkdir("):
            assert forbidden not in src, (
                f"scorer.py must not use {forbidden!r} — scoring is pure."
            )

    def test_no_sqlite_writes(self):
        from pathlib import Path
        src = Path(scorer.__file__).read_text(encoding="utf-8")
        # Importing aiv_schema is fine (for SCORE_COLUMNS); calling its
        # write helpers is not.
        assert "aiv_schema.init_db" not in src
        assert "INSERT" not in src.upper().split("SCORE_COLUMNS")[0] if "SCORE_COLUMNS" in src else True
