"""Tests for aimm.observer — paper-worthiness scoring for shipped stories.

Verifies:
    * Happy path: mocked ``claude -p`` returning valid JSON yields a
      populated Observation.
    * Malformed LLM responses → ``Observation(finding_worthy=False,
      reason='parse_failure')``, never raise.
    * Missing rubric, subprocess failure, unsuccessful result dict →
      each produces a distinct ``reason`` code.
    * The function makes **no** Jira mutations in any code path
      (imports and calls are checked — see TestNoJiraMutations).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimm import observer
from aimm.observer import (
    Observation,
    _build_prompt,
    _load_rubric,
    _parse_response,
    score_shipped_story,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_RUBRIC = (
    "# Paper-Worthiness Rubric\n\n"
    "Criteria: concrete, reproducible, narrative-forward, observable metric."
)


@pytest.fixture
def rubric_file(tmp_path: Path) -> Path:
    path = tmp_path / "paper_rubric.md"
    path.write_text(SAMPLE_RUBRIC, encoding="utf-8")
    return path


@pytest.fixture
def sample_story() -> dict:
    return {
        "key": "TK-660",
        "title": "Add observer scoring",
        "description": "Scores shipped stories for paper-worthiness.",
        "category": "feature",
        "run": {"cost_usd": 0.12, "phase_timings": {"tests": 7.4}},
    }


@pytest.fixture
def sample_commit() -> dict:
    return {
        "sha": "abc1234",
        "message": "[TK-660] observer",
        "diff": "+++ b/aimm/observer.py\n+def score_shipped_story(): ...",
    }


@pytest.fixture
def sample_test_output() -> str:
    return "============ 5 passed in 0.42s ============"


def _claude_ok(result_text: str) -> dict:
    return {
        "success": True,
        "result": result_text,
        "cost_usd": 0.0,
        "session_id": "sess-1",
        "duration": 0.1,
        "error": None,
    }


def _claude_err(error: str = "boom") -> dict:
    return {
        "success": False,
        "result": "",
        "cost_usd": 0.0,
        "session_id": "",
        "duration": 0.1,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------


class TestObservation:
    def test_defaults(self):
        obs = Observation()
        assert obs.finding_worthy is False
        assert obs.headline == ""
        assert obs.why_it_matters == ""
        assert obs.evidence_pointer == ""
        assert obs.theme == ""
        assert obs.reason == ""

    def test_to_dict_round_trip(self):
        obs = Observation(
            finding_worthy=True,
            headline="h",
            why_it_matters="w",
            evidence_pointer="e",
            theme="t",
            reason="ok",
        )
        d = obs.to_dict()
        assert d == {
            "finding_worthy": True,
            "headline": "h",
            "why_it_matters": "w",
            "evidence_pointer": "e",
            "theme": "t",
            "reason": "ok",
        }


# ---------------------------------------------------------------------------
# _load_rubric
# ---------------------------------------------------------------------------


class TestLoadRubric:
    def test_reads_existing_file(self, rubric_file):
        assert _load_rubric(rubric_file) == SAMPLE_RUBRIC

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist.md"
        assert _load_rubric(missing) == ""

    def test_default_path_module_constant(self):
        assert observer.RUBRIC_PATH.name == "paper_rubric.md"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_worthy_json(self):
        raw = (
            '{"finding_worthy": true, "headline": "H", "why_it_matters": "W",'
            ' "evidence_pointer": "E", "theme": "failure-mode discoveries"}'
        )
        obs = _parse_response(raw)
        assert obs.finding_worthy is True
        assert obs.headline == "H"
        assert obs.why_it_matters == "W"
        assert obs.evidence_pointer == "E"
        assert obs.theme == "failure-mode discoveries"
        assert obs.reason == "ok"

    def test_valid_not_worthy_json(self):
        raw = '{"finding_worthy": false, "headline": "", "theme": "other"}'
        obs = _parse_response(raw)
        assert obs.finding_worthy is False
        assert obs.theme == "other"
        assert obs.reason == "not_worthy"

    def test_json_embedded_in_prose(self):
        raw = (
            "Here is my verdict:\n"
            '{"finding_worthy": true, "headline": "h"}\n'
            "Thanks."
        )
        obs = _parse_response(raw)
        assert obs.finding_worthy is True
        assert obs.headline == "h"

    def test_malformed_json_returns_parse_failure(self):
        obs = _parse_response("{not valid json at all")
        assert obs.finding_worthy is False
        assert obs.reason == "parse_failure"

    def test_no_json_block_returns_parse_failure(self):
        obs = _parse_response("there is no json here")
        assert obs.finding_worthy is False
        assert obs.reason == "parse_failure"

    def test_empty_string_returns_parse_failure(self):
        obs = _parse_response("")
        assert obs.reason == "parse_failure"

    def test_non_string_returns_parse_failure(self):
        obs = _parse_response(None)  # type: ignore[arg-type]
        assert obs.reason == "parse_failure"

    def test_json_array_returns_parse_failure(self):
        obs = _parse_response("[1, 2, 3]")
        assert obs.reason == "parse_failure"

    def test_missing_optional_fields_default_to_empty(self):
        obs = _parse_response('{"finding_worthy": true}')
        assert obs.finding_worthy is True
        assert obs.headline == ""
        assert obs.why_it_matters == ""
        assert obs.evidence_pointer == ""
        assert obs.theme == ""

    def test_null_fields_coerced_to_empty_string(self):
        raw = (
            '{"finding_worthy": true, "headline": null, "why_it_matters": null,'
            ' "evidence_pointer": null, "theme": null}'
        )
        obs = _parse_response(raw)
        assert obs.headline == ""
        assert obs.theme == ""


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_includes_rubric_and_story_fields(
        self, sample_story, sample_commit, sample_test_output
    ):
        prompt = _build_prompt(
            SAMPLE_RUBRIC, sample_story, sample_commit, sample_test_output
        )
        assert SAMPLE_RUBRIC in prompt
        assert "TK-660" in prompt
        assert "Add observer scoring" in prompt
        assert "abc1234" in prompt
        assert "5 passed" in prompt

    def test_prompt_tolerates_missing_fields(self):
        prompt = _build_prompt(SAMPLE_RUBRIC, {}, {}, "")
        assert "Key: " in prompt
        assert "SHA: " in prompt

    def test_prompt_truncates_long_diff(self):
        huge = "x" * 20_000
        prompt = _build_prompt(
            SAMPLE_RUBRIC, {"key": "TK-1"}, {"diff": huge}, ""
        )
        assert "truncated" in prompt
        # Ensure we didn't embed the full 20k
        assert prompt.count("x") < 20_000

    def test_prompt_accepts_id_and_hash_aliases(self):
        story = {"id": "TK-77", "summary": "alt title"}
        commit = {"hash": "deadbee"}
        prompt = _build_prompt(SAMPLE_RUBRIC, story, commit, "")
        assert "TK-77" in prompt
        assert "alt title" in prompt
        assert "deadbee" in prompt


# ---------------------------------------------------------------------------
# score_shipped_story — end-to-end with mocked claude -p
# ---------------------------------------------------------------------------


class TestScoreShippedStory:
    def test_happy_path_worthy(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        fake = _claude_ok(
            '{"finding_worthy": true, "headline": "stall race",'
            ' "why_it_matters": "exposed a latent race", "evidence_pointer": "abc1234",'
            ' "theme": "failure-mode discoveries"}'
        )
        with patch("aimm.observer.run_claude_prompt", return_value=fake) as mock_run:
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )

        assert obs.finding_worthy is True
        assert obs.headline == "stall race"
        assert obs.theme == "failure-mode discoveries"
        assert obs.reason == "ok"
        mock_run.assert_called_once()
        # claude -p was invoked with a prompt that mentions the story key
        (prompt_arg,) = mock_run.call_args.args
        assert "TK-660" in prompt_arg

    def test_malformed_response_returns_parse_failure(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        fake = _claude_ok("definitely not json")
        with patch("aimm.observer.run_claude_prompt", return_value=fake):
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )

        assert obs.finding_worthy is False
        assert obs.reason == "parse_failure"
        assert obs.headline == ""

    def test_claude_failure_returns_llm_error(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        with patch("aimm.observer.run_claude_prompt", return_value=_claude_err()):
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )
        assert obs.finding_worthy is False
        assert obs.reason == "llm_error"

    def test_claude_raises_returns_llm_error(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        with patch(
            "aimm.observer.run_claude_prompt",
            side_effect=RuntimeError("subprocess exploded"),
        ):
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )
        assert obs.finding_worthy is False
        assert obs.reason == "llm_error"

    def test_missing_rubric_returns_rubric_unavailable(
        self, tmp_path, sample_story, sample_commit, sample_test_output
    ):
        missing = tmp_path / "nope.md"
        with patch("aimm.observer.run_claude_prompt") as mock_run:
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=missing,
            )
        assert obs.reason == "rubric_unavailable"
        mock_run.assert_not_called()

    def test_non_dict_result_returns_llm_error(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        with patch("aimm.observer.run_claude_prompt", return_value="oops"):
            obs = score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )
        assert obs.reason == "llm_error"

    def test_function_never_raises_on_any_input(self, rubric_file):
        """Scoring must swallow every exception path."""
        with patch(
            "aimm.observer.run_claude_prompt",
            side_effect=Exception("arbitrary"),
        ):
            obs = score_shipped_story({}, {}, "", rubric_path=rubric_file)
        assert isinstance(obs, Observation)
        assert obs.finding_worthy is False


# ---------------------------------------------------------------------------
# Contract: no Jira mutations anywhere in the module
# ---------------------------------------------------------------------------


class TestNoJiraMutations:
    """Observer is read-only. It must not import or call anything Jira-facing."""

    def test_module_does_not_import_jira_sync(self):
        src = Path(observer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from idea_board.jira_sync",
            "import idea_board.jira_sync",
            "from idea_board import jira_sync",
            "jira_sync.",
            "JiraProvider",
        ):
            assert forbidden not in src, (
                f"observer.py must not reference {forbidden!r} — it is a "
                "read-only scorer, no Jira mutations allowed."
            )

    def test_module_does_not_write_to_filesystem(self):
        src = Path(observer.__file__).read_text(encoding="utf-8")
        # Only read access is allowed; writes are forbidden in a pure scorer.
        for forbidden in (".write_text(", ".write_bytes(", "open(", ".mkdir("):
            assert forbidden not in src, (
                f"observer.py must not use {forbidden!r} — scoring is pure."
            )

    def test_score_does_not_trigger_imports_of_jira(
        self, rubric_file, sample_story, sample_commit, sample_test_output
    ):
        before = {m for m in sys.modules if "jira" in m.lower()}
        with patch(
            "aimm.observer.run_claude_prompt",
            return_value=_claude_ok('{"finding_worthy": false}'),
        ):
            score_shipped_story(
                sample_story,
                sample_commit,
                sample_test_output,
                rubric_path=rubric_file,
            )
        after = {m for m in sys.modules if "jira" in m.lower()}
        assert after == before, "scoring must not import any jira module"
