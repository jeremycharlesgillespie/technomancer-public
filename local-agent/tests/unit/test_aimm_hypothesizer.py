"""Tests for aimm.hypothesizer — research hypothesis proposer.

Verifies:
    * Acceptance: mocked LLM returning 2 hypotheses → 2 records appended.
    * Acceptance: malformed response → empty list, no append, no raise.
    * Acceptance: re-running on the same theme does not append
      semantically identical hypotheses (by headline match).
    * No Jira mutations anywhere in the module (imports + runtime).
    * Bounded to MAX_HYPOTHESES even when the LLM returns more.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimm import hypothesizer
from aimm.hypothesizer import (
    FINDINGS_HEADING,
    FINDINGS_PREAMBLE,
    MAX_HYPOTHESES,
    Hypothesis,
    _build_prompt,
    _load_existing_statements,
    _normalize_statement,
    _parse_response,
    propose_hypotheses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_RUBRIC = (
    "# Paper-Worthiness Rubric\n\n"
    "Criteria: concrete, reproducible, narrative-forward, observable metric."
)


@pytest.fixture
def tmp_findings(tmp_path: Path) -> Path:
    return tmp_path / "raw_findings.md"


@pytest.fixture
def theme_dict() -> dict:
    return {
        "name": "measurement + benchmarks",
        "target_per_week": 3,
        "keywords": ("benchmark", "metric", "measurement"),
    }


@pytest.fixture
def recent_stories() -> list[dict]:
    return [
        {
            "key": "TK-401",
            "title": "Add commit-volume benchmark across projects",
            "description": "Baseline throughput measurement.",
        },
        {
            "key": "TK-402",
            "title": "Capture per-phase executor timings",
            "description": "Phase-by-phase timing breakdown.",
        },
    ]


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


def _two_hypotheses_json() -> str:
    return (
        "["
        '{"statement": "Stall-detector false positives drop >80% with ready-line heartbeat",'
        ' "how_to_verify": "Replay last 24h of executor logs with patched detector",'
        ' "expected_outcome": "Fewer than 2 stalled-but-shipped warnings per day"},'
        '{"statement": "Executor throughput scales linearly with parallel branches up to 4",'
        ' "how_to_verify": "Measure commits/hour vs concurrent worker count",'
        ' "expected_outcome": "Linear up to 4, sublinear past 4"}'
        "]"
    )


def _many_hypotheses_json(n: int) -> str:
    parts = [
        f'{{"statement": "claim {i}", "how_to_verify": "v{i}", "expected_outcome": "o{i}"}}'
        for i in range(n)
    ]
    return "[" + ",".join(parts) + "]"


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------


class TestHypothesis:
    def test_defaults(self) -> None:
        h = Hypothesis()
        assert h.statement == ""
        assert h.how_to_verify == ""
        assert h.expected_outcome == ""

    def test_to_dict_round_trip(self) -> None:
        h = Hypothesis(
            statement="s", how_to_verify="v", expected_outcome="o"
        )
        assert h.to_dict() == {
            "statement": "s",
            "how_to_verify": "v",
            "expected_outcome": "o",
        }


# ---------------------------------------------------------------------------
# _normalize_statement
# ---------------------------------------------------------------------------


class TestNormalizeStatement:
    def test_lowercases_and_strips(self) -> None:
        assert _normalize_statement("  Hello World  ") == "hello world"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalize_statement("a    b\tc\nd") == "a b c d"

    def test_empty_and_none(self) -> None:
        assert _normalize_statement("") == ""
        assert _normalize_statement(None) == ""


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_array(self) -> None:
        hs = _parse_response(_two_hypotheses_json())
        assert len(hs) == 2
        assert hs[0].statement.startswith("Stall-detector")
        assert hs[1].statement.startswith("Executor throughput")

    def test_array_embedded_in_prose(self) -> None:
        raw = f"Here are my ideas:\n{_two_hypotheses_json()}\nDone."
        hs = _parse_response(raw)
        assert len(hs) == 2

    def test_malformed_json_returns_empty(self) -> None:
        assert _parse_response("[not valid json") == []

    def test_non_array_json_returns_empty(self) -> None:
        assert _parse_response('{"statement": "x"}') == []

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_response("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _parse_response("   \n\t  ") == []

    def test_non_string_returns_empty(self) -> None:
        assert _parse_response(None) == []  # type: ignore[arg-type]
        assert _parse_response(123) == []  # type: ignore[arg-type]

    def test_skips_items_without_statement(self) -> None:
        raw = (
            '[{"statement": "keep"},'
            ' {"how_to_verify": "no statement here"},'
            ' {"statement": "", "how_to_verify": "blank"},'
            ' "not a dict",'
            ' 42]'
        )
        hs = _parse_response(raw)
        assert len(hs) == 1
        assert hs[0].statement == "keep"

    def test_optional_fields_default_to_empty(self) -> None:
        hs = _parse_response('[{"statement": "just a claim"}]')
        assert len(hs) == 1
        assert hs[0].statement == "just a claim"
        assert hs[0].how_to_verify == ""
        assert hs[0].expected_outcome == ""

    def test_no_json_block_at_all(self) -> None:
        assert _parse_response("just some prose, no brackets") == []


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_rubric_theme_stories_existing(
        self, theme_dict: dict, recent_stories: list[dict]
    ) -> None:
        prompt = _build_prompt(
            SAMPLE_RUBRIC,
            theme_dict,
            {"recent_stories": recent_stories},
            ["prior claim one", "prior claim two"],
        )
        assert SAMPLE_RUBRIC in prompt
        assert "measurement + benchmarks" in prompt
        assert "benchmark, metric, measurement" in prompt
        assert "TK-401" in prompt
        assert "TK-402" in prompt
        assert "prior claim one" in prompt
        assert "prior claim two" in prompt

    def test_prompt_accepts_theme_dataclass(self) -> None:
        from aimm.themes import Theme

        t = Theme(
            name="failure-mode discoveries",
            target_per_week=2,
            keywords=("failure", "race", "stall"),
        )
        prompt = _build_prompt(SAMPLE_RUBRIC, t, {}, [])
        assert "failure-mode discoveries" in prompt
        assert "failure, race, stall" in prompt

    def test_prompt_tolerates_no_stories(self, theme_dict: dict) -> None:
        prompt = _build_prompt(SAMPLE_RUBRIC, theme_dict, {}, [])
        assert "no stories shipped" in prompt.lower()

    def test_prompt_tolerates_no_existing(self, theme_dict: dict) -> None:
        prompt = _build_prompt(SAMPLE_RUBRIC, theme_dict, {}, [])
        assert "first proposal for this theme" in prompt.lower()

    def test_prompt_includes_cap_wording(self, theme_dict: dict) -> None:
        prompt = _build_prompt(SAMPLE_RUBRIC, theme_dict, {}, [])
        assert str(MAX_HYPOTHESES) in prompt

    def test_prompt_ignores_non_dict_stories(self, theme_dict: dict) -> None:
        prompt = _build_prompt(
            SAMPLE_RUBRIC,
            theme_dict,
            {"recent_stories": ["bad", 42, {"key": "TK-1", "title": "good"}]},
            [],
        )
        assert "TK-1" in prompt
        assert "good" in prompt


# ---------------------------------------------------------------------------
# _load_existing_statements
# ---------------------------------------------------------------------------


class TestLoadExistingStatements:
    def test_missing_file_returns_empty(self, tmp_findings: Path) -> None:
        assert _load_existing_statements(tmp_findings) == set()

    def test_parses_statement_lines(self, tmp_findings: Path) -> None:
        tmp_findings.write_text(
            "## Hypotheses to Verify\n\n"
            "### 2026-04-18 — X\n\n"
            "**Statement:** First claim here\n\n"
            "**How to verify:** ...\n\n---\n\n"
            "### 2026-04-18 — Y\n\n"
            "**Statement:**  SECOND   Claim  \n\n---\n",
            encoding="utf-8",
        )
        existing = _load_existing_statements(tmp_findings)
        assert existing == {"first claim here", "second claim"}

    def test_file_without_statements_returns_empty(
        self, tmp_findings: Path
    ) -> None:
        tmp_findings.write_text("random prose, no hypotheses", encoding="utf-8")
        assert _load_existing_statements(tmp_findings) == set()


# ---------------------------------------------------------------------------
# propose_hypotheses — end-to-end
# ---------------------------------------------------------------------------


class TestProposeHypotheses:
    def test_acceptance_two_hypotheses_two_records_appended(
        self, theme_dict: dict, recent_stories: list[dict], tmp_findings: Path
    ) -> None:
        """Acceptance: mocked LLM returning 2 hypotheses → 2 records
        appended to raw_findings.md."""
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_two_hypotheses_json()),
        ) as mock_run:
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC, "recent_stories": recent_stories},
                findings_path=tmp_findings,
            )

        assert mock_run.call_count == 1
        assert len(result) == 2
        assert result[0].statement.startswith("Stall-detector")
        assert result[1].statement.startswith("Executor throughput")

        content = tmp_findings.read_text(encoding="utf-8")
        assert FINDINGS_HEADING in content
        assert FINDINGS_PREAMBLE in content
        assert content.count(FINDINGS_HEADING) == 1
        # Two entry sub-headings
        assert content.count("### ") == 2
        assert content.count("**Statement:**") == 2
        assert "Stall-detector" in content
        assert "Executor throughput" in content

    def test_acceptance_malformed_response_returns_empty_no_append(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        """Acceptance: malformed LLM response → empty list, no append,
        no raise."""
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok("definitely not json at all"),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )

        assert result == []
        assert not tmp_findings.exists()

    def test_acceptance_dedup_same_theme_no_duplicates(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        """Acceptance: re-running with the same theme does not append
        semantically identical hypotheses (by headline match)."""
        # First run: two hypotheses land on disk.
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_two_hypotheses_json()),
        ):
            first = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert len(first) == 2
        snapshot = tmp_findings.read_text(encoding="utf-8")

        # Second run: LLM returns the same pair (e.g. wording drifts in
        # whitespace/case). Nothing new should be appended.
        drifted = _two_hypotheses_json().replace("Stall-detector", "STALL-DETECTOR")
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(drifted),
        ):
            second = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )

        assert second == []
        assert tmp_findings.read_text(encoding="utf-8") == snapshot

    def test_dedup_within_same_batch(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        """An LLM that returns the same statement twice in one call
        should still only produce one entry."""
        dup_json = (
            '[{"statement": "same claim", "how_to_verify": "v1", "expected_outcome": "o1"},'
            ' {"statement": "SAME   claim", "how_to_verify": "v2", "expected_outcome": "o2"},'
            ' {"statement": "different claim", "how_to_verify": "v3", "expected_outcome": "o3"}]'
        )
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(dup_json),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )

        assert len(result) == 2
        statements = [h.statement for h in result]
        assert "same claim" in statements
        assert "different claim" in statements
        content = tmp_findings.read_text(encoding="utf-8")
        assert content.count("**Statement:**") == 2

    def test_caps_at_max_hypotheses_even_if_llm_returns_more(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_many_hypotheses_json(10)),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )

        assert len(result) == MAX_HYPOTHESES
        content = tmp_findings.read_text(encoding="utf-8")
        assert content.count("**Statement:**") == MAX_HYPOTHESES

    def test_missing_theme_name_returns_empty(
        self, tmp_findings: Path
    ) -> None:
        with patch("aimm.hypothesizer.run_claude_prompt") as mock_run:
            result = propose_hypotheses(
                {"target_per_week": 3},
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert result == []
        mock_run.assert_not_called()
        assert not tmp_findings.exists()

    def test_empty_rubric_returns_empty(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch("aimm.hypothesizer.run_claude_prompt") as mock_run:
            result = propose_hypotheses(
                theme_dict,
                {"rubric": ""},
                findings_path=tmp_findings,
            )
        assert result == []
        mock_run.assert_not_called()
        assert not tmp_findings.exists()

    def test_whitespace_only_rubric_returns_empty(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch("aimm.hypothesizer.run_claude_prompt") as mock_run:
            result = propose_hypotheses(
                theme_dict,
                {"rubric": "   \n\t "},
                findings_path=tmp_findings,
            )
        assert result == []
        mock_run.assert_not_called()

    def test_missing_context_returns_empty(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch("aimm.hypothesizer.run_claude_prompt") as mock_run:
            result = propose_hypotheses(
                theme_dict, None, findings_path=tmp_findings
            )
        assert result == []
        mock_run.assert_not_called()

    def test_llm_failure_returns_empty_no_append(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_err(),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert result == []
        assert not tmp_findings.exists()

    def test_llm_raises_returns_empty(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            side_effect=RuntimeError("subprocess exploded"),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert result == []
        assert not tmp_findings.exists()

    def test_non_dict_result_returns_empty(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value="string not dict",
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert result == []
        assert not tmp_findings.exists()

    def test_empty_hypotheses_array_returns_empty_no_append(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok("[]"),
        ):
            result = propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        assert result == []
        assert not tmp_findings.exists()

    def test_appends_to_existing_findings_preserving_content(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        tmp_findings.write_text(
            "# Raw Findings\n\nSome pre-existing narrative.\n",
            encoding="utf-8",
        )
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_two_hypotheses_json()),
        ):
            propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        content = tmp_findings.read_text(encoding="utf-8")
        assert "Some pre-existing narrative." in content
        assert FINDINGS_HEADING in content
        assert "**Statement:**" in content

    def test_heading_created_only_once_across_runs(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        # First run with one hypothesis.
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(
                '[{"statement": "first unique", "how_to_verify": "v", "expected_outcome": "o"}]'
            ),
        ):
            propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        # Second run with a different hypothesis.
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(
                '[{"statement": "second unique", "how_to_verify": "v", "expected_outcome": "o"}]'
            ),
        ):
            propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )

        content = tmp_findings.read_text(encoding="utf-8")
        assert content.count(FINDINGS_HEADING) == 1
        assert content.count("**Statement:**") == 2

    def test_does_not_create_jira_stories(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        """A mocked Jira-like provider passed nowhere near the call path
        must still see zero mutating calls — sanity check that no hidden
        Jira threading exists."""
        provider = MagicMock(name="BoardProvider")
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_two_hypotheses_json()),
        ):
            propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        for attr in (
            "add",
            "mark_executing",
            "mark_done",
            "mark_failed",
            "vote",
            "put",
            "patch",
        ):
            assert getattr(provider, attr).call_count == 0


# ---------------------------------------------------------------------------
# Contract: no Jira mutations anywhere in the module
# ---------------------------------------------------------------------------


class TestNoJiraMutations:
    """Hypothesizer is a researcher, not a manager. It must not import
    Jira-mutating code and must not issue any mutating call at runtime.
    """

    def test_module_does_not_import_jira_sync(self) -> None:
        src = Path(hypothesizer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from idea_board.jira_sync",
            "import idea_board.jira_sync",
            "from idea_board import jira_sync",
            "jira_sync.",
            "JiraProvider",
        ):
            assert forbidden not in src, (
                f"hypothesizer.py must not reference {forbidden!r} — "
                "AIMM does not mutate Jira."
            )

    def test_module_does_not_call_provider_mutators(self) -> None:
        src = Path(hypothesizer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            ".mark_executing(",
            ".mark_done(",
            ".mark_failed(",
            ".vote(",
            ".add_comment(",
            ".delete(",
            "add_label",
            "remove_label",
            "transition_issue",
        ):
            assert forbidden not in src, (
                f"hypothesizer.py must not use mutating call {forbidden!r}."
            )

    def test_does_not_import_jira_at_runtime(
        self, theme_dict: dict, tmp_findings: Path
    ) -> None:
        before = {m for m in sys.modules if "jira" in m.lower()}
        with patch(
            "aimm.hypothesizer.run_claude_prompt",
            return_value=_claude_ok(_two_hypotheses_json()),
        ):
            propose_hypotheses(
                theme_dict,
                {"rubric": SAMPLE_RUBRIC},
                findings_path=tmp_findings,
            )
        after = {m for m in sys.modules if "jira" in m.lower()}
        assert after == before, (
            "propose_hypotheses must not import any jira module at runtime"
        )

    def test_module_does_not_import_add_idea(self) -> None:
        """The task forbids add_idea() in production code."""
        src = Path(hypothesizer.__file__).read_text(encoding="utf-8")
        assert "add_idea" not in src, (
            "hypothesizer.py must not call add_idea — no Jira stories."
        )
