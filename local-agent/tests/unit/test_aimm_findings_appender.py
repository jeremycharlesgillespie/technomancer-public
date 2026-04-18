"""Tests for aimm.findings_appender — shared writer for raw_findings.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from aimm.findings_appender import (
    FINDINGS_HEADING,
    FINDINGS_PREAMBLE,
    HYPOTHESES_HEADING,
    HYPOTHESES_PREAMBLE,
    SUGGESTIONS_HEADING,
    SUGGESTIONS_PREAMBLE,
    append_finding,
    append_hypothesis,
    append_suggestion,
)


@pytest.fixture
def findings_path(tmp_path: Path) -> Path:
    return tmp_path / "raw_findings.md"


# ---------------------------------------------------------------------------
# TestAppendFinding — verify append_finding writes correct markdown
# under the "## Findings" heading.
# ---------------------------------------------------------------------------


class TestAppendFinding:
    def test_writes_entry_under_findings_heading(
        self, findings_path: Path
    ) -> None:
        assert (
            append_finding(
                {
                    "story_key": "TK-1",
                    "headline": "A thing happened",
                    "why_it_matters": "Because autonomy",
                    "evidence_pointer": "commit abc123",
                    "theme": "failure-mode",
                },
                findings_path=findings_path,
            )
            is True
        )
        text = findings_path.read_text(encoding="utf-8")
        assert FINDINGS_HEADING in text
        assert FINDINGS_PREAMBLE in text
        assert "- finding_key: TK-1" in text
        assert "A thing happened" in text
        assert "- why_it_matters: Because autonomy" in text
        assert "- evidence_pointer: commit abc123" in text
        assert "- theme: failure-mode" in text

    def test_each_field_renders_on_its_own_line(
        self, findings_path: Path
    ) -> None:
        append_finding(
            {
                "story_key": "TK-2",
                "headline": "Headline",
                "why_it_matters": "Reason",
            },
            findings_path=findings_path,
        )
        lines = findings_path.read_text(encoding="utf-8").splitlines()
        assert "- finding_key: TK-2" in lines
        assert "- why_it_matters: Reason" in lines

    def test_multiline_values_flattened_to_single_line(
        self, findings_path: Path
    ) -> None:
        append_finding(
            {
                "story_key": "TK-5",
                "why_it_matters": "line1\nline2\r\nline3",
            },
            findings_path=findings_path,
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "line1" in text and "line2" in text and "line3" in text
        for line in text.splitlines():
            if line.startswith("- why_it_matters:"):
                assert "\n" not in line and "\r" not in line

    def test_missing_story_key_returns_false(self, findings_path: Path) -> None:
        assert append_finding({}, findings_path=findings_path) is False
        assert (
            append_finding({"story_key": ""}, findings_path=findings_path)
            is False
        )
        assert (
            append_finding({"story_key": "   "}, findings_path=findings_path)
            is False
        )
        assert not findings_path.exists() or findings_path.read_text() == ""

    def test_non_dict_payload_returns_false(self, findings_path: Path) -> None:
        assert (
            append_finding("not a dict", findings_path=findings_path) is False
        )
        assert append_finding(None, findings_path=findings_path) is False
        assert append_finding(42, findings_path=findings_path) is False

    def test_accepts_dataclass_payload(self, findings_path: Path) -> None:
        @dataclass
        class FindingRecord:
            story_key: str
            headline: str = ""

        assert (
            append_finding(
                FindingRecord(story_key="TK-7", headline="dc"),
                findings_path=findings_path,
            )
            is True
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "- finding_key: TK-7" in text
        assert "dc" in text

    def test_accepts_to_dict_object(self, findings_path: Path) -> None:
        class Obs:
            def to_dict(self) -> dict:
                return {"story_key": "TK-11", "headline": "via to_dict"}

        assert append_finding(Obs(), findings_path=findings_path) is True
        assert "- finding_key: TK-11" in findings_path.read_text(
            encoding="utf-8"
        )

    def test_os_error_on_write_returns_false(
        self, findings_path: Path
    ) -> None:
        with patch.object(Path, "write_text", side_effect=OSError("disk")):
            assert (
                append_finding(
                    {"story_key": "TK-1"}, findings_path=findings_path
                )
                is False
            )


# ---------------------------------------------------------------------------
# TestAppendSuggestion — verify append_suggestion writes correct markdown
# under the "## Suggested Approvals" heading.
# ---------------------------------------------------------------------------


class TestAppendSuggestion:
    def test_writes_entry_under_suggestions_heading(
        self, findings_path: Path
    ) -> None:
        assert (
            append_suggestion(
                {
                    "story_key": "TK-42",
                    "recommend": True,
                    "reasoning": "Meets all four rubric criteria",
                    "title": "Add retry logic",
                },
                findings_path=findings_path,
            )
            is True
        )
        text = findings_path.read_text(encoding="utf-8")
        assert SUGGESTIONS_HEADING in text
        assert SUGGESTIONS_PREAMBLE in text
        assert "- suggestion_key: TK-42" in text
        assert "RECOMMEND APPROVE" in text
        assert "Meets all four rubric criteria" in text
        assert "Add retry logic" in text

    def test_skip_verdict_rendered_for_recommend_false(
        self, findings_path: Path
    ) -> None:
        append_suggestion(
            {"story_key": "TK-3", "recommend": False, "reasoning": "boring"},
            findings_path=findings_path,
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "SKIP" in text
        assert "- recommend: false" in text

    def test_missing_reasoning_renders_placeholder(
        self, findings_path: Path
    ) -> None:
        append_suggestion(
            {"story_key": "TK-4", "recommend": True},
            findings_path=findings_path,
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "- reasoning: (no reasoning provided)" in text

    def test_missing_story_key_returns_false(
        self, findings_path: Path
    ) -> None:
        assert append_suggestion({}, findings_path=findings_path) is False
        assert (
            append_suggestion({"story_key": ""}, findings_path=findings_path)
            is False
        )

    def test_non_dict_payload_returns_false(self, findings_path: Path) -> None:
        assert (
            append_suggestion("nope", findings_path=findings_path) is False
        )
        assert append_suggestion(None, findings_path=findings_path) is False


# ---------------------------------------------------------------------------
# TestAppendHypothesis — verify append_hypothesis writes correct markdown
# under the "## Hypotheses to Verify" heading.
# ---------------------------------------------------------------------------


class TestAppendHypothesis:
    def test_writes_entry_under_hypotheses_heading(
        self, findings_path: Path
    ) -> None:
        assert (
            append_hypothesis(
                {
                    "statement": "Haiku brain reduces weekly cost by 40%",
                    "how_to_verify": "Compare story_model_usage rollup",
                    "expected_outcome": "~40% drop in brain cost",
                    "theme": "measurement",
                },
                findings_path=findings_path,
            )
            is True
        )
        text = findings_path.read_text(encoding="utf-8")
        assert HYPOTHESES_HEADING in text
        assert HYPOTHESES_PREAMBLE in text
        assert "Haiku brain reduces weekly cost by 40%" in text
        assert "- how_to_verify: Compare story_model_usage rollup" in text
        assert "- expected_outcome: ~40% drop in brain cost" in text
        assert "- theme: measurement" in text

    def test_hypothesis_key_is_normalized_statement(
        self, findings_path: Path
    ) -> None:
        append_hypothesis(
            {"statement": "  Mixed CASE   Statement  "},
            findings_path=findings_path,
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "- hypothesis_key: mixed case statement" in text

    def test_missing_optional_fields_use_placeholders(
        self, findings_path: Path
    ) -> None:
        append_hypothesis(
            {"statement": "Bare statement"}, findings_path=findings_path
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "- how_to_verify: (not specified)" in text
        assert "- expected_outcome: (not specified)" in text

    def test_missing_statement_returns_false(
        self, findings_path: Path
    ) -> None:
        assert append_hypothesis({}, findings_path=findings_path) is False
        assert (
            append_hypothesis({"statement": ""}, findings_path=findings_path)
            is False
        )
        assert (
            append_hypothesis(
                {"statement": "   "}, findings_path=findings_path
            )
            is False
        )

    def test_non_dict_payload_returns_false(self, findings_path: Path) -> None:
        assert append_hypothesis("nope", findings_path=findings_path) is False
        assert append_hypothesis(None, findings_path=findings_path) is False


# ---------------------------------------------------------------------------
# TestIdempotency — re-appending a payload with an already-seen dedup key
# is a no-op. Scope is per-section: observer, suggester, and hypothesizer
# dedup independently so the same story_key can exist under two sections.
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_duplicate_finding_story_key_is_noop(
        self, findings_path: Path
    ) -> None:
        assert append_finding(
            {"story_key": "TK-9", "headline": "first"},
            findings_path=findings_path,
        ) is True
        assert append_finding(
            {"story_key": "TK-9", "headline": "second"},
            findings_path=findings_path,
        ) is False

        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- finding_key: TK-9") == 1
        assert "first" in text
        assert "second" not in text

    def test_duplicate_suggestion_story_key_is_noop(
        self, findings_path: Path
    ) -> None:
        assert append_suggestion(
            {"story_key": "TK-1", "reasoning": "first"},
            findings_path=findings_path,
        ) is True
        assert append_suggestion(
            {"story_key": "TK-1", "reasoning": "second"},
            findings_path=findings_path,
        ) is False
        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- suggestion_key: TK-1") == 1
        assert "first" in text
        assert "second" not in text

    def test_duplicate_hypothesis_statement_is_noop(
        self, findings_path: Path
    ) -> None:
        assert append_hypothesis(
            {"statement": "Same claim."}, findings_path=findings_path
        ) is True
        assert append_hypothesis(
            {"statement": "Same claim."}, findings_path=findings_path
        ) is False
        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- hypothesis_key:") == 1

    def test_hypothesis_dedup_is_case_and_whitespace_insensitive(
        self, findings_path: Path
    ) -> None:
        append_hypothesis(
            {"statement": "Recursive splitting shortens cycle time."},
            findings_path=findings_path,
        )
        assert (
            append_hypothesis(
                {"statement": "  RECURSIVE   splitting shortens CYCLE time.  "},
                findings_path=findings_path,
            )
            is False
        )
        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- hypothesis_key:") == 1

    def test_dedup_survives_across_process_restart(
        self, findings_path: Path
    ) -> None:
        """The module holds no in-memory cache — the dedup set is rebuilt
        from the file on every call, so dedup must still hold after the
        process is torn down and restarted."""
        append_finding({"story_key": "TK-1"}, findings_path=findings_path)
        assert (
            append_finding({"story_key": "TK-1"}, findings_path=findings_path)
            is False
        )

    def test_same_story_key_across_sections_does_not_collide(
        self, findings_path: Path
    ) -> None:
        """Observer and suggester can both log story ``TK-1`` — they
        write to separate sections with separate dedup scopes."""
        assert append_finding(
            {"story_key": "TK-1", "headline": "observation"},
            findings_path=findings_path,
        ) is True
        assert append_suggestion(
            {
                "story_key": "TK-1",
                "recommend": False,
                "reasoning": "already shipped",
            },
            findings_path=findings_path,
        ) is True
        text = findings_path.read_text(encoding="utf-8")
        assert "- finding_key: TK-1" in text
        assert "- suggestion_key: TK-1" in text


# ---------------------------------------------------------------------------
# TestSectionCreation — a missing section's heading + preamble must be
# created the first time a producer appends to it; subsequent appends
# under the same section must reuse the existing heading without
# duplicating it.
# ---------------------------------------------------------------------------


class TestSectionCreation:
    def test_findings_heading_created_on_first_append(
        self, findings_path: Path
    ) -> None:
        assert not findings_path.exists()
        append_finding({"story_key": "TK-1"}, findings_path=findings_path)
        text = findings_path.read_text(encoding="utf-8")
        assert FINDINGS_HEADING in text
        assert FINDINGS_PREAMBLE in text

    def test_suggestions_heading_created_on_first_append(
        self, findings_path: Path
    ) -> None:
        assert not findings_path.exists()
        append_suggestion(
            {"story_key": "TK-1", "recommend": True},
            findings_path=findings_path,
        )
        text = findings_path.read_text(encoding="utf-8")
        assert SUGGESTIONS_HEADING in text
        assert SUGGESTIONS_PREAMBLE in text

    def test_hypotheses_heading_created_on_first_append(
        self, findings_path: Path
    ) -> None:
        assert not findings_path.exists()
        append_hypothesis(
            {"statement": "A claim"}, findings_path=findings_path
        )
        text = findings_path.read_text(encoding="utf-8")
        assert HYPOTHESES_HEADING in text
        assert HYPOTHESES_PREAMBLE in text

    def test_parent_directory_created_if_missing(
        self, tmp_path: Path
    ) -> None:
        nested = tmp_path / "docs" / "raw_findings.md"
        assert not nested.parent.exists()
        append_finding({"story_key": "TK-1"}, findings_path=nested)
        assert nested.exists()

    def test_heading_appears_only_once_across_multiple_entries(
        self, findings_path: Path
    ) -> None:
        append_finding({"story_key": "TK-1"}, findings_path=findings_path)
        append_finding({"story_key": "TK-2"}, findings_path=findings_path)
        append_finding({"story_key": "TK-3"}, findings_path=findings_path)

        text = findings_path.read_text(encoding="utf-8")
        assert text.count(FINDINGS_HEADING) == 1
        assert text.count(FINDINGS_PREAMBLE) == 1
        assert text.count("- finding_key: TK-") == 3

    def test_all_three_sections_coexist_in_one_file(
        self, findings_path: Path
    ) -> None:
        """Three producers each appending once yields three distinct
        sections with their headings created on first write."""
        append_finding(
            {
                "story_key": "TK-100",
                "headline": "Shipped cleanly",
                "theme": "autonomy",
            },
            findings_path=findings_path,
        )
        append_suggestion(
            {
                "story_key": "TK-200",
                "recommend": True,
                "reasoning": "passes rubric",
            },
            findings_path=findings_path,
        )
        append_hypothesis(
            {
                "statement": "Throughput correlates with model tier.",
                "how_to_verify": "daily_stats join",
                "expected_outcome": "positive correlation",
            },
            findings_path=findings_path,
        )

        text = findings_path.read_text(encoding="utf-8")
        assert FINDINGS_HEADING in text
        assert SUGGESTIONS_HEADING in text
        assert HYPOTHESES_HEADING in text
        assert "- finding_key: TK-100" in text
        assert "- suggestion_key: TK-200" in text
        assert (
            "- hypothesis_key: throughput correlates with model tier." in text
        )
        assert text.count(FINDINGS_HEADING) == 1
        assert text.count(SUGGESTIONS_HEADING) == 1
        assert text.count(HYPOTHESES_HEADING) == 1
