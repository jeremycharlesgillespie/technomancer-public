"""Tests for the knowledge gap reporter module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.gap_reporter as gr_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_gap_reporter(temp_vault, monkeypatch):
    """Patch gap_reporter.py to use a temp vault and a temp facts DB."""
    temp_llm = temp_vault / "LLM Memory"
    monkeypatch.setattr(gr_module, "VAULT_PATH", temp_llm)
    monkeypatch.setattr(gr_module, "REPORTS_DIR", temp_llm / "Permanent" / "gap_reports")

    # Also patch the GAPS_FILE that gap_reporter imports from knowledge_gaps
    import agent.knowledge_gaps as kg_module
    monkeypatch.setattr(kg_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")
    monkeypatch.setattr(gr_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")

    return temp_llm


def _write_gaps_file(vault_path: Path, open_entries: str, resolved_entries: str = "") -> None:
    """Helper to write a knowledge_gaps.md with given content."""
    gaps_file = vault_path / "Permanent" / "knowledge_gaps.md"
    gaps_file.parent.mkdir(parents=True, exist_ok=True)
    gaps_file.write_text(
        f"# Knowledge Gap Log\n\n---\n\n## Open Gaps\n\n{open_entries}## Resolved\n\n{resolved_entries}",
        encoding="utf-8",
    )


def _make_entry(
    gap_type: str = "UNCERTAINTY",
    timestamp: str = "2026-04-01 10:00",
    query: str = "What is spaghetti?",
    snippet: str = "I'm not sure about that.",
    escalated: bool = False,
) -> str:
    esc = " [escalated]" if escalated else ""
    return (
        f"- **[{gap_type}]** ({timestamp}{esc})\n"
        f"  - **Query:** {query}\n"
        f"  - **Response:** {snippet}\n\n"
    )


# ---------------------------------------------------------------------------
# _parse_open_gaps
# ---------------------------------------------------------------------------


class TestParseOpenGaps:
    def test_no_file(self, patched_gap_reporter):
        assert gr_module._parse_open_gaps() == []

    def test_empty_open_section(self, patched_gap_reporter):
        _write_gaps_file(patched_gap_reporter, "")
        assert gr_module._parse_open_gaps() == []

    def test_single_uncertainty(self, patched_gap_reporter):
        entry = _make_entry()
        _write_gaps_file(patched_gap_reporter, entry)

        gaps = gr_module._parse_open_gaps()
        assert len(gaps) == 1
        assert gaps[0]["gap_type"] == "uncertainty"
        assert gaps[0]["query"] == "What is spaghetti?"
        assert gaps[0]["timestamp"] == "2026-04-01 10:00"
        assert gaps[0]["escalated"] == "False"

    def test_failure_and_escalated(self, patched_gap_reporter):
        entries = (
            _make_entry(gap_type="FAILURE", query="Explain quantum tunneling", escalated=True)
            + _make_entry(query="Define blockchain")
        )
        _write_gaps_file(patched_gap_reporter, entries)

        gaps = gr_module._parse_open_gaps()
        assert len(gaps) == 2
        assert gaps[0]["gap_type"] == "failure"
        assert gaps[0]["escalated"] == "True"
        assert gaps[1]["gap_type"] == "uncertainty"

    def test_ignores_resolved_section(self, patched_gap_reporter):
        resolved = _make_entry(query="Old resolved query")
        _write_gaps_file(patched_gap_reporter, "", resolved)

        gaps = gr_module._parse_open_gaps()
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# _cross_reference_gaps
# ---------------------------------------------------------------------------


class TestCrossReferenceGaps:
    def test_no_matches(self, patched_gap_reporter):
        gaps = [{"query": "xyzzy foobar", "gap_type": "uncertainty"}]
        with patch.object(gr_module, "lookup_fact", return_value=[]):
            result = gr_module._cross_reference_gaps(gaps)
        assert len(result) == 1
        assert result[0]["facts_matches"] == []

    def test_finds_matching_fact(self, patched_gap_reporter):
        gaps = [{"query": "What is spaghetti exactly?", "gap_type": "uncertainty"}]
        fake_fact = {"category": "definition", "key": "spaghetti", "value": "A pasta type"}

        def mock_lookup(term):
            if term == "spaghetti":
                return [fake_fact]
            return []

        with patch.object(gr_module, "lookup_fact", side_effect=mock_lookup):
            result = gr_module._cross_reference_gaps(gaps)

        assert len(result[0]["facts_matches"]) == 1
        assert result[0]["facts_matches"][0]["key"] == "spaghetti"

    def test_deduplicates_facts(self, patched_gap_reporter):
        gaps = [{"query": "spaghetti pasta dinner", "gap_type": "uncertainty"}]
        same_fact = {"category": "definition", "key": "spaghetti", "value": "A pasta"}

        with patch.object(gr_module, "lookup_fact", return_value=[same_fact]):
            result = gr_module._cross_reference_gaps(gaps)

        # Even though multiple terms could match the same fact, it's deduplicated
        assert len(result[0]["facts_matches"]) == 1


# ---------------------------------------------------------------------------
# _group_by_topic
# ---------------------------------------------------------------------------


class TestGroupByTopic:
    def test_no_recurring_keywords(self):
        gaps = [
            {"query": "What is alpha?"},
            {"query": "How does beta work?"},
        ]
        groups = gr_module._group_by_topic(gaps)
        assert "uncategorized" in groups
        assert len(groups["uncategorized"]) == 2

    def test_recurring_keyword_groups(self):
        gaps = [
            {"query": "What is Python used for?"},
            {"query": "How to install Python packages?"},
            {"query": "What is JavaScript?"},
        ]
        groups = gr_module._group_by_topic(gaps)
        assert "python" in groups
        assert len(groups["python"]) == 2

    def test_mixed_groups(self):
        gaps = [
            {"query": "Python error handling"},
            {"query": "Python async patterns"},
            {"query": "random unrelated topic"},
        ]
        groups = gr_module._group_by_topic(gaps)
        assert "python" in groups
        assert len(groups["python"]) == 2
        assert "uncategorized" in groups


# ---------------------------------------------------------------------------
# generate_gap_report
# ---------------------------------------------------------------------------


class TestGenerateGapReport:
    def test_empty_when_no_gaps(self, patched_gap_reporter):
        assert gr_module.generate_gap_report() == ""

    def test_empty_when_no_file(self, patched_gap_reporter):
        assert gr_module.generate_gap_report() == ""

    def test_report_contains_summary_table(self, patched_gap_reporter):
        entries = _make_entry() + _make_entry(gap_type="FAILURE", query="unknown topic")
        _write_gaps_file(patched_gap_reporter, entries)

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            report = gr_module.generate_gap_report()

        assert "# Knowledge Gap Report" in report
        assert "## Summary" in report
        assert "Total open gaps | 2" in report
        assert "Uncertainty signals | 1" in report
        assert "Outright failures | 1" in report

    def test_report_contains_topic_groups(self, patched_gap_reporter):
        entries = (
            _make_entry(query="Python async error")
            + _make_entry(query="Python threading issue")
        )
        _write_gaps_file(patched_gap_reporter, entries)

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            report = gr_module.generate_gap_report()

        assert "## Gaps by Topic" in report
        assert "Python" in report

    def test_report_shows_resolvable_section(self, patched_gap_reporter):
        entries = _make_entry(query="What is spaghetti exactly?")
        _write_gaps_file(patched_gap_reporter, entries)
        fake_fact = {"category": "definition", "key": "spaghetti", "value": "A pasta"}

        def mock_lookup(term):
            return [fake_fact] if term == "spaghetti" else []

        with patch.object(gr_module, "lookup_fact", side_effect=mock_lookup):
            report = gr_module.generate_gap_report()

        assert "## Potentially Resolvable Gaps" in report
        assert "spaghetti" in report

    def test_report_recommendations_failures(self, patched_gap_reporter):
        entries = _make_entry(gap_type="FAILURE", query="unknown topic")
        _write_gaps_file(patched_gap_reporter, entries)

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            report = gr_module.generate_gap_report()

        assert "## Recommendations" in report
        assert "failure" in report.lower()


# ---------------------------------------------------------------------------
# write_weekly_report
# ---------------------------------------------------------------------------


class TestWriteWeeklyReport:
    def test_writes_file(self, patched_gap_reporter):
        entries = _make_entry()
        _write_gaps_file(patched_gap_reporter, entries)

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            path = gr_module.write_weekly_report()

        assert path != ""
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Knowledge Gap Report" in content

    def test_creates_reports_dir(self, patched_gap_reporter):
        entries = _make_entry()
        _write_gaps_file(patched_gap_reporter, entries)
        reports_dir = patched_gap_reporter / "Permanent" / "gap_reports"
        assert not reports_dir.exists()

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            gr_module.write_weekly_report()

        assert reports_dir.exists()

    def test_returns_empty_when_no_gaps(self, patched_gap_reporter):
        assert gr_module.write_weekly_report() == ""


# ---------------------------------------------------------------------------
# format_discord_summary
# ---------------------------------------------------------------------------


class TestFormatDiscordSummary:
    def test_empty_when_no_gaps(self, patched_gap_reporter):
        assert gr_module.format_discord_summary("/some/path") == ""

    def test_includes_counts(self, patched_gap_reporter):
        entries = (
            _make_entry()
            + _make_entry(gap_type="FAILURE", query="another question")
        )
        _write_gaps_file(patched_gap_reporter, entries)

        with patch.object(gr_module, "lookup_fact", return_value=[]):
            summary = gr_module.format_discord_summary("/some/path")

        assert "Weekly Knowledge Gap Report" in summary
        assert "Open gaps: **2**" in summary
        assert "gap_reports/" in summary


# ---------------------------------------------------------------------------
# _seconds_until_next_report
# ---------------------------------------------------------------------------


class TestSecondsUntilNextReport:
    def test_returns_positive(self):
        seconds = gr_module._seconds_until_next_report()
        assert seconds > 0

    def test_within_one_week(self):
        seconds = gr_module._seconds_until_next_report()
        assert seconds <= 7 * 24 * 3600
