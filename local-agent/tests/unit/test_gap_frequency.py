"""Tests for the gap frequency tracker module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.gap_frequency as gf_module
import agent.knowledge_gaps as kg_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_gap_frequency(temp_vault, monkeypatch):
    """Patch gap_frequency.py to use a temp vault."""
    temp_llm = temp_vault / "LLM Memory"
    monkeypatch.setattr(gf_module, "VAULT_PATH", temp_llm)
    monkeypatch.setattr(gf_module, "CONVERSATIONS_DIR", temp_llm / "Conversations")
    monkeypatch.setattr(gf_module, "REPORTS_DIR", temp_llm / "Permanent" / "gap_frequency")
    monkeypatch.setattr(gf_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")
    monkeypatch.setattr(kg_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")
    return temp_llm


def _write_gaps_file(vault: Path, open_entries: str, resolved_entries: str = "") -> None:
    """Helper to write knowledge_gaps.md."""
    f = vault / "Permanent" / "knowledge_gaps.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        f"# Knowledge Gap Log\n\n---\n\n## Open Gaps\n\n{open_entries}"
        f"## Resolved\n\n{resolved_entries}",
        encoding="utf-8",
    )


def _gap(query: str, gap_type: str = "UNCERTAINTY", ts: str = "2026-04-01 10:00") -> str:
    return (
        f"- **[{gap_type}]** ({ts})\n"
        f"  - **Query:** {query}\n"
        f"  - **Response:** I'm not sure about that.\n\n"
    )


def _write_conv_log(vault: Path, date_str: str, entries: list[tuple[str, str, str]]) -> None:
    """Write a fake conversation log. entries = [(time, query, response), ...]"""
    d = vault / "Conversations"
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# Conversations - {date_str}\n\n---\n"]
    for t, q, r in entries:
        lines.append(f"### {t} - testuser\n**Q:** {q}\n**A:** {r}\n\n---\n")
    (d / f"{date_str}.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# _parse_all_gaps
# ---------------------------------------------------------------------------


class TestParseAllGaps:
    def test_no_file(self, patched_gap_frequency):
        assert gf_module._parse_all_gaps() == []

    def test_parses_open_gaps(self, patched_gap_frequency):
        _write_gaps_file(patched_gap_frequency, _gap("What is spaghetti?"))
        gaps = gf_module._parse_all_gaps()
        assert len(gaps) == 1
        assert gaps[0]["status"] == "open"
        assert gaps[0]["source"] == "gap_log"
        assert gaps[0]["query"] == "What is spaghetti?"

    def test_parses_resolved_gaps(self, patched_gap_frequency):
        _write_gaps_file(patched_gap_frequency, "", _gap("Old question"))
        gaps = gf_module._parse_all_gaps()
        assert len(gaps) == 1
        assert gaps[0]["status"] == "resolved"

    def test_parses_both_sections(self, patched_gap_frequency):
        _write_gaps_file(
            patched_gap_frequency,
            _gap("Open question"),
            _gap("Resolved question"),
        )
        gaps = gf_module._parse_all_gaps()
        assert len(gaps) == 2
        statuses = {g["status"] for g in gaps}
        assert statuses == {"open", "resolved"}


# ---------------------------------------------------------------------------
# _scan_conversations_for_gaps
# ---------------------------------------------------------------------------


class TestScanConversationsForGaps:
    def test_no_logs(self, patched_gap_frequency):
        assert gf_module._scan_conversations_for_gaps(days=1) == []

    def test_detects_uncertainty(self, patched_gap_frequency):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conv_log(patched_gap_frequency, today, [
            ("10:00:00", "What is quantum tunneling?", "I'm not sure about quantum tunneling details."),
        ])
        found = gf_module._scan_conversations_for_gaps(days=1)
        assert len(found) == 1
        assert found[0]["gap_type"] == "uncertainty"
        assert found[0]["source"] == "conversation_scan"

    def test_detects_failure(self, patched_gap_frequency):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conv_log(patched_gap_frequency, today, [
            ("10:00:00", "Explain dark matter", "I cannot answer that question."),
        ])
        found = gf_module._scan_conversations_for_gaps(days=1)
        assert len(found) == 1
        assert found[0]["gap_type"] == "failure"

    def test_skips_clean_responses(self, patched_gap_frequency):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conv_log(patched_gap_frequency, today, [
            ("10:00:00", "What is Python?", "Python is a programming language."),
        ])
        found = gf_module._scan_conversations_for_gaps(days=1)
        assert len(found) == 0


# ---------------------------------------------------------------------------
# Keyword extraction & similarity
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_removes_stop_words(self):
        kws = gf_module._extract_keywords("what is the definition of spaghetti")
        assert "what" not in kws
        assert "the" not in kws
        assert "spaghetti" in kws
        assert "definition" in kws

    def test_short_words_excluded(self):
        kws = gf_module._extract_keywords("is it ok to do so")
        assert len(kws) == 0  # All words are <= 2 chars or stop words


class TestQuerySimilarity:
    def test_identical(self):
        a = {"foo", "bar", "baz"}
        assert gf_module._query_similarity(a, a) == 1.0

    def test_no_overlap(self):
        assert gf_module._query_similarity({"foo"}, {"bar"}) == 0.0

    def test_partial_overlap(self):
        sim = gf_module._query_similarity({"foo", "bar"}, {"bar", "baz"})
        assert 0.3 < sim < 0.5  # 1/3 ≈ 0.33

    def test_empty_sets(self):
        assert gf_module._query_similarity(set(), {"foo"}) == 0.0


# ---------------------------------------------------------------------------
# cluster_gaps
# ---------------------------------------------------------------------------


class TestClusterGaps:
    def test_empty_input(self):
        assert gf_module.cluster_gaps([]) == []

    def test_single_gap(self):
        gaps = [{"query": "What is spaghetti?", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"}]
        clusters = gf_module.cluster_gaps(gaps)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 1

    def test_similar_queries_cluster(self):
        gaps = [
            {"query": "What is spaghetti pasta?", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"},
            {"query": "Tell me about spaghetti pasta dishes", "gap_type": "failure", "timestamp": "2026-04-02 10:00"},
        ]
        clusters = gf_module.cluster_gaps(gaps)
        # Should be 1 cluster since "spaghetti" and "pasta" overlap
        assert len(clusters) == 1
        assert clusters[0]["count"] == 2

    def test_dissimilar_queries_separate(self):
        gaps = [
            {"query": "What is spaghetti?", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"},
            {"query": "How does quantum physics work?", "gap_type": "uncertainty", "timestamp": "2026-04-02 10:00"},
        ]
        clusters = gf_module.cluster_gaps(gaps)
        assert len(clusters) == 2

    def test_sorted_by_frequency(self):
        gaps = [
            {"query": "quantum physics question", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"},
            {"query": "spaghetti recipe", "gap_type": "uncertainty", "timestamp": "2026-04-01 11:00"},
            {"query": "another spaghetti recipe question", "gap_type": "failure", "timestamp": "2026-04-02 10:00"},
            {"query": "spaghetti cooking method", "gap_type": "uncertainty", "timestamp": "2026-04-03 10:00"},
        ]
        clusters = gf_module.cluster_gaps(gaps)
        assert clusters[0]["count"] >= clusters[-1]["count"]

    def test_cluster_has_timestamps(self):
        gaps = [
            {"query": "spaghetti recipe", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"},
            {"query": "spaghetti cooking", "gap_type": "uncertainty", "timestamp": "2026-04-05 10:00"},
        ]
        clusters = gf_module.cluster_gaps(gaps)
        assert clusters[0]["first_seen"] == "2026-04-01"
        assert clusters[0]["last_seen"] == "2026-04-05"

    def test_cluster_has_domain(self):
        gaps = [
            {"query": "What is spaghetti?", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"},
        ]
        clusters = gf_module.cluster_gaps(gaps)
        assert clusters[0]["domain"] == "food_cooking"


# ---------------------------------------------------------------------------
# generate_frequency_report
# ---------------------------------------------------------------------------


class TestGenerateFrequencyReport:
    def test_empty_when_no_gaps(self, patched_gap_frequency):
        assert gf_module.generate_frequency_report(days=1) == ""

    def test_report_structure(self, patched_gap_frequency):
        _write_gaps_file(
            patched_gap_frequency,
            _gap("What is spaghetti?") + _gap("How to cook spaghetti?"),
        )
        report = gf_module.generate_frequency_report(days=1)
        assert "# Gap Frequency Report" in report
        assert "## Summary" in report
        assert "## Priority Ranking" in report
        assert "## Top Clusters" in report
        assert "## Gap Frequency by Domain" in report
        assert "## Recommendations" in report

    def test_deduplication(self, patched_gap_frequency):
        # Write a gap to the log AND create a matching conversation entry
        _write_gaps_file(patched_gap_frequency, _gap("What is spaghetti?", ts="2026-04-01 10:00"))
        _write_conv_log(patched_gap_frequency, "2026-04-01", [
            ("10:00:00", "What is spaghetti?", "I'm not sure about spaghetti."),
        ])
        report = gf_module.generate_frequency_report(days=30)
        # The duplicate should be deduped — total should be 1, not 2
        assert "Total gap occurrences | 1" in report

    def test_includes_conversation_scan_gaps(self, patched_gap_frequency):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conv_log(patched_gap_frequency, today, [
            ("10:00:00", "What is dark matter?", "I don't know about dark matter."),
        ])
        report = gf_module.generate_frequency_report(days=1)
        assert "From conversation scan | 1" in report


# ---------------------------------------------------------------------------
# write_frequency_report
# ---------------------------------------------------------------------------


class TestWriteFrequencyReport:
    def test_writes_file(self, patched_gap_frequency):
        _write_gaps_file(patched_gap_frequency, _gap("Test query"))
        path = gf_module.write_frequency_report(days=1)
        assert path != ""
        assert Path(path).exists()

    def test_creates_dir(self, patched_gap_frequency):
        d = patched_gap_frequency / "Permanent" / "gap_frequency"
        assert not d.exists()
        _write_gaps_file(patched_gap_frequency, _gap("Test query"))
        gf_module.write_frequency_report(days=1)
        assert d.exists()

    def test_empty_when_no_gaps(self, patched_gap_frequency):
        assert gf_module.write_frequency_report(days=1) == ""


# ---------------------------------------------------------------------------
# format_discord_summary
# ---------------------------------------------------------------------------


class TestFormatDiscordSummary:
    def test_with_clusters(self):
        clusters = [
            {"label": "spaghetti + pasta", "domain": "food_cooking", "count": 3,
             "gap_types": {"uncertainty": 2, "failure": 1}, "queries": [], "first_seen": "", "last_seen": ""},
            {"label": "quantum + physics", "domain": "science", "count": 1,
             "gap_types": {"uncertainty": 1}, "queries": [], "first_seen": "", "last_seen": ""},
        ]
        summary = gf_module.format_discord_summary(clusters=clusters)
        assert "Gap Frequency Report" in summary
        assert "spaghetti + pasta" in summary
        assert "**3x**" in summary

    def test_empty_clusters(self):
        assert gf_module.format_discord_summary(clusters=[]) == ""


# ---------------------------------------------------------------------------
# _tool_gap_frequency
# ---------------------------------------------------------------------------


class TestToolGapFrequency:
    def test_no_gaps(self, patched_gap_frequency):
        result = gf_module._tool_gap_frequency(days=1)
        assert "No knowledge gaps found" in result

    def test_with_gaps(self, patched_gap_frequency):
        _write_gaps_file(
            patched_gap_frequency,
            _gap("What is spaghetti?") + _gap("How to cook pasta?"),
        )
        result = gf_module._tool_gap_frequency(days=1)
        assert "Top Topics" in result
        assert "total occurrences" in result


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestScheduling:
    def test_returns_positive(self):
        assert gf_module._seconds_until_next_report() > 0

    def test_within_one_week(self):
        assert gf_module._seconds_until_next_report() <= 7 * 24 * 3600
