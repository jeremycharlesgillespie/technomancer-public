"""Tests for the domain coverage tracker module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.domain_coverage as dc_module
import agent.knowledge_gaps as kg_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_domain_coverage(temp_vault, monkeypatch):
    """Patch domain_coverage.py to use a temp vault."""
    temp_llm = temp_vault / "LLM Memory"
    monkeypatch.setattr(dc_module, "VAULT_PATH", temp_llm)
    monkeypatch.setattr(dc_module, "CONVERSATIONS_DIR", temp_llm / "Conversations")
    monkeypatch.setattr(dc_module, "REPORTS_DIR", temp_llm / "Permanent" / "domain_coverage")

    # Also patch knowledge_gaps GAPS_FILE
    monkeypatch.setattr(kg_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")
    monkeypatch.setattr(dc_module, "GAPS_FILE", temp_llm / "Permanent" / "knowledge_gaps.md")

    return temp_llm


def _write_conversation_log(vault_path: Path, date_str: str, entries: list[tuple[str, str, str]]) -> None:
    """Write a fake conversation log. entries = [(time, query, response), ...]"""
    conv_dir = vault_path / "Conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Conversations - {date_str}\n\n---\n"]
    for time_str, query, response in entries:
        lines.append(f"### {time_str} - testuser\n**Q:** {query}\n**A:** {response}\n\n---\n")
    (conv_dir / f"{date_str}.md").write_text("\n".join(lines), encoding="utf-8")


def _write_gaps_file(vault_path: Path, entries: str) -> None:
    """Write a knowledge_gaps.md with given open entries."""
    gaps_file = vault_path / "Permanent" / "knowledge_gaps.md"
    gaps_file.parent.mkdir(parents=True, exist_ok=True)
    gaps_file.write_text(
        f"# Knowledge Gap Log\n\n---\n\n## Open Gaps\n\n{entries}## Resolved\n\n",
        encoding="utf-8",
    )


def _gap_entry(query: str, gap_type: str = "UNCERTAINTY") -> str:
    return (
        f"- **[{gap_type}]** (2026-04-01 10:00)\n"
        f"  - **Query:** {query}\n"
        f"  - **Response:** I'm not sure about that.\n\n"
    )


# ---------------------------------------------------------------------------
# classify_text
# ---------------------------------------------------------------------------


class TestClassifyText:
    def test_food_keywords(self):
        hits = dc_module.classify_text("How do I cook spaghetti with tomato sauce?")
        assert "food_cooking" in hits
        assert hits["food_cooking"] >= 2  # cook + spaghetti (sauce also)

    def test_technology_keywords(self):
        hits = dc_module.classify_text("How to use Python with a database API?")
        assert "technology" in hits
        assert hits["technology"] >= 2

    def test_no_match(self):
        hits = dc_module.classify_text("xyz foobar baz")
        assert hits == {}

    def test_multi_word_keyword(self):
        hits = dc_module.classify_text("What is machine learning used for?")
        assert "technology" in hits

    def test_case_insensitive(self):
        hits = dc_module.classify_text("PYTHON DATABASE API")
        assert "technology" in hits


class TestClassifyToPrimaryDomain:
    def test_returns_best_match(self):
        domain = dc_module.classify_to_primary_domain("How to cook pasta and rice?")
        assert domain == "food_cooking"

    def test_uncategorized_for_no_match(self):
        domain = dc_module.classify_to_primary_domain("xyzzy foobar")
        assert domain == "uncategorized"


# ---------------------------------------------------------------------------
# Conversation scanning
# ---------------------------------------------------------------------------


class TestParseConversationLog:
    def test_empty_file(self, patched_domain_coverage):
        path = patched_domain_coverage / "Conversations" / "2026-04-01.md"
        assert dc_module._parse_conversation_log(path) == []

    def test_parses_entries(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "It is a type of pasta."),
            ("11:00:00", "How does gravity work?", "Gravity is a force."),
        ])
        entries = dc_module._parse_conversation_log(
            patched_domain_coverage / "Conversations" / f"{today}.md"
        )
        assert len(entries) == 2
        assert entries[0]["query"] == "What is spaghetti?"


class TestScanConversations:
    def test_counts_domains(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "A pasta dish."),
            ("11:00:00", "How to cook rice?", "Boil water first."),
            ("12:00:00", "What is Python?", "A programming language."),
        ])
        counts = dc_module.scan_conversations(days=1)
        assert counts.get("food_cooking", 0) >= 2
        assert counts.get("technology", 0) >= 1

    def test_empty_when_no_logs(self, patched_domain_coverage):
        counts = dc_module.scan_conversations(days=1)
        assert counts == {}


# ---------------------------------------------------------------------------
# Knowledge gap scanning
# ---------------------------------------------------------------------------


class TestScanKnowledgeGaps:
    def test_no_file(self, patched_domain_coverage):
        assert dc_module.scan_knowledge_gaps() == {}

    def test_counts_gap_domains(self, patched_domain_coverage):
        entries = (
            _gap_entry("What is spaghetti?")
            + _gap_entry("How to cook pasta?")
            + _gap_entry("What is quantum physics?")
        )
        _write_gaps_file(patched_domain_coverage, entries)
        counts = dc_module.scan_knowledge_gaps()
        assert counts.get("food_cooking", 0) >= 2
        assert counts.get("science", 0) >= 1


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------


class TestBuildCoverageMatrix:
    def test_includes_all_taxonomy_domains(self, patched_domain_coverage):
        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            matrix = dc_module.build_coverage_matrix(days=1)
        # Should include all taxonomy domains even with no data
        for domain in dc_module.DOMAIN_TAXONOMY:
            assert domain in matrix

    def test_merges_all_sources(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "A pasta."),
        ])
        _write_gaps_file(patched_domain_coverage, _gap_entry("What is pasta?"))

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            matrix = dc_module.build_coverage_matrix(days=1)

        food = matrix.get("food_cooking", {})
        assert food.get("conversations", 0) >= 1
        assert food.get("gaps", 0) >= 1


# ---------------------------------------------------------------------------
# Coverage level
# ---------------------------------------------------------------------------


class TestCoverageLevel:
    def test_none_gaps_but_no_facts(self):
        assert dc_module._coverage_level(facts=0, gaps=3) == "NONE"

    def test_empty_no_data(self):
        assert dc_module._coverage_level(facts=0, gaps=0) == "EMPTY"

    def test_low_more_gaps_than_facts(self):
        assert dc_module._coverage_level(facts=2, gaps=5) == "LOW"

    def test_partial_some_gaps(self):
        assert dc_module._coverage_level(facts=10, gaps=3) == "PARTIAL"

    def test_good_no_gaps(self):
        assert dc_module._coverage_level(facts=10, gaps=0) == "GOOD"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestGenerateCoverageReport:
    def test_empty_when_no_data(self, patched_domain_coverage):
        with patch.object(dc_module, "build_coverage_matrix", return_value={}):
            assert dc_module.generate_coverage_report() == ""

    def test_report_has_structure(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "A pasta."),
        ])
        _write_gaps_file(patched_domain_coverage, _gap_entry("What is pasta?"))

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            report = dc_module.generate_coverage_report(days=1)

        assert "# Domain Coverage Report" in report
        assert "## Coverage Matrix" in report
        assert "## Summary" in report
        assert "## Flagged Domains" in report

    def test_flags_domains_with_gaps_no_facts(self, patched_domain_coverage):
        _write_gaps_file(patched_domain_coverage, _gap_entry("What is spaghetti?"))

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            report = dc_module.generate_coverage_report(days=1)

        assert "NONE" in report


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------


class TestWriteCoverageReport:
    def test_writes_file(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "A pasta."),
        ])

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            path = dc_module.write_coverage_report(days=1)

        assert path != ""
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Domain Coverage Report" in content

    def test_creates_directory(self, patched_domain_coverage):
        reports_dir = patched_domain_coverage / "Permanent" / "domain_coverage"
        assert not reports_dir.exists()

        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "test query", "test response"),
        ])

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            dc_module.write_coverage_report(days=1)

        assert reports_dir.exists()


# ---------------------------------------------------------------------------
# Discord summary
# ---------------------------------------------------------------------------


class TestFormatDiscordSummary:
    def test_includes_stats(self, patched_domain_coverage):
        today = datetime.now().strftime("%Y-%m-%d")
        _write_conversation_log(patched_domain_coverage, today, [
            ("10:00:00", "What is spaghetti?", "A pasta."),
        ])
        _write_gaps_file(patched_domain_coverage, _gap_entry("What is pasta?"))

        with patch.object(dc_module, "get_facts_stats", return_value={"total_facts": 0, "categories": {}}):
            summary = dc_module.format_discord_summary()

        assert "Weekly Domain Coverage Report" in summary
        assert "domain_coverage/" in summary

    def test_empty_when_no_data(self, patched_domain_coverage):
        with patch.object(dc_module, "build_coverage_matrix", return_value={}):
            assert dc_module.format_discord_summary() == ""


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestSecondsUntilNextReport:
    def test_returns_positive(self):
        assert dc_module._seconds_until_next_report() > 0

    def test_within_one_week(self):
        assert dc_module._seconds_until_next_report() <= 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Suggest actions
# ---------------------------------------------------------------------------


class TestSuggestActions:
    def test_suggests_seed_facts_when_empty(self):
        suggestions = dc_module._suggest_actions("food_cooking", {"facts": 0, "gaps": 2, "conversations": 1})
        assert any("add seed facts" in s.lower() for s in suggestions)

    def test_suggests_review_when_some_facts(self):
        suggestions = dc_module._suggest_actions("science", {"facts": 3, "gaps": 5, "conversations": 2})
        assert any("review" in s.lower() for s in suggestions)

    def test_suggests_bulk_add_when_frequent(self):
        suggestions = dc_module._suggest_actions("history", {"facts": 2, "gaps": 1, "conversations": 5})
        assert any("frequently discussed" in s.lower() for s in suggestions)

    def test_suggests_research_for_high_gaps(self):
        suggestions = dc_module._suggest_actions("math", {"facts": 1, "gaps": 5, "conversations": 2})
        assert any("systematic" in s.lower() for s in suggestions)
