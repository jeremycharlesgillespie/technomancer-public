"""Tests for the knowledge_enrichment module — auto-enrichment of knowledge gaps."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.knowledge_enrichment import (
    _get_already_enriched_queries,
    _log_enrichment,
    _try_enrich_query,
    get_enrichment_report,
    get_knowledge_enrichment_tools,
    run_enrichment,
)


@pytest.fixture(autouse=True)
def _use_temp_vault(tmp_path, monkeypatch):
    """Redirect enrichment log to temp dir."""
    monkeypatch.setattr(
        "agent.knowledge_enrichment.ENRICHMENT_LOG",
        tmp_path / "enrichment_log.json",
    )
    monkeypatch.setattr(
        "agent.knowledge_enrichment.VAULT_PATH",
        tmp_path,
    )


MOCK_GAP = {
    "query": "What is spaghetti?",
    "gap_type": "failure",
    "timestamp": "2026-04-01 10:00",
    "response_snippet": "I don't have reliable information",
    "status": "open",
    "source": "gap_log",
}

MOCK_CLUSTER = {
    "label": "spaghetti + definition",
    "domain": "general",
    "count": 3,
    "gap_types": {"failure": 3},
    "queries": ["What is spaghetti?"],
    "first_seen": "2026-04-01",
    "last_seen": "2026-04-09",
}


class TestEnrichmentLog:
    def test_log_and_retrieve(self, tmp_path):
        _log_enrichment("test query", "enriched", "wikipedia", "Test")
        entries = json.loads((tmp_path / "enrichment_log.json").read_text())
        assert len(entries) == 1
        assert entries[0]["status"] == "enriched"
        assert entries[0]["query"] == "test query"

    def test_already_enriched(self, tmp_path):
        _log_enrichment("spaghetti", "enriched", "wikipedia", "Spaghetti")
        already = _get_already_enriched_queries()
        assert "spaghetti" in already

    def test_not_enriched_if_no_source(self, tmp_path):
        _log_enrichment("unknown thing", "no_source")
        already = _get_already_enriched_queries()
        assert "unknown thing" not in already


class TestTryEnrichQuery:
    @patch("agent.facts_db.lookup_fact", return_value=[
        {"key": "spaghetti", "value": "A pasta.", "category": "definition", "source": "seed"}
    ])
    def test_skips_existing_fact(self, mock_lookup):
        result = _try_enrich_query("spaghetti", "general")
        assert result["status"] == "already_exists"

    @patch("agent.knowledge_enrichment._write_vault_article")
    @patch("agent.facts_db.add_fact")
    @patch("agent.knowledge_fallback._query_wikipedia", return_value="A thin Italian pasta.")
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_enriches_from_wikipedia(self, mock_lookup, mock_wiki, mock_add, mock_vault):
        result = _try_enrich_query("spaghetti", "general")
        assert result["status"] == "enriched"
        assert result["source"] == "wikipedia"
        mock_add.assert_called_once()

    @patch("agent.knowledge_fallback._web_search_summary", return_value="A type of pasta")
    @patch("agent.knowledge_fallback._search_wikipedia", return_value=None)
    @patch("agent.knowledge_fallback._query_wikipedia", return_value=None)
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_falls_back_to_web_search(self, mock_lookup, mock_wiki, mock_search, mock_web):
        result = _try_enrich_query("spaghetti", "general")
        assert result["status"] == "enriched"
        assert result["source"] == "web_search"

    @patch("agent.knowledge_fallback._web_search_summary", return_value=None)
    @patch("agent.knowledge_fallback._search_wikipedia", return_value=None)
    @patch("agent.knowledge_fallback._query_wikipedia", return_value=None)
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_all_sources_fail(self, mock_lookup, mock_wiki, mock_search, mock_web):
        result = _try_enrich_query("xyznonexistent", "general")
        assert result["status"] == "no_source"


class TestRunEnrichment:
    @patch("agent.gap_frequency.cluster_gaps", return_value=[])
    @patch("agent.gap_frequency._scan_conversations_for_gaps", return_value=[])
    @patch("agent.gap_frequency._parse_all_gaps", return_value=[])
    def test_no_gaps(self, mock_parse, mock_scan, mock_cluster):
        result = run_enrichment(days=7)
        assert result["enriched"] == 0

    @patch("agent.knowledge_enrichment._try_enrich_query")
    @patch("agent.gap_frequency.cluster_gaps")
    @patch("agent.gap_frequency._scan_conversations_for_gaps", return_value=[])
    @patch("agent.gap_frequency._parse_all_gaps")
    def test_enriches_gaps(self, mock_parse, mock_scan, mock_cluster, mock_try):
        mock_parse.return_value = [MOCK_GAP]
        mock_cluster.return_value = [MOCK_CLUSTER]
        mock_try.return_value = {
            "status": "enriched",
            "title": "spaghetti",
            "source": "wikipedia",
            "category": "definition",
        }

        result = run_enrichment(days=7)
        assert result["enriched"] == 1
        assert len(result["details"]) == 1

    @patch("agent.knowledge_enrichment._try_enrich_query")
    @patch("agent.gap_frequency.cluster_gaps")
    @patch("agent.gap_frequency._scan_conversations_for_gaps", return_value=[])
    @patch("agent.gap_frequency._parse_all_gaps")
    def test_skips_already_enriched(self, mock_parse, mock_scan, mock_cluster, mock_try):
        # Pre-log an enrichment
        _log_enrichment("what is spaghetti?", "enriched", "wikipedia", "spaghetti")

        mock_parse.return_value = [MOCK_GAP]
        mock_cluster.return_value = [MOCK_CLUSTER]

        result = run_enrichment(days=7)
        assert result["skipped"] == 1
        assert result["enriched"] == 0
        mock_try.assert_not_called()

    @patch("agent.knowledge_enrichment._try_enrich_query")
    @patch("agent.gap_frequency.cluster_gaps")
    @patch("agent.gap_frequency._scan_conversations_for_gaps", return_value=[])
    @patch("agent.gap_frequency._parse_all_gaps")
    def test_respects_max_enrichments(self, mock_parse, mock_scan, mock_cluster, mock_try):
        gaps = [dict(MOCK_GAP, query=f"query {i}") for i in range(10)]
        clusters = [
            dict(MOCK_CLUSTER, queries=[f"query {i}"], label=f"cluster_{i}")
            for i in range(10)
        ]
        mock_parse.return_value = gaps
        mock_cluster.return_value = clusters
        mock_try.return_value = {
            "status": "enriched", "title": "test", "source": "wikipedia", "category": "definition",
        }

        result = run_enrichment(days=7, max_enrichments=3)
        assert result["enriched"] == 3


class TestGetEnrichmentReport:
    def test_empty_report(self):
        report = get_enrichment_report()
        assert "No enrichment runs" in report

    def test_report_with_data(self):
        _log_enrichment("spaghetti", "enriched", "wikipedia", "Spaghetti")
        _log_enrichment("unknown", "no_source")
        report = get_enrichment_report()
        assert "Knowledge Base Enrichment" in report
        assert "Spaghetti" in report
        assert "1" in report  # enriched count


class TestGetTools:
    def test_returns_two_tools(self):
        tools = get_knowledge_enrichment_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"enrich_knowledge_base", "enrichment_report"}

    def test_report_tool_runs(self):
        tools = get_knowledge_enrichment_tools()
        report_tool = next(t for t in tools if t.name == "enrichment_report")
        result = report_tool.function()
        assert isinstance(result, str)
