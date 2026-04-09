"""Tests for the knowledge_fallback module — tiered factual lookup pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from agent.knowledge_fallback import (
    _guess_category,
    _query_wikipedia,
    _search_wikipedia,
    _web_search_summary,
    get_knowledge_fallback_tools,
    knowledge_lookup,
)


class TestGuessCategory:
    def test_geography(self):
        assert _guess_category("France capital", "Paris is the capital") == "geography"

    def test_technology(self):
        assert _guess_category("Python", "A programming language") == "technology"

    def test_science(self):
        assert _guess_category("photosynthesis", "converts light energy in plant cells") == "science"

    def test_history(self):
        assert _guess_category("Lincoln", "Abraham Lincoln was president of the US") == "history"

    def test_default_definition(self):
        assert _guess_category("spaghetti", "a type of long thin pasta") == "definition"


class TestQueryWikipedia:
    @patch("requests.get")
    def test_successful_query(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "title": "Spaghetti",
                        "extract": "Spaghetti is a long, thin pasta of Italian origin.\n\nHistory section.",
                    }
                }
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _query_wikipedia("Spaghetti")
        assert result is not None
        assert "pasta" in result.lower()

    @patch("requests.get")
    def test_no_page_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {"pages": {"-1": {"title": "Nonexistent", "missing": ""}}}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _query_wikipedia("xyznonexistent12345")
        assert result is None

    @patch("requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = _query_wikipedia("test")
        assert result is None

    @patch("requests.get")
    def test_truncates_long_extracts(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {
                "pages": {
                    "1": {"title": "Test", "extract": "A" * 600}
                }
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _query_wikipedia("Test")
        assert result is not None
        assert len(result) <= 500


class TestSearchWikipedia:
    @patch("agent.knowledge_fallback._query_wikipedia")
    @patch("requests.get")
    def test_opensearch_then_query(self, mock_get, mock_query):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            "spaghet",
            ["Spaghetti"],
            ["Italian pasta"],
            ["https://en.wikipedia.org/wiki/Spaghetti"],
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        mock_query.return_value = "Spaghetti is a pasta."

        result = _search_wikipedia("spaghet")
        assert result == "Spaghetti is a pasta."
        mock_query.assert_called_once_with("Spaghetti")

    @patch("requests.get")
    def test_no_search_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = ["query", [], [], []]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _search_wikipedia("xyznonexistent")
        assert result is None


class TestWebSearchSummary:
    @patch("agent.web_search.web_search")
    def test_returns_description(self, mock_search):
        mock_search.return_value = (
            "Web search results for: spaghetti definition\n\n"
            "1. **Spaghetti - Wikipedia**\n"
            "   A long thin pasta of Italian origin\n"
            "   Source: https://en.wikipedia.org\n"
        )
        result = _web_search_summary("spaghetti")
        assert result is not None
        assert "pasta" in result.lower()

    @patch("agent.web_search.web_search")
    def test_no_results(self, mock_search):
        mock_search.return_value = "No results found for: xyzabc"
        result = _web_search_summary("xyzabc")
        assert result is None

    @patch("agent.web_search.web_search")
    def test_search_error(self, mock_search):
        mock_search.return_value = "Search error: timeout"
        result = _web_search_summary("test")
        assert result is None


class TestKnowledgeLookup:
    def test_empty_query(self):
        result = knowledge_lookup("")
        assert "provide a query" in result.lower()

    @patch("agent.facts_db.lookup_fact")
    def test_facts_db_hit(self, mock_lookup):
        mock_lookup.return_value = [
            {"key": "spaghetti", "category": "definition", "value": "A thin pasta.", "source": "seed"}
        ]
        result = knowledge_lookup("spaghetti")
        assert "spaghetti" in result.lower()
        assert "thin pasta" in result.lower()
        assert "seed" in result

    @patch("agent.knowledge_fallback._log_gap_resolved")
    @patch("agent.knowledge_fallback._cache_result")
    @patch("agent.knowledge_fallback._query_wikipedia", return_value="A type of pasta.")
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_wikipedia_fallback(self, mock_lookup, mock_wiki, mock_cache, mock_log):
        result = knowledge_lookup("spaghetti")
        assert "pasta" in result.lower()
        assert "Wikipedia" in result
        mock_cache.assert_called_once()
        mock_log.assert_called_once()

    @patch("agent.knowledge_fallback._log_gap_resolved")
    @patch("agent.knowledge_fallback._cache_result")
    @patch("agent.knowledge_fallback._web_search_summary", return_value="A popular Italian dish")
    @patch("agent.knowledge_fallback._search_wikipedia", return_value=None)
    @patch("agent.knowledge_fallback._query_wikipedia", return_value=None)
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_web_search_fallback(self, mock_lookup, mock_wiki, mock_wiki_search, mock_web, mock_cache, mock_log):
        result = knowledge_lookup("spaghetti")
        assert "Italian" in result
        assert "web search" in result
        mock_cache.assert_called_once()

    @patch("agent.knowledge_fallback._log_gap_unresolved")
    @patch("agent.knowledge_fallback._web_search_summary", return_value=None)
    @patch("agent.knowledge_fallback._search_wikipedia", return_value=None)
    @patch("agent.knowledge_fallback._query_wikipedia", return_value=None)
    @patch("agent.facts_db.lookup_fact", return_value=[])
    def test_all_tiers_fail(self, mock_lookup, mock_wiki, mock_wiki_search, mock_web, mock_log):
        result = knowledge_lookup("xyznonexistent12345")
        assert "could not find" in result.lower()
        mock_log.assert_called_once()


class TestCacheResult:
    @patch("agent.facts_db.add_fact")
    def test_caches_to_facts_db(self, mock_add):
        from agent.knowledge_fallback import _cache_result

        _cache_result("spaghetti", "A thin Italian pasta", "wikipedia")
        mock_add.assert_called_once_with(
            "definition", "spaghetti", "A thin Italian pasta", source="wikipedia"
        )

    @patch("agent.facts_db.add_fact", side_effect=Exception("DB error"))
    def test_handles_cache_failure(self, mock_add):
        from agent.knowledge_fallback import _cache_result

        # Should not raise
        _cache_result("test", "value", "test")


class TestGetTools:
    def test_returns_one_tool(self):
        tools = get_knowledge_fallback_tools()
        assert len(tools) == 1
        assert tools[0].name == "knowledge_lookup"

    @patch("agent.facts_db.lookup_fact", return_value=[
        {"key": "test", "category": "definition", "value": "A test.", "source": "seed"}
    ])
    def test_tool_function_runs(self, mock_lookup):
        tools = get_knowledge_fallback_tools()
        result = tools[0].function(query="test")
        assert isinstance(result, str)
        assert "test" in result.lower()
