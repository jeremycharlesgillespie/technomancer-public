"""Tests for the knowledge gap resolution workflow module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.gap_resolver as gr_module
import agent.knowledge_gaps as kg_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_gap_resolver(temp_vault, monkeypatch):
    """Patch gap_resolver.py to use a temp vault."""
    temp_llm = temp_vault / "LLM Memory"
    monkeypatch.setattr(gr_module, "VAULT_PATH", temp_llm)
    monkeypatch.setattr(
        gr_module, "RESOLUTIONS_FILE",
        temp_llm / "Permanent" / "gap_resolutions.json",
    )
    monkeypatch.setattr(
        gr_module, "GAPS_FILE",
        temp_llm / "Permanent" / "knowledge_gaps.md",
    )
    monkeypatch.setattr(
        kg_module, "GAPS_FILE",
        temp_llm / "Permanent" / "knowledge_gaps.md",
    )
    return temp_llm


def _write_gaps(vault: Path, open_entries: str, resolved: str = "") -> None:
    f = vault / "Permanent" / "knowledge_gaps.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        f"# Knowledge Gap Log\n\n---\n\n## Open Gaps\n\n{open_entries}"
        f"## Resolved\n\n{resolved}",
        encoding="utf-8",
    )


def _gap(query: str, gap_type: str = "UNCERTAINTY", ts: str = "2026-04-01 10:00") -> str:
    return (
        f"- **[{gap_type}]** ({ts})\n"
        f"  - **Query:** {query}\n"
        f"  - **Response:** I'm not sure about that.\n\n"
    )


def _write_resolutions(vault: Path, resolutions: list) -> None:
    f = vault / "Permanent" / "gap_resolutions.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(resolutions), encoding="utf-8")


def _read_resolutions(vault: Path) -> list:
    f = vault / "Permanent" / "gap_resolutions.json"
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


MOCK_WIKI = {
    "title": "Spaghetti",
    "extract": "Spaghetti is a long, thin, solid, cylindrical pasta.",
    "url": "https://en.wikipedia.org/wiki/Spaghetti",
    "search_term": "spaghetti",
}


# ---------------------------------------------------------------------------
# _parse_open_gaps
# ---------------------------------------------------------------------------


class TestParseOpenGaps:
    def test_no_file(self, patched_gap_resolver):
        assert gr_module._parse_open_gaps() == []

    def test_parses_gaps(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("What is spaghetti?"))
        gaps = gr_module._parse_open_gaps()
        assert len(gaps) == 1
        assert gaps[0]["query"] == "What is spaghetti?"
        assert gaps[0]["gap_type"] == "uncertainty"

    def test_skips_empty_queries(self, patched_gap_resolver):
        # An entry with no query line
        bad = "- **[UNCERTAINTY]** (2026-04-01 10:00)\n  - **Response:** Something\n\n"
        _write_gaps(patched_gap_resolver, bad)
        assert gr_module._parse_open_gaps() == []


# ---------------------------------------------------------------------------
# _extract_search_terms
# ---------------------------------------------------------------------------


class TestExtractSearchTerms:
    def test_extracts_keywords(self):
        terms = gr_module._extract_search_terms("What is spaghetti pasta?")
        assert "spaghetti" in terms or "spaghetti pasta" in terms

    def test_removes_stop_words(self):
        terms = gr_module._extract_search_terms("What is the meaning of life?")
        assert "what" not in terms
        assert "meaning" in terms or "life" in terms

    def test_limits_results(self):
        terms = gr_module._extract_search_terms(
            "very long query with many different words about various topics"
        )
        assert len(terms) <= 5


# ---------------------------------------------------------------------------
# fetch_wikipedia_suggestion (mocked)
# ---------------------------------------------------------------------------


class TestFetchWikipediaSuggestion:
    def test_returns_result(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "title": "Spaghetti",
            "extract": "Spaghetti is a long, thin pasta.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Spaghetti"}},
        }

        with patch("agent.gap_resolver.requests.get", return_value=mock_resp):
            result = gr_module.fetch_wikipedia_suggestion("What is spaghetti?")

        assert result is not None
        assert result["title"] == "Spaghetti"
        assert "search_term" in result

    def test_returns_none_on_failure(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_search_resp = MagicMock()
        mock_search_resp.json.return_value = {"query": {"search": []}}

        with patch("agent.gap_resolver.requests.get", side_effect=[mock_resp, mock_search_resp] * 5):
            result = gr_module.fetch_wikipedia_suggestion("xyzzy foobar gibberish")

        assert result is None

    def test_returns_none_on_exception(self):
        with patch("agent.gap_resolver.requests.get", side_effect=Exception("network error")):
            result = gr_module.fetch_wikipedia_suggestion("anything")
        assert result is None


# ---------------------------------------------------------------------------
# Resolution state management
# ---------------------------------------------------------------------------


class TestResolutionState:
    def test_load_empty(self, patched_gap_resolver):
        assert gr_module._load_resolutions() == []

    def test_save_and_load(self, patched_gap_resolver):
        data = [{"query": "test", "status": "suggested"}]
        gr_module._save_resolutions(data)
        loaded = gr_module._load_resolutions()
        assert loaded == data

    def test_find_resolution(self):
        resolutions = [
            {"query": "What is spaghetti?", "status": "suggested"},
            {"query": "How does gravity work?", "status": "suggested"},
        ]
        idx, entry = gr_module._find_resolution(resolutions, "spaghetti")
        assert idx == 0
        assert entry["query"] == "What is spaghetti?"

    def test_find_resolution_not_found(self):
        idx, entry = gr_module._find_resolution([], "anything")
        assert idx == -1
        assert entry is None


# ---------------------------------------------------------------------------
# generate_suggestions
# ---------------------------------------------------------------------------


class TestGenerateSuggestions:
    def test_no_gaps(self, patched_gap_resolver):
        assert gr_module.generate_suggestions() == []

    def test_creates_suggestions(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("What is spaghetti?"))

        with patch.object(gr_module, "lookup_fact", return_value=[]), \
             patch.object(gr_module, "fetch_wikipedia_suggestion", return_value=MOCK_WIKI):
            new = gr_module.generate_suggestions()

        assert len(new) == 1
        assert new[0]["status"] == "suggested"
        assert new[0]["suggestion"]["title"] == "Spaghetti"

        # Check persisted
        saved = _read_resolutions(patched_gap_resolver)
        assert len(saved) == 1

    def test_auto_covers_when_facts_exist(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("What is spaghetti?"))
        fake_fact = [{"category": "definition", "key": "spaghetti", "value": "A pasta"}]

        with patch.object(gr_module, "lookup_fact", return_value=fake_fact):
            new = gr_module.generate_suggestions()

        assert len(new) == 1
        assert new[0]["status"] == "auto_covered"

    def test_skips_already_processed(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("What is spaghetti?"))
        _write_resolutions(patched_gap_resolver, [
            {"query": "What is spaghetti?", "status": "suggested", "suggestion": MOCK_WIKI},
        ])

        with patch.object(gr_module, "lookup_fact", return_value=[]), \
             patch.object(gr_module, "fetch_wikipedia_suggestion", return_value=MOCK_WIKI):
            new = gr_module.generate_suggestions()

        assert len(new) == 0

    def test_no_source_when_wiki_fails(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("xyzzy gibberish question"))

        with patch.object(gr_module, "lookup_fact", return_value=[]), \
             patch.object(gr_module, "fetch_wikipedia_suggestion", return_value=None):
            new = gr_module.generate_suggestions()

        assert len(new) == 1
        assert new[0]["status"] == "no_source"


# ---------------------------------------------------------------------------
# accept_suggestion
# ---------------------------------------------------------------------------


class TestAcceptSuggestion:
    def test_accepts_and_resolves(self, patched_gap_resolver):
        _write_gaps(patched_gap_resolver, _gap("What is spaghetti?"))
        _write_resolutions(patched_gap_resolver, [{
            "query": "What is spaghetti?",
            "gap_type": "uncertainty",
            "domain": "food_cooking",
            "detected": "2026-04-01 10:00",
            "status": "suggested",
            "suggestion": MOCK_WIKI,
            "resolved_at": None,
            "resolution_note": None,
        }])

        with patch.object(gr_module, "add_fact") as mock_add, \
             patch.object(gr_module, "resolve_gap") as mock_resolve:
            result = gr_module.accept_suggestion("spaghetti")

        assert "Accepted" in result
        assert "Spaghetti" in result
        mock_add.assert_called_once()
        mock_resolve.assert_called_once()

        saved = _read_resolutions(patched_gap_resolver)
        assert saved[0]["status"] == "accepted"
        assert saved[0]["resolved_at"] is not None

    def test_not_found(self, patched_gap_resolver):
        _write_resolutions(patched_gap_resolver, [])
        result = gr_module.accept_suggestion("nonexistent")
        assert "No resolution found" in result

    def test_wrong_status(self, patched_gap_resolver):
        _write_resolutions(patched_gap_resolver, [{
            "query": "test", "status": "dismissed", "suggestion": MOCK_WIKI,
        }])
        result = gr_module.accept_suggestion("test")
        assert "status" in result.lower()


# ---------------------------------------------------------------------------
# dismiss_suggestion
# ---------------------------------------------------------------------------


class TestDismissSuggestion:
    def test_dismisses(self, patched_gap_resolver):
        _write_resolutions(patched_gap_resolver, [{
            "query": "What is spaghetti?", "status": "suggested",
            "suggestion": MOCK_WIKI,
        }])
        result = gr_module.dismiss_suggestion("spaghetti", "Not relevant")
        assert "Dismissed" in result

        saved = _read_resolutions(patched_gap_resolver)
        assert saved[0]["status"] == "dismissed"
        assert saved[0]["resolution_note"] == "Not relevant"

    def test_not_found(self, patched_gap_resolver):
        result = gr_module.dismiss_suggestion("nothing")
        assert "No resolution found" in result


# ---------------------------------------------------------------------------
# get_pending_suggestions
# ---------------------------------------------------------------------------


class TestGetPendingSuggestions:
    def test_no_pending(self, patched_gap_resolver):
        result = gr_module.get_pending_suggestions()
        assert "No pending" in result

    def test_shows_pending(self, patched_gap_resolver):
        _write_resolutions(patched_gap_resolver, [{
            "query": "What is spaghetti?",
            "gap_type": "uncertainty",
            "domain": "food_cooking",
            "status": "suggested",
            "suggestion": MOCK_WIKI,
        }])
        result = gr_module.get_pending_suggestions()
        assert "Pending Resolution Suggestions (1)" in result
        assert "Spaghetti" in result
        assert "spaghetti" in result.lower()


# ---------------------------------------------------------------------------
# get_resolution_stats
# ---------------------------------------------------------------------------


class TestGetResolutionStats:
    def test_no_data(self, patched_gap_resolver):
        result = gr_module.get_resolution_stats()
        assert "No resolution data" in result

    def test_with_data(self, patched_gap_resolver):
        _write_resolutions(patched_gap_resolver, [
            {"query": "q1", "status": "suggested"},
            {"query": "q2", "status": "accepted"},
            {"query": "q3", "status": "accepted"},
            {"query": "q4", "status": "dismissed"},
            {"query": "q5", "status": "auto_covered"},
        ])
        result = gr_module.get_resolution_stats()
        assert "Suggested | 1" in result
        assert "Accepted | 2" in result
        assert "Dismissed | 1" in result
        assert "Auto Covered | 1" in result
        assert "Resolution rate: **60%**" in result  # 3/5


# ---------------------------------------------------------------------------
# _domain_to_facts_category
# ---------------------------------------------------------------------------


class TestDomainToFactsCategory:
    def test_food_maps_to_definition(self):
        assert gr_module._domain_to_facts_category("food_cooking") == "definition"

    def test_geography_maps_to_geography(self):
        assert gr_module._domain_to_facts_category("geography") == "geography"

    def test_unknown_maps_to_definition(self):
        assert gr_module._domain_to_facts_category("unknown_domain") == "definition"

    def test_conversions(self):
        assert gr_module._domain_to_facts_category("conversions_units") == "conversion"


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestScheduling:
    def test_returns_positive(self):
        assert gr_module._seconds_until_next_run() > 0

    def test_within_24_hours(self):
        assert gr_module._seconds_until_next_run() <= 24 * 3600
