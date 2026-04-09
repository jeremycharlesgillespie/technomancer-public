"""Tests for the reference source enhancement system."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.ref_enrichment as re_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_ref_enrichment(temp_vault, monkeypatch):
    """Patch ref_enrichment.py to use a temp vault."""
    temp_llm = temp_vault / "LLM Memory"
    monkeypatch.setattr(re_module, "VAULT_PATH", temp_llm)
    monkeypatch.setattr(re_module, "REFERENCES_DIR", temp_llm / "Permanent" / "References")
    monkeypatch.setattr(re_module, "IMPORT_LOG_FILE", temp_llm / "Permanent" / "import_log.json")
    return temp_llm


MOCK_WIKI_ARTICLE = {
    "title": "Spaghetti",
    "extract": "Spaghetti is a long, thin, solid, cylindrical pasta of Italian origin.",
    "description": "Type of pasta",
    "url": "https://en.wikipedia.org/wiki/Spaghetti",
}


def _read_import_log(vault: Path) -> list:
    f = vault / "Permanent" / "import_log.json"
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def _write_import_log(vault: Path, entries: list) -> None:
    f = vault / "Permanent" / "import_log.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self):
        assert re_module._sanitize_filename("Hello World") == "Hello_World"

    def test_special_chars(self):
        result = re_module._sanitize_filename('File: "test" <stuff>')
        assert ":" not in result
        assert '"' not in result
        assert "<" not in result

    def test_long_title(self):
        result = re_module._sanitize_filename("A" * 100)
        assert len(result) <= 80

    def test_empty(self):
        assert re_module._sanitize_filename("") == "untitled"


# ---------------------------------------------------------------------------
# write_reference_article
# ---------------------------------------------------------------------------


class TestWriteReferenceArticle:
    def test_writes_file(self, patched_ref_enrichment):
        path = re_module.write_reference_article(
            title="Spaghetti",
            extract="A type of pasta.",
            url="https://example.com",
            source="wikipedia",
            domain="food_cooking",
        )
        assert path != ""
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "# Spaghetti" in content
        assert "A type of pasta." in content
        assert "source: wikipedia" in content
        assert "domain: food_cooking" in content

    def test_creates_directory(self, patched_ref_enrichment):
        refs_dir = patched_ref_enrichment / "Permanent" / "References"
        assert not refs_dir.exists()
        re_module.write_reference_article(
            title="Test", extract="Content", url="", source="test", domain="test",
        )
        assert refs_dir.exists()

    def test_no_overwrite(self, patched_ref_enrichment):
        re_module.write_reference_article(
            title="Test", extract="Original", url="", source="test", domain="test",
        )
        re_module.write_reference_article(
            title="Test", extract="Overwritten", url="", source="test", domain="test",
        )
        refs_dir = patched_ref_enrichment / "Permanent" / "References"
        content = (refs_dir / "Test.md").read_text(encoding="utf-8")
        assert "Original" in content
        assert "Overwritten" not in content

    def test_includes_frontmatter(self, patched_ref_enrichment):
        path = re_module.write_reference_article(
            title="Spaghetti",
            extract="Pasta content",
            url="https://en.wikipedia.org/wiki/Spaghetti",
            source="wikipedia",
            domain="food_cooking",
            original_query="What is spaghetti?",
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "---" in content
        assert 'title: "Spaghetti"' in content
        assert 'original_query: "What is spaghetti?"' in content


# ---------------------------------------------------------------------------
# Import log
# ---------------------------------------------------------------------------


class TestImportLog:
    def test_empty_log(self, patched_ref_enrichment):
        assert re_module._load_import_log() == []

    def test_log_import(self, patched_ref_enrichment):
        re_module._log_import(
            title="Spaghetti", source="wikipedia", domain="food_cooking",
            status="imported", query="What is spaghetti?", filepath="/path",
        )
        log = _read_import_log(patched_ref_enrichment)
        assert len(log) == 1
        assert log[0]["title"] == "Spaghetti"
        assert log[0]["status"] == "imported"

    def test_appends_entries(self, patched_ref_enrichment):
        re_module._log_import(title="A", source="s", domain="d", status="imported")
        re_module._log_import(title="B", source="s", domain="d", status="no_source")
        log = _read_import_log(patched_ref_enrichment)
        assert len(log) == 2


# ---------------------------------------------------------------------------
# fetch_reference (mocked)
# ---------------------------------------------------------------------------


class TestFetchReference:
    def test_wikipedia_success(self):
        with patch.object(re_module, "_fetch_wikipedia_article", return_value=MOCK_WIKI_ARTICLE):
            result = re_module.fetch_reference("spaghetti", source="wikipedia")
        assert result is not None
        assert result["title"] == "Spaghetti"
        assert result["source"] == "wikipedia"

    def test_wikipedia_failure(self):
        with patch.object(re_module, "_fetch_wikipedia_article", return_value=None):
            result = re_module.fetch_reference("xyzzy", source="wikipedia")
        assert result is None

    def test_unknown_source(self):
        result = re_module.fetch_reference("spaghetti", source="nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# enrich_from_query
# ---------------------------------------------------------------------------


class TestEnrichFromQuery:
    def test_imports_new_article(self, patched_ref_enrichment):
        mock_ref = dict(MOCK_WIKI_ARTICLE)
        mock_ref["source"] = "wikipedia"

        with patch.object(re_module, "fetch_reference", return_value=mock_ref), \
             patch.object(re_module, "add_fact"):
            result = re_module.enrich_from_query("What is spaghetti?")

        assert result["status"] == "imported"
        assert result["title"] == "Spaghetti"
        refs_dir = patched_ref_enrichment / "Permanent" / "References"
        assert (refs_dir / "Spaghetti.md").exists()

    def test_already_exists(self, patched_ref_enrichment):
        # Write an existing article
        re_module.write_reference_article(
            title="Spaghetti", extract="Existing", url="", source="test", domain="test",
        )

        mock_ref = dict(MOCK_WIKI_ARTICLE)
        mock_ref["source"] = "wikipedia"

        with patch.object(re_module, "fetch_reference", return_value=mock_ref):
            result = re_module.enrich_from_query("What is spaghetti?")

        assert result["status"] == "already_exists"

    def test_no_source(self, patched_ref_enrichment):
        with patch.object(re_module, "fetch_reference", return_value=None):
            result = re_module.enrich_from_query("xyzzy gibberish nonsense")

        assert result["status"] == "no_source"

    def test_logs_import(self, patched_ref_enrichment):
        mock_ref = dict(MOCK_WIKI_ARTICLE)
        mock_ref["source"] = "wikipedia"

        with patch.object(re_module, "fetch_reference", return_value=mock_ref), \
             patch.object(re_module, "add_fact"):
            re_module.enrich_from_query("What is spaghetti?")

        log = _read_import_log(patched_ref_enrichment)
        assert len(log) == 1
        assert log[0]["status"] == "imported"


# ---------------------------------------------------------------------------
# enrich_from_gap_resolutions
# ---------------------------------------------------------------------------


class TestEnrichFromGapResolutions:
    def test_no_resolutions(self, patched_ref_enrichment):
        with patch("agent.gap_resolver._load_resolutions", return_value=[]):
            results = re_module.enrich_from_gap_resolutions()
        assert results == []

    def test_writes_articles_for_accepted(self, patched_ref_enrichment):
        accepted = [{
            "query": "What is spaghetti?",
            "gap_type": "uncertainty",
            "domain": "food_cooking",
            "status": "accepted",
            "suggestion": MOCK_WIKI_ARTICLE,
        }]

        with patch("agent.gap_resolver._load_resolutions", return_value=accepted):
            results = re_module.enrich_from_gap_resolutions()

        assert len(results) == 1
        assert results[0]["status"] == "imported"
        refs_dir = patched_ref_enrichment / "Permanent" / "References"
        assert (refs_dir / "Spaghetti.md").exists()


# ---------------------------------------------------------------------------
# enrich_from_frequent_gaps
# ---------------------------------------------------------------------------


class TestEnrichFromFrequentGaps:
    def test_no_gaps(self, patched_ref_enrichment):
        with patch("agent.gap_frequency._parse_all_gaps", return_value=[]):
            results = re_module.enrich_from_frequent_gaps()
        assert results == []

    def test_enriches_recurring_clusters(self, patched_ref_enrichment):
        clusters = [
            {"label": "spaghetti", "domain": "food_cooking", "count": 3,
             "gap_types": {"uncertainty": 3}, "queries": ["What is spaghetti?"],
             "first_seen": "2026-04-01", "last_seen": "2026-04-05"},
        ]
        mock_ref = dict(MOCK_WIKI_ARTICLE)
        mock_ref["source"] = "wikipedia"

        with patch("agent.gap_frequency._parse_all_gaps", return_value=[{"query": "x", "gap_type": "uncertainty", "timestamp": "2026-04-01 10:00"}]), \
             patch("agent.gap_frequency.cluster_gaps", return_value=clusters), \
             patch.object(re_module, "fetch_reference", return_value=mock_ref), \
             patch.object(re_module, "add_fact"):
            results = re_module.enrich_from_frequent_gaps(top_n=3)

        assert len(results) == 1
        assert results[0]["status"] == "imported"

    def test_skips_single_occurrence(self, patched_ref_enrichment):
        clusters = [
            {"label": "test", "domain": "test", "count": 1,
             "gap_types": {"uncertainty": 1}, "queries": ["test query"],
             "first_seen": "", "last_seen": ""},
        ]

        with patch("agent.gap_frequency._parse_all_gaps", return_value=[{"query": "x", "gap_type": "uncertainty", "timestamp": ""}]), \
             patch("agent.gap_frequency.cluster_gaps", return_value=clusters):
            results = re_module.enrich_from_frequent_gaps()

        assert results == []


# ---------------------------------------------------------------------------
# get_import_metrics
# ---------------------------------------------------------------------------


class TestGetImportMetrics:
    def test_no_data(self, patched_ref_enrichment):
        result = re_module.get_import_metrics()
        assert "No imports recorded" in result

    def test_with_data(self, patched_ref_enrichment):
        _write_import_log(patched_ref_enrichment, [
            {"timestamp": "2026-04-01 10:00", "title": "A", "source": "wikipedia", "domain": "food_cooking", "status": "imported", "query": "", "filepath": "", "error": ""},
            {"timestamp": "2026-04-02 10:00", "title": "B", "source": "wikipedia", "domain": "science", "status": "imported", "query": "", "filepath": "", "error": ""},
            {"timestamp": "2026-04-03 10:00", "title": "", "source": "wikipedia", "domain": "history", "status": "no_source", "query": "q", "filepath": "", "error": ""},
        ])
        result = re_module.get_import_metrics()
        assert "Total attempts | 3" in result
        assert "Successfully imported | 2" in result
        assert "No source found | 1" in result
        assert "Success rate | 67%" in result


# ---------------------------------------------------------------------------
# list_reference_articles
# ---------------------------------------------------------------------------


class TestListReferenceArticles:
    def test_no_articles(self, patched_ref_enrichment):
        result = re_module.list_reference_articles()
        assert "No reference articles" in result

    def test_lists_articles(self, patched_ref_enrichment):
        re_module.write_reference_article(
            title="Spaghetti", extract="Pasta", url="", source="test", domain="test",
        )
        re_module.write_reference_article(
            title="Pizza", extract="Flatbread", url="", source="test", domain="test",
        )
        result = re_module.list_reference_articles()
        assert "Reference Articles (2)" in result
        assert "Spaghetti" in result
        assert "Pizza" in result


# ---------------------------------------------------------------------------
# _tool_import_reference
# ---------------------------------------------------------------------------


class TestToolImportReference:
    def test_imported(self, patched_ref_enrichment):
        mock_ref = dict(MOCK_WIKI_ARTICLE)
        mock_ref["source"] = "wikipedia"
        with patch.object(re_module, "fetch_reference", return_value=mock_ref), \
             patch.object(re_module, "add_fact"):
            result = re_module._tool_import_reference("spaghetti")
        assert "Imported" in result
        assert "Spaghetti" in result

    def test_no_source(self, patched_ref_enrichment):
        with patch.object(re_module, "fetch_reference", return_value=None):
            result = re_module._tool_import_reference("xyzzy gibberish")
        assert "No reference source found" in result


# ---------------------------------------------------------------------------
# _domain_to_facts_category
# ---------------------------------------------------------------------------


class TestDomainToFactsCategory:
    def test_geography(self):
        assert re_module._domain_to_facts_category("geography") == "geography"

    def test_default(self):
        assert re_module._domain_to_facts_category("food_cooking") == "definition"


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestScheduling:
    def test_returns_positive(self):
        assert re_module._seconds_until_next_run() > 0

    def test_within_24_hours(self):
        assert re_module._seconds_until_next_run() <= 24 * 3600
