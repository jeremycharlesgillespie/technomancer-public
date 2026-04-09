"""Tests for fact auto-enrichment: confidence scoring and real-time gap filling."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agent.facts_db import add_fact, init_db, lookup_fact, source_confidence


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirect facts_db to a temp directory."""
    monkeypatch.setattr("agent.facts_db.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.facts_db.DB_PATH", tmp_path / "facts.db")
    import agent.facts_db as mod
    if hasattr(mod._local, "conn"):
        del mod._local.conn
    init_db()


class TestSourceConfidence:
    def test_seed_highest(self):
        assert source_confidence("seed") == 1.0

    def test_user_high(self):
        assert source_confidence("user") == 0.95

    def test_wikipedia_high(self):
        assert source_confidence("wikipedia") == 0.9

    def test_auto_enrichment_medium(self):
        assert source_confidence("auto_enrichment") == 0.8

    def test_web_search_lower(self):
        assert source_confidence("web_search") == 0.6

    def test_unknown_source_default(self):
        assert source_confidence("random_source") == 0.7


class TestAddFactWithConfidence:
    def test_default_confidence_from_source(self):
        add_fact("definition", "test_key", "test value", source="wikipedia")
        results = lookup_fact("test_key")
        assert len(results) == 1
        assert results[0]["confidence"] == 0.9

    def test_explicit_confidence(self):
        add_fact("definition", "test_key2", "value", source="user", confidence=0.5)
        results = lookup_fact("test_key2")
        assert results[0]["confidence"] == 0.5

    def test_seed_confidence(self):
        add_fact("definition", "coffee", "A beverage", source="seed")
        results = lookup_fact("coffee")
        assert results[0]["confidence"] == 1.0

    def test_web_search_confidence(self):
        add_fact("definition", "obscure_term", "Something", source="web_search")
        results = lookup_fact("obscure_term")
        assert results[0]["confidence"] == 0.6

    def test_confidence_in_return_message(self):
        msg = add_fact("test", "key", "val", source="seed")
        assert "confidence" in msg.lower()
        assert "100%" in msg


class TestLookupReturnsConfidence:
    def test_exact_match_includes_confidence(self):
        add_fact("geography", "france capital", "Paris", source="seed")
        results = lookup_fact("france capital")
        assert "confidence" in results[0]
        assert results[0]["confidence"] == 1.0

    def test_like_match_includes_confidence(self):
        add_fact("definition", "spaghetti pasta", "Italian noodle", source="wikipedia")
        results = lookup_fact("spaghetti")
        assert len(results) >= 1
        assert "confidence" in results[0]


class TestMigrateExistingDb:
    def test_migration_adds_column(self, tmp_path):
        """Existing databases without confidence column get migrated."""
        # Create a DB without the confidence column
        db_path = tmp_path / "old_facts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'seed',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(category, key)
            )
        """)
        conn.execute(
            "INSERT INTO facts (category, key, value) VALUES ('test', 'old_key', 'old_value')"
        )
        conn.commit()
        conn.close()

        # Point facts_db at this old database
        import agent.facts_db as mod
        if hasattr(mod._local, "conn"):
            del mod._local.conn
        mod.DB_PATH = db_path
        mod.DB_DIR = tmp_path

        # init_db should migrate
        init_db()
        results = lookup_fact("old_key")
        assert len(results) == 1
        assert results[0]["confidence"] == 1.0  # default from migration


class TestAutoEnrichGap:
    @patch("agent.knowledge_enrichment._try_enrich_query")
    def test_enriches_on_gap(self, mock_try):
        mock_try.return_value = {
            "status": "enriched",
            "title": "spaghetti",
            "source": "wikipedia",
            "category": "definition",
        }

        from agent.knowledge_gaps import auto_enrich_gap

        # Mock resolve_gap since we don't have the vault file
        with patch("agent.knowledge_gaps.resolve_gap", return_value="Resolved"):
            result = auto_enrich_gap({
                "query": "What is spaghetti?",
                "gap_type": "failure",
                "response_snippet": "I don't know",
                "timestamp": "2026-04-09 15:00",
            })

        assert result is not None
        assert "spaghetti" in result.lower()
        assert "wikipedia" in result.lower()

    @patch("agent.knowledge_enrichment._try_enrich_query")
    def test_returns_none_on_no_source(self, mock_try):
        mock_try.return_value = {"status": "no_source", "title": ""}

        from agent.knowledge_gaps import auto_enrich_gap

        result = auto_enrich_gap({
            "query": "xyznonexistent12345",
            "gap_type": "failure",
            "response_snippet": "...",
            "timestamp": "2026-04-09 15:00",
        })
        assert result is None

    def test_returns_none_on_short_query(self):
        from agent.knowledge_gaps import auto_enrich_gap

        result = auto_enrich_gap({"query": "hi", "gap_type": "failure"})
        assert result is None

    def test_returns_none_on_empty_query(self):
        from agent.knowledge_gaps import auto_enrich_gap

        result = auto_enrich_gap({"query": "", "gap_type": "failure"})
        assert result is None

    @patch("agent.knowledge_enrichment._try_enrich_query", side_effect=Exception("boom"))
    def test_handles_error_gracefully(self, mock_try):
        from agent.knowledge_gaps import auto_enrich_gap

        # Should not raise
        result = auto_enrich_gap({
            "query": "something that breaks",
            "gap_type": "failure",
            "response_snippet": "...",
            "timestamp": "2026-04-09 15:00",
        })
        assert result is None
