"""Tests for the facts_db module — SQLite-backed reference facts database."""

import sqlite3

import pytest

from agent import facts_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point facts_db at a temporary SQLite DB for each test."""
    db_path = tmp_path / "facts.db"
    monkeypatch.setattr(facts_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(facts_db, "DB_PATH", db_path)
    # Clear any cached per-thread connection
    facts_db._local.__dict__.pop("conn", None)
    facts_db.init_db()
    yield
    conn = getattr(facts_db._local, "conn", None)
    if conn:
        conn.close()
        facts_db._local.conn = None


class TestWalMode:
    """WAL journal mode must be active so readers don't block on writers."""

    def test_journal_mode_is_wal(self):
        conn = facts_db._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_concurrent_read_during_open_write_transaction(self):
        """Separate reader connection must succeed while a writer holds an
        open write transaction — the contention scenario that causes
        `database is locked` under the default rollback journal."""
        facts_db.add_fact("test", "seed", "v", source="user")
        conn = getattr(facts_db._local, "conn", None)
        if conn:
            conn.close()
            facts_db._local.conn = None

        writer = sqlite3.connect(str(facts_db.DB_PATH), timeout=5)
        reader = sqlite3.connect(str(facts_db.DB_PATH), timeout=5)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO facts (category, key, value, source) "
                "VALUES (?, ?, ?, ?)",
                ("test", "concurrent", "v", "test"),
            )
            count = reader.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
            assert count >= 1
            writer.rollback()
        finally:
            writer.close()
            reader.close()


class TestInitDb:
    """Test database initialization."""

    def test_creates_facts_table(self):
        conn = facts_db._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
        )
        assert cursor.fetchone() is not None

    def test_creates_fts_table(self):
        conn = facts_db._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_fts'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent(self):
        facts_db.init_db()
        facts_db.init_db()
        conn = facts_db._get_conn()
        tables = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sqlite_master WHERE name='facts'"
        ).fetchone()
        assert tables["cnt"] == 1


class TestSeedDb:
    """Test seeding with reference data."""

    def test_seed_inserts_data(self):
        facts_db.seed_db()
        stats = facts_db.get_stats()
        assert stats["total_facts"] > 0

    def test_seed_has_definitions(self):
        facts_db.seed_db()
        results = facts_db.lookup_fact("spaghetti", category="definition")
        assert len(results) >= 1
        assert "pasta" in results[0]["value"].lower() or "italian" in results[0]["value"].lower()

    def test_seed_has_geography(self):
        facts_db.seed_db()
        results = facts_db.lookup_fact("United States capital", category="geography")
        assert len(results) >= 1
        assert "Washington" in results[0]["value"]

    def test_seed_has_timezones(self):
        facts_db.seed_db()
        results = facts_db.lookup_fact("EST", category="timezone")
        assert len(results) >= 1
        assert "UTC" in results[0]["value"]

    def test_seed_has_conversions(self):
        facts_db.seed_db()
        results = facts_db.lookup_fact("miles to kilometers", category="conversion")
        assert len(results) >= 1
        assert "1.609" in results[0]["value"]

    def test_seed_idempotent(self):
        facts_db.seed_db()
        count1 = facts_db.get_stats()["total_facts"]
        facts_db.seed_db()
        count2 = facts_db.get_stats()["total_facts"]
        assert count1 == count2


class TestLookupFact:
    """Test fact lookup with various strategies."""

    def test_exact_key_match(self):
        facts_db.add_fact("definition", "test_thing", "A thing used in tests.")
        results = facts_db.lookup_fact("test_thing")
        assert len(results) == 1
        assert results[0]["value"] == "A thing used in tests."

    def test_exact_key_case_insensitive(self):
        facts_db.add_fact("definition", "CamelCase", "A naming convention.")
        results = facts_db.lookup_fact("camelcase")
        assert len(results) >= 1

    def test_category_filter(self):
        facts_db.add_fact("definition", "mercury", "A chemical element (Hg).")
        facts_db.add_fact("geography", "mercury", "Not a real place, but a planet.")
        results = facts_db.lookup_fact("mercury", category="definition")
        assert len(results) == 1
        assert results[0]["category"] == "definition"

    def test_like_fallback(self):
        facts_db.add_fact("definition", "machine learning", "A subset of AI.")
        results = facts_db.lookup_fact("machine")
        assert len(results) >= 1

    def test_no_results(self):
        results = facts_db.lookup_fact("xyznonexistent123")
        assert results == []


class TestAddFact:
    """Test adding facts."""

    def test_add_new_fact(self):
        result = facts_db.add_fact("definition", "widget", "A small gadget.")
        assert "Saved fact" in result
        results = facts_db.lookup_fact("widget")
        assert len(results) == 1
        assert results[0]["value"] == "A small gadget."

    def test_update_existing_fact(self):
        facts_db.add_fact("definition", "widget", "A small gadget.")
        facts_db.add_fact("definition", "widget", "An updated gadget description.")
        results = facts_db.lookup_fact("widget")
        assert len(results) == 1
        assert "updated" in results[0]["value"].lower()

    def test_add_with_source(self):
        facts_db.add_fact("definition", "custom_term", "Custom definition.", source="manual")
        conn = facts_db._get_conn()
        row = conn.execute(
            "SELECT source FROM facts WHERE key = 'custom_term'"
        ).fetchone()
        assert row["source"] == "manual"


class TestGetCategories:
    """Test category listing."""

    def test_empty_db(self):
        cats = facts_db.get_categories()
        assert cats == []

    def test_with_data(self):
        facts_db.add_fact("definition", "a", "val")
        facts_db.add_fact("geography", "b", "val")
        facts_db.add_fact("timezone", "c", "val")
        cats = facts_db.get_categories()
        assert set(cats) == {"definition", "geography", "timezone"}


class TestGetStats:
    """Test statistics."""

    def test_empty_stats(self):
        stats = facts_db.get_stats()
        assert stats["total_facts"] == 0
        assert stats["categories"] == {}

    def test_stats_after_seed(self):
        facts_db.seed_db()
        stats = facts_db.get_stats()
        assert stats["total_facts"] == len(facts_db._SEED_DATA)
        assert "definition" in stats["categories"]
        assert "geography" in stats["categories"]
        assert "timezone" in stats["categories"]
        assert "conversion" in stats["categories"]


class TestToolWrappers:
    """Test the tool interface functions."""

    def test_tool_lookup_found(self):
        facts_db.add_fact("definition", "widget", "A small gadget.")
        result = facts_db._tool_lookup_fact("widget")
        assert "widget" in result
        assert "small gadget" in result

    def test_tool_lookup_not_found(self):
        result = facts_db._tool_lookup_fact("nonexistent_xyz")
        assert "No facts found" in result

    def test_tool_add_fact(self):
        result = facts_db._tool_add_fact("definition", "gizmo", "A fancy gadget.")
        assert "Saved fact" in result

    def test_tool_stats(self):
        facts_db.seed_db()
        result = facts_db._tool_facts_stats()
        assert "total_facts" in result

    def test_get_facts_tools_returns_tools(self):
        tools = facts_db.get_facts_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"lookup_fact", "add_fact", "facts_stats"}


    # TestKnowledgeGraphPreRetrieval removed — knowledge.py was deleted as legacy code
