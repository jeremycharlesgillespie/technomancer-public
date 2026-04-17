"""Tests for idea_board.jira_sync_dlq — SQLite dead-letter queue."""

from __future__ import annotations

import json
import sqlite3

import pytest

from idea_board import jira_sync_dlq


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point jira_sync_dlq at a temporary SQLite DB for each test."""
    db_path = tmp_path / "jira_sync_dlq.db"
    monkeypatch.setattr(jira_sync_dlq, "DB_DIR", tmp_path)
    monkeypatch.setattr(jira_sync_dlq, "DB_PATH", db_path)
    # Drop any cached per-thread connection from prior tests so the new
    # DB_PATH actually takes effect on the next _get_conn() call.
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    jira_sync_dlq._local.__dict__.pop("conn", None)
    yield
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        jira_sync_dlq._local.__dict__.pop("conn", None)


class TestInitDb:
    def test_creates_table(self):
        jira_sync_dlq.init_db()
        conn = jira_sync_dlq._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='jira_sync_dlq'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        """Calling init_db repeatedly must not raise or duplicate the table."""
        jira_sync_dlq.init_db()
        jira_sync_dlq.init_db()
        jira_sync_dlq.init_db()
        conn = jira_sync_dlq._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='jira_sync_dlq'"
        ).fetchall()
        assert len(tables) == 1

    def test_schema_columns_match_spec(self):
        """Schema must have id, idea_id, payload_json, error, attempts,
        first_failed_at, last_failed_at."""
        jira_sync_dlq.init_db()
        conn = jira_sync_dlq._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(jira_sync_dlq)").fetchall()
        }
        expected = {
            "id",
            "idea_id",
            "payload_json",
            "error",
            "attempts",
            "first_failed_at",
            "last_failed_at",
        }
        assert expected == cols


class TestAddDlqEntry:
    def test_inserts_exactly_one_row(self):
        jira_sync_dlq.add_dlq_entry(
            idea_id="TK-507",
            payload={"summary": "test", "fields": {"status": "todo"}},
            error="HTTP 500: Internal Server Error",
            attempts=5,
        )
        conn = jira_sync_dlq._get_conn()
        rows = conn.execute("SELECT * FROM jira_sync_dlq").fetchall()
        assert len(rows) == 1

    def test_columns_populated_correctly(self):
        payload = {"summary": "Sync failed", "fields": {"priority": "High"}}
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-999",
            payload=payload,
            error="connection refused",
            attempts=3,
        )
        assert rowid >= 1

        conn = jira_sync_dlq._get_conn()
        row = conn.execute(
            "SELECT * FROM jira_sync_dlq WHERE id = ?", (rowid,)
        ).fetchone()

        assert row["idea_id"] == "TK-999"
        assert json.loads(row["payload_json"]) == payload
        assert row["error"] == "connection refused"
        assert row["attempts"] == 3
        assert row["first_failed_at"]
        assert row["last_failed_at"]
        # On insert, both timestamps are set to "now" simultaneously.
        assert row["first_failed_at"] == row["last_failed_at"]

    def test_calls_init_db_lazily(self, tmp_path, monkeypatch):
        """add_dlq_entry should work even if init_db wasn't called explicitly."""
        # Repoint to a fresh DB and drop the cached connection so the
        # table doesn't exist yet at call time.
        new_db = tmp_path / "fresh.db"
        monkeypatch.setattr(jira_sync_dlq, "DB_PATH", new_db)
        conn = getattr(jira_sync_dlq._local, "conn", None)
        if conn is not None:
            conn.close()
        jira_sync_dlq._local.__dict__.pop("conn", None)

        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-1",
            payload={"x": 1},
            error="boom",
            attempts=1,
        )
        assert rowid >= 1
        rows = jira_sync_dlq._get_conn().execute(
            "SELECT COUNT(*) FROM jira_sync_dlq"
        ).fetchone()
        assert rows[0] == 1

    def test_non_serializable_payload_falls_back(self):
        """Stray non-JSON values must not block the insert."""

        class Thing:
            def __repr__(self) -> str:
                return "<Thing>"

        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-2",
            payload=Thing(),
            error="serialize-test",
            attempts=1,
        )
        conn = jira_sync_dlq._get_conn()
        row = conn.execute(
            "SELECT payload_json FROM jira_sync_dlq WHERE id = ?", (rowid,)
        ).fetchone()
        # default=str converts unknown objects, so this should be the repr.
        assert "Thing" in row["payload_json"]
