"""Tests for idea_board.jira_sync_dlq — SQLite dead-letter queue."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

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


def _set_last_failed_at(entry_id: int, value: str) -> None:
    """Directly set ``last_failed_at`` on an existing DLQ row.

    Lets tests simulate distinct timestamps without waiting for the
    clock to tick.
    """
    conn = jira_sync_dlq._get_conn()
    conn.execute(
        "UPDATE jira_sync_dlq SET last_failed_at = ? WHERE id = ?",
        (value, int(entry_id)),
    )
    conn.commit()


class TestGetJiraDlqEntries:
    def test_empty_db_returns_empty_list(self):
        """With no rows, get_jira_dlq_entries returns []."""
        assert jira_sync_dlq.get_jira_dlq_entries() == []

    def test_orders_by_last_failed_at_desc(self):
        """Rows come back newest-first by last_failed_at."""
        # Insert three rows, then mutate last_failed_at so the most
        # recently failed row is the one inserted FIRST. This proves
        # we're ordering by last_failed_at, not by id.
        a = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-A", payload={"x": "a"}, error="e", attempts=1
        )
        b = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-B", payload={"x": "b"}, error="e", attempts=1
        )
        c = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-C", payload={"x": "c"}, error="e", attempts=1
        )
        _set_last_failed_at(a, "2026-04-17T12:00:03+00:00")
        _set_last_failed_at(b, "2026-04-17T12:00:02+00:00")
        _set_last_failed_at(c, "2026-04-17T12:00:01+00:00")

        entries = jira_sync_dlq.get_jira_dlq_entries()
        assert [e["idea_id"] for e in entries] == ["TK-A", "TK-B", "TK-C"]

    def test_limit_is_honoured(self):
        """Only ``limit`` rows come back, still newest-first."""
        for i in range(5):
            jira_sync_dlq.add_dlq_entry(
                idea_id=f"TK-{i}",
                payload={"i": i},
                error=f"err-{i}",
                attempts=1,
            )
        entries = jira_sync_dlq.get_jira_dlq_entries(limit=2)
        assert len(entries) == 2
        # Highest id wins the same-timestamp tiebreaker.
        assert [e["idea_id"] for e in entries] == ["TK-4", "TK-3"]

    def test_payload_is_parsed_into_dict(self):
        """payload_json is parsed back into ``payload`` as a dict."""
        payload = {"summary": "hi", "fields": {"priority": "High"}}
        jira_sync_dlq.add_dlq_entry(
            idea_id="TK-42", payload=payload, error="x", attempts=1
        )
        [entry] = jira_sync_dlq.get_jira_dlq_entries()
        assert entry["payload"] == payload
        assert json.loads(entry["payload_json"]) == payload


class TestRetryJiraDlqEntry:
    def test_unknown_entry_id_returns_false(self):
        """Retrying an id that doesn't exist returns False."""
        assert jira_sync_dlq.retry_jira_dlq_entry(9999) is False

    def test_success_deletes_row(self):
        """A successful sync deletes the DLQ row and returns True."""
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-100",
            payload={"summary": "retry me"},
            error="transient",
            attempts=2,
        )
        fake_idea = MagicMock(id="TK-100")
        with patch.object(
            jira_sync_dlq, "get_idea", return_value=fake_idea
        ) as gi, patch.object(
            jira_sync_dlq, "sync_idea_to_jira", return_value="TK-100"
        ) as sync:
            ok = jira_sync_dlq.retry_jira_dlq_entry(rowid)

        assert ok is True
        gi.assert_called_once_with("TK-100")
        sync.assert_called_once_with(fake_idea)

        conn = jira_sync_dlq._get_conn()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM jira_sync_dlq WHERE id = ?", (rowid,)
        ).fetchone()
        assert remaining[0] == 0

    def test_failure_leaves_row_and_updates_last_failed_at(self):
        """A failed sync must leave the row and bump last_failed_at."""
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-200",
            payload={"summary": "still broken"},
            error="connection refused",
            attempts=3,
        )
        # Pin the original timestamp to a clearly-older value so we can
        # tell whether the retry bumped it forward.
        _set_last_failed_at(rowid, "2020-01-01T00:00:00+00:00")

        fake_idea = MagicMock(id="TK-200")
        with patch.object(
            jira_sync_dlq, "get_idea", return_value=fake_idea
        ), patch.object(
            jira_sync_dlq, "sync_idea_to_jira", return_value=None
        ):
            ok = jira_sync_dlq.retry_jira_dlq_entry(rowid)

        assert ok is False
        conn = jira_sync_dlq._get_conn()
        row = conn.execute(
            "SELECT idea_id, last_failed_at, first_failed_at "
            "FROM jira_sync_dlq WHERE id = ?",
            (rowid,),
        ).fetchone()
        assert row is not None
        assert row["idea_id"] == "TK-200"
        assert row["last_failed_at"] != "2020-01-01T00:00:00+00:00"
        # first_failed_at must not move.
        assert row["last_failed_at"] >= row["first_failed_at"]

    def test_sync_exception_treated_as_failure(self):
        """If sync raises, the row stays and last_failed_at is updated."""
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-300",
            payload={"summary": "raises"},
            error="boom",
            attempts=1,
        )
        _set_last_failed_at(rowid, "2020-01-01T00:00:00+00:00")

        fake_idea = MagicMock(id="TK-300")
        with patch.object(
            jira_sync_dlq, "get_idea", return_value=fake_idea
        ), patch.object(
            jira_sync_dlq, "sync_idea_to_jira",
            side_effect=RuntimeError("kaboom"),
        ):
            ok = jira_sync_dlq.retry_jira_dlq_entry(rowid)

        assert ok is False
        conn = jira_sync_dlq._get_conn()
        row = conn.execute(
            "SELECT last_failed_at FROM jira_sync_dlq WHERE id = ?",
            (rowid,),
        ).fetchone()
        assert row is not None
        assert row["last_failed_at"] != "2020-01-01T00:00:00+00:00"

    def test_missing_idea_leaves_row(self):
        """If get_idea returns None, sync is never called and row stays."""
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-GONE",
            payload={"summary": "idea deleted"},
            error="x",
            attempts=1,
        )
        with patch.object(
            jira_sync_dlq, "get_idea", return_value=None
        ), patch.object(
            jira_sync_dlq, "sync_idea_to_jira"
        ) as sync:
            ok = jira_sync_dlq.retry_jira_dlq_entry(rowid)

        assert ok is False
        sync.assert_not_called()

        conn = jira_sync_dlq._get_conn()
        exists = conn.execute(
            "SELECT COUNT(*) FROM jira_sync_dlq WHERE id = ?", (rowid,)
        ).fetchone()
        assert exists[0] == 1
