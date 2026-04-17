"""Integration test for GET /api/jira/dlq.

Inserts a row via ``add_dlq_entry`` (with DB_PATH monkeypatched to a temp
SQLite file), hits the endpoint via the Flask test client, and asserts
the response contains the inserted row with a parsed payload.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from idea_board import jira_sync_dlq
from idea_board.web import app


@pytest.fixture
def isolated_dlq(tmp_path, monkeypatch):
    """Point jira_sync_dlq at a temporary SQLite DB for a single test.

    The module keeps a per-thread connection in ``_local.conn``; we have
    to drop it before and after the test so the patched DB_PATH actually
    takes effect and so we don't leak state to later tests.
    """
    db_path = tmp_path / "jira_sync_dlq.db"
    monkeypatch.setattr(jira_sync_dlq, "DB_DIR", tmp_path)
    monkeypatch.setattr(jira_sync_dlq, "DB_PATH", db_path)

    def _drop_conn():
        conn = getattr(jira_sync_dlq._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        jira_sync_dlq._local.__dict__.pop("conn", None)

    _drop_conn()
    yield
    _drop_conn()


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestJiraDlqEndpoint:
    def test_empty_when_no_entries(self, client, isolated_dlq):
        """Returns an empty entries list when the DLQ has no rows."""
        resp = client.get("/api/jira/dlq")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"entries": []}

    def test_returns_inserted_row_with_parsed_payload(self, client, isolated_dlq):
        """A row written via add_dlq_entry surfaces on the endpoint."""
        payload = {"summary": "sync failed", "fields": {"priority": "High"}}
        rowid = jira_sync_dlq.add_dlq_entry(
            idea_id="TK-999",
            payload=payload,
            error="HTTP 500: Internal Server Error",
            attempts=5,
        )
        assert rowid >= 1

        resp = client.get("/api/jira/dlq")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "entries" in data
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) == 1

        entry = data["entries"][0]
        assert entry["id"] == rowid
        assert entry["idea_id"] == "TK-999"
        assert entry["error"] == "HTTP 500: Internal Server Error"
        assert entry["attempts"] == 5
        # Payload is parsed back into a dict (not just the JSON string).
        assert entry["payload"] == payload
        # The raw JSON column is preserved too.
        assert json.loads(entry["payload_json"]) == payload
        assert entry["first_failed_at"]
        assert entry["last_failed_at"]

    def test_newest_first_ordering(self, client, isolated_dlq):
        """Multiple entries come back newest-id first, capped at 100."""
        for i in range(3):
            jira_sync_dlq.add_dlq_entry(
                idea_id=f"TK-{i}",
                payload={"i": i},
                error=f"err-{i}",
                attempts=i + 1,
            )

        resp = client.get("/api/jira/dlq")
        assert resp.status_code == 200
        entries = resp.get_json()["entries"]
        assert len(entries) == 3
        # Newest first: last insert (i=2) shows up first.
        assert [e["idea_id"] for e in entries] == ["TK-2", "TK-1", "TK-0"]
        assert entries[0]["payload"] == {"i": 2}
