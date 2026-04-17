"""Jira sync dead-letter queue — SQLite-backed record of failed Jira writes.

Each row captures a payload that could not be delivered to Jira after all
retries, along with the error, the number of attempts spent, and the
first/last failure timestamps so recurring failures can be deduplicated
in a follow-up story.

Database: ``local-agent/data/jira_sync_dlq.db``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "jira_sync_dlq.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (created on first use)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create the jira_sync_dlq table if it doesn't exist. Idempotent."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jira_sync_dlq (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id         TEXT    NOT NULL,
            payload_json    TEXT    NOT NULL,
            error           TEXT    NOT NULL,
            attempts        INTEGER NOT NULL,
            first_failed_at TEXT    NOT NULL,
            last_failed_at  TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jira_sync_dlq_idea_id
        ON jira_sync_dlq (idea_id)
    """)
    conn.commit()


def add_dlq_entry(
    idea_id: str,
    payload: Any,
    error: str,
    attempts: int,
) -> int:
    """Insert a new dead-letter row and return its rowid.

    ``payload`` is serialized to JSON; non-serializable values fall back to
    ``str()`` via ``default=str`` so a stray object never blocks the insert.
    On insert, ``first_failed_at`` and ``last_failed_at`` are both set to
    the current UTC time.
    """
    init_db()
    conn = _get_conn()

    payload_json = json.dumps(payload, default=str)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO jira_sync_dlq
            (idea_id, payload_json, error, attempts, first_failed_at, last_failed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (idea_id, payload_json, error, int(attempts), now, now),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def get_jira_dlq_entries(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent dead-letter entries, newest first.

    Each entry is a dict with the row columns plus a ``payload`` field
    holding the parsed JSON payload. If ``payload_json`` cannot be parsed
    (shouldn't happen, but guard anyway), ``payload`` is ``None`` and the
    raw string is preserved in ``payload_json``.
    """
    init_db()
    conn = _get_conn()
    try:
        safe_limit = max(1, int(limit))
    except (TypeError, ValueError):
        safe_limit = 100

    rows = conn.execute(
        """
        SELECT id, idea_id, payload_json, error, attempts,
               first_failed_at, last_failed_at
        FROM jira_sync_dlq
        ORDER BY id DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()

    entries: list[dict[str, Any]] = []
    for row in rows:
        raw_payload = row["payload_json"]
        try:
            parsed = json.loads(raw_payload) if raw_payload else None
        except (TypeError, ValueError):
            parsed = None
        entries.append({
            "id": row["id"],
            "idea_id": row["idea_id"],
            "payload_json": raw_payload,
            "payload": parsed,
            "error": row["error"],
            "attempts": row["attempts"],
            "first_failed_at": row["first_failed_at"],
            "last_failed_at": row["last_failed_at"],
        })
    return entries
