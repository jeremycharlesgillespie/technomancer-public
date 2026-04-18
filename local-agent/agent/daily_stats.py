"""
Daily Stats DB — SQLite persistence for per-day, per-project aggregate metrics.

Single source of truth for downstream consumers — rollup jobs, the hub's CSV
export, and PPTX slide generators — so they don't each re-derive the same
numbers from raw Jira + git + executor logs.

One row per ``(date, project)``. Columns cover throughput (shipped, failed,
split_children), spend (cost_usd), latency (p50/p95 wall-clock seconds),
code churn (loc_added/removed), and quality signals (first-attempt success
rate, splitter child outcomes).

Database: ``local-agent/data/daily_stats.db``

This module is schema-only — no other module imports it yet. Follow-up
stories will add the rollup writer and reader paths.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "daily_stats.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection (created on first use)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create the ``daily_stats`` table if missing. Idempotent."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date                   TEXT    NOT NULL,
            project                TEXT    NOT NULL,
            shipped                INTEGER NOT NULL DEFAULT 0,
            failed                 INTEGER NOT NULL DEFAULT 0,
            split_children         INTEGER NOT NULL DEFAULT 0,
            cost_usd               REAL    NOT NULL DEFAULT 0.0,
            p50_wall_s             REAL    NOT NULL DEFAULT 0.0,
            p95_wall_s             REAL    NOT NULL DEFAULT 0.0,
            loc_added              INTEGER NOT NULL DEFAULT 0,
            loc_removed            INTEGER NOT NULL DEFAULT 0,
            first_attempt_success  INTEGER NOT NULL DEFAULT 0,
            splitter_child_success INTEGER NOT NULL DEFAULT 0,
            splitter_child_fail    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (date, project)
        )
    """)
    conn.commit()
