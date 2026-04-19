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
            splitter_child_success INTEGER,
            splitter_child_fail    INTEGER,
            phase_timings_json     TEXT,
            PRIMARY KEY (date, project)
        )
    """)
    # Migrate pre-existing databases that don't have phase_timings_json yet.
    # ALTER TABLE raises OperationalError when the column already exists —
    # that's the expected idempotency signal, so swallow it.
    try:
        conn.execute("ALTER TABLE daily_stats ADD COLUMN phase_timings_json TEXT")
    except sqlite3.OperationalError:
        pass
    # Splitter columns were originally NOT NULL DEFAULT 0. TK-618 made them
    # nullable so "Jira unreachable" can be recorded as NULL (distinct from
    # "Jira said zero"). Rebuild the table if a legacy row still has the
    # NOT NULL constraint.
    _relax_splitter_nullability(conn)
    conn.commit()


CSV_COLUMNS: list[str] = [
    "date", "project", "shipped", "failed", "split_children",
    "cost_usd", "p50_wall_s", "p95_wall_s",
    "loc_added", "loc_removed", "first_attempt_success",
    "splitter_child_success", "splitter_child_fail",
    "phase_timings_json",
]


def get_rows(
    since: str | None = None,
    project: str | None = None,
) -> list[dict]:
    """Return daily_stats rows as dicts, ordered by date then project.

    since   — ISO date (YYYY-MM-DD); returns rows where date >= since.
    project — exact project key filter.
    """
    init_db()
    conn = _get_conn()
    query = "SELECT * FROM daily_stats WHERE 1=1"
    params: list[str] = []
    if since:
        query += " AND date >= ?"
        params.append(since)
    if project:
        query += " AND project = ?"
        params.append(project)
    query += " ORDER BY date ASC, project ASC"
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def _relax_splitter_nullability(conn: sqlite3.Connection) -> None:
    """Rebuild daily_stats if splitter columns were created as NOT NULL.

    SQLite can't drop a NOT NULL constraint in place, so we create a
    sibling table with the new schema, copy rows over, and rename.
    No-op when the columns are already nullable.
    """
    info = conn.execute("PRAGMA table_info(daily_stats)").fetchall()
    needs_rebuild = any(
        row[1] in {"splitter_child_success", "splitter_child_fail"} and row[3]
        for row in info
    )
    if not needs_rebuild:
        return
    conn.execute("""
        CREATE TABLE daily_stats_v2 (
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
            splitter_child_success INTEGER,
            splitter_child_fail    INTEGER,
            phase_timings_json     TEXT,
            PRIMARY KEY (date, project)
        )
    """)
    conn.execute("""
        INSERT INTO daily_stats_v2 (
            date, project, shipped, failed, split_children,
            cost_usd, p50_wall_s, p95_wall_s,
            loc_added, loc_removed, first_attempt_success,
            splitter_child_success, splitter_child_fail,
            phase_timings_json
        )
        SELECT
            date, project, shipped, failed, split_children,
            cost_usd, p50_wall_s, p95_wall_s,
            loc_added, loc_removed, first_attempt_success,
            splitter_child_success, splitter_child_fail,
            phase_timings_json
        FROM daily_stats
    """)
    conn.execute("DROP TABLE daily_stats")
    conn.execute("ALTER TABLE daily_stats_v2 RENAME TO daily_stats")
