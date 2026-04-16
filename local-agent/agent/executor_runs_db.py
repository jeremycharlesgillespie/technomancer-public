"""
Executor Runs DB — SQLite persistence for executor run metadata.

Records a structured trace for every executor / claude_code_runner run so
cost, duration, and outcome trends can be analyzed without scraping logs.

Typical usage — insert at start, update on completion:

    run_id = record_run(
        jira_key="TK-447",
        branch="2026-04-16-052408-TK-447",
        started_at=datetime.now().isoformat(),
        status="running",
    )
    # ... do the run ...
    record_run(
        id=run_id,
        ended_at=datetime.now().isoformat(),
        duration_ms=12345,
        cost_usd=0.42,
        status="success",
        exit_code=0,
        tests_passed=True,
        deployed=True,
    )

Database: ``local-agent/data/executor_runs.db``
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "executor_runs.db"

_local = threading.local()

# Whitelist of legal column names — prevents SQL injection via **kwargs keys.
_COLUMNS: frozenset[str] = frozenset({
    "id",
    "jira_key",
    "branch",
    "started_at",
    "ended_at",
    "duration_ms",
    "cost_usd",
    "status",
    "exit_code",
    "tests_passed",
    "deployed",
})


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
    """Create the executor_runs table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executor_runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            jira_key       TEXT,
            branch         TEXT,
            started_at     TEXT,
            ended_at       TEXT,
            duration_ms    INTEGER,
            cost_usd       REAL,
            status         TEXT,
            exit_code      INTEGER,
            tests_passed   INTEGER,
            deployed       INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_started
        ON executor_runs (started_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_jira
        ON executor_runs (jira_key)
    """)
    conn.commit()


def _coerce(key: str, value: Any) -> Any:
    """Coerce booleans to 0/1 for INTEGER columns — sqlite stores either but
    queries are more predictable when we normalize."""
    if key in ("tests_passed", "deployed") and isinstance(value, bool):
        return 1 if value else 0
    return value


def record_run(**fields: Any) -> int:
    """Insert a new run row, or update an existing row if ``id`` is provided.

    All fields are optional. Unknown keys are silently ignored (whitelist-only)
    so callers can safely pass the same kwargs to start and end a run.

    Returns:
        The row id of the inserted or updated run.
    """
    init_db()
    conn = _get_conn()

    # Drop any unknown keys — prevents SQL injection via kwargs keys
    safe_fields = {
        k: _coerce(k, v) for k, v in fields.items() if k in _COLUMNS
    }
    run_id = safe_fields.pop("id", None)

    if run_id is not None:
        if safe_fields:
            set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
            params = list(safe_fields.values()) + [run_id]
            conn.execute(
                f"UPDATE executor_runs SET {set_clause} WHERE id = ?",
                params,
            )
            conn.commit()
        return int(run_id)

    # Insert path — default started_at if caller didn't supply one
    if "started_at" not in safe_fields:
        safe_fields["started_at"] = datetime.now().isoformat()

    columns = ", ".join(safe_fields.keys())
    placeholders = ", ".join("?" * len(safe_fields))
    cursor = conn.execute(
        f"INSERT INTO executor_runs ({columns}) VALUES ({placeholders})",
        list(safe_fields.values()),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def get_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent runs, newest first.

    Args:
        limit: Max rows to return (default 20).
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed
           FROM executor_runs
           ORDER BY id DESC
           LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [dict(r) for r in rows]
