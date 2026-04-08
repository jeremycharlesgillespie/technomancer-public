"""
SQLite-backed persistence for LLM call metrics.

Stores every LLM API call record (Ollama, Claude) in a local SQLite database
for trend analysis across restarts.  The in-memory PerfMonitor forwards each
record here automatically.

Database lives at ``local-agent/profiling/metrics.db``.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "profiling"
DB_PATH = DB_DIR / "metrics.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (created on first use)."""
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
    """Create the metrics table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            endpoint    TEXT    NOT NULL,
            model       TEXT    NOT NULL DEFAULT '',
            duration    REAL    NOT NULL,
            success     INTEGER NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            error       TEXT    NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_ts
        ON llm_calls (timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_endpoint
        ON llm_calls (endpoint, timestamp)
    """)
    conn.commit()


def record(
    endpoint: str,
    duration: float,
    success: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    error: str = "",
) -> None:
    """Persist a single LLM call record to SQLite."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO llm_calls
               (timestamp, endpoint, model, duration, success, input_tokens, output_tokens, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                endpoint,
                model,
                round(duration, 4),
                1 if success else 0,
                input_tokens,
                output_tokens,
                error[:500] if error else "",
            ),
        )
        conn.commit()
    except Exception:
        log.exception("Failed to write metric to SQLite")


def _query_rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Execute a query and return results as list of dicts."""
    conn = _get_conn()
    init_db()  # ensure table exists
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_endpoint_stats(
    endpoint: str | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    """Aggregated stats for a given endpoint over the last N hours."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    where = "WHERE timestamp >= ?"
    params: list[Any] = [since]
    if endpoint:
        where += " AND endpoint = ?"
        params.append(endpoint)

    rows = _query_rows(
        f"""SELECT
                COUNT(*)          AS calls,
                SUM(success)      AS successes,
                AVG(duration)     AS avg_latency,
                MIN(duration)     AS min_latency,
                MAX(duration)     AS max_latency,
                SUM(input_tokens) AS total_input_tokens,
                SUM(output_tokens) AS total_output_tokens
            FROM llm_calls {where}""",
        tuple(params),
    )
    if not rows or rows[0]["calls"] == 0:
        return {"calls": 0, "hours": hours}

    r = rows[0]
    calls = r["calls"]
    successes = r["successes"] or 0
    return {
        "calls": calls,
        "successes": successes,
        "failures": calls - successes,
        "success_rate": round(successes / calls * 100, 1),
        "avg_latency": round(r["avg_latency"], 2),
        "min_latency": round(r["min_latency"], 2),
        "max_latency": round(r["max_latency"], 2),
        "total_input_tokens": r["total_input_tokens"] or 0,
        "total_output_tokens": r["total_output_tokens"] or 0,
        "hours": hours,
    }


def get_percentiles(
    endpoint: str | None = None,
    hours: int = 24,
) -> dict[str, float]:
    """Calculate p50, p95, p99 latencies from raw durations."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    where = "WHERE timestamp >= ? AND success = 1"
    params: list[Any] = [since]
    if endpoint:
        where += " AND endpoint = ?"
        params.append(endpoint)

    rows = _query_rows(
        f"SELECT duration FROM llm_calls {where} ORDER BY duration",
        tuple(params),
    )
    if not rows:
        return {}

    durations = [r["duration"] for r in rows]
    n = len(durations)
    return {
        "p50": round(durations[n // 2], 2),
        "p95": round(durations[int(n * 0.95)], 2) if n >= 20 else round(durations[-1], 2),
        "p99": round(durations[int(n * 0.99)], 2) if n >= 100 else round(durations[-1], 2),
    }


def get_hourly_trend(
    endpoint: str | None = None,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Average latency per hour for the last N hours."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    where = "WHERE timestamp >= ?"
    params: list[Any] = [since]
    if endpoint:
        where += " AND endpoint = ?"
        params.append(endpoint)

    return _query_rows(
        f"""SELECT
                SUBSTR(timestamp, 1, 13) AS hour,
                endpoint,
                COUNT(*)     AS calls,
                ROUND(AVG(duration), 2) AS avg_latency,
                ROUND(MAX(duration), 2) AS max_latency
            FROM llm_calls {where}
            GROUP BY hour, endpoint
            ORDER BY hour""",
        tuple(params),
    )


def get_slowest_calls(n: int = 5, hours: int = 24) -> list[dict[str, Any]]:
    """Return the N slowest successful calls in the last N hours."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    return _query_rows(
        """SELECT timestamp, endpoint, model, duration, input_tokens, output_tokens
           FROM llm_calls
           WHERE timestamp >= ? AND success = 1
           ORDER BY duration DESC
           LIMIT ?""",
        (since, n),
    )


def get_recent_errors(n: int = 5) -> list[dict[str, Any]]:
    """Return the N most recent errors."""
    return _query_rows(
        """SELECT timestamp, endpoint, model, duration, error
           FROM llm_calls
           WHERE success = 0
           ORDER BY timestamp DESC
           LIMIT ?""",
        (n,),
    )


def get_summary(hours: int = 24) -> str:
    """Human-readable metrics summary for the Discord 'metrics' command."""
    conn = _get_conn()
    init_db()

    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    endpoints = _query_rows(
        "SELECT DISTINCT endpoint FROM llm_calls WHERE timestamp >= ? ORDER BY endpoint",
        (since,),
    )
    if not endpoints:
        return f"No LLM call data in the last {hours}h."

    lines = [f"**LLM Metrics** (last {hours}h)", ""]

    for ep_row in endpoints:
        ep = ep_row["endpoint"]
        stats = get_endpoint_stats(ep, hours)
        pcts = get_percentiles(ep, hours)

        lines.append(f"**{ep}** -- {stats['calls']} calls, {stats['success_rate']}% success")
        if pcts:
            lines.append(
                f"  Latency: avg {stats['avg_latency']}s, "
                f"p50 {pcts.get('p50', '-')}s, "
                f"p95 {pcts.get('p95', '-')}s, "
                f"max {stats['max_latency']}s"
            )
        else:
            lines.append(f"  Latency: avg {stats['avg_latency']}s, max {stats['max_latency']}s")
        if stats["total_input_tokens"] or stats["total_output_tokens"]:
            lines.append(
                f"  Tokens: {stats['total_input_tokens']:,} in / "
                f"{stats['total_output_tokens']:,} out"
            )
        lines.append("")

    # Hourly trend (last 6 hours, compact)
    trend = get_hourly_trend(hours=6)
    if trend:
        lines.append("**Hourly trend** (last 6h):")
        for t in trend:
            hour_label = t["hour"][-5:]  # "HH:MM" portion
            lines.append(
                f"  {hour_label} | {t['endpoint']:10s} | "
                f"{t['calls']} calls, avg {t['avg_latency']}s, max {t['max_latency']}s"
            )
        lines.append("")

    # Slowest calls
    slow = get_slowest_calls(3, hours)
    if slow:
        lines.append("**Slowest calls:**")
        for s in slow:
            ts_short = s["timestamp"][11:19]  # HH:MM:SS
            lines.append(
                f"  {ts_short} {s['endpoint']} ({s['model'] or '?'}) "
                f"-- {s['duration']}s, {s['input_tokens']+s['output_tokens']} tokens"
            )
        lines.append("")

    # Recent errors
    errors = get_recent_errors(3)
    if errors:
        lines.append("**Recent errors:**")
        for e in errors:
            ts_short = e["timestamp"][11:19]
            lines.append(f"  {ts_short} [{e['endpoint']}] {e['error'][:80]}")

    return "\n".join(lines)


def get_total_call_count() -> int:
    """Total number of records in the database (for diagnostics)."""
    rows = _query_rows("SELECT COUNT(*) AS cnt FROM llm_calls")
    return rows[0]["cnt"] if rows else 0
