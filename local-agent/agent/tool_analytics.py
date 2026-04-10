"""
Tool Usage Analytics — Track which tools the LLM calls, how often,
and whether they succeed. Identify never-called and high-failure tools
for deprecation review.

Database: data/tool_usage.db
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "tool_usage.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "tool_conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.tool_conn = conn
    return conn


def init_db() -> None:
    """Create tool usage table."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tool_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            tool_name   TEXT NOT NULL,
            success     INTEGER NOT NULL DEFAULT 1,
            duration_ms REAL,
            result_size INTEGER NOT NULL DEFAULT 0,
            error       TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_tc_ts ON tool_calls (timestamp);
        CREATE INDEX IF NOT EXISTS idx_tc_name ON tool_calls (tool_name, timestamp);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_tool_call(
    tool_name: str,
    success: bool = True,
    duration_ms: float = 0.0,
    result_size: int = 0,
    error: str = "",
) -> None:
    """Record a tool invocation."""
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "INSERT INTO tool_calls (timestamp, tool_name, success, duration_ms, result_size, error) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), tool_name, int(success), duration_ms, result_size, error[:200]),
        )
        conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def get_tool_stats(days: int = 7) -> list[dict[str, Any]]:
    """Get tool usage stats sorted by frequency."""
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT tool_name,
                  COUNT(*) AS calls,
                  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes,
                  SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures,
                  ROUND(AVG(duration_ms), 1) AS avg_duration_ms,
                  ROUND(AVG(result_size), 0) AS avg_result_size
           FROM tool_calls WHERE timestamp >= ?
           GROUP BY tool_name ORDER BY calls DESC""",
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_never_called_tools(days: int = 14) -> list[str]:
    """Find registered tools with zero invocations in the given period.

    Compares the agent's registered tool list against actual usage data.
    """
    stats = get_tool_stats(days)
    called = {r["tool_name"] for r in stats}

    # Get registered tool names from the agent
    try:
        from .discord_memory_bot import agent
        if agent and hasattr(agent, "tools"):
            registered = set(agent.tools.keys())
            # Exclude internal tools
            registered.discard("get_stored_result")
            return sorted(registered - called)
    except Exception:
        pass
    return []


def get_high_failure_tools(days: int = 7, min_calls: int = 3) -> list[dict[str, Any]]:
    """Find tools with >50% failure rate (with minimum call count)."""
    stats = get_tool_stats(days)
    return [
        r for r in stats
        if r["calls"] >= min_calls and r["failures"] > r["successes"]
    ]


def get_tool_usage_report(days: int = 7) -> str:
    """Generate a human-readable tool usage report."""
    stats = get_tool_stats(days)
    never_called = get_never_called_tools(days)
    high_fail = get_high_failure_tools(days)

    total_calls = sum(r["calls"] for r in stats)
    total_tools = len(stats)

    lines = [f"**Tool Usage Report** (last {days} days)", ""]

    lines.append(f"**Total:** {total_calls} tool calls across {total_tools} tools")
    lines.append("")

    if stats:
        lines.append("**Most used:**")
        for r in stats[:10]:
            fail_pct = round(r["failures"] / r["calls"] * 100) if r["calls"] else 0
            lines.append(
                f"  `{r['tool_name']}`: {r['calls']}x "
                f"({fail_pct}% fail, avg {r['avg_duration_ms']}ms)"
            )
        lines.append("")

    if never_called:
        lines.append(f"**Never called** ({len(never_called)} tools, deprecation candidates):")
        lines.append(f"  {', '.join(f'`{t}`' for t in never_called[:15])}")
        lines.append("")

    if high_fail:
        lines.append("**High failure rate (>50%):**")
        for r in high_fail:
            lines.append(f"  `{r['tool_name']}`: {r['failures']}/{r['calls']} failed")

    if not stats:
        lines.append("No tool usage data yet. Data will accumulate as the bot processes messages.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

def get_tool_analytics_tools() -> list:
    """Return tool analytics tools for the LLM agent."""
    from .core import create_tool

    return [
        create_tool(
            "tool_usage_report",
            "Show which tools are most/least used, failure rates, and deprecation candidates",
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Days of history to analyze (default 7)",
                    },
                },
                "required": [],
            },
            function=lambda days=7: get_tool_usage_report(days),
        ),
    ]
