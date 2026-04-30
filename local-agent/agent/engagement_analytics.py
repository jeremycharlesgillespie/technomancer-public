"""
Engagement Analytics — Track command usage, message interactions, and
feature adoption to enable data-driven development prioritization.

Collects:
    - Command invocations (who, what, when, success)
    - Message volume per user
    - Feature adoption rates

Aggregates:
    - Most/least used commands (daily, weekly)
    - User activity trends
    - Underused features for deprecation candidates

Database: data/engagement.db
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .core import Tool

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "engagement.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "eng_conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.eng_conn = conn
    return conn


def init_db() -> None:
    """Create engagement tables."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS command_usage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            command     TEXT NOT NULL,
            user_name   TEXT NOT NULL DEFAULT '',
            args        TEXT NOT NULL DEFAULT '',
            success     INTEGER NOT NULL DEFAULT 1,
            duration_ms REAL
        );
        CREATE INDEX IF NOT EXISTS idx_cu_ts ON command_usage (timestamp);
        CREATE INDEX IF NOT EXISTS idx_cu_cmd ON command_usage (command, timestamp);

        CREATE TABLE IF NOT EXISTS message_activity (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            user_name   TEXT NOT NULL,
            channel     TEXT NOT NULL DEFAULT '',
            has_attachment INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_ma_ts ON message_activity (timestamp);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def track_command(
    command: str,
    user: str = "",
    args: str = "",
    success: bool = True,
    duration_ms: float | None = None,
) -> None:
    """Record a command invocation."""
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "INSERT INTO command_usage (timestamp, command, user_name, args, success, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), command, user[:50], args[:200], int(success), duration_ms),
        )
        conn.commit()
    except Exception:
        pass  # best-effort


def track_message(user: str, channel: str = "", has_attachment: bool = False) -> None:
    """Record a message for activity tracking."""
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "INSERT INTO message_activity (timestamp, user_name, channel, has_attachment) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), user[:50], channel[:50], int(has_attachment)),
        )
        conn.commit()
    except Exception:
        pass  # best-effort


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_command_stats(days: int = 7) -> list[dict[str, Any]]:
    """Get command usage stats for the last N days, sorted by frequency."""
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT command, COUNT(*) AS cnt,
                  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes,
                  COUNT(DISTINCT user_name) AS unique_users,
                  AVG(duration_ms) AS avg_duration
           FROM command_usage WHERE timestamp >= ?
           GROUP BY command ORDER BY cnt DESC""",
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_underused_commands(days: int = 14) -> list[str]:
    """Identify commands that exist but have zero or very few invocations."""
    try:
        from .command_suggestions import COMMANDS
        all_commands = {c.name.lower() for c in COMMANDS}
    except Exception:
        return []

    stats = get_command_stats(days)
    used = {row["command"].lower() for row in stats}
    return sorted(all_commands - used)


def get_unused_commands_detailed(days: int = 14) -> list[dict[str, Any]]:
    """Detailed info for each unused command.

    Returns a list of dicts with: name, description, category, last_seen (ISO
    string or None), invocations. "Unused" means zero invocations in the last
    ``days`` days. ``last_seen`` looks at all history, not just the window.
    """
    try:
        from .command_suggestions import COMMANDS
    except Exception:
        return []

    stats = get_command_stats(days)
    used = {row["command"].lower() for row in stats}

    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT command, MAX(timestamp) AS last_seen, COUNT(*) AS total "
        "FROM command_usage GROUP BY command"
    ).fetchall()
    history = {r["command"].lower(): (r["last_seen"], r["total"]) for r in rows}

    result: list[dict[str, Any]] = []
    for cmd in COMMANDS:
        key = cmd.name.lower()
        if key in used:
            continue
        last_seen, total = history.get(key, (None, 0))
        result.append({
            "name": cmd.name,
            "description": cmd.description,
            "category": cmd.category,
            "last_seen": last_seen,
            "invocations": total,
            "days": days,
        })
    result.sort(key=lambda r: r["name"].lower())
    return result


def get_daily_activity(days: int = 14) -> list[dict[str, Any]]:
    """Get daily message and command counts."""
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(days=days)).isoformat()

    msg_rows = conn.execute(
        """SELECT DATE(timestamp) AS day, COUNT(*) AS messages, COUNT(DISTINCT user_name) AS users
           FROM message_activity WHERE timestamp >= ?
           GROUP BY day ORDER BY day""",
        (since,),
    ).fetchall()

    cmd_rows = conn.execute(
        """SELECT DATE(timestamp) AS day, COUNT(*) AS commands
           FROM command_usage WHERE timestamp >= ?
           GROUP BY day ORDER BY day""",
        (since,),
    ).fetchall()
    cmd_by_day = {r["day"]: r["commands"] for r in cmd_rows}

    result = []
    for r in msg_rows:
        result.append({
            "day": r["day"],
            "messages": r["messages"],
            "commands": cmd_by_day.get(r["day"], 0),
            "users": r["users"],
        })
    return result


def get_top_users(days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
    """Get most active users by message + command count."""
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(days=days)).isoformat()

    rows = conn.execute(
        """SELECT user_name,
                  (SELECT COUNT(*) FROM message_activity WHERE user_name = u.user_name AND timestamp >= ?) AS messages,
                  (SELECT COUNT(*) FROM command_usage WHERE user_name = u.user_name AND timestamp >= ?) AS commands
           FROM (SELECT DISTINCT user_name FROM message_activity WHERE timestamp >= ?
                 UNION SELECT DISTINCT user_name FROM command_usage WHERE timestamp >= ?) u
           ORDER BY messages + commands DESC LIMIT ?""",
        (since, since, since, since, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def get_engagement_report(days: int = 7) -> str:
    """Generate a human-readable engagement report."""
    cmd_stats = get_command_stats(days)
    daily = get_daily_activity(days)
    underused = get_underused_commands(days)
    top_users = get_top_users(days, limit=5)

    lines = [f"**Community Engagement Report** (last {days} days)", ""]

    # Command usage
    total_cmds = sum(r["cnt"] for r in cmd_stats)
    lines.append(f"**Commands:** {total_cmds} invocations across {len(cmd_stats)} commands")
    if cmd_stats:
        lines.append("Top commands:")
        for r in cmd_stats[:8]:
            users = r["unique_users"]
            lines.append(f"  `{r['command']}`: {r['cnt']}x ({users} user{'s' if users != 1 else ''})")
    lines.append("")

    # Daily activity
    if daily:
        total_msgs = sum(d["messages"] for d in daily)
        avg_daily = total_msgs / len(daily) if daily else 0
        lines.append(f"**Messages:** {total_msgs} total, {avg_daily:.0f}/day average")
        lines.append("")

    # Underused features
    if underused:
        lines.append(f"**Underused features** (0 invocations in {days}d):")
        lines.append(f"  {', '.join(f'`{c}`' for c in underused[:10])}")
        lines.append("")

    # Top users
    if top_users:
        lines.append("**Most active users:**")
        for u in top_users:
            lines.append(f"  {u['user_name']}: {u['messages']} msgs, {u['commands']} cmds")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

def get_engagement_tools() -> list[Tool]:
    """Return engagement analytics tools for the LLM agent."""
    from .core import create_tool

    return [
        create_tool(
            "engagement_report",
            "Show community engagement analytics: command usage, activity trends, underused features",
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
            function=lambda days=7: get_engagement_report(days),
        ),
    ]
