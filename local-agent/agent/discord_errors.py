"""
Discord Error Resilience — Unified error categorization, recovery strategies,
context buffering, gateway health tracking, and incident logging.

Consolidates all Discord error handling into a single module that:
1. Categorizes errors by type (rate_limit, empty_message, auth, permission, etc.)
2. Applies recovery strategies per category
3. Buffers recent messages for context recovery on 50006 errors
4. Tracks gateway health (connection state, error frequency)
5. Logs incidents to SQLite for trend analysis
6. Sends Discord notifications on critical errors

Covers epic stories: idea-062, idea-063, idea-074, idea-075, idea-081,
idea-082, idea-087, idea-089, idea-096, idea-099, idea-105.
"""

import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "discord_errors.db"

_local = threading.local()


# ---------------------------------------------------------------------------
# Error categories and recovery strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorCategory:
    """A Discord error classification with its recovery strategy."""

    name: str
    description: str
    recovery: str  # what to do
    severity: str  # "low", "medium", "high", "critical"


# Maps HTTP status codes and Discord error codes to categories
ERROR_CATALOG: dict[int, ErrorCategory] = {
    # HTTP status codes
    400: ErrorCategory(
        "bad_request", "Malformed request (often empty message)",
        "Validate content before sending; use fallback response", "medium",
    ),
    401: ErrorCategory(
        "unauthorized", "Invalid or expired authentication token",
        "Check bot token in .env; regenerate in Discord Developer Portal", "critical",
    ),
    403: ErrorCategory(
        "forbidden", "Missing permissions for the requested action",
        "Check bot role permissions in server settings", "high",
    ),
    404: ErrorCategory(
        "not_found", "Channel, message, or resource not found",
        "Verify channel ID; resource may have been deleted", "medium",
    ),
    429: ErrorCategory(
        "rate_limited", "Too many requests; rate limit hit",
        "Respect Retry-After header; reduce request frequency", "low",
    ),
    500: ErrorCategory(
        "discord_server_error", "Discord internal server error",
        "Retry after brief delay; Discord-side issue", "medium",
    ),
    502: ErrorCategory(
        "bad_gateway", "Discord gateway unavailable",
        "Retry with exponential backoff; transient issue", "medium",
    ),
    503: ErrorCategory(
        "service_unavailable", "Discord service temporarily down",
        "Wait and retry; check Discord status page", "high",
    ),
    # Discord-specific gateway close codes
    4004: ErrorCategory(
        "auth_failed", "Gateway authentication failed",
        "Bot token is invalid; regenerate in Discord Developer Portal and update .env", "critical",
    ),
    4008: ErrorCategory(
        "rate_limited_gateway", "Gateway rate limited (too many resumes)",
        "Reduce reconnection frequency; wait before resuming", "medium",
    ),
    4009: ErrorCategory(
        "session_timeout", "Gateway session timed out",
        "Reconnect with new session; old session expired", "medium",
    ),
    4014: ErrorCategory(
        "disallowed_intents", "Bot requires privileged intents not enabled",
        "Enable required intents in Discord Developer Portal", "critical",
    ),
    # Discord API error codes (from JSON body)
    50006: ErrorCategory(
        "empty_message", "Cannot send an empty message",
        "Pre-validate content; use fallback response if empty", "low",
    ),
    50007: ErrorCategory(
        "cannot_dm", "Cannot send DMs to this user",
        "User has DMs disabled; send to channel instead", "low",
    ),
    50013: ErrorCategory(
        "missing_permissions", "Missing required permissions",
        "Check bot role has Send Messages + Read History", "high",
    ),
    50035: ErrorCategory(
        "invalid_form", "Invalid form body (message too long, etc.)",
        "Truncate message to 2000 chars before sending", "medium",
    ),
}


def categorize_error(
    status_code: int | None = None,
    error_code: int | None = None,
    error_text: str = "",
) -> ErrorCategory:
    """Categorize a Discord error into a known category.

    Checks error_code first (more specific), then status_code.
    Falls back to a generic category if unknown.
    """
    if error_code and error_code in ERROR_CATALOG:
        return ERROR_CATALOG[error_code]
    if status_code and status_code in ERROR_CATALOG:
        return ERROR_CATALOG[status_code]

    # Try to infer from error text
    text_lower = error_text.lower()
    if "empty message" in text_lower:
        return ERROR_CATALOG[50006]
    if "rate limit" in text_lower:
        return ERROR_CATALOG[429]
    if "authentication" in text_lower or "4004" in text_lower:
        return ERROR_CATALOG[4004]

    return ErrorCategory(
        "unknown", f"Unclassified error: {error_text[:100]}",
        "Log and investigate manually", "medium",
    )


# ---------------------------------------------------------------------------
# Message context buffer for 50006 recovery
# ---------------------------------------------------------------------------

_message_buffer: deque[dict[str, str]] = deque(maxlen=20)
_buffer_lock = threading.Lock()


def buffer_message(user: str, content: str, message_id: str = "") -> None:
    """Buffer a recent message for context recovery on send failures."""
    with _buffer_lock:
        _message_buffer.append({
            "user": user,
            "content": content[:500],
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
        })


def get_recent_context(n: int = 5) -> list[dict[str, str]]:
    """Get the N most recent buffered messages."""
    with _buffer_lock:
        return list(_message_buffer)[-n:]


def suggest_recovery_content() -> str:
    """Suggest recovery content based on recent message context.

    Used when a 50006 error occurs — reconstructs what the user likely
    expects based on recent conversation.
    """
    recent = get_recent_context(3)
    if not recent:
        return "I'm here and ready to help. What would you like to discuss?"

    last = recent[-1]
    user = last.get("user", "you")
    snippet = last.get("content", "")[:100]
    return (
        f"I was responding to {user}'s message about: \"{snippet}...\" "
        f"— let me try that again. Could you rephrase your question?"
    )


# ---------------------------------------------------------------------------
# Gateway health tracking
# ---------------------------------------------------------------------------

@dataclass
class GatewayHealth:
    """Tracks Discord gateway connection health."""

    connected: bool = False
    last_connect: float = 0.0
    last_disconnect: float = 0.0
    disconnect_count: int = 0
    error_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_connect(self) -> None:
        with self._lock:
            self.connected = True
            self.last_connect = time.monotonic()

    def record_disconnect(self) -> None:
        with self._lock:
            self.connected = False
            self.last_disconnect = time.monotonic()
            self.disconnect_count += 1

    def record_error(self, error: str) -> None:
        with self._lock:
            self.error_count += 1
            self.last_error = error[:200]
            self.last_error_time = time.monotonic()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            uptime = 0.0
            if self.connected and self.last_connect:
                uptime = time.monotonic() - self.last_connect
            return {
                "connected": self.connected,
                "uptime_seconds": round(uptime),
                "disconnect_count": self.disconnect_count,
                "error_count": self.error_count,
                "last_error": self.last_error,
            }


_gateway_health = GatewayHealth()


def get_gateway_health() -> GatewayHealth:
    return _gateway_health


# ---------------------------------------------------------------------------
# Incident database
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
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
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS discord_incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
            category    TEXT NOT NULL,
            severity    TEXT NOT NULL,
            status_code INTEGER,
            error_code  INTEGER,
            error_text  TEXT NOT NULL DEFAULT '',
            recovery    TEXT NOT NULL DEFAULT '',
            resolved    INTEGER NOT NULL DEFAULT 0,
            context     TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_di_ts ON discord_incidents (timestamp);
        CREATE INDEX IF NOT EXISTS idx_di_cat ON discord_incidents (category);
    """)
    conn.commit()


def log_incident(
    category: str,
    severity: str,
    status_code: int | None = None,
    error_code: int | None = None,
    error_text: str = "",
    recovery: str = "",
    context: str = "",
) -> int:
    """Log a Discord error incident. Returns the incident ID."""
    init_db()
    conn = _get_conn()
    now = datetime.now().isoformat()
    cur = conn.execute(
        """INSERT INTO discord_incidents
           (timestamp, category, severity, status_code, error_code, error_text, recovery, context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (now, category, severity, status_code, error_code,
         error_text[:500], recovery[:200], context[:500]),
    )
    conn.commit()
    return cur.lastrowid or 0


def get_incident_summary(hours: int = 24) -> dict[str, Any]:
    """Get incident summary for the last N hours."""
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    total = conn.execute(
        "SELECT COUNT(*) AS cnt FROM discord_incidents WHERE timestamp >= ?",
        (since,),
    ).fetchone()["cnt"]

    by_category = conn.execute(
        """SELECT category, COUNT(*) AS cnt, severity
           FROM discord_incidents WHERE timestamp >= ?
           GROUP BY category ORDER BY cnt DESC""",
        (since,),
    ).fetchall()

    recent = conn.execute(
        """SELECT timestamp, category, severity, error_text, recovery
           FROM discord_incidents WHERE timestamp >= ?
           ORDER BY timestamp DESC LIMIT 5""",
        (since,),
    ).fetchall()

    return {
        "total": total,
        "hours": hours,
        "by_category": [dict(r) for r in by_category],
        "recent": [dict(r) for r in recent],
    }


# ---------------------------------------------------------------------------
# Unified error handler — call this from discord_memory_bot.py
# ---------------------------------------------------------------------------

def handle_discord_error(
    error: Exception,
    context: str = "",
) -> ErrorCategory:
    """Categorize and log a Discord error. Returns the category for caller to act on.

    Call this from any except block that catches discord.HTTPException or
    similar Discord errors. It will:
    1. Extract status/error codes from the exception
    2. Categorize the error
    3. Log an incident to SQLite
    4. Update gateway health
    5. Notify on critical errors
    """
    status_code = None
    error_code = None
    error_text = str(error)

    # Extract Discord-specific fields if available
    if hasattr(error, "status"):
        status_code = error.status
    if hasattr(error, "code"):
        error_code = error.code
    if hasattr(error, "text") and error.text:
        error_text = error.text

    cat = categorize_error(status_code, error_code, error_text)

    # Log incident
    log_incident(
        category=cat.name,
        severity=cat.severity,
        status_code=status_code,
        error_code=error_code,
        error_text=error_text,
        recovery=cat.recovery,
        context=context,
    )

    # Update health tracker
    _gateway_health.record_error(error_text)

    # Notify on critical/high severity
    if cat.severity in ("critical", "high"):
        _notify_critical(cat, error_text)

    log.warning(
        "Discord error [%s/%s]: %s — recovery: %s",
        cat.name, cat.severity, error_text[:100], cat.recovery,
    )

    return cat


def _notify_critical(cat: ErrorCategory, error_text: str) -> None:
    """Send Discord notification for critical errors via bridge API."""
    try:
        from pathlib import Path as _Path
        token_file = _Path(__file__).parent.parent / ".bridge_token"
        if not token_file.exists():
            return
        import requests
        token = token_file.read_text(encoding="utf-8").strip()
        msg = (
            f"**Discord Error Alert** [{cat.severity.upper()}]\n"
            f"**{cat.name}**: {error_text[:200]}\n"
            f"**Recovery:** {cat.recovery}"
        )
        requests.post(
            "http://127.0.0.1:8321/api/send",
            headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
            json={"message": msg},
            timeout=5,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_error_report(hours: int = 24) -> str:
    """Human-readable error report for Discord."""
    summary = get_incident_summary(hours)

    if summary["total"] == 0:
        return f"No Discord errors in the last {hours} hours."

    lines = [f"**Discord Error Report** (last {hours}h)", ""]
    lines.append(f"Total incidents: **{summary['total']}**")
    lines.append("")

    if summary["by_category"]:
        lines.append("**By Category:**")
        for row in summary["by_category"]:
            lines.append(f"  {row['category']}: {row['cnt']} ({row['severity']})")
        lines.append("")

    if summary["recent"]:
        lines.append("**Recent:**")
        for row in summary["recent"]:
            ts = row["timestamp"][11:19] if len(row["timestamp"]) > 11 else row["timestamp"]
            lines.append(f"  [{ts}] {row['category']}: {row['error_text'][:80]}")

    # Gateway health
    health = _gateway_health.get_status()
    lines.append("")
    lines.append("**Gateway Health:**")
    status = "Connected" if health["connected"] else "Disconnected"
    lines.append(f"  Status: {status}")
    if health["uptime_seconds"] > 0:
        hours_up = health["uptime_seconds"] / 3600
        lines.append(f"  Uptime: {hours_up:.1f}h")
    lines.append(f"  Disconnects: {health['disconnect_count']}")
    lines.append(f"  Errors: {health['error_count']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def get_discord_error_tools() -> list:
    """Get Discord error resilience tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "discord_error_report",
            "Show Discord error incidents, categories, and gateway health",
            parameters={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Hours of history to report (default 24)",
                    },
                },
                "required": [],
            },
            function=lambda hours=24: get_error_report(hours),
        ),
    ]
