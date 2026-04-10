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
    """Tracks Discord gateway connection health with predictive monitoring.

    Records connect/disconnect/resume events, heartbeat latency, and error
    codes. Persists events to SQLite for trend analysis. Detects degradation
    patterns (frequent disconnects, rising latency) and fires proactive alerts.
    """

    connected: bool = False
    last_connect: float = 0.0
    last_disconnect: float = 0.0
    disconnect_count: int = 0
    resume_count: int = 0
    error_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Rolling windows for trend detection (last 60 minutes)
    _disconnect_times: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)
    _latency_samples: deque = field(default_factory=lambda: deque(maxlen=300), repr=False)
    _error_codes: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)

    # Alert cooldown (don't spam alerts)
    _last_alert_time: float = 0.0
    ALERT_COOLDOWN: float = 600.0  # 10 minutes between alerts

    def record_connect(self) -> None:
        with self._lock:
            self.connected = True
            self.last_connect = time.monotonic()
        _log_gateway_event("connect")

    def record_disconnect(self) -> None:
        now = time.monotonic()
        with self._lock:
            self.connected = False
            self.last_disconnect = now
            self.disconnect_count += 1
            self._disconnect_times.append(now)
        _log_gateway_event("disconnect")
        self._check_health()

    def record_resume(self) -> None:
        """Record a gateway resume (reconnection without full re-identify)."""
        with self._lock:
            self.connected = True
            self.resume_count += 1
        _log_gateway_event("resume")

    def record_latency(self, latency_ms: float) -> None:
        """Record a heartbeat latency measurement."""
        with self._lock:
            self._latency_samples.append((time.monotonic(), latency_ms))
        _log_gateway_event("heartbeat", latency_ms=latency_ms)

    def record_error(self, error: str, code: int | None = None) -> None:
        with self._lock:
            self.error_count += 1
            self.last_error = error[:200]
            self.last_error_time = time.monotonic()
            if code is not None:
                self._error_codes.append((time.monotonic(), code))
        _log_gateway_event("error", error_code=code, detail=error[:200])
        self._check_health()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            uptime = 0.0
            if self.connected and self.last_connect:
                uptime = time.monotonic() - self.last_connect
            avg_latency = self._avg_latency()
            return {
                "connected": self.connected,
                "uptime_seconds": round(uptime),
                "disconnect_count": self.disconnect_count,
                "resume_count": self.resume_count,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "avg_latency_ms": avg_latency,
                "health_score": self._compute_health_score(),
                "prediction": self._predict(),
            }

    def _avg_latency(self) -> float:
        """Average heartbeat latency over recent samples."""
        if not self._latency_samples:
            return 0.0
        cutoff = time.monotonic() - 3600  # last hour
        recent = [ms for t, ms in self._latency_samples if t > cutoff]
        return round(sum(recent) / len(recent), 1) if recent else 0.0

    def _recent_disconnect_rate(self) -> float:
        """Disconnects per hour over the last 30 minutes."""
        now = time.monotonic()
        cutoff = now - 1800  # 30 min
        recent = sum(1 for t in self._disconnect_times if t > cutoff)
        return recent * 2  # extrapolate to per-hour

    def _compute_health_score(self) -> int:
        """Compute a 0-100 health score based on recent metrics.

        100 = perfect health, 0 = critical degradation.
        """
        score = 100

        # Penalty for disconnects (up to -40)
        dc_rate = self._recent_disconnect_rate()
        if dc_rate >= 6:
            score -= 40
        elif dc_rate >= 3:
            score -= 25
        elif dc_rate >= 1:
            score -= 10

        # Penalty for high latency (up to -30)
        avg_lat = self._avg_latency()
        if avg_lat > 500:
            score -= 30
        elif avg_lat > 250:
            score -= 15
        elif avg_lat > 100:
            score -= 5

        # Penalty for recent errors (up to -30)
        now = time.monotonic()
        recent_errors = sum(1 for t, _ in self._error_codes if now - t < 1800)
        if recent_errors >= 5:
            score -= 30
        elif recent_errors >= 2:
            score -= 15
        elif recent_errors >= 1:
            score -= 5

        return max(0, score)

    def _predict(self) -> str:
        """Predict connection health trajectory."""
        score = self._compute_health_score()
        dc_rate = self._recent_disconnect_rate()

        if score >= 90:
            return "stable"
        if score >= 70:
            return "minor_degradation"
        if score >= 40:
            if dc_rate >= 3:
                return "disconnect_pattern_detected"
            return "degraded"
        return "critical"

    def _check_health(self) -> None:
        """Check health and fire an alert if degraded."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_alert_time < self.ALERT_COOLDOWN:
                return
            score = self._compute_health_score()
            prediction = self._predict()

        if score < 70:
            with self._lock:
                self._last_alert_time = now
            try:
                from .alerts import send_alert
                send_alert(
                    f"Health score: **{score}/100** — {prediction}\n"
                    f"Disconnects (30min): {self._recent_disconnect_rate():.0f}/hr\n"
                    f"Avg latency: {self._avg_latency():.0f}ms\n"
                    f"Recent errors: {sum(1 for t, _ in self._error_codes if now - t < 1800)}",
                    title="Gateway Health Warning",
                    level="warning" if score >= 40 else "error",
                )
            except Exception:
                pass  # alerts are best-effort


_gateway_health = GatewayHealth()


def get_gateway_health() -> GatewayHealth:
    return _gateway_health


def _log_gateway_event(
    event: str,
    latency_ms: float | None = None,
    error_code: int | None = None,
    detail: str = "",
) -> None:
    """Persist a gateway event to SQLite for historical trend analysis."""
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            """INSERT INTO gateway_events (timestamp, event, latency_ms, error_code, detail)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), event, latency_ms, error_code, detail[:200]),
        )
        conn.commit()
    except Exception:
        pass  # best-effort persistence


def get_gateway_trend(hours: int = 24) -> dict[str, Any]:
    """Analyze gateway health trends over the given time window.

    Returns disconnect frequency, latency percentiles, error code
    distribution, and hourly breakdown.
    """
    init_db()
    conn = _get_conn()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    # Disconnect count
    dc_count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM gateway_events WHERE event='disconnect' AND timestamp >= ?",
        (since,),
    ).fetchone()["cnt"]

    # Resume count
    resume_count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM gateway_events WHERE event='resume' AND timestamp >= ?",
        (since,),
    ).fetchone()["cnt"]

    # Latency stats
    latency_rows = conn.execute(
        "SELECT latency_ms FROM gateway_events WHERE event='heartbeat' AND latency_ms IS NOT NULL AND timestamp >= ?",
        (since,),
    ).fetchall()
    latencies = sorted(row["latency_ms"] for row in latency_rows)

    latency_stats: dict[str, float] = {}
    if latencies:
        latency_stats = {
            "avg": round(sum(latencies) / len(latencies), 1),
            "p50": round(latencies[len(latencies) // 2], 1),
            "p95": round(latencies[int(len(latencies) * 0.95)], 1),
            "max": round(latencies[-1], 1),
            "samples": len(latencies),
        }

    # Error code distribution
    error_rows = conn.execute(
        "SELECT error_code, COUNT(*) AS cnt FROM gateway_events WHERE event='error' AND error_code IS NOT NULL AND timestamp >= ? GROUP BY error_code ORDER BY cnt DESC",
        (since,),
    ).fetchall()
    error_dist = {str(row["error_code"]): row["cnt"] for row in error_rows}

    # Hourly disconnect rate
    hourly = conn.execute(
        """SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour, COUNT(*) AS cnt
           FROM gateway_events WHERE event='disconnect' AND timestamp >= ?
           GROUP BY hour ORDER BY hour""",
        (since,),
    ).fetchall()
    hourly_disconnects = {row["hour"]: row["cnt"] for row in hourly}

    # Current health
    health = _gateway_health.get_status()

    return {
        "hours": hours,
        "disconnects": dc_count,
        "resumes": resume_count,
        "latency": latency_stats,
        "error_codes": error_dist,
        "hourly_disconnects": hourly_disconnects,
        "current_health": health,
    }


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

        CREATE TABLE IF NOT EXISTS gateway_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            event       TEXT NOT NULL,
            latency_ms  REAL,
            error_code  INTEGER,
            detail      TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_ge_ts ON gateway_events (timestamp);
        CREATE INDEX IF NOT EXISTS idx_ge_event ON gateway_events (event, timestamp);
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

    # Correlate 50006 empty message errors with recent knowledge gaps —
    # if the bot failed to send because the LLM produced nothing useful,
    # the recent context may reveal what topic caused the gap
    if cat.name == "empty_message":
        _correlate_empty_with_gaps(context)

    log.warning(
        "Discord error [%s/%s]: %s — recovery: %s",
        cat.name, cat.severity, error_text[:100], cat.recovery,
    )

    return cat


def _correlate_empty_with_gaps(context: str) -> None:
    """When a 50006 error occurs, try to enrich whatever the user asked about."""
    if not context or len(context) < 10:
        return
    try:
        from .knowledge_gaps import auto_enrich_gap

        auto_enrich_gap({
            "query": context[:500],
            "response_snippet": "Triggered by 50006 empty message error — response was empty",
            "gap_type": "failure",
            "was_escalated": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    except Exception:
        pass


def _notify_critical(cat: ErrorCategory, error_text: str) -> None:
    """Send Discord notification for critical errors to the alerts channel."""
    try:
        from .alerts import send_alert

        msg = (
            f"**{cat.name}**: {error_text[:200]}\n"
            f"**Recovery:** {cat.recovery}"
        )
        send_alert(msg, title=f"Discord Error [{cat.severity.upper()}]", level="error")
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
    lines.append(f"  Disconnects: {health['disconnect_count']} | Resumes: {health['resume_count']}")
    lines.append(f"  Errors: {health['error_count']}")
    if health["avg_latency_ms"] > 0:
        lines.append(f"  Avg Latency: {health['avg_latency_ms']}ms")
    lines.append(f"  Health Score: {health['health_score']}/100 ({health['prediction']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def get_discord_error_tools() -> list:
    """Get Discord error resilience tools for the agent."""
    from .core import create_tool

    def _gateway_health_report(hours: int = 24) -> str:
        trend = get_gateway_trend(hours)
        h = trend["current_health"]
        lines = [
            f"**Gateway Health Report** (last {hours}h)",
            f"Status: {'Connected' if h['connected'] else 'Disconnected'} | Score: {h['health_score']}/100 ({h['prediction']})",
            f"Disconnects: {trend['disconnects']} | Resumes: {trend['resumes']}",
        ]
        if trend["latency"]:
            lat = trend["latency"]
            lines.append(f"Latency: avg {lat['avg']}ms, p50 {lat['p50']}ms, p95 {lat['p95']}ms, max {lat['max']}ms ({lat['samples']} samples)")
        if trend["error_codes"]:
            codes = ", ".join(f"{k}: {v}" for k, v in trend["error_codes"].items())
            lines.append(f"Error codes: {codes}")
        if trend["hourly_disconnects"]:
            lines.append("Hourly disconnects:")
            for hour, cnt in list(trend["hourly_disconnects"].items())[-6:]:
                lines.append(f"  {hour}: {cnt}")
        return "\n".join(lines)

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
        create_tool(
            "gateway_health",
            "Show gateway connection health: score, prediction, latency, disconnect trends",
            parameters={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Hours of history to analyze (default 24)",
                    },
                },
                "required": [],
            },
            function=lambda hours=24: _gateway_health_report(hours),
        ),
    ]


# ---------------------------------------------------------------------------
# Response feedback tracking (thumbs up/down on bot responses)
# ---------------------------------------------------------------------------

_response_messages: deque[str] = deque(maxlen=200)
_response_lock = threading.Lock()


def _init_feedback_table() -> None:
    """Create the feedback table if it doesn't exist."""
    init_db()
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  TEXT NOT NULL,
            emoji       TEXT NOT NULL,
            user_name   TEXT NOT NULL DEFAULT '',
            positive    INTEGER NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rf_msg ON response_feedback (message_id)"
    )
    conn.commit()


def track_bot_response(message_id: str) -> None:
    """Mark a message ID as a bot response so reactions on it are tracked."""
    with _response_lock:
        _response_messages.append(str(message_id))


def is_bot_response(message_id: str) -> bool:
    """Check if a message ID is a tracked bot response."""
    with _response_lock:
        return str(message_id) in _response_messages


def record_response_feedback(message_id: str, emoji: str, user_name: str = "") -> None:
    """Record a thumbs-up/down reaction on a bot response."""
    positive = 1 if emoji in ("\U0001f44d", "\u2705", "\u2b50", "\U0001f525") else 0
    _init_feedback_table()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO response_feedback (message_id, emoji, user_name, positive, created_at) VALUES (?, ?, ?, ?, ?)",
        (str(message_id), emoji, user_name, positive, datetime.now().isoformat()),
    )
    conn.commit()

    # If negative feedback, boost priority for knowledge gap enrichment
    if not positive:
        try:
            from .knowledge_gaps import log_knowledge_gap

            log_knowledge_gap({
                "query": f"[user feedback: negative reaction on response {message_id}]",
                "response_snippet": f"User {user_name} reacted with {emoji}",
                "gap_type": "uncertainty",
                "was_escalated": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            pass


def get_feedback_summary(days: int = 30) -> dict[str, Any]:
    """Get response feedback summary."""
    _init_feedback_table()
    conn = _get_conn()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    total = conn.execute(
        "SELECT COUNT(*) AS cnt FROM response_feedback WHERE created_at >= ?", (since,)
    ).fetchone()["cnt"]
    positive = conn.execute(
        "SELECT COUNT(*) AS cnt FROM response_feedback WHERE created_at >= ? AND positive = 1", (since,)
    ).fetchone()["cnt"]
    negative = total - positive
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / total * 100, 1) if total else 0.0,
    }
