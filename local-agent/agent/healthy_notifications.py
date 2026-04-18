"""
Healthy Notifications — SQLite-backed log of ``bot online`` Discord webhook
posts.

The ``/errors`` empty state calls :func:`query_healthy_notifications` so
operators can click through to the most recent Discord message confirming
the bot is healthy — the point is not a fresh ping, but proof that the
monitor is still posting. Each row captures the timestamp the notification
was sent and a Discord message URL (when the webhook posted with
``?wait=true`` and the response carried ``id`` / ``channel_id``).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

DB_DIR: Path = Path(__file__).parent.parent / "data"
DB_PATH: Path = DB_DIR / "healthy_notifications.db"

# Cap rows so the table never grows unbounded. The /errors page only renders
# the most recent handful, so older rows are dead weight.
MAX_ROWS: int = 50

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
    """Create the ``healthy_notifications`` table if it doesn't exist."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS healthy_notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            message_url  TEXT,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hn_ts ON healthy_notifications (timestamp DESC)"
    )
    conn.commit()


def record_healthy_notification(
    timestamp: datetime | str | None = None, message_url: str | None = None
) -> None:
    """Record that a ``bot online`` webhook was posted.

    ``timestamp`` may be a ``datetime`` or a pre-formatted ISO string; defaults
    to ``datetime.now()``. ``message_url`` is optional — stored as ``NULL`` when
    the webhook was posted without ``?wait=true`` (in which case Discord
    returns ``204 No Content`` and no message URL is available).
    """
    if timestamp is None:
        ts_str = datetime.now().isoformat(timespec="seconds")
    elif isinstance(timestamp, datetime):
        ts_str = timestamp.isoformat(timespec="seconds")
    else:
        ts_str = str(timestamp)

    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "INSERT INTO healthy_notifications (timestamp, message_url) VALUES (?, ?)",
            (ts_str, message_url),
        )
        conn.execute(
            """
            DELETE FROM healthy_notifications
            WHERE id NOT IN (
                SELECT id FROM healthy_notifications
                ORDER BY id DESC LIMIT ?
            )
            """,
            (MAX_ROWS,),
        )
        conn.commit()
    except sqlite3.Error as e:
        log.warning("Failed to record healthy notification: %s", e)


def query_healthy_notifications(
    limit: int = 5, max_age_hours: int | None = 24
) -> list[dict[str, str | None]]:
    """Return recent healthy-notification rows, newest first.

    ``limit`` caps the number of rows. ``max_age_hours`` filters out rows
    older than that window; pass ``None`` to disable the window filter. Each
    row is ``{"timestamp": <iso str>, "message_url": <str|None>}``. Returns
    an empty list if the table doesn't exist or no rows match.
    """
    try:
        init_db()
        conn = _get_conn()
        if max_age_hours is None:
            rows = conn.execute(
                """
                SELECT timestamp, message_url
                FROM healthy_notifications
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat(
                timespec="seconds"
            )
            rows = conn.execute(
                """
                SELECT timestamp, message_url
                FROM healthy_notifications
                WHERE timestamp >= ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
    except sqlite3.Error as e:
        log.warning("Failed to query healthy notifications: %s", e)
        return []

    return [
        {"timestamp": row["timestamp"], "message_url": row["message_url"]}
        for row in rows
    ]
