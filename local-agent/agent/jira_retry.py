"""Jira request wrapper — timeout-bounded, retrying, failure-recording.

Wraps individual Jira REST calls issued from the background sync thread so
a transient network blip or hung socket cannot pin the thread or silently
drop a state change. Each call:

- enforces ``timeout=(5, 15)`` (connect, read) so hung reads can never
  exceed 15 seconds
- retries up to 3 attempts on ``ConnectionError``, ``ReadTimeout``, or
  5xx responses, with exponential backoff (1s, 3s, 9s)
- on final failure, logs a single WARNING with ``idea_id`` and
  ``jira_key`` and records a row in ``jira_sync_failures`` for a future
  reconciliation job to replay
- never propagates the underlying requests exception to the caller —
  returns ``None`` instead

Non-retryable responses (2xx/3xx/4xx other than the retryable 5xx set)
are returned to the caller as-is so it can apply its own logic (for
example, ``_post_with_retry`` honours ``Retry-After`` on 429).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

# Connect timeout / read timeout. The read timeout must stay small enough
# that the background sync thread can't be pinned for minutes by a hung
# connection.
DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 15.0)

MAX_ATTEMPTS: int = 3
BACKOFF_SECONDS: tuple[float, ...] = (1.0, 3.0, 9.0)
RETRYABLE_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})

DB_DIR: Path = Path(__file__).parent.parent / "data"
DB_PATH: Path = DB_DIR / "jira_sync_failures.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, opening it on first use."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def _init_db() -> None:
    """Create the ``jira_sync_failures`` table if absent. Idempotent."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jira_sync_failures (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id      TEXT,
            jira_key     TEXT,
            attempted_at TEXT NOT NULL,
            error        TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jira_sync_failures_idea_id
        ON jira_sync_failures (idea_id)
        """
    )
    conn.commit()


def record_failure(
    idea_id: str | None,
    jira_key: str | None,
    error: str,
) -> int:
    """Insert a ``jira_sync_failures`` row. Returns the new rowid."""
    _init_db()
    conn = _get_conn()
    attempted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO jira_sync_failures (idea_id, jira_key, attempted_at, error)
        VALUES (?, ?, ?, ?)
        """,
        (idea_id, jira_key, attempted_at, (error or "")[:2000]),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def get_failures(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent failure rows, newest first."""
    _init_db()
    conn = _get_conn()
    try:
        safe_limit = max(1, int(limit))
    except (TypeError, ValueError):
        safe_limit = 100
    rows = conn.execute(
        """
        SELECT id, idea_id, jira_key, attempted_at, error
        FROM jira_sync_failures
        ORDER BY attempted_at DESC, id DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _backoff(attempt: int) -> float:
    """Return seconds to wait after ``attempt`` (1-indexed)."""
    idx = min(max(attempt, 1) - 1, len(BACKOFF_SECONDS) - 1)
    return BACKOFF_SECONDS[idx]


def jira_request(
    method: str,
    url: str,
    *,
    idea_id: str | None = None,
    jira_key: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> requests.Response | None:
    """Issue a Jira REST request with timeout, retries, and failure recording.

    Retries up to ``max_attempts`` times on ``requests.ConnectionError``,
    ``requests.ReadTimeout``, or any 5xx response, waiting 1s / 3s / 9s
    between attempts. A response with a non-retryable status (including
    2xx and 4xx such as 429) is returned to the caller on the first try
    so upstream logic can apply request-specific handling.

    On final failure this logs a single WARNING tagged with ``idea_id``
    and ``jira_key``, records a row in ``jira_sync_failures``, and
    returns ``None``. It never raises ``requests`` exceptions.
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    requester = getattr(requests, method.lower())

    last_error: str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requester(url, **kwargs)
        except (requests.ConnectionError, requests.ReadTimeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_attempts:
                break
            delay = _backoff(attempt)
            logger.warning(
                "[JiraRetry] %s %s network error (%s), retry in %.1fs "
                "(attempt %d/%d)",
                method.upper(), url, type(exc).__name__, delay,
                attempt, max_attempts,
            )
            sleep_fn(delay)
            continue

        if resp.status_code not in RETRYABLE_STATUS:
            return resp

        last_error = f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
        if attempt >= max_attempts:
            break

        delay = _backoff(attempt)
        logger.warning(
            "[JiraRetry] %s %s -> %d, retry in %.1fs (attempt %d/%d)",
            method.upper(), url, resp.status_code, delay,
            attempt, max_attempts,
        )
        sleep_fn(delay)

    logger.warning(
        "[JiraRetry] %s %s failed after %d attempts "
        "(idea_id=%s jira_key=%s): %s",
        method.upper(), url, max_attempts, idea_id, jira_key, last_error,
    )
    try:
        record_failure(idea_id, jira_key, last_error)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[JiraRetry] Failed to record failure: %s", exc)
    return None
