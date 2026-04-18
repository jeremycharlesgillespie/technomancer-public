"""Unit tests for the 24h time-window scoping on the Jira sync DLQ health check.

Covers acceptance criteria for TK-675: the red indicator must reflect
*recent* sync problems, not cumulative history. A row older than 24h does
not count toward the reported depth, and a DLQ that only has stale rows
reports ``ok=True``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from idea_board import health, jira_sync_dlq


@pytest.fixture(autouse=True)
def _reset_health_cache():
    health.clear_cache()
    yield
    health.clear_cache()


@pytest.fixture(autouse=True)
def _isolate_jira_dlq_db(tmp_path, monkeypatch):
    """Route jira_sync_dlq at a temp SQLite so the real DLQ is untouched."""
    db_path = tmp_path / "jira_sync_dlq.db"
    monkeypatch.setattr(jira_sync_dlq, "DB_PATH", db_path)
    jira_sync_dlq._local.__dict__.pop("conn", None)
    yield
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn:
        conn.close()
        jira_sync_dlq._local.conn = None


def _insert_row(idea_id: str, last_failed_at: str) -> None:
    """Insert a DLQ row with an explicit ``last_failed_at`` timestamp.

    The public :func:`add_dlq_entry` always stamps ``now()``, so direct SQL
    is the only way to plant rows older than the 24h window.
    """
    jira_sync_dlq.init_db()
    conn = jira_sync_dlq._get_conn()
    conn.execute(
        """
        INSERT INTO jira_sync_dlq
            (idea_id, payload_json, error, attempts,
             first_failed_at, last_failed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (idea_id, "{}", "test-error", 3, last_failed_at, last_failed_at),
    )
    conn.commit()


def test_mixed_rows_only_count_last_24h():
    """5 DLQ rows total, 3 within 24h -> reports 3 recent and ok=False."""
    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    older_ts = (now - timedelta(days=7)).isoformat(timespec="seconds")

    for i in range(3):
        _insert_row(f"TK-recent-{i}", recent_ts)
    for i in range(2):
        _insert_row(f"TK-stale-{i}", older_ts)

    with patch.object(health, "is_jira_configured", return_value=True):
        result = health.check_jira()

    assert result["ok"] is False
    assert result["dlq_depth"] == 3
    assert result["detail"] == "3 DLQ entries in last 24h"
    assert result["window_hours"] == 24


def test_all_rows_older_than_24h_report_ok():
    """10 DLQ rows, all older than 24h -> reports 0 recent and ok=True."""
    now = datetime.now(timezone.utc)
    stale_ts = (now - timedelta(days=3)).isoformat(timespec="seconds")

    for i in range(10):
        _insert_row(f"TK-stale-{i}", stale_ts)

    with patch.object(health, "is_jira_configured", return_value=True):
        result = health.check_jira()

    assert result["ok"] is True
    assert result["dlq_depth"] == 0
    assert result["detail"] == "0 DLQ entries in last 24h"
