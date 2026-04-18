"""Tests for agent/healthy_notifications.py — SQLite-backed online-ping log.

Exercises the record/query round-trip plus the row-cap trim so callers can
rely on ``query_healthy_notifications`` never returning more than ``MAX_ROWS``
even after a long run of lifecycle events.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent import healthy_notifications


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point healthy_notifications at a per-test SQLite file."""
    db_path = tmp_path / "healthy_notifications.db"
    monkeypatch.setattr(healthy_notifications, "DB_DIR", tmp_path)
    monkeypatch.setattr(healthy_notifications, "DB_PATH", db_path)
    healthy_notifications._local.__dict__.pop("conn", None)
    yield
    conn = getattr(healthy_notifications._local, "conn", None)
    if conn is not None:
        conn.close()
        healthy_notifications._local.conn = None


class TestInitDb:
    def test_creates_table(self):
        healthy_notifications.init_db()
        conn = healthy_notifications._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='healthy_notifications'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        healthy_notifications.init_db()
        healthy_notifications.init_db()
        conn = healthy_notifications._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM healthy_notifications").fetchone()[0] == 0


class TestRecordHealthyNotification:
    def test_default_timestamp_uses_now(self):
        before = datetime.now()
        healthy_notifications.record_healthy_notification(
            message_url="https://discord.com/channels/1/2/3"
        )
        rows = healthy_notifications.query_healthy_notifications()
        assert len(rows) == 1
        recorded = datetime.fromisoformat(rows[0]["timestamp"])
        assert abs((recorded - before).total_seconds()) < 5

    def test_stores_explicit_datetime(self):
        ts = datetime(2026, 4, 18, 12, 30, 0)
        healthy_notifications.record_healthy_notification(timestamp=ts)
        rows = healthy_notifications.query_healthy_notifications(max_age_hours=None)
        assert rows[0]["timestamp"] == "2026-04-18T12:30:00"
        assert rows[0]["message_url"] is None

    def test_stores_null_url_when_not_provided(self):
        healthy_notifications.record_healthy_notification()
        rows = healthy_notifications.query_healthy_notifications()
        assert rows[0]["message_url"] is None

    def test_round_trip_preserves_url(self):
        url = "https://discord.com/channels/111/222/333"
        healthy_notifications.record_healthy_notification(message_url=url)
        rows = healthy_notifications.query_healthy_notifications()
        assert rows[0]["message_url"] == url

    def test_trims_to_max_rows(self):
        # Write more than MAX_ROWS and confirm only the newest survive.
        for i in range(healthy_notifications.MAX_ROWS + 10):
            healthy_notifications.record_healthy_notification(
                message_url=f"url-{i}"
            )
        conn = healthy_notifications._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM healthy_notifications"
        ).fetchone()[0]
        assert total == healthy_notifications.MAX_ROWS
        # The very oldest row (url-0) must have been trimmed.
        oldest = conn.execute(
            "SELECT message_url FROM healthy_notifications ORDER BY id ASC LIMIT 1"
        ).fetchone()
        assert oldest[0] != "url-0"


class TestQueryHealthyNotifications:
    def test_empty_when_table_empty(self):
        assert healthy_notifications.query_healthy_notifications() == []

    def test_returns_newest_first(self):
        old = datetime.now() - timedelta(hours=2)
        newer = datetime.now() - timedelta(hours=1)
        healthy_notifications.record_healthy_notification(
            timestamp=old, message_url="url-old"
        )
        healthy_notifications.record_healthy_notification(
            timestamp=newer, message_url="url-new"
        )
        rows = healthy_notifications.query_healthy_notifications()
        assert [r["message_url"] for r in rows] == ["url-new", "url-old"]

    def test_respects_limit(self):
        for i in range(5):
            healthy_notifications.record_healthy_notification(message_url=f"u{i}")
        rows = healthy_notifications.query_healthy_notifications(limit=2)
        assert len(rows) == 2

    def test_filters_by_max_age(self):
        # Aged row should be filtered; fresh row should remain.
        old = datetime.now() - timedelta(hours=48)
        fresh = datetime.now() - timedelta(minutes=10)
        healthy_notifications.record_healthy_notification(
            timestamp=old, message_url="old"
        )
        healthy_notifications.record_healthy_notification(
            timestamp=fresh, message_url="fresh"
        )
        rows = healthy_notifications.query_healthy_notifications(max_age_hours=24)
        assert [r["message_url"] for r in rows] == ["fresh"]

    def test_max_age_none_disables_filter(self):
        old = datetime.now() - timedelta(days=365)
        healthy_notifications.record_healthy_notification(
            timestamp=old, message_url="ancient"
        )
        rows = healthy_notifications.query_healthy_notifications(max_age_hours=None)
        assert rows and rows[0]["message_url"] == "ancient"
