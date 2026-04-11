"""Tests for agent/engagement_analytics.py — command and message tracking."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.engagement_analytics as ea


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB to a temp directory for each test, clearing cached connections."""
    monkeypatch.setattr(ea, "DB_DIR", tmp_path)
    monkeypatch.setattr(ea, "DB_PATH", tmp_path / "engagement.db")
    if hasattr(ea._local, "eng_conn"):
        try:
            ea._local.eng_conn.close()
        except Exception:
            pass
        del ea._local.eng_conn
    ea.init_db()


class TestInitDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "engagement.db"
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "command_usage" in table_names
        assert "message_activity" in table_names
        conn.close()


class TestTrackCommand:
    def test_tracks_command(self):
        ea.track_command("test_cmd", user="bob", success=True, duration_ms=42.0)
        stats = ea.get_command_stats(days=1)
        assert len(stats) == 1
        assert stats[0]["command"] == "test_cmd"

    def test_tracks_multiple_commands(self):
        ea.track_command("cmd_a")
        ea.track_command("cmd_b")
        ea.track_command("cmd_a")
        stats = ea.get_command_stats(days=1)
        cmd_a = [s for s in stats if s["command"] == "cmd_a"]
        assert cmd_a[0]["cnt"] == 2


class TestTrackMessage:
    def test_tracks_message(self):
        ea.track_message("alice", channel="general")
        activity = ea.get_daily_activity(days=1)
        assert len(activity) >= 1


class TestGetEngagementReport:
    def test_returns_string(self):
        report = ea.get_engagement_report(days=1)
        assert isinstance(report, str)

    def test_report_with_data(self):
        ea.track_command("web_search", user="bob")
        ea.track_message("bob")
        report = ea.get_engagement_report(days=1)
        assert isinstance(report, str)


class TestGetCommandStats:
    def test_empty_db_returns_empty(self):
        stats = ea.get_command_stats(days=1)
        assert stats == []

    def test_stats_have_expected_keys(self):
        ea.track_command("test", user="u", success=True, duration_ms=10)
        stats = ea.get_command_stats(days=1)
        assert "command" in stats[0]
        assert "cnt" in stats[0]


class TestGetDailyActivity:
    def test_empty_returns_empty(self):
        activity = ea.get_daily_activity(days=1)
        assert isinstance(activity, list)
