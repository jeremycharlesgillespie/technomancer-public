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


class TestGetUnusedCommandsDetailed:
    def test_returns_list_of_dicts(self):
        result = ea.get_unused_commands_detailed(days=14)
        assert isinstance(result, list)
        # The command registry has real commands, so at least one should exist
        # and it should be unused (no data in fresh DB).
        assert len(result) > 0
        for row in result:
            assert "name" in row
            assert "description" in row
            assert "category" in row
            assert "last_seen" in row
            assert "invocations" in row
            assert "days" in row

    def test_excludes_recently_used_commands(self):
        # Pick a real command from the registry and record usage
        from agent.command_suggestions import COMMANDS
        name = COMMANDS[0].name
        ea.track_command(name, user="bob")
        result = ea.get_unused_commands_detailed(days=14)
        names_lower = {r["name"].lower() for r in result}
        assert name.lower() not in names_lower

    def test_last_seen_none_for_never_used(self):
        result = ea.get_unused_commands_detailed(days=14)
        # In a fresh DB, everything should be "never"
        assert all(r["last_seen"] is None for r in result)
        assert all(r["invocations"] == 0 for r in result)

    def test_last_seen_populated_from_older_usage(self, monkeypatch):
        # Track a command, then check that with days=1 it still shows up as
        # unused (no recent usage) but last_seen reflects the older call.
        from agent.command_suggestions import COMMANDS
        name = COMMANDS[0].name

        # Insert a row with a timestamp well outside the 1-day window
        from datetime import datetime, timedelta
        old_ts = (datetime.now() - timedelta(days=30)).isoformat()
        conn = ea._get_conn()
        conn.execute(
            "INSERT INTO command_usage (timestamp, command, user_name, args, success, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (old_ts, name, "bob", "", 1, None),
        )
        conn.commit()

        result = ea.get_unused_commands_detailed(days=1)
        hit = next((r for r in result if r["name"].lower() == name.lower()), None)
        assert hit is not None, f"{name} should be listed as unused"
        assert hit["last_seen"] == old_ts
        assert hit["invocations"] == 1

    def test_sorted_by_name(self):
        result = ea.get_unused_commands_detailed(days=14)
        names = [r["name"].lower() for r in result]
        assert names == sorted(names)

    def test_days_field_matches_input(self):
        result = ea.get_unused_commands_detailed(days=7)
        assert all(r["days"] == 7 for r in result)
