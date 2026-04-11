"""Tests for agent/tool_analytics.py — tool usage tracking."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.tool_analytics as ta


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB to a temp directory for each test, clearing cached connections."""
    monkeypatch.setattr(ta, "DB_DIR", tmp_path)
    monkeypatch.setattr(ta, "DB_PATH", tmp_path / "tool_usage.db")
    # Clear thread-local cached connection so it reconnects to new DB
    if hasattr(ta._local, "tool_conn"):
        try:
            ta._local.tool_conn.close()
        except Exception:
            pass
        del ta._local.tool_conn
    ta.init_db()


class TestInitDb:
    def test_creates_table(self, tmp_path):
        db_path = tmp_path / "tool_usage.db"
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "tool_calls" in table_names
        conn.close()


class TestRecordToolCall:
    def test_records_success(self):
        ta.record_tool_call("web_search", success=True, duration_ms=150.0)
        stats = ta.get_tool_stats(days=1)
        assert len(stats) == 1
        assert stats[0]["tool_name"] == "web_search"
        assert stats[0]["successes"] == 1

    def test_records_failure(self):
        ta.record_tool_call("read_file", success=False, error="not found")
        stats = ta.get_tool_stats(days=1)
        assert stats[0]["failures"] == 1


class TestGetToolStats:
    def test_empty_db(self):
        stats = ta.get_tool_stats(days=1)
        assert stats == []

    def test_stats_keys(self):
        ta.record_tool_call("test_tool", success=True, duration_ms=100)
        stats = ta.get_tool_stats(days=1)
        assert "tool_name" in stats[0]
        assert "calls" in stats[0]
        assert "successes" in stats[0]
        assert "failures" in stats[0]

    def test_aggregates_multiple_calls(self):
        ta.record_tool_call("web_search", success=True, duration_ms=100)
        ta.record_tool_call("web_search", success=True, duration_ms=200)
        ta.record_tool_call("web_search", success=False, duration_ms=50)
        stats = ta.get_tool_stats(days=1)
        ws = [s for s in stats if s["tool_name"] == "web_search"][0]
        assert ws["calls"] == 3
        assert ws["successes"] == 2
        assert ws["failures"] == 1


class TestGetHighFailureTools:
    def test_no_high_failure(self):
        ta.record_tool_call("good_tool", success=True)
        ta.record_tool_call("good_tool", success=True)
        ta.record_tool_call("good_tool", success=True)
        result = ta.get_high_failure_tools(days=1, min_calls=3)
        assert result == []

    def test_detects_high_failure(self):
        ta.record_tool_call("bad_tool", success=False)
        ta.record_tool_call("bad_tool", success=False)
        ta.record_tool_call("bad_tool", success=True)
        result = ta.get_high_failure_tools(days=1, min_calls=3)
        assert len(result) == 1
        assert result[0]["tool_name"] == "bad_tool"


class TestGetToolUsageReport:
    def test_returns_string(self):
        report = ta.get_tool_usage_report(days=1)
        assert isinstance(report, str)
