"""Tests for the metrics_db module — SQLite-backed LLM call persistence."""

import sqlite3
import threading
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from agent import metrics_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point metrics_db at a temporary SQLite DB for each test."""
    db_path = tmp_path / "metrics.db"
    monkeypatch.setattr(metrics_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(metrics_db, "DB_PATH", db_path)
    # Clear any cached per-thread connection so we get a fresh one
    metrics_db._local.__dict__.pop("conn", None)
    metrics_db.init_db()
    yield
    # Close the per-thread connection to release the file
    conn = getattr(metrics_db._local, "conn", None)
    if conn:
        conn.close()
        metrics_db._local.conn = None


class TestInitDb:
    """Test database initialization."""

    def test_creates_table(self):
        # Table should exist after init_db (called by fixture)
        conn = metrics_db._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent(self):
        # Calling init_db twice should not raise
        metrics_db.init_db()
        metrics_db.init_db()


class TestRecord:
    """Test writing records to SQLite."""

    def test_basic_record(self):
        metrics_db.record("ollama", 1.5, True, input_tokens=100, output_tokens=50, model="qwen")
        rows = metrics_db._query_rows("SELECT * FROM llm_calls")
        assert len(rows) == 1
        assert rows[0]["endpoint"] == "ollama"
        assert rows[0]["duration"] == 1.5
        assert rows[0]["success"] == 1
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["model"] == "qwen"

    def test_failure_record(self):
        metrics_db.record("claude_api", 0.3, False, error="Rate limited")
        rows = metrics_db._query_rows("SELECT * FROM llm_calls")
        assert len(rows) == 1
        assert rows[0]["success"] == 0
        assert rows[0]["error"] == "Rate limited"

    def test_error_truncated_to_500(self):
        long_error = "x" * 1000
        metrics_db.record("ollama", 0.1, False, error=long_error)
        rows = metrics_db._query_rows("SELECT * FROM llm_calls")
        assert len(rows) == 1
        assert len(rows[0]["error"]) == 500

    def test_multiple_records(self):
        for i in range(5):
            metrics_db.record("ollama", float(i), True)
        assert metrics_db.get_total_call_count() == 5


class TestEndpointStats:
    """Test aggregated endpoint statistics."""

    def _insert_records(self, endpoint, durations, success=True):
        for d in durations:
            metrics_db.record(endpoint, d, success, input_tokens=100, output_tokens=50)

    def test_basic_stats(self):
        self._insert_records("ollama", [1.0, 2.0, 3.0])
        stats = metrics_db.get_endpoint_stats("ollama", hours=1)
        assert stats["calls"] == 3
        assert stats["successes"] == 3
        assert stats["failures"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["avg_latency"] == 2.0
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150

    def test_empty_endpoint(self):
        stats = metrics_db.get_endpoint_stats("nonexistent", hours=1)
        assert stats["calls"] == 0

    def test_filter_by_endpoint(self):
        self._insert_records("ollama", [1.0, 2.0])
        self._insert_records("claude_api", [0.5])
        stats = metrics_db.get_endpoint_stats("ollama", hours=1)
        assert stats["calls"] == 2

    def test_all_endpoints(self):
        self._insert_records("ollama", [1.0])
        self._insert_records("claude_api", [0.5])
        stats = metrics_db.get_endpoint_stats(None, hours=1)
        assert stats["calls"] == 2

    def test_time_window_filtering(self):
        # Insert a record with an old timestamp directly
        conn = metrics_db._get_conn()
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute(
            "INSERT INTO llm_calls (timestamp, endpoint, duration, success) VALUES (?, ?, ?, ?)",
            (old_ts, "ollama", 5.0, 1),
        )
        conn.commit()

        # Insert a recent record
        metrics_db.record("ollama", 1.0, True)

        stats = metrics_db.get_endpoint_stats("ollama", hours=24)
        assert stats["calls"] == 1  # Only the recent one


class TestPercentiles:
    """Test latency percentile calculations."""

    def test_basic_percentiles(self):
        for d in [float(i) for i in range(1, 101)]:
            metrics_db.record("ollama", d, True)

        pcts = metrics_db.get_percentiles("ollama", hours=1)
        assert pcts["p50"] == 51.0  # index 50 of [1..100] = 51
        assert pcts["p95"] == 96.0  # index 95 of [1..100] = 96
        assert pcts["p99"] == 100.0  # index 99 of [1..100] = 100

    def test_empty_percentiles(self):
        pcts = metrics_db.get_percentiles("ollama", hours=1)
        assert pcts == {}

    def test_few_records_use_max(self):
        for d in [1.0, 2.0, 3.0]:
            metrics_db.record("ollama", d, True)

        pcts = metrics_db.get_percentiles("ollama", hours=1)
        assert pcts["p50"] == 2.0
        # With < 20 records, p95 falls back to max
        assert pcts["p95"] == 3.0


class TestHourlyTrend:
    """Test hourly trend breakdown."""

    def test_returns_grouped_data(self):
        metrics_db.record("ollama", 1.0, True)
        metrics_db.record("ollama", 2.0, True)
        metrics_db.record("claude_api", 0.5, True)

        trend = metrics_db.get_hourly_trend(hours=1)
        assert len(trend) >= 1
        # All should be in the current hour
        for t in trend:
            assert "hour" in t
            assert "calls" in t
            assert "avg_latency" in t

    def test_empty_trend(self):
        trend = metrics_db.get_hourly_trend(hours=1)
        assert trend == []


class TestSlowestCalls:
    """Test slowest calls query."""

    def test_returns_slowest(self):
        for d in [1.0, 5.0, 2.0, 10.0, 0.5]:
            metrics_db.record("ollama", d, True, input_tokens=100, output_tokens=50)

        slow = metrics_db.get_slowest_calls(3, hours=1)
        assert len(slow) == 3
        assert slow[0]["duration"] == 10.0
        assert slow[1]["duration"] == 5.0
        assert slow[2]["duration"] == 2.0

    def test_empty_slowest(self):
        slow = metrics_db.get_slowest_calls(5, hours=1)
        assert slow == []


class TestRecentErrors:
    """Test recent errors query."""

    def test_returns_errors(self):
        metrics_db.record("ollama", 0.1, False, error="Error A")
        metrics_db.record("ollama", 0.2, True)
        metrics_db.record("claude_api", 0.3, False, error="Error B")

        errors = metrics_db.get_recent_errors(5)
        assert len(errors) == 2
        # Most recent first
        assert errors[0]["error"] == "Error B"

    def test_limit(self):
        for i in range(10):
            metrics_db.record("ollama", 0.1, False, error=f"Error {i}")

        errors = metrics_db.get_recent_errors(3)
        assert len(errors) == 3


class TestGetSummary:
    """Test human-readable summary generation."""

    def test_empty_summary(self):
        summary = metrics_db.get_summary(hours=1)
        assert "No LLM call data" in summary

    def test_summary_with_data(self):
        metrics_db.record("ollama", 1.5, True, input_tokens=100, output_tokens=50, model="qwen")
        metrics_db.record("claude_api", 0.8, True, input_tokens=500, output_tokens=200)
        metrics_db.record("claude_api", 0.3, False, error="Rate limit")

        summary = metrics_db.get_summary(hours=1)
        assert "LLM Metrics" in summary
        assert "ollama" in summary
        assert "claude_api" in summary
        assert "Rate limit" in summary

    def test_summary_includes_trend(self):
        metrics_db.record("ollama", 1.0, True)
        summary = metrics_db.get_summary(hours=1)
        assert "Hourly trend" in summary or "Slowest calls" in summary


class TestGetTotalCallCount:
    """Test total call count."""

    def test_empty(self):
        assert metrics_db.get_total_call_count() == 0

    def test_count(self):
        for _ in range(7):
            metrics_db.record("ollama", 1.0, True)
        assert metrics_db.get_total_call_count() == 7


class TestThreadSafety:
    """Test concurrent writes don't corrupt the database."""

    def test_concurrent_writes(self):
        errors = []

        def write_many(endpoint, count):
            try:
                # Need a fresh connection for this thread
                metrics_db._local.__dict__.pop("conn", None)
                for i in range(count):
                    metrics_db.record(endpoint, 0.01 * i, True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_many, args=("ollama", 50)),
            threading.Thread(target=write_many, args=("claude_api", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert metrics_db.get_total_call_count() == 100
