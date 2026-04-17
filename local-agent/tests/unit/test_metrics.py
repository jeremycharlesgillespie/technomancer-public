"""Tests for agent.metrics — snapshot aggregator + Prometheus rendering."""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import metrics


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_metrics_cache():
    """Drop the module-level cache between cases so each test starts clean."""
    metrics._reset_cache()
    yield
    metrics._reset_cache()


@pytest.fixture
def patched_sources(monkeypatch):
    """Patch all four metrics sources with controllable mocks.

    Returns a SimpleNamespace with one MagicMock per source so the test
    body can flip return values or side effects per case.
    """
    executor_mock = MagicMock(return_value={
        "total_runs_24h": 10,
        "successes_24h": 8,
        "success_rate": 0.8,
        "p50_latency_ms": 1500.0,
        "p95_latency_ms": 9000.0,
    })
    vault_mock = MagicMock(return_value={
        "calls": 42,
        "cache_hit_rate": 0.84,
        "input_tokens": 1000,
        "cache_read_tokens": 8400,
        "cache_creation_tokens": 600,
    })
    board_mock = MagicMock(return_value={
        "queue_depth": 5,
        "oldest_top_ranked_wait_seconds": 3600.0,
    })
    ollama_mock = MagicMock(return_value={
        "status": "healthy",
        "last_checked": 0.0,
        "last_success": 0.0,
        "last_error": "",
        "consecutive_failures": 0,
    })

    monkeypatch.setattr(metrics, "_executor_metrics", executor_mock)
    monkeypatch.setattr(metrics, "_claude_vault_metrics", vault_mock)
    monkeypatch.setattr(metrics, "_board_metrics", board_mock)
    monkeypatch.setattr(metrics, "_ollama_metrics", ollama_mock)

    return SimpleNamespace(
        executor=executor_mock,
        claude_vault=vault_mock,
        board=board_mock,
        ollama=ollama_mock,
    )


# =============================================================================
# CACHE BEHAVIOR
# =============================================================================


class TestCache:
    def test_first_call_builds_fresh_payload(self, patched_sources):
        snap = metrics.get_snapshot()

        assert snap["executor"]["success_rate"] == 0.8
        assert snap["claude_vault"]["cache_hit_rate"] == 0.84
        assert snap["board"]["queue_depth"] == 5
        assert snap["ollama"]["status"] == "healthy"
        assert snap["stale_seconds"] == 0.0
        # Each source called exactly once.
        assert patched_sources.executor.call_count == 1
        assert patched_sources.claude_vault.call_count == 1
        assert patched_sources.board.call_count == 1
        assert patched_sources.ollama.call_count == 1

    def test_second_call_within_ttl_hits_cache(self, patched_sources):
        first = metrics.get_snapshot()
        second = metrics.get_snapshot()

        # Underlying sources are NOT re-invoked.
        assert patched_sources.executor.call_count == 1
        assert patched_sources.claude_vault.call_count == 1
        # Same payload values, but a non-zero stale_seconds on the cached read.
        assert second["executor"] == first["executor"]
        assert second["stale_seconds"] >= 0.0

    def test_force_refresh_bypasses_cache(self, patched_sources):
        metrics.get_snapshot()
        metrics.get_snapshot(force_refresh=True)

        assert patched_sources.executor.call_count == 2

    def test_cache_expires_after_ttl(self, patched_sources, monkeypatch):
        """When the cached_at timestamp is older than CACHE_TTL_SECONDS,
        the next call must rebuild from the live sources."""
        metrics.get_snapshot()
        # Push the cache age past the TTL by rewinding the recorded timestamp.
        with metrics._cache_lock:
            metrics._cached_at = _time.monotonic() - (metrics.CACHE_TTL_SECONDS + 5)

        metrics.get_snapshot()
        assert patched_sources.executor.call_count == 2


# =============================================================================
# STALE FALLBACK
# =============================================================================


class TestStaleFallback:
    def test_failure_after_good_payload_serves_stale(self, patched_sources):
        """If a source raises after a successful call, the next refresh
        must fall back to the prior payload and report stale_seconds."""
        # First call seeds the cache.
        metrics.get_snapshot()

        # Force the cache to be considered expired so the next call
        # actually attempts a refresh (which we'll make blow up).
        with metrics._cache_lock:
            metrics._cached_at = _time.monotonic() - (metrics.CACHE_TTL_SECONDS + 1)

        patched_sources.executor.side_effect = RuntimeError("sqlite locked")

        snap = metrics.get_snapshot()
        # Falls back to last_good payload.
        assert snap["executor"]["success_rate"] == 0.8
        assert snap["claude_vault"]["cache_hit_rate"] == 0.84
        # Reports the cache age in seconds.
        assert isinstance(snap["stale_seconds"], float)
        assert snap["stale_seconds"] > 0

    def test_failure_with_no_cache_returns_empty_shell(self, patched_sources):
        """First-ever call that fails returns an empty-but-valid payload
        with stale_seconds=None so callers can tell there's no good data."""
        metrics._reset_cache()
        patched_sources.board.side_effect = RuntimeError("provider gone")

        snap = metrics.get_snapshot()
        assert snap["executor"] == {}
        assert snap["board"] == {}
        assert snap["stale_seconds"] is None
        assert "provider gone" in snap.get("error", "")


# =============================================================================
# PERCENTILE / EXECUTOR AGGREGATION
# =============================================================================


class TestPercentile:
    def test_empty_returns_zero(self):
        assert metrics._percentile([], 50) == 0.0
        assert metrics._percentile([], 95) == 0.0

    def test_single_value(self):
        assert metrics._percentile([42.0], 50) == 42.0
        assert metrics._percentile([42.0], 95) == 42.0

    def test_p50_p95(self):
        # Ten evenly-spaced values: nearest-rank p50 picks the 5th (500),
        # p95 picks the 10th (1000).
        values = [100.0, 200.0, 300.0, 400.0, 500.0,
                  600.0, 700.0, 800.0, 900.0, 1000.0]
        assert metrics._percentile(values, 50) == 500.0
        assert metrics._percentile(values, 95) == 1000.0

    def test_executor_metrics_with_no_rows_returns_zero_success_rate(
        self, monkeypatch
    ):
        """A fresh database (no executor_runs in the last 24h) returns
        zeroes everywhere instead of dividing by zero."""
        # Fake the DB layer with a connection whose execute() returns no rows.
        class _FakeCursor:
            def fetchall(self) -> list:
                return []

        class _FakeConn:
            def execute(self, *args, **kwargs):
                return _FakeCursor()

        monkeypatch.setattr(metrics.executor_runs_db, "init_db", lambda: None)
        monkeypatch.setattr(
            metrics.executor_runs_db, "_get_conn", lambda: _FakeConn()
        )

        result = metrics._executor_metrics()
        assert result["total_runs_24h"] == 0
        assert result["successes_24h"] == 0
        assert result["success_rate"] == 0.0
        assert result["p50_latency_ms"] == 0.0
        assert result["p95_latency_ms"] == 0.0


# =============================================================================
# BOARD METRICS — oldest top-ranked wait
# =============================================================================


class TestBoardMetrics:
    def test_queue_depth_counts_proposed_and_approved(self, monkeypatch):
        items = [
            SimpleNamespace(state="proposed", created="2026-04-17T00:00:00"),
            SimpleNamespace(state="approved", created="2026-04-17T01:00:00"),
            SimpleNamespace(state="approved", created="2026-04-16T12:00:00"),
            SimpleNamespace(state="executing", created="2026-04-17T02:00:00"),
            SimpleNamespace(state="done", created="2026-04-17T03:00:00"),
        ]
        provider = MagicMock()
        provider.load_all.return_value = items
        monkeypatch.setattr(metrics, "get_provider", lambda: provider)

        result = metrics._board_metrics()
        assert result["queue_depth"] == 3  # 1 proposed + 2 approved
        # Oldest approved is 2026-04-16T12:00:00, so the wait is positive.
        assert result["oldest_top_ranked_wait_seconds"] is not None
        assert result["oldest_top_ranked_wait_seconds"] > 0

    def test_no_approved_items_means_no_wait(self, monkeypatch):
        items = [
            SimpleNamespace(state="proposed", created="2026-04-17T00:00:00"),
            SimpleNamespace(state="done", created="2026-04-17T03:00:00"),
        ]
        provider = MagicMock()
        provider.load_all.return_value = items
        monkeypatch.setattr(metrics, "get_provider", lambda: provider)

        result = metrics._board_metrics()
        assert result["queue_depth"] == 1
        assert result["oldest_top_ranked_wait_seconds"] is None

    def test_bad_timestamp_does_not_crash(self, monkeypatch):
        items = [
            SimpleNamespace(state="approved", created="not-a-date"),
        ]
        provider = MagicMock()
        provider.load_all.return_value = items
        monkeypatch.setattr(metrics, "get_provider", lambda: provider)

        result = metrics._board_metrics()
        # 'approved' counts toward queue depth but the malformed timestamp
        # must not propagate as a ValueError out of the metrics layer.
        assert result["queue_depth"] == 1
        assert result["oldest_top_ranked_wait_seconds"] is None


# =============================================================================
# PROMETHEUS RENDERING
# =============================================================================


class TestPrometheusFormat:
    def test_render_emits_help_type_value_per_metric(self, patched_sources):
        text = metrics.render_prometheus()

        # Every metric line must come with a HELP and TYPE line.
        assert "# HELP technomancer_executor_success_rate" in text
        assert "# TYPE technomancer_executor_success_rate gauge" in text
        assert "technomancer_executor_success_rate 0.8" in text

        assert "# HELP technomancer_claude_vault_cache_hit_rate" in text
        assert "technomancer_claude_vault_cache_hit_rate 0.84" in text

        assert "technomancer_board_queue_depth 5.0" in text

        # Trailing newline is required by the Prometheus text exposition.
        assert text.endswith("\n")

    def test_render_handles_none_values(self, patched_sources):
        # Simulate "no approved items" — wait field is None.
        patched_sources.board.return_value = {
            "queue_depth": 0,
            "oldest_top_ranked_wait_seconds": None,
        }
        # Force a rebuild so the new return value is picked up.
        text = metrics.render_prometheus()

        assert "technomancer_board_oldest_top_ranked_wait_seconds NaN" in text

    def test_render_emits_one_line_per_ollama_status(self, patched_sources):
        patched_sources.ollama.return_value = {
            "status": "degraded",
            "consecutive_failures": 3,
        }
        text = metrics.render_prometheus()

        assert "technomancer_ollama_status_healthy 0" in text
        assert "technomancer_ollama_status_degraded 1" in text
        assert "technomancer_ollama_status_down 0" in text
        assert "technomancer_ollama_status_unknown 0" in text
        assert "technomancer_ollama_consecutive_failures 3.0" in text

    def test_render_does_not_import_prometheus_client(self, patched_sources):
        """The whole point of this module is being usable without the
        ``prometheus_client`` dependency installed. Sanity-check that
        ``render_prometheus`` doesn't reach for it."""
        import sys

        sentinel = object()
        original = sys.modules.pop("prometheus_client", sentinel)
        try:
            # Block re-import attempts during the call.
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            text = metrics.render_prometheus()
            assert "technomancer_executor_success_rate" in text
        finally:
            sys.modules.pop("prometheus_client", None)
            if original is not sentinel:
                sys.modules["prometheus_client"] = original  # type: ignore[assignment]
