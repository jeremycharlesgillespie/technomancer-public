"""Tests for the /api/metrics and /metrics Flask endpoints in idea_board.web."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import metrics
from idea_board.web import app


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
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


_FAKE_SNAPSHOT = {
    "executor": {
        "total_runs_24h": 10,
        "successes_24h": 8,
        "success_rate": 0.8,
        "p50_latency_ms": 1500.0,
        "p95_latency_ms": 9000.0,
    },
    "claude_vault": {
        "calls": 42,
        "cache_hit_rate": 0.84,
        "input_tokens": 1000,
        "cache_read_tokens": 8400,
        "cache_creation_tokens": 600,
    },
    "board": {
        "queue_depth": 5,
        "oldest_top_ranked_wait_seconds": 3600.0,
    },
    "ollama": {
        "status": "healthy",
        "last_checked": 0.0,
        "last_success": 0.0,
        "last_error": "",
        "consecutive_failures": 0,
    },
    "generated_at": "2026-04-17T00:00:00",
    "stale_seconds": 0.0,
}


# =============================================================================
# /api/metrics — JSON endpoint
# =============================================================================


class TestApiMetricsJson:
    def test_returns_200_and_application_json(self, client):
        with patch("idea_board.web.metrics.get_snapshot",
                   return_value=_FAKE_SNAPSHOT):
            resp = client.get("/api/metrics")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type

    def test_body_matches_snapshot(self, client):
        with patch("idea_board.web.metrics.get_snapshot",
                   return_value=_FAKE_SNAPSHOT):
            resp = client.get("/api/metrics")
        data = resp.get_json()
        assert data["executor"]["success_rate"] == 0.8
        assert data["claude_vault"]["cache_hit_rate"] == 0.84
        assert data["board"]["queue_depth"] == 5
        assert data["ollama"]["status"] == "healthy"

    def test_calls_get_snapshot_exactly_once(self, client):
        with patch("idea_board.web.metrics.get_snapshot",
                   return_value=_FAKE_SNAPSHOT) as mock_snap:
            client.get("/api/metrics")
        assert mock_snap.call_count == 1

    def test_second_call_reuses_cache(self, client):
        """Two requests in quick succession should hit the metrics cache
        rather than rebuilding — the endpoint must not force_refresh."""
        with patch("idea_board.web.metrics.get_snapshot",
                   return_value=_FAKE_SNAPSHOT) as mock_snap:
            client.get("/api/metrics")
            client.get("/api/metrics")
        # Endpoint calls get_snapshot() each hit; the cache lives
        # inside get_snapshot() itself. Ensure no force_refresh=True
        # was passed, which would defeat the cache.
        for call in mock_snap.call_args_list:
            assert call.kwargs.get("force_refresh", False) is False


# =============================================================================
# /metrics — Prometheus exposition
# =============================================================================


class TestPrometheusEndpoint:
    def test_returns_200_and_plain_text(self, client):
        with patch("idea_board.web.metrics.render_prometheus",
                   return_value="# HELP x test\n# TYPE x gauge\nx 1.0\n"):
            resp = client.get("/metrics")
        assert resp.status_code == 200
        # Flask appends charset=utf-8 — match on the mimetype prefix only.
        assert "text/plain" in resp.content_type
        assert "version=0.0.4" in resp.content_type

    def test_body_matches_render_prometheus(self, client):
        body = (
            "# HELP technomancer_executor_success_rate test\n"
            "# TYPE technomancer_executor_success_rate gauge\n"
            "technomancer_executor_success_rate 0.8\n"
        )
        with patch("idea_board.web.metrics.render_prometheus",
                   return_value=body):
            resp = client.get("/metrics")
        assert resp.data.decode("utf-8") == body

    def test_real_render_produces_valid_prometheus_text(self, client):
        """End-to-end: patch the snapshot sources but let render_prometheus
        run for real, then assert the response looks like valid exposition."""
        with patch("agent.metrics._executor_metrics",
                   return_value=_FAKE_SNAPSHOT["executor"]), \
             patch("agent.metrics._claude_vault_metrics",
                   return_value=_FAKE_SNAPSHOT["claude_vault"]), \
             patch("agent.metrics._board_metrics",
                   return_value=_FAKE_SNAPSHOT["board"]), \
             patch("agent.metrics._ollama_metrics",
                   return_value=_FAKE_SNAPSHOT["ollama"]):
            resp = client.get("/metrics")

        assert resp.status_code == 200
        text = resp.data.decode("utf-8")
        # Must contain HELP/TYPE/sample triples for at least one metric.
        assert "# HELP technomancer_executor_success_rate" in text
        assert "# TYPE technomancer_executor_success_rate gauge" in text
        assert "technomancer_executor_success_rate 0.8" in text
        # Every non-comment, non-empty line must parse as "name value".
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            assert len(parts) == 2, f"Bad Prometheus line: {line!r}"
            name, value = parts
            assert name, f"Empty metric name: {line!r}"
            # Value must be a float, int, or "NaN".
            if value != "NaN":
                float(value)  # raises if not a valid number


# =============================================================================
# Both endpoints share the 30s snapshot cache
# =============================================================================


class TestSharedCache:
    def test_api_metrics_and_prometheus_share_cache(self, client):
        """Hitting /api/metrics then /metrics should build the snapshot
        only once — the prior story's 30s cache is what both endpoints
        depend on to avoid slamming SQLite."""
        with patch("agent.metrics._executor_metrics",
                   return_value=_FAKE_SNAPSHOT["executor"]) as exec_mock, \
             patch("agent.metrics._claude_vault_metrics",
                   return_value=_FAKE_SNAPSHOT["claude_vault"]), \
             patch("agent.metrics._board_metrics",
                   return_value=_FAKE_SNAPSHOT["board"]), \
             patch("agent.metrics._ollama_metrics",
                   return_value=_FAKE_SNAPSHOT["ollama"]):
            r1 = client.get("/api/metrics")
            r2 = client.get("/metrics")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Single source build across both endpoints, served from cache.
        assert exec_mock.call_count == 1
