"""Tests for agent/ollama_health.py — retries, backoff, and health monitor."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from ollama._types import ResponseError

from agent import ollama_health
from agent.ollama_health import (
    STATUS_DEGRADED,
    STATUS_DOWN,
    STATUS_HEALTHY,
    STATUS_UNKNOWN,
    OllamaHealthMonitor,
    _compute_backoff,
    check_ollama_ready,
    is_transient_error,
    ollama_call_with_retries,
)


@pytest.fixture(autouse=True)
def _fresh_monitor(monkeypatch):
    """Swap the module-level singleton for a fresh monitor in each test."""
    fresh = OllamaHealthMonitor()
    monkeypatch.setattr(ollama_health, "_monitor", fresh)
    yield fresh
    fresh.stop()


@pytest.fixture
def _no_sleep(monkeypatch):
    """Skip time.sleep so retry tests don't block."""
    monkeypatch.setattr(ollama_health.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# is_transient_error
# ---------------------------------------------------------------------------


class TestIsTransientError:
    def test_connection_error_is_transient(self):
        assert is_transient_error(requests.ConnectionError("refused")) is True

    def test_requests_timeout_is_transient(self):
        assert is_transient_error(requests.Timeout("timed out")) is True

    def test_stdlib_connection_error_is_transient(self):
        assert is_transient_error(ConnectionRefusedError("nope")) is True

    def test_stdlib_timeout_is_transient(self):
        assert is_transient_error(TimeoutError("slow")) is True

    def test_response_error_503_is_transient(self):
        assert is_transient_error(ResponseError("unavailable", 503)) is True

    def test_response_error_429_is_transient(self):
        assert is_transient_error(ResponseError("rate limited", 429)) is True

    def test_response_error_500_is_transient(self):
        assert is_transient_error(ResponseError("server error", 500)) is True

    def test_response_error_404_not_transient(self):
        # Unknown model / not found — don't retry.
        assert is_transient_error(ResponseError("model not found", 404)) is False

    def test_response_error_400_not_transient(self):
        assert is_transient_error(ResponseError("bad request", 400)) is False

    def test_value_error_not_transient(self):
        assert is_transient_error(ValueError("bad arg")) is False

    def test_httpx_connect_error_is_transient(self):
        import httpx

        assert is_transient_error(httpx.ConnectError("unreachable")) is True

    def test_httpx_read_timeout_is_transient(self):
        import httpx

        assert is_transient_error(httpx.ReadTimeout("slow")) is True


# ---------------------------------------------------------------------------
# _compute_backoff
# ---------------------------------------------------------------------------


class TestComputeBackoff:
    def test_first_attempt_uses_base(self):
        # attempt=0, jitter=0 → exactly base.
        with patch("agent.ollama_health.random.uniform", return_value=0.0):
            assert _compute_backoff(0, base_delay=1.0, max_delay=30.0) == 1.0

    def test_second_attempt_doubles(self):
        with patch("agent.ollama_health.random.uniform", return_value=0.0):
            assert _compute_backoff(1, base_delay=1.0, max_delay=30.0) == 2.0

    def test_third_attempt_quadruples(self):
        with patch("agent.ollama_health.random.uniform", return_value=0.0):
            assert _compute_backoff(2, base_delay=1.0, max_delay=30.0) == 4.0

    def test_capped_at_max_delay(self):
        # attempt=10, base=1 → would be 1024s without cap.
        with patch("agent.ollama_health.random.uniform", return_value=0.0):
            assert _compute_backoff(10, base_delay=1.0, max_delay=5.0) == 5.0

    def test_jitter_applied(self):
        # With a fixed random.uniform, jitter is deterministic.
        with patch("agent.ollama_health.random.uniform", return_value=0.5):
            # attempt=1: delay=2, jitter=0.5 → 2.5
            assert _compute_backoff(1, base_delay=1.0, max_delay=30.0) == 2.5

    def test_jitter_nonnegative_and_bounded(self):
        # No mock — real jitter. Run many times; every result in [delay, 1.5*delay].
        for _ in range(50):
            r = _compute_backoff(0, base_delay=2.0, max_delay=30.0, jitter_factor=0.5)
            assert 2.0 <= r <= 3.0


# ---------------------------------------------------------------------------
# ollama_call_with_retries
# ---------------------------------------------------------------------------


class TestOllamaCallWithRetries:
    def test_success_first_try_no_retry(self, _no_sleep):
        fn = MagicMock(return_value={"message": {"content": "ok"}})
        result = ollama_call_with_retries(fn, model="x", max_retries=3, base_delay=0.001)
        assert result == {"message": {"content": "ok"}}
        assert fn.call_count == 1

    def test_retries_then_succeeds(self, _fresh_monitor, _no_sleep):
        fn = MagicMock(side_effect=[
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            {"message": {"content": "recovered"}},
        ])
        result = ollama_call_with_retries(fn, max_retries=3, base_delay=0.001, max_delay=0.01)
        assert result == {"message": {"content": "recovered"}}
        assert fn.call_count == 3
        # Recovery after retries → healthy.
        assert _fresh_monitor.get_status() == STATUS_HEALTHY

    def test_permanent_error_fails_fast(self, _fresh_monitor, _no_sleep):
        """404 / unknown model is never retried."""
        fn = MagicMock(side_effect=ResponseError("model not found", 404))
        with pytest.raises(ResponseError):
            ollama_call_with_retries(fn, max_retries=5, base_delay=0.001)
        # Fail-fast means exactly one call.
        assert fn.call_count == 1

    def test_exhausted_retries_raises_and_marks_down(self, _fresh_monitor, _no_sleep):
        fn = MagicMock(side_effect=requests.ConnectionError("nope"))
        with pytest.raises(requests.ConnectionError):
            ollama_call_with_retries(fn, max_retries=2, base_delay=0.001, max_delay=0.01)
        # retries=2 → 3 total attempts.
        assert fn.call_count == 3
        assert _fresh_monitor.get_status() == STATUS_DOWN

    def test_transient_marks_degraded_between_attempts(self, _fresh_monitor, _no_sleep):
        statuses: list[str] = []

        def record_then_succeed(*a, **k):
            # Capture status after each degraded mark.
            statuses.append(_fresh_monitor.get_status())
            if len(statuses) < 3:
                raise requests.ConnectionError("refused")
            return "ok"

        result = ollama_call_with_retries(
            record_then_succeed, max_retries=5, base_delay=0.001, max_delay=0.01
        )
        assert result == "ok"
        # First call: unknown. Subsequent: degraded after mark.
        assert STATUS_DEGRADED in statuses[1:]
        # After final success, monitor is healthy.
        assert _fresh_monitor.get_status() == STATUS_HEALTHY

    def test_on_transient_error_hook_invoked(self, _no_sleep):
        hook = MagicMock()
        fn = MagicMock(side_effect=[
            requests.ConnectionError("x"),
            requests.ConnectionError("x"),
            "ok",
        ])
        ollama_call_with_retries(
            fn, max_retries=3, base_delay=0.001, max_delay=0.01,
            on_transient_error=hook,
        )
        assert hook.call_count == 2
        # Hook receives (exception, attempt_index).
        for call_args in hook.call_args_list:
            exc, attempt = call_args.args
            assert isinstance(exc, requests.ConnectionError)
            assert isinstance(attempt, int)

    def test_hook_exceptions_do_not_break_retries(self, _no_sleep):
        fn = MagicMock(side_effect=[requests.ConnectionError("x"), "ok"])
        hook = MagicMock(side_effect=RuntimeError("hook blew up"))
        # Should still succeed on the second attempt even though hook raised.
        assert ollama_call_with_retries(
            fn, max_retries=2, base_delay=0.001, on_transient_error=hook
        ) == "ok"

    def test_uses_settings_defaults_when_args_none(self, _no_sleep):
        """When retry args are omitted, values come from settings."""
        fn = MagicMock(side_effect=[requests.ConnectionError("x"), "ok"])
        with patch("agent.ollama_health.settings.ollama_max_retries", 1), \
             patch("agent.ollama_health.settings.ollama_retry_base_delay", 0.001), \
             patch("agent.ollama_health.settings.ollama_retry_max_delay", 0.01):
            assert ollama_call_with_retries(fn) == "ok"
        assert fn.call_count == 2

    def test_kwargs_passed_through(self, _no_sleep):
        fn = MagicMock(return_value="ok")
        ollama_call_with_retries(fn, model="m", messages=[1, 2], think=True,
                                 max_retries=1, base_delay=0.001)
        fn.assert_called_once_with(model="m", messages=[1, 2], think=True)


# ---------------------------------------------------------------------------
# OllamaHealthMonitor — state transitions
# ---------------------------------------------------------------------------


class TestHealthMonitorState:
    def test_initial_status_unknown(self):
        m = OllamaHealthMonitor()
        assert m.get_status() == STATUS_UNKNOWN

    def test_mark_healthy_transitions(self):
        m = OllamaHealthMonitor()
        m.mark_healthy()
        assert m.get_status() == STATUS_HEALTHY
        assert m.is_healthy() is True
        state = m.get_state()
        assert state["consecutive_failures"] == 0
        assert state["last_success"] > 0

    def test_mark_degraded_transitions(self):
        m = OllamaHealthMonitor()
        m.mark_degraded("refused")
        assert m.get_status() == STATUS_DEGRADED
        state = m.get_state()
        assert state["last_error"] == "refused"
        assert state["consecutive_failures"] == 1

    def test_mark_down_overrides_degraded(self):
        m = OllamaHealthMonitor()
        m.mark_degraded("first")
        m.mark_down("exhausted")
        assert m.get_status() == STATUS_DOWN

    def test_degraded_does_not_override_down(self):
        m = OllamaHealthMonitor()
        m.mark_down("dead")
        m.mark_degraded("blip")
        # Once down, stay down until explicitly healthy.
        assert m.get_status() == STATUS_DOWN

    def test_healthy_clears_failures(self):
        m = OllamaHealthMonitor()
        m.mark_degraded("1")
        m.mark_degraded("2")
        assert m.get_state()["consecutive_failures"] == 2
        m.mark_healthy()
        assert m.get_state()["consecutive_failures"] == 0
        assert m.get_state()["last_error"] == ""

    def test_snapshot_contains_expected_keys(self):
        m = OllamaHealthMonitor()
        state = m.get_state()
        for key in ("status", "last_checked", "last_success",
                    "last_error", "consecutive_failures"):
            assert key in state


# ---------------------------------------------------------------------------
# OllamaHealthMonitor.check_once — HTTP probe
# ---------------------------------------------------------------------------


class TestHealthMonitorCheckOnce:
    def test_200_marks_healthy(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.requests.get", return_value=resp):
            status = m.check_once()
        assert status == STATUS_HEALTHY

    def test_503_marks_degraded(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=503)
        with patch("agent.ollama_health.requests.get", return_value=resp):
            status = m.check_once()
        assert status == STATUS_DEGRADED
        assert "503" in m.get_state()["last_error"]

    def test_connection_error_marks_down(self):
        m = OllamaHealthMonitor()
        with patch("agent.ollama_health.requests.get",
                   side_effect=requests.ConnectionError("refused")):
            status = m.check_once()
        assert status == STATUS_DOWN
        assert "ConnectionError" in m.get_state()["last_error"]

    def test_timeout_marks_down(self):
        m = OllamaHealthMonitor()
        with patch("agent.ollama_health.requests.get",
                   side_effect=requests.Timeout("slow")):
            status = m.check_once()
        assert status == STATUS_DOWN

    def test_4xx_marks_degraded_not_down(self):
        """Server responded — reachable but unhappy. Degraded, not down."""
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=404)
        with patch("agent.ollama_health.requests.get", return_value=resp):
            status = m.check_once()
        assert status == STATUS_DEGRADED

    def test_check_uses_configured_host(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.settings.ollama_host", "http://custom:9999/"), \
             patch("agent.ollama_health.requests.get", return_value=resp) as mock_get:
            m.check_once()
        # Trailing slash stripped, /api/tags appended.
        called_url = mock_get.call_args.args[0]
        assert called_url == "http://custom:9999/api/tags"

    def test_check_uses_configured_timeout(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.settings.ollama_health_check_timeout", 2.5), \
             patch("agent.ollama_health.requests.get", return_value=resp) as mock_get:
            m.check_once()
        assert mock_get.call_args.kwargs["timeout"] == 2.5

    def test_last_checked_updated_even_on_failure(self):
        m = OllamaHealthMonitor()
        before = time.time()
        with patch("agent.ollama_health.requests.get",
                   side_effect=requests.ConnectionError("x")):
            m.check_once()
        assert m.get_state()["last_checked"] >= before


# ---------------------------------------------------------------------------
# OllamaHealthMonitor — lifecycle
# ---------------------------------------------------------------------------


class TestHealthMonitorLifecycle:
    def test_start_is_idempotent(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.requests.get", return_value=resp), \
             patch("agent.ollama_health.settings.ollama_health_check_interval", 3600):
            m.start()
            t1 = m._thread
            m.start()  # Second start is a no-op.
            t2 = m._thread
        assert t1 is t2
        m.stop()

    def test_stop_sets_stop_event(self):
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.requests.get", return_value=resp), \
             patch("agent.ollama_health.settings.ollama_health_check_interval", 3600):
            m.start()
            assert not m._stop_event.is_set()
            m.stop()
            assert m._stop_event.is_set()

    def test_thread_runs_check_once_on_start(self):
        """Monitor performs an immediate probe so status flips off 'unknown'."""
        m = OllamaHealthMonitor()
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.requests.get", return_value=resp), \
             patch("agent.ollama_health.settings.ollama_health_check_interval", 3600):
            m.start()
            # Give the thread a beat to run its first check.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if m.get_status() != STATUS_UNKNOWN:
                    break
                time.sleep(0.01)
            m.stop()
        assert m.get_status() == STATUS_HEALTHY


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_get_monitor_returns_singleton(self, _fresh_monitor):
        assert ollama_health.get_monitor() is _fresh_monitor

    def test_get_ollama_status_returns_snapshot(self, _fresh_monitor):
        _fresh_monitor.mark_healthy()
        snap = ollama_health.get_ollama_status()
        assert snap["status"] == STATUS_HEALTHY
        # Snapshot is a plain dict (safe to serialize).
        assert isinstance(snap, dict)

    def test_start_monitor_delegates(self, _fresh_monitor):
        resp = MagicMock(status_code=200)
        with patch("agent.ollama_health.requests.get", return_value=resp), \
             patch("agent.ollama_health.settings.ollama_health_check_interval", 3600):
            ollama_health.start_monitor()
            assert _fresh_monitor._thread is not None
            assert _fresh_monitor._thread.is_alive()
            ollama_health.stop_monitor()


# ---------------------------------------------------------------------------
# check_ollama_ready — pre-flight readiness probe
# ---------------------------------------------------------------------------


def _tags_response(model_names: list[str]) -> MagicMock:
    """Helper: build a mock requests.get response with the given models."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return resp


class TestCheckOllamaReady:
    def test_success_returns_ready(self):
        """Happy path: /api/tags lists the model and generate succeeds."""
        tags_resp = _tags_response(["qwen3.5:27b", "llava-llama3:latest"])
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "ok"}

        with patch("agent.ollama_health.requests.get", return_value=tags_resp), \
             patch("agent.core._ollama_client", mock_client):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=2.0)

        assert ok is True
        assert "ready" in reason
        # Generate called with 1-token limit (the actual point of warmup).
        args, kwargs = mock_client.generate.call_args
        assert kwargs["model"] == "qwen3.5:27b"
        assert kwargs["options"] == {"num_predict": 1}

    def test_connection_refused_returns_false(self):
        """Ollama server down: requests raises ConnectionError."""
        with patch(
            "agent.ollama_health.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=1.0)

        assert ok is False
        assert "connection refused" in reason.lower()

    def test_missing_model_returns_false(self):
        """Server is up, but requested model isn't in /api/tags."""
        tags_resp = _tags_response(["llava-llama3:latest", "nomic-embed-text:latest"])
        with patch("agent.ollama_health.requests.get", return_value=tags_resp):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=1.0)

        assert ok is False
        assert "not loaded" in reason
        assert "qwen3.5:27b" in reason

    def test_generate_timeout_returns_false(self):
        """Generate warmup exceeds the configured timeout."""
        tags_resp = _tags_response(["qwen3.5:27b"])
        mock_client = MagicMock()

        def _slow_generate(*_args, **_kwargs):
            time.sleep(0.5)  # Longer than the 0.05s timeout below.
            return {"response": "finally"}

        mock_client.generate.side_effect = _slow_generate

        with patch("agent.ollama_health.requests.get", return_value=tags_resp), \
             patch("agent.core._ollama_client", mock_client):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=0.05)

        assert ok is False
        assert "timeout" in reason.lower()

    def test_matches_model_by_base_name(self):
        """A bare name like 'qwen3.5' matches a listed 'qwen3.5:27b'."""
        tags_resp = _tags_response(["qwen3.5:27b"])
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "ok"}

        with patch("agent.ollama_health.requests.get", return_value=tags_resp), \
             patch("agent.core._ollama_client", mock_client):
            ok, reason = check_ollama_ready("qwen3.5", timeout=2.0)

        assert ok is True, reason

    def test_tags_http_error_returns_false(self):
        """/api/tags returning non-200 is a failure."""
        resp = MagicMock(status_code=500)
        with patch("agent.ollama_health.requests.get", return_value=resp):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=1.0)

        assert ok is False
        assert "500" in reason

    def test_generate_response_error_returns_false(self):
        """Generate raising ResponseError (e.g. model 404) is handled."""
        tags_resp = _tags_response(["qwen3.5:27b"])
        mock_client = MagicMock()
        mock_client.generate.side_effect = ResponseError("not found", 404)

        with patch("agent.ollama_health.requests.get", return_value=tags_resp), \
             patch("agent.core._ollama_client", mock_client):
            ok, reason = check_ollama_ready("qwen3.5:27b", timeout=1.0)

        assert ok is False
        assert "404" in reason
