"""Tests for the fallback_orchestrator module — local-first Claude API fallback."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.fallback_orchestrator import (
    FallbackOrchestrator,
    get_fallback_status,
    get_fallback_tools,
)


class TestFallbackOrchestrator:
    """Test the FallbackOrchestrator class."""

    def _make_orch(self, **kwargs):
        defaults = {
            "max_consecutive_errors": 3,
            "latency_threshold": 10.0,
            "latency_window": 3,
            "recovery_cooldown": 60,
        }
        defaults.update(kwargs)
        return FallbackOrchestrator(**defaults)

    def test_no_fallback_initially(self):
        orch = self._make_orch()
        assert orch.should_use_fallback() is False

    def test_no_fallback_after_success(self):
        orch = self._make_orch()
        orch.record_claude_result(success=True, latency=1.0)
        assert orch.should_use_fallback() is False

    def test_fallback_after_consecutive_errors(self):
        orch = self._make_orch(max_consecutive_errors=3)
        orch.record_claude_result(success=False, error="rate limited")
        orch.record_claude_result(success=False, error="rate limited")
        assert orch.should_use_fallback() is False  # only 2 errors
        orch.record_claude_result(success=False, error="rate limited")
        assert orch.should_use_fallback() is True  # 3 errors -> activated

    def test_success_resets_error_count(self):
        orch = self._make_orch(max_consecutive_errors=3)
        orch.record_claude_result(success=False, error="err1")
        orch.record_claude_result(success=False, error="err2")
        orch.record_claude_result(success=True, latency=1.0)  # resets count
        orch.record_claude_result(success=False, error="err3")
        assert orch.should_use_fallback() is False  # only 1 consecutive error

    def test_fallback_on_high_latency(self):
        orch = self._make_orch(latency_threshold=5.0, latency_window=3)
        orch.record_claude_result(success=True, latency=6.0)
        orch.record_claude_result(success=True, latency=7.0)
        assert orch.should_use_fallback() is False  # only 2 readings
        orch.record_claude_result(success=True, latency=8.0)
        assert orch.should_use_fallback() is True  # avg 7.0 > threshold 5.0

    def test_no_fallback_on_acceptable_latency(self):
        orch = self._make_orch(latency_threshold=10.0, latency_window=3)
        orch.record_claude_result(success=True, latency=3.0)
        orch.record_claude_result(success=True, latency=4.0)
        orch.record_claude_result(success=True, latency=5.0)
        assert orch.should_use_fallback() is False  # avg 4.0 < threshold

    def test_fallback_stays_active_during_cooldown(self):
        orch = self._make_orch(max_consecutive_errors=2, recovery_cooldown=9999)
        orch.record_claude_result(success=False, error="err")
        orch.record_claude_result(success=False, error="err")
        # During cooldown, should route to fallback
        assert orch.should_use_fallback() is True

    def test_probe_allowed_after_cooldown(self):
        orch = self._make_orch(max_consecutive_errors=2, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="err")
        orch.record_claude_result(success=False, error="err")
        # Cooldown is 0 — probe immediately allowed (returns False to let Claude try)
        time.sleep(0.05)
        assert orch.should_use_fallback() is False

    def test_recovery_on_successful_probe(self):
        orch = self._make_orch(max_consecutive_errors=2, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="err")
        orch.record_claude_result(success=False, error="err")
        # Probe succeeds
        time.sleep(0.05)
        orch.record_claude_result(success=True, latency=1.0)
        # Fully recovered
        assert orch.should_use_fallback() is False

    def test_probe_failure_keeps_fallback_active(self):
        orch = self._make_orch(max_consecutive_errors=2, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="err1")
        orch.record_claude_result(success=False, error="err2")
        # Probe fails — fallback should remain active
        time.sleep(0.05)
        orch.record_claude_result(success=False, error="err3")
        status = orch.get_status()
        assert status["fallback_active"] is True
        assert status["consecutive_errors"] == 3

    def test_force_fallback(self):
        orch = self._make_orch()
        orch.force_fallback("manual test")
        assert orch.should_use_fallback() is True

    def test_force_recover(self):
        orch = self._make_orch()
        orch.force_fallback("test")
        orch.force_recover()
        assert orch.should_use_fallback() is False

    def test_reset(self):
        orch = self._make_orch(max_consecutive_errors=2)
        orch.record_claude_result(success=False, error="err")
        orch.record_claude_result(success=False, error="err")
        assert orch.should_use_fallback() is True

        orch.reset()
        assert orch.should_use_fallback() is False
        status = orch.get_status()
        assert status["consecutive_errors"] == 0
        assert status["total_activations"] == 0

    def test_get_status(self):
        orch = self._make_orch()
        orch.record_claude_result(success=True, latency=2.5)
        orch.record_claude_result(success=False, error="timeout")

        status = orch.get_status()
        assert status["fallback_active"] is False
        assert status["consecutive_errors"] == 1
        assert status["last_error"] == "timeout"
        assert "max_consecutive_errors" in status
        assert "recent_events" in status

    def test_status_tracks_activations(self):
        orch = self._make_orch(max_consecutive_errors=1, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="err1")  # activates

        status = orch.get_status()
        assert status["fallback_active"] is True
        assert status["total_activations"] == 1
        assert len(status["recent_events"]) == 1
        assert status["recent_events"][0]["type"] == "activated"

    def test_status_tracks_recovery(self):
        orch = self._make_orch(max_consecutive_errors=1, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="err1")
        time.sleep(0.05)
        orch.should_use_fallback()  # allow probe
        orch.record_claude_result(success=True, latency=1.0)

        status = orch.get_status()
        assert status["fallback_active"] is False
        events = status["recent_events"]
        types = [e["type"] for e in events]
        assert "activated" in types
        assert "recovered" in types

    def test_total_fallback_requests_counted(self):
        orch = self._make_orch(max_consecutive_errors=1, recovery_cooldown=9999)
        orch.record_claude_result(success=False, error="err")
        # These calls should count as fallback requests
        orch.should_use_fallback()
        orch.should_use_fallback()
        orch.should_use_fallback()

        status = orch.get_status()
        assert status["total_fallback_requests"] == 3

    def test_latency_window_trimmed(self):
        orch = self._make_orch(latency_window=3)
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0]:
            orch.record_claude_result(success=True, latency=lat)
        # Only last 3 should be kept
        with orch._lock:
            assert len(orch._recent_latencies) == 3
            assert orch._recent_latencies == [3.0, 4.0, 5.0]

    def test_error_clears_latency_window(self):
        orch = self._make_orch()
        orch.record_claude_result(success=True, latency=1.0)
        orch.record_claude_result(success=True, latency=2.0)
        orch.record_claude_result(success=False, error="err")
        with orch._lock:
            assert len(orch._recent_latencies) == 0

    def test_thread_safety(self):
        orch = self._make_orch(max_consecutive_errors=100)
        errors = []

        def record_many(success: bool, count: int):
            try:
                for _ in range(count):
                    orch.record_claude_result(success=success, latency=0.1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=(True, 50)),
            threading.Thread(target=record_many, args=(False, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_events_capped_at_50(self):
        orch = self._make_orch(max_consecutive_errors=1, recovery_cooldown=0)
        # Generate many events by toggling fallback
        for _ in range(60):
            orch.record_claude_result(success=False, error="err")
            time.sleep(0.01)
            orch.should_use_fallback()
            orch.record_claude_result(success=True, latency=0.1)

        with orch._lock:
            assert len(orch._events) <= 50


class TestNotificationIntegration:
    """Test that alerts are sent to the dedicated alerts channel."""

    @patch("agent.alerts.send_alert")
    def test_activation_sends_alert(self, mock_send):
        orch = FallbackOrchestrator(max_consecutive_errors=1)
        orch.record_claude_result(success=False, error="blocked")

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "fallback activated" in call_args[0][0].lower()
        assert call_args[1]["title"] == "API Fallback"
        assert call_args[1]["level"] == "warning"

    @patch("agent.alerts.send_alert")
    def test_recovery_sends_alert(self, mock_send):
        orch = FallbackOrchestrator(max_consecutive_errors=1, recovery_cooldown=0)
        orch.record_claude_result(success=False, error="blocked")
        mock_send.reset_mock()

        time.sleep(0.05)
        orch.should_use_fallback()
        orch.record_claude_result(success=True, latency=1.0)

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "available again" in call_args[0][0].lower()
        assert call_args[1]["title"] == "API Fallback Recovered"
        assert call_args[1]["level"] == "success"

    @patch("agent.alerts.send_alert", side_effect=Exception("webhook down"))
    def test_alert_failure_doesnt_crash(self, mock_send):
        orch = FallbackOrchestrator(max_consecutive_errors=1)
        # Should not raise
        orch.record_claude_result(success=False, error="blocked")


class TestConvenienceFunctions:
    """Test module-level functions and tools."""

    def test_get_fallback_status_returns_string(self):
        status = get_fallback_status()
        assert isinstance(status, str)
        assert "Claude API Fallback Orchestrator" in status

    def test_get_fallback_tools_returns_list(self):
        tools = get_fallback_tools()
        assert len(tools) == 1
        assert tools[0].name == "fallback_status"

    def test_fallback_tool_function_runs(self):
        tools = get_fallback_tools()
        result = tools[0].function()
        assert "Fallback Orchestrator" in result


class TestClaudeBridgeFallbackIntegration:
    """Test that ClaudeBridge integrates with the fallback orchestrator."""

    @patch("agent.claude_bridge.anthropic", create=True)
    @patch("agent.claude_bridge.HAS_ANTHROPIC", True)
    def test_bridge_checks_fallback_before_api(self, mock_anthropic):
        """ClaudeBridge.send() checks should_use_fallback before calling Claude."""
        from agent.claude_bridge import ClaudeBridge

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        bridge = ClaudeBridge(mode="api", api_key="fake-key", use_vault_context=False)
        bridge.client = mock_client

        with patch("agent.fallback_orchestrator.should_use_fallback", return_value=False):
            mock_usage = MagicMock(input_tokens=10, output_tokens=20)
            mock_content = MagicMock(text="Claude response")
            mock_response = MagicMock(content=[mock_content], usage=mock_usage)
            mock_client.messages.create.return_value = mock_response

            result = bridge.send("test question")
            assert "Claude response" in result
            mock_client.messages.create.assert_called_once()

    @patch("agent.claude_bridge.anthropic", create=True)
    @patch("agent.claude_bridge.HAS_ANTHROPIC", True)
    def test_bridge_uses_ollama_when_fallback_active(self, mock_anthropic):
        """When fallback is active, route to Ollama instead of Claude."""
        from agent.claude_bridge import ClaudeBridge

        bridge = ClaudeBridge(mode="api", api_key="fake-key", use_vault_context=False)
        bridge.client = MagicMock()

        with patch("agent.fallback_orchestrator.should_use_fallback", return_value=True), \
             patch.object(bridge, "_send_ollama_fallback", return_value="[Ollama fallback]\nOllama says hi") as mock_fb:
            result = bridge.send("test question")
            mock_fb.assert_called_once()
            assert "Ollama" in result
            # Claude API should NOT have been called
            bridge.client.messages.create.assert_not_called()
