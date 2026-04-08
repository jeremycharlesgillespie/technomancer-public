"""Tests for the prometheus_metrics module — Prometheus metrics for LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from agent.perf_monitor import PerfMonitor


class TestRecordPrometheus:
    """Test that record_prometheus updates Prometheus counters/histograms."""

    def test_record_success_updates_metrics(self):
        """A successful call updates duration histogram, call counter, and token counters."""
        mock_histogram = MagicMock()
        mock_calls_counter = MagicMock()
        mock_errors_counter = MagicMock()
        mock_input_counter = MagicMock()
        mock_output_counter = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._llm_call_duration", mock_histogram), \
             patch("agent.prometheus_metrics._llm_calls_total", mock_calls_counter), \
             patch("agent.prometheus_metrics._llm_call_errors_total", mock_errors_counter), \
             patch("agent.prometheus_metrics._llm_input_tokens_total", mock_input_counter), \
             patch("agent.prometheus_metrics._llm_output_tokens_total", mock_output_counter):
            from agent.prometheus_metrics import record_prometheus

            record_prometheus(
                endpoint="ollama",
                duration=1.5,
                success=True,
                input_tokens=100,
                output_tokens=50,
                model="qwen3.5:27b",
            )

            mock_histogram.labels.assert_called_with(endpoint="ollama", model="qwen3.5:27b")
            mock_histogram.labels().observe.assert_called_with(1.5)

            mock_calls_counter.labels.assert_called_with(
                endpoint="ollama", model="qwen3.5:27b", status="success"
            )
            mock_calls_counter.labels().inc.assert_called_once()

            # No error counter on success
            mock_errors_counter.labels.assert_not_called()

            mock_input_counter.labels().inc.assert_called_with(100)
            mock_output_counter.labels().inc.assert_called_with(50)

    def test_record_failure_updates_error_counter(self):
        """A failed call increments the error counter."""
        mock_histogram = MagicMock()
        mock_calls_counter = MagicMock()
        mock_errors_counter = MagicMock()
        mock_input_counter = MagicMock()
        mock_output_counter = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._llm_call_duration", mock_histogram), \
             patch("agent.prometheus_metrics._llm_calls_total", mock_calls_counter), \
             patch("agent.prometheus_metrics._llm_call_errors_total", mock_errors_counter), \
             patch("agent.prometheus_metrics._llm_input_tokens_total", mock_input_counter), \
             patch("agent.prometheus_metrics._llm_output_tokens_total", mock_output_counter):
            from agent.prometheus_metrics import record_prometheus

            record_prometheus(
                endpoint="claude_api",
                duration=0.3,
                success=False,
                model="sonnet",
            )

            mock_calls_counter.labels.assert_called_with(
                endpoint="claude_api", model="sonnet", status="error"
            )
            mock_errors_counter.labels.assert_called_with(
                endpoint="claude_api", model="sonnet"
            )
            mock_errors_counter.labels().inc.assert_called_once()

    def test_record_zero_tokens_skips_token_counters(self):
        """Zero tokens should not increment token counters."""
        mock_histogram = MagicMock()
        mock_calls_counter = MagicMock()
        mock_errors_counter = MagicMock()
        mock_input_counter = MagicMock()
        mock_output_counter = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._llm_call_duration", mock_histogram), \
             patch("agent.prometheus_metrics._llm_calls_total", mock_calls_counter), \
             patch("agent.prometheus_metrics._llm_call_errors_total", mock_errors_counter), \
             patch("agent.prometheus_metrics._llm_input_tokens_total", mock_input_counter), \
             patch("agent.prometheus_metrics._llm_output_tokens_total", mock_output_counter):
            from agent.prometheus_metrics import record_prometheus

            record_prometheus(
                endpoint="ollama",
                duration=1.0,
                success=True,
                input_tokens=0,
                output_tokens=0,
                model="qwen",
            )

            mock_input_counter.labels().inc.assert_not_called()
            mock_output_counter.labels().inc.assert_not_called()

    def test_noop_when_prometheus_not_installed(self):
        """record_prometheus is a no-op when prometheus_client is not installed."""
        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", False):
            from agent.prometheus_metrics import record_prometheus

            # Should not raise
            record_prometheus(
                endpoint="ollama", duration=1.0, success=True, model="qwen"
            )


class TestStartMetricsServer:
    """Test start_metrics_server."""

    def test_start_server_success(self):
        """start_metrics_server returns True on success."""
        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics.start_http_server") as mock_start:
            from agent.prometheus_metrics import start_metrics_server

            result = start_metrics_server(port=9999)
            assert result is True
            mock_start.assert_called_once_with(9999)

    def test_start_server_port_in_use(self):
        """start_metrics_server returns False if port is in use."""
        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics.start_http_server", side_effect=OSError("in use")):
            from agent.prometheus_metrics import start_metrics_server

            result = start_metrics_server(port=9999)
            assert result is False

    def test_start_server_no_prometheus(self):
        """start_metrics_server returns False when prometheus_client not installed."""
        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", False):
            from agent.prometheus_metrics import start_metrics_server

            result = start_metrics_server()
            assert result is False


class TestPerfMonitorPrometheusIntegration:
    """Test that PerfMonitor.record() forwards to Prometheus metrics."""

    def test_record_calls_prometheus(self):
        """PerfMonitor.record() calls record_prometheus."""
        mon = PerfMonitor()
        with patch("agent.prometheus_metrics.record_prometheus") as mock_prom:
            mon.record("ollama", 1.5, True, input_tokens=100, output_tokens=50, model="qwen")

            mock_prom.assert_called_once_with(
                endpoint="ollama",
                duration=1.5,
                success=True,
                input_tokens=100,
                output_tokens=50,
                model="qwen",
            )

    def test_track_context_manager_calls_prometheus(self):
        """PerfMonitor.track() context manager also forwards to Prometheus."""
        mon = PerfMonitor()
        with patch("agent.prometheus_metrics.record_prometheus") as mock_prom:
            with mon.track("claude_api", model="sonnet") as ctx:
                ctx["input_tokens"] = 500
                ctx["output_tokens"] = 200

            mock_prom.assert_called_once()
            call_kwargs = mock_prom.call_args[1]
            assert call_kwargs["endpoint"] == "claude_api"
            assert call_kwargs["success"] is True
            assert call_kwargs["input_tokens"] == 500
            assert call_kwargs["output_tokens"] == 200
