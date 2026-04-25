"""Tests for Ollama model metrics in prometheus_metrics module."""

from unittest.mock import MagicMock, patch

import pytest

from agent.prometheus_metrics import (
    record_ollama_model_load,
    record_ollama_model_unload,
    set_ollama_model_resident,
)


class TestOllamaModelMetrics:
    """Test Ollama model load/unload/residency metrics."""

    def test_record_ollama_model_load_updates_counter(self):
        """record_ollama_model_load increments the load counter."""
        mock_loads_counter = MagicMock()
        mock_unloads_counter = MagicMock()
        mock_resident_gauge = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._ollama_model_loads_total", mock_loads_counter), \
             patch("agent.prometheus_metrics._ollama_model_unloads_total", mock_unloads_counter), \
             patch("agent.prometheus_metrics._ollama_model_resident", mock_resident_gauge):
            record_ollama_model_load("qwen3.5:27b")

            mock_loads_counter.labels.assert_called_with(model="qwen3.5:27b")
            mock_loads_counter.labels().inc.assert_called_once()

            # Unloads counter should not be called
            mock_unloads_counter.labels().inc.assert_not_called()
            # Gauge should not be called
            mock_resident_gauge.labels().set.assert_not_called()

    def test_record_ollama_model_unload_updates_counter(self):
        """record_ollama_model_unload increments the unload counter."""
        mock_loads_counter = MagicMock()
        mock_unloads_counter = MagicMock()
        mock_resident_gauge = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._ollama_model_loads_total", mock_loads_counter), \
             patch("agent.prometheus_metrics._ollama_model_unloads_total", mock_unloads_counter), \
             patch("agent.prometheus_metrics._ollama_model_resident", mock_resident_gauge):
            record_ollama_model_unload("llama3.2")

            mock_unloads_counter.labels.assert_called_with(model="llama3.2")
            mock_unloads_counter.labels().inc.assert_called_once()

            # Loads counter should not be called
            mock_loads_counter.labels().inc.assert_not_called()
            # Gauge should not be called
            mock_resident_gauge.labels().set.assert_not_called()

    def test_set_ollama_model_resident_updates_gauge(self):
        """set_ollama_model_resident sets the gauge value."""
        mock_loads_counter = MagicMock()
        mock_unloads_counter = MagicMock()
        mock_resident_gauge = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._ollama_model_loads_total", mock_loads_counter), \
             patch("agent.prometheus_metrics._ollama_model_unloads_total", mock_unloads_counter), \
             patch("agent.prometheus_metrics._ollama_model_resident", mock_resident_gauge):
            set_ollama_model_resident("qwen3.5:27b", True)

            mock_resident_gauge.labels.assert_called_with(model="qwen3.5:27b")
            mock_resident_gauge.labels().set.assert_called_with(1)

            # Counter should not be called
            mock_loads_counter.labels().inc.assert_not_called()
            mock_unloads_counter.labels().inc.assert_not_called()

    def test_set_ollama_model_resident_false(self):
        """set_ollama_model_resident sets gauge to 0 when resident=False."""
        mock_loads_counter = MagicMock()
        mock_unloads_counter = MagicMock()
        mock_resident_gauge = MagicMock()

        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", True), \
             patch("agent.prometheus_metrics._ollama_model_loads_total", mock_loads_counter), \
             patch("agent.prometheus_metrics._ollama_model_unloads_total", mock_unloads_counter), \
             patch("agent.prometheus_metrics._ollama_model_resident", mock_resident_gauge):
            set_ollama_model_resident("llama3.2", False)

            mock_resident_gauge.labels.assert_called_with(model="llama3.2")
            mock_resident_gauge.labels().set.assert_called_with(0)

            # Counter should not be called
            mock_loads_counter.labels().inc.assert_not_called()
            mock_unloads_counter.labels().inc.assert_not_called()

    def test_noop_when_prometheus_not_installed(self):
        """All functions are no-ops when prometheus_client is not installed."""
        with patch("agent.prometheus_metrics._initialized", True), \
             patch("agent.prometheus_metrics.HAS_PROMETHEUS", False):
            # Should not raise
            record_ollama_model_load("qwen")
            record_ollama_model_unload("llama")
            set_ollama_model_resident("qwen", True)