"""Tests for the perf_monitor module — LLM endpoint performance tracking."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.perf_monitor import CallRecord, PerfMonitor, get_endpoint_summary, get_monitor


class TestCallRecord:
    """Test CallRecord dataclass."""

    def test_create_record(self):
        rec = CallRecord(
            timestamp="2026-04-06T12:00:00",
            endpoint="ollama",
            duration=1.5,
            success=True,
            input_tokens=100,
            output_tokens=50,
            model="qwen3.5:27b",
        )
        assert rec.endpoint == "ollama"
        assert rec.duration == 1.5
        assert rec.success is True
        assert rec.input_tokens == 100

    def test_default_values(self):
        rec = CallRecord(
            timestamp="2026-04-06T12:00:00",
            endpoint="claude_api",
            duration=0.5,
            success=False,
        )
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.model == ""
        assert rec.error == ""


class TestPerfMonitor:
    """Test PerfMonitor class."""

    def test_record_and_get_stats(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.0, True, input_tokens=100, output_tokens=50, model="qwen")
        mon.record("ollama", 2.0, True, input_tokens=200, output_tokens=100, model="qwen")
        mon.record("ollama", 0.5, False, model="qwen", error="Connection refused")

        stats = mon.get_endpoint_stats("ollama")
        assert stats["calls"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["success_rate"] == pytest.approx(66.7, abs=0.1)
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150

    def test_empty_stats(self):
        mon = PerfMonitor()
        stats = mon.get_endpoint_stats("ollama")
        assert stats == {"calls": 0}

    def test_filter_by_endpoint(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.0, True)
        mon.record("claude_api", 0.5, True)
        mon.record("ollama", 2.0, True)

        ollama_stats = mon.get_endpoint_stats("ollama")
        assert ollama_stats["calls"] == 2

        claude_stats = mon.get_endpoint_stats("claude_api")
        assert claude_stats["calls"] == 1

        all_stats = mon.get_endpoint_stats()
        assert all_stats["calls"] == 3

    def test_latency_stats(self):
        mon = PerfMonitor()
        for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mon.record("ollama", d, True)

        stats = mon.get_endpoint_stats("ollama")
        assert stats["min_latency"] == 1.0
        assert stats["max_latency"] == 5.0
        assert stats["avg_latency"] == 3.0
        assert stats["p50_latency"] == 3.0

    def test_max_records_trim(self):
        mon = PerfMonitor(max_records=5)
        for i in range(10):
            mon.record("ollama", float(i), True)

        stats = mon.get_endpoint_stats()
        assert stats["calls"] == 5
        # Should keep the last 5 (durations 5-9)
        assert stats["min_latency"] == 5.0

    def test_get_recent_errors(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.0, True)
        mon.record("ollama", 0.5, False, error="Timeout")
        mon.record("claude_api", 0.3, False, error="Rate limited")

        errors = mon.get_recent_errors(5)
        assert len(errors) == 2
        assert errors[0]["error"] == "Timeout"
        assert errors[1]["endpoint"] == "claude_api"

    def test_get_recent_errors_limit(self):
        mon = PerfMonitor()
        for i in range(10):
            mon.record("ollama", 0.1, False, error=f"Error {i}")

        errors = mon.get_recent_errors(3)
        assert len(errors) == 3
        assert errors[0]["error"] == "Error 7"

    def test_error_field_cleared_on_success(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.0, True, error="should be cleared")

        with mon._lock:
            assert mon._records[0].error == ""

    def test_reset(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.0, True)
        mon.record("claude_api", 0.5, True)
        mon.reset()

        stats = mon.get_endpoint_stats()
        assert stats == {"calls": 0}

    def test_get_summary_empty(self):
        mon = PerfMonitor()
        summary = mon.get_summary()
        assert "No LLM call data" in summary

    def test_get_summary_with_data(self):
        mon = PerfMonitor()
        mon.record("ollama", 1.5, True, input_tokens=100, output_tokens=50, model="qwen")
        mon.record("claude_api", 0.8, True, input_tokens=500, output_tokens=200, model="sonnet")
        mon.record("claude_api", 0.3, False, model="sonnet", error="Rate limit")

        summary = mon.get_summary()
        assert "Endpoint Metrics" in summary
        assert "ollama" in summary
        assert "claude_api" in summary
        assert "3 total calls" in summary
        assert "Rate limit" in summary

    def test_track_context_manager_success(self):
        mon = PerfMonitor()
        with mon.track("ollama", model="qwen") as ctx:
            time.sleep(0.01)
            ctx["input_tokens"] = 42
            ctx["output_tokens"] = 10

        stats = mon.get_endpoint_stats("ollama")
        assert stats["calls"] == 1
        assert stats["successes"] == 1
        assert stats["total_input_tokens"] == 42
        assert stats["avg_latency"] >= 0.01

    def test_track_context_manager_failure(self):
        mon = PerfMonitor()
        with pytest.raises(ValueError, match="boom"):
            with mon.track("claude_api", model="sonnet"):
                raise ValueError("boom")

        stats = mon.get_endpoint_stats("claude_api")
        assert stats["calls"] == 1
        assert stats["failures"] == 1

        errors = mon.get_recent_errors()
        assert "boom" in errors[0]["error"]

    def test_thread_safety(self):
        mon = PerfMonitor()
        errors = []

        def record_many(endpoint: str, count: int):
            try:
                for i in range(count):
                    mon.record(endpoint, 0.01 * i, True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=("ollama", 100)),
            threading.Thread(target=record_many, args=("claude_api", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = mon.get_endpoint_stats()
        assert stats["calls"] == 200


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    def test_get_monitor_returns_singleton(self):
        m1 = get_monitor()
        m2 = get_monitor()
        assert m1 is m2

    def test_get_endpoint_summary_returns_string(self):
        # Reset global monitor to avoid pollution from other tests
        get_monitor().reset()
        result = get_endpoint_summary()
        assert isinstance(result, str)


class TestInstrumentationIntegration:
    """Test that perf_monitor instrumentation is wired into LLM call sites."""

    def test_image_identification_vision_records_perf(self):
        """analyze_with_vision_model records to perf_monitor on success."""
        mon = PerfMonitor()
        mock_response = {"message": {"content": "A cat sitting on a couch"}}
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response

        with patch("agent.image_identification._vision_client", mock_client), \
             patch("agent.image_identification._record_perf") as mock_perf:
            from agent.image_identification import analyze_with_vision_model

            result = analyze_with_vision_model([b"fake_image"])
            assert "cat" in result

            mock_perf.assert_called_once()
            args, kwargs = mock_perf.call_args
            assert args[0] == "ollama_vision"
            assert kwargs["success"] is True
            assert kwargs["model"] == "llava-llama3"

    def test_image_identification_vision_records_failure(self):
        """analyze_with_vision_model records failure on exception."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Ollama down")

        with patch("agent.image_identification._vision_client", mock_client), \
             patch("agent.image_identification._record_perf") as mock_perf:
            from agent.image_identification import analyze_with_vision_model

            result = analyze_with_vision_model([b"fake_image"])
            assert "Error" in result

            mock_perf.assert_called_once()
            args, kwargs = mock_perf.call_args
            assert args[0] == "ollama_vision"
            assert kwargs["success"] is False
            assert "Ollama down" in kwargs["error"]

    def test_image_identification_claude_records_perf(self):
        """ask_claude_with_image records to perf_monitor on success (via Pro sub)."""
        fake_result = {"success": True, "result": "This is Pikachu from Pokemon", "cost_usd": 0}

        with patch("agent.claude_code_runner.run_claude_prompt", return_value=fake_result), \
             patch("agent.image_identification._record_perf") as mock_perf:

            from agent.image_identification import ask_claude_with_image

            result = ask_claude_with_image(b"\x89PNG" + b"\x00" * 100, "Who is this?")
            assert "Pikachu" in result

            mock_perf.assert_called_once()
            args, kwargs = mock_perf.call_args
            assert args[0] == "claude_pro_sub"
            assert kwargs["success"] is True

    def test_capability_request_records_perf(self, mock_ollama_client):
        """request_capability records to perf_monitor (via Ollama)."""
        mock_ollama_client.set_responses([
            {"message": {"content": "IMPLEMENT: NO\nREASON: Not feasible", "tool_calls": []}}
        ])

        with patch("agent.capability_request._record_perf") as mock_perf, \
             patch("agent.capability_request.log_request"):

            from agent.capability_request import request_capability

            result = request_capability("get weather", "What's the weather?")
            assert "CANNOT IMPLEMENT" in result

            mock_perf.assert_called_once()
            args, kwargs = mock_perf.call_args
            assert args[0] == "ollama"
            assert kwargs["success"] is True

    # test_claude_coder_records_perf_per_turn removed — claude_coder.py was deleted as legacy code
