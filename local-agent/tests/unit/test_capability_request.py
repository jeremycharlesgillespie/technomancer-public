"""Tests for capability_request — self-improvement capability evaluation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.capability_request import get_capability_tools, log_request, read_current_tools, request_capability


class TestLogRequest:
    def test_logs_to_file(self, tmp_path, monkeypatch):
        log_file = tmp_path / "capability_requests.log"
        monkeypatch.setattr("agent.capability_request.LOG_FILE", log_file)

        log_request("new_tool", "Add a weather tool", "APPROVED")
        assert log_file.exists()
        content = log_file.read_text()
        assert "new_tool" in content
        assert "Add a weather tool" in content

    def test_appends_to_existing(self, tmp_path, monkeypatch):
        log_file = tmp_path / "capability_requests.log"
        monkeypatch.setattr("agent.capability_request.LOG_FILE", log_file)

        log_request("type1", "desc1", "OK")
        log_request("type2", "desc2", "OK")
        content = log_file.read_text()
        assert "type1" in content
        assert "type2" in content


class TestReadCurrentTools:
    def test_returns_string(self):
        result = read_current_tools()
        assert isinstance(result, str)
        assert len(result) > 0


class TestRequestCapability:
    @patch("agent.capability_request.log_request")
    def test_approved_capability(self, mock_log, mock_ollama_client, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.json")
        mock_ollama_client.set_responses([
            {"message": {"content": "IMPLEMENT: YES\nREASON: Good idea\n\ndef new_tool(): pass", "tool_calls": []}}
        ])

        with patch("agent.capability_request._record_perf"):
            result = request_capability("weather tool", "What's the weather?")
        assert isinstance(result, str)

    @patch("agent.capability_request.log_request")
    def test_rejected_capability(self, mock_log, mock_ollama_client, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.json")
        mock_ollama_client.set_responses([
            {"message": {"content": "IMPLEMENT: NO\nREASON: Not feasible\nALTERNATIVE: Use existing tool", "tool_calls": []}}
        ])

        with patch("agent.capability_request._record_perf"):
            result = request_capability("time travel", "Go back in time")
        assert "CANNOT IMPLEMENT" in result

    def test_handles_ollama_error(self, mock_ollama_client, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.json")
        mock_ollama_client.set_responses([])  # Empty = will raise

        with patch("agent.capability_request._record_perf"):
            with patch("agent.capability_request.log_request"):
                result = request_capability("test", "test")
        assert "CANNOT IMPLEMENT" in result or "Error" in result


class TestGetCapabilityTools:
    def test_returns_tool(self):
        tools = get_capability_tools()
        assert len(tools) >= 1
        names = {t.name for t in tools}
        assert "request_capability" in names
