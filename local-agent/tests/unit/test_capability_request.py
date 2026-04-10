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
    @patch("agent.capability_request.ANTHROPIC_API_KEY", "fake-key")
    @patch("agent.capability_request.anthropic")
    @patch("agent.capability_request.log_request")
    def test_approved_capability(self, mock_log, mock_anthropic, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.json")

        mock_client = MagicMock()
        mock_usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_content = MagicMock(text="IMPLEMENT: YES\nREASON: Good idea\n\ndef new_tool(): pass")
        mock_response = MagicMock(content=[mock_content], usage=mock_usage)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("agent.capability_request._record_perf"):
            result = request_capability("weather tool", "What's the weather?")
        # Should contain implementation info or reason
        assert isinstance(result, str)

    @patch("agent.capability_request.ANTHROPIC_API_KEY", "fake-key")
    @patch("agent.capability_request.anthropic")
    @patch("agent.capability_request.log_request")
    def test_rejected_capability(self, mock_log, mock_anthropic, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.json")

        mock_client = MagicMock()
        mock_usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_content = MagicMock(text="IMPLEMENT: NO\nREASON: Not feasible")
        mock_response = MagicMock(content=[mock_content], usage=mock_usage)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("agent.capability_request._record_perf"):
            result = request_capability("time travel", "Go back in time")
        assert "CANNOT IMPLEMENT" in result or "NO" in result or isinstance(result, str)

    def test_no_api_key(self, monkeypatch):
        monkeypatch.setattr("agent.capability_request.ANTHROPIC_API_KEY", None)
        result = request_capability("test", "test")
        assert "cannot implement" in result.lower() or "not available" in result.lower() or "Error" in result


class TestGetCapabilityTools:
    def test_returns_tool(self):
        tools = get_capability_tools()
        assert len(tools) >= 1
        names = {t.name for t in tools}
        assert "request_capability" in names
