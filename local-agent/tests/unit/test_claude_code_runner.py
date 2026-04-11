"""Tests for agent/claude_code_runner.py — session management and run_claude_prompt."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_code_runner import (
    ChatSession,
    end_session,
    get_active_session,
    run_claude_prompt,
    _active_sessions,
)


class TestSessionManagement:
    """In-memory dict operations — no mocks needed."""

    def setup_method(self):
        _active_sessions.clear()

    def test_no_active_session(self):
        assert get_active_session(12345) is None

    def test_get_active_session(self):
        session = ChatSession(session_id="abc", user="test", channel_id=12345)
        _active_sessions[12345] = session
        assert get_active_session(12345) is session

    def test_end_session(self):
        session = ChatSession(session_id="abc", user="test", channel_id=12345)
        _active_sessions[12345] = session
        ended = end_session(12345)
        assert ended is session
        assert get_active_session(12345) is None

    def test_end_nonexistent_session(self):
        assert end_session(99999) is None


class TestChatSession:
    def test_dataclass_fields(self):
        s = ChatSession(session_id="abc", user="test", channel_id=1)
        assert s.session_id == "abc"
        assert s.user == "test"
        assert s.channel_id == 1
        assert s.turn_count == 0
        assert s.total_cost_usd == 0.0


class TestRunClaudePrompt:
    """Mock subprocess to test the shared utility."""

    def test_binary_not_found(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = run_claude_prompt("test")
            assert result["success"] is False
            assert "not found" in result["error"]

    def test_successful_json_response(self):
        import json

        fake_output = json.dumps({
            "result": "Hello world!",
            "session_id": "sess-123",
            "total_cost_usd": 0.01,
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("Say hello")
                assert result["success"] is True
                assert result["result"] == "Hello world!"
                assert result["cost_usd"] == 0.01
                assert result["session_id"] == "sess-123"

    def test_nonzero_exit_code(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Some error"

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is False
                assert "Some error" in result["error"]

    def test_timeout(self):
        import subprocess

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch(
                "agent.claude_code_runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=30),
            ):
                result = run_claude_prompt("test", timeout=30)
                assert result["success"] is False
                assert "Timed out" in result["error"]

    def test_malformed_json_fallback(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Not valid JSON but still output"
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is True
                assert result["result"] == "Not valid JSON but still output"

    def test_strips_api_key_from_env(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "ok"}'
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result) as mock_run:
                with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "secret123", "CLAUDECODE": "1"}):
                    run_claude_prompt("test")
                    call_env = mock_run.call_args.kwargs.get("env", {})
                    assert "ANTHROPIC_API_KEY" not in call_env
                    assert "CLAUDECODE" not in call_env
