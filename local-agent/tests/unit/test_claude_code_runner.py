"""Tests for agent/claude_code_runner.py — session management and run_claude_prompt."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_code_runner import (
    ChatSession,
    ChatResult,
    _find_claude_binary,
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


class TestFindClaudeBinary:
    def test_returns_none_when_no_extensions(self):
        with patch("pathlib.Path.exists", return_value=False):
            result = _find_claude_binary()
            # Either None or a valid path (if actually installed)
            assert result is None or result.exists()


class TestChatResult:
    def test_dataclass_fields(self):
        r = ChatResult(
            success=True, response="hello", session_id="abc",
            duration=1.5, cost_usd=0.01, is_new_session=True,
        )
        assert r.success is True
        assert r.response == "hello"
        assert r.session_id == "abc"
        assert r.cost_usd == 0.01

    def test_failed_result(self):
        r = ChatResult(
            success=False, response="Error: timeout",
            session_id="", duration=30.0, cost_usd=0, is_new_session=False,
        )
        assert r.success is False
        assert "Error" in r.response


class TestRunClaudePromptEdgeCases:
    def test_empty_stdout(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is True
                assert result["result"] == ""

    def test_general_exception(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", side_effect=OSError("Permission denied")):
                result = run_claude_prompt("test")
                assert result["success"] is False
                assert "Permission denied" in result["error"]

    def test_result_dict_always_has_all_keys(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = run_claude_prompt("test")
            assert "success" in result
            assert "result" in result
            assert "cost_usd" in result
            assert "session_id" in result
            assert "duration" in result
            assert "error" in result
