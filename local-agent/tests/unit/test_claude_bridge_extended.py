"""Extended tests for claude_bridge — CLI mode, fallback, tool registration."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_bridge import ClaudeBridge, ask_claude, escalate_to_claude, get_claude_tools, report_to_claude


class TestClaudeBridgeInit:
    def test_auto_mode_with_api(self, monkeypatch):
        monkeypatch.setattr("agent.claude_bridge.HAS_ANTHROPIC", True)
        bridge = ClaudeBridge(api_key="fake-key")
        assert bridge.mode == "api"

    def test_auto_mode_without_api(self, monkeypatch):
        monkeypatch.setattr("agent.claude_bridge.HAS_ANTHROPIC", False)
        bridge = ClaudeBridge(mode="auto", api_key=None)
        assert bridge.mode == "cli"

    def test_explicit_cli_mode(self):
        bridge = ClaudeBridge(mode="cli")
        assert bridge.mode == "cli"


class TestBuildPrompt:
    def test_escalate_type(self):
        bridge = ClaudeBridge(mode="cli")
        prompt = bridge._build_prompt("Fix this bug", "Some context", "escalate")
        assert "ESCALATED" in prompt
        assert "Fix this bug" in prompt
        assert "Some context" in prompt

    def test_report_type(self):
        bridge = ClaudeBridge(mode="cli")
        prompt = bridge._build_prompt("Found a bug", "", "report")
        assert "REPORT" in prompt

    def test_question_type(self):
        bridge = ClaudeBridge(mode="cli")
        prompt = bridge._build_prompt("How does X work?", "", "question")
        assert "QUESTION" in prompt

    def test_general_type(self):
        bridge = ClaudeBridge(mode="cli")
        prompt = bridge._build_prompt("Hello", "", "general")
        assert "Hello" in prompt


class TestSendCli:
    @patch("subprocess.run")
    def test_successful_cli(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Claude says hello", stderr=""
        )
        bridge = ClaudeBridge(mode="cli")
        result = bridge._send_cli("test prompt")
        assert "Claude says hello" in result

    @patch("subprocess.run")
    def test_cli_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        bridge = ClaudeBridge(mode="cli", timeout=120)
        result = bridge._send_cli("test prompt")
        assert "Timed out" in result or "Error" in result

    @patch("subprocess.run", side_effect=Exception("binary not found"))
    def test_cli_exception(self, mock_run):
        bridge = ClaudeBridge(mode="cli")
        result = bridge._send_cli("test prompt")
        assert "Error" in result


class TestSendOllamaFallback:
    @patch("agent.core.Agent")
    def test_fallback_returns_prefixed(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Ollama response here"
        mock_agent_cls.return_value = mock_agent

        bridge = ClaudeBridge(mode="cli")
        result = bridge._send_ollama_fallback("What is Python?")
        assert "Ollama fallback" in result or "Ollama response" in result

    @patch("agent.core.Agent")
    def test_fallback_handles_error(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("Ollama down")
        mock_agent_cls.return_value = mock_agent

        bridge = ClaudeBridge(mode="cli")
        result = bridge._send_ollama_fallback("test")
        assert "Error" in result or "failed" in result.lower()


class TestMaybeOllamaFallback:
    @patch("agent.fallback_orchestrator.should_use_fallback", return_value=True)
    @patch.object(ClaudeBridge, "_send_ollama_fallback", return_value="fallback response")
    def test_uses_fallback_when_active(self, mock_fb, mock_should):
        bridge = ClaudeBridge(mode="cli")
        result = bridge._maybe_ollama_fallback("prompt", "Error: rate limited")
        assert result == "fallback response"

    @patch("agent.fallback_orchestrator.should_use_fallback", return_value=False)
    def test_returns_error_when_not_active(self, mock_should):
        bridge = ClaudeBridge(mode="cli")
        result = bridge._maybe_ollama_fallback("prompt", "Error: something")
        assert result == "Error: something"


class TestConvenienceFunctions:
    @patch("agent.claude_bridge._get_bridge")
    def test_escalate(self, mock_bridge):
        mock_bridge.return_value = MagicMock()
        mock_bridge.return_value.escalate.return_value = "done"
        assert escalate_to_claude("task") == "done"

    @patch("agent.claude_bridge._get_bridge")
    def test_report(self, mock_bridge):
        mock_bridge.return_value = MagicMock()
        mock_bridge.return_value.report.return_value = "noted"
        assert report_to_claude("findings") == "noted"

    @patch("agent.claude_bridge._get_bridge")
    def test_ask(self, mock_bridge):
        mock_bridge.return_value = MagicMock()
        mock_bridge.return_value.ask.return_value = "answer"
        assert ask_claude("question") == "answer"


class TestGetClaudeTools:
    def test_returns_three_tools(self):
        tools = get_claude_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "escalate_to_claude" in names
        assert "report_to_claude" in names
        assert "ask_claude" in names
