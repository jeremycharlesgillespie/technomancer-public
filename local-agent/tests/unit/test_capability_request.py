"""Tests for capability_request — self-improvement capability evaluation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.capability_request import (
    _implement_capability,
    get_capability_tools,
    log_request,
    read_current_tools,
    request_capability,
)
# Aliased to avoid pytest auto-collecting it as a test function
from agent.capability_request import test_capability as run_test_capability


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


# =============================================================================
# _implement_capability
# =============================================================================


SAMPLE_TOOLS_FILE = '''"""Dummy tools module for tests."""

from .core import Tool, create_tool


def existing_tool():
    """Example."""
    return "ok"


def get_system_tools() -> list[Tool]:
    return [
        create_tool(
            name="existing",
            description="",
            parameters={},
            function=existing_tool,
        ),
    ]


def get_all_tools() -> list[Tool]:
    return get_system_tools()
'''


@pytest.fixture
def fake_tools_file(tmp_path, monkeypatch):
    """Write a minimal tools.py to a temp path and patch TOOLS_FILE."""
    path = tmp_path / "tools.py"
    path.write_text(SAMPLE_TOOLS_FILE, encoding="utf-8")
    monkeypatch.setattr("agent.capability_request.TOOLS_FILE", path)
    monkeypatch.setattr("agent.capability_request.LOG_FILE", tmp_path / "log.log")
    return path


class TestImplementCapability:
    def test_fails_when_function_code_missing(self, fake_tools_file):
        response = "IMPLEMENT: YES\nREGISTRATION_CODE:\n```python\nfoo\n```"
        result = _implement_capability(response, "test")
        assert "FAILED" in result
        assert "function code" in result

    def test_fails_when_registration_code_missing(self, fake_tools_file):
        response = "IMPLEMENT: YES\nFUNCTION_CODE:\n```python\ndef foo(): pass\n```"
        result = _implement_capability(response, "test")
        assert "FAILED" in result
        assert "registration code" in result

    def test_fails_when_function_has_syntax_error(self, fake_tools_file):
        response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef broken(:\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='x')\n```\n"
        )
        result = _implement_capability(response, "test")
        assert "FAILED" in result
        assert "Syntax error" in result

    def test_rolls_back_on_reload_failure(self, fake_tools_file, monkeypatch):
        """When reload raises, tools.py is restored to original content."""
        original = fake_tools_file.read_text(encoding="utf-8")
        response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef new_tool():\n    return 'x'\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='new_tool')\n```\n"
            "INSERT_AFTER: END\n"
        )

        import importlib

        def fake_reload(_mod):
            raise RuntimeError("reload exploded")

        monkeypatch.setattr(importlib, "reload", fake_reload)
        result = _implement_capability(response, "test")
        assert "FAILED" in result
        assert "rolled back" in result.lower()
        # Verify file was restored
        assert fake_tools_file.read_text(encoding="utf-8") == original

    def test_success_with_insert_end(self, fake_tools_file, monkeypatch):
        """Successful implementation: inserts before get_all_tools marker."""
        response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef brand_new_tool():\n    return 'ok'\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='brand_new_tool')\n```\n"
            "INSERT_AFTER: END\n"
        )

        import importlib

        monkeypatch.setattr(importlib, "reload", lambda mod: mod)
        result = _implement_capability(response, "brand new tool")
        assert "SUCCESS" in result
        new_content = fake_tools_file.read_text(encoding="utf-8")
        assert "brand_new_tool" in new_content
        # Registration is added to get_system_tools list
        assert new_content.count("create_tool(name='brand_new_tool')") >= 1

    def test_success_with_insert_after_named_function(self, fake_tools_file, monkeypatch):
        """When INSERT_AFTER names a real function, inserts after it."""
        response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef next_tool():\n    return 'x'\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='next_tool')\n```\n"
            "INSERT_AFTER: existing_tool\n"
        )
        import importlib

        monkeypatch.setattr(importlib, "reload", lambda mod: mod)
        result = _implement_capability(response, "next tool")
        assert "SUCCESS" in result
        assert "next_tool" in fake_tools_file.read_text(encoding="utf-8")

    def test_unknown_insert_after_falls_back_to_marker(self, fake_tools_file, monkeypatch):
        """INSERT_AFTER: unknown_function falls back to before get_all_tools."""
        response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef stray():\n    return 'y'\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='stray')\n```\n"
            "INSERT_AFTER: totally_missing_function\n"
        )
        import importlib

        monkeypatch.setattr(importlib, "reload", lambda mod: mod)
        result = _implement_capability(response, "stray tool")
        assert "SUCCESS" in result

    def test_outer_exception_returns_failed(self, fake_tools_file, monkeypatch):
        """If anything inside _implement_capability raises, returns FAILED."""
        # Force re.search to raise
        monkeypatch.setattr(
            "agent.capability_request.re.search",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        result = _implement_capability("whatever", "test")
        assert "FAILED" in result


# =============================================================================
# test_capability
# =============================================================================


class TestTestCapability:
    def test_reports_missing_function(self):
        ok, msg = run_test_capability("nonexistent_function_xyz", {})
        assert ok is False
        assert "not found" in msg

    def test_runs_existing_function(self, monkeypatch):
        # Inject a fake tools module attribute for the duration of the test
        from agent import tools as tools_module

        def good(a, b):
            return a + b

        monkeypatch.setattr(tools_module, "_test_sum", good, raising=False)
        ok, msg = run_test_capability("_test_sum", {"a": 2, "b": 3})
        assert ok is True
        assert "5" in msg

    def test_returns_false_on_exception(self, monkeypatch):
        from agent import tools as tools_module

        def bomb():
            raise ValueError("no good")

        monkeypatch.setattr(tools_module, "_bomb", bomb, raising=False)
        ok, msg = run_test_capability("_bomb", {})
        assert ok is False
        assert "no good" in msg


# =============================================================================
# request_capability — YES-branch integration (covers _implement_capability call)
# =============================================================================


class TestRequestCapabilityYes:
    def test_yes_branch_routes_to_implement(
        self, mock_ollama_client, fake_tools_file, monkeypatch
    ):
        """When the LLM returns IMPLEMENT: YES, _implement_capability runs."""
        llm_response = (
            "IMPLEMENT: YES\n"
            "FUNCTION_CODE:\n```python\ndef auto_tool():\n    return 'a'\n```\n"
            "REGISTRATION_CODE:\n```python\ncreate_tool(name='auto_tool')\n```\n"
            "INSERT_AFTER: END\n"
        )
        mock_ollama_client.set_responses([
            {"message": {"content": llm_response, "tool_calls": []}}
        ])

        import importlib

        monkeypatch.setattr(importlib, "reload", lambda mod: mod)
        with patch("agent.capability_request._record_perf"):
            result = request_capability("auto tool", "please add one")

        # Either SUCCESS or one of the FAILED states — ensures the YES branch
        # is reached and the helper gets exercised
        assert any(kw in result for kw in ("SUCCESS", "FAILED"))
