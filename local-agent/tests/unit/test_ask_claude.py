"""Tests for agent.ask_claude — Anthropic SDK path.

Verifies the public ``ask_claude(question, context)`` interface is preserved
after replacing the ``claude -p`` subprocess with a direct SDK call.
"""

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def reset_ask_claude_client(monkeypatch):
    """Reset the module-level Anthropic client singleton between tests."""
    import agent.ask_claude as ac_module

    monkeypatch.setattr(ac_module, "_client", None)
    return ac_module


@pytest.fixture
def mock_ask_claude_client(monkeypatch, reset_ask_claude_client):
    """Mock anthropic.Anthropic so ask_claude uses a fake client.

    Returns the fake client. `.messages.call_history` captures the kwargs
    passed to `messages.create` so tests can assert on the payload shape.
    """
    import agent.ask_claude as ac_module

    class MockContent:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class MockResponse:
        def __init__(self, text):
            self.content = [MockContent(text)]
            self.stop_reason = "end_turn"

    class MockMessages:
        def __init__(self):
            self.call_history = []
            self.response_text = "Claude's answer"
            self.raise_on_create = None

        def create(self, **kwargs):
            self.call_history.append(kwargs)
            if self.raise_on_create is not None:
                raise self.raise_on_create
            return MockResponse(self.response_text)

    class MockClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.messages = MockMessages()

    client = MockClient()

    def mock_init(*args, **kwargs):
        client.init_kwargs = kwargs
        return client

    monkeypatch.setattr("anthropic.Anthropic", mock_init)
    monkeypatch.setattr(ac_module, "HAS_ANTHROPIC", True)

    fake_settings = MagicMock()
    fake_settings.anthropic_api_key = "test-api-key"
    monkeypatch.setattr(ac_module, "settings", fake_settings)

    return client


@pytest.fixture
def patched_log_file(tmp_path, monkeypatch):
    """Redirect the log file so tests don't write to the real project dir."""
    import agent.ask_claude as ac_module

    log_path = tmp_path / "claude_queries.log"
    monkeypatch.setattr(ac_module, "LOG_FILE", log_path)
    return log_path


# =============================================================================
# TESTS: ask_claude basic flow
# =============================================================================


class TestAskClaudeBasic:
    """Happy-path tests for ask_claude with the SDK."""

    def test_returns_response_text(self, mock_ask_claude_client, patched_log_file):
        """ask_claude should return the text of the first content block, stripped."""
        mock_ask_claude_client.messages.response_text = "  Hello world  "

        from agent.ask_claude import ask_claude

        result = ask_claude("hi")

        assert result == "Hello world"

    def test_sends_expected_payload(self, mock_ask_claude_client, patched_log_file):
        """messages.create should be called with model, system, and user messages."""
        from agent.ask_claude import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, ask_claude

        ask_claude("What is 2+2?")

        assert len(mock_ask_claude_client.messages.call_history) == 1
        call = mock_ask_claude_client.messages.call_history[0]
        assert call["model"] == DEFAULT_MODEL
        assert call["system"] == DEFAULT_SYSTEM_PROMPT
        assert call["max_tokens"] == 4096
        assert call["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_context_is_prepended(self, mock_ask_claude_client, patched_log_file):
        """When context is provided, it should appear in the user message."""
        from agent.ask_claude import ask_claude

        ask_claude("What does it do?", context="def foo(): return 42")

        call = mock_ask_claude_client.messages.call_history[0]
        user_content = call["messages"][0]["content"]
        assert "def foo(): return 42" in user_content
        assert "What does it do?" in user_content

    def test_no_context_sends_bare_question(
        self, mock_ask_claude_client, patched_log_file
    ):
        """Without context, the user message is just the question."""
        from agent.ask_claude import ask_claude

        ask_claude("plain question")

        call = mock_ask_claude_client.messages.call_history[0]
        assert call["messages"][0]["content"] == "plain question"

    def test_timeout_passed_to_sdk(self, mock_ask_claude_client, patched_log_file):
        """The SDK timeout param should be set (not a subprocess timeout)."""
        from agent.ask_claude import ask_claude

        ask_claude("test")

        call = mock_ask_claude_client.messages.call_history[0]
        assert "timeout" in call
        assert call["timeout"] > 0

    def test_client_initialized_with_api_key(
        self, mock_ask_claude_client, patched_log_file
    ):
        """The Anthropic client should be constructed with the configured key."""
        from agent.ask_claude import ask_claude

        ask_claude("test")

        assert mock_ask_claude_client.init_kwargs.get("api_key") == "test-api-key"

    def test_logs_query_on_success(self, mock_ask_claude_client, patched_log_file):
        """log_query should be called with success=True and write to the log."""
        from agent.ask_claude import ask_claude

        mock_ask_claude_client.messages.response_text = "ok"
        ask_claude("ping")

        assert patched_log_file.exists()
        content = patched_log_file.read_text(encoding="utf-8")
        assert "SUCCESS" in content
        assert "ping" in content


# =============================================================================
# TESTS: error paths
# =============================================================================


class TestAskClaudeErrors:
    """Error-path tests for ask_claude."""

    def test_missing_sdk_returns_install_hint(
        self, reset_ask_claude_client, patched_log_file, monkeypatch
    ):
        """When the SDK isn't installed, return a helpful message (no crash)."""
        monkeypatch.setattr(reset_ask_claude_client, "HAS_ANTHROPIC", False)

        from agent.ask_claude import ask_claude

        result = ask_claude("question")

        assert "Anthropic SDK" in result
        assert "pip install" in result.lower()

    def test_missing_api_key_returns_hint(
        self, reset_ask_claude_client, patched_log_file, monkeypatch
    ):
        """When the API key isn't set, return a helpful message (no crash)."""
        monkeypatch.setattr(reset_ask_claude_client, "HAS_ANTHROPIC", True)
        fake_settings = MagicMock()
        fake_settings.anthropic_api_key = ""
        monkeypatch.setattr(reset_ask_claude_client, "settings", fake_settings)

        from agent.ask_claude import ask_claude

        result = ask_claude("question")

        assert "ANTHROPIC_API_KEY" in result

    def test_timeout_returns_timeout_message(
        self, mock_ask_claude_client, patched_log_file
    ):
        """A timeout exception should surface as a timeout error message."""
        import anthropic

        mock_ask_claude_client.messages.raise_on_create = anthropic.APITimeoutError(
            request=MagicMock()
        )

        from agent.ask_claude import ask_claude

        result = ask_claude("slow question")

        assert "timeout" in result.lower()

    def test_generic_exception_is_caught(
        self, mock_ask_claude_client, patched_log_file
    ):
        """Unexpected exceptions return an error string rather than propagating."""
        mock_ask_claude_client.messages.raise_on_create = RuntimeError("boom")

        from agent.ask_claude import ask_claude

        result = ask_claude("test")

        assert "Error" in result
        assert "boom" in result

    def test_logs_failure(self, mock_ask_claude_client, patched_log_file):
        """Failed calls should still be logged, with FAILED status."""
        mock_ask_claude_client.messages.raise_on_create = RuntimeError("boom")

        from agent.ask_claude import ask_claude

        ask_claude("test")

        content = patched_log_file.read_text(encoding="utf-8")
        assert "FAILED" in content


# =============================================================================
# TESTS: no subprocess
# =============================================================================


class TestNoSubprocess:
    """Verify the refactor removed all subprocess usage from ask_claude."""

    def test_subprocess_run_not_called(self, mock_ask_claude_client, patched_log_file):
        """subprocess.run must not be invoked anywhere in the code path."""
        with patch("subprocess.run") as mock_run:
            from agent.ask_claude import ask_claude

            ask_claude("hello")

            mock_run.assert_not_called()

    def test_module_does_not_import_subprocess(self):
        """ask_claude.py should not import subprocess at all."""
        import agent.ask_claude as ac_module

        source = (
            __import__("pathlib").Path(ac_module.__file__).read_text(encoding="utf-8")
        )
        assert "import subprocess" not in source
        assert "subprocess.run" not in source


# =============================================================================
# TESTS: client singleton
# =============================================================================


class TestClientSingleton:
    """The Anthropic client should be built once and reused."""

    def test_client_reused_across_calls(
        self, mock_ask_claude_client, patched_log_file
    ):
        """Multiple ask_claude() calls should hit the same client instance."""
        from agent.ask_claude import ask_claude

        ask_claude("q1")
        ask_claude("q2")

        # Two calls on the same messages object
        assert len(mock_ask_claude_client.messages.call_history) == 2


# =============================================================================
# TESTS: tool registration
# =============================================================================


class TestGetClaudeTools:
    """get_claude_tools should still expose the ask_claude tool."""

    def test_returns_ask_claude_tool(self):
        from agent.ask_claude import get_claude_tools

        tools = get_claude_tools()
        names = {t.name for t in tools}

        assert "ask_claude" in names

    def test_tool_function_is_ask_claude(self):
        from agent.ask_claude import ask_claude, get_claude_tools

        tools = get_claude_tools()
        tool = next(t for t in tools if t.name == "ask_claude")

        assert tool.function is ask_claude
