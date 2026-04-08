"""
Tests for agent/core.py - Agent class, Tool dataclass, and utilities.
"""

import time

from agent.core import (
    Agent,
    AgentConfig,
    Tool,
    ToolResultStorage,
    create_tool,
    strip_thinking_tags,
)


class TestStripThinkingTags:
    """Tests for strip_thinking_tags function."""

    def test_removes_simple_thinking(self):
        """Removes <think>...</think> tags."""
        text = "Hello <think>internal reasoning</think> world"
        result = strip_thinking_tags(text)
        assert result == "Hello  world"

    def test_removes_multiline_thinking(self):
        """Handles multiline thinking blocks."""
        text = """Start
<think>
Line 1
Line 2
Line 3
</think>
End"""
        result = strip_thinking_tags(text)
        assert "Line 1" not in result
        assert "Start" in result
        assert "End" in result

    def test_preserves_content_outside(self):
        """Content outside tags preserved."""
        text = "Before <think>hidden</think> After"
        result = strip_thinking_tags(text)
        assert "Before" in result
        assert "After" in result
        assert "hidden" not in result

    def test_no_tags_unchanged(self):
        """Text without tags unchanged."""
        text = "No thinking tags here"
        result = strip_thinking_tags(text)
        assert result == text

    def test_multiple_thinking_blocks(self):
        """Handles multiple thinking blocks."""
        text = "A <think>1</think> B <think>2</think> C"
        result = strip_thinking_tags(text)
        assert "1" not in result
        assert "2" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_strips_whitespace(self):
        """Result is stripped of leading/trailing whitespace."""
        text = "  <think>stuff</think>  Result  "
        result = strip_thinking_tags(text)
        assert result == "Result"


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self, sample_tool_function, sample_tool_schema):
        """Tool can be created with all required fields."""
        tool = Tool(
            name="greet",
            description="Greet someone",
            parameters=sample_tool_schema,
            function=sample_tool_function,
        )

        assert tool.name == "greet"
        assert tool.description == "Greet someone"
        assert tool.parameters == sample_tool_schema
        assert tool.function("World") == "Hello, World!"

    def test_create_tool_helper(self, sample_tool_function, sample_tool_schema):
        """create_tool() helper creates valid Tool."""
        tool = create_tool("greet", "Greet someone", sample_tool_schema, sample_tool_function)

        assert isinstance(tool, Tool)
        assert tool.name == "greet"
        assert tool.function("Test") == "Hello, Test!"


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = AgentConfig()

        assert config.model == "llama3.1"
        assert config.temperature == 0.7
        assert config.max_turns == 20
        assert config.system_prompt == ""
        assert config.verbose is True

    def test_custom_config(self):
        """Custom config values are applied."""
        config = AgentConfig(
            model="qwen3.5:27b",
            temperature=0.5,
            max_turns=10,
            system_prompt="Custom prompt",
            verbose=False,
        )

        assert config.model == "qwen3.5:27b"
        assert config.temperature == 0.5
        assert config.max_turns == 10
        assert config.system_prompt == "Custom prompt"
        assert config.verbose is False


class TestAgent:
    """Tests for Agent class."""

    def test_agent_init_default_config(self, mock_ollama_client):
        """Agent initializes with default AgentConfig."""
        agent = Agent()

        assert agent.config is not None
        assert agent.config.model == "llama3.1"
        # Built-in get_stored_result tool is always registered
        assert "get_stored_result" in agent.tools
        assert agent.messages == []
        assert agent.turn_count == 0

    def test_agent_init_custom_config(self, mock_ollama_client):
        """Agent accepts custom AgentConfig."""
        config = AgentConfig(model="custom-model", verbose=False)
        agent = Agent(config)

        assert agent.config.model == "custom-model"
        assert agent.config.verbose is False

    def test_register_tool(self, mock_ollama_client, sample_tool_function, sample_tool_schema):
        """register_tool adds tool to agent.tools dict."""
        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool("greet", "Greet someone", sample_tool_schema, sample_tool_function)

        agent.register_tool(tool)

        assert "greet" in agent.tools
        assert agent.tools["greet"] == tool

    def test_register_tool_duplicate(
        self, mock_ollama_client, sample_tool_function, sample_tool_schema
    ):
        """Registering same tool name overwrites."""
        agent = Agent(AgentConfig(verbose=False))

        tool1 = create_tool("greet", "First", sample_tool_schema, sample_tool_function)
        tool2 = create_tool("greet", "Second", sample_tool_schema, lambda name: f"Hi, {name}!")

        agent.register_tool(tool1)
        agent.register_tool(tool2)

        # 1 built-in (get_stored_result) + 1 user tool (greet, overwritten)
        assert len(agent.tools) == 2
        assert agent.tools["greet"].description == "Second"

    def test_get_ollama_tools_format(
        self, mock_ollama_client, sample_tool_function, sample_tool_schema
    ):
        """_get_ollama_tools returns correct Ollama API format."""
        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool("greet", "Greet someone", sample_tool_schema, sample_tool_function)
        agent.register_tool(tool)

        ollama_tools = agent._get_ollama_tools()

        # 1 built-in + 1 user tool
        assert len(ollama_tools) == 2
        greet_tool = [t for t in ollama_tools if t["function"]["name"] == "greet"][0]
        assert greet_tool["type"] == "function"
        assert greet_tool["function"]["description"] == "Greet someone"
        assert greet_tool["function"]["parameters"] == sample_tool_schema

    def test_execute_tool_success(
        self, mock_ollama_client, sample_tool_function, sample_tool_schema
    ):
        """_execute_tool calls function with arguments."""
        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool("greet", "Greet someone", sample_tool_schema, sample_tool_function)
        agent.register_tool(tool)

        result = agent._execute_tool("greet", {"name": "World"})

        assert result == "Hello, World!"

    def test_execute_tool_unknown(self, mock_ollama_client):
        """_execute_tool returns error for unknown tool."""
        agent = Agent(AgentConfig(verbose=False))

        result = agent._execute_tool("nonexistent", {})

        assert "Error" in result
        assert "Unknown tool" in result

    def test_execute_tool_exception(self, mock_ollama_client):
        """_execute_tool handles function exceptions gracefully."""

        def failing_func(**kwargs):
            raise ValueError("Intentional error")

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "fail",
            "A failing tool",
            {"type": "object", "properties": {}, "required": []},
            failing_func,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("fail", {})

        assert "Error" in result
        assert "Intentional error" in result

    def test_clear_history(self, mock_ollama_client):
        """clear_history resets messages and turn count."""
        agent = Agent(AgentConfig(verbose=False))
        agent.messages = [{"role": "user", "content": "test"}]
        agent.turn_count = 5

        agent.clear_history()

        assert agent.messages == []
        assert agent.turn_count == 0

    def test_get_history(self, mock_ollama_client):
        """get_history returns a copy of messages."""
        agent = Agent(AgentConfig(verbose=False))
        agent.messages = [{"role": "user", "content": "test"}]

        history = agent.get_history()

        assert history == [{"role": "user", "content": "test"}]
        # Verify it's a copy
        history.append({"role": "assistant", "content": "response"})
        assert len(agent.messages) == 1


class TestAgentRun:
    """Tests for Agent.run() method."""

    def test_run_simple_response(self, mock_ollama_client):
        """Agent returns response when no tool calls."""
        mock_ollama_client.set_responses(
            [{"message": {"content": "This is my response", "tool_calls": []}}]
        )

        agent = Agent(AgentConfig(verbose=False))
        result = agent.run("Hello")

        assert result == "This is my response"

    def test_run_with_tool_call(self, mock_ollama_client, sample_tool_function, sample_tool_schema):
        """Agent executes tool when Ollama returns tool_calls."""
        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "greet", "arguments": {"name": "World"}}}
                        ],
                    }
                },
                {"message": {"content": "I greeted the world!", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool("greet", "Greet someone", sample_tool_schema, sample_tool_function)
        agent.register_tool(tool)

        result = agent.run("Greet someone")

        assert result == "I greeted the world!"
        # Verify tool was called
        assert len(mock_ollama_client.call_history) == 2

    def test_run_strips_thinking_tags(self, mock_ollama_client):
        """Agent strips thinking tags from response."""
        mock_ollama_client.set_responses(
            [{"message": {"content": "<think>reasoning</think>Final answer", "tool_calls": []}}]
        )

        agent = Agent(AgentConfig(verbose=False))
        result = agent.run("Question")

        assert result == "Final answer"
        assert "reasoning" not in result

    def test_run_max_turns(self, mock_ollama_client, sample_tool_function, sample_tool_schema):
        """Agent stops at max_turns."""
        # Set up infinite tool calls
        tool_call_response = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "greet", "arguments": {"name": "Test"}}}],
            }
        }
        mock_ollama_client.responses = [tool_call_response] * 100

        agent = Agent(AgentConfig(verbose=False, max_turns=3))
        tool = create_tool("greet", "Greet", sample_tool_schema, sample_tool_function)
        agent.register_tool(tool)

        agent.run("Keep greeting")

        # Should have stopped at 3 turns
        assert agent.turn_count == 3


# =============================================================================
# TOOL RESULT STORAGE TESTS
# =============================================================================


class TestToolResultStorage:
    """Tests for ToolResultStorage truncation and retrieval."""

    def test_small_result_not_truncated(self):
        """Results under threshold are returned unchanged."""
        storage = ToolResultStorage()
        result = "Hello, this is a short result."
        output = storage.maybe_truncate("web_fetch", result)
        assert output == result

    def test_large_result_truncated(self):
        """Results over threshold are truncated with preview."""
        storage = ToolResultStorage()
        result = "x" * 10000  # Way over any threshold
        output = storage.maybe_truncate("web_fetch", result)
        assert "TRUNCATED" in output
        assert "10,000 chars total" in output
        assert "get_stored_result" in output
        assert len(output) < len(result)

    def test_truncated_result_retrievable(self):
        """Full result can be retrieved after truncation."""
        storage = ToolResultStorage()
        original = "Important data " * 1000  # ~15KB
        truncated = storage.maybe_truncate("web_fetch", original)

        # Extract the result ID from the truncation notice
        import re

        match = re.search(r"ID: ([a-f0-9]+)", truncated)
        assert match is not None
        result_id = match.group(1)

        # Retrieve full result
        full = storage.get_full_result(result_id)
        assert full == original

    def test_unknown_id_returns_error(self):
        """Requesting a non-existent ID returns error message."""
        storage = ToolResultStorage()
        result = storage.get_full_result("nonexistent123")
        assert "Error" in result

    def test_stats_tracking(self):
        """Stats correctly track truncation counts."""
        storage = ToolResultStorage()

        # One small result
        storage.maybe_truncate("web_fetch", "small")
        assert storage.stats["total_results"] == 1
        assert storage.stats["truncated_results"] == 0

        # One large result
        storage.maybe_truncate("web_fetch", "x" * 10000)
        assert storage.stats["total_results"] == 2
        assert storage.stats["truncated_results"] == 1
        assert storage.stats["bytes_saved"] > 0

    def test_per_tool_thresholds(self):
        """Different tools have different thresholds."""
        storage = ToolResultStorage()

        # 2500 chars: under read_file threshold (4000) but over web_fetch (2000)
        medium_result = "y" * 2500

        web_output = storage.maybe_truncate("web_fetch", medium_result)
        assert "TRUNCATED" in web_output

        file_output = storage.maybe_truncate("read_file", medium_result)
        assert file_output == medium_result  # Not truncated

    def test_preview_has_head_and_tail(self):
        """Truncated preview includes beginning and end of content."""
        storage = ToolResultStorage()
        # Build a result with distinct head and tail
        result = "HEAD_MARKER " + ("x" * 10000) + " TAIL_MARKER"
        output = storage.maybe_truncate("web_fetch", result)
        assert "HEAD_MARKER" in output
        assert "TAIL_MARKER" in output

    def test_cleanup_clears_store(self):
        """cleanup_session clears all stored results."""
        storage = ToolResultStorage()
        storage.maybe_truncate("web_fetch", "x" * 10000)
        assert len(storage._mem_store) > 0
        storage.cleanup_session()
        assert len(storage._mem_store) == 0

    def test_get_stats_format(self):
        """get_stats returns expected keys."""
        storage = ToolResultStorage()
        storage.maybe_truncate("web_fetch", "small")
        stats = storage.get_stats()
        assert "backend" in stats
        assert "truncation_rate" in stats
        assert "stored_results" in stats
        assert stats["backend"] in ("redis", "memory")


class TestAgentWithTruncation:
    """Tests that Agent properly integrates tool result truncation."""

    def test_agent_has_get_stored_result_tool(self, mock_ollama_client):
        """Agent automatically registers the get_stored_result tool."""
        agent = Agent(AgentConfig(verbose=False))
        assert "get_stored_result" in agent.tools

    def test_agent_truncates_large_tool_result(self, mock_ollama_client):
        """Agent truncates large results during tool execution."""

        def big_result(**kwargs):
            return "data " * 5000  # ~25KB

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "big_tool",
            "Returns big data",
            {"type": "object", "properties": {}, "required": []},
            big_result,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("big_tool", {})
        assert "TRUNCATED" in result
        assert agent.result_storage.stats["truncated_results"] == 1

    def test_agent_does_not_truncate_small_result(self, mock_ollama_client):
        """Agent passes through small results unchanged."""

        def small_result(**kwargs):
            return "ok"

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "small_tool",
            "Returns small data",
            {"type": "object", "properties": {}, "required": []},
            small_result,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("small_tool", {})
        assert result == "ok"


# =============================================================================
# TOOL TIMEOUT TESTS
# =============================================================================


class TestToolTimeout:
    """Tests for per-tool timeout mechanism."""

    def test_tool_timeout_field_default(self):
        """Tool.timeout defaults to None (no limit)."""
        tool = Tool(
            name="t", description="d",
            parameters={}, function=lambda: "ok",
        )
        assert tool.timeout is None

    def test_create_tool_with_timeout(self):
        """create_tool accepts optional timeout parameter."""
        tool = create_tool("t", "d", {}, lambda: "ok", timeout=30)
        assert tool.timeout == 30

    def test_fast_tool_with_timeout_returns_normally(self, mock_ollama_client):
        """A tool that finishes within its timeout returns the normal result."""
        def fast_func(**kwargs):
            return "fast result"

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "fast", "Fast tool",
            {"type": "object", "properties": {}, "required": []},
            fast_func, timeout=5,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("fast", {})
        assert result == "fast result"

    def test_slow_tool_returns_timeout_fallback(self, mock_ollama_client):
        """A tool that exceeds its timeout returns a fallback message."""
        def slow_func(**kwargs):
            time.sleep(10)
            return "should not see this"

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "slow", "Slow tool",
            {"type": "object", "properties": {}, "required": []},
            slow_func, timeout=1,
        )
        agent.register_tool(tool)

        start = time.monotonic()
        result = agent._execute_tool("slow", {})
        elapsed = time.monotonic() - start

        assert "timed out" in result
        assert "1s" in result
        # Should return in ~1s, not 10s
        assert elapsed < 3

    def test_tool_without_timeout_runs_normally(self, mock_ollama_client):
        """A tool with no timeout set (None) runs without the thread wrapper."""
        def normal_func(**kwargs):
            return "normal result"

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "normal", "Normal tool",
            {"type": "object", "properties": {}, "required": []},
            normal_func,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("normal", {})
        assert result == "normal result"

    def test_tool_timeout_exception_propagates(self, mock_ollama_client):
        """A tool that raises within the timeout wrapper still reports the error."""
        def error_func(**kwargs):
            raise ValueError("broken")

        agent = Agent(AgentConfig(verbose=False))
        tool = create_tool(
            "err", "Error tool",
            {"type": "object", "properties": {}, "required": []},
            error_func, timeout=5,
        )
        agent.register_tool(tool)

        result = agent._execute_tool("err", {})
        assert "Error" in result
        assert "broken" in result
