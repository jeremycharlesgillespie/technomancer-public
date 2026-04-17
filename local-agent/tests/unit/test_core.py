"""
Tests for agent/core.py - Agent class, Tool dataclass, and utilities.
"""

import re
import threading
import time
from unittest.mock import patch

import pytest

from agent.core import (
    Agent,
    AgentConfig,
    CircuitBreaker,
    OllamaCircuitOpenError,
    Tool,
    ToolResultStorage,
    create_tool,
    extract_thinking,
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


# =============================================================================
# EXTRACT_THINKING TESTS
# =============================================================================


class TestExtractThinking:
    """Tests for extract_thinking utility — mirror of strip_thinking_tags."""

    def test_extracts_single_block(self):
        assert extract_thinking("pre <think>reasoning here</think> post") == "reasoning here"

    def test_multiline_block(self):
        text = "pre <think>\nline one\nline two\n</think> post"
        assert "line one" in extract_thinking(text)
        assert "line two" in extract_thinking(text)

    def test_no_block_returns_empty(self):
        assert extract_thinking("no tags here") == ""

    def test_strips_surrounding_whitespace(self):
        assert extract_thinking("<think>   hello   </think>") == "hello"


# =============================================================================
# CTX SIZE ESTIMATION
# =============================================================================


class TestEstimateCtxSize:
    """Agent._estimate_ctx_size should snap to power-of-two sizes."""

    def test_empty_messages_returns_minimum(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        agent.messages = []
        assert agent._estimate_ctx_size() == 8192

    def test_small_messages_stay_small(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        agent.messages = [{"role": "user", "content": "hi"}]
        assert agent._estimate_ctx_size() == 8192

    def test_medium_messages_step_up(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        # ~5000 tokens of content
        agent.messages = [{"role": "user", "content": "x" * 20_000}]
        ctx = agent._estimate_ctx_size()
        assert ctx in (16384, 32768)

    def test_huge_messages_max_out(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        # ~50K tokens worth of content
        agent.messages = [{"role": "user", "content": "x" * 200_000}]
        assert agent._estimate_ctx_size() == 131072


# =============================================================================
# TEMPERATURE OVERRIDE
# =============================================================================


class TestTemperatureOverride:
    def test_default_returns_config_temperature(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False, temperature=0.42))
        assert agent._get_temperature() == 0.42

    def test_override_applied_once(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False, temperature=0.7))
        agent.set_temperature(0.1)
        # First call consumes the override.
        assert agent._get_temperature() == 0.1
        # Second call falls back to config default.
        assert agent._get_temperature() == 0.7


# =============================================================================
# PROMPT COMPRESSION INTEGRATION
# =============================================================================


class TestAgentCompression:
    def test_compress_noop_when_under_threshold(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False, compression_threshold_chars=10_000))
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "x" * 100},
        ]
        before = list(agent.messages)
        agent._maybe_compress_messages()
        assert agent.messages == before

    def test_compress_collapses_when_over_threshold(self, mock_ollama_client):
        # Small threshold forces compression on next call.
        agent = Agent(AgentConfig(
            verbose=False,
            compression_threshold_chars=500,
            compression_keep_recent=1,
        ))
        big = "x" * 2000
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": big},
            {"role": "tool", "content": big},
            {"role": "user", "content": "latest"},
        ]
        agent._maybe_compress_messages()
        # The oldest tool result should have been collapsed.
        tools = [m for m in agent.messages if m.get("role") == "tool"]
        assert any("compressed" in m["content"] for m in tools)
        # Latest user message preserved.
        assert agent.messages[-1] == {"role": "user", "content": "latest"}


# =============================================================================
# OLLAMA CLIENT TIMEOUT (TK-390)
# =============================================================================


class TestOllamaClientTimeout:
    """Regression tests: the Ollama client must be built with a finite timeout.

    Without one, a stalled Ollama server can freeze the bot indefinitely
    (observed p95 latency of 1464s before this fix).
    """

    def test_build_client_applies_configured_timeout(self):
        """_build_ollama_client reads the timeout from settings."""
        from agent import core

        with patch("agent.core._settings.ollama_host", "http://127.0.0.1:11434"), \
             patch("agent.core._settings.ollama_request_timeout", 42.0):
            client = core._build_ollama_client()

        # Ollama wraps httpx — the timeout is stored on the underlying client.
        assert client._client.timeout.connect == 42.0
        assert client._client.timeout.read == 42.0

    def test_build_client_uses_configured_host(self):
        from agent import core

        with patch("agent.core._settings.ollama_host", "http://example.com:9999"), \
             patch("agent.core._settings.ollama_request_timeout", 60.0):
            client = core._build_ollama_client()

        # Host should be applied to the underlying httpx base_url.
        assert "example.com" in str(client._client.base_url)

    def test_module_client_has_finite_timeout(self):
        """The singleton client used at runtime must not have None timeout."""
        from agent import core

        timeout = core._ollama_client._client.timeout
        assert timeout.connect is not None
        assert timeout.read is not None


# =============================================================================
# RESPONSE CACHE INTEGRATION
# =============================================================================


class TestAgentResponseCache:
    def test_cache_disabled_by_default(self, mock_ollama_client):
        """By default the agent doesn't consult the cache — keeps behavior unchanged."""
        agent = Agent(AgentConfig(verbose=False))
        assert agent.config.enable_response_cache is False

    def test_cache_hit_skips_ollama(self, mock_ollama_client):
        """When the cache returns a hit, no Ollama call is issued."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))

        with patch("agent.llm_optimizer.cache_lookup", return_value="cached answer"):
            result = agent.run("What is 2+2?")

        assert result == "cached answer"
        # Mock Ollama was never asked.
        assert mock_ollama_client.call_history == []

    def test_cache_miss_calls_ollama_and_stores(self, mock_ollama_client):
        """On cache miss the agent runs Ollama and writes the result back."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        mock_ollama_client.set_responses(
            [{"message": {"content": "fresh answer", "tool_calls": []}}]
        )

        with patch("agent.llm_optimizer.cache_lookup", return_value=None) as mock_lookup, \
             patch("agent.llm_optimizer.cache_store") as mock_store:
            result = agent.run("What is 2+2?")

        assert result == "fresh answer"
        mock_lookup.assert_called_once()
        mock_store.assert_called_once()
        # Second positional arg is the response string.
        args, kwargs = mock_store.call_args
        assert "fresh answer" in (args[1] if len(args) > 1 else kwargs.get("response", ""))

    def test_multi_turn_not_cached(self, mock_ollama_client):
        """Responses that required tool calls shouldn't be cached — they're not deterministic."""
        def sample_func(**kwargs):
            return "tool result"

        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        tool = create_tool(
            "sample", "sample tool",
            {"type": "object", "properties": {}, "required": []},
            sample_func,
        )
        agent.register_tool(tool)

        mock_ollama_client.set_responses([
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "sample", "arguments": {}}}
            ]}},
            {"message": {"content": "done", "tool_calls": []}},
        ])

        with patch("agent.llm_optimizer.cache_lookup", return_value=None), \
             patch("agent.llm_optimizer.cache_store") as mock_store:
            result = agent.run("Run the tool")

        assert result == "done"
        mock_store.assert_not_called()

    def test_cache_skipped_for_images(self, mock_ollama_client):
        """Vision prompts never hit the cache — the hash would ignore the image."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        mock_ollama_client.set_responses(
            [{"message": {"content": "vision answer", "tool_calls": []}}]
        )

        with patch("agent.llm_optimizer.cache_lookup") as mock_lookup, \
             patch("agent.llm_optimizer.cache_store") as mock_store:
            result = agent.run("describe this", images=[b"\x89PNG"])

        assert result == "vision answer"
        mock_lookup.assert_not_called()
        mock_store.assert_not_called()

    def test_cache_skipped_for_long_tasks(self, mock_ollama_client):
        """Long prompts aren't cache candidates — the normalized hash is too coarse."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        mock_ollama_client.set_responses(
            [{"message": {"content": "long answer", "tool_calls": []}}]
        )

        long_task = "x" * 2000

        with patch("agent.llm_optimizer.cache_lookup") as mock_lookup, \
             patch("agent.llm_optimizer.cache_store") as mock_store:
            result = agent.run(long_task)

        assert result == "long answer"
        mock_lookup.assert_not_called()
        mock_store.assert_not_called()

    def test_cache_lookup_exception_falls_through(self, mock_ollama_client):
        """If the cache backend errors we still complete the request via Ollama."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        mock_ollama_client.set_responses(
            [{"message": {"content": "fallback answer", "tool_calls": []}}]
        )

        with patch("agent.llm_optimizer.cache_lookup", side_effect=RuntimeError("db locked")), \
             patch("agent.llm_optimizer.cache_store"):
            result = agent.run("hi")

        assert result == "fallback answer"

    def test_cache_store_exception_does_not_fail_request(self, mock_ollama_client):
        """A cache write failure must not change what the user sees."""
        agent = Agent(AgentConfig(verbose=False, enable_response_cache=True))
        mock_ollama_client.set_responses(
            [{"message": {"content": "still ok", "tool_calls": []}}]
        )

        with patch("agent.llm_optimizer.cache_lookup", return_value=None), \
             patch("agent.llm_optimizer.cache_store", side_effect=RuntimeError("disk full")):
            result = agent.run("hi")

        assert result == "still ok"


# =============================================================================
# PARALLEL TOOL EXECUTION (TK-328)
# =============================================================================


class TestParallelToolExecution:
    """Agent should execute independent tool calls concurrently to cut latency."""

    def _make_call(self, name: str, args: dict | None = None) -> dict:
        return {"function": {"name": name, "arguments": args or {}}}

    def test_parallel_enabled_by_default(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        assert agent.config.parallel_tool_execution is True

    def test_empty_tool_calls_returns_empty(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        assert agent._execute_tool_calls([]) == []

    def test_single_tool_call_returns_one_message(
        self, mock_ollama_client, sample_tool_function, sample_tool_schema
    ):
        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(
            create_tool("greet", "Greet", sample_tool_schema, sample_tool_function)
        )
        msgs = agent._execute_tool_calls([self._make_call("greet", {"name": "A"})])
        assert msgs == [{"role": "tool", "content": "Hello, A!"}]

    def test_results_preserve_call_order(self, mock_ollama_client):
        """Parallel results come back in the same order as the input calls."""
        order_record = []

        def slow_a(**kwargs):
            time.sleep(0.15)
            order_record.append("A")
            return "result_A"

        def slow_b(**kwargs):
            time.sleep(0.05)
            order_record.append("B")
            return "result_B"

        schema = {"type": "object", "properties": {}, "required": []}
        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("tool_a", "A", schema, slow_a))
        agent.register_tool(create_tool("tool_b", "B", schema, slow_b))

        msgs = agent._execute_tool_calls(
            [self._make_call("tool_a"), self._make_call("tool_b")]
        )

        # Output order matches input order, even though B finished first.
        assert [m["content"] for m in msgs] == ["result_A", "result_B"]
        # Real concurrency: B completed before A.
        assert order_record == ["B", "A"]

    def test_parallel_cuts_latency(self, mock_ollama_client):
        """Running two 200ms tools concurrently finishes well under 400ms."""
        schema = {"type": "object", "properties": {}, "required": []}

        def slow(**kwargs):
            time.sleep(0.2)
            return "done"

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("s1", "", schema, slow))
        agent.register_tool(create_tool("s2", "", schema, slow))
        agent.register_tool(create_tool("s3", "", schema, slow))

        calls = [self._make_call("s1"), self._make_call("s2"), self._make_call("s3")]

        start = time.monotonic()
        msgs = agent._execute_tool_calls(calls)
        elapsed = time.monotonic() - start

        assert len(msgs) == 3
        # Sequential would take ~0.6s. Parallel should be closer to 0.2s.
        assert elapsed < 0.45, f"Expected parallel speed-up, took {elapsed:.2f}s"

    def test_parallel_disabled_runs_sequentially(self, mock_ollama_client):
        """With parallel_tool_execution=False, tools run in sequence."""
        schema = {"type": "object", "properties": {}, "required": []}
        order_record = []

        def slow_a(**kwargs):
            time.sleep(0.1)
            order_record.append("A")
            return "A"

        def fast_b(**kwargs):
            order_record.append("B")
            return "B"

        agent = Agent(AgentConfig(verbose=False, parallel_tool_execution=False))
        agent.register_tool(create_tool("tool_a", "", schema, slow_a))
        agent.register_tool(create_tool("tool_b", "", schema, fast_b))

        agent._execute_tool_calls(
            [self._make_call("tool_a"), self._make_call("tool_b")]
        )

        # Sequential: A blocks, so A finishes before B starts.
        assert order_record == ["A", "B"]

    def test_dependency_detection_no_references(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        calls = [
            self._make_call("web_search", {"query": "python asyncio"}),
            self._make_call("read_file", {"path": "/etc/hosts"}),
        ]
        assert agent._tool_calls_have_dependencies(calls) is False

    def test_dependency_detection_placeholder_reference(self, mock_ollama_client):
        """When a later call's args reference an earlier tool by placeholder."""
        agent = Agent(AgentConfig(verbose=False))
        calls = [
            self._make_call("web_search", {"query": "python"}),
            self._make_call("summarize", {"text": "{{web_search_result}}"}),
        ]
        assert agent._tool_calls_have_dependencies(calls) is True

    def test_dependency_detection_output_of_pattern(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        calls = [
            self._make_call("fetch_data", {}),
            self._make_call("post", {"body": "output_of_fetch_data"}),
        ]
        assert agent._tool_calls_have_dependencies(calls) is True

    def test_single_call_never_has_dependencies(self, mock_ollama_client):
        agent = Agent(AgentConfig(verbose=False))
        assert agent._tool_calls_have_dependencies([self._make_call("x")]) is False

    def test_dependent_calls_run_sequentially(self, mock_ollama_client):
        """Dependent calls fall back to serial execution even with parallel enabled."""
        schema = {"type": "object", "properties": {}, "required": []}
        order_record = []

        def slow_a(**kwargs):
            time.sleep(0.1)
            order_record.append("A")
            return "A_result"

        def slow_b(**kwargs):
            order_record.append("B")
            return kwargs.get("body", "")

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("fetch_a", "", schema, slow_a))
        agent.register_tool(
            create_tool(
                "use_b",
                "",
                {
                    "type": "object",
                    "properties": {"body": {"type": "string"}},
                    "required": [],
                },
                slow_b,
            )
        )

        calls = [
            self._make_call("fetch_a"),
            self._make_call("use_b", {"body": "{{fetch_a"}),
        ]
        agent._execute_tool_calls(calls)
        # Detected dependency ⇒ sequential execution.
        assert order_record == ["A", "B"]

    def test_parallel_handles_tool_errors(self, mock_ollama_client):
        """If one tool raises, others still complete and the error is surfaced as a string."""
        schema = {"type": "object", "properties": {}, "required": []}

        def boom(**kwargs):
            raise ValueError("kaboom")

        def ok(**kwargs):
            return "ok_result"

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("boom", "", schema, boom))
        agent.register_tool(create_tool("ok", "", schema, ok))

        msgs = agent._execute_tool_calls(
            [self._make_call("boom"), self._make_call("ok")]
        )

        assert len(msgs) == 2
        assert "kaboom" in msgs[0]["content"]
        assert msgs[1]["content"] == "ok_result"

    def test_run_parallel_integration(self, mock_ollama_client):
        """Full run() loop dispatches multiple tool calls in parallel."""
        schema = {"type": "object", "properties": {}, "required": []}
        order_record = []

        def slow(**kwargs):
            time.sleep(0.15)
            order_record.append(kwargs.get("tag", "?"))
            return "done"

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(
            create_tool(
                "slow",
                "",
                {
                    "type": "object",
                    "properties": {"tag": {"type": "string"}},
                    "required": [],
                },
                slow,
            )
        )

        mock_ollama_client.set_responses([
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "slow", "arguments": {"tag": "a"}}},
                        {"function": {"name": "slow", "arguments": {"tag": "b"}}},
                    ],
                }
            },
            {"message": {"content": "all done", "tool_calls": []}},
        ])

        start = time.monotonic()
        result = agent.run("do both")
        elapsed = time.monotonic() - start

        assert result == "all done"
        # Two 150ms calls in parallel should finish well under 300ms.
        assert elapsed < 0.4, f"Parallel run took {elapsed:.2f}s"

    def test_max_parallel_tools_caps_workers(self, mock_ollama_client):
        """max_parallel_tools limits the thread pool size."""
        schema = {"type": "object", "properties": {}, "required": []}
        active = {"count": 0, "peak": 0}
        lock = threading.Lock()

        def tracked(**kwargs):
            with lock:
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
            time.sleep(0.05)
            with lock:
                active["count"] -= 1
            return "ok"

        agent = Agent(AgentConfig(verbose=False, max_parallel_tools=2))
        agent.register_tool(create_tool("t", "", schema, tracked))

        calls = [self._make_call("t") for _ in range(6)]
        agent._execute_tool_calls(calls)

        assert active["peak"] <= 2


class TestParallelExecutionThreadSafety:
    """ToolResultStorage stats must stay consistent under concurrent writes."""

    def test_concurrent_small_results_stats_consistent(self, mock_ollama_client):
        """Total results counter matches the number of concurrent calls."""
        schema = {"type": "object", "properties": {}, "required": []}

        def quick(**kwargs):
            return "ok"

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("q", "", schema, quick))

        calls = [{"function": {"name": "q", "arguments": {}}} for _ in range(20)]
        agent._execute_tool_calls(calls)

        assert agent.result_storage.stats["total_results"] == 20
        assert agent.result_storage.stats["truncated_results"] == 0

    def test_concurrent_large_results_are_all_stored(self, mock_ollama_client):
        """Every truncated result gets its own retrievable ID — no collisions."""
        schema = {"type": "object", "properties": {}, "required": []}
        big = "y" * 20_000

        def heavy(**kwargs):
            return big

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("heavy", "", schema, heavy))

        calls = [{"function": {"name": "heavy", "arguments": {}}} for _ in range(8)]
        msgs = agent._execute_tool_calls(calls)

        ids = []
        for m in msgs:
            match = re.search(r"ID: ([a-f0-9]+)", m["content"])
            assert match, f"no truncation id in {m['content'][:120]}"
            ids.append(match.group(1))

        # All IDs unique.
        assert len(set(ids)) == 8
        # Every ID retrieves the original content.
        for rid in ids:
            assert agent.result_storage.get_full_result(rid) == big
        assert agent.result_storage.stats["truncated_results"] == 8


class TestOllamaCircuitOpenError:
    """Tests for OllamaCircuitOpenError exception."""

    def test_is_exception(self):
        """OllamaCircuitOpenError is a plain Exception subclass."""
        assert issubclass(OllamaCircuitOpenError, Exception)

    def test_can_be_raised_and_caught(self):
        """Can be raised and caught with a message."""
        with pytest.raises(OllamaCircuitOpenError, match="nope"):
            raise OllamaCircuitOpenError("nope")


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_defaults(self):
        """Default fields match the spec."""
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.opened_at == 0.0
        assert cb.threshold == 5
        assert cb.window_seconds == 60.0
        assert cb.reset_seconds == 30.0

    def test_closed_circuit_is_not_open(self):
        """A fresh breaker does not block calls."""
        cb = CircuitBreaker()
        assert cb.is_open() is False

    def test_record_failure_below_threshold_stays_closed(self):
        """Failures below threshold keep the circuit closed."""
        cb = CircuitBreaker(threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 4
        assert cb.is_open() is False

    def test_record_failure_at_threshold_opens(self):
        """Hitting threshold opens the circuit."""
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.opened_at > 0.0
        assert cb.is_open() is True

    def test_record_success_resets(self):
        """Success clears the failure count and closes the circuit."""
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.opened_at == 0.0

    def test_failures_outside_window_start_fresh_count(self):
        """Failures older than window_seconds reset the rolling counter."""
        cb = CircuitBreaker(threshold=5, window_seconds=60.0)
        t = [1000.0]

        def fake_monotonic():
            return t[0]

        with patch("agent.core._time.monotonic", fake_monotonic):
            cb.record_failure()
            cb.record_failure()
            assert cb.failure_count == 2
            # Jump past the window — next failure resets the count.
            t[0] = 1000.0 + 61.0
            cb.record_failure()
            assert cb.failure_count == 1
            assert cb.state == "closed"

    def test_failures_inside_window_accumulate_and_open(self):
        """Failures within the window accumulate to threshold and open."""
        cb = CircuitBreaker(threshold=3, window_seconds=60.0)
        t = [500.0]

        def fake_monotonic():
            return t[0]

        with patch("agent.core._time.monotonic", fake_monotonic):
            cb.record_failure()
            t[0] += 10
            cb.record_failure()
            t[0] += 10
            cb.record_failure()
            assert cb.state == "open"
            assert cb.opened_at == t[0]

    def test_open_transitions_to_half_open_after_reset(self):
        """After reset_seconds an open circuit transitions to half-open."""
        cb = CircuitBreaker(threshold=2, reset_seconds=30.0)
        t = [100.0]

        def fake_monotonic():
            return t[0]

        with patch("agent.core._time.monotonic", fake_monotonic):
            cb.record_failure()
            cb.record_failure()
            assert cb.state == "open"
            # Still inside reset window — stays open.
            t[0] += 29
            assert cb.is_open() is True
            # Past reset window — flips to half-open, is_open returns False.
            t[0] += 2
            assert cb.is_open() is False
            assert cb.state == "half-open"

    def test_half_open_success_closes(self):
        """A success while half-open fully closes the circuit."""
        cb = CircuitBreaker(threshold=2, reset_seconds=30.0)
        t = [0.0]

        def fake_monotonic():
            return t[0]

        with patch("agent.core._time.monotonic", fake_monotonic):
            cb.record_failure()
            cb.record_failure()
            t[0] += 31
            cb.is_open()  # transitions to half-open
            assert cb.state == "half-open"
            cb.record_success()
            assert cb.state == "closed"
            assert cb.failure_count == 0

    def test_half_open_failure_reopens(self):
        """A failure while half-open re-opens the circuit immediately."""
        cb = CircuitBreaker(threshold=5, reset_seconds=30.0)
        t = [0.0]

        def fake_monotonic():
            return t[0]

        with patch("agent.core._time.monotonic", fake_monotonic):
            for _ in range(5):
                cb.record_failure()
            assert cb.state == "open"
            t[0] += 31
            cb.is_open()
            assert cb.state == "half-open"
            t[0] += 1
            cb.record_failure()
            assert cb.state == "open"
            assert cb.opened_at == t[0]
            # Circuit is immediately blocking again.
            assert cb.is_open() is True
