"""
Integration tests for full agent workflow with mocked Ollama.
"""

from unittest.mock import MagicMock

import requests

from agent.core import Agent, AgentConfig, create_tool


class TestAgentToolExecution:
    """Tests for agent executing tools end-to-end."""

    def test_agent_executes_single_tool(self, mock_ollama_client):
        """Agent executes a tool and returns final response."""

        # Create a simple tool
        def add_numbers(a: int, b: int) -> str:
            return f"The sum is {a + b}"

        tool = create_tool(
            "add_numbers",
            "Add two numbers",
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            add_numbers,
        )

        # Set up mock to call tool, then return response
        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "add_numbers", "arguments": {"a": 5, "b": 3}}}
                        ],
                    }
                },
                {"message": {"content": "I calculated that 5 + 3 = 8", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(tool)

        result = agent.run("What is 5 + 3?")

        assert "8" in result
        assert len(mock_ollama_client.call_history) == 2

    def test_agent_executes_multiple_tools_sequentially(self, mock_ollama_client):
        """Agent handles multiple sequential tool calls."""
        call_log = []

        def tool_a() -> str:
            call_log.append("A")
            return "Result A"

        def tool_b() -> str:
            call_log.append("B")
            return "Result B"

        schema = {"type": "object", "properties": {}, "required": []}

        # Set up mock for two tool calls
        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "tool_a", "arguments": {}}}],
                    }
                },
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "tool_b", "arguments": {}}}],
                    }
                },
                {"message": {"content": "Completed both tasks", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("tool_a", "Tool A", schema, tool_a))
        agent.register_tool(create_tool("tool_b", "Tool B", schema, tool_b))

        result = agent.run("Run both tools")

        assert call_log == ["A", "B"]
        assert "Completed" in result

    def test_agent_handles_tool_error(self, mock_ollama_client):
        """Agent gracefully handles tool execution errors."""

        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        schema = {"type": "object", "properties": {}, "required": []}

        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "fail", "arguments": {}}}],
                    }
                },
                {"message": {"content": "The tool failed, here's what happened", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        agent.register_tool(create_tool("fail", "A failing tool", schema, failing_tool))

        result = agent.run("Use the failing tool")

        # Agent should handle error and continue
        assert result is not None
        assert len(mock_ollama_client.call_history) == 2


class TestAgentConversation:
    """Tests for agent conversation management."""

    def test_agent_chat_maintains_history(self, mock_ollama_client):
        """chat() maintains message history across calls."""
        mock_ollama_client.set_responses(
            [
                {"message": {"content": "Hello!", "tool_calls": []}},
                {"message": {"content": "Your name is Alice", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))

        agent.run("My name is Alice")
        agent.chat("What is my name?")

        # Second call should have history from first
        second_call = mock_ollama_client.call_history[1]
        messages = second_call["messages"]

        # Should have system + user + assistant + user messages
        assert len(messages) >= 3

    def test_agent_run_resets_history(self, mock_ollama_client):
        """run() starts fresh each time."""
        mock_ollama_client.set_responses(
            [
                {"message": {"content": "First response", "tool_calls": []}},
                {"message": {"content": "Second response", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))

        agent.run("First task")
        agent.run("Second task")

        # Second run should start fresh - verify by checking the user message
        second_call = mock_ollama_client.call_history[1]
        messages = second_call["messages"]

        # Find user messages - should only have "Second task" not "First task"
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert len(user_messages) == 1
        assert "Second task" in user_messages[0]["content"]
        assert "First task" not in str(messages)  # No history from first run


class TestAgentWithFileTools:
    """Integration tests with file system tools."""

    def test_agent_reads_and_processes_file(self, mock_ollama_client, tmp_path):
        """Agent can read a file and process its contents."""
        # Create a test file
        test_file = tmp_path / "data.txt"
        test_file.write_text("The answer is 42")

        from agent.tools import get_file_tools

        # Mock: first call tool to read file, then respond
        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": str(test_file)},
                                }
                            }
                        ],
                    }
                },
                {"message": {"content": "The file says the answer is 42", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        for tool in get_file_tools():
            agent.register_tool(tool)

        result = agent.run(f"Read the file at {test_file}")

        assert "42" in result


class TestAgentWithMemoryTools:
    """Integration tests with memory system tools."""

    def test_agent_saves_and_retrieves_memory(
        self, mock_ollama_client, temp_vault, reset_memory_system
    ):
        """Agent can save and retrieve memories."""
        from agent.memory_system import get_memory_tools

        # Mock: save memory, then retrieve it
        mock_ollama_client.set_responses(
            [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "remember_permanently",
                                    "arguments": {
                                        "content": "User likes Python",
                                        "category": "preferences",
                                    },
                                }
                            }
                        ],
                    }
                },
                {"message": {"content": "I've saved that you like Python", "tool_calls": []}},
            ]
        )

        agent = Agent(AgentConfig(verbose=False))
        for tool in get_memory_tools(str(temp_vault)):
            agent.register_tool(tool)

        result = agent.run("Remember that I like Python")

        assert "saved" in result.lower() or "Python" in result

        # Verify the memory was actually saved
        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        assert memories_file.exists()
        assert "Python" in memories_file.read_text()


class TestAgentSystemPrompt:
    """Tests for system prompt customization."""

    def test_custom_system_prompt(self, mock_ollama_client):
        """Agent uses custom system prompt."""
        mock_ollama_client.set_responses(
            [{"message": {"content": "Arrr, hello!", "tool_calls": []}}]
        )

        config = AgentConfig(
            system_prompt="You are a pirate. Always speak like a pirate.", verbose=False
        )
        agent = Agent(config)

        agent.run("Hello")

        # Check the system prompt was sent
        call = mock_ollama_client.call_history[0]
        system_msg = next(m for m in call["messages"] if m["role"] == "system")
        assert "pirate" in system_msg["content"]

    def test_default_system_prompt(self, mock_ollama_client):
        """Agent has default system prompt."""
        mock_ollama_client.set_responses([{"message": {"content": "Hello", "tool_calls": []}}])

        agent = Agent(AgentConfig(verbose=False))
        agent.run("Hello")

        # Check a system prompt was sent
        call = mock_ollama_client.call_history[0]
        system_msg = next(m for m in call["messages"] if m["role"] == "system")
        assert "autonomous" in system_msg["content"].lower()


class TestOllamaOutageGracefulFallback:
    """A sustained Ollama outage must produce a clean fallback string and must
    not corrupt the vault's crash log with records of expected transport errors.
    """

    def test_repeated_connection_errors_return_fallback_not_traceback(
        self, mock_ollama_client, tmp_path, monkeypatch
    ):
        """Agent.run() swallows requests.ConnectionError and returns a string.

        Patches ``_ollama_client.chat`` to always raise, fires six consecutive
        ``Agent.run()`` calls, and asserts:

        * every result (including the 6th) is a short fallback string — never
          a raw Python traceback leaked back to the caller;
        * ``crash_log.md`` pre-seeded in a temp vault is not touched by
          Agent.run() — mtime and contents stay at the baseline, confirming
          expected Ollama outages don't poison the crash log.
        """
        import agent.ollama_health as health_module

        # Keep the test fast: zero retries means one attempt per call, no
        # exponential-backoff sleep between them. The retry path itself is
        # already covered by ollama_health unit tests; here we care about
        # the failure being wrapped cleanly by Agent.run().
        monkeypatch.setattr(health_module.settings, "ollama_max_retries", 0)

        # Reset the singleton health monitor so prior tests that marked
        # Ollama ``down`` don't leak into this one, and so we leave a clean
        # state behind.
        fresh_state = health_module._HealthState()
        monkeypatch.setattr(health_module._monitor, "_state", fresh_state)

        # Every chat call blows up with a transport-layer error that
        # ``is_transient_error`` recognizes — the retry path will burn through
        # its budget and re-raise, then Agent.run() catches it.
        mock_ollama_client.chat = MagicMock(
            side_effect=requests.ConnectionError("Ollama unreachable")
        )

        # Seed a crash_log.md in a temp vault so we can prove Agent.run()
        # didn't touch it. We check both mtime and contents — mtime alone is
        # a weak signal on filesystems with coarse timestamp resolution.
        crash_log = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        crash_log.parent.mkdir(parents=True)
        crash_log.write_text("# sentinel — must not be overwritten\n", encoding="utf-8")
        baseline_mtime = crash_log.stat().st_mtime
        baseline_contents = crash_log.read_text(encoding="utf-8")

        agent = Agent(AgentConfig(verbose=False))

        results = [agent.run("ping") for _ in range(6)]

        # Each call attempted Ollama exactly once (max_retries=0 → 1 attempt).
        assert mock_ollama_client.chat.call_count == 6

        # Every result — crucially the 6th — is a clean fallback string.
        for idx, result in enumerate(results):
            assert isinstance(result, str), f"call {idx + 1} did not return a string"
            assert result, f"call {idx + 1} returned an empty string"
            assert "Traceback" not in result, (
                f"call {idx + 1} leaked a Python traceback: {result!r}"
            )

        sixth = results[5]
        assert "Agent error" in sixth, (
            f"6th call missing fallback sentinel: {sixth!r}"
        )

        # The vault's crash_log.md was not written to by Agent.run(): the
        # fallback path lives entirely in-process. write_crash_log is reserved
        # for uncaught exceptions escaping to sys.excepthook — not expected
        # transport failures that the agent handles gracefully.
        assert crash_log.stat().st_mtime == baseline_mtime
        assert crash_log.read_text(encoding="utf-8") == baseline_contents
