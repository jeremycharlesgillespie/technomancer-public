"""
Integration tests for full agent workflow with mocked Ollama.
"""

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
