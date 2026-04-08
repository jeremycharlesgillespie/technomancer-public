"""
Shared test fixtures for local-agent test suite.

Provides mocks for:
- Ollama client (avoids needing Ollama running)
- Anthropic client (avoids needing API key)
- Temporary Obsidian vault structure
- Memory system initialized with temp vault
"""

from unittest.mock import MagicMock

import pytest

# =============================================================================
# OLLAMA MOCK
# =============================================================================


class MockOllamaResponse:
    """Mock response from Ollama chat API."""

    def __init__(self, content: str = "", tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockOllamaClient:
    """Mock ollama.Client that returns configurable responses."""

    def __init__(self):
        self.responses = []  # Queue of responses to return
        self.call_history = []  # Track calls for assertions
        self.default_response = {
            "message": {"content": "Mock response from Ollama", "tool_calls": []}
        }

    def set_responses(self, responses: list):
        """Set a queue of responses to return in order."""
        self.responses = list(responses)

    def set_tool_call_response(self, tool_name: str, arguments: dict, final_response: str = "Done"):
        """Helper to set up a tool call followed by final response."""
        self.responses = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
                }
            },
            {"message": {"content": final_response, "tool_calls": []}},
        ]

    def chat(self, model: str, messages: list, tools: list = None, options: dict = None, **kwargs):
        """Mock chat endpoint."""
        self.call_history.append(
            {"model": model, "messages": messages, "tools": tools, "options": options}
        )

        if self.responses:
            return self.responses.pop(0)
        return self.default_response


@pytest.fixture
def mock_ollama_client(monkeypatch):
    """
    Mock the global _ollama_client in agent.core.

    Usage:
        def test_something(mock_ollama_client):
            mock_ollama_client.set_responses([...])
            # Agent will use mock responses
    """
    mock = MockOllamaClient()
    monkeypatch.setattr("agent.core._ollama_client", mock)
    return mock


# =============================================================================
# ANTHROPIC MOCK
# =============================================================================


class MockAnthropicContent:
    """Mock content block from Anthropic response."""

    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class MockAnthropicResponse:
    """Mock response from Anthropic messages API."""

    def __init__(self, text: str):
        self.content = [MockAnthropicContent(text)]
        self.stop_reason = "end_turn"


class MockAnthropicMessages:
    """Mock messages API."""

    def __init__(self):
        self.response_text = "Mock response from Claude"
        self.call_history = []

    def create(self, **kwargs):
        self.call_history.append(kwargs)
        return MockAnthropicResponse(self.response_text)


class MockAnthropicClient:
    """Mock anthropic.Anthropic client."""

    def __init__(self, **kwargs):
        self.messages = MockAnthropicMessages()


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """
    Mock the anthropic.Anthropic client.

    Usage:
        def test_claude(mock_anthropic_client):
            mock_anthropic_client.messages.response_text = "Custom response"
    """
    mock_instance = MockAnthropicClient()

    def mock_init(*args, **kwargs):
        return mock_instance

    monkeypatch.setattr("anthropic.Anthropic", mock_init)
    return mock_instance


# =============================================================================
# TEMPORARY VAULT
# =============================================================================


@pytest.fixture
def temp_vault(tmp_path):
    """
    Create a temporary Obsidian vault structure.

    Structure:
        tmp_path/
            LLM Memory/
                Conversations/
                Context/
                Permanent/
                    memories.md (empty header)
                    enhancements.md (empty header)

    Returns:
        Path to the vault root (tmp_path, not tmp_path/LLM Memory)
    """
    vault_root = tmp_path
    memory_root = vault_root / "LLM Memory"

    # Create directory structure
    (memory_root / "Conversations").mkdir(parents=True)
    (memory_root / "Context").mkdir(parents=True)
    (memory_root / "Permanent").mkdir(parents=True)

    return vault_root


@pytest.fixture
def temp_vault_with_files(temp_vault):
    """
    Temp vault with some pre-populated files for testing.
    """
    memory_root = temp_vault / "LLM Memory"

    # Create memories.md with some content
    memories_file = memory_root / "Permanent" / "memories.md"
    memories_file.write_text(
        """# Permanent Memories

Important information to always remember.

---

## 2026-03-14 10:00 - user_preferences
User prefers dark mode and concise responses.

## 2026-03-14 11:00 - facts
The project uses Python 3.10+ and Ollama.
""",
        encoding="utf-8",
    )

    # Create enhancements.md
    enhancements_file = memory_root / "Permanent" / "enhancements.md"
    enhancements_file.write_text(
        """# Enhancement Queue

Ideas and feature requests for Claude to implement.

---

## Pending

- [ ] **#1:** Add dark mode support (2026-03-14)
- [ ] **#2:** Improve error messages (2026-03-14)

## In Progress

## Completed

- [x] **#0:** Initial setup (2026-03-13)
""",
        encoding="utf-8",
    )

    return temp_vault


# =============================================================================
# MEMORY SYSTEM
# =============================================================================


@pytest.fixture
def memory_system(temp_vault, monkeypatch):
    """
    MemorySystem initialized with temporary vault.

    Also resets the global _memory_system to avoid test pollution.
    """
    # Reset global state
    import agent.memory_system as mem_module

    monkeypatch.setattr(mem_module, "_memory_system", None)

    # Import and initialize
    from agent.memory_system import MemorySystem

    return MemorySystem(str(temp_vault))


@pytest.fixture
def reset_memory_system(monkeypatch):
    """Reset the global memory system state between tests."""
    import agent.memory_system as mem_module

    monkeypatch.setattr(mem_module, "_memory_system", None)
    yield
    monkeypatch.setattr(mem_module, "_memory_system", None)


# =============================================================================
# ENHANCEMENTS - Patched vault path
# =============================================================================


@pytest.fixture
def patched_enhancements(temp_vault, monkeypatch):
    """
    Patch enhancements.py to use temp vault instead of hardcoded path.
    """
    import agent.enhancements as enh_module

    temp_vault_path = temp_vault / "LLM Memory"
    monkeypatch.setattr(enh_module, "VAULT_PATH", temp_vault_path)
    monkeypatch.setattr(
        enh_module, "ENHANCEMENTS_FILE", temp_vault_path / "Permanent" / "enhancements.md"
    )

    return temp_vault_path


# =============================================================================
# ACCOUNTABILITY - Patched vault path
# =============================================================================


@pytest.fixture
def patched_accountability(temp_vault, monkeypatch):
    """
    Patch accountability.py to use temp vault instead of hardcoded path.
    """
    import agent.accountability as acc_module

    temp_vault_path = temp_vault / "LLM Memory"
    monkeypatch.setattr(acc_module, "VAULT_PATH", temp_vault_path)

    return temp_vault_path


# =============================================================================
# KNOWLEDGE GAPS - Patched vault path
# =============================================================================


@pytest.fixture
def patched_knowledge_gaps(temp_vault, monkeypatch):
    """
    Patch knowledge_gaps.py to use temp vault instead of hardcoded path.
    """
    import agent.knowledge_gaps as kg_module

    temp_vault_path = temp_vault / "LLM Memory"
    monkeypatch.setattr(kg_module, "VAULT_PATH", temp_vault_path)
    monkeypatch.setattr(
        kg_module, "GAPS_FILE", temp_vault_path / "Permanent" / "knowledge_gaps.md"
    )

    return temp_vault_path


# =============================================================================
# DISCORD MOCKS
# =============================================================================


class MockDiscordUser:
    """Mock Discord user."""

    def __init__(self, name: str = "testuser", bot: bool = False):
        self.name = name
        self.bot = bot
        self.id = 12345


class MockDiscordChannel:
    """Mock Discord channel."""

    def __init__(self, name: str = "llm_chat"):
        self.name = name
        self.id = 67890
        self.sent_messages = []

    async def send(self, content: str = None, file=None):
        self.sent_messages.append({"content": content, "file": file})
        return MagicMock()


class MockDiscordMessage:
    """Mock Discord message."""

    def __init__(
        self,
        content: str = "",
        author_name: str = "testuser",
        channel_name: str = "llm_chat",
        is_bot: bool = False,
    ):
        self.content = content
        self.author = MockDiscordUser(author_name, is_bot)
        self.channel = MockDiscordChannel(channel_name)
        self.attachments = []
        self.reference = None
        self.replied_to = None

    async def reply(self, content: str):
        self.replied_to = content
        return MagicMock()


@pytest.fixture
def mock_discord_message():
    """Factory fixture for creating mock Discord messages."""
    return MockDiscordMessage


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def sample_tool_function():
    """A simple tool function for testing."""

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return greet


@pytest.fixture
def sample_tool_schema():
    """A simple tool schema for testing."""
    return {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Name to greet"}},
        "required": ["name"],
    }


# =============================================================================
# DEV LEARNING - Patched state file
# =============================================================================


@pytest.fixture
def patched_dev_learning(tmp_path, monkeypatch):
    """
    Patch dev_learning.py to use temp directory for state files and articles.

    Also patches the profile loading to return a test profile.
    """
    import agent.dev_learning as dl_module

    # Use temp path for state file
    monkeypatch.setattr(dl_module, "SENT_TOPICS_FILE", tmp_path / "sent_topics.json")

    # Use temp path for learning articles (prevents tests from writing to real vault)
    monkeypatch.setattr(dl_module, "LEARNING_ARTICLES_DIR", tmp_path / "Learning")

    # Mock user profile to avoid reading from actual vault
    test_profile = {
        "role": "software developer",
        "stack": ["Python", "Django", "PostgreSQL"],
        "interests": ["AI", "automation"],
    }

    def mock_load_profile():
        return test_profile

    monkeypatch.setattr(dl_module, "load_user_profile", mock_load_profile)

    return tmp_path
