"""
Shared test fixtures for local-agent test suite.

Provides mocks for:
- Ollama client (avoids needing Ollama running)
- Anthropic client (avoids needing API key)
- Temporary Obsidian vault structure
- Memory system initialized with temp vault
"""

# Install the claude -p shims before any test module imports ollama /
# anthropic. Production wires them in agent/__init__.py; this ensures
# tests that import ollama directly (without going through the agent
# package first) also pick them up.
from agent.ollama_shim import install_as_ollama as _install_ollama_shim
from agent.anthropic_shim import install_as_anthropic as _install_anthropic_shim
_install_ollama_shim()
_install_anthropic_shim()

# Import to access leak_counter for test reset
import agent.executor_runs_db as executor_runs_db

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _block_git_cleanup():
    """Prevent AIM/Worker git cleanup from running real `git checkout --force main`.

    The production cleanup wipes the current branch when a worker fails.
    Without this guard, any test that exercises handle_worker_failure or
    _ensure_git_clean wipes the feature branch pytest is running from.
    """
    with patch("aim.manager._cleanup_git_and_executions"), \
         patch("aim.worker._ensure_git_clean"):
        yield


@pytest.fixture(autouse=True)
def _force_local_board_provider():
    """Force LocalProvider in tests so Jira API is never hit.

    Production flips to JiraProvider when is_jira_configured() is true.
    Tests use LocalProvider via existing mocks of idea_board.models.*,
    so we force that path regardless of the .env state.
    """
    from board import reset_provider

    reset_provider()
    with patch("board.factory.is_jira_configured", return_value=False):
        yield
    reset_provider()

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
# EMBEDDING STORE - Isolated DB
# =============================================================================


@pytest.fixture(autouse=True)
def _block_jira_sync(monkeypatch):
    """Prevent Jira sync from firing during tests.

    Blocks both the background sync thread used by LocalProvider writes
    AND any direct Jira API call made by JiraProvider, create_jira_issue,
    or other code paths. If a test needs real Jira-like responses, it
    must opt in with its own ``patch("...jira_sync._api")`` or
    ``patch("board.jira_provider._api")``.

    Without the _api guard, tests that exercise JiraProvider code paths
    without mocking create_jira_issue can accidentally create real Jira
    issues (we lost TK-364 and TK-365 this way during Phase 2).
    """
    monkeypatch.setattr(
        "idea_board.models._jira_sync_background",
        lambda idea: None,
    )

    def _raise_on_real_jira(*args, **kwargs):
        raise RuntimeError(
            "Test attempted a real Jira API call. Mock _api in your test "
            "(patch 'idea_board.jira_sync._api' and/or 'board.jira_provider._api')."
        )

    monkeypatch.setattr("idea_board.jira_sync._api", _raise_on_real_jira)
    monkeypatch.setattr("board.jira_provider._api", _raise_on_real_jira)


@pytest.fixture
def mock_dedup_llm(monkeypatch):
    """Control the dedup verdict used by ``review_queue`` and ``add_idea``.

    Patches every layer of the dedup seam with a single configurable
    ``MagicMock``. The default side-effect normalises whatever shape a
    test sets via ``mock_dedup_llm.return_value = True/False`` into the
    ``tuple[bool, str]`` shape that both seams now expose (TK-766), so
    legacy tests written against the bool API keep working.

      * ``idea_board.models._is_duplicate`` — the public seam every
        caller (``add_idea``, queue review, ``/api/jira/create``) goes
        through. Returns ``tuple[bool, str]`` (TK-766).
      * ``idea_board.dedup_llm.is_near_exact_duplicate`` — the LLM
        near-exact judge. Already returns ``tuple[bool, str]``.
        Patched with ``raising=False`` and guarded with an import
        probe so the fixture is safe to load even when the module is
        not importable in the current environment.

    The same ``MagicMock`` instance is bound to both targets so a test
    can ``return_value=`` / ``side_effect=`` once and have it apply
    everywhere, and ``call_count`` / ``call_args`` aggregate across
    layers.

    Usage:

        def test_something(mock_dedup_llm):
            mock_dedup_llm.return_value = True   # treat pair as duplicate
            ...
            mock_dedup_llm.return_value = False  # treat pair as distinct
            # Or set the full tuple explicitly:
            mock_dedup_llm.return_value = (True, "title_overlap=0.9")
    """

    def _coerce(*args, **kwargs):
        # Honor side_effect if a test set one, else use the stored
        # return_value (which may be True/False from legacy tests or a
        # tuple from newer callers).
        raw = mock.return_value
        if isinstance(raw, tuple):
            return raw
        if isinstance(raw, bool):
            return raw, "mocked"
        # Fallback for MagicMock sentinel / arbitrary values
        return bool(raw), "mocked"

    mock = MagicMock()
    mock.return_value = False  # default: not a duplicate
    mock.side_effect = _coerce

    # Legacy seam — the function every caller already routes through.
    monkeypatch.setattr("idea_board.models._is_duplicate", mock)

    # Future LLM judge target. The module is introduced by a separate
    # story (TK-743); guard the patch so this fixture is safe to add
    # before that lands. Once the module exists, the attribute may or
    # may not have been bound at import time, so use ``raising=False``
    # to cover the bootstrap window too.
    try:
        import importlib

        importlib.import_module("idea_board.dedup_llm")
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            "idea_board.dedup_llm.is_near_exact_duplicate", mock, raising=False
        )

    return mock


@pytest.fixture(autouse=True)
def _isolate_embedding_store(tmp_path, monkeypatch):
    """Point embedding_store at a temporary SQLite DB for each test."""
    import agent.embedding_store as es_module

    monkeypatch.setattr(es_module, "DB_DIR", tmp_path)
    monkeypatch.setattr(es_module, "DB_PATH", tmp_path / "embeddings.db")
    es_module._local.__dict__.pop("emb_conn", None)
    yield
    conn = getattr(es_module._local, "emb_conn", None)
    if conn:
        conn.close()
        es_module._local.emb_conn = None


@pytest.fixture(autouse=True)
def _reset_leak_counter():
    """Reset the leak_counter at the start of each test to ensure clean state."""
    # Reset the leak_counter for executor_runs_db
    # Since leak_counter is a threading.local(), we need to clear its attributes
    # by setting them to None or default values
    try:
        # Clear any existing attributes on the threading.local object
        for attr in dir(executor_runs_db.leak_counter):
            if not attr.startswith('_'):
                setattr(executor_runs_db.leak_counter, attr, None)
    except Exception:
        # If there's an error, just continue - the fixture should not break tests
        pass


# =============================================================================
# DISCORD MOCKS
# =============================================================================


class MockDiscordUser:
    """Mock Discord user."""

    def __init__(self, name: str = "testuser", bot: bool = False, user_id: int = 12345):
        self.name = name
        self.display_name = name
        self.bot = bot
        self.id = user_id

    def __eq__(self, other):
        if isinstance(other, MockDiscordUser):
            return self.id == other.id
        return NotImplemented


class MockMessageReference:
    """Mock message reference (for replies)."""

    def __init__(self, message_id: int = 0):
        self.message_id = message_id


class _TypingCtx:
    """Async context manager for channel.typing()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSentMessage:
    """Mock return value from channel.send() or message.reply()."""

    _counter = 0

    def __init__(self, content: str = ""):
        MockSentMessage._counter += 1
        self.id = 100000 + MockSentMessage._counter
        self.content = content


class MockDiscordChannel:
    """Mock Discord channel with full API surface."""

    def __init__(self, name: str = "llm_chat"):
        self.name = name
        self.id = 67890
        self.sent_messages = []
        self._stored_messages = {}  # id -> MockDiscordMessage for fetch_message

    async def send(self, content: str = None, file=None, files=None, **kwargs):
        sent = MockSentMessage(content or "")
        self.sent_messages.append({"content": content, "file": file, "files": files})
        return sent

    def typing(self):
        return _TypingCtx()

    async def fetch_message(self, message_id: int):
        return self._stored_messages.get(message_id)

    def store_message(self, msg):
        """Test helper: store a message for fetch_message to return."""
        self._stored_messages[msg.id if hasattr(msg, "id") else 0] = msg


class MockDiscordAttachment:
    """Mock Discord attachment."""

    def __init__(self, filename: str = "file.txt", url: str = "https://cdn.discord.com/file.txt"):
        self.filename = filename
        self.url = url
        self.size = 1000

    async def save(self, path):
        pass


class MockDiscordMessage:
    """Mock Discord message with full API surface."""

    _counter = 0

    def __init__(
        self,
        content: str = "",
        author_name: str = "testuser",
        channel_name: str = "llm_chat",
        is_bot: bool = False,
    ):
        MockDiscordMessage._counter += 1
        self.id = 200000 + MockDiscordMessage._counter
        self.content = content
        self.author = MockDiscordUser(author_name, is_bot)
        self.channel = MockDiscordChannel(channel_name)
        self.attachments = []
        self.reference = None
        self.replied_to = None
        self._replies = []

    async def reply(self, content: str, **kwargs):
        self.replied_to = content
        self._replies.append(content)
        return MockSentMessage(content)


class MockRawReactionPayload:
    """Mock discord.RawReactionActionEvent."""

    def __init__(self, user_id: int = 12345, message_id: int = 100001,
                 emoji: str = "\U0001f44d", member_name: str = "testuser"):
        self.user_id = user_id
        self.message_id = message_id
        self.emoji = MagicMock(__str__=MagicMock(return_value=emoji))
        self.member = MockDiscordUser(member_name) if member_name else None


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
# /live PAGE DATA - Mocked executor_runs_db + AIM state files
# =============================================================================


@pytest.fixture
def mock_executor_runs_db(monkeypatch):
    """Patch ``executor_runs_db.get_current_execution_per_project`` so /live
    tests never touch real AIM state files.

    Returns a ``MagicMock`` seeded with one executing row so any test hitting
    the /live landing page sees a populated "Currently Executing" table
    without disk I/O. Override per-test via ``mock.return_value = [...]``.
    """
    import agent.executor_runs_db as db_module

    default_rows = [
        {
            "project": "TK",
            "status": "executing",
            "current_idea_id": "TK-793",
            "started_at": "2026-04-19T00:00:00",
            "last_observation": "running tests",
            "is_executing": True,
        },
    ]
    mock = MagicMock(return_value=default_rows)
    monkeypatch.setattr(db_module, "get_current_execution_per_project", mock)
    return mock


@pytest.fixture
def mock_aim_state_files(monkeypatch):
    """Patch ``executor_runs_db._read_aim_state_file`` so /live tests never
    open real AIM state files on disk.

    Returns a ``MagicMock`` whose default payload matches the AIM schema
    (``worker`` + ``board_snapshot.recent_completions``). Override per-test
    via ``mock.return_value = {...}`` or ``mock.side_effect = [...]``.
    """
    import agent.executor_runs_db as db_module

    default_state = {
        "worker": {
            "status": "executing",
            "current_idea_id": "TK-793",
            "started_at": "2026-04-19T00:00:00",
            "last_observation": "running tests",
        },
        "board_snapshot": {
            "recent_completions": [
                {
                    "key": "TK-792",
                    "summary": "Previous story",
                    "resolved": "2026-04-18T23:00:00+0900",
                },
            ],
        },
    }
    mock = MagicMock(return_value=default_state)
    monkeypatch.setattr(db_module, "_read_aim_state_file", mock)
    return mock


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
