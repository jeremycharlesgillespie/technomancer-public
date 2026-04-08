"""
Tests for claude_vault module.

Tests vault data extraction, tool functions, and ClaudeVaultSession.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def vault_with_profile(temp_vault):
    """Temp vault with profile.md and memories.md."""
    memory_root = temp_vault / "LLM Memory"

    # Create profile.md
    profile_file = memory_root / "Permanent" / "profile.md"
    profile_file.write_text(
        """# User Profile

## Role
software developer

## Tech Stack
- Python
- Django
- PostgreSQL
- AWS

## Interests
- AI
- automation
- LLM

## Currently Learning
- Rust
""",
        encoding="utf-8",
    )

    # Create memories.md with resume-like content
    memories_file = memory_root / "Permanent" / "memories.md"
    memories_file.write_text(
        """# Permanent Memories

---

## 2026-03-14 10:00 - user_info
Name: Jeremy Gillespie
Discord: testuser

---

## 2026-03-14 10:00 - resume
**Current Role:** Senior Software Engineer at Acme Corp (2022-present)
- Working on cloud infrastructure with AWS CDK
- Python, Django, PostgreSQL stack

**Previous Experience:**
- Additech Inc. (2020-2022): DevOps Engineer
- 7-Eleven Corporate (2018-2020): Systems Administrator
- US Army (2010-2014): IT Specialist 25B

---

## 2026-03-14 11:00 - extracted_facts
User prefers dark mode and concise responses.
""",
        encoding="utf-8",
    )

    # Create a conversation file
    conversations_dir = memory_root / "Conversations"
    today = datetime.now().strftime("%Y-%m-%d")
    conv_file = conversations_dir / f"{today}.md"
    conv_file.write_text(
        """# Conversations

### 10:00:00 - testuser
**Q:** Hello!
**A:** Hi there! How can I help you today?

### 10:05:00 - testuser
**Q:** What's the weather like?
**A:** I don't have access to weather data, but you can check weather.com.
""",
        encoding="utf-8",
    )

    return temp_vault


@pytest.fixture
def mock_anthropic_with_cache(monkeypatch):
    """Mock Anthropic client that includes cache usage stats."""

    class MockUsage:
        def __init__(self):
            self.input_tokens = 500
            self.output_tokens = 200
            self.cache_read_input_tokens = 3000
            self.cache_creation_input_tokens = 0

    class MockContent:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class MockResponse:
        def __init__(self, text="Mock response"):
            self.content = [MockContent(text)]
            self.usage = MockUsage()
            self.stop_reason = "end_turn"

    class MockMessages:
        def __init__(self):
            self.call_history = []
            self.response_text = "Mock response from Claude"

        def create(self, **kwargs):
            self.call_history.append(kwargs)
            return MockResponse(self.response_text)

    class MockClient:
        def __init__(self, **kwargs):
            self.messages = MockMessages()

    mock_client = MockClient()

    def mock_init(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("anthropic.Anthropic", mock_init)
    return mock_client


@pytest.fixture
def patched_claude_vault(vault_with_profile, monkeypatch):
    """Patch claude_vault to use temp vault."""
    import agent.claude_vault as cv_module

    vault_path = vault_with_profile / "LLM Memory"
    monkeypatch.setattr(cv_module, "_session", None)

    # Patch settings to use temp vault
    mock_settings = MagicMock()
    mock_settings.llm_memory_path = vault_path
    mock_settings.anthropic_api_key = "test-api-key"
    mock_settings.claude_cache_ttl = "5m"
    mock_settings.claude_use_vault_context = True
    monkeypatch.setattr(cv_module, "settings", mock_settings)

    return vault_path


# =============================================================================
# TESTS: VAULT DATA EXTRACTION
# =============================================================================


class TestLoadProfile:
    """Tests for load_profile function."""

    def test_loads_role(self, vault_with_profile):
        """Should extract user role from profile."""
        from agent.claude_vault import load_profile

        vault_path = vault_with_profile / "LLM Memory"
        profile = load_profile(vault_path)

        assert profile["role"] == "software developer"

    def test_loads_stack(self, vault_with_profile):
        """Should extract tech stack list."""
        from agent.claude_vault import load_profile

        vault_path = vault_with_profile / "LLM Memory"
        profile = load_profile(vault_path)

        assert "Python" in profile["stack"]
        assert "Django" in profile["stack"]
        assert "PostgreSQL" in profile["stack"]

    def test_loads_interests(self, vault_with_profile):
        """Should extract interests list."""
        from agent.claude_vault import load_profile

        vault_path = vault_with_profile / "LLM Memory"
        profile = load_profile(vault_path)

        assert "AI" in profile["interests"]
        assert "automation" in profile["interests"]

    def test_handles_missing_profile(self, temp_vault):
        """Should return defaults when profile.md missing."""
        from agent.claude_vault import load_profile

        vault_path = temp_vault / "LLM Memory"
        profile = load_profile(vault_path)

        assert profile["role"] == "software developer"
        assert profile["stack"] == []


class TestBuildUserSummary:
    """Tests for build_user_summary function."""

    def test_includes_profile_data(self, vault_with_profile):
        """Should include profile data in summary."""
        from agent.claude_vault import build_user_summary

        vault_path = vault_with_profile / "LLM Memory"
        summary = build_user_summary(vault_path)

        assert "Quick Reference" in summary
        assert "Python" in summary or "Stack:" in summary

    def test_handles_missing_files(self, temp_vault):
        """Should handle missing profile gracefully."""
        from agent.claude_vault import build_user_summary

        vault_path = temp_vault / "LLM Memory"
        summary = build_user_summary(vault_path)

        assert "Quick Reference" in summary


class TestBuildCachedPrefix:
    """Tests for build_cached_prefix function."""

    def test_returns_list_of_blocks(self, vault_with_profile):
        """Should return list of content blocks."""
        from agent.claude_vault import build_cached_prefix

        vault_path = vault_with_profile / "LLM Memory"
        blocks = build_cached_prefix(vault_path)

        assert isinstance(blocks, list)
        assert len(blocks) >= 3

    def test_has_cache_control_on_last_block(self, vault_with_profile):
        """Should have cache_control on the final block."""
        from agent.claude_vault import build_cached_prefix

        vault_path = vault_with_profile / "LLM Memory"
        blocks = build_cached_prefix(vault_path)

        # Last block should have cache_control
        last_block = blocks[-1]
        assert "cache_control" in last_block
        assert last_block["cache_control"]["type"] == "ephemeral"

    def test_includes_permanent_memories(self, vault_with_profile):
        """Should include memories.md content."""
        from agent.claude_vault import build_cached_prefix

        vault_path = vault_with_profile / "LLM Memory"
        blocks = build_cached_prefix(vault_path)

        # Check that memories content is in one of the blocks
        full_text = "".join(b.get("text", "") for b in blocks)
        assert "Jeremy Gillespie" in full_text or "Permanent" in full_text


# =============================================================================
# TESTS: VAULT TOOLS
# =============================================================================


class TestReadVaultFile:
    """Tests for read_vault_file function."""

    def test_reads_existing_file(self, vault_with_profile):
        """Should read existing vault file."""
        from agent.claude_vault import read_vault_file

        vault_path = vault_with_profile / "LLM Memory"
        content = read_vault_file("Permanent/profile.md", vault_path)

        assert "software developer" in content
        assert "Python" in content

    def test_handles_missing_file(self, vault_with_profile):
        """Should return error for missing file."""
        from agent.claude_vault import read_vault_file

        vault_path = vault_with_profile / "LLM Memory"
        result = read_vault_file("nonexistent.md", vault_path)

        assert "not found" in result.lower()


class TestSearchVault:
    """Tests for search_vault function."""

    def test_finds_matching_content(self, vault_with_profile):
        """Should find content matching query."""
        from agent.claude_vault import search_vault

        vault_path = vault_with_profile / "LLM Memory"
        result = search_vault("Python", vault_path=vault_path)

        assert "Found" in result
        assert "profile.md" in result or "memories.md" in result

    def test_returns_no_results_message(self, vault_with_profile):
        """Should return message when no matches."""
        from agent.claude_vault import search_vault

        vault_path = vault_with_profile / "LLM Memory"
        result = search_vault("xyznonexistent123", vault_path=vault_path)

        assert "No results" in result


class TestListVaultContents:
    """Tests for list_vault_contents function."""

    def test_lists_root_directory(self, vault_with_profile):
        """Should list root directory contents."""
        from agent.claude_vault import list_vault_contents

        vault_path = vault_with_profile / "LLM Memory"
        result = list_vault_contents("", vault_path)

        assert "Permanent" in result or "Conversations" in result

    def test_lists_subdirectory(self, vault_with_profile):
        """Should list subdirectory contents."""
        from agent.claude_vault import list_vault_contents

        vault_path = vault_with_profile / "LLM Memory"
        result = list_vault_contents("Permanent", vault_path)

        assert "profile.md" in result or "memories.md" in result

    def test_handles_missing_directory(self, vault_with_profile):
        """Should handle missing directory."""
        from agent.claude_vault import list_vault_contents

        vault_path = vault_with_profile / "LLM Memory"
        result = list_vault_contents("nonexistent_dir", vault_path)

        assert "not found" in result.lower()


class TestGetRecentConversations:
    """Tests for get_recent_conversations function."""

    def test_retrieves_conversations(self, vault_with_profile):
        """Should retrieve recent conversations."""
        from agent.claude_vault import get_recent_conversations

        vault_path = vault_with_profile / "LLM Memory"
        result = get_recent_conversations(hours=24, vault_path=vault_path)

        assert "Hello" in result or "weather" in result or "Conversations" in result

    def test_handles_no_conversations(self, temp_vault):
        """Should handle empty conversations directory."""
        from agent.claude_vault import get_recent_conversations

        vault_path = temp_vault / "LLM Memory"
        result = get_recent_conversations(hours=24, vault_path=vault_path)

        assert "No conversations" in result


# =============================================================================
# TESTS: CLAUDE VAULT SESSION
# =============================================================================


class TestClaudeVaultSession:
    """Tests for ClaudeVaultSession class."""

    def test_init_builds_prefix(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should build cached prefix on init."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)

        assert session._cached_prefix is not None
        assert len(session._cached_prefix) >= 3

    def test_ask_uses_client(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should call Anthropic API with cached prefix."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        response = session.ask("What is my name?")

        assert response == "Mock response from Claude"
        assert len(mock_anthropic_with_cache.messages.call_history) == 1

        # Verify system prompt was passed
        call = mock_anthropic_with_cache.messages.call_history[0]
        assert "system" in call
        assert isinstance(call["system"], list)

    def test_tracks_usage_stats(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should track token usage statistics."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("Test question")

        stats = session.get_usage_stats()
        assert stats["cache_read_tokens"] == 3000
        assert stats["total_input_tokens"] == 500
        assert stats["estimated_savings_tokens"] == 2700  # 90% of 3000

    def test_refresh_cache(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should rebuild prefix when refreshed."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        original_time = session._prefix_built_at

        session.refresh_cache()

        assert session._prefix_built_at > original_time

    def test_handles_missing_api_key(self, patched_claude_vault, monkeypatch):
        """Should handle missing API key gracefully."""
        # Remove API key
        import agent.claude_vault as cv_module
        from agent.claude_vault import ClaudeVaultSession

        cv_module.settings.anthropic_api_key = None

        session = ClaudeVaultSession(vault_path=patched_claude_vault, api_key=None)
        response = session.ask("Test")

        assert "Error" in response or "not available" in response


# =============================================================================
# TESTS: CONVENIENCE FUNCTIONS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_vault_session_creates_singleton(
        self, patched_claude_vault, mock_anthropic_with_cache
    ):
        """Should create and return singleton session."""
        # Reset global state
        import agent.claude_vault as cv_module
        from agent.claude_vault import get_vault_session, init_vault_session

        cv_module._session = None

        # Initialize
        init_vault_session(vault_path=patched_claude_vault)

        # Get should return same instance
        session1 = get_vault_session()
        session2 = get_vault_session()

        assert session1 is session2

    def test_ask_claude_with_vault(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should work as convenience function."""
        import agent.claude_vault as cv_module
        from agent.claude_vault import ask_claude_with_vault, init_vault_session

        cv_module._session = None

        init_vault_session(vault_path=patched_claude_vault)
        response = ask_claude_with_vault("Test question")

        assert response == "Mock response from Claude"


# =============================================================================
# TESTS: TOOL DEFINITIONS
# =============================================================================


class TestToolDefinitions:
    """Tests for VAULT_TOOLS definitions."""

    def test_all_tools_have_required_fields(self):
        """All tools should have name, description, input_schema."""
        from agent.claude_vault import VAULT_TOOLS

        for tool in VAULT_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_functions_map_exists(self):
        """Tool function map should include all tools."""
        from agent.claude_vault import VAULT_TOOLS, get_vault_tool_functions

        functions = get_vault_tool_functions()

        for tool in VAULT_TOOLS:
            assert tool["name"] in functions


# =============================================================================
# TESTS: COST TRACKING
# =============================================================================


class TestCostTracking:
    """Tests for per-request cost tracking."""

    def test_tracks_per_request_tokens(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should track tokens for last request separately."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("Test question")

        assert session.last_cache_read_tokens == 3000
        assert session.last_input_tokens == 500
        assert session.last_output_tokens == 200

    def test_get_last_request_cost(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should calculate cost of last request."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("Test question")

        cost = session.get_last_request_cost()
        # Cost should be positive and reasonable
        assert cost > 0
        assert cost < 1.0  # Should be well under $1 for a simple request

    def test_format_last_cost(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should format cost as readable string."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("Test question")

        formatted = session.format_last_cost()
        assert formatted.startswith("$")
        assert len(formatted) > 1

    def test_get_total_cost(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should calculate cumulative session cost."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("First question")
        session.ask("Second question")

        total = session.get_total_cost()
        last = session.get_last_request_cost()

        # Total should be approximately 2x last (same mock response each time)
        assert total >= last

    def test_usage_stats_includes_cost(self, patched_claude_vault, mock_anthropic_with_cache):
        """Usage stats should include total cost."""
        from agent.claude_vault import ClaudeVaultSession

        session = ClaudeVaultSession(vault_path=patched_claude_vault)
        session.ask("Test question")

        stats = session.get_usage_stats()
        assert "total_cost_usd" in stats
        assert stats["total_cost_usd"] > 0

    def test_get_last_request_cost_function(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should expose cost via module-level function."""
        import agent.claude_vault as cv_module
        from agent.claude_vault import get_last_request_cost, init_vault_session

        cv_module._session = None
        init_vault_session(vault_path=patched_claude_vault)

        # Make a request first
        cv_module.get_vault_session().ask("Test")

        cost = get_last_request_cost()
        assert cost.startswith("$")

    def test_ask_claude_with_cost(self, patched_claude_vault, mock_anthropic_with_cache):
        """Should return both response and cost."""
        import agent.claude_vault as cv_module
        from agent.claude_vault import ask_claude_with_cost, init_vault_session

        cv_module._session = None
        init_vault_session(vault_path=patched_claude_vault)

        response, cost = ask_claude_with_cost("Test question")

        assert response == "Mock response from Claude"
        assert cost.startswith("$")
