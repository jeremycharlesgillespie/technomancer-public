"""
Tests for agent/conversation_context.py - Conversation context persistence.
"""

import json
from datetime import datetime

import pytest

from agent.conversation_context import (
    build_summary_prompt,
    buffer_for_summary,
    fallback_summaries,
    get_conversation_context_tools,
    get_recent_summaries,
    parse_summary_response,
    save_summaries,
    search_summaries,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def patched_conversation_context(tmp_path, monkeypatch):
    """Patch conversation_context.py to use temp vault."""
    import agent.conversation_context as cc_module

    summaries_file = tmp_path / "Permanent" / "conversation_summaries.md"
    summaries_file.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cc_module, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(cc_module, "SUMMARIES_FILE", summaries_file)

    # Reset module-level buffer between tests
    monkeypatch.setattr(cc_module, "_summary_buffer", [])
    monkeypatch.setattr(cc_module, "_summary_running", False)

    return tmp_path


@pytest.fixture
def sample_conversations():
    """Sample conversation data for testing."""
    return [
        {
            "user": "gman386",
            "message": "How do I set up Django REST framework with token auth?",
            "response": "You can use Django REST framework's TokenAuthentication...",
        },
        {
            "user": "gman386",
            "message": "Should I use class-based or function-based views?",
            "response": "Class-based views offer more reusability...",
        },
        {
            "user": "gman386",
            "message": "What's the best way to handle pagination?",
            "response": "DRF provides several pagination classes...",
        },
    ]


@pytest.fixture
def summaries_file_with_content(patched_conversation_context):
    """Create a summaries file with pre-populated content."""
    import agent.conversation_context as cc_module

    content = (
        "# Conversation Summaries\n\n"
        "Structured summaries of past conversations for long-term context.\n\n---\n"
        "\n## 2026-04-01 10:30 - gman386\n"
        "**Topic:** Discussed Django REST framework setup with token authentication\n"
        "**Decisions:** Will use TokenAuthentication over SessionAuthentication\n"
        "**Preferences:** Prefers class-based views for API endpoints\n"
        "\n## 2026-04-02 14:00 - gman386\n"
        "**Topic:** Explored Python asyncio patterns for background tasks\n"
        "**Decisions:** None\n"
        "**Preferences:** Prefers async/await over threading for I/O-bound tasks\n"
        "\n## 2026-04-03 09:15 - gman386\n"
        "**Topic:** Asked about PostgreSQL indexing strategies\n"
        "**Decisions:** Will add composite index on orders table\n"
        "**Preferences:** None\n"
    )
    cc_module.SUMMARIES_FILE.write_text(content, encoding="utf-8")
    return patched_conversation_context


# =============================================================================
# PROMPT BUILDING
# =============================================================================


class TestBuildSummaryPrompt:
    """Tests for build_summary_prompt."""

    def test_includes_conversations(self, sample_conversations):
        """Prompt includes conversation content."""
        prompt = build_summary_prompt(sample_conversations)

        assert "Django REST framework" in prompt
        assert "class-based or function-based" in prompt
        assert "gman386" in prompt

    def test_includes_instructions(self, sample_conversations):
        """Prompt includes summary instructions."""
        prompt = build_summary_prompt(sample_conversations)

        assert "Topic" in prompt
        assert "Key Decisions" in prompt
        assert "User Preferences" in prompt
        assert "JSON" in prompt

    def test_truncates_long_responses(self):
        """Long bot responses are truncated in the prompt."""
        convos = [
            {
                "user": "test",
                "message": "Short question",
                "response": "x" * 1000,
            }
        ]
        prompt = build_summary_prompt(convos)

        # Response should be truncated to 500 chars
        assert len(prompt) < 1500

    def test_empty_conversations(self):
        """Handles empty conversation list."""
        prompt = build_summary_prompt([])

        assert "Topic" in prompt  # Instructions still present


# =============================================================================
# RESPONSE PARSING
# =============================================================================


class TestParseSummaryResponse:
    """Tests for parse_summary_response."""

    def test_parses_json_in_code_block(self):
        """Parses JSON wrapped in markdown code block."""
        response = (
            'Some text\n```json\n[{"user": "test", "topic": "Python basics", '
            '"decisions": "None", "preferences": "Likes concise code"}]\n```'
        )
        result = parse_summary_response(response)

        assert len(result) == 1
        assert result[0]["topic"] == "Python basics"
        assert result[0]["preferences"] == "Likes concise code"

    def test_parses_bare_json(self):
        """Parses bare JSON array without code block."""
        response = (
            '[{"user": "test", "topic": "Django setup", '
            '"decisions": "Use DRF", "preferences": "None"}]'
        )
        result = parse_summary_response(response)

        assert len(result) == 1
        assert result[0]["topic"] == "Django setup"

    def test_handles_missing_fields(self):
        """Entries without topic field are skipped."""
        response = '[{"user": "test", "decisions": "None"}]'
        result = parse_summary_response(response)

        assert len(result) == 0

    def test_defaults_missing_optional_fields(self):
        """Missing optional fields get defaults."""
        response = '[{"topic": "Test topic"}]'
        result = parse_summary_response(response)

        assert len(result) == 1
        assert result[0]["user"] == "unknown"
        assert result[0]["decisions"] == "None"
        assert result[0]["preferences"] == "None"

    def test_handles_invalid_json(self):
        """Returns empty list for invalid JSON."""
        result = parse_summary_response("This is not JSON at all")

        assert result == []

    def test_handles_non_array_json(self):
        """Returns empty list for non-array JSON."""
        result = parse_summary_response('{"topic": "test"}')

        assert result == []

    def test_handles_empty_response(self):
        """Returns empty list for empty string."""
        result = parse_summary_response("")

        assert result == []

    def test_multiple_summaries(self):
        """Parses multiple summaries correctly."""
        response = (
            "```json\n"
            "[\n"
            '  {"user": "alice", "topic": "Topic A", "decisions": "Decision A", "preferences": "None"},\n'
            '  {"user": "bob", "topic": "Topic B", "decisions": "None", "preferences": "Pref B"}\n'
            "]\n"
            "```"
        )
        result = parse_summary_response(response)

        assert len(result) == 2
        assert result[0]["user"] == "alice"
        assert result[1]["user"] == "bob"


# =============================================================================
# SAVING SUMMARIES
# =============================================================================


class TestSaveSummaries:
    """Tests for save_summaries."""

    def test_saves_to_file(self, patched_conversation_context):
        """Summaries are written to the vault file."""
        import agent.conversation_context as cc_module

        summaries = [
            {
                "user": "gman386",
                "topic": "Discussed Python decorators",
                "decisions": "Will use functools.wraps",
                "preferences": "Prefers decorator factories",
            }
        ]

        count = save_summaries(summaries)

        assert count == 1
        content = cc_module.SUMMARIES_FILE.read_text(encoding="utf-8")
        assert "Python decorators" in content
        assert "functools.wraps" in content
        assert "gman386" in content

    def test_creates_file_if_missing(self, patched_conversation_context):
        """Creates summaries file with header if it doesn't exist."""
        import agent.conversation_context as cc_module

        summaries = [{"user": "test", "topic": "Test", "decisions": "None", "preferences": "None"}]
        save_summaries(summaries)

        content = cc_module.SUMMARIES_FILE.read_text(encoding="utf-8")
        assert "# Conversation Summaries" in content

    def test_appends_to_existing(self, patched_conversation_context):
        """Appends new summaries to existing file."""
        summaries1 = [{"user": "a", "topic": "First topic", "decisions": "None", "preferences": "None"}]
        summaries2 = [{"user": "b", "topic": "Second topic", "decisions": "None", "preferences": "None"}]

        save_summaries(summaries1)
        save_summaries(summaries2)

        import agent.conversation_context as cc_module

        content = cc_module.SUMMARIES_FILE.read_text(encoding="utf-8")
        assert "First topic" in content
        assert "Second topic" in content

    def test_returns_zero_for_empty(self, patched_conversation_context):
        """Returns 0 when no summaries to save."""
        assert save_summaries([]) == 0

    def test_saves_multiple(self, patched_conversation_context):
        """Saves multiple summaries at once."""
        summaries = [
            {"user": "a", "topic": "Topic A", "decisions": "Dec A", "preferences": "None"},
            {"user": "b", "topic": "Topic B", "decisions": "None", "preferences": "Pref B"},
        ]

        count = save_summaries(summaries)
        assert count == 2


# =============================================================================
# READING SUMMARIES
# =============================================================================


class TestGetRecentSummaries:
    """Tests for get_recent_summaries."""

    def test_returns_message_when_no_file(self, patched_conversation_context):
        """Returns helpful message when no summaries file exists."""
        result = get_recent_summaries()

        assert "No conversation summaries" in result

    def test_reads_existing_summaries(self, summaries_file_with_content):
        """Reads and formats existing summaries."""
        result = get_recent_summaries()

        assert "Django REST framework" in result
        assert "asyncio" in result
        assert "PostgreSQL" in result

    def test_limits_count(self, summaries_file_with_content):
        """Respects count parameter."""
        result = get_recent_summaries(count=1)

        # Should only have the most recent entry
        assert "PostgreSQL" in result
        assert "1 entries" in result

    def test_returns_message_for_empty_file(self, patched_conversation_context):
        """Returns message for file with no entries."""
        import agent.conversation_context as cc_module

        cc_module.SUMMARIES_FILE.write_text("# Empty file\n", encoding="utf-8")
        result = get_recent_summaries()

        assert "No conversation summaries" in result


# =============================================================================
# SEARCHING SUMMARIES
# =============================================================================


class TestSearchSummaries:
    """Tests for search_summaries."""

    def test_finds_matching_topic(self, summaries_file_with_content):
        """Finds summaries matching topic keyword."""
        result = search_summaries("Django")

        assert "Found" in result
        assert "Django" in result

    def test_finds_matching_decision(self, summaries_file_with_content):
        """Finds summaries matching decision keyword."""
        result = search_summaries("TokenAuthentication")

        assert "Found" in result
        assert "TokenAuthentication" in result

    def test_finds_matching_preference(self, summaries_file_with_content):
        """Finds summaries matching preference keyword."""
        result = search_summaries("async/await")

        assert "Found" in result

    def test_no_matches(self, summaries_file_with_content):
        """Returns message when no matches found."""
        result = search_summaries("nonexistent_xyz_12345")

        assert "No conversation summaries matching" in result

    def test_no_file(self, patched_conversation_context):
        """Returns message when summaries file doesn't exist."""
        result = search_summaries("anything")

        assert "No conversation summaries to search" in result

    def test_case_insensitive(self, summaries_file_with_content):
        """Search is case-insensitive."""
        result = search_summaries("django")

        assert "Found" in result


# =============================================================================
# FALLBACK SUMMARIES
# =============================================================================


class TestFallbackSummaries:
    """Tests for fallback_summaries."""

    def test_generates_from_message(self):
        """Generates topic from first words of message."""
        convos = [
            {
                "user": "test",
                "message": "How do I configure Django settings for production",
                "response": "You should...",
            }
        ]
        result = fallback_summaries(convos)

        assert len(result) == 1
        assert result[0]["user"] == "test"
        assert "Django" in result[0]["topic"] or "configure" in result[0]["topic"]
        assert result[0]["decisions"] == "None"
        assert result[0]["preferences"] == "None"

    def test_truncates_long_topic(self):
        """Topic is truncated for long messages."""
        convos = [
            {
                "user": "test",
                "message": "This is a very long message with many words that should be truncated",
                "response": "OK",
            }
        ]
        result = fallback_summaries(convos)

        assert result[0]["topic"].endswith("...")

    def test_multiple_conversations(self):
        """Generates a summary for each conversation."""
        convos = [
            {"user": "a", "message": "Hello there", "response": "Hi"},
            {"user": "b", "message": "Goodbye now", "response": "Bye"},
        ]
        result = fallback_summaries(convos)

        assert len(result) == 2


# =============================================================================
# BUFFER
# =============================================================================


class TestBufferForSummary:
    """Tests for buffer_for_summary."""

    def test_adds_to_buffer(self, patched_conversation_context):
        """Adds conversation to the buffer."""
        import agent.conversation_context as cc_module

        buffer_for_summary("user1", "A sufficiently long message about something interesting", "A response that is also long enough to pass the threshold")

        assert len(cc_module._summary_buffer) == 1
        assert cc_module._summary_buffer[0]["user"] == "user1"

    def test_skips_short_conversations(self, patched_conversation_context):
        """Skips conversations shorter than MIN_CONVERSATION_LENGTH."""
        import agent.conversation_context as cc_module

        buffer_for_summary("user1", "Hi", "Hey")

        assert len(cc_module._summary_buffer) == 0


# =============================================================================
# TOOLS
# =============================================================================


class TestConversationContextTools:
    """Tests for tool registration."""

    def test_returns_tools(self):
        """get_conversation_context_tools returns list of Tool objects."""
        tools = get_conversation_context_tools()

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "get_conversation_summaries" in tool_names
        assert "search_conversation_summaries" in tool_names

    def test_get_summaries_tool_callable(self, summaries_file_with_content):
        """get_conversation_summaries tool function is callable."""
        tools = get_conversation_context_tools()
        get_tool = next(t for t in tools if t.name == "get_conversation_summaries")

        result = get_tool.function(count=5)
        assert "Django" in result

    def test_search_tool_callable(self, summaries_file_with_content):
        """search_conversation_summaries tool function is callable."""
        tools = get_conversation_context_tools()
        search_tool = next(t for t in tools if t.name == "search_conversation_summaries")

        result = search_tool.function("PostgreSQL")
        assert "Found" in result
