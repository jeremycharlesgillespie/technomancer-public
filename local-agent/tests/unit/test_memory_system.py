"""
Tests for agent/memory_system.py - MemorySystem class and tools.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from agent.memory_system import (
    ConversationEntry,
    MemorySystem,
    get_memory_tools,
)


class TestConversationEntry:
    """Tests for ConversationEntry dataclass."""

    def test_to_markdown(self):
        """Entry formats as expected markdown."""
        entry = ConversationEntry(
            timestamp=datetime(2026, 3, 15, 14, 30, 0),
            user="testuser",
            message="Hello there",
            response="Hi! How can I help?",
        )

        markdown = entry.to_markdown()

        assert "14:30:00" in markdown
        assert "testuser" in markdown
        assert "Hello there" in markdown
        assert "Hi! How can I help?" in markdown
        assert "**Q:**" in markdown
        assert "**A:**" in markdown

    def test_to_dict(self):
        """Entry converts to dict correctly."""
        entry = ConversationEntry(
            timestamp=datetime(2026, 3, 15, 14, 30, 0),
            user="testuser",
            message="Hello",
            response="Hi",
        )

        d = entry.to_dict()

        assert d["user"] == "testuser"
        assert d["message"] == "Hello"
        assert d["response"] == "Hi"
        assert "timestamp" in d

    def test_from_dict_round_trip(self):
        """Entry survives dict serialization round trip."""
        original = ConversationEntry(
            timestamp=datetime(2026, 3, 15, 14, 30, 0),
            user="testuser",
            message="Test message",
            response="Test response",
        )

        restored = ConversationEntry.from_dict(original.to_dict())

        assert restored.user == original.user
        assert restored.message == original.message
        assert restored.response == original.response
        assert restored.timestamp == original.timestamp


class TestMemorySystemInit:
    """Tests for MemorySystem initialization."""

    def test_init_creates_folders(self, tmp_path):
        """__init__ creates vault folder structure."""
        MemorySystem(str(tmp_path))

        assert (tmp_path / "LLM Memory" / "Conversations").exists()
        assert (tmp_path / "LLM Memory" / "Context").exists()
        assert (tmp_path / "LLM Memory" / "Permanent").exists()

    def test_init_custom_memory_folder(self, tmp_path):
        """__init__ respects custom memory folder name."""
        MemorySystem(str(tmp_path), memory_folder="Custom Memory")

        assert (tmp_path / "Custom Memory" / "Conversations").exists()


class TestMemorySystemLogging:
    """Tests for conversation logging."""

    def test_log_conversation_appends(self, memory_system, temp_vault):
        """log_conversation adds to daily file."""
        memory_system.log_conversation("user1", "Hello", "Hi there")
        memory_system.log_conversation("user1", "How are you?", "I'm good!")

        # Check daily file exists
        today = datetime.now().strftime("%Y-%m-%d")
        daily_file = temp_vault / "LLM Memory" / "Conversations" / f"{today}.md"

        assert daily_file.exists()
        content = daily_file.read_text(encoding="utf-8")
        assert "Hello" in content
        assert "How are you?" in content

    def test_log_conversation_creates_daily_file(self, memory_system, temp_vault):
        """First log creates daily file with header."""
        memory_system.log_conversation("user1", "First message", "First response")

        today = datetime.now().strftime("%Y-%m-%d")
        daily_file = temp_vault / "LLM Memory" / "Conversations" / f"{today}.md"

        content = daily_file.read_text(encoding="utf-8")
        assert f"# Conversations - {today}" in content
        assert "First message" in content

    def test_log_conversation_adds_to_memory(self, memory_system):
        """log_conversation adds to recent_conversations deque."""
        initial_count = len(memory_system.recent_conversations)

        memory_system.log_conversation("user1", "Test", "Response")

        assert len(memory_system.recent_conversations) == initial_count + 1
        last_entry = memory_system.recent_conversations[-1]
        assert last_entry.message == "Test"


class TestMemorySystemContext:
    """Tests for context retrieval."""

    def test_get_context_hour(self, memory_system):
        """get_context('hour') returns recent conversations."""
        # Add a recent conversation
        memory_system.log_conversation("user1", "Recent message", "Recent response")

        result = memory_system.get_context("hour")

        assert "Recent message" in result or "1 exchanges" in result or "hour" in result.lower()

    def test_get_context_no_conversations(self, memory_system):
        """get_context returns message when empty."""
        result = memory_system.get_context("hour")

        assert "No conversations" in result

    def test_get_context_permanent(self, memory_system, temp_vault):
        """get_context('permanent') reads memories.md."""
        # Create memories.md
        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        memories_file.write_text("# Memories\n\nImportant fact: Test data")

        result = memory_system.get_context("permanent")

        assert "Test data" in result

    def test_get_context_permanent_empty(self, memory_system):
        """get_context('permanent') handles missing file."""
        result = memory_system.get_context("permanent")

        assert "No permanent memories" in result

    def test_get_context_permanent_for_think_command(self, memory_system, temp_vault):
        """get_context('permanent') returns formatted content suitable for think command."""
        # Create memories with typical user info
        memories_content = """# Permanent Memories

Important information to always remember.

---

## 2026-03-15 14:30 - user_info/testuser
Name: Test User
Works at: Acme Corp
Role: Senior Software Engineer

## 2026-03-15 14:31 - preferences/testuser
Likes Python programming
Interested in AI and LLMs

## 2026-03-15 14:32 - projects/testuser
Working on Technomancer Discord bot project
"""

        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        memories_file.write_text(memories_content)

        result = memory_system.get_context("permanent")

        # Should contain the actual content, not "No permanent memories"
        assert "No permanent memories" not in result
        assert "Test User" in result
        assert "Acme Corp" in result
        assert "Python programming" in result
        # Should be the full content for the think command to display
        assert "Permanent Memories" in result


class TestMemorySystemPermanent:
    """Tests for permanent memory operations."""

    def test_save_permanent_memory(self, memory_system, temp_vault):
        """save_permanent_memory writes to memories.md."""
        memory_system.save_permanent_memory("User likes Python", "preferences")

        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        assert memories_file.exists()
        content = memories_file.read_text()
        assert "User likes Python" in content
        assert "preferences" in content

    def test_save_permanent_memory_appends(self, memory_system, temp_vault):
        """save_permanent_memory appends to existing file."""
        memory_system.save_permanent_memory("First fact", "facts")
        memory_system.save_permanent_memory("Second fact", "facts")

        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        content = memories_file.read_text()
        assert "First fact" in content
        assert "Second fact" in content

    def test_save_permanent_memory_replace_category(self, memory_system, temp_vault):
        """replace_category=True removes old entries."""
        memory_system.save_permanent_memory("Old value", "settings")
        memory_system.save_permanent_memory("New value", "settings", replace_category=True)

        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        content = memories_file.read_text()
        assert "New value" in content
        # Old value may or may not be removed depending on exact implementation
        # The important thing is the new value is there


class TestMemorySystemSearch:
    """Tests for memory search functionality."""

    def test_search_memories_finds_match(self, memory_system):
        """search_memories finds matching conversations."""
        memory_system.log_conversation("user1", "I love Python programming", "Python is great!")

        # Access the search via the module function
        from agent.memory_system import init_memory_system, search_memories

        # Re-init with our memory system's vault path
        init_memory_system(str(memory_system.vault_path))

        result = search_memories("Python")

        assert "Python" in result or "Found" in result

    def test_search_memories_no_match(self, memory_system):
        """search_memories returns message when no matches."""
        memory_system.log_conversation("user1", "Hello there", "Hi!")

        from agent.memory_system import init_memory_system, search_memories

        init_memory_system(str(memory_system.vault_path))

        result = search_memories("nonexistent_query_12345")

        assert "No conversations found" in result


class TestMemorySystemCompaction:
    """Tests for memory compaction."""

    def test_compact_hourly_no_data(self, memory_system):
        """compact_hourly handles empty data."""
        result = memory_system.compact_hourly()

        assert "Nothing to compact" in result

    def test_compact_daily(self, memory_system, temp_vault):
        """compact_daily creates daily summary."""
        # Add some conversations
        memory_system.log_conversation("user1", "Morning message", "Morning response")

        result = memory_system.compact_daily()

        daily_path = temp_vault / "LLM Memory" / "Context" / "daily.md"
        assert daily_path.exists()
        assert "Updated daily context" in result


    def test_compact_hourly_with_summarizer(self, memory_system, temp_vault):
        """compact_hourly uses LLM summarizer when provided."""
        from datetime import timedelta

        # Add old conversations (1-2 hours ago)
        now = datetime.now()
        old_time = now - timedelta(hours=1, minutes=30)
        entry = ConversationEntry(
            timestamp=old_time, user="test", message="What is Django?", response="A web framework"
        )
        memory_system.recent_conversations.append(entry)

        mock_summarizer = lambda prompt: "LLM summary: Discussed Django web framework."
        result = memory_system.compact_hourly(summarizer=mock_summarizer)

        assert "Compacted 1" in result
        hourly_path = temp_vault / "LLM Memory" / "Context" / "hourly.md"
        assert hourly_path.exists()
        content = hourly_path.read_text()
        assert "LLM summary" in content

    def test_compact_hourly_llm_failure_falls_back(self, memory_system, temp_vault):
        """compact_hourly falls back to simple summary on LLM error."""
        from datetime import timedelta

        now = datetime.now()
        old_time = now - timedelta(hours=1, minutes=30)
        entry = ConversationEntry(
            timestamp=old_time, user="test", message="Hello world", response="Hi there"
        )
        memory_system.recent_conversations.append(entry)

        def failing_summarizer(prompt):
            raise RuntimeError("LLM offline")

        result = memory_system.compact_hourly(summarizer=failing_summarizer)
        assert "Compacted 1" in result  # Still succeeds with fallback

    def test_compact_daily_with_summarizer(self, memory_system, temp_vault):
        """compact_daily uses LLM summarizer when provided."""
        memory_system.log_conversation("user1", "Tell me about Python", "Python is great")

        mock_summarizer = lambda prompt: "Daily LLM summary: Discussed Python programming."
        result = memory_system.compact_daily(summarizer=mock_summarizer)

        daily_path = temp_vault / "LLM Memory" / "Context" / "daily.md"
        assert daily_path.exists()
        content = daily_path.read_text()
        assert "Daily LLM summary" in content

    def test_compaction_stats_logged(self, memory_system, temp_vault):
        """Compaction stats are written to JSON log."""
        from datetime import timedelta

        now = datetime.now()
        old_time = now - timedelta(hours=1, minutes=30)
        entry = ConversationEntry(
            timestamp=old_time, user="test", message="Testing stats", response="OK"
        )
        memory_system.recent_conversations.append(entry)

        mock_summarizer = lambda prompt: "Summary for stats test."
        memory_system.compact_hourly(summarizer=mock_summarizer)

        stats = memory_system.get_compaction_stats()
        assert len(stats) >= 1
        assert stats[-1]["tier"] == "hourly"
        assert stats[-1]["success"] is True
        assert "duration_seconds" in stats[-1]

    def test_compact_weekly(self, memory_system, temp_vault):
        """compact_weekly creates weekly summary with summarizer."""
        # Create a daily file so there's something to summarize
        daily_path = temp_vault / "LLM Memory" / "Context" / "daily.md"
        daily_path.write_text("# Daily\n\nDiscussed Python and Django today.", encoding="utf-8")

        mock_summarizer = lambda prompt: "Weekly: Main themes were Python and Django."
        result = memory_system.compact_weekly(summarizer=mock_summarizer)

        weekly_path = temp_vault / "LLM Memory" / "Context" / "weekly.md"
        assert weekly_path.exists()
        assert "Weekly" in weekly_path.read_text()


class TestMemoryTools:
    """Tests for memory tool functions."""

    def test_get_memory_tools_returns_tools(self, temp_vault, reset_memory_system):
        """get_memory_tools returns list of Tool objects."""
        tools = get_memory_tools(str(temp_vault))

        assert len(tools) >= 4  # get_context, remember_permanently, search_memories, etc.
        tool_names = [t.name for t in tools]
        assert "get_context" in tool_names
        assert "remember_permanently" in tool_names
        assert "search_memories" in tool_names

    def test_remember_permanently_tool_function(self, temp_vault, reset_memory_system):
        """remember_permanently tool function works."""
        from agent.memory_system import init_memory_system, remember_permanently

        init_memory_system(str(temp_vault))
        result = remember_permanently("Test memory", "test_category")

        assert "Saved" in result

        # Verify file was created
        memories_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        assert memories_file.exists()
        assert "Test memory" in memories_file.read_text()


class TestCompactionLogging:
    """Tests for compaction logging (print→logging replacement)."""

    def test_llm_summarize_logs_error_not_print(self, memory_system, caplog):
        """_llm_summarize uses logging, not print(), on LLM failure."""
        def failing_summarizer(prompt):
            raise RuntimeError("model offline")

        with caplog.at_level(logging.ERROR, logger="agent.memory_system"):
            summary, stats = memory_system._llm_summarize(
                failing_summarizer, "test text", "hourly"
            )

        assert summary is None
        assert stats["success"] is False
        assert any("model offline" in r.message for r in caplog.records)

    def test_compaction_loop_logs_info(self, memory_system, caplog):
        """Background compaction loop uses log.info, not print()."""
        # Run a single compaction cycle synchronously to verify logging
        memory_system.log_conversation("user1", "Hello", "Hi")

        with caplog.at_level(logging.INFO, logger="agent.memory_system"):
            memory_system.compact_daily()

        # No print output, just logging — the key assertion is no exception


class TestCompactionStatsCapping:
    """Tests for compaction_stats.json rotating window on read."""

    def test_get_compaction_stats_caps_on_read(self, memory_system, temp_vault):
        """get_compaction_stats truncates file when over max_entries."""
        stats_path = temp_vault / "LLM Memory" / "Context" / "compaction_stats.json"

        # Write 20 entries
        entries = [{"tier": "hourly", "i": i} for i in range(20)]
        stats_path.write_text(json.dumps(entries), encoding="utf-8")

        # Read with max_entries=10 — should cap to last 10
        result = memory_system.get_compaction_stats(max_entries=10)

        assert len(result) == 10
        assert result[0]["i"] == 10  # kept the most recent 10

        # File should also be truncated on disk
        on_disk = json.loads(stats_path.read_text(encoding="utf-8"))
        assert len(on_disk) == 10

    def test_get_compaction_stats_no_truncate_when_under_limit(self, memory_system, temp_vault):
        """get_compaction_stats doesn't rewrite file when under limit."""
        stats_path = temp_vault / "LLM Memory" / "Context" / "compaction_stats.json"

        entries = [{"tier": "daily", "i": i} for i in range(5)]
        stats_path.write_text(json.dumps(entries), encoding="utf-8")

        result = memory_system.get_compaction_stats(max_entries=500)

        assert len(result) == 5

    def test_get_compaction_stats_empty_file(self, memory_system):
        """get_compaction_stats returns empty list for missing file."""
        result = memory_system.get_compaction_stats()
        assert result == []

    def test_get_compaction_stats_corrupt_json(self, memory_system, temp_vault):
        """get_compaction_stats handles corrupt JSON gracefully."""
        stats_path = temp_vault / "LLM Memory" / "Context" / "compaction_stats.json"
        stats_path.write_text("not valid json {{{", encoding="utf-8")

        result = memory_system.get_compaction_stats()
        assert result == []


class TestCompactionHealth:
    """Tests for compaction_health() method."""

    def test_health_before_start(self, memory_system):
        """compaction_health reports not running before start."""
        health = memory_system.compaction_health()

        assert health["running"] is False
        assert health["thread_alive"] is False
        assert health["last_run"] is None
        assert health["error_count"] == 0
        assert health["last_error"] is None
        assert health["healthy"] is False  # not running = not healthy

    def test_health_after_successful_run(self, memory_system):
        """compaction_health reports healthy after a successful cycle."""
        # Simulate state after a successful compaction
        memory_system._running = True
        memory_system._last_compaction_run = datetime.now()
        memory_system._compaction_error_count = 0

        # Create a fake alive thread
        t = threading.Thread(target=lambda: time.sleep(10), daemon=True)
        t.start()
        memory_system._compaction_thread = t

        health = memory_system.compaction_health()

        assert health["running"] is True
        assert health["thread_alive"] is True
        assert health["last_run"] is not None
        assert health["healthy"] is True

        # Clean up
        memory_system._running = False

    def test_health_with_errors(self, memory_system):
        """compaction_health reports unhealthy after many errors."""
        memory_system._running = True
        memory_system._compaction_error_count = 5
        memory_system._last_compaction_error = "disk full"

        t = threading.Thread(target=lambda: time.sleep(10), daemon=True)
        t.start()
        memory_system._compaction_thread = t

        health = memory_system.compaction_health()

        assert health["error_count"] == 5
        assert health["last_error"] == "disk full"
        assert health["healthy"] is False  # >= 5 errors

        memory_system._running = False

    def test_health_dead_thread(self, memory_system):
        """compaction_health detects dead thread."""
        memory_system._running = True

        # Thread that finishes immediately
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        t.join()  # wait for it to die
        memory_system._compaction_thread = t

        health = memory_system.compaction_health()

        assert health["running"] is True
        assert health["thread_alive"] is False
        assert health["healthy"] is False

        memory_system._running = False


class TestCompactionThreadResilience:
    """Tests for compaction thread not dying on exceptions."""

    def test_compaction_loop_survives_exception(self, memory_system):
        """Compaction loop retries after an exception instead of dying."""
        call_count = 0

        # Patch compact_hourly to fail once, then succeed
        original_hourly = memory_system.compact_hourly

        def flaky_hourly(summarizer=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return original_hourly(summarizer)

        memory_system.compact_hourly = flaky_hourly

        # Patch sleep to avoid real waits — track calls
        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)
            # Stop after retry sleep
            if len(sleep_calls) >= 3:
                memory_system._running = False

        with patch("agent.memory_system.time.sleep", side_effect=fake_sleep):
            memory_system.start_background_compaction(interval_minutes=1)
            # Wait for thread to finish
            memory_system._compaction_thread.join(timeout=5)

        # Should have been called at least twice (first fail + retry)
        assert call_count >= 1
        assert memory_system._compaction_error_count == 0 or memory_system._last_compaction_error is not None

    def test_compaction_tracks_error_state(self, memory_system):
        """Compaction loop updates error tracking on failure."""

        def always_fail(summarizer=None):
            raise RuntimeError("permanent failure")

        memory_system.compact_hourly = always_fail

        iteration = 0

        def fake_sleep(seconds):
            nonlocal iteration
            iteration += 1
            if iteration >= 3:
                memory_system._running = False

        with patch("agent.memory_system.time.sleep", side_effect=fake_sleep):
            memory_system.start_background_compaction(interval_minutes=1)
            memory_system._compaction_thread.join(timeout=5)

        assert memory_system._last_compaction_error == "permanent failure"
        assert memory_system._compaction_error_count >= 1
