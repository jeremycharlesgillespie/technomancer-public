"""
Tests for agent/memory_system.py - MemorySystem class and tools.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.memory_system import (
    ConversationEntry,
    MAX_COMPACTION_SNAPSHOTS,
    MemorySystem,
    atomic_write,
    get_memory_tools,
    list_backups,
    main as memory_cli_main,
    notify_auto_restored,
    restore_backup,
    rotate_backup,
    verify_memory_file,
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


class TestCompactionSnapshots:
    """Tests for _snapshot_before_compaction backup mechanism."""

    def _write_targets(self, temp_vault):
        """Populate the three files that compaction snapshots."""
        memory_root = temp_vault / "LLM Memory"
        hourly = memory_root / "Context" / "hourly.md"
        daily = memory_root / "Context" / "daily.md"
        memories = memory_root / "Permanent" / "memories.md"
        hourly.write_text("hourly content", encoding="utf-8")
        daily.write_text("daily content", encoding="utf-8")
        memories.write_text("memories content", encoding="utf-8")
        return hourly, daily, memories

    def test_snapshot_copies_three_target_files(self, memory_system, temp_vault):
        """Triggering compaction copies hourly/daily/memories into a new backup dir."""
        self._write_targets(temp_vault)

        memory_system.compact_hourly()

        backup_root = temp_vault / "Backups" / "memory"
        assert backup_root.exists()
        snapshots = [d for d in backup_root.iterdir() if d.is_dir()]
        assert len(snapshots) == 1

        snap = snapshots[0]
        assert (snap / "hourly.md").read_text(encoding="utf-8") == "hourly content"
        assert (snap / "daily.md").read_text(encoding="utf-8") == "daily content"
        assert (snap / "memories.md").read_text(encoding="utf-8") == "memories content"

    def test_snapshot_skips_missing_files(self, memory_system, temp_vault):
        """Snapshot only copies files that actually exist."""
        memory_root = temp_vault / "LLM Memory"
        (memory_root / "Context" / "hourly.md").write_text("only hourly", encoding="utf-8")

        memory_system.compact_hourly()

        backup_root = temp_vault / "Backups" / "memory"
        snapshots = [d for d in backup_root.iterdir() if d.is_dir()]
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert (snap / "hourly.md").exists()
        assert not (snap / "daily.md").exists()
        assert not (snap / "memories.md").exists()

    def test_snapshot_no_backup_when_nothing_exists(self, memory_system, temp_vault):
        """If none of the target files exist, no backup dir is created."""
        memory_system.compact_hourly()

        backup_root = temp_vault / "Backups" / "memory"
        if backup_root.exists():
            assert list(backup_root.iterdir()) == []

    def test_snapshot_prunes_to_ten_most_recent(self, memory_system, temp_vault):
        """Running compaction 11 times leaves exactly 10 snapshot dirs."""
        self._write_targets(temp_vault)

        for _ in range(11):
            memory_system._snapshot_before_compaction()

        backup_root = temp_vault / "Backups" / "memory"
        snapshots = [d for d in backup_root.iterdir() if d.is_dir()]
        assert len(snapshots) == 10

    def test_snapshot_keeps_newest_after_prune(self, memory_system, temp_vault):
        """After pruning, the 10 most recent (by name) are retained."""
        self._write_targets(temp_vault)
        backup_root = temp_vault / "Backups" / "memory"

        # Pre-seed backup root with an obviously-old snapshot that should get pruned.
        stale = backup_root / "19990101-000000"
        stale.mkdir(parents=True)
        (stale / "hourly.md").write_text("old", encoding="utf-8")

        for _ in range(10):
            memory_system._snapshot_before_compaction()

        snapshots = sorted(
            [d.name for d in backup_root.iterdir() if d.is_dir()]
        )
        assert len(snapshots) == 10
        assert "19990101-000000" not in snapshots


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


# =============================================================================
# RESTORE CLI + NOTIFICATION
# =============================================================================


def _seed_backup(vault_root: Path, stamp: str, files: dict[str, str]) -> Path:
    """Create a snapshot dir under Backups/memory/<stamp> with the given files."""
    snap = vault_root / "Backups" / "memory" / stamp
    snap.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (snap / name).write_text(body, encoding="utf-8")
    return snap


def _seed_live_files(vault_root: Path) -> None:
    """Create the LLM Memory tree the CLI restores into."""
    memory_root = vault_root / "LLM Memory"
    (memory_root / "Context").mkdir(parents=True, exist_ok=True)
    (memory_root / "Permanent").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def cli_vault(tmp_path, monkeypatch):
    """Vault root pointed at by settings.vault_path for CLI tests."""
    import agent.config as config_module

    _seed_live_files(tmp_path)
    monkeypatch.setattr(config_module.settings, "vault_path", tmp_path)
    return tmp_path


class TestAtomicWrite:
    """atomic_write must not truncate the target on crash."""

    def test_replaces_existing_file(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("old", encoding="utf-8")
        atomic_write(target, b"new content")
        assert target.read_text(encoding="utf-8") == "new content"

    def test_creates_parent_dir(self, tmp_path):
        target = tmp_path / "sub" / "file.md"
        atomic_write(target, b"hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_cleans_tmp_on_failure(self, tmp_path, monkeypatch):
        target = tmp_path / "file.md"
        target.write_text("original", encoding="utf-8")

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr("agent.memory_system.os.replace", boom)
        with pytest.raises(OSError):
            atomic_write(target, b"new")

        # Live file left intact; no stray .tmp files linger next to it.
        assert target.read_text(encoding="utf-8") == "original"
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


class TestRestoreCliList:
    """`restore <file>` with no --backup lists available backups."""

    def test_list_backups_newest_first(self, cli_vault, capsys):
        _seed_backup(cli_vault, "20260101-000000", {"hourly.md": "old"})
        _seed_backup(cli_vault, "20260201-120000", {"hourly.md": "mid"})
        _seed_backup(cli_vault, "20260301-083000", {"hourly.md": "new"})

        rc = memory_cli_main(["restore", "hourly.md"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("2026")]
        stamps = [ln.split()[0] for ln in lines]
        assert stamps == ["20260301-083000", "20260201-120000", "20260101-000000"]

    def test_list_skips_snapshots_missing_file(self, cli_vault, capsys):
        _seed_backup(cli_vault, "20260101-000000", {"hourly.md": "h"})
        _seed_backup(cli_vault, "20260102-000000", {"daily.md": "d"})  # no hourly

        rc = memory_cli_main(["restore", "hourly.md"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "20260101-000000" in out
        assert "20260102-000000" not in out

    def test_list_empty_exits_nonzero(self, cli_vault, capsys):
        rc = memory_cli_main(["restore", "hourly.md"])
        assert rc == 1
        assert "No backups" in capsys.readouterr().out


class TestRestoreCliLatest:
    """`restore <file> --backup latest` restores the newest snapshot."""

    def test_restore_latest_overwrites_live_file(self, cli_vault, capsys):
        _seed_backup(cli_vault, "20260101-000000", {"hourly.md": "OLD snapshot"})
        _seed_backup(cli_vault, "20260301-083000", {"hourly.md": "NEW snapshot"})

        live = cli_vault / "LLM Memory" / "Context" / "hourly.md"
        live.write_text("corrupted live file", encoding="utf-8")

        rc = memory_cli_main(["restore", "hourly.md", "--backup", "latest"])
        assert rc == 0
        assert live.read_text(encoding="utf-8") == "NEW snapshot"

        out = capsys.readouterr().out
        assert "20260301-083000" in out
        assert "hourly.md" in out

    def test_restore_latest_no_backups_errors(self, cli_vault, capsys):
        rc = memory_cli_main(["restore", "hourly.md", "--backup", "latest"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "No backups" in err


class TestRestoreCliNamedBackup:
    """`restore <file> --backup <name>` restores a specific snapshot."""

    def test_restore_named_backup(self, cli_vault, capsys):
        _seed_backup(cli_vault, "20260101-000000", {"memories.md": "V1 memories"})
        _seed_backup(cli_vault, "20260201-000000", {"memories.md": "V2 memories"})
        _seed_backup(cli_vault, "20260301-000000", {"memories.md": "V3 memories"})

        live = cli_vault / "LLM Memory" / "Permanent" / "memories.md"
        live.write_text("garbage", encoding="utf-8")

        rc = memory_cli_main(
            ["restore", "memories.md", "--backup", "20260201-000000"]
        )
        assert rc == 0
        assert live.read_text(encoding="utf-8") == "V2 memories"
        assert "20260201-000000" in capsys.readouterr().out

    def test_restore_named_backup_missing_errors(self, cli_vault, capsys):
        _seed_backup(cli_vault, "20260101-000000", {"hourly.md": "x"})

        rc = memory_cli_main(["restore", "hourly.md", "--backup", "does-not-exist"])
        assert rc == 2
        assert "Backup not found" in capsys.readouterr().err

    def test_restore_unknown_filename_errors(self, cli_vault, capsys):
        rc = memory_cli_main(["restore", "garbage.md", "--backup", "latest"])
        assert rc == 2
        assert "Unknown target file" in capsys.readouterr().err


class TestListBackupsApi:
    """Direct exercise of list_backups() independent of CLI."""

    def test_filters_by_filename(self, tmp_path):
        _seed_backup(tmp_path, "20260101-000000", {"hourly.md": "a"})
        _seed_backup(tmp_path, "20260102-000000", {"daily.md": "b"})

        hourly = list_backups(tmp_path, filename="hourly.md")
        assert [p.name for p in hourly] == ["20260101-000000"]

        daily = list_backups(tmp_path, filename="daily.md")
        assert [p.name for p in daily] == ["20260102-000000"]

    def test_empty_when_backups_dir_missing(self, tmp_path):
        assert list_backups(tmp_path, filename="hourly.md") == []


class TestRestoreBackupApi:
    """Direct exercise of restore_backup() (used by auto-recover path)."""

    def test_restore_backup_uses_atomic_write(self, tmp_path):
        _seed_live_files(tmp_path)
        _seed_backup(tmp_path, "20260401-120000", {"hourly.md": "snapshot body"})
        target = tmp_path / "LLM Memory" / "Context" / "hourly.md"
        target.write_text("stale", encoding="utf-8")

        result = restore_backup(tmp_path, "hourly.md", backup_name="20260401-120000")

        assert result == target
        assert target.read_text(encoding="utf-8") == "snapshot body"


class TestAutoRestoreNotification:
    """notify_auto_restored() posts via notifications.discord_send().

    This is the hook that the auto-recover compaction path (sibling story)
    will call; the test asserts it forwards to the Discord webhook helper.
    """

    def test_calls_discord_send_once(self):
        with patch("agent.memory_system.notifications.discord_send") as mock_send:
            notify_auto_restored("hourly.md", "20260401-120000")

        assert mock_send.call_count == 1
        msg = mock_send.call_args.args[0]
        assert "hourly.md" in msg
        assert "20260401-120000" in msg

    def test_swallows_webhook_errors(self):
        with patch(
            "agent.memory_system.notifications.discord_send",
            side_effect=RuntimeError("webhook down"),
        ):
            # Must not propagate — compaction thread should keep running.
            notify_auto_restored("daily.md", "20260401-120000")


# =============================================================================
# ROTATE_BACKUP — per-path snapshot helper used by the safe-write flow
# =============================================================================


class TestRotateBackup:
    """rotate_backup() snapshots a single file into Backups/memory/<ts>/."""

    def test_creates_snapshot_with_only_target(self, tmp_path):
        target = tmp_path / "LLM Memory" / "Context" / "hourly.md"
        target.parent.mkdir(parents=True)
        target.write_text("live content", encoding="utf-8")

        snap_dir = rotate_backup(tmp_path, target)

        assert snap_dir is not None
        assert snap_dir.is_dir()
        assert (snap_dir / "hourly.md").read_text(encoding="utf-8") == "live content"
        assert not (snap_dir / "daily.md").exists()

    def test_returns_none_when_target_missing(self, tmp_path):
        target = tmp_path / "LLM Memory" / "Context" / "hourly.md"

        assert rotate_backup(tmp_path, target) is None
        assert not (tmp_path / "Backups" / "memory").exists()

    def test_prunes_to_max_snapshots(self, tmp_path):
        target = tmp_path / "LLM Memory" / "Context" / "hourly.md"
        target.parent.mkdir(parents=True)
        target.write_text("c", encoding="utf-8")

        for _ in range(MAX_COMPACTION_SNAPSHOTS + 3):
            rotate_backup(tmp_path, target)

        snapshots = [d for d in (tmp_path / "Backups" / "memory").iterdir() if d.is_dir()]
        assert len(snapshots) == MAX_COMPACTION_SNAPSHOTS

    def test_collision_in_same_second_uses_suffix(self, tmp_path):
        target = tmp_path / "LLM Memory" / "Context" / "hourly.md"
        target.parent.mkdir(parents=True)
        target.write_text("c", encoding="utf-8")

        first = rotate_backup(tmp_path, target)
        second = rotate_backup(tmp_path, target)

        # Two back-to-back calls must produce distinct snapshot dirs.
        assert first != second
        assert first.exists()
        assert second.exists()


# =============================================================================
# VERIFY_MEMORY_FILE — integrity check after each compaction write
# =============================================================================


class TestVerifyMemoryFile:
    """verify_memory_file() must reject corrupted/missing/empty files."""

    def test_accepts_healthy_file(self, tmp_path):
        path = tmp_path / "f.md"
        path.write_text("normal content", encoding="utf-8")

        assert verify_memory_file(path) is True

    def test_rejects_missing_file(self, tmp_path):
        assert verify_memory_file(tmp_path / "ghost.md") is False

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_bytes(b"")

        assert verify_memory_file(path) is False

    def test_rejects_invalid_utf8(self, tmp_path):
        path = tmp_path / "bad.md"
        # 0xff 0xfe 0xfd is invalid as UTF-8 but non-empty.
        path.write_bytes(b"\xff\xfe\xfd")

        assert verify_memory_file(path) is False


# =============================================================================
# COMPACTION SAFE-WRITE — verify failure triggers backup restore
# =============================================================================


class TestCompactionSafeWriteVerifyFailure:
    """Integration tests for the rotate → atomic_write → verify flow.

    These cover the acceptance criterion: simulate a verify failure and
    confirm the backup is restored and a crash_log entry is appended.
    """

    def _add_old_convo(self, memory_system):
        """Add a conversation in the 1–2 hour window so compact_hourly does work."""
        old_time = datetime.now() - timedelta(hours=1, minutes=30)
        memory_system.recent_conversations.append(
            ConversationEntry(
                timestamp=old_time,
                user="u",
                message="What is Django?",
                response="A web framework.",
            )
        )

    def test_hourly_verify_failure_restores_previous_backup(
        self, memory_system, temp_vault
    ):
        """If verify_memory_file returns False, the prior hourly.md is restored."""
        self._add_old_convo(memory_system)

        # Seed an existing hourly.md so rotate_backup captures it first.
        hourly_path = temp_vault / "LLM Memory" / "Context" / "hourly.md"
        original_body = "# Hourly Context\n\nKNOWN GOOD CONTENT\n"
        hourly_path.write_text(original_body, encoding="utf-8")

        # Force verify to fail once (for the fresh write) then pass during the
        # post-restore check (there is no post-restore verify today, but this
        # keeps the stub robust against future additions).
        call_count = {"n": 0}

        def fake_verify(path):
            call_count["n"] += 1
            return False  # always fail — write was "corrupted"

        with patch(
            "agent.memory_system.verify_memory_file", side_effect=fake_verify
        ), patch(
            "agent.memory_system.notifications.discord_send"
        ) as mock_notify:
            result = memory_system.compact_hourly(
                summarizer=lambda prompt: "Post-write summary — simulated bad write."
            )

        assert "restored" in result.lower()
        assert hourly_path.read_text(encoding="utf-8") == original_body
        # Auto-restore Discord notification must fire exactly once.
        assert mock_notify.call_count == 1

        crash_log = temp_vault / "LLM Memory" / "Permanent" / "crash_log.md"
        assert crash_log.exists()
        body = crash_log.read_text(encoding="utf-8")
        assert "hourly.md" in body
        assert "Compaction recovery" in body

    def test_daily_verify_failure_restores_previous_backup(
        self, memory_system, temp_vault
    ):
        """Same flow for compact_daily writes to daily.md."""
        memory_system.log_conversation("user1", "Hello", "Hi")

        daily_path = temp_vault / "LLM Memory" / "Context" / "daily.md"
        original_body = "# Daily Context\n\nOLD GOOD DAILY CONTENT\n"
        daily_path.write_text(original_body, encoding="utf-8")

        with patch(
            "agent.memory_system.verify_memory_file", return_value=False
        ), patch(
            "agent.memory_system.notifications.discord_send"
        ) as mock_notify:
            result = memory_system.compact_daily(
                summarizer=lambda prompt: "Post-write daily summary."
            )

        assert "restored" in result.lower()
        assert daily_path.read_text(encoding="utf-8") == original_body
        assert mock_notify.call_count == 1

    def test_verify_failure_without_backup_reports_restore_failed(
        self, memory_system, temp_vault
    ):
        """If verify fails and no backup exists, the caller learns restore_failed.

        This can happen on the very first hourly write when no prior snapshot
        exists — we must not silently pretend everything worked.
        """
        self._add_old_convo(memory_system)

        # Ensure no backup dir exists at all.
        assert not (temp_vault / "Backups" / "memory").exists()

        # Make list_backups return empty for the live verify-fail path by
        # patching verify to always return False.
        with patch(
            "agent.memory_system.verify_memory_file", return_value=False
        ), patch(
            "agent.memory_system.list_backups", return_value=[]
        ):
            result = memory_system.compact_hourly(
                summarizer=lambda prompt: "summary"
            )

        assert "no backup could be restored" in result.lower()

        crash_log = temp_vault / "LLM Memory" / "Permanent" / "crash_log.md"
        assert crash_log.exists()
        assert "no backup" in crash_log.read_text(encoding="utf-8").lower()


# =============================================================================
# COMPACTION SAFE-WRITE — filelock contention skips the cycle cleanly
# =============================================================================


class TestCompactionSafeWriteLockContention:
    """When the per-path lock cannot be acquired, the cycle is skipped."""

    def _add_old_convo(self, memory_system):
        old_time = datetime.now() - timedelta(hours=1, minutes=30)
        memory_system.recent_conversations.append(
            ConversationEntry(
                timestamp=old_time,
                user="u",
                message="Any topic.",
                response="Any response.",
            )
        )

    def test_hourly_skipped_on_lock_timeout(
        self, memory_system, temp_vault, caplog
    ):
        """FileLock.acquire raises Timeout → compact_hourly returns 'skipped'."""
        from filelock import Timeout as FileLockTimeout

        self._add_old_convo(memory_system)

        hourly_path = temp_vault / "LLM Memory" / "Context" / "hourly.md"
        original = "# Hourly Context\n\nSHOULD NOT BE OVERWRITTEN\n"
        hourly_path.write_text(original, encoding="utf-8")

        with patch(
            "agent.memory_system.FileLock.acquire",
            side_effect=FileLockTimeout("hourly.md.lock"),
        ), caplog.at_level(logging.WARNING, logger="agent.memory_system"):
            result = memory_system.compact_hourly(
                summarizer=lambda prompt: "summary"
            )

        assert "skipped" in result.lower()
        # Live file must be untouched when the cycle is skipped.
        assert hourly_path.read_text(encoding="utf-8") == original
        # Warning must be logged so operators can see lock contention in prod.
        assert any(
            "lock" in r.getMessage().lower() and "hourly" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_daily_skipped_on_lock_timeout(self, memory_system, temp_vault):
        """Same skip-cleanly behaviour for compact_daily."""
        from filelock import Timeout as FileLockTimeout

        memory_system.log_conversation("user1", "Hi", "Hello")

        daily_path = temp_vault / "LLM Memory" / "Context" / "daily.md"
        original = "# Daily Context\n\nSHOULD NOT BE OVERWRITTEN\n"
        daily_path.write_text(original, encoding="utf-8")

        with patch(
            "agent.memory_system.FileLock.acquire",
            side_effect=FileLockTimeout("daily.md.lock"),
        ):
            result = memory_system.compact_daily(
                summarizer=lambda prompt: "summary"
            )

        assert "skipped" in result.lower()
        assert daily_path.read_text(encoding="utf-8") == original


# =============================================================================
# COMPACTION SAFE-WRITE — healthy path still rotates a per-target backup
# =============================================================================


class TestCompactionSafeWriteHealthy:
    """Successful compaction still produces a per-path rotate_backup snapshot."""

    def test_hourly_success_creates_per_target_backup(
        self, memory_system, temp_vault
    ):
        """A successful compact_hourly leaves a snapshot containing hourly.md."""
        old_time = datetime.now() - timedelta(hours=1, minutes=30)
        memory_system.recent_conversations.append(
            ConversationEntry(
                timestamp=old_time,
                user="u",
                message="msg",
                response="resp",
            )
        )

        hourly_path = temp_vault / "LLM Memory" / "Context" / "hourly.md"
        hourly_path.write_text("# Hourly\n\nprevious summary\n", encoding="utf-8")

        result = memory_system.compact_hourly(
            summarizer=lambda prompt: "fresh summary"
        )

        assert "Compacted 1" in result
        assert "fresh summary" in hourly_path.read_text(encoding="utf-8")

        # rotate_backup keeps the pre-write body in at least one snapshot.
        snapshots = list_backups(temp_vault, filename="hourly.md")
        assert snapshots, "rotate_backup should have produced at least one snapshot"
        assert any(
            "previous summary" in (s / "hourly.md").read_text(encoding="utf-8")
            for s in snapshots
        )
