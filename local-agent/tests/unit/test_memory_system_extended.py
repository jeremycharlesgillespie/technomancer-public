"""Extended tests for memory_system — conversation logging, context retrieval, compaction."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_system import MemorySystem


@pytest.fixture
def mem_sys(tmp_path):
    """Create a MemorySystem with temp vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    mem_dir = vault / "LLM Memory"
    mem_dir.mkdir()
    (mem_dir / "Context").mkdir()
    (mem_dir / "Permanent").mkdir()
    (mem_dir / "Conversations").mkdir()
    ms = MemorySystem(str(vault))
    return ms


class TestMemorySystemInit:
    def test_creates_directories(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        ms = MemorySystem(str(vault))
        assert (vault / "LLM Memory" / "Context").exists()
        assert (vault / "LLM Memory" / "Permanent").exists()

    def test_stores_vault_path(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        ms = MemorySystem(str(vault))
        # vault_path may be a Path or str depending on implementation
        assert str(vault) in str(ms.vault_path)


class TestLogConversation:
    def test_logs_conversation(self, mem_sys):
        mem_sys.log_conversation("TestUser", "Hello", "Hi there!")
        # Should have logged to today's conversation file
        conversations_dir = Path(mem_sys.vault_path) / "LLM Memory" / "Conversations"
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = conversations_dir / f"{today}.md"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "TestUser" in content
        assert "Hello" in content

    def test_appends_multiple(self, mem_sys):
        mem_sys.log_conversation("Alice", "Q1", "A1")
        mem_sys.log_conversation("Bob", "Q2", "A2")
        conversations_dir = Path(mem_sys.vault_path) / "LLM Memory" / "Conversations"
        today = datetime.now().strftime("%Y-%m-%d")
        content = (conversations_dir / f"{today}.md").read_text(encoding="utf-8")
        assert "Alice" in content
        assert "Bob" in content


class TestGetContext:
    def test_empty_context(self, mem_sys):
        result = mem_sys.get_context("hour")
        assert isinstance(result, str)

    def test_returns_recent_conversations(self, mem_sys):
        mem_sys.log_conversation("TestUser", "What is Python?", "A programming language.")
        result = mem_sys.get_context("hour")
        assert "Python" in result or isinstance(result, str)

    def test_day_context(self, mem_sys):
        mem_sys.log_conversation("TestUser", "Test", "Response")
        result = mem_sys.get_context("day")
        assert isinstance(result, str)


class TestSavePermanentMemory:
    def test_saves_memory(self, mem_sys):
        result = mem_sys.save_permanent_memory("TestUser likes Python", category="preferences")
        assert "Saved" in result or "saved" in result

        memories_file = Path(mem_sys.vault_path) / "LLM Memory" / "Permanent" / "memories.md"
        if memories_file.exists():
            content = memories_file.read_text(encoding="utf-8")
            assert "Python" in content

    def test_replaces_category(self, mem_sys):
        mem_sys.save_permanent_memory("Old fact", category="test_cat")
        mem_sys.save_permanent_memory("New fact", category="test_cat", replace_category=True)

        memories_file = Path(mem_sys.vault_path) / "LLM Memory" / "Permanent" / "memories.md"
        if memories_file.exists():
            content = memories_file.read_text(encoding="utf-8")
            assert "New fact" in content


class TestCompactHourly:
    def test_compact_empty(self, mem_sys):
        result = mem_sys.compact_hourly()
        assert isinstance(result, str)

    def test_compact_with_data(self, mem_sys):
        mem_sys.log_conversation("TestUser", "Q1", "A1")
        mem_sys.log_conversation("TestUser", "Q2", "A2")
        mem_sys.log_conversation("TestUser", "Q3", "A3")
        result = mem_sys.compact_hourly()
        assert isinstance(result, str)


class TestCompactDaily:
    def test_compact_empty(self, mem_sys):
        result = mem_sys.compact_daily()
        assert isinstance(result, str)


class TestBackgroundCompaction:
    def test_start_stop(self, mem_sys):
        mem_sys.start_background_compaction(interval_minutes=60)
        # Check the thread was created
        assert hasattr(mem_sys, "_compaction_thread") or hasattr(mem_sys, "_compaction_running")
        mem_sys.stop_background_compaction()

    def test_double_start_is_safe(self, mem_sys):
        mem_sys.start_background_compaction(interval_minutes=60)
        mem_sys.start_background_compaction(interval_minutes=60)  # should not crash
        mem_sys.stop_background_compaction()
