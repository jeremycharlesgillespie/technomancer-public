"""Tests for agent/auto_memory.py - Auto memory extraction system."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.auto_memory import (
    MEMORY_CATEGORIES,
    build_extraction_prompt,
    buffer_conversation,
    extract_memories,
    get_identity_summary,
    merge_facts_into_files,
    parse_extraction_response,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_identity(tmp_path, monkeypatch):
    """Patch identity dir and extraction log to use temp directory."""
    import agent.auto_memory as am_module

    identity_dir = tmp_path / "Permanent" / "identity"
    identity_dir.mkdir(parents=True)

    monkeypatch.setattr(am_module, "IDENTITY_DIR", identity_dir)
    monkeypatch.setattr(am_module, "EXTRACTION_LOG", tmp_path / "extraction_log.md")
    monkeypatch.setattr(am_module, "VAULT_PATH", tmp_path)

    return identity_dir


@pytest.fixture
def sample_conversations():
    """Sample conversations for testing extraction."""
    return [
        {
            "user": "gman386",
            "message": "I really prefer Django over Flask for bigger projects. The ORM is just better.",
            "response": "That makes sense - Django's ORM is more full-featured for complex data models.",
        },
        {
            "user": "gman386",
            "message": "Working on a new RPA project at Franklin Templeton using Python and AWS Lambda.",
            "response": "Sounds like a great use case for serverless with your automation work!",
        },
        {
            "user": "gman386",
            "message": "I've been into retro gaming lately, playing a lot of SNES stuff on weekends.",
            "response": "Nice! What games have you been playing?",
        },
    ]


@pytest.fixture
def mock_agent_with_facts():
    """Mock agent that returns JSON extraction results."""
    agent = MagicMock()
    agent.run = MagicMock(
        return_value=json.dumps([
            {"category": "preferences", "fact": "Prefers Django over Flask for large projects", "confidence": "high"},
            {"category": "experiences", "fact": "Works at Franklin Templeton on RPA projects", "confidence": "high"},
            {"category": "preferences", "fact": "Enjoys retro gaming, especially SNES, on weekends", "confidence": "high"},
        ])
    )
    return agent


@pytest.fixture
def mock_agent_empty():
    """Mock agent that returns no facts."""
    agent = MagicMock()
    agent.run = MagicMock(return_value="[]")
    return agent


# =============================================================================
# PARSE EXTRACTION RESPONSE TESTS
# =============================================================================


class TestParseExtractionResponse:
    """Tests for parsing LLM extraction responses."""

    def test_parse_json_array(self):
        response = json.dumps([
            {"category": "preferences", "fact": "Likes Python", "confidence": "high"}
        ])
        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0]["fact"] == "Likes Python"

    def test_parse_json_in_code_block(self):
        response = '```json\n[{"category": "skills", "fact": "Expert in Django", "confidence": "high"}]\n```'
        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0]["category"] == "skills"

    def test_parse_empty_array(self):
        facts = parse_extraction_response("[]")
        assert facts == []

    def test_parse_invalid_json(self):
        facts = parse_extraction_response("This is not JSON at all")
        assert facts == []

    def test_parse_invalid_category(self):
        response = json.dumps([
            {"category": "invalid_category", "fact": "Something", "confidence": "high"}
        ])
        facts = parse_extraction_response(response)
        assert facts == []  # Invalid category filtered out

    def test_parse_missing_fields(self):
        response = json.dumps([{"category": "preferences"}])  # Missing "fact"
        facts = parse_extraction_response(response)
        assert facts == []

    def test_parse_mixed_valid_invalid(self):
        response = json.dumps([
            {"category": "preferences", "fact": "Valid fact", "confidence": "high"},
            {"category": "bogus", "fact": "Invalid category"},
            {"not_a_fact": True},
        ])
        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0]["fact"] == "Valid fact"


# =============================================================================
# MERGE FACTS TESTS
# =============================================================================


class TestMergeFacts:
    """Tests for merging extracted facts into identity files."""

    def test_merge_creates_file(self, temp_identity):
        facts = [{"category": "preferences", "fact": "Loves dark mode", "confidence": "high"}]
        updated = merge_facts_into_files(facts)
        assert "preferences" in updated

        content = (temp_identity / "preferences.md").read_text(encoding="utf-8")
        assert "Loves dark mode" in content

    def test_merge_appends_to_existing(self, temp_identity):
        # Pre-populate
        (temp_identity / "skills.md").write_text(
            "# Skills\n\n- Expert in Python [high, 2026-01-01]\n", encoding="utf-8"
        )

        facts = [{"category": "skills", "fact": "Learning Rust", "confidence": "medium"}]
        merge_facts_into_files(facts)

        content = (temp_identity / "skills.md").read_text(encoding="utf-8")
        assert "Expert in Python" in content  # Original preserved
        assert "Learning Rust" in content  # New fact added

    def test_merge_skips_duplicates(self, temp_identity):
        (temp_identity / "preferences.md").write_text(
            "# Preferences\n\n- Prefers Django over Flask for large projects [high, 2026-01-01]\n",
            encoding="utf-8",
        )

        facts = [
            {"category": "preferences", "fact": "Prefers Django over Flask for large projects", "confidence": "high"}
        ]
        updated = merge_facts_into_files(facts)
        assert "preferences" not in updated  # Nothing new was added

    def test_merge_multiple_categories(self, temp_identity):
        facts = [
            {"category": "preferences", "fact": "Likes VSCode", "confidence": "high"},
            {"category": "goals", "fact": "Wants to learn Kubernetes", "confidence": "medium"},
        ]
        updated = merge_facts_into_files(facts)
        assert "preferences" in updated
        assert "goals" in updated

    def test_merge_includes_timestamp(self, temp_identity):
        facts = [{"category": "habits", "fact": "Works late at night", "confidence": "high"}]
        merge_facts_into_files(facts)

        content = (temp_identity / "habits.md").read_text(encoding="utf-8")
        assert "2026-" in content  # Has a date stamp


# =============================================================================
# BUILD PROMPT TESTS
# =============================================================================


class TestBuildPrompt:
    """Tests for extraction prompt construction."""

    def test_prompt_includes_conversations(self, sample_conversations):
        prompt = build_extraction_prompt(sample_conversations, {})
        assert "Django over Flask" in prompt
        assert "Franklin Templeton" in prompt
        assert "SNES" in prompt

    def test_prompt_includes_existing_facts(self, sample_conversations):
        existing = {"preferences": "# Preferences\n\n- Already known fact\n"}
        prompt = build_extraction_prompt(sample_conversations, existing)
        assert "Already known fact" in prompt

    def test_prompt_includes_all_categories(self, sample_conversations):
        prompt = build_extraction_prompt(sample_conversations, {})
        for cat in MEMORY_CATEGORIES:
            assert cat in prompt


# =============================================================================
# FULL EXTRACTION PIPELINE
# =============================================================================


class TestExtractMemories:
    """Tests for the full extraction pipeline."""

    def test_extract_returns_stats(self, temp_identity, mock_agent_with_facts, sample_conversations):
        result = asyncio.run(extract_memories(mock_agent_with_facts, sample_conversations))
        assert result["facts_extracted"] == 3
        assert len(result["categories_updated"]) > 0
        assert result["duration_seconds"] >= 0

    def test_extract_writes_to_files(self, temp_identity, mock_agent_with_facts, sample_conversations):
        asyncio.run(extract_memories(mock_agent_with_facts, sample_conversations))

        # Check that identity files were created
        assert (temp_identity / "preferences.md").exists()
        assert (temp_identity / "experiences.md").exists()

    def test_extract_empty_response(self, temp_identity, mock_agent_empty, sample_conversations):
        result = asyncio.run(extract_memories(mock_agent_empty, sample_conversations))
        assert result["facts_extracted"] == 0

    def test_extract_agent_error(self, temp_identity, sample_conversations):
        agent = MagicMock()
        agent.run = MagicMock(side_effect=Exception("LLM down"))
        result = asyncio.run(extract_memories(agent, sample_conversations))
        assert "error" in result

    def test_extraction_log_created(self, temp_identity, mock_agent_with_facts, sample_conversations):
        asyncio.run(extract_memories(mock_agent_with_facts, sample_conversations))
        log_path = temp_identity.parent / "extraction_log.md"
        # The log is at EXTRACTION_LOG which we patched
        import agent.auto_memory as am_module
        assert am_module.EXTRACTION_LOG.exists()


# =============================================================================
# IDENTITY SUMMARY
# =============================================================================


class TestIdentitySummary:
    """Tests for the identity summary used in context injection."""

    def test_summary_empty_when_no_data(self, temp_identity):
        summary = get_identity_summary()
        assert "No identity data" in summary

    def test_summary_includes_facts(self, temp_identity):
        (temp_identity / "preferences.md").write_text(
            "# Preferences\n\n- Loves Python\n- Prefers dark mode\n", encoding="utf-8"
        )
        summary = get_identity_summary()
        assert "Loves Python" in summary
        assert "Prefers dark mode" in summary
        assert "Preferences" in summary


# =============================================================================
# CONVERSATION BUFFER
# =============================================================================


class TestConversationBuffer:
    """Tests for the conversation buffering system."""

    def test_buffer_adds_conversation(self, monkeypatch):
        import agent.auto_memory as am_module
        monkeypatch.setattr(am_module, "_conversation_buffer", [])

        buffer_conversation("user", "I really prefer Django over Flask for bigger projects because the ORM is so much better", "That makes sense for complex data models and large applications")
        assert len(am_module._conversation_buffer) == 1

    def test_buffer_skips_short_conversations(self, monkeypatch):
        import agent.auto_memory as am_module
        monkeypatch.setattr(am_module, "_conversation_buffer", [])

        buffer_conversation("user", "hi", "hey")
        assert len(am_module._conversation_buffer) == 0  # Too short
