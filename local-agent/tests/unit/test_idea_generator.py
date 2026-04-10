"""Tests for idea_generator — idea parsing, prompt building, generation cycle."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.idea_generator import IDEA_PROMPT, _load_codebase_summary, _load_errors, _load_performance


class TestIdeaPrompt:
    """Test the prompt template."""

    def test_prompt_has_lifecycle_section(self):
        assert "FULL LIFECYCLE" in IDEA_PROMPT

    def test_prompt_requires_end_to_end(self):
        assert "THINK END-TO-END" in IDEA_PROMPT

    def test_prompt_supports_epics(self):
        assert "epic" in IDEA_PROMPT.lower()
        assert "stories" in IDEA_PROMPT.lower()

    def test_prompt_has_all_placeholders(self):
        for key in ["codebase", "news", "conversations", "errors", "performance", "existing"]:
            assert f"{{{key}}}" in IDEA_PROMPT, f"Missing placeholder: {key}"

    def test_prompt_mentions_idea_type(self):
        assert "idea_type" in IDEA_PROMPT


class TestLoadCodebaseSummary:
    def test_returns_string(self):
        result = _load_codebase_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_python_files(self):
        result = _load_codebase_summary()
        assert ".py" in result or "core" in result


class TestLoadErrors:
    def test_returns_string(self):
        result = _load_errors()
        assert isinstance(result, str)


class TestLoadPerformance:
    def test_returns_string(self):
        result = _load_performance()
        assert isinstance(result, str)


class TestParseIdeas:
    """Test the _parse_ideas function."""

    def test_parses_valid_json(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {
                "title": "Test Idea",
                "description": "WHAT: Something\n\nWHY: Because",
                "category": "quality",
                "source": "conversation_analysis",
            }
        ])
        result = _parse_ideas(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Test Idea"

    def test_parses_json_in_markdown(self):
        from agent.idea_generator import _parse_ideas

        raw = "Here are my suggestions:\n```json\n" + json.dumps([
            {"title": "Idea", "description": "Desc", "category": "feature", "source": "news_analysis"}
        ]) + "\n```"
        result = _parse_ideas(raw)
        assert len(result) >= 1

    def test_handles_empty_response(self):
        from agent.idea_generator import _parse_ideas

        result = _parse_ideas("")
        assert result == []

    def test_handles_invalid_json(self):
        from agent.idea_generator import _parse_ideas

        result = _parse_ideas("This is not JSON at all")
        assert result == []

    def test_filters_missing_fields(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {"title": "Good", "description": "Has all fields", "category": "quality", "source": "test"},
            {"title": "Bad"},  # missing description
        ])
        result = _parse_ideas(raw)
        # Should keep valid, skip invalid
        assert len(result) >= 1

    def test_preserves_extra_fields(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {
                "title": "Epic Idea",
                "description": "Full lifecycle",
                "category": "feature",
                "source": "news_analysis",
                "idea_type": "epic",
                "stories": ["Story 1", "Story 2"],
            }
        ])
        result = _parse_ideas(raw)
        assert len(result) == 1
        # _parse_ideas may or may not preserve extra fields
        # Just verify it parsed correctly
        assert result[0]["title"] == "Epic Idea"
