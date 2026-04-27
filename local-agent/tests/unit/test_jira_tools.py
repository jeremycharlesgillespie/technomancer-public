"""Tests for agent/jira_tools.py — LLM-facing list_ideas / get_idea tools."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.jira_tools import _get_idea_impl, _list_ideas_impl, get_jira_tools


@pytest.fixture
def fake_provider(monkeypatch):
    """Patch board.get_provider to return a MagicMock."""
    provider = MagicMock()
    monkeypatch.setattr("board.get_provider", lambda: provider)
    monkeypatch.setattr("board.factory.get_provider", lambda: provider)
    return provider


class TestGetJiraTools:
    """Test the tool registration shape."""

    def test_returns_list_ideas_and_get_idea(self):
        tools = get_jira_tools()
        names = [t.name for t in tools]
        assert names == ["list_ideas", "get_idea"]

    def test_get_idea_has_required_idea_id(self):
        tools = get_jira_tools()
        get_tool = next(t for t in tools if t.name == "get_idea")
        assert "idea_id" in get_tool.parameters["properties"]
        assert get_tool.parameters["required"] == ["idea_id"]

    def test_list_ideas_has_optional_state(self):
        tools = get_jira_tools()
        list_tool = next(t for t in tools if t.name == "list_ideas")
        assert "state" in list_tool.parameters["properties"]
        assert list_tool.parameters["required"] == []


class TestListIdeas:
    """Tests for _list_ideas_impl."""

    def test_calls_provider_with_state(self, fake_provider):
        fake_provider.list_ideas_for_llm.return_value = "**Board** — 0"
        result = _list_ideas_impl(state="approved")
        fake_provider.list_ideas_for_llm.assert_called_once_with(state="approved")
        assert result == "**Board** — 0"

    def test_calls_provider_with_empty_state_by_default(self, fake_provider):
        fake_provider.list_ideas_for_llm.return_value = "active"
        result = _list_ideas_impl()
        fake_provider.list_ideas_for_llm.assert_called_once_with(state="")
        assert result == "active"

    def test_returns_error_string_on_provider_failure(self, fake_provider):
        fake_provider.list_ideas_for_llm.side_effect = RuntimeError("Jira down")
        result = _list_ideas_impl()
        assert result.startswith("Board lookup failed")
        assert "Jira down" in result


class TestGetIdea:
    """Tests for _get_idea_impl."""

    def test_returns_required_message_when_blank(self, fake_provider):
        result = _get_idea_impl("")
        assert "required" in result.lower()
        fake_provider.get.assert_not_called()

    def test_returns_required_when_whitespace(self, fake_provider):
        result = _get_idea_impl("   ")
        assert "required" in result.lower()
        fake_provider.get.assert_not_called()

    def test_returns_not_found_when_provider_returns_none(self, fake_provider):
        fake_provider.get.return_value = None
        result = _get_idea_impl("TK-999")
        assert "TK-999" in result
        assert "not found" in result.lower()

    def test_strips_whitespace_before_lookup(self, fake_provider):
        fake_provider.get.return_value = None
        _get_idea_impl("  TK-42  ")
        fake_provider.get.assert_called_once_with("TK-42")

    def test_formats_full_idea(self, fake_provider):
        item = SimpleNamespace(
            id="TK-42",
            title="Speed up search",
            state="approved",
            idea_type="story",
            category="performance",
            source="planning",
            parent_id=None,
            description="WHAT: Optimize\nWHY: Slow",
        )
        fake_provider.get.return_value = item
        result = _get_idea_impl("TK-42")
        assert "# TK-42: Speed up search" in result
        assert "State: approved" in result
        assert "Type: story" in result
        assert "Category: performance" in result
        assert "Source: planning" in result
        assert "WHAT: Optimize" in result

    def test_includes_parent_when_present(self, fake_provider):
        item = SimpleNamespace(
            id="TK-42",
            title="Sub-story",
            state="approved",
            idea_type="story",
            category="feature",
            source="planning",
            parent_id="TK-10",
            description="",
        )
        fake_provider.get.return_value = item
        result = _get_idea_impl("TK-42")
        assert "Parent: TK-10" in result

    def test_returns_error_string_on_provider_failure(self, fake_provider):
        fake_provider.get.side_effect = RuntimeError("network")
        result = _get_idea_impl("TK-42")
        assert result.startswith("Lookup failed")
        assert "network" in result
