"""Tests for itinerary module — Claude-powered travel itinerary generation."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent.itinerary import ITINERARY_SYSTEM_PROMPT, ITINERARY_HTML_TEMPLATE


class TestItineraryConstants:
    """Test prompt and template content."""

    def test_system_prompt_has_structure(self):
        assert "Day-by-Day" in ITINERARY_SYSTEM_PROMPT
        assert "Day 1" in ITINERARY_SYSTEM_PROMPT
        assert "Morning" in ITINERARY_SYSTEM_PROMPT

    def test_html_template_valid(self):
        assert "<html" in ITINERARY_HTML_TEMPLATE
        assert "{title}" in ITINERARY_HTML_TEMPLATE
        assert "{content}" in ITINERARY_HTML_TEMPLATE
        assert "{duration}" in ITINERARY_HTML_TEMPLATE


class TestGenerateItinerary:
    """Test the generate_itinerary async function."""

    def test_no_api_key_returns_error(self, monkeypatch):
        from agent.itinerary import generate_itinerary
        monkeypatch.setattr("agent.itinerary.settings", MagicMock(anthropic_api_key=None))

        with patch("agent.itinerary.ClaudeBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge.client = None  # No API key
            mock_bridge_cls.return_value = mock_bridge

            md, html, dur = asyncio.run(generate_itinerary("Japan trip"))
            assert "Error" in md or "not configured" in md

    def test_function_is_async(self):
        from agent.itinerary import generate_itinerary
        import inspect
        assert inspect.iscoroutinefunction(generate_itinerary)

    @patch("agent.itinerary.ClaudeBridge")
    def test_error_handling(self, mock_bridge_cls, monkeypatch):
        from agent.itinerary import generate_itinerary
        monkeypatch.setattr("agent.itinerary.settings", MagicMock(
            anthropic_api_key="fake-key",
        ))

        mock_bridge = MagicMock()
        mock_bridge.client = MagicMock()
        mock_bridge.send.return_value = "Error: API timeout"
        mock_bridge_cls.return_value = mock_bridge

        with patch("agent.itinerary._record_perf"):
            md, html, dur = asyncio.run(generate_itinerary("test"))
        assert isinstance(md, str)
