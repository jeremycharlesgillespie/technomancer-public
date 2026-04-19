"""Tests for get_execution_detail_href helper (TK-781).

Acceptance criteria:
- Returns "/live/<idea_id>" for valid idea_id
- Returns empty string for None, empty string, and whitespace-only inputs
- Handles special characters (passed through unchanged — callers escape)
"""

from __future__ import annotations

import pytest

from idea_board.web import get_execution_detail_href


class TestGetExecutionDetailHref:
    def test_jira_key_returns_live_href(self):
        assert get_execution_detail_href("TK-781") == "/live/TK-781"

    def test_numeric_id_returns_live_href(self):
        assert get_execution_detail_href("42") == "/live/42"

    def test_integer_id_returns_live_href(self):
        assert get_execution_detail_href(42) == "/live/42"

    def test_none_returns_empty_string(self):
        assert get_execution_detail_href(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert get_execution_detail_href("") == ""

    def test_whitespace_only_returns_empty_string(self):
        assert get_execution_detail_href("   ") == ""

    def test_special_chars_passed_through(self):
        assert get_execution_detail_href("idea/foo") == "/live/idea/foo"

    def test_href_starts_with_slash_live(self):
        href = get_execution_detail_href("TK-1")
        assert href.startswith("/live/")
