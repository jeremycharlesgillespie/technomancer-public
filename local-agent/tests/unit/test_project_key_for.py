"""Tests for _project_key_for function in idea_board.executor."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for


class TestProjectKeyFor:
    """Test _project_key_for function behavior."""

    def test_returns_none_for_plain_text_without_digits(self):
        """Test that plain text inputs without digits return None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("abc") is None
            assert _project_key_for("hello") is None
            assert _project_key_for("test") is None
            assert _project_key_for("xyz") is None

    def test_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("") is None

    def test_returns_none_for_whitespace_only(self):
        """Test that whitespace-only inputs return None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("   ") is None
            assert _project_key_for("\t\n") is None

    def test_returns_none_for_none_input(self):
        """Test that None input returns None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for(None) is None

    def test_returns_project_key_when_set(self):
        """Test that when jira_project_key is set, it returns that value."""
        with patch("idea_board.executor.settings.jira_project_key", "TK"):
            assert _project_key_for("any-id") == "TK"

    def test_returns_prefix_for_valid_id_with_digits(self):
        """Test that valid IDs with digits return the prefix."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("TK-123") == "TK"
            assert _project_key_for("FA-456") == "FA"
            assert _project_key_for("ABC-789") == "ABC"

    def test_returns_none_for_ids_with_only_digits(self):
        """Test that IDs with only digits return None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("123") is None
            assert _project_key_for("456") is None
            assert _project_key_for("789") is None

    def test_returns_none_for_ids_with_no_prefix(self):
        """Test that IDs without a prefix return None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("123-456") is None
            assert _project_key_for("789-012") is None

    def test_returns_none_for_ids_with_non_alpha_prefix(self):
        """Test that IDs with non-alpha prefix return None."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            assert _project_key_for("123-abc") is None
            assert _project_key_for("456-xyz") is None
            assert _project_key_for("123abc") is None