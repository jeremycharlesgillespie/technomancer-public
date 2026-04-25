"""Comprehensive tests for _project_key_for function in idea_board.executor."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for


class TestProjectKeyForComprehensive:
    """Comprehensive test for _project_key_for function behavior."""

    def test_all_cases(self):
        """Test all edge cases for _project_key_for."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            # Should return None for inputs without digits
            assert _project_key_for("abc") is None
            assert _project_key_for("hello") is None
            assert _project_key_for("test") is None
            assert _project_key_for("xyz") is None
            assert _project_key_for("") is None
            assert _project_key_for("   ") is None
            assert _project_key_for("\t\n") is None
            assert _project_key_for(None) is None
            assert _project_key_for("123") is None  # Only digits, no prefix
            assert _project_key_for("123-456") is None  # No alpha prefix
            assert _project_key_for("123abc") is None  # No dash, just digits + alpha
            
            # Should return prefix for valid inputs
            assert _project_key_for("TK-123") == "TK"
            assert _project_key_for("FA-456") == "FA"
            assert _project_key_for("ABC-789") == "ABC"
            assert _project_key_for("TK-1") == "TK"
            assert _project_key_for("FA-0") == "FA"
            
            # Should return project key when set
            with patch("idea_board.executor.settings.jira_project_key", "CUSTOM"):
                assert _project_key_for("any-id") == "CUSTOM"
                assert _project_key_for("TK-123") == "CUSTOM"  # Should override prefix logic

    def test_whitespace_only_inputs(self):
        """Test that whitespace-only inputs return None (TK-1194 requirement)."""
        with patch("idea_board.executor.settings.jira_project_key", None):
            # These are the exact cases mentioned in the story
            whitespace_cases = [
                "   ",      # spaces only
                "\t\n",     # tabs and newlines  
                " \t \n ",  # mixed whitespace
                "",         # empty string
                " ",        # single space
                "\t",       # tab only
                "\n",       # newline only
            ]
            
            for case in whitespace_cases:
                assert _project_key_for(case) is None, f"Expected None for {repr(case)}"