"""Tests for idea_board.executor._project_key_for — project key extraction."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for, _is_valid_idea_id, _PhaseMarker, ExecutionState
from agent.config import settings


# ---------------------------------------------------------------------------
# Helper function for missing project key scenarios
# ---------------------------------------------------------- -------------------

def _assert_missing_key_handling(
    test_func,
    expected_result: str | None = None,
    message: str = "Expected None for missing project key"
):
    """Reusable assertion helper for testing missing project key scenarios.

    This helper encapsulates the common pattern of testing that _project_key_for
    returns None (or a specific fallback) when given inputs that should not
    produce a valid project key. It provides a clear contract for failure
    scenarios and ensures consistent test logic.

    Args:
        test_func: A callable that takes a string input and returns the result
                   of _project_key_for(input).
        expected_result: The expected result (typically None, but can be a
                         fallback string like "FA" when jira_project_key is set).
        message: Custom assertion message describing the expected behavior.

    Example:
        _assert_missing_key_handling(
            lambda x: _project_key_for(x),
            expected_result=None,
            message="Should return None for empty string"
        )
    """
    with patch.object(settings, 'jira_project_key', None):
        result = test_func()
        assert result == expected_result, message


class TestProjectKeyFor:
    """Test _project_key_for function with various inputs."""

    def test_returns_none_for_none_input(self):
        """_project_key_for should return None for None input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for(None) is None

    def test_returns_none_for_empty_string(self):
        """_project_key_for should return None for empty string input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("") is None

    def test_returns_none_for_whitespace_only(self):
        """_project_key_for should return None for whitespace-only input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("   ") is None

    def test_returns_project_key_when_present(self):
        """_project_key_for should return the project key when found."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("FA-123") == "FA"

    def test_returns_none_when_project_key_not_found(self):
        """_project_key_for should return None when project key is not found."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("TK-123") is None

    def test_returns_none_when_project_key_is_not_at_start(self):
        """_project_key_for should return None when project key is not at the start."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("Some text FA-123 more text") is None

    def test_handles_multiple_project_keys(self):
        """_project_key_for should handle multiple potential project keys."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("FA-123") == "FA"
            assert _project_key_for("FA-456") == "FA"
            assert _project_key_for("TK-123") is None

    def test_returns_none_when_jira_project_key_is_none(self):
        """_project_key_for should return None when jira_project_key is None."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-123") is None

    def test_returns_none_when_jira_project_key_is_empty_string(self):
        """_project_key_for should return None when jira_project_key is empty string."""
        with patch.object(settings, 'jira_project_key', ""):
            assert _project_key_for("FA-123") is None

    def test_returns_none_for_non_string_input(self):
        """_project_key_for should return None for non-string inputs."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for(123) is None
            assert _project_key_for([]) is None
            assert _project_key_for({}) is None

    def test_returns_FA_for_FA_1029(self):
        """_project_key_for should extract FA from FA-1029."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-1029") == "FA"

    def test_returns_TK_for_TK_500(self):
        """_project_key_for should extract TK from TK-500."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("TK-500") == "TK"

    def test_is_valid_idea_id_returns_false_for_text_only(self):
        """_is_valid_idea_id should return False for 'text-only'."""
        assert _is_valid_idea_id("text-only") is False

    def test_is_valid_idea_id_returns_true_for_FA_100(self):
        """_is_valid_idea_id should return True for 'FA-100'."""
        assert _is_valid_idea_id("FA-100") is True

    def test_project_key_for_returns_none_for_text_only(self):
        """_project_key_for should return None for 'text-only'."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("text-only") is None

    def test_project_key_for_returns_FA_for_FA_100(self):
        """_project_key_for should return 'FA' for 'FA-100'."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-100") == "FA"

    def test_is_valid_idea_id_edge_cases(self):
        """Test edge cases for _is_valid_idea_id function."""
        # Valid cases
        assert _is_valid_idea_id("FA-100") is True
        assert _is_valid_idea_id("TK-500") is True
        assert _is_valid_idea_id("ABC-123") is True
        
        # Invalid cases
        assert _is_valid_idea_id("text-only") is False
        assert _is_valid_idea_id("") is False
        assert _is_valid_idea_id("   ") is False
        assert _is_valid_idea_id("123-456") is False  # No alphabetic prefix
        assert _is_valid_idea_id("FA") is False  # No hyphen
        assert _is_valid_idea_id("FA-") is False  # No suffix
        assert _is_valid_idea_id("-123") is False  # No prefix
        assert _is_valid_idea_id("FA-123abc") is True  # Has digits in suffix
        assert _is_valid_idea_id("FA-abc") is False  # No digits in suffix
        assert _is_valid_idea_id(123) is False  # Not a string
        assert _is_valid_idea_id(None) is False  # Not a string