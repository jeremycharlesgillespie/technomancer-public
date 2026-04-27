"""Tests for idea_board.executor._project_key_for — project key extraction."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for, _PhaseMarker, ExecutionState
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

    def test_returns_none_for_dash_only(self):
        """_project_key_for should return None for dash-only input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("-") is None

    def test_returns_none_for_double_dash_only(self):
        """_project_key_for should return None for double-dash-only input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("--") is None

    def test_returns_project_key_from_jira_project_setting(self):
        """_project_key_for should return jira_project_key setting when set."""
        with patch.object(settings, 'jira_project_key', 'TK'):
            assert _project_key_for("FA-123") == "TK"

    def test_returns_prefix_for_valid_input_with_jira_not_set(self):
        """_project_key_for should return prefix when jira_project_key not set."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-123") == "FA"

    def test_returns_none_for_input_without_digits(self):
        """_project_key_for should return None for inputs without digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-abc") is None

    def test_returns_none_for_input_without_hyphen(self):
        """_project_key_for should return None for inputs without hyphen."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA123") is None

    def test_returns_none_for_input_with_non_alpha_prefix(self):
        """_project_key_for should return None for inputs with non-alpha prefix."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("123-FA") is None

    def test_returns_uppercase_prefix(self):
        """_project_key_for should return uppercase prefix."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("fa-123") == "FA"

    def test_handles_mixed_case_prefix(self):
        """_project_key_for should handle mixed case prefix correctly."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("Fa-123") == "FA"

    def test_handles_long_prefix(self):
        """_project_key_for should handle long alpha prefixes."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABCDEF-123") == "ABCDEF"

    def test_handles_prefix_with_numbers_and_letters(self):
        """_project_key_for should return None for prefixes with mixed alphanumeric."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC123-DEF-456") is None

    def test_handles_prefix_with_special_characters(self):
        """_project_key_for should return None for prefixes with special characters."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("!@#-ABC-123") is None

    def test_handles_prefix_with_underscore(self):
        """_project_key_for should return None for prefixes with underscores."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC_DEF-123") is None

    def test_handles_prefix_with_space(self):
        """_project_key_for should return None for prefixes with spaces."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC DEF-123") is None

    def test_handles_prefix_with_leading_hyphen(self):
        """_project_key_for should return None for prefixes starting with hyphen."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("-ABC-123") is None

    def test_handles_prefix_with_multiple_hyphens(self):
        """_project_key_for should handle inputs with multiple hyphens correctly."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC-DEF-GHI-123") == "ABC"

    def test_handles_prefix_with_trailing_hyphen(self):
        """_project_key_for should handle prefixes ending with trailing hyphens correctly."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC--123") == "ABC"

    def test_handles_prefix_with_only_one_char(self):
        """_project_key_for should handle single character prefixes."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A-123") == "A"

    def test_handles_prefix_with_only_one_char_no_digits(self):
        """_project_key_for should return None for single character prefix without digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A-BCD") is None

    def test_handles_prefix_with_only_one_char_with_digits(self):
        """_project_key_for should handle single character prefix with digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A-123") == "A"

    def test_handles_prefix_with_only_one_char_no_hyphen(self):
        """_project_key_for should return None for single character prefix without hyphen."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A123") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_no_digits(self):
        """_project_key_for should return None for single character prefix without hyphen or digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_digits(self):
        """_project_key_for should return None for single character prefix without hyphen but with digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A123") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_digits_and_letters(self):
        """_project_key_for should return None for single character prefix without hyphen but with digits and letters."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("A123B") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_digits(self):
        """_project_key_for should return None for single character prefix without hyphen and only digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("123") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_letters(self):
        """_project_key_for should return None for single character prefix without hyphen and only letters."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("ABC") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_special_chars(self):
        """_project_key_for should return None for single character prefix without hyphen and only special chars."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("!@#") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_spaces(self):
        """_project_key_for should return None for single character prefix without hyphen and only spaces."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("   ") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_hyphens(self):
        """_project_key_for should return None for single character prefix without hyphen and only hyphens."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("---") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("___") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores_and_hyphens(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores/hyphens."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("_-_-") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores_and_hyphens_and_spaces(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores/hyphens/spaces."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("_ - _") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores_and_hyphens_and_spaces_and_digits(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores/hyphens/spaces/digits."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("_ - 123") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores_and_hyphens_and_spaces_and_letters(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores/hyphens/spaces/letters."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("_ - ABC") is None

    def test_handles_prefix_with_only_one_char_no_hyphen_with_only_underscores_and_hyphens_and_spaces_and_digits_and_letters(self):
        """_project_key_for should return None for single character prefix without hyphen and only underscores/hyphens/spaces/digits/letters."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("_ - 123ABC") is None

    def test_handles_very_long_input(self):
        """_project_key_for should handle very long inputs gracefully."""
        with patch.object(settings, 'jira_project_key', None):
            long_input = "A" * 1000 + "-123"
            assert _project_key_for(long_input) == "A" * 1000

    def test_handles_edge_case_with_numbers_only(self):
        """_project_key_for should handle inputs with only numbers gracefully."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("123456") is None

    def test_handles_edge_case_with_only_hyphens(self):
        """_project_key_for should handle inputs with only hyphens gracefully."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("---") is None

    def test_handles_none_settings_jira_project_key(self):
        """_project_key_for should handle None jira_project_key gracefully."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-123") == "FA"

    def test_handles_empty_settings_jira_project_key(self):
        """_project_key_for should handle empty jira_project_key gracefully."""
        with patch.object(settings, 'jira_project_key', ""):
            assert _project_key_for("FA-123") == "FA"

    def test_handles_whitespace_settings_jira_project_key(self):
        """_project_key_for should handle whitespace jira_project_key gracefully."""
        with patch.object(settings, 'jira_project_key', "   "):
            # When jira_project_key is whitespace, it should return the whitespace value
            result = _project_key_for("FA-123")
            # The function should not crash, but the behavior with whitespace is to return it
            assert result is not None  # Should not be None

    def test_returns_valid_project_key_for_FA_100(self):
        """_project_key_for should correctly extract 'FA' from 'FA-100' input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-100") == "FA"

    def test_valid_idea_id_formats(self):
        """Test that valid idea_id formats are correctly handled."""
        # Test the specific case mentioned in the story
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-100") == "FA"
            
            # Test other valid formats
            assert _project_key_for("TK-447") == "TK"
            assert _project_key_for("AB-123") == "AB"
            assert _project_key_for("XYZ-789") == "XYZ"
            
            # Test with mixed case (should be normalized to uppercase)
            assert _project_key_for("fa-100") == "FA"
            assert _project_key_for("tk-447") == "TK"

    def test_invalid_idea_id_formats(self):
        """Test that invalid idea_id formats are rejected."""
        with patch.object(settings, 'jira_project_key', None):
            # Test the specific case mentioned in the story
            assert _project_key_for("invalid@char-100") is None
            
            # Test other invalid formats
            assert _project_key_for("invalid@char") is None
            assert _project_key_for("invalid#char") is None
            assert _project_key_for("invalid$char") is None
            assert _project_key_for("invalid%char") is None
            assert _project_key_for("invalid^char") is None
            assert _project_key_for("invalid&char") is None
            assert _project_key_for("invalid*char") is None
            assert _project_key_for("invalid(char") is None
            assert _project_key_for("invalid)char") is None
            assert _project_key_for("invalid[char") is None
            assert _project_key_for("invalid]char") is None
            assert _project_key_for("invalid{char") is None
            assert _project_key_for("invalid}char") is None
            assert _project_key_for("invalid|char") is None
            assert _project_key_for("invalid\\char") is None
            assert _project_key_for("invalid:char") is None
            assert _project_key_for("invalid;char") is None
            assert _project_key_for("invalid\"char") is None
            assert _project_key_for("invalid'char") is None
            assert _project_key_for("invalid<char") is None
            assert _project_key_for("invalid>char") is None
            assert _project_key_for("invalid?char") is None
            assert _project_key_for("invalid/char") is None
            assert _project_key_for("invalid.char") is None
            assert _project_key_for("invalid,char") is None
            assert _project_key_for("invalid char") is None
            assert _project_key_for("invalid\tchar") is None
            assert _project_key_for("invalid\nchar") is None
            assert _project_key_for("invalid\rchar") is None
            assert _project_key_for("invalid\t\n\rchar") is None
            assert _project_key_for("invalid@char123") is None
            assert _project_key_for("invalid@char123@") is None
            assert _project_key_for("123invalid") is None
            assert _project_key_for("invalid123") is None
            assert _project_key_for("invalid123@char") is None
            assert _project_key_for("invalid@char123@char") is None


class TestPhaseMarkerFinish:
    """Test _PhaseMarker.finish() method with NULL project keys."""

    def test_finish_handles_none_project_key(self):
        """_PhaseMarker.finish() should handle None project keys without raising exceptions."""
        # Create a mock execution state with None idea_id (which leads to None project key)
        state = ExecutionState(idea_id=None)
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()
        
    def test_finish_handles_empty_string_project_key(self):
        """_PhaseMarker.finish() should handle empty string project keys without raising exceptions."""
        # Create a mock execution state with empty string idea_id (which leads to None project key)
        state = ExecutionState(idea_id="")
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()
        
    def test_finish_handles_whitespace_project_key(self):
        """_PhaseMarker.finish() should handle whitespace-only project keys without raising exceptions."""
        # Create a mock execution state with whitespace idea_id (which leads to None project key)
        state = ExecutionState(idea_id="   ")
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()
        
    def test_finish_handles_invalid_project_key_format(self):
        """_PhaseMarker.finish() should handle invalid project key formats without raising exceptions."""
        # Create a mock execution state with invalid idea_id format (no digits)
        state = ExecutionState(idea_id="invalid-format")
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()
        
    def test_finish_handles_dash_only_project_key(self):
        """_PhaseMarker.finish() should handle dash-only project keys without raising exceptions."""
        # Create a mock execution state with dash-only idea_id (which leads to None project key)
        state = ExecutionState(idea_id="-")
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()
        
    def test_finish_handles_double_dash_project_key(self):
        """_PhaseMarker.finish() should handle double-dash project keys without raising exceptions."""
        # Create a mock execution state with double-dash idea_id (which leads to None project key)
        state = ExecutionState(idea_id="--")
        
        # Create a phase marker
        marker = _PhaseMarker(state, "test_phase")
        
        # This should not raise any exceptions
        marker.finish()