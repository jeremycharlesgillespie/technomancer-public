"""Tests for null project handling in idea_board.executor."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for, _state_timer, ExecutionState
from agent.story_timings import phase_timer, record_phase


class TestNullProjectHandling:
    """Test that null project handling works correctly."""

    def test_project_key_for_none_returns_none(self):
        """Test that _project_key_for returns None for None input."""
        assert _project_key_for(None) is None

    def test_project_key_for_empty_string_returns_none(self):
        """Test that _project_key_for returns None for empty string input."""
        assert _project_key_for("") is None

    def test_project_key_for_invalid_format_returns_none(self):
        """Test that _project_key_for returns None for invalid format."""
        assert _project_key_for("invalid") is None
        assert _project_key_for("123-456") is None
        assert _project_key_for("abc-") is None

    def test_project_key_for_valid_format_returns_prefix(self):
        """Test that _project_key_for returns correct prefix for valid format."""
        assert _project_key_for("TK-123") == "TK"
        assert _project_key_for("FA-456") == "FA"
        assert _project_key_for("ABC-789") == "ABC"

    def test_state_timer_with_none_idea_id(self):
        """Test that _state_timer correctly handles None idea_id."""
        # Create an ExecutionState with None idea_id
        state = ExecutionState(idea_id=None)
        
        # Mock phase_timer to capture the call
        with patch('idea_board.executor.phase_timer') as mock_phase_timer:
            # Call _state_timer - this should not raise an exception
            _state_timer(state, "test_phase")
            
            # Verify that phase_timer was called with project=None
            mock_phase_timer.assert_called_once()
            call_args = mock_phase_timer.call_args
            assert call_args[1]['project'] is None
            assert call_args[1]['story_id'] is None