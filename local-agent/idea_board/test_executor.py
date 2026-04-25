"""Tests for idea_board.executor._project_key_for — project key extraction."""

import pytest
from unittest.mock import patch

from idea_board.executor import _project_key_for
from agent.config import settings


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