"""Tests for Jira summary update functionality."""

from unittest.mock import patch, MagicMock

import pytest
import requests

from idea_board.jira_sync import update_jira_summary, is_jira_configured


class TestJiraSummaryUpdate:
    """Tests for update_jira_summary function."""

    def test_update_jira_summary_not_configured(self):
        """Test that update_jira_summary returns False when Jira is not configured."""
        with patch("idea_board.jira_sync.is_jira_configured", return_value=False):
            result = update_jira_summary("TK-1077", "Split scope: null-safe behavior and iterative fix plan")
            assert result is False

    def test_update_jira_summary_success(self):
        """Test successful Jira summary update."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.text = ""

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.return_value = mock_response
                result = update_jira_summary("TK-1077", "Split scope: null-safe behavior and iterative fix plan")
                assert result is True
                mock_api.assert_called_once_with(
                    "put",
                    "/issue/TK-1077",
                    json={"fields": {"summary": "Split scope: null-safe behavior and iterative fix plan"}}
                )

    def test_update_jira_summary_failure(self):
        """Test failed Jira summary update."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.return_value = mock_response
                result = update_jira_summary("TK-1077", "Split scope: null-safe behavior and iterative fix plan")
                assert result is False
                mock_api.assert_called_once_with(
                    "put",
                    "/issue/TK-1077",
                    json={"fields": {"summary": "Split scope: null-safe behavior and iterative fix plan"}}
                )

    def test_update_jira_summary_exception(self):
        """Test Jira summary update with exception."""
        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.side_effect = Exception("Network error")
                result = update_jira_summary("TK-1077", "Split scope: null-safe behavior and iterative fix plan")
                assert result is False