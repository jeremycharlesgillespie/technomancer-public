"""Tests for Jira label add functionality."""

from unittest.mock import patch, MagicMock

import pytest
import requests

from idea_board.jira_sync import add_jira_label, is_jira_configured


class TestJiraLabelAdd:
    """Tests for add_jira_label function."""

    def test_add_jira_label_not_configured(self):
        """Test that add_jira_label returns False when Jira is not configured."""
        with patch("idea_board.jira_sync.is_jira_configured", return_value=False):
            result = add_jira_label("TK-1077", "src:splitter")
            assert result is False

    def test_add_jira_label_success_new_label(self):
        """Test successful Jira label addition with new label."""
        # Mock the GET request to retrieve current issue
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "fields": {
                "labels": ["existing-label"]
            }
        }
        
        # Mock the PUT request to add the label
        mock_put_response = MagicMock()
        mock_put_response.status_code = 204
        mock_put_response.text = ""

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.side_effect = [mock_get_response, mock_put_response]
                result = add_jira_label("TK-1077", "src:splitter")
                assert result is True
                # Should make two API calls: GET and PUT
                assert mock_api.call_count == 2
                
                # First call should be GET
                mock_api.assert_any_call(
                    "get",
                    "/issue/TK-1077",
                    jira_key="TK-1077"
                )
                
                # Second call should be PUT with updated labels
                mock_api.assert_any_call(
                    "put",
                    "/issue/TK-1077",
                    json={"fields": {"labels": ["existing-label", "src:splitter"]}}
                )

    def test_add_jira_label_success_existing_label(self):
        """Test that add_jira_label returns True when label already exists."""
        # Mock the GET request to retrieve current issue
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "fields": {
                "labels": ["existing-label", "src:splitter"]
            }
        }

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.return_value = mock_get_response
                result = add_jira_label("TK-1077", "src:splitter")
                assert result is True
                # Should only make one GET call, no PUT call
                assert mock_api.call_count == 1
                mock_api.assert_called_once_with(
                    "get",
                    "/issue/TK-1077",
                    jira_key="TK-1077"
                )

    def test_add_jira_label_get_failure(self):
        """Test Jira label addition when GET request fails."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 404
        mock_get_response.text = "Not Found"

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.return_value = mock_get_response
                result = add_jira_label("TK-1077", "src:splitter")
                assert result is False
                mock_api.assert_called_once_with(
                    "get",
                    "/issue/TK-1077",
                    jira_key="TK-1077"
                )

    def test_add_jira_label_put_failure(self):
        """Test Jira label addition when PUT request fails."""
        # Mock the GET request to retrieve current issue
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "fields": {
                "labels": ["existing-label"]
            }
        }
        
        # Mock the PUT request to add the label - this will fail
        mock_put_response = MagicMock()
        mock_put_response.status_code = 400
        mock_put_response.text = "Bad Request"

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.side_effect = [mock_get_response, mock_put_response]
                result = add_jira_label("TK-1077", "src:splitter")
                assert result is False
                # Should make two API calls: GET and PUT
                assert mock_api.call_count == 2

    def test_add_jira_label_exception(self):
        """Test Jira label addition with exception."""
        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.side_effect = Exception("Network error")
                result = add_jira_label("TK-1077", "src:splitter")
                assert result is False