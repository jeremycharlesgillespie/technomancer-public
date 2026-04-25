"""Specific tests for updating TK-1077 summary as mentioned in TK-1144."""

from unittest.mock import patch, MagicMock

import pytest

from idea_board.jira_sync import update_jira_summary


class TestTK1077SummaryUpdate:
    """Tests for updating the specific TK-1077 ticket summary."""

    def test_update_tk1077_summary(self):
        """Test updating the specific TK-1077 ticket to the required summary."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.text = ""

        with patch("idea_board.jira_sync.is_jira_configured", return_value=True):
            with patch("idea_board.jira_sync._api") as mock_api:
                mock_api.return_value = mock_response
                result = update_jira_summary(
                    "TK-1077", 
                    "Split scope: null-safe behavior and iterative fix plan"
                )
                assert result is True
                # Verify the exact summary was passed
                mock_api.assert_called_once()
                call_args = mock_api.call_args
                assert call_args[1]['json']['fields']['summary'] == "Split scope: null-safe behavior and iterative fix plan"
                # Check that the method and path were called correctly
                assert call_args[0][0] == "put"  # method
                assert call_args[0][1] == "/issue/TK-1077"  # path