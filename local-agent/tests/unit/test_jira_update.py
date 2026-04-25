"""Tests for board.jira_update — Jira update utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from board.jira_update import (
    update_jira_ticket_description,
    update_ticket_with_subtask_references,
    update_tk_1077_description_with_subtasks,
    find_ticket_by_id,
    search_ticket_by_jql,
)


@pytest.fixture(autouse=True)
def _mock_jira_configured(monkeypatch):
    """Pretend Jira is configured in every test."""
    monkeypatch.setattr("idea_board.jira_sync.is_jira_configured", lambda: True)
    monkeypatch.setattr("board.jira_provider.is_jira_configured", lambda: True)


@pytest.fixture
def mock_api():
    """Mock board.jira_provider._api — covers all HTTP calls inside the module."""
    with patch("board.jira_update._api") as m:
        yield m


class TestUpdateJiraTicketDescription:
    def test_update_description_success(self, mock_api):
        resp = MagicMock()
        resp.status_code = 200
        mock_api.return_value = resp
        
        result = update_jira_ticket_description(
            jira_key="TK-1077",
            new_description="New description text"
        )
        
        assert result is True
        mock_api.assert_called_once()
        
    def test_update_description_failure(self, mock_api):
        resp = MagicMock()
        resp.status_code = 400
        mock_api.return_value = resp
        
        result = update_jira_ticket_description(
            jira_key="TK-1077",
            new_description="New description text"
        )
        
        assert result is False
        
    def test_update_description_with_labels(self, mock_api):
        resp = MagicMock()
        resp.status_code = 200
        mock_api.return_value = resp
        
        result = update_jira_ticket_description(
            jira_key="TK-1077",
            new_description="New description text",
            add_labels=["src:splitter"],
            remove_labels=["old-label"]
        )
        
        assert result is True
        call_args = mock_api.call_args
        assert "update" in call_args.kwargs["json"]
        assert "description" in call_args.kwargs["json"]["update"]
        assert "labels" in call_args.kwargs["json"]["update"]


class TestUpdateTicketWithSubtaskReferences:
    def test_update_with_subtasks_success(self, mock_api):
        resp = MagicMock()
        resp.status_code = 200
        mock_api.return_value = resp
        
        result = update_ticket_with_subtask_references(
            ticket_key="TK-1077",
            subtask_keys=["TK-1077-1", "TK-1077-2"]
        )
        
        assert result is True
        
    def test_update_with_empty_subtasks(self, mock_api, caplog):
        result = update_ticket_with_subtask_references(
            ticket_key="TK-1077",
            subtask_keys=[]
        )
        
        assert result is False
        assert "No sub-tasks provided" in caplog.text


class TestUpdateTk1077DescriptionWithSubtasks:
    def test_update_tk_1077_success(self, mock_api):
        resp = MagicMock()
        resp.status_code = 200
        mock_api.return_value = resp
        
        result = update_tk_1077_description_with_subtasks(
            subtask_keys=["TK-1077-1", "TK-1077-2"]
        )
        
        assert result is True
        # Verify it called with the right ticket key
        call_args = mock_api.call_args
        assert "/issue/TK-1077" in call_args.args[1]


class TestFindTicketById:
    def test_find_ticket_success(self, mock_api):
        with patch("idea_board.jira_sync.find_jira_issue", return_value="TK-1077"):
            result = find_ticket_by_id("test-idea")
            assert result == "TK-1077"


class TestSearchTicketByJql:
    def test_search_success(self, mock_api):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"issues": [{"key": "TK-1077"}]}
        mock_api.return_value = mock_response
        
        with patch("board.jira_provider._search", return_value=[{"key": "TK-1077"}]):
            result = search_ticket_by_jql("project = TK AND summary ~ 'test'")
            assert result is not None
            assert len(result) == 1