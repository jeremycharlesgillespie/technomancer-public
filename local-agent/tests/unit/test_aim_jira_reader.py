"""Tests for aim.jira_reader — Jira board read APIs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Mock jira_sync._api and is_jira_configured before importing
@pytest.fixture(autouse=True)
def _mock_jira(monkeypatch):
    """Ensure Jira is 'configured' and _api is mocked for all tests."""
    monkeypatch.setattr("idea_board.jira_sync.is_jira_configured", lambda: True)


@pytest.fixture
def mock_api():
    """Provide a mock for the Jira _api function."""
    with patch("aim.jira_reader._api") as m:
        yield m


class TestCountIssuesByStatus:
    def test_empty_project(self, mock_api):
        from aim.jira_reader import count_issues_by_status

        # First call: total count
        resp_total = MagicMock()
        resp_total.status_code = 200
        resp_total.json.return_value = {"total": 0}

        mock_api.return_value = resp_total

        result = count_issues_by_status()
        assert result == {}

    def test_counts_multiple_statuses(self, mock_api):
        from aim.jira_reader import count_issues_by_status

        # First call returns total
        resp_total = MagicMock()
        resp_total.status_code = 200
        resp_total.json.return_value = {"total": 3}

        # Second call returns paginated results
        resp_page = MagicMock()
        resp_page.status_code = 200
        resp_page.json.return_value = {
            "issues": [
                {"fields": {"status": {"name": "To Do"}}},
                {"fields": {"status": {"name": "To Do"}}},
                {"fields": {"status": {"name": "Done"}}},
            ]
        }

        mock_api.side_effect = [resp_total, resp_page]

        result = count_issues_by_status()
        assert result == {"To Do": 2, "Done": 1}

    def test_api_failure_returns_empty(self, mock_api):
        from aim.jira_reader import count_issues_by_status

        resp = MagicMock()
        resp.status_code = 500
        mock_api.return_value = resp

        result = count_issues_by_status()
        assert result == {}

    def test_exception_returns_empty(self, mock_api):
        from aim.jira_reader import count_issues_by_status

        mock_api.side_effect = ConnectionError("Network down")

        result = count_issues_by_status()
        assert result == {}

    def test_jira_not_configured(self, monkeypatch):
        from aim.jira_reader import count_issues_by_status

        monkeypatch.setattr("aim.jira_reader.is_jira_configured", lambda: False)
        result = count_issues_by_status()
        assert result == {}


class TestListTodoIssues:
    def test_returns_issues(self, mock_api):
        from aim.jira_reader import list_todo_issues

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "issues": [
                {
                    "key": "TK-1",
                    "fields": {
                        "summary": "[idea-001] Fix bug",
                        "issuetype": {"name": "Story"},
                        "labels": ["quality"],
                        "created": "2026-04-14T10:00:00Z",
                    },
                },
                {
                    "key": "TK-2",
                    "fields": {
                        "summary": "[idea-002] Add feature",
                        "issuetype": {"name": "Epic"},
                        "labels": [],
                        "created": "2026-04-14T11:00:00Z",
                    },
                },
            ]
        }
        mock_api.return_value = resp

        result = list_todo_issues()
        assert len(result) == 2
        assert result[0]["key"] == "TK-1"
        assert result[0]["summary"] == "[idea-001] Fix bug"
        assert result[0]["issue_type"] == "Story"
        assert result[1]["labels"] == []

    def test_empty_board(self, mock_api):
        from aim.jira_reader import list_todo_issues

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"issues": []}
        mock_api.return_value = resp

        result = list_todo_issues()
        assert result == []

    def test_api_failure(self, mock_api):
        from aim.jira_reader import list_todo_issues

        resp = MagicMock()
        resp.status_code = 403
        mock_api.return_value = resp

        result = list_todo_issues()
        assert result == []

    def test_jira_not_configured(self, monkeypatch):
        from aim.jira_reader import list_todo_issues

        monkeypatch.setattr("aim.jira_reader.is_jira_configured", lambda: False)
        result = list_todo_issues()
        assert result == []


class TestGetRecentCompletions:
    def test_returns_completions(self, mock_api):
        from aim.jira_reader import get_recent_completions

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "issues": [
                {
                    "key": "TK-5",
                    "fields": {
                        "summary": "Done task",
                        "resolutiondate": "2026-04-14T12:00:00Z",
                        "updated": "2026-04-14T12:00:00Z",
                    },
                },
            ]
        }
        mock_api.return_value = resp

        result = get_recent_completions(hours=24)
        assert len(result) == 1
        assert result[0]["key"] == "TK-5"

    def test_uses_updated_as_fallback(self, mock_api):
        from aim.jira_reader import get_recent_completions

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "issues": [
                {
                    "key": "TK-6",
                    "fields": {
                        "summary": "Done task",
                        "resolutiondate": None,
                        "updated": "2026-04-14T13:00:00Z",
                    },
                },
            ]
        }
        mock_api.return_value = resp

        result = get_recent_completions()
        assert result[0]["resolved"] == "2026-04-14T13:00:00Z"


class TestGetBoardSummary:
    def test_combines_counts_and_completions(self, mock_api):
        from aim.jira_reader import get_board_summary

        # Mock count_issues_by_status (calls _api twice for pagination)
        resp_total = MagicMock()
        resp_total.status_code = 200
        resp_total.json.return_value = {"total": 2}

        resp_page = MagicMock()
        resp_page.status_code = 200
        resp_page.json.return_value = {
            "issues": [
                {"fields": {"status": {"name": "To Do"}}},
                {"fields": {"status": {"name": "Done"}}},
            ]
        }

        # Mock get_recent_completions
        resp_completions = MagicMock()
        resp_completions.status_code = 200
        resp_completions.json.return_value = {
            "issues": [
                {
                    "key": "TK-1",
                    "fields": {
                        "summary": "Done",
                        "resolutiondate": "2026-04-14T12:00:00Z",
                        "updated": "2026-04-14T12:00:00Z",
                    },
                },
            ]
        }

        mock_api.side_effect = [resp_total, resp_page, resp_completions]

        result = get_board_summary()
        assert result["todo"] == 1
        assert result["done_total"] == 1
        assert result["done_last_24h"] == 1
        assert result["jira_available"] is True
