"""Tests for idea_board.jira_sync — Jira integration."""

from unittest.mock import MagicMock, patch, call
import pytest

from idea_board.jira_sync import (
    is_jira_configured,
    find_jira_issue,
    create_jira_issue,
    transition_jira_issue,
    sync_idea_to_jira,
    TYPE_MAP,
    STATE_MAP,
    _build_description_adf,
)


# ---------------------------------------------------------------------------
# is_jira_configured
# ---------------------------------------------------------------------------


class TestIsJiraConfigured:
    @patch("idea_board.jira_sync.settings")
    def test_configured_when_all_set(self, mock_settings):
        mock_settings.jira_url = "https://test.atlassian.net"
        mock_settings.jira_email = "test@test.com"
        mock_settings.jira_api_token = "token123"
        mock_settings.jira_project_key = "TK"
        assert is_jira_configured() is True

    @patch("idea_board.jira_sync.settings")
    def test_not_configured_when_missing_url(self, mock_settings):
        mock_settings.jira_url = None
        mock_settings.jira_email = "test@test.com"
        mock_settings.jira_api_token = "token123"
        mock_settings.jira_project_key = "TK"
        assert is_jira_configured() is False

    @patch("idea_board.jira_sync.settings")
    def test_not_configured_when_missing_token(self, mock_settings):
        mock_settings.jira_url = "https://test.atlassian.net"
        mock_settings.jira_email = "test@test.com"
        mock_settings.jira_api_token = None
        mock_settings.jira_project_key = "TK"
        assert is_jira_configured() is False


# ---------------------------------------------------------------------------
# _build_description_adf
# ---------------------------------------------------------------------------


class TestBuildDescriptionAdf:
    def test_basic_text(self):
        result = _build_description_adf("Hello world")
        assert result["type"] == "doc"
        assert result["version"] == 1
        assert result["content"][0]["type"] == "paragraph"
        assert result["content"][0]["content"][0]["text"] == "Hello world"

    def test_truncates_long_text(self):
        long_text = "x" * 40000
        result = _build_description_adf(long_text)
        assert len(result["content"][0]["content"][0]["text"]) == 30000


# ---------------------------------------------------------------------------
# find_jira_issue
# ---------------------------------------------------------------------------


class TestFindJiraIssue:
    @patch("idea_board.jira_sync.is_jira_configured", return_value=False)
    def test_returns_none_when_not_configured(self, mock_conf):
        assert find_jira_issue("idea-001") is None

    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_returns_key_when_found(self, mock_conf, mock_api):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "issues": [{"key": "TK-42"}]
        }
        mock_api.return_value = mock_resp
        assert find_jira_issue("idea-001") == "TK-42"

    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_returns_none_when_not_found(self, mock_conf, mock_api):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"issues": []}
        mock_api.return_value = mock_resp
        assert find_jira_issue("idea-999") is None


# ---------------------------------------------------------------------------
# create_jira_issue
# ---------------------------------------------------------------------------


class TestCreateJiraIssue:
    @patch("idea_board.jira_sync.is_jira_configured", return_value=False)
    def test_returns_none_when_not_configured(self, mock_conf):
        assert create_jira_issue("idea-001", "Test", "Desc") is None

    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_creates_story(self, mock_settings, mock_conf, mock_api, mock_find):
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "localhost"
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"key": "TK-10"}
        mock_api.return_value = mock_resp

        result = create_jira_issue("idea-001", "My Story", "Description here")
        assert result == "TK-10"

        # First API call is the create, second is the comment
        create_call = mock_api.call_args_list[0]
        fields = create_call[1]["json"]["fields"]
        assert fields["issuetype"]["name"] == "Story"
        assert "[idea-001]" in fields["summary"]

    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_creates_epic(self, mock_settings, mock_conf, mock_api, mock_find):
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "localhost"
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"key": "TK-11"}
        mock_api.return_value = mock_resp

        result = create_jira_issue("idea-001", "My Epic", "Desc", idea_type="epic")
        assert result == "TK-11"
        create_call = mock_api.call_args_list[0]
        fields = create_call[1]["json"]["fields"]
        assert fields["issuetype"]["name"] == "Epic"

    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_adds_execute_comment(self, mock_settings, mock_conf, mock_api, mock_find):
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "10.0.0.1"
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"key": "TK-20"}
        mock_api.return_value = mock_resp

        create_jira_issue("idea-042", "Test", "Desc")

        # Second API call should be the comment
        assert mock_api.call_count == 2
        comment_call = mock_api.call_args_list[1]
        assert comment_call[0] == ("post", "/issue/TK-20/comment")

    @patch("idea_board.jira_sync.find_jira_issue", return_value="TK-5")
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_links_to_parent_epic(self, mock_settings, mock_conf, mock_api, mock_find):
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "localhost"
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"key": "TK-12"}
        mock_api.return_value = mock_resp

        result = create_jira_issue(
            "idea-002", "Child Story", "Desc",
            parent_idea_id="idea-001"
        )
        assert result == "TK-12"
        create_call = mock_api.call_args_list[0]
        fields = create_call[1]["json"]["fields"]
        assert fields["parent"]["key"] == "TK-5"


# ---------------------------------------------------------------------------
# transition_jira_issue
# ---------------------------------------------------------------------------


class TestTransitionJiraIssue:
    @patch("idea_board.jira_sync.is_jira_configured", return_value=False)
    def test_returns_false_when_not_configured(self, mock_conf):
        assert transition_jira_issue("TK-1", "Done") is False

    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_direct_transition(self, mock_conf, mock_api):
        # First call: get transitions
        trans_resp = MagicMock()
        trans_resp.status_code = 200
        trans_resp.json.return_value = {
            "transitions": [
                {"name": "Done", "id": "31"},
                {"name": "In Progress", "id": "21"},
            ]
        }
        # Second call: post transition
        post_resp = MagicMock()
        post_resp.status_code = 204

        mock_api.side_effect = [trans_resp, post_resp]
        assert transition_jira_issue("TK-1", "Done") is True

    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_two_step_transition_to_done(self, mock_conf, mock_api):
        # First get: only In Progress available
        trans_resp1 = MagicMock()
        trans_resp1.status_code = 200
        trans_resp1.json.return_value = {
            "transitions": [{"name": "In Progress", "id": "21"}]
        }
        # Post: transition to In Progress
        post_resp1 = MagicMock()
        post_resp1.status_code = 204
        # Second get: Done now available
        trans_resp2 = MagicMock()
        trans_resp2.status_code = 200
        trans_resp2.json.return_value = {
            "transitions": [{"name": "Done", "id": "31"}]
        }
        # Post: transition to Done
        post_resp2 = MagicMock()
        post_resp2.status_code = 204

        mock_api.side_effect = [trans_resp1, post_resp1, trans_resp2, post_resp2]
        assert transition_jira_issue("TK-1", "Done") is True


# ---------------------------------------------------------------------------
# sync_idea_to_jira (integration)
# ---------------------------------------------------------------------------


class TestSyncIdeaToJira:
    @patch("idea_board.jira_sync.is_jira_configured", return_value=False)
    def test_returns_none_when_not_configured(self, mock_conf):
        idea = MagicMock(id="idea-001", state="done")
        assert sync_idea_to_jira(idea) is None

    @patch("idea_board.jira_sync.transition_jira_issue")
    @patch("idea_board.jira_sync.create_jira_issue", return_value="TK-50")
    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_creates_and_transitions_new_idea(
        self, mock_conf, mock_find, mock_create, mock_trans
    ):
        idea = MagicMock(
            id="idea-001", title="Test", description="Desc",
            idea_type="story", state="done", parent_id=None, category="feature"
        )
        result = sync_idea_to_jira(idea)
        assert result == "TK-50"
        mock_create.assert_called_once()
        mock_trans.assert_called_once_with("TK-50", "Done")

    @patch("idea_board.jira_sync.transition_jira_issue")
    @patch("idea_board.jira_sync.find_jira_issue", return_value="TK-42")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_transitions_existing_idea(self, mock_conf, mock_find, mock_trans):
        idea = MagicMock(
            id="idea-001", title="Test", description="Desc",
            idea_type="story", state="executing", parent_id=None, category=""
        )
        result = sync_idea_to_jira(idea)
        assert result == "TK-42"
        mock_trans.assert_called_once_with("TK-42", "In Progress")

    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_skips_vetoed(self, mock_conf):
        idea = MagicMock(id="idea-001", state="vetoed")
        assert sync_idea_to_jira(idea) is None

    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_skips_failed(self, mock_conf):
        idea = MagicMock(id="idea-001", state="failed")
        assert sync_idea_to_jira(idea) is None

    @patch("idea_board.jira_sync.transition_jira_issue")
    @patch("idea_board.jira_sync.create_jira_issue", return_value="TK-60")
    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_works_with_dict_input(
        self, mock_conf, mock_find, mock_create, mock_trans
    ):
        """sync_idea_to_jira should work with both Idea objects and dicts."""
        idea = {
            "id": "idea-001", "title": "Test", "description": "Desc",
            "idea_type": "story", "state": "approved", "parent_id": None,
            "category": "feature",
        }
        result = sync_idea_to_jira(idea)
        assert result == "TK-60"
        mock_trans.assert_called_once_with("TK-60", "To Do")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_type_map_covers_all_types(self):
        assert "epic" in TYPE_MAP
        assert "story" in TYPE_MAP
        assert "task" in TYPE_MAP

    def test_state_map_covers_active_states(self):
        for state in ("proposed", "refining", "approved", "executing", "done"):
            assert state in STATE_MAP
