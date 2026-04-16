"""
Tests for POST /api/jira/create endpoint in idea_board/web.py.

Validates:
- Required field validation (title, description)
- Optional field defaults (category, source, idea_type)
- Successful creation via BoardProvider
- Jira browse URL generation
- rank_position handling (top, after:<KEY>)
- Error handling for provider failures
- idea_type validation
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_board.models import Idea
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_item(key="TK-100", title="Test story", state="proposed", **kwargs):
    """Create a minimal Idea for use as a mock return value."""
    return Idea(
        id=key,
        title=title,
        description=kwargs.get("description", "A test"),
        state=state,
        source=kwargs.get("source", "planning"),
        category=kwargs.get("category", "quality"),
        idea_type=kwargs.get("idea_type", "story"),
        parent_id=kwargs.get("parent_id"),
    )


# ============================================================================
# FIELD VALIDATION
# ============================================================================


class TestJiraCreateValidation:
    """Tests for input validation on POST /api/jira/create."""

    def test_missing_title_returns_400(self, client):
        resp = client.post(
            "/api/jira/create",
            data=json.dumps({"description": "A test"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "title" in resp.get_json()["error"]

    def test_empty_title_returns_400(self, client):
        resp = client.post(
            "/api/jira/create",
            data=json.dumps({"title": "  ", "description": "A test"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "title" in resp.get_json()["error"]

    def test_missing_description_returns_400(self, client):
        resp = client.post(
            "/api/jira/create",
            data=json.dumps({"title": "A story"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "description" in resp.get_json()["error"]

    def test_empty_description_returns_400(self, client):
        resp = client.post(
            "/api/jira/create",
            data=json.dumps({"title": "A story", "description": "  "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "description" in resp.get_json()["error"]

    def test_invalid_idea_type_returns_400(self, client):
        resp = client.post(
            "/api/jira/create",
            data=json.dumps({
                "title": "A story",
                "description": "Desc",
                "idea_type": "task",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "idea_type" in resp.get_json()["error"]

    def test_no_json_body_returns_400(self, client):
        resp = client.post("/api/jira/create", content_type="application/json")
        assert resp.status_code == 400


# ============================================================================
# SUCCESSFUL CREATION
# ============================================================================


class TestJiraCreateSuccess:
    """Tests for successful story/epic creation."""

    def test_creates_story_with_defaults(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({"title": "Test story", "description": "A test"}),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["key"] == "TK-100"
        assert data["title"] == "Test story"
        assert data["state"] == "proposed"
        assert data["url"] == "https://test.atlassian.net/browse/TK-100"

        # Verify provider was called with correct defaults
        mock_provider.add.assert_called_once_with(
            title="Test story",
            description="A test",
            source="planning",
            category="quality",
            idea_type="story",
            parent_id=None,
        )

    def test_creates_epic_with_custom_fields(self, client):
        item = _make_item(key="TK-200", idea_type="epic", category="feature",
                          source="conversation_analysis")
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Big epic",
                    "description": "An epic",
                    "idea_type": "epic",
                    "category": "feature",
                    "source": "conversation_analysis",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["key"] == "TK-200"
        mock_provider.add.assert_called_once_with(
            title="Big epic",
            description="An epic",
            source="conversation_analysis",
            category="feature",
            idea_type="epic",
            parent_id=None,
        )

    def test_creates_story_with_parent_key(self, client):
        item = _make_item(parent_id="TK-10")
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Child story",
                    "description": "Under an epic",
                    "parent_key": "TK-10",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        mock_provider.add.assert_called_once_with(
            title="Child story",
            description="Under an epic",
            source="planning",
            category="quality",
            idea_type="story",
            parent_id="TK-10",
        )

    def test_no_browse_url_for_local_provider(self, client):
        """When Jira isn't configured, URL is empty."""
        item = _make_item(key="idea-001")
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = None
            mock_settings.jira_project_key = None

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({"title": "Local story", "description": "Desc"}),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["url"] == ""


# ============================================================================
# ERROR HANDLING
# ============================================================================


class TestJiraCreateErrors:
    """Tests for error conditions."""

    def test_provider_add_failure_returns_500(self, client):
        mock_provider = MagicMock()
        mock_provider.add.side_effect = RuntimeError("Jira API down")

        with patch("idea_board.web._get_board_provider", return_value=mock_provider):
            resp = client.post(
                "/api/jira/create",
                data=json.dumps({"title": "Fail story", "description": "Desc"}),
                content_type="application/json",
            )

        assert resp.status_code == 500
        assert "Failed to create issue" in resp.get_json()["error"]


# ============================================================================
# DUPLICATE COMMENT ATTACHMENT (TK-475)
# ============================================================================


class TestJiraCreateDuplicateComment:
    """On duplicate match, the incoming idea's body is posted as a comment
    on the canonical issue — unless the caller opts out via ``force=true``.
    """

    def test_duplicate_posts_comment_with_incoming_body(self, client):
        """409 path attaches exactly one [Duplicate Match] comment on the
        matched issue, carrying the incoming title and description."""
        existing = _make_item(key="TK-100", title="Existing issue")
        mock_provider = MagicMock()
        # load_all reports the existing item; provider.add returns the same
        # id, which is how the endpoint detects a dedup hit.
        mock_provider.load_all.return_value = [existing]
        mock_provider.add.return_value = existing

        payload = {
            "title": "Near duplicate",
            "description": "WHY: repeat of the existing item. HOW: dedup it.",
        }

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps(payload),
                content_type="application/json",
            )

        assert resp.status_code == 409
        body = resp.get_json()
        assert body["key"] == "TK-100"

        expected_text = (
            "[Duplicate Match] Incoming idea matched this issue.\n\n"
            f"Title: {payload['title']}\n\n{payload['description']}"
        )
        mock_provider.add_comment.assert_called_once_with(
            "TK-100", "jira_create", expected_text
        )

    def test_duplicate_with_force_true_skips_comment(self, client):
        """When ``force=true`` is supplied, the endpoint returns 409 as
        usual but does not attach a comment — callers that already intend
        to re-post the same content shouldn't spam the matched issue."""
        existing = _make_item(key="TK-101", title="Existing issue")
        mock_provider = MagicMock()
        mock_provider.load_all.return_value = [existing]
        mock_provider.add.return_value = existing

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Another duplicate",
                    "description": "Body that should not be posted.",
                    "force": True,
                }),
                content_type="application/json",
            )

        assert resp.status_code == 409
        mock_provider.add_comment.assert_not_called()


# ============================================================================
# RANKING
# ============================================================================


class TestJiraCreateRanking:
    """Tests for rank_position handling."""

    def test_rank_top_calls_agile_api(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        # Mock the JQL search to return a top To Do item
        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {
            "issues": [{"key": "TK-50", "fields": {"summary": "Top item"}}]
        }

        # Mock the rank PUT call
        mock_rank_resp = MagicMock()
        mock_rank_resp.status_code = 204

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", return_value=mock_search_resp), \
             patch("idea_board.web._requests_lib.put", return_value=mock_rank_resp) as mock_put:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"
            mock_settings.jira_email = "test@test.com"
            mock_settings.jira_api_token = "token"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Ranked story",
                    "description": "Desc",
                    "rank_position": "top",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["rank_result"] == "ok"
        mock_put.assert_called_once()
        call_kwargs = mock_put.call_args
        assert "rankBeforeIssue" in call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))

    def test_rank_after_key_calls_agile_api(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        mock_rank_resp = MagicMock()
        mock_rank_resp.status_code = 204

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._requests_lib.put", return_value=mock_rank_resp) as mock_put:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"
            mock_settings.jira_email = "test@test.com"
            mock_settings.jira_api_token = "token"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "After story",
                    "description": "Desc",
                    "rank_position": "after:TK-42",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["rank_result"] == "ok"
        call_kwargs = mock_put.call_args
        assert "rankAfterIssue" in call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))

    def test_rank_not_requested_no_rank_result(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings:
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({"title": "No rank", "description": "Desc"}),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert "rank_result" not in data

    def test_rank_top_no_todo_items(self, client):
        """When no To Do items exist, ranking is silently skipped."""
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {"issues": []}

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", return_value=mock_search_resp):
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Ranked story",
                    "description": "Desc",
                    "rank_position": "top",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["rank_result"] == "ok"

    def test_rank_invalid_position_returns_warning(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web.is_jira_configured", return_value=True):
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Bad rank",
                    "description": "Desc",
                    "rank_position": "middle",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert "warning" in data["rank_result"]

    def test_rank_after_empty_key_returns_warning(self, client):
        item = _make_item()
        mock_provider = MagicMock()
        mock_provider.add.return_value = item

        with patch("idea_board.web._get_board_provider", return_value=mock_provider), \
             patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web.is_jira_configured", return_value=True):
            mock_settings.jira_url = "https://test.atlassian.net"
            mock_settings.jira_project_key = "TK"

            resp = client.post(
                "/api/jira/create",
                data=json.dumps({
                    "title": "Bad rank",
                    "description": "Desc",
                    "rank_position": "after:",
                }),
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert "warning" in data["rank_result"]
