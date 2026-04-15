"""Tests for board.jira_provider — JiraProvider backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_jira_configured(monkeypatch):
    """Pretend Jira is configured in every test."""
    monkeypatch.setattr("idea_board.jira_sync.is_jira_configured", lambda: True)
    monkeypatch.setattr("board.jira_provider.is_jira_configured", lambda: True)


@pytest.fixture
def mock_api():
    """Mock board.jira_provider._api — covers all HTTP calls inside the module."""
    with patch("board.jira_provider._api") as m:
        yield m


@pytest.fixture
def provider():
    from board.jira_provider import JiraProvider
    return JiraProvider()


def _issue(key="TK-1", summary="Add caching layer", status="To Do",
           issuetype="Story", labels=None, created="2026-04-15T10:00:00.000+0000",
           parent=None, description="Speed up responses"):
    fields = {
        "summary": summary,
        "status": {"name": status},
        "issuetype": {"name": issuetype},
        "labels": labels if labels is not None else ["cat:performance", "src:llm_analysis"],
        "created": created,
        "description": {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
        },
    }
    if parent:
        fields["parent"] = {"key": parent}
    return {"key": key, "fields": fields}


def _search_response(issues):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"issues": issues}
    return resp


# ---------------------------------------------------------------------------
# State derivation
# ---------------------------------------------------------------------------

class TestStateDerivation:
    def test_in_progress_maps_to_executing(self):
        from board.jira_provider import _jira_status_to_state
        assert _jira_status_to_state("In Progress", []) == "executing"

    def test_done_maps_to_done(self):
        from board.jira_provider import _jira_status_to_state
        assert _jira_status_to_state("Done", []) == "done"

    def test_failed_maps_to_failed(self):
        from board.jira_provider import _jira_status_to_state
        assert _jira_status_to_state("Failed", []) == "failed"

    def test_to_do_without_gate_is_approved(self):
        from board.jira_provider import _jira_status_to_state
        assert _jira_status_to_state("To Do", ["cat:quality"]) == "approved"

    def test_to_do_with_pending_approval_is_proposed(self):
        from board.jira_provider import _jira_status_to_state
        assert _jira_status_to_state("To Do", ["pending-approval"]) == "proposed"


# ---------------------------------------------------------------------------
# Issue → BoardItem
# ---------------------------------------------------------------------------

class TestIssueToItem:
    def test_parses_core_fields(self):
        from board.jira_provider import _issue_to_item
        item = _issue_to_item(_issue(key="TK-42"))
        assert item.id == "TK-42"
        assert item.title == "Add caching layer"
        assert item.state == "approved"
        assert item.category == "performance"
        assert item.source == "llm_analysis"
        assert "Speed up responses" in item.description

    def test_strips_idea_id_prefix_from_summary(self):
        from board.jira_provider import _issue_to_item
        item = _issue_to_item(_issue(summary="[idea-225] Knowledge remediation"))
        assert item.title == "Knowledge remediation"

    def test_epic_issuetype_sets_idea_type(self):
        from board.jira_provider import _issue_to_item
        item = _issue_to_item(_issue(issuetype="Epic"))
        assert item.idea_type == "epic"

    def test_parent_link_populates_parent_id(self):
        from board.jira_provider import _issue_to_item
        item = _issue_to_item(_issue(parent="TK-5"))
        assert item.parent_id == "TK-5"


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

class TestReads:
    def test_load_all_maps_every_issue(self, provider, mock_api):
        mock_api.return_value = _search_response([_issue("TK-1"), _issue("TK-2")])
        items = provider.load_all()
        assert [i.id for i in items] == ["TK-1", "TK-2"]

    def test_list_by_state_approved_filters_out_pending(self, provider, mock_api):
        mock_api.return_value = _search_response([
            _issue("TK-1", labels=["cat:quality"]),
            _issue("TK-2", labels=["cat:feature", "pending-approval"]),
        ])
        items = provider.list_by_state("approved")
        assert [i.id for i in items] == ["TK-1"]

    def test_list_by_state_proposed_filters_to_pending(self, provider, mock_api):
        mock_api.return_value = _search_response([
            _issue("TK-1", labels=["cat:quality"]),
            _issue("TK-2", labels=["cat:feature", "pending-approval"]),
        ])
        items = provider.list_by_state("proposed")
        assert [i.id for i in items] == ["TK-2"]

    def test_get_returns_board_item_on_200(self, provider, mock_api):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _issue("TK-99")
        mock_api.return_value = resp
        item = provider.get("TK-99")
        assert item is not None
        assert item.id == "TK-99"

    def test_get_returns_none_on_404(self, provider, mock_api):
        resp = MagicMock()
        resp.status_code = 404
        mock_api.return_value = resp
        assert provider.get("TK-doesnt-exist") is None


# ---------------------------------------------------------------------------
# Writes — add / vote / transitions
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_creates_and_labels(self, provider, mock_api):
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = _issue(
            "TK-500", labels=["cat:quality", "src:aim_manager", "type:story"]
        )
        mock_api.return_value = get_resp

        with patch("board.jira_provider.create_jira_issue", return_value="TK-500") as create_mock:
            item = provider.add("New quality thing", "Desc", source="aim_manager",
                                 category="quality", idea_type="story")

        assert item.id == "TK-500"
        assert item.category == "quality"
        labels = create_mock.call_args.kwargs["labels"]
        assert "cat:quality" in labels
        assert "src:aim_manager" in labels
        assert "type:story" in labels
        assert "pending-approval" not in labels  # quality is safe

    def test_add_gates_non_safe_categories_with_pending_approval(self, provider, mock_api):
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = _issue(
            "TK-501", labels=["cat:feature", "src:llm_analysis", "type:story", "pending-approval"]
        )
        mock_api.return_value = get_resp

        with patch("board.jira_provider.create_jira_issue", return_value="TK-501") as create_mock:
            item = provider.add("Shiny feature", "Desc", category="feature")

        assert item.state == "proposed"
        assert "pending-approval" in create_mock.call_args.kwargs["labels"]


class TestVote:
    def test_owner_approve_removes_pending_label(self, provider, mock_api):
        put_resp = MagicMock()
        put_resp.status_code = 204
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = _issue("TK-1", labels=["cat:feature"])
        mock_api.side_effect = [put_resp, get_resp]

        provider.vote("TK-1", "owner", "approve")

        put_call = mock_api.call_args_list[0]
        assert put_call.args[0] == "put"
        assert "/issue/TK-1" in put_call.args[1]
        assert {"remove": "pending-approval"} in put_call.kwargs["json"]["update"]["labels"]

    def test_owner_veto_transitions_to_failed(self, provider, mock_api):
        # add-label PUT then final get()
        put_resp = MagicMock()
        put_resp.status_code = 204
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = _issue("TK-1", status="Failed", labels=["vetoed"])
        mock_api.side_effect = [put_resp, get_resp]

        with patch("board.jira_provider.transition_jira_issue", return_value=True) as t:
            provider.vote("TK-1", "owner", "veto")

        add_call = mock_api.call_args_list[0]
        assert {"add": "vetoed"} in add_call.kwargs["json"]["update"]["labels"]
        t.assert_called_once_with("TK-1", "Failed")


class TestTransitions:
    def test_mark_executing_transitions_to_in_progress(self, provider, mock_api):
        final_get = MagicMock(status_code=200)
        final_get.json.return_value = _issue("TK-1", status="In Progress")
        mock_api.return_value = final_get

        with patch("board.jira_provider.transition_jira_issue", return_value=True) as t:
            item = provider.mark_executing("TK-1")

        t.assert_called_once_with("TK-1", "In Progress")
        assert item.state == "executing"

    def test_mark_done_posts_log_then_transitions(self, provider, mock_api):
        post_comment = MagicMock()
        post_comment.status_code = 201
        post_comment.json.return_value = {"id": "12345"}
        final_get = MagicMock(status_code=200)
        final_get.json.return_value = _issue("TK-1", status="Done")
        mock_api.side_effect = [post_comment, final_get]

        with patch("board.jira_provider.transition_jira_issue", return_value=True) as t:
            item = provider.mark_done("TK-1", "execution log body")

        first_call = mock_api.call_args_list[0]
        assert first_call.args[0] == "post"
        assert "/issue/TK-1/comment" in first_call.args[1]
        t.assert_called_once_with("TK-1", "Done")
        assert item.state == "done"

    def test_mark_failed_posts_error_then_transitions(self, provider, mock_api):
        post_comment = MagicMock(status_code=201)
        post_comment.json.return_value = {"id": "999"}
        final_get = MagicMock(status_code=200)
        final_get.json.return_value = _issue("TK-1", status="Failed")
        mock_api.side_effect = [post_comment, final_get]

        with patch("board.jira_provider.transition_jira_issue", return_value=True) as t:
            item = provider.mark_failed("TK-1", "boom")

        t.assert_called_once_with("TK-1", "Failed")
        assert item.state == "failed"


# ---------------------------------------------------------------------------
# Progress comment (edit in place)
# ---------------------------------------------------------------------------

class TestAppendProgressComment:
    def test_first_call_posts_comment_and_caches_id(self, provider, mock_api):
        post_resp = MagicMock(status_code=201)
        post_resp.json.return_value = {"id": "7777"}
        mock_api.return_value = post_resp

        provider.append_progress_comment("TK-1", "line 1\nline 2")

        assert provider._progress_comment_ids["TK-1"] == 7777
        call = mock_api.call_args
        assert call.args[0] == "post"
        assert "/issue/TK-1/comment" in call.args[1]

    def test_second_call_edits_cached_comment(self, provider, mock_api):
        post_resp = MagicMock(status_code=201)
        post_resp.json.return_value = {"id": "7777"}
        put_resp = MagicMock(status_code=200)
        mock_api.side_effect = [post_resp, put_resp]

        provider.append_progress_comment("TK-1", "one")
        provider.append_progress_comment("TK-1", "one\ntwo")

        put_call = mock_api.call_args_list[1]
        assert put_call.args[0] == "put"
        assert "/issue/TK-1/comment/7777" in put_call.args[1]

    def test_edit_failure_falls_back_to_fresh_post(self, provider, mock_api):
        first_post = MagicMock(status_code=201)
        first_post.json.return_value = {"id": "7777"}
        failed_edit = MagicMock(status_code=404)
        new_post = MagicMock(status_code=201)
        new_post.json.return_value = {"id": "9999"}
        mock_api.side_effect = [first_post, failed_edit, new_post]

        provider.append_progress_comment("TK-1", "one")
        provider.append_progress_comment("TK-1", "two")

        assert provider._progress_comment_ids["TK-1"] == 9999


# ---------------------------------------------------------------------------
# Provider interface compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_is_a_board_provider(self):
        from board import BoardProvider
        from board.jira_provider import JiraProvider
        assert isinstance(JiraProvider(), BoardProvider)
