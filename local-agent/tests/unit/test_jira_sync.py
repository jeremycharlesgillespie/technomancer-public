"""Tests for idea_board.jira_sync — Jira integration."""

import json
import sqlite3
from unittest.mock import MagicMock, patch, call
import pytest
import requests

from idea_board import jira_sync_dlq
from idea_board.jira_sync import (
    is_jira_configured,
    find_jira_issue,
    create_jira_issue,
    transition_jira_issue,
    sync_idea_to_jira,
    TYPE_MAP,
    STATE_MAP,
    _build_description_adf,
    _post_with_retry,
    _write_deadletter,
    _backoff_for,
    _retry_after_seconds,
    JiraRetryExhausted,
    MAX_RETRY_ATTEMPTS,
    BACKOFF_SECONDS,
    RETRY_STATUS_CODES,
)


@pytest.fixture(autouse=True)
def _isolate_dlq_db(tmp_path, monkeypatch):
    """Redirect the jira_sync_dlq SQLite DB at a temp path for each test.

    jira_sync now writes to the DLQ from the permanent-4xx branch of
    ``create_jira_issue`` and the JiraRetryExhausted handler of
    ``sync_idea_to_jira``. Without this fixture those writes would land
    in the real ``data/jira_sync_dlq.db`` during the test run.
    """
    db_path = tmp_path / "jira_sync_dlq.db"
    monkeypatch.setattr(jira_sync_dlq, "DB_DIR", tmp_path)
    monkeypatch.setattr(jira_sync_dlq, "DB_PATH", db_path)
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    jira_sync_dlq._local.__dict__.pop("conn", None)
    yield
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        jira_sync_dlq._local.__dict__.pop("conn", None)


def _resp(status: int, text: str = "", headers: dict | None = None):
    """Build a MagicMock imitating ``requests.Response``."""
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = text
    r.headers = headers or {}
    return r


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


# ---------------------------------------------------------------------------
# _post_with_retry — retry policy on 429 / 5xx
# ---------------------------------------------------------------------------


class TestPostWithRetry:
    @patch("idea_board.jira_sync._api")
    def test_success_on_first_try_returns_response(self, mock_api):
        mock_api.return_value = _resp(201, text="{}")
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {"x": 1}, sleep=sleeps.append)

        assert resp.status_code == 201
        assert mock_api.call_count == 1
        assert sleeps == []

    @patch("idea_board.jira_sync._api")
    def test_retries_on_429_then_succeeds(self, mock_api):
        """A 429 followed by a 200 should return the 200 and sleep once."""
        mock_api.side_effect = [_resp(429, headers={"Retry-After": "2"}), _resp(200)]
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert resp.status_code == 200
        assert mock_api.call_count == 2
        assert sleeps == [2.0]  # honoured Retry-After

    @patch("idea_board.jira_sync._api")
    def test_429_without_retry_after_uses_backoff(self, mock_api):
        mock_api.side_effect = [_resp(429), _resp(200)]
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert resp.status_code == 200
        # Falls back to the first backoff entry
        assert sleeps == [BACKOFF_SECONDS[0]]

    @patch("idea_board.jira_sync._api")
    def test_429_with_invalid_retry_after_uses_backoff(self, mock_api):
        mock_api.side_effect = [
            _resp(429, headers={"Retry-After": "not-a-number"}),
            _resp(200),
        ]
        sleeps: list[float] = []

        _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert sleeps == [BACKOFF_SECONDS[0]]

    @patch("idea_board.jira_sync._api")
    def test_retries_on_5xx_then_succeeds(self, mock_api):
        mock_api.side_effect = [_resp(503), _resp(502), _resp(201)]
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert resp.status_code == 201
        assert mock_api.call_count == 3
        assert sleeps == [BACKOFF_SECONDS[0], BACKOFF_SECONDS[1]]

    @patch("idea_board.jira_sync._api")
    def test_500_all_attempts_raises_retry_exhausted(self, mock_api):
        """Five 500s must raise JiraRetryExhausted (one per attempt)."""
        mock_api.side_effect = [_resp(500, text=f"err{i}") for i in range(5)]
        sleeps: list[float] = []

        with pytest.raises(JiraRetryExhausted) as exc_info:
            _post_with_retry("/issue", {}, sleep=sleeps.append)

        err = exc_info.value
        assert err.attempts == MAX_RETRY_ATTEMPTS == 5
        assert err.status == 500
        assert "err4" in err.body  # body of the last attempt preserved
        assert err.path == "/issue"
        # Four sleeps between five attempts, no sleep after the last
        assert len(sleeps) == 4
        assert mock_api.call_count == 5

    @patch("idea_board.jira_sync._api")
    def test_retries_on_requests_exception_then_succeeds(self, mock_api):
        """Transient connection error should be retried, then succeed."""
        mock_api.side_effect = [
            requests.ConnectionError("boom"),
            _resp(200),
        ]
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert resp.status_code == 200
        assert sleeps == [BACKOFF_SECONDS[0]]

    @patch("idea_board.jira_sync._api")
    def test_all_network_errors_raises_exhausted(self, mock_api):
        mock_api.side_effect = [
            requests.ConnectionError(f"boom-{i}") for i in range(5)
        ]
        sleeps: list[float] = []

        with pytest.raises(JiraRetryExhausted) as exc_info:
            _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert exc_info.value.status is None
        assert "boom-4" in exc_info.value.body
        assert mock_api.call_count == 5

    @patch("idea_board.jira_sync._api")
    def test_4xx_other_than_429_returned_without_retry(self, mock_api):
        """404/400 etc. are caller errors — don't retry, let caller handle."""
        mock_api.return_value = _resp(404, text="Not Found")
        sleeps: list[float] = []

        resp = _post_with_retry("/issue", {}, sleep=sleeps.append)

        assert resp.status_code == 404
        assert mock_api.call_count == 1
        assert sleeps == []

    @patch("idea_board.jira_sync._api")
    def test_max_attempts_override(self, mock_api):
        mock_api.side_effect = [_resp(500)] * 2
        sleeps: list[float] = []

        with pytest.raises(JiraRetryExhausted) as exc_info:
            _post_with_retry("/issue", {}, max_attempts=2, sleep=sleeps.append)

        assert exc_info.value.attempts == 2
        assert mock_api.call_count == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestBackoffFor:
    def test_first_attempt_uses_first_backoff(self):
        assert _backoff_for(1) == BACKOFF_SECONDS[0]

    def test_caps_at_last_entry(self):
        assert _backoff_for(100) == BACKOFF_SECONDS[-1]

    def test_zero_and_negative_clamped_to_first(self):
        assert _backoff_for(0) == BACKOFF_SECONDS[0]
        assert _backoff_for(-5) == BACKOFF_SECONDS[0]


class TestRetryAfterSeconds:
    def test_valid_integer_seconds(self):
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "5"})) == 5.0

    def test_missing_header_returns_none(self):
        assert _retry_after_seconds(_resp(429)) is None

    def test_invalid_header_returns_none(self):
        r = _resp(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
        assert _retry_after_seconds(r) is None

    def test_negative_clamped_to_zero(self):
        assert _retry_after_seconds(_resp(429, headers={"Retry-After": "-3"})) == 0.0


# ---------------------------------------------------------------------------
# Dead-letter writer
# ---------------------------------------------------------------------------


class TestWriteDeadletter:
    def test_writes_one_json_line(self, tmp_path):
        dl = tmp_path / "logs" / "jira_sync_deadletter.jsonl"

        _write_deadletter("idea-001", "Done", "boom", path=dl)

        lines = dl.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["idea_id"] == "idea-001"
        assert entry["target_state"] == "Done"
        assert entry["last_error"] == "boom"
        assert "timestamp" in entry

    def test_appends_on_second_call(self, tmp_path):
        dl = tmp_path / "jira_sync_deadletter.jsonl"

        _write_deadletter("idea-1", "To Do", "e1", path=dl)
        _write_deadletter("idea-2", "Done", "e2", path=dl)

        lines = dl.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["idea_id"] == "idea-1"
        assert json.loads(lines[1])["idea_id"] == "idea-2"

    def test_creates_parent_directory(self, tmp_path):
        dl = tmp_path / "nested" / "deep" / "dead.jsonl"

        _write_deadletter("idea-x", "To Do", "fail", path=dl)

        assert dl.exists()

    def test_truncates_very_long_error(self, tmp_path):
        dl = tmp_path / "dead.jsonl"

        _write_deadletter("idea-x", "Done", "x" * 5000, path=dl)

        entry = json.loads(dl.read_text(encoding="utf-8").splitlines()[0])
        assert len(entry["last_error"]) == 2000


# ---------------------------------------------------------------------------
# sync_idea_to_jira — dead-letter on retry exhaustion
# ---------------------------------------------------------------------------


class TestSyncIdeaDeadletter:
    @patch("idea_board.jira_sync._write_deadletter")
    @patch(
        "idea_board.jira_sync.create_jira_issue",
        side_effect=JiraRetryExhausted(500, "oops", 5, path="/issue"),
    )
    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_create_exhaustion_writes_deadletter(
        self, mock_conf, mock_find, mock_create, mock_dl
    ):
        idea = MagicMock(
            id="idea-001", title="T", description="D",
            idea_type="story", state="approved", parent_id=None, category="feature",
        )
        result = sync_idea_to_jira(idea)
        assert result is None
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["idea_id"] == "idea-001"
        assert kwargs["target_state"] == "To Do"
        assert "oops" in kwargs["last_error"] or "500" in kwargs["last_error"]

    @patch("idea_board.jira_sync._write_deadletter")
    @patch(
        "idea_board.jira_sync.transition_jira_issue",
        side_effect=JiraRetryExhausted(503, "down", 5, path="/issue/TK-1/transitions"),
    )
    @patch("idea_board.jira_sync.find_jira_issue", return_value="TK-1")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_transition_exhaustion_writes_deadletter(
        self, mock_conf, mock_find, mock_trans, mock_dl
    ):
        idea = MagicMock(
            id="idea-042", title="T", description="D",
            idea_type="story", state="done", parent_id=None, category="",
        )
        result = sync_idea_to_jira(idea)
        assert result is None
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["idea_id"] == "idea-042"
        assert kwargs["target_state"] == "Done"


# ---------------------------------------------------------------------------
# create_jira_issue / transition_jira_issue propagate JiraRetryExhausted
# ---------------------------------------------------------------------------


class TestWritesPropagateExhaustion:
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_create_propagates_retry_exhausted(
        self, mock_settings, mock_conf, mock_api
    ):
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "localhost"
        mock_api.side_effect = [_resp(500) for _ in range(MAX_RETRY_ATTEMPTS)]

        with patch("idea_board.jira_sync.time.sleep"):
            with pytest.raises(JiraRetryExhausted):
                create_jira_issue("idea-001", "T", "D")

        assert mock_api.call_count == MAX_RETRY_ATTEMPTS

    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    def test_transition_propagates_retry_exhausted(self, mock_conf, mock_api):
        # 1 GET for transitions, then 5 POSTs that all 503
        trans_resp = _resp(200)
        trans_resp.json = MagicMock(
            return_value={"transitions": [{"name": "Done", "id": "31"}]}
        )
        mock_api.side_effect = [trans_resp] + [_resp(503) for _ in range(MAX_RETRY_ATTEMPTS)]

        with patch("idea_board.jira_sync.time.sleep"):
            with pytest.raises(JiraRetryExhausted):
                transition_jira_issue("TK-1", "Done")


# ---------------------------------------------------------------------------
# DLQ writes from permanent-4xx branches
# ---------------------------------------------------------------------------


class TestCreate4xxWritesDlq:
    @patch("idea_board.jira_sync._api")
    @patch("idea_board.jira_sync.find_jira_issue", return_value=None)
    @patch("idea_board.jira_sync.is_jira_configured", return_value=True)
    @patch("idea_board.jira_sync.settings")
    def test_400_lands_one_dlq_row(
        self, mock_settings, mock_conf, mock_find, mock_api
    ):
        """A permanent 400 from Jira lands exactly one DLQ row.

        ``_post_with_retry`` doesn't retry non-429 4xx responses, so
        ``create_jira_issue`` sees the 400 on the first attempt. The
        DLQ row must carry the idea_id and ``attempts=1``.
        """
        mock_settings.jira_project_key = "TK"
        mock_settings.server_host = "localhost"
        mock_api.return_value = _resp(400, text="Bad Request")

        result = create_jira_issue("idea-400", "Broken", "desc")

        assert result is None

        conn = jira_sync_dlq._get_conn()
        rows = conn.execute(
            "SELECT idea_id, attempts, error FROM jira_sync_dlq"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["idea_id"] == "idea-400"
        assert rows[0]["attempts"] == 1
        assert "400" in rows[0]["error"]
