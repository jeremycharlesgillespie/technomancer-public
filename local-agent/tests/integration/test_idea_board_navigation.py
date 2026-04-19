"""Integration tests for home card and execution navigation flow.

Verifies:
- Home page (/) renders the "View Live Executions" card linking to /live
- /live renders execution list with hrefs to /live/<key>
- /live/<idea_id> renders without error
- All tests complete in <5 seconds (no blocking subprocess/network calls)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app, get_execution_detail_href


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Flask test client with TESTING mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_subprocess_mock():
    """Return a mock subprocess.CompletedProcess-like result for git calls."""
    result = MagicMock()
    result.stdout = "abc1234\n"
    result.returncode = 0
    return result


def _raise_oserror(*args, **kwargs):
    raise OSError("mocked: no network in tests")


@pytest.fixture(autouse=True)
def _mock_blocking_calls(monkeypatch):
    """Prevent network and subprocess calls that would slow tests.

    The hub renderer makes two slow calls:
    - subprocess.run(['git', 'rev-parse', ...]) — has a 5s timeout
    - urllib.request.urlopen(discord_bridge_url, timeout=2) — connection attempt

    Both are wrapped in try/except in production; raising here exercises
    the fallback paths without incurring real latency.
    """
    git_mock = _make_subprocess_mock()
    monkeypatch.setattr("idea_board.web.subprocess.run", lambda *a, **kw: git_mock)
    monkeypatch.setattr("urllib.request.urlopen", _raise_oserror)


# ---------------------------------------------------------------------------
# Home page — View Live card
# ---------------------------------------------------------------------------


class TestHomePageLiveCard:
    """The hub home page must render the View Live Executions card."""

    def test_home_returns_200(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_renders_view_live_card(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        html = resp.data.decode()
        assert "/live" in html
        assert "View Live" in html

    def test_home_live_card_has_correct_href(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        html = resp.data.decode()
        assert 'href="/live"' in html

    def test_home_content_type_is_html(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        assert "text/html" in resp.content_type


# ---------------------------------------------------------------------------
# /live — execution list page
# ---------------------------------------------------------------------------


class TestLivePage:
    """The /live landing page renders execution list with navigable hrefs."""

    def test_live_returns_200(self, client):
        empty_data = {"executing": [], "recent": []}
        with patch("idea_board.web._collect_live_executions", return_value=empty_data):
            resp = client.get("/live")
        assert resp.status_code == 200

    def test_live_content_type_is_html(self, client):
        empty_data = {"executing": [], "recent": []}
        with patch("idea_board.web._collect_live_executions", return_value=empty_data):
            resp = client.get("/live")
        assert "text/html" in resp.content_type

    def test_live_empty_state_renders_no_executions_message(self, client):
        empty_data = {"executing": [], "recent": []}
        with patch("idea_board.web._collect_live_executions", return_value=empty_data):
            resp = client.get("/live")
        html = resp.data.decode()
        assert "No executions in flight" in html

    def test_live_executing_row_renders_href(self, client):
        """An executing story with a key gets a /live/<key> link in the table."""
        data = {
            "executing": [
                {
                    "project": "TK",
                    "key": "TK-101",
                    "status": "executing",
                    "started_at": "2026-04-19T10:00:00",
                    "last_observation": "Running tests",
                }
            ],
            "recent": [],
        }
        with patch("idea_board.web._collect_live_executions", return_value=data):
            with patch("idea_board.web._live_route_accessible", return_value=True):
                resp = client.get("/live")
        html = resp.data.decode()
        assert 'href="/live/TK-101"' in html
        assert "TK-101" in html

    def test_live_recent_row_renders_href(self, client):
        """A recently completed story gets a /live/<key> link in the table."""
        data = {
            "executing": [],
            "recent": [
                {
                    "project": "TK",
                    "key": "TK-55",
                    "title": "Add retry logic",
                    "resolved": "2026-04-18T09:30:00",
                }
            ],
        }
        with patch("idea_board.web._collect_live_executions", return_value=data):
            with patch("idea_board.web._live_route_accessible", return_value=True):
                resp = client.get("/live")
        html = resp.data.decode()
        assert 'href="/live/TK-55"' in html
        assert "TK-55" in html

    def test_live_row_without_accessible_key_omits_href(self, client):
        """When _live_route_accessible returns False, the key is rendered as plain text."""
        data = {
            "executing": [
                {
                    "project": "TK",
                    "key": "TK-200",
                    "status": "executing",
                    "started_at": "2026-04-19T10:00:00",
                    "last_observation": "",
                }
            ],
            "recent": [],
        }
        with patch("idea_board.web._collect_live_executions", return_value=data):
            with patch("idea_board.web._live_route_accessible", return_value=False):
                resp = client.get("/live")
        html = resp.data.decode()
        assert "TK-200" in html
        assert 'href="/live/TK-200"' not in html


# ---------------------------------------------------------------------------
# /live/<idea_id> — per-story log viewer
# ---------------------------------------------------------------------------


class TestLiveLogViewer:
    """The /live/<idea_id> route renders the log viewer without errors."""

    def test_live_local_id_returns_200(self, client):
        resp = client.get("/live/idea-123")
        assert resp.status_code == 200

    def test_live_jira_key_returns_200(self, client):
        resp = client.get("/live/TK-42")
        assert resp.status_code == 200

    def test_live_log_viewer_contains_item_id(self, client):
        resp = client.get("/live/TK-99")
        html = resp.data.decode()
        assert "TK-99" in html

    def test_live_log_viewer_content_type_is_html(self, client):
        resp = client.get("/live/TK-99")
        assert "text/html" in resp.content_type

    def test_live_log_viewer_alphanumeric_id(self, client):
        """Any alphanumeric idea_id renders without error."""
        resp = client.get("/live/idea-abc-456")
        assert resp.status_code == 200
        assert "idea-abc-456" in resp.data.decode()


# ---------------------------------------------------------------------------
# get_execution_detail_href unit tests
# ---------------------------------------------------------------------------


class TestGetExecutionDetailHref:
    """Unit tests for the get_execution_detail_href helper."""

    def test_valid_jira_key_returns_live_href(self):
        assert get_execution_detail_href("TK-123") == "/live/TK-123"

    def test_valid_local_id_returns_live_href(self):
        assert get_execution_detail_href("idea-42") == "/live/idea-42"

    def test_none_returns_empty_string(self):
        assert get_execution_detail_href(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert get_execution_detail_href("") == ""

    def test_whitespace_only_returns_empty_string(self):
        assert get_execution_detail_href("   ") == ""
