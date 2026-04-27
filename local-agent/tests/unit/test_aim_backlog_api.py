"""Tests for the GET /api/aim/backlog endpoint in idea_board/web.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture(autouse=True)
def _stub_jira_project_key():
    """Force a valid Jira project key for every test in this module.

    The /api/aim/backlog code path resolves the project key via
    ``_jira_project_key_for_project(project)`` which falls back to
    ``settings.jira_project_key``. In the AIW worktree (and on any host
    without a loaded ``.env``) that setting is None, which short-circuits
    the ``if is_jira_configured() and jira_project_key:`` gate and makes
    every "what happens when Jira IS configured" test fail with
    ``in_progress is None``. Patching the resolver itself isolates the
    tests from real env state — the previous individual ``patch(...)``
    blocks already mock ``is_jira_configured`` and ``_jira_api``, so the
    project-key resolver is the last leak.
    """
    with patch("idea_board.web._jira_project_key_for_project", return_value="TK"):
        yield


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _mock_api_response(status_code: int = 200, payload: dict | None = None):
    """Shape a requests.Response-like mock for _jira_api."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {"issues": []}
    return resp


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_returns_200(self, client):
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            resp = client.get("/api/aim/backlog")
            assert resp.status_code == 200

    def test_content_type_is_json(self, client):
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            resp = client.get("/api/aim/backlog")
            assert "application/json" in resp.content_type

    def test_acceptance_shape(self, client):
        """Acceptance: response has counts, in_progress, today.{done,failed}."""
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            data = client.get("/api/aim/backlog").get_json()
            assert "counts" in data
            assert "in_progress" in data
            assert "today" in data
            assert "done" in data["today"]
            assert "failed" in data["today"]


# ---------------------------------------------------------------------------
# Counts passthrough
# ---------------------------------------------------------------------------


class TestCounts:
    def test_counts_are_forwarded_from_jira_reader(self, client):
        fake_counts = {"To Do": 15, "In Progress": 2, "Done": 47, "Failed": 3, "Veto": 1}
        with patch(
            "idea_board.web.aim_jira_reader.count_issues_by_status",
            return_value=fake_counts,
        ), patch("idea_board.web.is_jira_configured", return_value=False):
            data = client.get("/api/aim/backlog").get_json()
            assert data["counts"] == fake_counts

    def test_counts_default_to_empty_when_jira_unavailable(self, client):
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            data = client.get("/api/aim/backlog").get_json()
            assert data["counts"] == {}


# ---------------------------------------------------------------------------
# In-progress lookup
# ---------------------------------------------------------------------------


class TestInProgress:
    def test_in_progress_null_when_jira_not_configured(self, client):
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            data = client.get("/api/aim/backlog").get_json()
            assert data["in_progress"] is None

    def test_in_progress_null_when_no_active_issue(self, client):
        """Idle worker ⇒ no In Progress items ⇒ in_progress is null."""
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch(
                 "idea_board.web._jira_api",
                 return_value=_mock_api_response(payload={"issues": []}),
             ):
            data = client.get("/api/aim/backlog").get_json()
            assert data["in_progress"] is None

    def test_in_progress_populated_when_worker_active(self, client):
        """Worker running ⇒ in_progress has the key and title of that issue."""
        in_progress_payload = {
            "issues": [
                {
                    "key": "TK-415",
                    "fields": {"summary": "Add /api/aim/backlog endpoint"},
                }
            ]
        }

        def fake_api(method, path, **kwargs):
            jql = kwargs.get("json", {}).get("jql", "")
            if "In Progress" in jql and "resolutiondate" not in jql:
                return _mock_api_response(payload=in_progress_payload)
            return _mock_api_response(payload={"issues": []})

        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", side_effect=fake_api):
            data = client.get("/api/aim/backlog").get_json()

        assert data["in_progress"] == {
            "key": "TK-415",
            "title": "Add /api/aim/backlog endpoint",
        }

    def test_in_progress_survives_api_error(self, client):
        """A failed JQL call shouldn't blow up the endpoint."""
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", side_effect=RuntimeError("boom")):
            resp = client.get("/api/aim/backlog")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["in_progress"] is None

    def test_in_progress_null_on_non_200(self, client):
        """A non-200 response leaves in_progress as null."""
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch(
                 "idea_board.web._jira_api",
                 return_value=_mock_api_response(status_code=500),
             ):
            data = client.get("/api/aim/backlog").get_json()
            assert data["in_progress"] is None


# ---------------------------------------------------------------------------
# Today counts
# ---------------------------------------------------------------------------


class TestTodayCounts:
    def test_today_zero_when_jira_not_configured(self, client):
        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            data = client.get("/api/aim/backlog").get_json()
            assert data["today"] == {"done": 0, "failed": 0}

    def test_today_counts_by_status(self, client):
        """Given a mix of Done and Failed issues today, count each."""
        today_payload = {
            "issues": [
                {"fields": {"status": {"name": "Done"}}},
                {"fields": {"status": {"name": "Done"}}},
                {"fields": {"status": {"name": "Done"}}},
                {"fields": {"status": {"name": "Failed"}}},
            ]
        }

        def fake_api(method, path, **kwargs):
            jql = kwargs.get("json", {}).get("jql", "")
            if "resolutiondate" in jql:
                return _mock_api_response(payload=today_payload)
            return _mock_api_response(payload={"issues": []})

        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", side_effect=fake_api):
            data = client.get("/api/aim/backlog").get_json()

        assert data["today"] == {"done": 3, "failed": 1}

    def test_today_jql_uses_startofday(self, client):
        """The JQL must filter resolutiondate >= startOfDay()."""
        captured: dict = {}

        def fake_api(method, path, **kwargs):
            jql = kwargs.get("json", {}).get("jql", "")
            if "resolutiondate" in jql:
                captured["jql"] = jql
            return _mock_api_response(payload={"issues": []})

        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", side_effect=fake_api):
            client.get("/api/aim/backlog")

        assert "resolutiondate >= startOfDay()" in captured["jql"]
        assert '"Done"' in captured["jql"] and '"Failed"' in captured["jql"]

    def test_today_zero_on_api_error(self, client):
        """If the today JQL call raises, both counts default to 0."""

        def fake_api(method, path, **kwargs):
            jql = kwargs.get("json", {}).get("jql", "")
            if "resolutiondate" in jql:
                raise RuntimeError("network down")
            return _mock_api_response(payload={"issues": []})

        with patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=True), \
             patch("idea_board.web._jira_api", side_effect=fake_api):
            resp = client.get("/api/aim/backlog")
            assert resp.status_code == 200
            assert resp.get_json()["today"] == {"done": 0, "failed": 0}
