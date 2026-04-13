"""Tests for hub action endpoints — restart, cleanup, github-sync."""

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture()
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestActionRestart:
    """POST /api/actions/restart — restart the bot."""

    def test_restart_success(self, client):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Bot started successfully (PID: 1234)"
        mock_result.stderr = ""

        with patch("idea_board.web.subprocess.run", return_value=mock_result) as mock_run:
            resp = client.post("/api/actions/restart")

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["action"] == "restart"
        assert data["success"] is True
        assert "Bot started" in data["output"]

        # Verify it called bot_service.py start
        args = mock_run.call_args
        assert "bot_service.py" in args[0][0][1]
        assert "start" in args[0][0][2]

    def test_restart_failure(self, client):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Failed: Process exited with code 1"

        with patch("idea_board.web.subprocess.run", return_value=mock_result):
            resp = client.post("/api/actions/restart")

        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert data["success"] is False

    def test_restart_timeout(self, client):
        import subprocess

        with patch("idea_board.web.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 60)):
            resp = client.post("/api/actions/restart")

        assert resp.status_code == 504
        data = json.loads(resp.data)
        assert data["success"] is False
        assert "Timed out" in data["output"]

    def test_restart_only_post(self, client):
        resp = client.get("/api/actions/restart")
        assert resp.status_code == 405


class TestActionCleanup:
    """POST /api/actions/cleanup — run cleanup.py."""

    def test_cleanup_success(self, client):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Killed 2 orphans\nBot restarted"
        mock_result.stderr = ""

        with patch("idea_board.web.subprocess.run", return_value=mock_result) as mock_run:
            resp = client.post("/api/actions/cleanup")

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["action"] == "cleanup"
        assert data["success"] is True

        args = mock_run.call_args
        assert "cleanup.py" in args[0][0][1]

    def test_cleanup_failure(self, client):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error during cleanup"

        with patch("idea_board.web.subprocess.run", return_value=mock_result):
            resp = client.post("/api/actions/cleanup")

        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert data["success"] is False

    def test_cleanup_timeout(self, client):
        import subprocess

        with patch("idea_board.web.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 120)):
            resp = client.post("/api/actions/cleanup")

        assert resp.status_code == 504
        data = json.loads(resp.data)
        assert data["success"] is False
        assert "Timed out" in data["output"]


class TestActionGithubSync:
    """POST /api/actions/github-sync — trigger sync_all_projects."""

    def test_github_sync_success(self, client):
        mock_results = {"technomancer": True, "other-project": True}

        with patch("idea_board.web.asyncio.new_event_loop") as mock_loop_factory:
            mock_loop = MagicMock()
            mock_loop.run_until_complete.return_value = mock_results
            mock_loop_factory.return_value = mock_loop

            with patch("agent.project_tracker.sync_all_projects") as mock_sync:
                resp = client.post("/api/actions/github-sync")

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["action"] == "github-sync"
        assert data["success"] is True
        assert "2/2" in data["output"]
        assert data["details"] == mock_results

    def test_github_sync_partial(self, client):
        mock_results = {"technomancer": True, "broken-repo": False}

        with patch("idea_board.web.asyncio.new_event_loop") as mock_loop_factory:
            mock_loop = MagicMock()
            mock_loop.run_until_complete.return_value = mock_results
            mock_loop_factory.return_value = mock_loop

            with patch("agent.project_tracker.sync_all_projects"):
                resp = client.post("/api/actions/github-sync")

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
        assert "1/2" in data["output"]

    def test_github_sync_import_error(self, client):
        with patch("idea_board.web.asyncio.new_event_loop") as mock_loop_factory:
            mock_loop = MagicMock()
            mock_loop.run_until_complete.side_effect = Exception("No GITHUB_TOKEN configured")
            mock_loop_factory.return_value = mock_loop

            with patch.dict("sys.modules", {"agent.project_tracker": MagicMock()}):
                resp = client.post("/api/actions/github-sync")

        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert data["success"] is False


class TestHubRendersActions:
    """GET / — hub page includes action buttons."""

    def test_hub_has_action_buttons(self, client):
        with patch("idea_board.web.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="abc1234", stderr="", returncode=0)
            resp = client.get("/")

        html = resp.data.decode()
        assert "Quick Actions" in html
        assert "action-restart" in html
        assert "action-cleanup" in html
        assert "action-github-sync" in html
        assert "runAction" in html

    def test_restart_has_confirm(self, client):
        with patch("idea_board.web.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="abc1234", stderr="", returncode=0)
            resp = client.get("/")

        html = resp.data.decode()
        # Restart button passes true for needsConfirm
        assert "runAction('restart', true)" in html
        # Others don't
        assert "runAction('cleanup')" in html
        assert "runAction('github-sync')" in html
