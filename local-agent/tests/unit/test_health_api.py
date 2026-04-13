"""
Tests for the /api/health endpoint in idea_board/web.py.

Validates the aggregated health check returns correct structure
and handles service up/down states properly.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_returns_200(self, client):
        """Health endpoint always returns 200 even when services are down."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            assert resp.status_code == 200

    def test_response_structure(self, client):
        """Response has overall status, services list, and timestamp."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            assert "overall" in data
            assert "services" in data
            assert "checked_at" in data
            assert isinstance(data["services"], list)

    def test_all_four_services_present(self, client):
        """Should check bot, ollama, bridge, and idea_board."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            ids = [s["id"] for s in data["services"]]
            assert ids == ["bot", "ollama", "bridge", "idea_board"]

    def test_each_service_has_required_fields(self, client):
        """Every service entry must have name, id, healthy, status."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            for svc in data["services"]:
                assert "name" in svc
                assert "id" in svc
                assert "healthy" in svc
                assert "status" in svc

    def test_idea_board_always_healthy(self, client):
        """Idea board is healthy if we can respond (self-check)."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            board = next(s for s in data["services"] if s["id"] == "idea_board")
            assert board["healthy"] is True
            assert board["idea_count"] == 0

    def test_overall_degraded_when_service_down(self, client):
        """Overall status is 'degraded' when any service is unhealthy."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            # Bot, Ollama, Bridge are all down (mocked failures)
            assert data["overall"] == "degraded"

    def test_bot_healthy_when_pid_alive(self, client, tmp_path):
        """Bot shows healthy when PID file exists and process is alive."""
        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("12345")
        state_file = tmp_path / "service_state.json"
        state_file.write_text(json.dumps({"total_restarts": 3, "consecutive_failures": 0}))

        def mock_exists(self_path):
            name = self_path.name
            if name == "bot.pid":
                return True
            if name == "service_state.json":
                return True
            return Path.exists(self_path)

        def mock_read_text(self_path, *args, **kwargs):
            name = self_path.name
            if name == "bot.pid":
                return "12345"
            if name == "service_state.json":
                return json.dumps({"total_restarts": 3, "consecutive_failures": 0})
            return Path.read_text(self_path, *args, **kwargs)

        with patch.object(Path, "exists", mock_exists), \
             patch.object(Path, "read_text", mock_read_text), \
             patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            # tasklist returns the PID in output
            mock_run.return_value = MagicMock(stdout="python.exe  12345 Console")
            resp = client.get("/api/health")
            data = resp.get_json()
            bot = next(s for s in data["services"] if s["id"] == "bot")
            assert bot["healthy"] is True
            assert bot["status"] == "running"
            assert bot["restarts"] == 3

    def test_bot_stopped_when_no_pid_file(self, client):
        """Bot shows stopped when PID file doesn't exist."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            bot = next(s for s in data["services"] if s["id"] == "bot")
            assert bot["healthy"] is False
            assert bot["status"] == "stopped"

    def test_ollama_healthy_with_models(self, client):
        """Ollama shows healthy with model count when API responds."""
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.read.return_value = json.dumps({
            "models": [{"name": "qwen3.5:27b"}, {"name": "llava-llama3"}]
        }).encode()
        mock_ollama_resp.__enter__ = lambda s: s
        mock_ollama_resp.__exit__ = MagicMock(return_value=False)

        # Bridge will fail
        call_count = [0]
        def mock_urlopen(url, **kwargs):
            call_count[0] += 1
            if "11434" in str(url):
                return mock_ollama_resp
            raise Exception("conn refused")

        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=mock_urlopen), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            ollama = next(s for s in data["services"] if s["id"] == "ollama")
            assert ollama["healthy"] is True
            assert "2 models" in ollama["status"]

    def test_bridge_healthy_with_uptime(self, client):
        """Bridge shows healthy with uptime when API responds."""
        mock_bridge_resp = MagicMock()
        mock_bridge_resp.read.return_value = json.dumps({
            "status": "ok", "uptime": 3661
        }).encode()
        mock_bridge_resp.__enter__ = lambda s: s
        mock_bridge_resp.__exit__ = MagicMock(return_value=False)

        def mock_urlopen(url, **kwargs):
            if "8321" in str(url):
                return mock_bridge_resp
            raise Exception("conn refused")

        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("idea_board.web.Path.exists", return_value=False), \
             patch("urllib.request.urlopen", side_effect=mock_urlopen), \
             patch("idea_board.web.load_ideas", return_value=[]):
            mock_run.return_value = MagicMock(stdout="")
            resp = client.get("/api/health")
            data = resp.get_json()
            bridge = next(s for s in data["services"] if s["id"] == "bridge")
            assert bridge["healthy"] is True
            assert bridge["uptime_seconds"] == 3661

    def test_hub_page_contains_health_panel(self, client):
        """The hub HTML includes the health panel div."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")):
            mock_run.return_value = MagicMock(stdout="abc1234")
            resp = client.get("/")
            html = resp.data.decode()
            assert "health-panel" in html
            assert "health-grid" in html
            assert "health-bot" in html
            assert "health-ollama" in html
            assert "health-bridge" in html
            assert "health-idea_board" in html
            assert "updateHealth" in html
            assert "setInterval(updateHealth, 10000)" in html
