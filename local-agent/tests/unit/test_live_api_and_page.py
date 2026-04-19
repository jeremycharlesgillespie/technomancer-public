"""Tests for /api/live JSON endpoint and /live auto-refresh wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    fake_git = MagicMock(stdout="abc1234\n", returncode=0)
    monkeypatch.setattr("idea_board.web.subprocess.run", lambda *a, **kw: fake_git)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("no network")),
    )


class TestApiLive:
    def test_returns_json_with_executing_and_recent(self, client):
        data = {
            "executing": [
                {
                    "project": "TK",
                    "key": "TK-42",
                    "status": "executing",
                    "started_at": "2026-04-19T10:00:00",
                    "last_observation": "working",
                }
            ],
            "recent": [
                {
                    "project": "TK",
                    "key": "TK-41",
                    "title": "Prev story",
                    "resolved": "2026-04-19T09:00:00",
                }
            ],
        }
        with patch("idea_board.web._collect_live_executions", return_value=data), \
             patch("idea_board.web._live_route_accessible", return_value=True):
            resp = client.get("/api/live")
        assert resp.status_code == 200
        j = resp.get_json()
        assert len(j["executing"]) == 1
        assert j["executing"][0]["key"] == "TK-42"
        assert j["executing"][0]["log_href"] == "/live/TK-42"
        assert len(j["recent"]) == 1
        assert j["recent"][0]["log_href"] == "/live/TK-41"
        assert "generated_at" in j

    def test_log_href_null_when_no_log_on_disk(self, client):
        data = {
            "executing": [{
                "project": "TK", "key": "TK-99", "status": "assigned",
                "started_at": "", "last_observation": "",
            }],
            "recent": [],
        }
        with patch("idea_board.web._collect_live_executions", return_value=data), \
             patch("idea_board.web._live_route_accessible", return_value=False):
            resp = client.get("/api/live")
        j = resp.get_json()
        assert j["executing"][0]["log_href"] is None

    def test_empty_when_nothing_executing(self, client):
        with patch("idea_board.web._collect_live_executions",
                   return_value={"executing": [], "recent": []}):
            resp = client.get("/api/live")
        j = resp.get_json()
        assert j["executing"] == []
        assert j["recent"] == []


class TestLivePagePollScript:
    def test_live_page_includes_polling_script(self, client):
        with patch("idea_board.web._collect_live_executions",
                   return_value={"executing": [], "recent": []}):
            resp = client.get("/live")
        body = resp.get_data(as_text=True)
        assert "/api/live" in body
        assert "setInterval" in body
        assert 'id="executing-body"' in body
        assert 'id="recent-body"' in body
        assert 'id="live-indicator"' in body
