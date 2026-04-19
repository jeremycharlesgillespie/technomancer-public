"""Tests for the /live landing page (TK-553).

Exercises:
- /live returns HTML aggregating AIM state files across projects
- Executing workers across projects are listed with click-through links
- Recent completions are merged, sorted newest-first, and capped at 10
- Empty states are rendered when no data is available
- The hub home page exposes a "View Live Executions" card linking to /live
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board.web import app, _collect_live_executions


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def fake_agent_root(tmp_path, monkeypatch):
    """Point ``_AGENT_ROOT`` at a fresh throwaway directory per test."""
    (tmp_path / "aim").mkdir()
    (tmp_path / "aim" / "projects").mkdir()
    monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
    return tmp_path


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def _empty_ideas():
    """Keep the hub render path independent of real board data."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


class TestCollectLiveExecutions:
    def test_collects_executing_worker_from_primary(self, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {
                    "status": "executing",
                    "current_idea_id": "TK-553",
                    "started_at": "2026-04-17T16:20:00",
                    "last_observation": "running tests",
                },
                "board_snapshot": {"recent_completions": []},
            },
        )
        with patch("idea_board.web.settings.jira_project_key", "TK"):
            data = _collect_live_executions()

        assert len(data["executing"]) == 1
        row = data["executing"][0]
        assert row["project"] == "TK"
        assert row["key"] == "TK-553"
        assert row["status"] == "executing"

    def test_skips_idle_workers(self, fake_agent_root):
        """Idle workers (no current_idea_id or status=idle) must not appear."""
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {
                    "status": "idle",
                    "current_idea_id": None,
                },
                "board_snapshot": {"recent_completions": []},
            },
        )
        data = _collect_live_executions()
        assert data["executing"] == []

    def test_skips_dead_worker_even_with_assigned_idea(self, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {
                    "status": "dead",
                    "current_idea_id": "TK-999",
                },
                "board_snapshot": {"recent_completions": []},
            },
        )
        data = _collect_live_executions()
        assert data["executing"] == []

    def test_merges_projects_and_sorts_recent_desc(self, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": "TK-1", "summary": "old TK", "resolved": "2026-04-17T10:00:00+0900"},
                        {"key": "TK-2", "summary": "newer TK", "resolved": "2026-04-18T05:00:00+0900"},
                    ]
                },
            },
        )
        _write_state(
            fake_agent_root / "aim" / "projects" / "40acres" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": "FA-1", "summary": "FA item", "resolved": "2026-04-18T06:00:00+0900"},
                    ]
                },
            },
        )
        with patch("idea_board.web.settings.jira_project_key", "TK"):
            data = _collect_live_executions()

        keys = [r["key"] for r in data["recent"]]
        assert keys == ["FA-1", "TK-2", "TK-1"]
        assert data["recent"][0]["project"] == "40acres"

    def test_caps_recent_at_ten(self, fake_agent_root):
        completions = [
            {"key": f"TK-{i}", "summary": f"item {i}", "resolved": f"2026-04-{i:02d}T00:00:00+0900"}
            for i in range(1, 15)
        ]
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {"recent_completions": completions},
            },
        )
        data = _collect_live_executions()
        assert len(data["recent"]) == 10

    def test_ignores_unreadable_state_files(self, fake_agent_root):
        """A malformed JSON file must not crash collection."""
        primary = fake_agent_root / "aim" / ".aim_state.json"
        primary.write_text("{not valid json", encoding="utf-8")
        data = _collect_live_executions()
        assert data["executing"] == []
        assert data["recent"] == []


class TestLiveLandingRoute:
    def test_route_returns_html(self, client, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {"recent_completions": []},
            },
        )
        resp = client.get("/live")
        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_renders_executing_link_back_to_live_log(self, client, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {
                    "status": "executing",
                    "current_idea_id": "TK-553",
                    "started_at": "2026-04-17T16:20:00",
                    "last_observation": "running tests",
                },
                "board_snapshot": {"recent_completions": []},
            },
        )
        with patch("idea_board.web._live_route_accessible", return_value=True):
            body = client.get("/live").get_data(as_text=True)
        assert 'href="/live/TK-553"' in body
        assert "TK-553" in body

    def test_renders_recent_completion_row_with_link(self, client, fake_agent_root):
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": "TK-550", "summary": "Add hover tooltips", "resolved": "2026-04-18T06:19:29+0900"},
                    ]
                },
            },
        )
        with patch("idea_board.web._live_route_accessible", return_value=True):
            body = client.get("/live").get_data(as_text=True)
        assert 'href="/live/TK-550"' in body
        assert "Add hover tooltips" in body

    def test_empty_state_messages(self, client, fake_agent_root):
        """With no state files at all, both sections render a friendly empty row."""
        # Remove the projects dir too so nothing is discovered.
        (fake_agent_root / "aim" / "projects").rmdir()
        body = client.get("/live").get_data(as_text=True)
        assert "No executions in flight." in body
        assert "No recent completions recorded." in body

    def test_escapes_untrusted_summary(self, client, fake_agent_root):
        """Summaries are echoed into HTML — the escape step must neutralise tags."""
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": "TK-1", "summary": "<script>alert(1)</script>", "resolved": "2026-04-18T00:00:00+0900"},
                    ]
                },
            },
        )
        body = client.get("/live").get_data(as_text=True)
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


class TestHubLiveCard:
    def test_hub_has_view_live_executions_card(self, client):
        body = client.get("/").get_data(as_text=True)
        assert 'href="/live"' in body
        assert "View Live Executions" in body
