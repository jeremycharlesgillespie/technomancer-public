"""Tests for GET /api/aim/projects (TK-770).

Verifies dynamic project discovery from aim/projects/ subdirectories.

Covers:
- 200 with JSON array including "technomancer" as first entry
- Subdirs with .aim_state.json are included; without are skipped
- Missing aim/projects/ directory returns just ["technomancer"]
- 500 when list_aim_projects raises unexpectedly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import idea_board.web as web_module
from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def fake_agent_root(tmp_path):
    """Return a tmp_path wired as _AGENT_ROOT with a minimal aim/ layout."""
    aim_dir = tmp_path / "aim"
    aim_dir.mkdir()
    (aim_dir / "projects").mkdir()
    return tmp_path


class TestAimProjectsEndpoint:
    def test_returns_200_with_technomancer_by_default(self, client, fake_agent_root):
        """technomancer is always first even when aim/projects/ is empty."""
        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert data[0] == {"name": "technomancer"}

    def test_discovers_project_with_state_file(self, client, fake_agent_root):
        """Subdirectory with .aim_state.json appears in response."""
        proj = fake_agent_root / "aim" / "projects" / "40acres"
        proj.mkdir()
        (proj / ".aim_state.json").write_text("{}", encoding="utf-8")

        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.get_json()]
        assert "technomancer" in names
        assert "40acres" in names

    def test_skips_project_without_state_file(self, client, fake_agent_root):
        """Subdirectory without .aim_state.json is not advertised."""
        (fake_agent_root / "aim" / "projects" / "ghost").mkdir()

        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        names = [p["name"] for p in resp.get_json()]
        assert "ghost" not in names

    def test_missing_projects_dir_returns_only_technomancer(self, client, tmp_path):
        """No aim/projects/ directory → just technomancer, no error."""
        aim_dir = tmp_path / "aim"
        aim_dir.mkdir()
        # No projects/ subdirectory created.
        with patch.object(web_module, "_AGENT_ROOT", tmp_path):
            resp = client.get("/api/aim/projects")
        assert resp.status_code == 200
        assert resp.get_json() == [{"name": "technomancer"}]

    def test_technomancer_is_always_first(self, client, fake_agent_root):
        """technomancer precedes alphabetically earlier names."""
        for name in ["aaa", "zzz"]:
            p = fake_agent_root / "aim" / "projects" / name
            p.mkdir()
            (p / ".aim_state.json").write_text("{}", encoding="utf-8")

        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        names = [p["name"] for p in resp.get_json()]
        assert names[0] == "technomancer"

    def test_projects_sorted_alphabetically_after_technomancer(self, client, fake_agent_root):
        """Non-primary projects are returned in sorted order."""
        for name in ["zzz", "aaa", "mmm"]:
            p = fake_agent_root / "aim" / "projects" / name
            p.mkdir()
            (p / ".aim_state.json").write_text("{}", encoding="utf-8")

        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        names = [p["name"] for p in resp.get_json()]
        assert names == ["technomancer", "aaa", "mmm", "zzz"]

    def test_returns_500_on_unexpected_error(self, client):
        """list_aim_projects raising returns 500 with error key."""
        with patch.object(web_module, "list_aim_projects", side_effect=RuntimeError("boom")):
            resp = client.get("/api/aim/projects")
        assert resp.status_code == 500
        assert resp.get_json()["error"] == "failed to list projects"

    def test_response_is_json(self, client, fake_agent_root):
        """Content-Type is application/json."""
        with patch.object(web_module, "_AGENT_ROOT", fake_agent_root):
            resp = client.get("/api/aim/projects")
        assert "application/json" in resp.content_type
