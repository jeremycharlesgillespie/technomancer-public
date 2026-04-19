"""Tests for set_state_dir() and its invocation in /api/aim/* endpoints (TK-816).

Verifies:
- set_state_dir() maps project names to the correct state directory paths
- All 5 /api/aim/* endpoints call set_state_dir() before any state read
- Default ('technomancer') and per-project cases both work correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aim.state import AIMState, WorkerState
from idea_board.web import app, set_state_dir


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# set_state_dir() unit tests
# ---------------------------------------------------------------------------


class TestSetStateDir:
    """Unit tests for the set_state_dir() path-resolution utility."""

    def test_technomancer_returns_aim_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert set_state_dir("technomancer") == tmp_path / "aim"

    def test_primary_alias_returns_aim_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert set_state_dir("primary") == tmp_path / "aim"

    def test_empty_string_returns_aim_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert set_state_dir("") == tmp_path / "aim"

    def test_non_default_returns_projects_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert set_state_dir("40acres") == tmp_path / "aim" / "projects" / "40acres"

    def test_hyphenated_name_returns_projects_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert set_state_dir("my-project") == tmp_path / "aim" / "projects" / "my-project"

    def test_returns_path_object_for_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert isinstance(set_state_dir("technomancer"), Path)

    def test_returns_path_object_for_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        assert isinstance(set_state_dir("myproject"), Path)


# ---------------------------------------------------------------------------
# Verify set_state_dir() is called before state reads in each endpoint
# ---------------------------------------------------------------------------


class TestAimStatusCallsSetStateDir:
    """/api/aim/status must call set_state_dir() before _load_aim_state_for_project."""

    def test_default_project_calls_set_state_dir_with_technomancer(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web.aim_state.load_state", return_value=AIMState()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("technomancer")

    def test_per_project_calls_set_state_dir_with_project_name(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        state_dir = tmp_path / "aim" / "projects" / "myproj"
        state_dir.mkdir(parents=True)
        (state_dir / ".aim_state.json").write_text("{}", encoding="utf-8")

        with patch("idea_board.web.set_state_dir", wraps=set_state_dir) as mock_ssd, \
             patch("idea_board.web._read_aim_events_for_project", return_value=[]):
            resp = client.get("/api/aim/status?project=myproj")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("myproj")

    def test_set_state_dir_called_before_state_read(self, client):
        call_order: list[str] = []

        def track_set_state_dir(project: str) -> Path:
            call_order.append("set_state_dir")
            return Path("/fake/aim")

        def track_load_state() -> AIMState:
            call_order.append("load_state")
            return AIMState()

        with patch("idea_board.web.set_state_dir", side_effect=track_set_state_dir), \
             patch("idea_board.web.aim_state.load_state", side_effect=track_load_state), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            client.get("/api/aim/status")

        assert call_order.index("set_state_dir") < call_order.index("load_state"), (
            "set_state_dir must be called before aim_state.load_state"
        )


class TestAimMetricsCallsSetStateDir:
    """/api/aim/metrics must call set_state_dir() before reading events."""

    def test_default_project_calls_set_state_dir(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/metrics")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("technomancer")

    def test_per_project_calls_set_state_dir_with_project_name(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        with patch("idea_board.web.set_state_dir", wraps=set_state_dir) as mock_ssd, \
             patch("idea_board.web._read_aim_events_for_project", return_value=[]):
            resp = client.get("/api/aim/metrics?project=40acres")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("40acres")


class TestAimBacklogCallsSetStateDir:
    """/api/aim/backlog must call set_state_dir() before reading state."""

    def test_default_project_calls_set_state_dir(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web.aim_jira_reader.count_issues_by_status", return_value={}), \
             patch("idea_board.web.is_jira_configured", return_value=False):
            resp = client.get("/api/aim/backlog")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("technomancer")

    def test_per_project_calls_set_state_dir_with_project_name(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        state_dir = tmp_path / "aim" / "projects" / "myproj"
        state_dir.mkdir(parents=True)
        (state_dir / ".aim_state.json").write_text("{}", encoding="utf-8")

        with patch("idea_board.web.set_state_dir", wraps=set_state_dir) as mock_ssd, \
             patch("idea_board.web.is_jira_configured", return_value=False):
            resp = client.get("/api/aim/backlog?project=myproj")
        assert resp.status_code == 200
        mock_ssd.assert_called_once_with("myproj")


class TestAimProjectsCallsSetStateDir:
    """/api/aim/projects must call set_state_dir() before listing projects."""

    def test_calls_set_state_dir(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web.list_aim_projects", return_value=["technomancer"]):
            resp = client.get("/api/aim/projects")
        assert resp.status_code == 200
        mock_ssd.assert_called_once()

    def test_calls_set_state_dir_before_list_projects(self, client):
        call_order: list[str] = []

        def track_set_state_dir(project: str) -> Path:
            call_order.append("set_state_dir")
            return Path("/fake/aim")

        def track_list_projects() -> list[str]:
            call_order.append("list_aim_projects")
            return ["technomancer"]

        with patch("idea_board.web.set_state_dir", side_effect=track_set_state_dir), \
             patch("idea_board.web.list_aim_projects", side_effect=track_list_projects):
            client.get("/api/aim/projects")

        assert call_order.index("set_state_dir") < call_order.index("list_aim_projects"), (
            "set_state_dir must be called before list_aim_projects"
        )


class TestAimLogsTailCallsSetStateDir:
    """/api/aim/logs/tail must call set_state_dir() before opening log files."""

    def test_calls_set_state_dir_with_project_param(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web._aim_log_paths_for_project", return_value=[]):
            client.get("/api/aim/logs/tail?idea=TK-816&project=technomancer")
        mock_ssd.assert_called_once_with("technomancer")

    def test_calls_set_state_dir_with_detected_project(self, client):
        with patch("idea_board.web.set_state_dir", return_value=Path("/fake/aim")) as mock_ssd, \
             patch("idea_board.web._detect_project_for_idea", return_value="40acres"), \
             patch("idea_board.web._aim_log_paths_for_project", return_value=[]):
            client.get("/api/aim/logs/tail?idea=TK-816")
        mock_ssd.assert_called_once_with("40acres")

    def test_calls_set_state_dir_before_log_path_resolution(self, client):
        call_order: list[str] = []

        def track_set_state_dir(project: str) -> Path:
            call_order.append("set_state_dir")
            return Path("/fake/aim")

        def track_log_paths(project: str) -> list:
            call_order.append("log_paths")
            return []

        with patch("idea_board.web.set_state_dir", side_effect=track_set_state_dir), \
             patch("idea_board.web._aim_log_paths_for_project", side_effect=track_log_paths):
            client.get("/api/aim/logs/tail?idea=TK-816&project=technomancer")

        assert call_order.index("set_state_dir") < call_order.index("log_paths"), (
            "set_state_dir must be called before _aim_log_paths_for_project"
        )


# ---------------------------------------------------------------------------
# Existing functionality preserved — no project param uses default path
# ---------------------------------------------------------------------------


class TestDefaultProjectFallback:
    """When no ?project= param is given, all endpoints use the default path."""

    def test_status_uses_aim_root_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        with patch("idea_board.web.aim_state.load_state", return_value=AIMState()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            with app.test_request_context("/api/aim/status"):
                from idea_board.web import get_project_param
                project = get_project_param()
                resolved = set_state_dir(project)
        assert resolved == tmp_path / "aim"

    def test_metrics_uses_aim_root_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        with app.test_request_context("/api/aim/metrics"):
            from idea_board.web import get_project_param
            project = get_project_param()
            resolved = set_state_dir(project)
        assert resolved == tmp_path / "aim"

    def test_backlog_uses_aim_root_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
        with app.test_request_context("/api/aim/backlog"):
            from idea_board.web import get_project_param
            project = get_project_param()
            resolved = set_state_dir(project)
        assert resolved == tmp_path / "aim"
