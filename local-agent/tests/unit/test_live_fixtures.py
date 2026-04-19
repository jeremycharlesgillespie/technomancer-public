"""Tests for the /live page data-mocking fixtures (TK-793).

Verifies that ``mock_executor_runs_db`` and ``mock_aim_state_files`` short-
circuit real disk + DB I/O so /live tests run instantly instead of hanging
on file reads. Each test sets a strict wall-clock budget so a regression
that reintroduces real I/O fails loudly instead of timing out the whole
suite.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agent import executor_runs_db
from idea_board.web import _collect_live_executions, app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _empty_ideas():
    """Keep the hub/home renderers independent of real board data."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


class TestMockExecutorRunsDB:
    def test_returns_non_empty_execution_dicts_by_default(
        self, mock_executor_runs_db
    ):
        rows = executor_runs_db.get_current_execution_per_project()
        assert isinstance(rows, list) and rows, "expected non-empty list"
        row = rows[0]
        for key in (
            "project",
            "status",
            "current_idea_id",
            "started_at",
            "last_observation",
            "is_executing",
        ):
            assert key in row, f"missing key {key!r} in executor row"
        assert row["is_executing"] is True

    def test_return_value_is_overridable(self, mock_executor_runs_db):
        mock_executor_runs_db.return_value = [
            {
                "project": "FA",
                "status": "watching",
                "current_idea_id": "FA-1",
                "started_at": "",
                "last_observation": "",
                "is_executing": False,
            }
        ]
        rows = executor_runs_db.get_current_execution_per_project()
        assert rows[0]["project"] == "FA"
        assert rows[0]["is_executing"] is False

    def test_returns_instantly(self, mock_executor_runs_db):
        """No real I/O: call must complete well under 1 second."""
        start = time.monotonic()
        for _ in range(100):
            executor_runs_db.get_current_execution_per_project()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"mock took {elapsed:.2f}s — real I/O leaked?"


class TestMockAimStateFiles:
    def test_returns_valid_aim_schema_by_default(self, mock_aim_state_files):
        data = executor_runs_db._read_aim_state_file(
            # Pass any path — the mock ignores it.
            pytest.importorskip("pathlib").Path("unused")
        )
        assert isinstance(data, dict) and data
        assert "worker" in data
        assert "board_snapshot" in data
        assert data["board_snapshot"]["recent_completions"], (
            "default payload should include at least one completion"
        )

    def test_return_value_is_overridable(self, mock_aim_state_files):
        mock_aim_state_files.return_value = {
            "worker": {"status": "idle", "current_idea_id": None},
            "board_snapshot": {"recent_completions": []},
        }
        data = executor_runs_db._read_aim_state_file(
            pytest.importorskip("pathlib").Path("unused")
        )
        assert data["worker"]["status"] == "idle"
        assert data["board_snapshot"]["recent_completions"] == []

    def test_returns_instantly(self, mock_aim_state_files):
        from pathlib import Path

        start = time.monotonic()
        for _ in range(100):
            executor_runs_db._read_aim_state_file(Path("unused"))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"mock took {elapsed:.2f}s — real I/O leaked?"


class TestFixturesWithLiveRoute:
    """Wire both fixtures through the real /live collection path."""

    def test_collect_live_executions_uses_mocked_db(
        self, mock_executor_runs_db, mock_aim_state_files, tmp_path, monkeypatch
    ):
        # Point _AGENT_ROOT at a throwaway dir so the recent-completions
        # second pass finds no state files — the mock would cover it
        # anyway, but this keeps the test explicit about what's exercised.
        (tmp_path / "aim" / "projects").mkdir(parents=True)
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)

        start = time.monotonic()
        data = _collect_live_executions()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"/live collection took {elapsed:.2f}s"
        assert data["executing"], "fixture should produce an executing row"
        assert data["executing"][0]["key"] == "TK-793"

    def test_live_route_renders_fast(
        self, mock_executor_runs_db, mock_aim_state_files, client, tmp_path, monkeypatch
    ):
        (tmp_path / "aim" / "projects").mkdir(parents=True)
        monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)

        start = time.monotonic()
        resp = client.get("/live")
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert "text/html" in resp.content_type
        assert elapsed < 2.0, f"/live rendered in {elapsed:.2f}s"
