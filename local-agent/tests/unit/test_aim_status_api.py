"""Tests for the GET /api/aim/status snapshot endpoint in idea_board/web.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aim.state import AIMState, WorkerState
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_state(**worker_kwargs) -> AIMState:
    worker = WorkerState(
        pid=worker_kwargs.pop("pid", 4321),
        status=worker_kwargs.pop("status", "executing"),
        current_idea_id=worker_kwargs.pop("current_idea_id", "TK-376"),
        started_at=worker_kwargs.pop("started_at", "2026-04-15T15:40:00"),
        last_heartbeat=worker_kwargs.pop("last_heartbeat", "2026-04-15T15:41:00"),
        last_observation=worker_kwargs.pop("last_observation", "Executing idea"),
        consecutive_failures=worker_kwargs.pop("consecutive_failures", 0),
    )
    return AIMState(
        manager_pid=1234,
        manager_started_at="2026-04-15T15:00:00",
        last_cycle="2026-04-15T15:41:10",
        last_completion="2026-04-15T15:30:00",
        completions_today=2,
        cycle_count=42,
        worker=worker,
        last_error="",
    )


class TestAimStatusBasics:
    def test_returns_200(self, client):
        with patch("idea_board.web.aim_state.load_state", return_value=_make_state()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            assert resp.status_code == 200

    def test_content_type_is_json(self, client):
        with patch("idea_board.web.aim_state.load_state", return_value=_make_state()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            assert "application/json" in resp.content_type

    def test_acceptance_keys_present(self, client):
        """Acceptance: response contains worker.pid, worker.status,
        current_idea_id, cycle_count, last_decisions."""
        with patch("idea_board.web.aim_state.load_state", return_value=_make_state()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert "worker" in data
            assert "pid" in data["worker"]
            assert "status" in data["worker"]
            assert "current_idea_id" in data
            assert "cycle_count" in data
            assert "last_decisions" in data


class TestAimStatusWorkerFields:
    def test_worker_pid_and_status(self, client):
        with patch(
            "idea_board.web.aim_state.load_state",
            return_value=_make_state(pid=9999, status="watching"),
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert data["worker"]["pid"] == 9999
            assert data["worker"]["status"] == "watching"

    def test_worker_heartbeat_and_observation_exposed(self, client):
        with patch(
            "idea_board.web.aim_state.load_state",
            return_value=_make_state(
                last_heartbeat="2026-04-15T16:00:00",
                last_observation="Ran tests",
                consecutive_failures=1,
            ),
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert data["worker"]["last_heartbeat"] == "2026-04-15T16:00:00"
            assert data["worker"]["last_observation"] == "Ran tests"
            assert data["worker"]["consecutive_failures"] == 1

    def test_aiw_worker_backend_present(self, client):
        """Acceptance: response contains aiw_worker_backend field."""
        with patch("idea_board.web.aim_state.load_state", return_value=_make_state()), \
             patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert "aiw_worker_backend" in data
            assert data["aiw_worker_backend"] == "ollama"

    def test_current_idea_id_mirrored_at_top_level(self, client):
        with patch(
            "idea_board.web.aim_state.load_state",
            return_value=_make_state(current_idea_id="TK-999"),
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert data["current_idea_id"] == "TK-999"
            assert data["worker"]["current_idea_id"] == "TK-999"

    def test_idle_worker_has_null_idea(self, client):
        with patch(
            "idea_board.web.aim_state.load_state",
            return_value=_make_state(status="idle", current_idea_id=None, pid=None),
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert data["worker"]["status"] == "idle"
            assert data["worker"]["pid"] is None
            assert data["current_idea_id"] is None


class TestAimStatusManagerFields:
    def test_manager_and_cycle_fields(self, client):
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            data = resp.get_json()
            assert data["manager_pid"] == 1234
            assert data["manager_started_at"] == "2026-04-15T15:00:00"
            assert data["cycle_count"] == 42
            assert data["last_cycle"] == "2026-04-15T15:41:10"
            assert data["last_completion"] == "2026-04-15T15:30:00"
            assert data["completions_today"] == 2

    def test_last_error_defaults_to_empty(self, client):
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/status").get_json()
            assert data["last_error"] == ""

    def test_snapshot_at_present(self, client):
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/status").get_json()
            assert "snapshot_at" in data
            assert isinstance(data["snapshot_at"], str)
            assert len(data["snapshot_at"]) > 0


class TestAimStatusDecisions:
    def test_no_decisions_returns_empty_list(self, client):
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch(
            "idea_board.web.aim_event_log.read_events",
            return_value=[
                {"timestamp": "2026-04-15T10:00:00", "type": "worker_started", "data": {}},
                {"timestamp": "2026-04-15T10:01:00", "type": "cycle_started", "data": {}},
            ],
        ):
            data = client.get("/api/aim/status").get_json()
            assert data["last_decisions"] == []

    def test_filters_to_decision_made_only(self, client):
        events = [
            {"timestamp": "2026-04-15T10:00:00", "type": "worker_started", "data": {"pid": 1}},
            {"timestamp": "2026-04-15T10:01:00", "type": "decision_made", "data": {"d": "a"}},
            {"timestamp": "2026-04-15T10:02:00", "type": "cycle_started", "data": {}},
            {"timestamp": "2026-04-15T10:03:00", "type": "decision_made", "data": {"d": "b"}},
        ]
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/status").get_json()
            assert len(data["last_decisions"]) == 2
            for item in data["last_decisions"]:
                assert item["type"] == "decision_made"

    def test_keeps_only_last_three(self, client):
        events = [
            {"timestamp": f"2026-04-15T10:{i:02d}:00", "type": "decision_made",
             "data": {"i": i}}
            for i in range(10)
        ]
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/status").get_json()
            assert len(data["last_decisions"]) == 3

    def test_last_decisions_newest_first(self, client):
        events = [
            {"timestamp": "2026-04-15T10:00:00", "type": "decision_made", "data": {"i": 0}},
            {"timestamp": "2026-04-15T10:01:00", "type": "decision_made", "data": {"i": 1}},
            {"timestamp": "2026-04-15T10:02:00", "type": "decision_made", "data": {"i": 2}},
            {"timestamp": "2026-04-15T10:03:00", "type": "decision_made", "data": {"i": 3}},
        ]
        with patch(
            "idea_board.web.aim_state.load_state", return_value=_make_state()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/status").get_json()
            # Newest first — most recent decision at index 0
            assert [d["data"]["i"] for d in data["last_decisions"]] == [3, 2, 1]


class TestAimStatusResilience:
    def test_state_load_returns_defaults_when_missing(self, client):
        """If no state file exists, load_state returns an AIMState() default."""
        with patch(
            "idea_board.web.aim_state.load_state", return_value=AIMState()
        ), patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["worker"]["status"] == "idle"
            assert data["worker"]["pid"] is None
            assert data["cycle_count"] == 0
            assert data["last_decisions"] == []
