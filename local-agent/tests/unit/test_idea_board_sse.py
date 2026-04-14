"""
Tests for SSE log streaming and polling fallback in idea_board/web.py.

Validates that:
- /api/ideas/<id>/log/stream returns proper Server-Sent Events
- /api/ideas/<id>/log returns idea_state for polling fallback
"""

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch
import time as _time

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []
    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event is not None:
            events.append({
                "event": current_event,
                "data": json.loads("".join(current_data)),
            })
            current_event = None
            current_data = []
    return events


@dataclass
class FakeExecutionState:
    """Minimal stand-in for executor.ExecutionState."""
    idea_id: str
    pid: int = 1234
    log_lines: list[str] = field(default_factory=list)
    _is_alive: bool = True
    _started_at: float = field(default_factory=_time.time)

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def elapsed(self) -> float:
        return _time.time() - self._started_at


class TestLogStreamSSE:
    """Tests for GET /api/ideas/<id>/log/stream."""

    def test_stored_log_returns_done(self, client):
        """When idea is not executing, returns stored log and a done event."""
        fake_idea = MagicMock()
        fake_idea.execution_log = "line1\nline2\nline3"
        fake_idea.state = "done"

        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-001/log/stream")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.content_type

            events = _parse_sse(resp.get_data(as_text=True))
            assert len(events) == 2

            assert events[0]["event"] == "log"
            assert events[0]["data"]["lines"] == ["line1", "line2", "line3"]

            assert events[1]["event"] == "done"
            assert events[1]["data"]["idea_state"] == "done"
            assert events[1]["data"]["is_alive"] is False

    def test_no_idea_returns_empty_log(self, client):
        """When idea doesn't exist, returns empty log and done."""
        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=None):
            resp = client.get("/api/ideas/idea-999/log/stream")
            events = _parse_sse(resp.get_data(as_text=True))

            assert events[0]["event"] == "log"
            assert events[0]["data"]["lines"] == []

            assert events[1]["event"] == "done"
            assert events[1]["data"]["idea_state"] == "unknown"

    def test_no_stored_log_returns_empty(self, client):
        """Completed idea with no execution_log returns empty lines."""
        fake_idea = MagicMock()
        fake_idea.execution_log = None
        fake_idea.state = "proposed"

        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-002/log/stream")
            events = _parse_sse(resp.get_data(as_text=True))
            assert events[0]["data"]["lines"] == []

    def test_live_execution_streams_lines(self, client):
        """Live execution sends log and state events, then done on finish."""
        state = FakeExecutionState(idea_id="idea-010")
        state.log_lines = ["Starting...", "Phase 1"]
        # Will become not-alive after the first loop iteration
        call_count = [0]
        original_is_alive = type(state).is_alive

        def _is_alive_side_effect(self):
            call_count[0] += 1
            # First call: alive. Second+ calls: dead.
            return call_count[0] <= 1

        fake_idea = MagicMock()
        fake_idea.state = "executing"

        # On the second check, idea is done
        done_idea = MagicMock()
        done_idea.state = "done"

        with patch("idea_board.web.get_execution", return_value=state), \
             patch("idea_board.web.get_idea", side_effect=[fake_idea, done_idea]), \
             patch("idea_board.web.time.sleep"), \
             patch.object(type(state), "is_alive", new_callable=lambda: property(_is_alive_side_effect)):
            resp = client.get("/api/ideas/idea-010/log/stream")
            events = _parse_sse(resp.get_data(as_text=True))

        event_types = [e["event"] for e in events]
        assert "log" in event_types
        assert "state" in event_types
        assert event_types[-1] == "done"

        # The done event should report done state
        done_ev = [e for e in events if e["event"] == "done"][0]
        assert done_ev["data"]["is_alive"] is False

    def test_sse_content_type(self, client):
        """Response has text/event-stream MIME type."""
        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=None):
            resp = client.get("/api/ideas/idea-001/log/stream")
            assert "text/event-stream" in resp.content_type

    def test_cache_headers(self, client):
        """SSE response has no-cache header."""
        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=None):
            resp = client.get("/api/ideas/idea-001/log/stream")
            assert resp.headers.get("Cache-Control") == "no-cache"

    def test_failed_execution_sends_failed_state(self, client):
        """When execution is alive but idea state is failed, stream closes."""
        state = FakeExecutionState(idea_id="idea-020")
        state.log_lines = ["Error occurred"]
        state._is_alive = False

        fake_idea = MagicMock()
        fake_idea.state = "failed"

        with patch("idea_board.web.get_execution", return_value=state), \
             patch("idea_board.web.get_idea", return_value=fake_idea), \
             patch("idea_board.web.time.sleep"):
            resp = client.get("/api/ideas/idea-020/log/stream")
            events = _parse_sse(resp.get_data(as_text=True))

        done_ev = [e for e in events if e["event"] == "done"][0]
        assert done_ev["data"]["idea_state"] == "failed"


class TestLogPollingEndpoint:
    """Tests for GET /api/ideas/<id>/log — polling fallback endpoint."""

    def test_active_execution_includes_idea_state(self, client):
        """Polling response includes idea_state when execution is active."""
        state = FakeExecutionState(idea_id="idea-030")
        state.log_lines = ["Working..."]

        fake_idea = MagicMock()
        fake_idea.state = "executing"

        with patch("idea_board.web.get_execution", return_value=state), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-030/log")
            data = resp.get_json()
            assert data["idea_state"] == "executing"
            assert data["is_alive"] is True
            assert data["lines"] == ["Working..."]

    def test_stored_log_includes_idea_state(self, client):
        """Polling response includes idea_state for completed ideas."""
        fake_idea = MagicMock()
        fake_idea.state = "done"
        fake_idea.execution_log = "line1\nline2"

        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-031/log")
            data = resp.get_json()
            assert data["idea_state"] == "done"
            assert data["is_alive"] is False
            assert data["lines"] == ["line1", "line2"]

    def test_failed_idea_state(self, client):
        """Polling response shows failed state correctly."""
        fake_idea = MagicMock()
        fake_idea.state = "failed"
        fake_idea.execution_log = "Error: boom"

        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-032/log")
            data = resp.get_json()
            assert data["idea_state"] == "failed"

    def test_no_idea_returns_unknown_state(self, client):
        """Polling response returns unknown state when idea doesn't exist."""
        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=None):
            resp = client.get("/api/ideas/idea-999/log")
            data = resp.get_json()
            assert data["idea_state"] == "unknown"
            assert data["lines"] == []

    def test_no_execution_log_returns_empty(self, client):
        """Polling response returns empty lines when idea has no log."""
        fake_idea = MagicMock()
        fake_idea.state = "proposed"
        fake_idea.execution_log = None

        with patch("idea_board.web.get_execution", return_value=None), \
             patch("idea_board.web.get_idea", return_value=fake_idea):
            resp = client.get("/api/ideas/idea-033/log")
            data = resp.get_json()
            assert data["idea_state"] == "proposed"
            assert data["lines"] == []
