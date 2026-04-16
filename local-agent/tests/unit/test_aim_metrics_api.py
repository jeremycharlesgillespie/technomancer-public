"""Tests for the GET /api/aim/metrics endpoint in idea_board/web.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _event(ts: datetime, etype: str) -> dict:
    """Shape a raw event dict the way read_events returns them."""
    return {
        "timestamp": ts.isoformat(timespec="seconds"),
        "type": etype,
        "data": {},
    }


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_returns_200(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/metrics")
            assert resp.status_code == 200

    def test_content_type_is_json(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            resp = client.get("/api/aim/metrics")
            assert "application/json" in resp.content_type

    def test_acceptance_keys_present(self, client):
        """Acceptance: response has timestamps, completions, failures, totals."""
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics").get_json()
            assert "timestamps" in data
            assert "completions" in data
            assert "failures" in data
            assert "totals" in data
            assert "completions" in data["totals"]
            assert "failures" in data["totals"]
            assert "completion_rate" in data["totals"]

    def test_arrays_are_equal_length(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=24").get_json()
            assert len(data["timestamps"]) == 24
            assert len(data["completions"]) == 24
            assert len(data["failures"]) == 24


# ---------------------------------------------------------------------------
# Hours param handling
# ---------------------------------------------------------------------------


class TestHoursParam:
    def test_default_is_24(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics").get_json()
            assert data["hours"] == 24
            assert len(data["timestamps"]) == 24

    @pytest.mark.parametrize("hours", [1, 6, 24, 168])
    def test_accepts_documented_values(self, client, hours):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get(f"/api/aim/metrics?hours={hours}").get_json()
            assert data["hours"] == hours
            assert len(data["timestamps"]) == hours

    def test_clamps_above_max_to_168(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=9999").get_json()
            assert data["hours"] == 168
            assert len(data["timestamps"]) == 168

    def test_clamps_zero_to_one(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=0").get_json()
            assert data["hours"] == 1
            assert len(data["timestamps"]) == 1

    def test_clamps_negative_to_one(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=-5").get_json()
            assert data["hours"] == 1

    def test_non_numeric_defaults_to_24(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=abc").get_json()
            assert data["hours"] == 24


# ---------------------------------------------------------------------------
# Bucketing correctness
# ---------------------------------------------------------------------------


class TestBucketing:
    def test_seeded_events_over_three_hours_bucket_correctly(self, client):
        """Seed a fake event log with a mix of completed/failed events
        across 3 distinct hour buckets and assert per-bucket counts."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        h0 = now                      # current hour
        h1 = now - timedelta(hours=1)  # one hour ago
        h2 = now - timedelta(hours=2)  # two hours ago

        events = [
            # h2: 1 completed, 2 failed
            _event(h2 + timedelta(minutes=5), "execution_completed"),
            _event(h2 + timedelta(minutes=10), "execution_failed"),
            _event(h2 + timedelta(minutes=45), "execution_failed"),
            # h1: 3 completed, 0 failed
            _event(h1 + timedelta(minutes=1), "execution_completed"),
            _event(h1 + timedelta(minutes=20), "execution_completed"),
            _event(h1 + timedelta(minutes=59), "execution_completed"),
            # h0: 2 completed, 1 failed
            _event(h0 + timedelta(minutes=2), "execution_completed"),
            _event(h0 + timedelta(minutes=3), "execution_completed"),
            _event(h0 + timedelta(minutes=4), "execution_failed"),
        ]

        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=3").get_json()

        # Buckets are oldest-first: [h2, h1, h0]
        assert data["completions"] == [1, 3, 2]
        assert data["failures"] == [2, 0, 1]

    def test_timestamps_are_oldest_first_and_hour_floored(self, client):
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=3").get_json()

        ts = [datetime.fromisoformat(s) for s in data["timestamps"]]
        assert ts == sorted(ts)  # oldest-first
        for t in ts:
            assert t.minute == 0 and t.second == 0

        # Spacing is exactly one hour
        assert ts[1] - ts[0] == timedelta(hours=1)
        assert ts[2] - ts[1] == timedelta(hours=1)

    def test_ignores_events_outside_window(self, client):
        """Events older than the requested window must not be counted."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        way_old = now - timedelta(hours=100)

        events = [
            _event(way_old, "execution_completed"),
            _event(way_old, "execution_failed"),
        ]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert sum(data["completions"]) == 0
        assert sum(data["failures"]) == 0

    def test_filters_out_unrelated_event_types(self, client):
        """Only execution_completed / execution_failed contribute."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [
            _event(now, "execution_completed"),
            _event(now, "worker_spawned"),
            _event(now, "decision_made"),
            _event(now, "cycle_started"),
            _event(now, "execution_failed"),
        ]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=1").get_json()

        assert data["completions"] == [1]
        assert data["failures"] == [1]

    def test_malformed_timestamp_is_skipped(self, client):
        """A bad ISO string shouldn't crash the endpoint."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [
            {"timestamp": "not-a-date", "type": "execution_completed", "data": {}},
            {"timestamp": None, "type": "execution_failed", "data": {}},
            _event(now, "execution_completed"),
        ]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            resp = client.get("/api/aim/metrics?hours=1")
            assert resp.status_code == 200
            data = resp.get_json()

        assert data["completions"] == [1]
        assert data["failures"] == [0]


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


class TestTotals:
    def test_totals_sum_matches_arrays(self, client):
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [
            _event(now, "execution_completed"),
            _event(now, "execution_completed"),
            _event(now, "execution_completed"),
            _event(now, "execution_failed"),
        ]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert data["totals"]["completions"] == sum(data["completions"]) == 3
        assert data["totals"]["failures"] == sum(data["failures"]) == 1

    def test_completion_rate_calculated(self, client):
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [
            _event(now, "execution_completed"),
            _event(now, "execution_completed"),
            _event(now, "execution_completed"),
            _event(now, "execution_failed"),
        ]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert data["totals"]["completion_rate"] == pytest.approx(0.75)

    def test_completion_rate_zero_when_no_events(self, client):
        """Empty window ⇒ rate is 0.0, not a ZeroDivisionError."""
        with patch("idea_board.web.aim_event_log.read_events", return_value=[]):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert data["totals"]["completions"] == 0
        assert data["totals"]["failures"] == 0
        assert data["totals"]["completion_rate"] == 0.0

    def test_completion_rate_one_when_all_succeed(self, client):
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [_event(now, "execution_completed") for _ in range(5)]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert data["totals"]["completion_rate"] == 1.0

    def test_completion_rate_zero_when_all_fail(self, client):
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        events = [_event(now, "execution_failed") for _ in range(3)]
        with patch("idea_board.web.aim_event_log.read_events", return_value=events):
            data = client.get("/api/aim/metrics?hours=24").get_json()

        assert data["totals"]["completion_rate"] == 0.0
