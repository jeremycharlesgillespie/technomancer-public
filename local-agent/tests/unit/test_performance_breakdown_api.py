"""Tests for /performance/breakdown — page + 3 JSON endpoints.

Covers:
- /api/performance/breakdown returns last-N run phase rows
- /api/performance/percentiles returns sorted per-phase percentiles
- /api/performance/idle-gaps returns the top-N gaps
- /performance/breakdown renders HTML with Chart.js
- Hub card shows p50 badge for executor.claude_work
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent import story_timings
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point story_timings at a temp DB per test."""
    db_path = tmp_path / "story_timings.db"
    monkeypatch.setattr(story_timings, "DB_DIR", tmp_path)
    monkeypatch.setattr(story_timings, "DB_PATH", db_path)
    story_timings._local.__dict__.pop("conn", None)
    story_timings.init_db()
    yield
    conn = getattr(story_timings._local, "conn", None)
    if conn:
        conn.close()
        story_timings._local.__dict__.pop("conn", None)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _seed_run(run_id: str, story_id: str, start: datetime,
              phases: list[tuple[str, int, bool]]) -> None:
    cursor = start
    for name, dur_ms, success in phases:
        ended = cursor + timedelta(milliseconds=dur_ms)
        story_timings.record_phase(
            run_id=run_id, story_id=story_id, project="TK", phase=name,
            started_at=_iso(cursor), ended_at=_iso(ended),
            duration_ms=dur_ms, success=success,
        )
        cursor = ended


# =============================================================================
# /api/performance/breakdown
# =============================================================================


class TestBreakdownEndpoint:
    def test_empty_db_returns_empty_runs(self, client):
        resp = client.get("/api/performance/breakdown")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        data = resp.get_json()
        assert data["runs"] == []
        assert data["limit"] == 50

    def test_returns_runs_with_phases(self, client):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-A", "TK-1", base, [
            ("plan", 500, True),
            ("code", 2000, True),
            ("test", 700, True),
        ])
        data = client.get("/api/performance/breakdown").get_json()
        assert len(data["runs"]) == 1
        run = data["runs"][0]
        assert run["run_id"] == "run-A"
        assert run["story_id"] == "TK-1"
        assert len(run["phases"]) == 3
        assert [p["phase"] for p in run["phases"]] == ["plan", "code", "test"]
        assert run["total_ms"] == 3200
        assert run["success"] is True

    def test_limit_clamps_to_500(self, client):
        resp = client.get("/api/performance/breakdown?limit=9999")
        assert resp.status_code == 200
        assert resp.get_json()["limit"] == 500

    def test_limit_clamps_to_1(self, client):
        resp = client.get("/api/performance/breakdown?limit=0")
        assert resp.get_json()["limit"] == 1

    def test_invalid_limit_falls_back_to_default(self, client):
        resp = client.get("/api/performance/breakdown?limit=abc")
        assert resp.get_json()["limit"] == 50

    def test_failed_runs_included_with_success_false(self, client):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-fail", "TK-F", base, [
            ("plan", 100, True),
            ("code", 500, False),
        ])
        data = client.get("/api/performance/breakdown").get_json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["success"] is False


# =============================================================================
# /api/performance/percentiles
# =============================================================================


class TestPercentilesEndpoint:
    def test_empty_db_returns_empty_rows(self, client):
        resp = client.get("/api/performance/percentiles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["rows"] == []
        assert data["days"] == 7

    def test_returns_sorted_rows(self, client):
        now = datetime.now()
        for phase, dur in [("fast", 50), ("medium", 500), ("slow", 5000)]:
            story_timings.record_phase(
                run_id="r", story_id="TK-1", project="TK", phase=phase,
                started_at=_iso(now - timedelta(hours=1)),
                ended_at=_iso(now),
                duration_ms=dur, success=True,
            )
        data = client.get("/api/performance/percentiles").get_json()
        phases = [r["phase"] for r in data["rows"]]
        assert phases == ["slow", "medium", "fast"]
        for r in data["rows"]:
            assert set(r.keys()) == {"phase", "count", "p50_ms", "p95_ms", "p99_ms"}

    def test_days_param_honored(self, client):
        now = datetime.now()
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(now - timedelta(days=5)),
            ended_at=_iso(now - timedelta(days=5)),
            duration_ms=100, success=True,
        )
        rows_1d = client.get("/api/performance/percentiles?days=1").get_json()
        rows_7d = client.get("/api/performance/percentiles?days=7").get_json()
        assert rows_1d["rows"] == []
        assert len(rows_7d["rows"]) == 1

    def test_days_clamped_to_max_90(self, client):
        assert client.get(
            "/api/performance/percentiles?days=9999"
        ).get_json()["days"] == 90

    def test_days_clamped_to_min_1(self, client):
        assert client.get(
            "/api/performance/percentiles?days=0"
        ).get_json()["days"] == 1


# =============================================================================
# /api/performance/idle-gaps
# =============================================================================


class TestIdleGapsEndpoint:
    def test_empty_db_returns_empty_gaps(self, client):
        resp = client.get("/api/performance/idle-gaps")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["gaps"] == []
        assert data["days"] == 7
        assert data["limit"] == 10

    def test_returns_detected_gaps(self, client):
        base = datetime.now() - timedelta(hours=1)
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(base),
            ended_at=_iso(base + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="code",
            started_at=_iso(base + timedelta(seconds=10)),
            ended_at=_iso(base + timedelta(seconds=12)),
            duration_ms=2000, success=True,
        )
        data = client.get("/api/performance/idle-gaps").get_json()
        assert len(data["gaps"]) == 1
        g = data["gaps"][0]
        assert g["from_phase"] == "plan"
        assert g["to_phase"] == "code"
        assert 8500 <= g["gap_ms"] <= 9500

    def test_limit_clamps_to_100(self, client):
        assert client.get(
            "/api/performance/idle-gaps?limit=999"
        ).get_json()["limit"] == 100


# =============================================================================
# /performance/breakdown HTML page
# =============================================================================


class TestPerformanceBreakdownPage:
    def test_returns_200_html(self, client):
        resp = client.get("/performance/breakdown")
        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_embeds_chartjs(self, client):
        body = client.get("/performance/breakdown").data.decode()
        assert "chart.js" in body.lower() or "chart.umd" in body.lower()

    def test_has_three_panels(self, client):
        body = client.get("/performance/breakdown").data.decode()
        assert "stacked" in body.lower()
        assert "percentile" in body.lower()
        assert "idle gap" in body.lower() or "idle-gap" in body.lower() or "idle gaps" in body.lower()

    def test_links_to_companion_apis(self, client):
        body = client.get("/performance/breakdown").data.decode()
        assert "/api/performance/breakdown" in body
        assert "/api/performance/percentiles" in body
        assert "/api/performance/idle-gaps" in body

    def test_back_link_to_hub(self, client):
        body = client.get("/performance/breakdown").data.decode()
        assert 'href="/"' in body


# =============================================================================
# Hub card badge — executor.claude_work p50
# =============================================================================


class TestHubCardBadge:
    def test_hub_shows_performance_breakdown_card(self, client):
        body = client.get("/").data.decode()
        assert "Performance Breakdown" in body
        assert "/performance/breakdown" in body

    def test_hub_badge_shows_no_data_when_empty(self, client):
        body = client.get("/").data.decode()
        # Empty DB → "no data" string in the badge
        assert "no data" in body

    def test_hub_badge_reflects_seeded_p50(self, client):
        now = datetime.now()
        for i in range(1, 11):
            story_timings.record_phase(
                run_id=f"r{i}", story_id=f"TK-{i}", project="TK",
                phase="executor.claude_work",
                started_at=_iso(now - timedelta(minutes=10)),
                ended_at=_iso(now),
                duration_ms=i * 1000, success=True,
            )
        body = client.get("/").data.decode()
        # p50 of [1000..10000] ≈ 5500ms → "5.5s" via _fmt_ms_for_badge
        assert "claude_work p50:" in body
        # Should render as seconds, not "no data"
        assert "no data" not in body.split("claude_work p50:")[1][:50]
