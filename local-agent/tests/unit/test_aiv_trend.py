"""Tests for /quality trend chart + /api/aiv/trend (TK-696).

The case-study dashboard needs a chart of the seven AIV axes averaged per
day over a selectable window so the paper can show whether quality is
drifting as throughput climbs.

Covered here:

- GET /api/aiv/trend returns JSON with 7 series (one per axis), each with
  ``{"date": "YYYY-MM-DD", "value": <float>}`` datapoints.
- Daily averaging collapses multiple stories on the same day into one
  point per axis.
- Sentinel ``-1`` and ``NULL`` scores are excluded from the mean.
- ``window=7d`` excludes older rows; ``window=all`` keeps them.
- Unknown ``window`` values fall back to the 7d default.
- Missing AIV DB collapses to seven empty series, not a 500.
- GET /quality renders a ``<canvas>`` tag for the chart and loads Chart.js.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent import aiv_schema
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    """Point aiv_schema at a per-test temporary SQLite DB."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _insert_row(
    *,
    story_key: str,
    validated_at: str | None = None,
    scores: tuple[int, int, int, int, int, int, int] = (9, 8, 7, 10, 9, 6, 8),
    overall: float = 8.1,
) -> None:
    """Insert a synthetic story_quality row for trend tests."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "story_key": story_key,
        "story_title": "trend test",
        "merged_at": validated_at,
        "validated_at": validated_at,
        "meets_requirements": scores[0],
        "code_quality": scores[1],
        "test_quality": scores[2],
        "security_safety": scores[3],
        "scope_discipline": scores[4],
        "edge_cases": scores[5],
        "product_impact": scores[6],
        "overall_score": overall,
        "red_flags_json": "[]",
        "verification_method": "tests-only",
        "verification_output": "",
        "reasoning_json": "{}",
        "error": None,
    }
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(f":{k}" for k in payload)
    conn.execute(
        f"INSERT INTO story_quality ({columns}) VALUES ({placeholders})",
        payload,
    )
    conn.commit()


EXPECTED_AXES = {
    "meets_requirements",
    "code_quality",
    "test_quality",
    "security_safety",
    "scope_discipline",
    "edge_cases",
    "product_impact",
}


def _points_for(body: dict, axis: str) -> list[dict]:
    """Pluck the points list for ``axis`` out of the trend envelope."""
    for s in body["series"]:
        if s["axis"] == axis:
            return s["points"]
    raise AssertionError(f"axis {axis!r} missing from trend payload")


class TestApiAivTrend:
    """GET /api/aiv/trend — per-axis daily means as JSON."""

    def test_returns_seven_series_on_empty_db(self, client):
        """Acceptance: seven series returned even when no rows exist."""
        resp = client.get("/api/aiv/trend")

        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        body = resp.get_json()
        assert "series" in body
        axes = {s["axis"] for s in body["series"]}
        assert axes == EXPECTED_AXES
        for s in body["series"]:
            assert s["points"] == []

    def test_default_window_is_7d(self, client):
        """No ?window arg — envelope reports ``window: "7d"``."""
        resp = client.get("/api/aiv/trend")
        body = resp.get_json()
        assert body["window"] == "7d"

    def test_daily_average_per_axis(self, client):
        """Acceptance: each series point carries ``date`` + averaged ``value``."""
        _insert_row(
            story_key="TK-1001",
            validated_at="2026-04-15T10:00:00+00:00",
            scores=(8, 6, 7, 9, 8, 5, 7),
        )
        _insert_row(
            story_key="TK-1002",
            validated_at="2026-04-15T14:30:00+00:00",
            scores=(10, 8, 9, 10, 9, 7, 9),
        )

        resp = client.get("/api/aiv/trend?window=all")
        body = resp.get_json()

        req_points = _points_for(body, "meets_requirements")
        assert len(req_points) == 1
        assert req_points[0]["date"] == "2026-04-15"
        assert req_points[0]["value"] == pytest.approx(9.0)  # mean(8, 10)

        code_points = _points_for(body, "code_quality")
        assert code_points[0]["value"] == pytest.approx(7.0)  # mean(6, 8)

    def test_points_span_multiple_days_in_order(self, client):
        """Multiple days -> multiple points, sorted ASC by date."""
        _insert_row(
            story_key="TK-1010",
            validated_at="2026-04-14T09:00:00+00:00",
            scores=(7, 7, 7, 7, 7, 7, 7),
        )
        _insert_row(
            story_key="TK-1011",
            validated_at="2026-04-16T09:00:00+00:00",
            scores=(9, 9, 9, 9, 9, 9, 9),
        )

        resp = client.get("/api/aiv/trend?window=all")
        body = resp.get_json()

        points = _points_for(body, "meets_requirements")
        dates = [p["date"] for p in points]
        assert dates == ["2026-04-14", "2026-04-16"]
        assert dates == sorted(dates)
        assert points[0]["value"] == pytest.approx(7.0)
        assert points[1]["value"] == pytest.approx(9.0)

    def test_excludes_sentinel_scores(self, client):
        """Sentinel -1 must not contribute to that axis's daily mean."""
        _insert_row(
            story_key="TK-1020",
            validated_at="2026-04-15T10:00:00+00:00",
            scores=(-1, 8, 7, 9, 8, 5, 7),
        )
        _insert_row(
            story_key="TK-1021",
            validated_at="2026-04-15T11:00:00+00:00",
            scores=(-1, 10, 9, 10, 9, 7, 9),
        )

        resp = client.get("/api/aiv/trend?window=all")
        body = resp.get_json()

        # meets_requirements is -1 for both rows -> no datapoint that day.
        req_points = _points_for(body, "meets_requirements")
        assert req_points == []
        # Other axes still average normally.
        code_points = _points_for(body, "code_quality")
        assert len(code_points) == 1
        assert code_points[0]["value"] == pytest.approx(9.0)

    def test_window_7d_excludes_older_rows(self, client):
        """?window=7d drops rows validated more than 7 days ago."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-1030",
            validated_at=(now - timedelta(days=30)).isoformat(),
            scores=(3, 3, 3, 3, 3, 3, 3),
        )
        _insert_row(
            story_key="TK-1031",
            validated_at=(now - timedelta(hours=1)).isoformat(),
            scores=(9, 9, 9, 9, 9, 9, 9),
        )

        resp = client.get("/api/aiv/trend?window=7d")
        body = resp.get_json()

        req_points = _points_for(body, "meets_requirements")
        assert len(req_points) == 1
        assert req_points[0]["value"] == pytest.approx(9.0)

    def test_window_30d_excludes_rows_older_than_30_days(self, client):
        """?window=30d keeps a 10-day-old row but drops a 60-day-old row."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-1040",
            validated_at=(now - timedelta(days=60)).isoformat(),
            scores=(2, 2, 2, 2, 2, 2, 2),
        )
        _insert_row(
            story_key="TK-1041",
            validated_at=(now - timedelta(days=10)).isoformat(),
            scores=(8, 8, 8, 8, 8, 8, 8),
        )

        resp = client.get("/api/aiv/trend?window=30d")
        body = resp.get_json()

        assert body["window"] == "30d"
        req_points = _points_for(body, "meets_requirements")
        assert len(req_points) == 1
        assert req_points[0]["value"] == pytest.approx(8.0)

    def test_window_all_includes_old_rows(self, client):
        """?window=all keeps rows regardless of age."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-1050",
            validated_at=(now - timedelta(days=120)).isoformat(),
            scores=(5, 5, 5, 5, 5, 5, 5),
        )

        resp = client.get("/api/aiv/trend?window=all")
        body = resp.get_json()

        assert body["window"] == "all"
        req_points = _points_for(body, "meets_requirements")
        assert len(req_points) == 1

    def test_bad_window_falls_back_to_7d(self, client):
        """Unknown ?window value -> default (7d), no 500."""
        resp = client.get("/api/aiv/trend?window=bogus")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["window"] == "7d"

    def test_survives_missing_db(self, client, tmp_path, monkeypatch):
        """No DB file -> empty series for every axis, status 200."""
        ghost = tmp_path / "does-not-exist.db"
        monkeypatch.setattr(aiv_schema, "DB_PATH", ghost)

        resp = client.get("/api/aiv/trend")

        assert resp.status_code == 200
        body = resp.get_json()
        axes = {s["axis"] for s in body["series"]}
        assert axes == EXPECTED_AXES
        for s in body["series"]:
            assert s["points"] == []


class TestQualityChartMarkup:
    """Acceptance: /quality HTML contains a <canvas> for the chart."""

    def test_quality_page_contains_trend_canvas(self, client):
        """<canvas id='quality-trend-chart'> must be present."""
        resp = client.get("/quality")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "<canvas" in body
        assert "quality-trend-chart" in body

    def test_quality_page_loads_chartjs(self, client):
        """Chart.js CDN bundle must be referenced so the canvas can render."""
        resp = client.get("/quality")
        body = resp.get_data(as_text=True).lower()

        assert "chart.js" in body or "chart.umd" in body

    def test_quality_page_has_window_selector(self, client):
        """Window selector control (7d/30d/all) must be rendered."""
        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert 'id="trend-window"' in body
        assert 'value="7d"' in body
        assert 'value="30d"' in body
        assert 'value="all"' in body
