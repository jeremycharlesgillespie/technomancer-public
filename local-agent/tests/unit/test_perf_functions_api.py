"""Tests for GET /api/perf/functions — per-function perf metrics endpoint."""

from __future__ import annotations

import json
from statistics import pstdev

import pytest

import agent.fn_profiler as fp
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_fn_profiler(tmp_path, monkeypatch):
    """Point fn_profiler at a tmp DB and reset state between tests."""
    monkeypatch.setattr(fp, "DB_DIR", tmp_path)
    monkeypatch.setattr(fp, "DB_PATH", tmp_path / "fn_stats.db")
    fp._reset_connection()
    fp.reset_registry()
    fp.stop_background_flush()
    yield
    fp.stop_background_flush()
    fp.reset_registry()
    fp._reset_connection()


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _seed(name: str, call_count: int, total_seconds: float, durations: list[float],
          updated_at: str = "2026-04-16T00:00:00") -> None:
    """Seed one row of the fn_stats SQLite table directly."""
    fp.init_db()
    conn = fp._get_conn()
    # Compute p50/p95 via the real helper so the seeded row matches what
    # flush_stats() would write.
    p50 = fp._percentile(list(durations), 50.0)
    p95 = fp._percentile(list(durations), 95.0)
    conn.execute(
        """INSERT INTO fn_stats
               (name, call_count, total_seconds, p50_seconds,
                p95_seconds, last_n_durations, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               call_count       = excluded.call_count,
               total_seconds    = excluded.total_seconds,
               p50_seconds      = excluded.p50_seconds,
               p95_seconds      = excluded.p95_seconds,
               last_n_durations = excluded.last_n_durations,
               updated_at       = excluded.updated_at""",
        (name, call_count, total_seconds, p50, p95, json.dumps(durations), updated_at),
    )
    conn.commit()


# =========================================================================
# Basics
# =========================================================================


class TestPerfFunctionsBasics:
    def test_empty_db_returns_empty_lists(self, client):
        fp.init_db()
        resp = client.get("/api/perf/functions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["by_total_time"] == []
        assert data["by_call_count"] == []
        assert data["by_variance"] == []

    def test_returns_200_and_json(self, client):
        _seed("mod.fast", 5, 0.5, [0.1, 0.1, 0.1, 0.1, 0.1])
        resp = client.get("/api/perf/functions")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type

    def test_response_has_three_arrays_and_meta(self, client):
        _seed("mod.a", 10, 1.0, [0.1] * 10)
        data = client.get("/api/perf/functions").get_json()
        assert "by_total_time" in data
        assert "by_call_count" in data
        assert "by_variance" in data
        assert "meta" in data
        assert "collected_since" in data["meta"]

    def test_item_shape(self, client):
        _seed("mod.a", 10, 1.0, [0.1] * 10)
        data = client.get("/api/perf/functions").get_json()
        item = data["by_total_time"][0]
        for field in ("name", "call_count", "total_seconds",
                      "p50_seconds", "p95_seconds", "stddev_seconds"):
            assert field in item
        assert item["name"] == "mod.a"
        assert item["call_count"] == 10


# =========================================================================
# Ordering — the acceptance criterion
# =========================================================================


class TestPerfFunctionsOrdering:
    """Seed SQLite with 3 fake functions and assert each list's ordering."""

    def _seed_three(self):
        # SLOW: highest total time, moderate call count, low variance.
        _seed("mod.slow", call_count=10, total_seconds=100.0,
              durations=[10.0] * 10)
        # CHATTY: highest call count, low total time, zero variance.
        _seed("mod.chatty", call_count=1000, total_seconds=5.0,
              durations=[0.005] * 50)
        # SPIKY: high variance (very uneven durations), middle total / count.
        _seed("mod.spiky", call_count=50, total_seconds=20.0,
              durations=[0.01, 0.02, 5.0, 0.01, 0.03, 4.5, 0.02, 0.01, 5.2, 0.01])

    def test_by_total_time_order(self, client):
        self._seed_three()
        data = client.get("/api/perf/functions").get_json()
        names = [x["name"] for x in data["by_total_time"]]
        assert names == ["mod.slow", "mod.spiky", "mod.chatty"]

    def test_by_call_count_order(self, client):
        self._seed_three()
        data = client.get("/api/perf/functions").get_json()
        names = [x["name"] for x in data["by_call_count"]]
        assert names == ["mod.chatty", "mod.spiky", "mod.slow"]

    def test_by_variance_order(self, client):
        self._seed_three()
        data = client.get("/api/perf/functions").get_json()
        names = [x["name"] for x in data["by_variance"]]
        # mod.spiky has the widest distribution, mod.chatty all equal -> 0.
        assert names[0] == "mod.spiky"
        assert names[-1] == "mod.chatty"

    def test_stddev_value_matches_pstdev(self, client):
        durations = [0.01, 0.02, 5.0, 0.01, 0.03, 4.5, 0.02, 0.01, 5.2, 0.01]
        _seed("mod.spiky", 50, 20.0, durations)
        data = client.get("/api/perf/functions").get_json()
        item = next(x for x in data["by_variance"] if x["name"] == "mod.spiky")
        assert item["stddev_seconds"] == pytest.approx(pstdev(durations), rel=1e-9)


# =========================================================================
# Limit handling
# =========================================================================


class TestPerfFunctionsLimit:
    def test_default_limit_is_twenty(self, client):
        for i in range(25):
            _seed(f"mod.fn{i:02d}", 10, float(i + 1), [float(i + 1) / 10] * 5)
        data = client.get("/api/perf/functions").get_json()
        assert len(data["by_total_time"]) == 20
        assert len(data["by_call_count"]) == 20
        assert len(data["by_variance"]) == 20
        assert data["meta"]["limit"] == 20

    def test_limit_query_param_truncates_all_three_lists(self, client):
        for i in range(10):
            _seed(f"mod.fn{i:02d}", 10, float(i + 1), [float(i + 1) / 10] * 5)
        data = client.get("/api/perf/functions?limit=5").get_json()
        assert len(data["by_total_time"]) == 5
        assert len(data["by_call_count"]) == 5
        assert len(data["by_variance"]) == 5
        assert data["meta"]["limit"] == 5

    def test_limit_clamped_to_max_100(self, client):
        data = client.get("/api/perf/functions?limit=999").get_json()
        assert data["meta"]["limit"] == 100

    def test_limit_clamped_to_min_1(self, client):
        data = client.get("/api/perf/functions?limit=0").get_json()
        assert data["meta"]["limit"] == 1

    def test_invalid_limit_falls_back_to_default(self, client):
        data = client.get("/api/perf/functions?limit=abc").get_json()
        assert data["meta"]["limit"] == 20


# =========================================================================
# Filtering: zero-call functions excluded
# =========================================================================


class TestPerfFunctionsFiltering:
    def test_uncalled_function_does_not_appear(self, client):
        _seed("mod.called", 10, 1.0, [0.1] * 10)
        _seed("mod.uncalled", 0, 0.0, [])
        data = client.get("/api/perf/functions").get_json()
        for key in ("by_total_time", "by_call_count", "by_variance"):
            names = [x["name"] for x in data[key]]
            assert "mod.uncalled" not in names
            assert "mod.called" in names


# =========================================================================
# Meta block
# =========================================================================


class TestPerfFunctionsMeta:
    def test_collected_since_reflects_min_updated_at(self, client):
        _seed("mod.old", 5, 1.0, [0.2] * 5, updated_at="2026-01-01T00:00:00")
        _seed("mod.newer", 5, 1.0, [0.2] * 5, updated_at="2026-04-15T12:00:00")
        data = client.get("/api/perf/functions").get_json()
        assert data["meta"]["collected_since"] == "2026-01-01T00:00:00"

    def test_collected_since_none_on_empty_db(self, client):
        fp.init_db()
        data = client.get("/api/perf/functions").get_json()
        assert data["meta"]["collected_since"] is None
