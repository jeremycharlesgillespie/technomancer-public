"""Tests for the aggregation helpers on agent.story_timings.

Covers the queries that feed /performance/breakdown:
- get_recent_runs_breakdown
- get_phase_percentiles
- get_phase_p50
- get_idle_gaps
- _percentile (edge cases)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent import story_timings


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point story_timings at a temp SQLite DB per test."""
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


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _seed_run(run_id: str, story_id: str, start: datetime,
              phases: list[tuple[str, int, bool]]) -> None:
    """Seed N phases for one run_id.

    Each phase tuple is (name, duration_ms, success).
    Phases are laid out back-to-back starting at ``start``.
    """
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
# _percentile
# =============================================================================


class TestPercentileHelper:
    def test_empty_list_returns_zero(self):
        assert story_timings._percentile([], 50) == 0
        assert story_timings._percentile([], 99) == 0

    def test_single_element(self):
        assert story_timings._percentile([42], 50) == 42
        assert story_timings._percentile([42], 99) == 42

    def test_matches_expected_values(self):
        values = list(range(1, 101))  # 1..100, pre-sorted
        # With inclusive linear interp on 1..100, p50 ≈ 50.5 → 50 (rounded)
        assert story_timings._percentile(values, 50) == 50
        assert story_timings._percentile(values, 95) == 95
        assert story_timings._percentile(values, 99) == 99

    def test_p50_of_two_elements_interpolates(self):
        """Two-element list: p50 interpolates halfway between the two values."""
        out = story_timings._percentile([100, 300], 50)
        assert out == 200

    def test_p99_never_exceeds_max(self):
        values = sorted([1, 1, 1, 1, 9999])
        assert story_timings._percentile(values, 99) <= 9999


# =============================================================================
# get_recent_runs_breakdown
# =============================================================================


class TestRecentRunsBreakdown:
    def test_empty_returns_empty_list(self):
        assert story_timings.get_recent_runs_breakdown(limit=50) == []

    def test_returns_runs_newest_first(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-old", "TK-1", base,
                  [("plan", 500, True), ("code", 2000, True)])
        _seed_run("run-new", "TK-2", base + timedelta(hours=1),
                  [("plan", 300, True), ("code", 1000, True)])

        runs = story_timings.get_recent_runs_breakdown(limit=50)
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-new"
        assert runs[1]["run_id"] == "run-old"

    def test_phases_sorted_ascending_within_run(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-A", "TK-A", base, [
            ("plan", 500, True),
            ("code", 2000, True),
            ("test", 700, True),
            ("deploy", 300, True),
        ])
        runs = story_timings.get_recent_runs_breakdown(limit=10)
        assert len(runs) == 1
        phases = [p["phase"] for p in runs[0]["phases"]]
        assert phases == ["plan", "code", "test", "deploy"]

    def test_total_ms_sums_phase_durations(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-A", "TK-A", base, [
            ("plan", 500, True),
            ("code", 2000, True),
            ("test", 700, True),
        ])
        runs = story_timings.get_recent_runs_breakdown(limit=5)
        assert runs[0]["total_ms"] == 3200

    def test_success_false_when_any_phase_failed(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-fail", "TK-F", base, [
            ("plan", 500, True),
            ("code", 2000, False),
            ("test", 700, True),
        ])
        runs = story_timings.get_recent_runs_breakdown(limit=5)
        assert runs[0]["success"] is False

    def test_success_true_when_all_phases_succeeded(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        _seed_run("run-ok", "TK-OK", base, [("plan", 100, True)])
        runs = story_timings.get_recent_runs_breakdown(limit=5)
        assert runs[0]["success"] is True

    def test_limit_clamped_to_positive(self):
        """Zero/negative limit must not SQL-inject or crash."""
        assert story_timings.get_recent_runs_breakdown(limit=0) == []  # empty DB still returns []
        # Seed something so we can prove the clamp returns at least 1
        _seed_run("r1", "TK-1", datetime(2026, 4, 17), [("plan", 1, True)])
        runs = story_timings.get_recent_runs_breakdown(limit=-5)
        assert len(runs) == 1

    def test_skips_rows_without_run_id(self):
        """Rows with NULL run_id don't belong in the stacked-bar view."""
        story_timings.record_phase(
            run_id=None, story_id="TK-N", project="TK", phase="lonely",
            started_at="2026-04-17T10:00:00", ended_at="2026-04-17T10:00:01",
            duration_ms=1000, success=True,
        )
        _seed_run("real-run", "TK-R", datetime(2026, 4, 17, 11),
                  [("plan", 1, True)])
        runs = story_timings.get_recent_runs_breakdown(limit=50)
        assert [r["run_id"] for r in runs] == ["real-run"]

    def test_respects_limit(self):
        base = datetime(2026, 4, 17, 10, 0, 0)
        for i in range(5):
            _seed_run(f"r{i}", f"TK-{i}", base + timedelta(minutes=i),
                      [("plan", 100, True)])
        runs = story_timings.get_recent_runs_breakdown(limit=3)
        assert len(runs) == 3
        # Newest 3: r4, r3, r2
        assert [r["run_id"] for r in runs] == ["r4", "r3", "r2"]


# =============================================================================
# get_phase_percentiles
# =============================================================================


class TestPhasePercentiles:
    def test_empty_returns_empty_list(self):
        assert story_timings.get_phase_percentiles(days=7) == []

    def test_computes_per_phase_percentiles(self):
        now = datetime.now()
        for i in range(1, 11):  # 10 rows per phase
            story_timings.record_phase(
                run_id=f"r-{i}", story_id=f"TK-{i}", project="TK",
                phase="plan",
                started_at=_iso(now - timedelta(hours=1)),
                ended_at=_iso(now),
                duration_ms=i * 100, success=True,  # 100..1000
            )
            story_timings.record_phase(
                run_id=f"r-{i}", story_id=f"TK-{i}", project="TK",
                phase="code",
                started_at=_iso(now - timedelta(hours=1)),
                ended_at=_iso(now),
                duration_ms=i * 1000, success=True,  # 1000..10000
            )

        rows = story_timings.get_phase_percentiles(days=7)
        assert {r["phase"] for r in rows} == {"plan", "code"}
        plan = next(r for r in rows if r["phase"] == "plan")
        code = next(r for r in rows if r["phase"] == "code")
        assert plan["count"] == 10
        assert code["count"] == 10
        # Code is 10x plan — percentiles should reflect that
        assert code["p50_ms"] > plan["p50_ms"]
        assert code["p95_ms"] > plan["p95_ms"]
        assert code["p99_ms"] >= code["p95_ms"] >= code["p50_ms"]

    def test_sorted_by_p50_descending(self):
        now = datetime.now()
        specs = [("fast", 100), ("medium", 500), ("slow", 5000)]
        for phase, dur in specs:
            story_timings.record_phase(
                run_id="r", story_id="TK-1", project="TK", phase=phase,
                started_at=_iso(now - timedelta(hours=1)),
                ended_at=_iso(now),
                duration_ms=dur, success=True,
            )

        rows = story_timings.get_phase_percentiles(days=7)
        phases = [r["phase"] for r in rows]
        assert phases == ["slow", "medium", "fast"]

    def test_excludes_rows_outside_window(self):
        now = datetime.now()
        story_timings.record_phase(
            run_id="r-old", story_id="TK-O", project="TK", phase="plan",
            started_at=_iso(now - timedelta(days=30)),
            ended_at=_iso(now - timedelta(days=30)),
            duration_ms=99999, success=True,
        )
        story_timings.record_phase(
            run_id="r-new", story_id="TK-N", project="TK", phase="plan",
            started_at=_iso(now - timedelta(hours=1)),
            ended_at=_iso(now),
            duration_ms=100, success=True,
        )

        rows = story_timings.get_phase_percentiles(days=7)
        assert len(rows) == 1
        assert rows[0]["count"] == 1
        assert rows[0]["p50_ms"] == 100

    def test_ignores_rows_with_null_duration(self):
        """record_phase int()s duration, but direct DB writes could be NULL."""
        now = datetime.now()
        conn = story_timings._get_conn()
        conn.execute(
            "INSERT INTO story_phase_timings "
            "(run_id, story_id, project, phase, started_at, ended_at, "
            " duration_ms, success) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("r", "TK-1", "TK", "plan", _iso(now), _iso(now), None, 1),
        )
        conn.commit()
        rows = story_timings.get_phase_percentiles(days=7)
        assert rows == []


# =============================================================================
# get_phase_p50
# =============================================================================


class TestPhaseP50:
    def test_empty_returns_zero(self):
        assert story_timings.get_phase_p50("executor.claude_work", days=1) == 0

    def test_returns_p50_for_phase(self):
        now = datetime.now()
        for i in range(1, 11):
            story_timings.record_phase(
                run_id=f"r-{i}", story_id=f"TK-{i}", project="TK",
                phase="executor.claude_work",
                started_at=_iso(now - timedelta(minutes=10)),
                ended_at=_iso(now),
                duration_ms=i * 1000, success=True,
            )
        p50 = story_timings.get_phase_p50("executor.claude_work", days=1)
        assert 4000 <= p50 <= 6000

    def test_unknown_phase_returns_zero(self):
        now = datetime.now()
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(now), ended_at=_iso(now),
            duration_ms=1000, success=True,
        )
        assert story_timings.get_phase_p50("nonexistent", days=7) == 0

    def test_respects_days_window(self):
        now = datetime.now()
        story_timings.record_phase(
            run_id="r-old", story_id="TK-O", project="TK", phase="code",
            started_at=_iso(now - timedelta(days=10)),
            ended_at=_iso(now - timedelta(days=10)),
            duration_ms=5000, success=True,
        )
        assert story_timings.get_phase_p50("code", days=1) == 0
        assert story_timings.get_phase_p50("code", days=30) == 5000


# =============================================================================
# get_idle_gaps
# =============================================================================


class TestIdleGaps:
    def test_empty_returns_empty(self):
        assert story_timings.get_idle_gaps(days=7, limit=10) == []

    def test_detects_gap_between_consecutive_phases(self):
        """Plan ends at 10:00:01, code starts at 10:00:05 → 4s gap."""
        base = datetime.now() - timedelta(hours=1)
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(base),
            ended_at=_iso(base + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="code",
            started_at=_iso(base + timedelta(seconds=5)),
            ended_at=_iso(base + timedelta(seconds=10)),
            duration_ms=5000, success=True,
        )

        gaps = story_timings.get_idle_gaps(days=7, limit=10)
        assert len(gaps) == 1
        g = gaps[0]
        assert g["from_phase"] == "plan"
        assert g["to_phase"] == "code"
        assert g["run_id"] == "r"
        assert 3500 <= g["gap_ms"] <= 4500

    def test_drops_negative_and_zero_gaps(self):
        """Overlapping or back-to-back phases must not appear in the list."""
        base = datetime.now() - timedelta(hours=1)
        # back-to-back: ended_at == next.started_at
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="a",
            started_at=_iso(base),
            ended_at=_iso(base + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="b",
            started_at=_iso(base + timedelta(seconds=1)),
            ended_at=_iso(base + timedelta(seconds=2)),
            duration_ms=1000, success=True,
        )
        assert story_timings.get_idle_gaps(days=7) == []

    def test_sorted_by_gap_desc(self):
        base = datetime.now() - timedelta(hours=1)
        # Run with a 10s gap and a 2s gap
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="a",
            started_at=_iso(base),
            ended_at=_iso(base + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="b",
            started_at=_iso(base + timedelta(seconds=11)),
            ended_at=_iso(base + timedelta(seconds=12)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="c",
            started_at=_iso(base + timedelta(seconds=14)),
            ended_at=_iso(base + timedelta(seconds=15)),
            duration_ms=1000, success=True,
        )

        gaps = story_timings.get_idle_gaps(days=7, limit=10)
        assert len(gaps) == 2
        assert gaps[0]["gap_ms"] > gaps[1]["gap_ms"]
        assert gaps[0]["from_phase"] == "a"  # 10s gap
        assert gaps[1]["from_phase"] == "b"  # 2s gap

    def test_respects_limit(self):
        base = datetime.now() - timedelta(hours=1)
        # 5 gaps in one run
        for i in range(6):
            story_timings.record_phase(
                run_id="r", story_id="TK-1", project="TK", phase=f"p{i}",
                started_at=_iso(base + timedelta(seconds=i * 10)),
                ended_at=_iso(base + timedelta(seconds=i * 10 + 1)),
                duration_ms=1000, success=True,
            )

        gaps = story_timings.get_idle_gaps(days=7, limit=3)
        assert len(gaps) == 3

    def test_does_not_cross_run_boundaries(self):
        """Gap between runs is not an idle gap — it's just time between runs."""
        base = datetime.now() - timedelta(hours=1)
        story_timings.record_phase(
            run_id="run-1", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(base),
            ended_at=_iso(base + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="run-2", story_id="TK-2", project="TK", phase="plan",
            started_at=_iso(base + timedelta(seconds=100)),
            ended_at=_iso(base + timedelta(seconds=101)),
            duration_ms=1000, success=True,
        )
        assert story_timings.get_idle_gaps(days=7) == []

    def test_skips_unparseable_timestamps(self):
        """Bad ISO strings shouldn't crash the dashboard."""
        base = datetime.now() - timedelta(hours=1)
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="plan",
            started_at=_iso(base), ended_at="not-a-timestamp",
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-1", project="TK", phase="code",
            started_at=_iso(base + timedelta(seconds=10)),
            ended_at=_iso(base + timedelta(seconds=11)),
            duration_ms=1000, success=True,
        )
        # Should return no gaps rather than crash — bad timestamp on phase 1
        # prevents the diff from being computed
        assert story_timings.get_idle_gaps(days=7) == []

    def test_excludes_rows_outside_window(self):
        old = datetime.now() - timedelta(days=30)
        story_timings.record_phase(
            run_id="old-run", story_id="TK-X", project="TK", phase="a",
            started_at=_iso(old),
            ended_at=_iso(old + timedelta(seconds=1)),
            duration_ms=1000, success=True,
        )
        story_timings.record_phase(
            run_id="old-run", story_id="TK-X", project="TK", phase="b",
            started_at=_iso(old + timedelta(seconds=100)),
            ended_at=_iso(old + timedelta(seconds=101)),
            duration_ms=1000, success=True,
        )
        assert story_timings.get_idle_gaps(days=7) == []
