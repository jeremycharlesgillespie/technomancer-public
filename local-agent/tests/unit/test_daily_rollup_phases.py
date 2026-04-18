"""Tests for agent.daily_rollup phase-timings aggregation.

These tests verify the second writer step of ``compute_and_write``: reading
``story_phase_timings`` for the day, computing per-phase p50/p95, and
writing JSON into the ``phase_timings_json`` column on ``daily_stats``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent import daily_rollup, daily_stats, executor_runs_db, story_timings


@pytest.fixture(autouse=True)
def _isolate_dbs(tmp_path, monkeypatch):
    """Point all three SQLite databases at temp paths and reset conn caches.

    compute_and_write now touches three DBs: executor_runs (read),
    story_phase_timings (read), and daily_stats (write). Each module has its
    own per-thread connection cache that has to be cleared so the new
    DB_PATH takes effect.
    """
    executor_db = tmp_path / "executor_runs.db"
    stats_db = tmp_path / "daily_stats.db"
    timings_db = tmp_path / "story_timings.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", executor_db)
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", stats_db)
    monkeypatch.setattr(story_timings, "DB_DIR", tmp_path)
    monkeypatch.setattr(story_timings, "DB_PATH", timings_db)
    for mod in (executor_runs_db, daily_stats, story_timings):
        mod._local.__dict__.pop("conn", None)
    yield
    for mod in (executor_runs_db, daily_stats, story_timings):
        conn = getattr(mod._local, "conn", None)
        if conn is not None:
            conn.close()
            mod._local.__dict__.pop("conn", None)


def _insert_phase(**fields: Any) -> None:
    """Insert a synthetic story_phase_timings row for test setup."""
    story_timings.init_db()
    conn = story_timings._get_conn()
    defaults: dict[str, Any] = {
        "run_id": "run-1",
        "story_id": "TK-1",
        "project": "TK",
        "phase": "code",
        "started_at": "2026-04-17T10:00:00",
        "ended_at": "2026-04-17T10:05:00",
        "duration_ms": 1000,
        "success": 1,
        "metadata": None,
    }
    defaults.update(fields)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    conn.execute(
        f"INSERT INTO story_phase_timings ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


def _get_phase_timings_json(date: str, project: str) -> str | None:
    """Fetch the phase_timings_json cell for one (date, project) row."""
    conn = daily_stats._get_conn()
    row = conn.execute(
        "SELECT phase_timings_json FROM daily_stats "
        "WHERE date = ? AND project = ?",
        (date, project),
    ).fetchone()
    if row is None:
        return None
    return row["phase_timings_json"]


class TestSchemaMigration:
    def test_column_exists_after_init(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(daily_stats)").fetchall()
        }
        assert "phase_timings_json" in cols

    def test_column_is_text_and_nullable(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        info = next(
            r for r in conn.execute("PRAGMA table_info(daily_stats)").fetchall()
            if r["name"] == "phase_timings_json"
        )
        assert info["type"].upper() == "TEXT"
        assert info["notnull"] == 0

    def test_migration_adds_column_to_preexisting_db(self):
        """Opening a db that predates the column must ALTER TABLE idempotently."""
        import sqlite3 as _sqlite3

        raw = _sqlite3.connect(str(daily_stats.DB_PATH))
        raw.execute(
            "CREATE TABLE daily_stats ("
            "date TEXT NOT NULL, project TEXT NOT NULL, "
            "PRIMARY KEY (date, project))"
        )
        raw.commit()
        raw.close()
        daily_stats._local.__dict__.pop("conn", None)

        daily_stats.init_db()

        conn = daily_stats._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(daily_stats)").fetchall()
        }
        assert "phase_timings_json" in cols

    def test_init_db_idempotent_with_new_column(self):
        daily_stats.init_db()
        daily_stats.init_db()
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='daily_stats'"
        ).fetchall()
        assert len(tables) == 1


class TestPhaseTimingsWritten:
    def test_acceptance_single_phase_produces_p50_and_p95(self):
        """AC: synthetic rows produce phase_timings_json with p50/p95."""
        for dur in (1000, 2000, 3000, 4000, 5000):
            _insert_phase(
                project="TK", phase="plan",
                started_at="2026-04-17T10:00:00",
                duration_ms=dur,
            )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        blob = _get_phase_timings_json("2026-04-17", "TK")
        assert blob is not None
        data = json.loads(blob)
        assert "plan" in data
        assert data["plan"]["count"] == 5
        assert data["plan"]["p50_ms"] == 3000
        assert data["plan"]["p95_ms"] > data["plan"]["p50_ms"]

    def test_multiple_phases_each_get_their_own_percentiles(self):
        """Each phase name is an independent grouping key."""
        for dur in (1000, 2000, 3000):
            _insert_phase(
                project="TK", phase="plan",
                started_at="2026-04-17T10:00:00",
                duration_ms=dur,
            )
        for dur in (5000, 10000, 15000):
            _insert_phase(
                project="TK", phase="code",
                started_at="2026-04-17T11:00:00",
                duration_ms=dur,
            )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert set(data.keys()) == {"plan", "code"}
        assert data["plan"]["count"] == 3
        assert data["plan"]["p50_ms"] == 2000
        assert data["code"]["count"] == 3
        assert data["code"]["p50_ms"] == 10000

    def test_single_sample_p50_equals_p95(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=7500,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data["plan"]["count"] == 1
        assert data["plan"]["p50_ms"] == 7500
        assert data["plan"]["p95_ms"] == 7500

    def test_stored_value_is_valid_json(self):
        _insert_phase(
            project="TK", phase="deploy",
            started_at="2026-04-17T10:00:00",
            duration_ms=42,
        )
        daily_rollup.compute_and_write("2026-04-17", "TK")
        blob = _get_phase_timings_json("2026-04-17", "TK")
        assert isinstance(blob, str)
        parsed = json.loads(blob)
        assert isinstance(parsed, dict)


class TestMissingPhaseData:
    def test_no_phase_rows_writes_null(self):
        """AC: missing phase data for a day doesn't crash — writes null."""
        executor_runs_db.init_db()
        story_timings.init_db()

        daily_rollup.compute_and_write("2026-04-17", "TK")

        blob = _get_phase_timings_json("2026-04-17", "TK")
        assert blob is None

    def test_story_timings_db_absent_still_succeeds(self):
        """Even if nothing has ever inited story_timings, rollup must run."""
        executor_runs_db.init_db()

        # Don't touch story_timings — let compute_and_write init it.
        result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["shipped"] == 0
        assert _get_phase_timings_json("2026-04-17", "TK") is None

    def test_all_phase_rows_have_null_duration(self):
        """A row with NULL duration_ms shouldn't count — if it's the only row,
        phase_timings_json must be null."""
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=None,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        assert _get_phase_timings_json("2026-04-17", "TK") is None


class TestFilters:
    def test_filters_out_different_date(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-16T10:00:00",
            duration_ms=1000,
        )
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=2000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data["plan"]["count"] == 1
        assert data["plan"]["p50_ms"] == 2000

    def test_filters_out_different_project(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )
        _insert_phase(
            project="FA", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=5000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data["plan"]["count"] == 1
        assert data["plan"]["p50_ms"] == 1000

    def test_ignores_null_duration_rows_while_keeping_others(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=None,
        )
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=2000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data["plan"]["count"] == 1
        assert data["plan"]["p50_ms"] == 2000

    def test_ignores_null_phase_rows(self):
        _insert_phase(
            project="TK", phase=None,
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=2000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert set(data.keys()) == {"plan"}

    def test_handles_fractional_second_timestamp(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00.123456",
            duration_ms=1000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        data = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data["plan"]["count"] == 1


class TestIdempotency:
    def test_rerun_updates_phase_timings(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")
        data1 = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data1["plan"]["count"] == 1

        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T11:00:00",
            duration_ms=3000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")
        data2 = json.loads(_get_phase_timings_json("2026-04-17", "TK"))
        assert data2["plan"]["count"] == 2

    def test_rerun_clears_phase_timings_when_rows_disappear(self):
        """Deleting every phase row must roll the column back to NULL so the
        slide generator never shows stale phase data."""
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")
        assert _get_phase_timings_json("2026-04-17", "TK") is not None

        story_conn = story_timings._get_conn()
        story_conn.execute("DELETE FROM story_phase_timings")
        story_conn.commit()

        daily_rollup.compute_and_write("2026-04-17", "TK")
        assert _get_phase_timings_json("2026-04-17", "TK") is None

    def test_phase_timings_does_not_clobber_throughput_columns(self):
        """Writing phase_timings_json must leave shipped/failed/cost_usd alone."""
        executor_runs_db.init_db()
        exec_conn = executor_runs_db._get_conn()
        exec_conn.execute(
            "INSERT INTO executor_runs (jira_key, status, cost_usd, "
            "duration_ms, started_at) VALUES (?, ?, ?, ?, ?)",
            ("TK-1", "success", 0.42, 5000, "2026-04-17T10:00:00"),
        )
        exec_conn.commit()

        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )

        daily_rollup.compute_and_write("2026-04-17", "TK")

        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT shipped, failed, cost_usd, phase_timings_json "
            "FROM daily_stats WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["shipped"] == 1
        assert row["failed"] == 0
        assert row["cost_usd"] == pytest.approx(0.42)
        assert row["phase_timings_json"] is not None


class TestHelperFunction:
    """Direct tests for the _compute_phase_timings_json helper."""

    def test_returns_none_when_no_rows(self):
        story_timings.init_db()
        assert daily_rollup._compute_phase_timings_json("2026-04-17", "TK") is None

    def test_returns_json_string_when_rows_present(self):
        _insert_phase(
            project="TK", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )
        blob = daily_rollup._compute_phase_timings_json("2026-04-17", "TK")
        assert isinstance(blob, str)
        parsed = json.loads(blob)
        assert parsed == {"plan": {"count": 1, "p50_ms": 1000, "p95_ms": 1000}}

    def test_returns_none_for_project_with_no_matching_rows(self):
        _insert_phase(
            project="FA", phase="plan",
            started_at="2026-04-17T10:00:00",
            duration_ms=1000,
        )
        assert daily_rollup._compute_phase_timings_json("2026-04-17", "TK") is None
