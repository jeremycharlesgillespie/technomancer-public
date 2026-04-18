"""Tests for agent.daily_rollup.compute_and_write — core throughput/cost rollup."""

from __future__ import annotations

import runpy
import sys
from typing import Any

import pytest

from agent import daily_rollup, daily_stats, executor_runs_db, story_timings


@pytest.fixture(autouse=True)
def _isolate_dbs(tmp_path, monkeypatch):
    """Point all SQLite databases at temp paths and reset connection caches.

    compute_and_write touches three DBs: executor_runs (read),
    story_phase_timings (read), and daily_stats (write). Each module has
    its own per-thread connection cache that has to be cleared so the new
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


def _insert_run(**fields: Any) -> None:
    """Insert a single synthetic executor_runs row for test setup."""
    executor_runs_db.init_db()
    conn = executor_runs_db._get_conn()
    defaults: dict[str, Any] = {
        "status": "success",
        "started_at": "2026-04-17T10:00:00",
    }
    defaults.update(fields)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    conn.execute(
        f"INSERT INTO executor_runs ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


class TestPercentile:
    def test_empty_list_returns_zero(self):
        assert daily_rollup._percentile([], 50) == 0.0
        assert daily_rollup._percentile([], 95) == 0.0

    def test_single_value_returns_that_value(self):
        assert daily_rollup._percentile([2.5], 50) == 2.5
        assert daily_rollup._percentile([2.5], 95) == 2.5

    def test_p50_of_odd_sequence_is_middle(self):
        assert daily_rollup._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p95_uses_linear_interpolation(self):
        # numpy.percentile([1,2,3,4,5], 95) == 4.8
        assert daily_rollup._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8)

    def test_percentile_handles_unsorted_input(self):
        assert daily_rollup._percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0


class TestComputeAndWriteReturnValue:
    def test_acceptance_criteria_shipped_failed_cost_and_non_null_percentiles(self):
        """AC: shipped=3, failed=1, cost_usd=0.42, non-NULL (non-zero) percentiles."""
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="TK-2", status="success", cost_usd=0.15,
                    duration_ms=12_000, started_at="2026-04-17T11:00:00")
        _insert_run(jira_key="TK-3", status="success", cost_usd=0.12,
                    duration_ms=20_000, started_at="2026-04-17T12:00:00")
        _insert_run(jira_key="TK-4", status="failed", cost_usd=0.05,
                    duration_ms=3_000, started_at="2026-04-17T13:00:00")

        result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["shipped"] == 3
        assert result["failed"] == 1
        assert result["cost_usd"] == pytest.approx(0.42)
        assert result["p50_wall_s"] > 0.0
        assert result["p95_wall_s"] > 0.0
        assert result["date"] == "2026-04-17"
        assert result["project"] == "TK"

    def test_empty_day_produces_zeros(self):
        executor_runs_db.init_db()
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result == {
            "date": "2026-04-17",
            "project": "TK",
            "shipped": 0,
            "failed": 0,
            "cost_usd": 0.0,
            "p50_wall_s": 0.0,
            "p95_wall_s": 0.0,
        }

    def test_null_cost_and_duration_handled(self):
        _insert_run(jira_key="TK-1", status="success",
                    cost_usd=None, duration_ms=None,
                    started_at="2026-04-17T10:00:00")
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["shipped"] == 1
        assert result["cost_usd"] == 0.0
        assert result["p50_wall_s"] == 0.0
        assert result["p95_wall_s"] == 0.0


class TestComputeAndWriteFilters:
    def test_excludes_different_project(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="FA-1", status="success", cost_usd=0.50,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")

        result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["shipped"] == 1
        assert result["cost_usd"] == pytest.approx(0.10)

    def test_excludes_different_date(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-16T23:59:00")
        _insert_run(jira_key="TK-2", status="success", cost_usd=0.20,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")

        result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["shipped"] == 1
        assert result["cost_usd"] == pytest.approx(0.20)

    def test_excludes_runs_with_null_started_at(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at=None)
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["shipped"] == 0

    def test_counts_every_terminal_failure_status(self):
        for status in ("failed", "error", "crashed", "killed", "timeout"):
            _insert_run(
                jira_key=f"TK-{status}", status=status, cost_usd=0.01,
                duration_ms=1_000, started_at="2026-04-17T10:00:00",
            )
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["failed"] == 5
        assert result["shipped"] == 0

    def test_ignores_running_or_queued_statuses(self):
        """In-flight rows shouldn't inflate either throughput bucket."""
        _insert_run(jira_key="TK-1", status="running", cost_usd=0.00,
                    duration_ms=None, started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="TK-2", status="queued", cost_usd=0.00,
                    duration_ms=None, started_at="2026-04-17T10:00:00")
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["shipped"] == 0
        assert result["failed"] == 0

    def test_handles_fractional_second_timestamp(self):
        """SQLite date() must still parse microsecond timestamps."""
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.01,
                    duration_ms=1_000,
                    started_at="2026-04-17T10:00:00.123456")
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["shipped"] == 1


class TestDailyStatsWrite:
    def test_writes_a_row_to_daily_stats(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")

        daily_rollup.compute_and_write("2026-04-17", "TK")

        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT * FROM daily_stats WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row is not None
        assert row["shipped"] == 1
        assert row["failed"] == 0
        assert row["cost_usd"] == pytest.approx(0.10)
        assert row["p50_wall_s"] == pytest.approx(5.0)
        assert row["p95_wall_s"] == pytest.approx(5.0)

    def test_other_columns_retain_zero_defaults(self):
        """LOC / first-attempt / splitter columns belong to future stories."""
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        daily_rollup.compute_and_write("2026-04-17", "TK")
        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT split_children, loc_added, loc_removed, "
            "first_attempt_success, splitter_child_success, splitter_child_fail "
            "FROM daily_stats WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["split_children"] == 0
        assert row["loc_added"] == 0
        assert row["loc_removed"] == 0
        assert row["first_attempt_success"] == 0
        assert row["splitter_child_success"] == 0
        assert row["splitter_child_fail"] == 0


class TestIdempotency:
    def test_rerun_does_not_create_duplicate_rows(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")

        daily_rollup.compute_and_write("2026-04-17", "TK")
        daily_rollup.compute_and_write("2026-04-17", "TK")
        daily_rollup.compute_and_write("2026-04-17", "TK")

        conn = daily_stats._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM daily_stats "
            "WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()["n"]
        assert count == 1

    def test_rerun_updates_values_when_underlying_runs_change(self):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        first = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert first["shipped"] == 1
        assert first["cost_usd"] == pytest.approx(0.10)

        _insert_run(jira_key="TK-2", status="success", cost_usd=0.25,
                    duration_ms=10_000, started_at="2026-04-17T11:00:00")
        _insert_run(jira_key="TK-3", status="failed", cost_usd=0.00,
                    duration_ms=1_000, started_at="2026-04-17T12:00:00")

        second = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert second["shipped"] == 2
        assert second["failed"] == 1
        assert second["cost_usd"] == pytest.approx(0.35)

        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT shipped, failed, cost_usd FROM daily_stats "
            "WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["shipped"] == 2
        assert row["failed"] == 1
        assert row["cost_usd"] == pytest.approx(0.35)


class TestCLI:
    def test_main_happy_path(self, capsys):
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.42,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")

        rc = daily_rollup._main(["--date", "2026-04-17", "--project", "TK"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "2026-04-17" in out
        assert "TK" in out
        assert "shipped=1" in out
        assert "cost=$0.4200" in out

    def test_main_requires_date(self):
        with pytest.raises(SystemExit):
            daily_rollup._main(["--project", "TK"])

    def test_main_requires_project(self):
        with pytest.raises(SystemExit):
            daily_rollup._main(["--date", "2026-04-17"])

    def test_module_run_as_main_invokes_cli(self, monkeypatch, capsys):
        """`python -m agent.daily_rollup ...` must wire __main__ to the CLI."""
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        monkeypatch.setattr(
            sys, "argv",
            ["agent.daily_rollup", "--date", "2026-04-17", "--project", "TK"],
        )
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("agent.daily_rollup", run_name="__main__")
        assert excinfo.value.code == 0
        assert "shipped=1" in capsys.readouterr().out
