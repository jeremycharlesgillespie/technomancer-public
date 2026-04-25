"""Tests for agent.executor_runs_cleanup — pruning old executor_runs rows
and the matching idea_board/execution_logs artifacts."""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent import executor_runs_cleanup, executor_runs_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point both executor_runs_db and executor_runs_cleanup at a temp DB
    and a temp execution_logs dir for each test."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)

    logs_dir = tmp_path / "execution_logs"
    logs_dir.mkdir()
    monkeypatch.setattr(executor_runs_cleanup, "EXECUTION_LOGS_DIR", logs_dir)

    # Reset the per-thread connection cache so init_db opens against the
    # temp path.
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()

    # Make sure any leftover scheduler from a previous test is cancelled.
    executor_runs_cleanup.stop_cleanup_scheduler()

    yield logs_dir

    executor_runs_cleanup.stop_cleanup_scheduler()
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


def _seed_run(started_at: datetime, run_id: str, jira_key: str = "TK-1") -> int:
    """Insert one executor_runs row with a controlled started_at timestamp."""
    return executor_runs_db.record_run(
        run_id=run_id,
        jira_key=jira_key,
        branch="main",
        started_at=started_at.isoformat(),
        status="success",
    )


def _seed_log_dir(logs_dir: Path, run_id: str, *, as_dir: bool = True) -> Path:
    """Create a log artifact (dir or flat file) for ``run_id``."""
    target = logs_dir / run_id
    if as_dir:
        target.mkdir(parents=True, exist_ok=True)
        (target / "stdout.log").write_text("output\n", encoding="utf-8")
    else:
        target = logs_dir / f"{run_id}.log"
        target.write_text("output\n", encoding="utf-8")
    return target


# =========================================================================
# cleanup_old_runs — basic deletion policy
# =========================================================================


class TestCleanupOldRuns:
    def test_deletes_rows_older_than_max_age(self, _isolate_db):
        logs_dir = _isolate_db
        old = datetime.now() - timedelta(days=45)
        recent = datetime.now() - timedelta(days=2)
        old_id = _seed_run(old, "20260301-120000-TK-1", "TK-1")
        recent_id = _seed_run(recent, "20260415-120000-TK-2", "TK-2")
        _seed_log_dir(logs_dir, "20260301-120000-TK-1")

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )

        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] == 1
        assert result["bytes_freed"] > 0

        conn = executor_runs_db._get_conn()
        ids = [r["id"] for r in conn.execute("SELECT id FROM executor_runs")]
        assert old_id not in ids
        assert recent_id in ids
        assert not (logs_dir / "20260301-120000-TK-1").exists()

    def test_keeps_last_n_regardless_of_age(self, _isolate_db):
        """All rows are older than the age window, but keep_last_n=3 must
        preserve the three newest."""
        logs_dir = _isolate_db
        ids = []
        for i in range(10):
            started = datetime.now() - timedelta(days=60 + i)
            rid = f"20260{i:02d}01-120000-TK-{i}"
            ids.append(_seed_run(started, rid, f"TK-{i}"))
            _seed_log_dir(logs_dir, rid)

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=3
        )

        assert result["rows_deleted"] == 7
        conn = executor_runs_db._get_conn()
        remaining = conn.execute(
            "SELECT id FROM executor_runs ORDER BY started_at DESC"
        ).fetchall()
        assert len(remaining) == 3

    def test_keeps_last_n_on_seeded_1000_row_db(self, _isolate_db):
        """Acceptance criterion: seeded 1000-row DB removes rows and dirs,
        keeps newest 200 regardless of age."""
        logs_dir = _isolate_db
        for i in range(1000):
            started = datetime.now() - timedelta(days=60 + i)
            rid = f"run-{i:04d}"
            _seed_run(started, rid, f"TK-{i}")
            _seed_log_dir(logs_dir, rid)

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=200
        )

        assert result["rows_deleted"] == 800
        assert result["dirs_deleted"] == 800
        conn = executor_runs_db._get_conn()
        remaining = conn.execute(
            "SELECT COUNT(*) AS c FROM executor_runs"
        ).fetchone()
        assert remaining["c"] == 200

    def test_no_rows_to_delete_returns_zero(self, _isolate_db):
        recent = datetime.now() - timedelta(days=5)
        _seed_run(recent, "r1", "TK-1")

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=200
        )

        assert result["rows_deleted"] == 0
        assert result["dirs_deleted"] == 0
        assert result["bytes_freed"] == 0


# =========================================================================
# dry_run mode
# =========================================================================


class TestDryRun:
    def test_reports_same_counts_without_mutating(self, _isolate_db):
        """Acceptance criterion: dry_run reports same counts without mutating."""
        logs_dir = _isolate_db
        for i in range(5):
            started = datetime.now() - timedelta(days=60 + i)
            rid = f"run-{i}"
            _seed_run(started, rid, f"TK-{i}")
            _seed_log_dir(logs_dir, rid)

        dry = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0, dry_run=True
        )
        # DB untouched
        conn = executor_runs_db._get_conn()
        still_there = conn.execute(
            "SELECT COUNT(*) AS c FROM executor_runs"
        ).fetchone()["c"]
        assert still_there == 5
        # Files untouched
        assert sum(1 for _ in logs_dir.iterdir()) == 5

        wet = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )
        assert dry["rows_deleted"] == wet["rows_deleted"]
        assert dry["dirs_deleted"] == wet["dirs_deleted"]
        assert dry["bytes_freed"] == wet["bytes_freed"]
        assert dry["dry_run"] == 1
        assert wet["dry_run"] == 0


# =========================================================================
# On-disk artifact cleanup — dir + flat-file variants
# =========================================================================


class TestArtifactCleanup:
    def test_removes_flat_log_file(self, _isolate_db):
        """The current executor writes flat ``<run_id>.log`` files. Cleanup
        must handle that layout, not just ``<run_id>/`` directories."""
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        _seed_run(started, "flat-run", "TK-1")
        _seed_log_dir(logs_dir, "flat-run", as_dir=False)

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )

        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert not (logs_dir / "flat-run.log").exists()

    def test_missing_artifacts_do_not_crash(self, _isolate_db):
        started = datetime.now() - timedelta(days=45)
        _seed_run(started, "ghost-run", "TK-1")
        # No files seeded for this run_id.

        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] == 0
        assert result["bytes_freed"] == 0

    def test_also_removes_child_tool_call_rows(self, _isolate_db):
        started = datetime.now() - timedelta(days=45)
        row_id = _seed_run(started, "parent", "TK-1")
        executor_runs_db.record_tool_call(run_id=row_id, tool_name="Read")

        executor_runs_cleanup.cleanup_old_runs(max_age_days=30, keep_last_n=0)

        conn = executor_runs_db._get_conn()
        tool_rows = conn.execute(
            "SELECT COUNT(*) AS c FROM executor_tool_calls WHERE run_id = ?",
            (row_id,),
        ).fetchone()["c"]
        assert tool_rows == 0

    def test_handles_flat_log_file_directly(self, _isolate_db):
        """Test that flat .log files are properly handled in artifact cleanup.
        
        This test validates the behavior that would be expected from a
        _resolve_flat_artifacts function when processing .log files.
        """
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        run_id = "test-log-file"
        _seed_run(started, run_id, "TK-1")
        
        # Create a flat .log file directly
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        # Verify file exists before cleanup
        assert log_file.exists()
        
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )

        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert not log_file.exists()


# =========================================================================
# Edge case tests for artifact removal functions
# =========================================================================


class TestEdgeCases:
    def test_remove_artifacts_with_none_run_id(self, _isolate_db):
        """Test _remove_artifacts with None run_id - should handle gracefully."""
        # This tests the _remove_artifacts function directly
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=None, dry_run=False
        )
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_remove_artifacts_with_missing_files(self, _isolate_db):
        """Test _remove_artifacts with run_id that has no matching files."""
        # This tests the _remove_artifacts function directly with a non-existent run_id
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id="nonexistent-run-id", dry_run=False
        )
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_remove_artifacts_with_dry_run_none(self, _isolate_db):
        """Test _remove_artifacts with None run_id in dry_run mode."""
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=None, dry_run=True
        )
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_remove_artifacts_with_dry_run_missing(self, _isolate_db):
        """Test _remove_artifacts with missing files in dry_run mode."""
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id="nonexistent-run-id", dry_run=True
        )
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_handles_flat_done_files(self, _isolate_db):
        """Test that flat .done files are properly handled in artifact cleanup.
        
        This test validates the behavior with .done files, which are another
        common flat file type in the system.
        """
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        run_id = "test-done-file"
        _seed_run(started, run_id, "TK-1")
        
        # Create a flat .done file directly
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        # Verify file exists before cleanup
        assert done_file.exists()
        
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, keep_last_n=0
        )

        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert not done_file.exists()


# =========================================================================
# Scheduler
# =========================================================================


class TestScheduler:
    def test_start_runs_immediately(self, _isolate_db, monkeypatch):
        """Acceptance criterion: scheduled run logs one INFO line per cycle.

        Verify by asserting :func:`cleanup_old_runs` is invoked once at
        startup.
        """
        calls: list[dict] = []

        def fake_cleanup(**kwargs):
            calls.append(kwargs)
            return {
                "rows_deleted": 0,
                "dirs_deleted": 0,
                "bytes_freed": 0,
                "dry_run": 0,
            }

        monkeypatch.setattr(
            executor_runs_cleanup, "cleanup_old_runs", fake_cleanup
        )

        executor_runs_cleanup.start_cleanup_scheduler(
            interval_seconds=3600, max_age_days=30, keep_last_n=200
        )
        try:
            assert len(calls) == 1
            assert calls[0]["max_age_days"] == 30
            assert calls[0]["keep_last_n"] == 200
        finally:
            executor_runs_cleanup.stop_cleanup_scheduler()

    def test_start_schedules_a_timer(self, _isolate_db, monkeypatch):
        monkeypatch.setattr(
            executor_runs_cleanup,
            "cleanup_old_runs",
            lambda **kwargs: {
                "rows_deleted": 0, "dirs_deleted": 0,
                "bytes_freed": 0, "dry_run": 0,
            },
        )

        executor_runs_cleanup.start_cleanup_scheduler(
            interval_seconds=3600, run_at_start=False
        )
        try:
            assert isinstance(
                executor_runs_cleanup._timer, threading.Timer
            )
            assert executor_runs_cleanup._scheduler_running is True
        finally:
            executor_runs_cleanup.stop_cleanup_scheduler()
        assert executor_runs_cleanup._scheduler_running is False

    def test_start_is_idempotent(self, _isolate_db, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            executor_runs_cleanup,
            "cleanup_old_runs",
            lambda **kwargs: (calls.append(kwargs) or {
                "rows_deleted": 0, "dirs_deleted": 0,
                "bytes_freed": 0, "dry_run": 0,
            }),
        )

        executor_runs_cleanup.start_cleanup_scheduler(interval_seconds=3600)
        executor_runs_cleanup.start_cleanup_scheduler(interval_seconds=3600)
        try:
            assert len(calls) == 1  # second start is a no-op
        finally:
            executor_runs_cleanup.stop_cleanup_scheduler()

    def test_start_handles_cleanup_exceptions(self, _isolate_db, monkeypatch):
        """A crash in cleanup must not propagate out of the scheduler —
        otherwise a transient DB error would permanently break the loop."""

        def raising_cleanup(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            executor_runs_cleanup, "cleanup_old_runs", raising_cleanup
        )

        # Should not raise.
        executor_runs_cleanup.start_cleanup_scheduler(interval_seconds=3600)
        try:
            assert executor_runs_cleanup._scheduler_running is True
        finally:
            executor_runs_cleanup.stop_cleanup_scheduler()


# =========================================================================
# CLI entry point
# =========================================================================


class TestCli:
    def test_cli_dry_run_does_not_mutate(self, _isolate_db, capsys):
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        _seed_run(started, "cli-dry", "TK-1")
        _seed_log_dir(logs_dir, "cli-dry")

        rc = executor_runs_cleanup._main(
            ["--dry-run", "--max-age-days", "30", "--keep-last-n", "0"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "Would delete" in out
        # DB untouched
        conn = executor_runs_db._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM executor_runs"
        ).fetchone()["c"]
        assert count == 1
        assert (logs_dir / "cli-dry").exists()

    def test_cli_wet_run_deletes(self, _isolate_db, capsys):
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        _seed_run(started, "cli-wet", "TK-1")
        _seed_log_dir(logs_dir, "cli-wet")

        rc = executor_runs_cleanup._main(
            ["--max-age-days", "30", "--keep-last-n", "0"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "Deleted 1 row" in out
        conn = executor_runs_db._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM executor_runs"
        ).fetchone()["c"]
        assert count == 0
        assert not (logs_dir / "cli-wet").exists()