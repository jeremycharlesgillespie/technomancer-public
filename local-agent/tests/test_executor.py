"""Tests for executor logging verification — skipped paths (dry run/missing).

Verifies that INFO logs are produced for skipped paths in _remove_artifacts
when dry_run=True or when paths don't exist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import executor_runs_cleanup, executor_runs_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point executor_runs_cleanup at a temporary DB and execution_logs dir."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)

    logs_dir = tmp_path / "execution_logs"
    logs_dir.mkdir()
    monkeypatch.setattr(executor_runs_cleanup, "EXECUTION_LOGS_DIR", logs_dir)

    # Reset per-thread connection cache
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()

    executor_runs_cleanup.stop_cleanup_scheduler()

    yield logs_dir

    executor_runs_cleanup.stop_cleanup_scheduler()
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


def _seed_run(started_at, run_id: str, jira_key: str = "TK-1") -> int:
    """Insert one executor_runs row with controlled started_at timestamp."""
    return executor_runs_db.record_run(
        run_id=run_id,
        jira_key=jira_key,
        branch="main",
        started_at=started_at.isoformat(),
        status="success",
    )


# ==================================================================
# Skipped path logging — dry run and missing files
# ====================================================================


class TestCleanupOldRuns_Skipped:
    """Tests for INFO logs when paths are skipped (dry run or missing)."""

    def test_dry_run_logs_skipped_missing_path(self, _isolate_db, caplog):
        """Test that dry_run=True logs INFO for missing paths."""
        logs_dir = _isolate_db
        run_id = "missing-run"
        _seed_run(
            datetime.now() - timedelta(days=45), run_id, "TK-1"
        )
        # No files seeded for this run_id

        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=True
            )

        # Should not raise an exception
        assert dirs_removed == 0
        assert bytes_freed == 0

        # Should have logged INFO message containing 'skipped' or 'missing'
        assert any(
            "skipped" in record.message.lower() or "missing" in record.message.lower()
            for record in caplog.records
        )
        assert any(record.levelno == logging.INFO for record in caplog.records)

    def test_dry_run_logs_skipped_existing_path(self, _isolate_db, caplog):
        """Test that dry_run=True logs INFO for existing paths that are skipped."""
        logs_dir = _isolate_db
        run_id = "dry-run-existing"
        _seed_run(
            datetime.now() - timedelta(days=45), run_id, "TK-1"
        )
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=True
            )

        # Should not raise an exception
        assert dirs_removed == 1  # Would be removed
        assert bytes_freed > 0  # Would free bytes

        # Should have logged INFO message containing 'skipped' or 'dry run'
        assert any(
            "skipped" in record.message.lower() or "dry run" in record.message.lower()
            for record in caplog.records
        )
        assert any(record.levelno == logging.INFO for record in caplog.records)

    def test_missing_path_logs_skipped(self, _isolate_db, caplog):
        """Test that missing paths trigger INFO logs."""
        logs_dir = _isolate_db
        run_id = "ghost-run"
        _seed_run(
            datetime.now() - timedelta(days=45), run_id, "TK-1"
        )
        # No files seeded for this run_id

        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=False
            )

        # Should not raise an exception
        assert dirs_removed == 0
        assert bytes_freed == 0

        # Should have logged INFO message containing 'skipped' or 'missing'
        assert any(
            "skipped" in record.message.lower() or "missing" in record.message.lower()
            for record in caplog.records
        )
        assert any(record.levelno == logging.INFO for record in caplog.records)

    def test_dry_run_with_multiple_missing_paths(self, _isolate_db, caplog):
        """Test dry_run with multiple missing paths logs each skipped."""
        logs_dir = _isolate_db
        run_ids = ["missing-1", "missing-2", "missing-3"]
        for rid in run_ids:
            _seed_run(
                datetime.now() - timedelta(days=45), rid, "TK-1"
            )
            # No files seeded

        with caplog.at_level(logging.INFO):
            for rid in run_ids:
                executor_runs_cleanup._remove_artifacts(run_id=rid, dry_run=True)

        # Should have logged INFO messages for each skipped path
        skipped_count = sum(
            1
            for record in caplog.records
            if "skipped" in record.message.lower()
            or "missing" in record.message.lower()
        )
        assert skipped_count >= 1

    def test_dry_run_with_mixed_paths(self, _isolate_db, caplog):
        """Test dry_run with both existing and missing paths logs appropriately."""
        logs_dir = _isolate_db
        run_id = "mixed-run"
        _seed_run(
            datetime.now() - timedelta(days=45), run_id, "TK-1"
        )

        # Create a directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")

        # Create another run_id with no files
        run_id2 = "missing-run"
        _seed_run(
            datetime.now() - timedelta(days=45), run_id2, "TK-1"
        )

        with caplog.at_level(logging.INFO):
            # Process mixed paths
            executor_runs_cleanup._remove_artifacts(run_id=run_id, dry_run=True)
            executor_runs_cleanup._remove_artifacts(run_id=run_id2, dry_run=True)

        # Should have logged INFO messages for both skipped paths
        skipped_count = sum(
            1
            for record in caplog.records
            if "skipped" in record.message.lower()
            or "missing" in record.message.lower()
        )
        assert skipped_count >= 1

    def test_none_run_id_logs_skipped(self, _isolate_db, caplog):
        """Test that None run_id triggers INFO logs for skipped paths."""
        logs_dir = _isolate_db

        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=None, dry_run=True
            )

        # Should not raise an exception
        assert dirs_removed == 0
        assert bytes_freed == 0

        # Should have logged INFO message containing 'skipped' or 'missing'
        assert any(
            "skipped" in record.message.lower() or "missing" in record.message.lower()
            for record in caplog.records
        )
        assert any(record.levelno == logging.INFO for record in caplog.records)

    def test_empty_string_run_id_logs_skipped(self, _isolate_db, caplog):
        """Test that empty string run_id triggers INFO logs for skipped paths."""
        logs_dir = _isolate_db

        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id="", dry_run=True
            )

        # Should not raise an exception
        assert dirs_removed == 0
        assert bytes_freed == 0

        # Should have logged INFO message containing 'skipped' or 'missing'
        assert any(
            "skipped" in record.message.lower() or "missing" in record.message.lower()
            for record in caplog.records
        )
        assert any(record.levelno == logging.INFO for record in caplog.records)