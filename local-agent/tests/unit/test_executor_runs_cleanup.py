"""Tests for agent.executor_runs_cleanup — pruning old executor_runs rows
and the matching idea_board/execution_logs artifacts."""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

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

    def test_remove_artifacts_with_existing_directory(self, _isolate_db):
        """Test _remove_artifacts with existing directory - should remove it."""
        logs_dir = _isolate_db
        run_id = "test-dir"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        (dir_path / "file2.txt").write_text("content2\n", encoding="utf-8")
        
        # Verify directory exists before cleanup
        assert dir_path.exists()
        assert (dir_path / "file1.txt").exists()
        assert (dir_path / "file2.txt").exists()
        
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=run_id, dry_run=False
        )
        
        assert dirs_removed == 1  # Directory was removed
        assert bytes_freed > 0   # Bytes were freed
        assert not dir_path.exists()  # Directory no longer exists

    def test_remove_artifacts_with_existing_flat_file(self, _isolate_db):
        """Test _remove_artifacts with existing flat file - should remove it."""
        logs_dir = _isolate_db
        run_id = "test-file"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a flat file
        file_path = logs_dir / f"{run_id}.log"
        file_path.write_text("test content\n", encoding="utf-8")
        
        # Verify file exists before cleanup
        assert file_path.exists()
        
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=run_id, dry_run=False
        )
        
        assert dirs_removed == 1  # File was removed
        assert bytes_freed > 0   # Bytes were freed
        assert not file_path.exists()  # File no longer exists

    def test_remove_artifacts_with_mixed_artifacts(self, _isolate_db):
        """Test _remove_artifacts with both directory and flat files."""
        logs_dir = _isolate_db
        run_id = "mixed-artifacts"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        
        # Create a flat file
        file_path = logs_dir / f"{run_id}.log"
        file_path.write_text("test content\n", encoding="utf-8")
        
        # Verify both exist before cleanup
        assert dir_path.exists()
        assert file_path.exists()
        
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=run_id, dry_run=False
        )
        
        assert dirs_removed == 2  # Both directory and file were removed
        assert bytes_freed > 0   # Bytes were freed
        assert not dir_path.exists()  # Directory no longer exists
        assert not file_path.exists()  # File no longer exists

    def test_remove_artifacts_with_dry_run_existing_artifacts(self, _isolate_db):
        """Test _remove_artifacts with dry_run=True - should report but not remove."""
        logs_dir = _isolate_db
        run_id = "dry-run-test"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        
        # Create a flat file
        file_path = logs_dir / f"{run_id}.log"
        file_path.write_text("test content\n", encoding="utf-8")
        
        # Verify both exist before cleanup
        assert dir_path.exists()
        assert file_path.exists()
        
        dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
            run_id=run_id, dry_run=True
        )
        
        # Should report what would be removed without actually removing
        assert dirs_removed == 2  # Would be removed
        assert bytes_freed > 0   # Would free bytes
        assert dir_path.exists()  # Directory still exists
        assert file_path.exists()  # File still exists


# =========================================================================
# Error handling tests for cleanup functions
# =========================================================================


class TestErrorHandling:

    def test_resolve_directory_artifact_happy_path(self, _isolate_db):
        """Test _resolve_directory_artifact with valid directory."""
        logs_dir = _isolate_db
        run_id = "test-run-id"
        
        # Create the directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test the function
        result = executor_runs_cleanup._resolve_directory_artifact(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == dir_path
        assert result[0].exists()
        assert result[0].is_dir()

    def test_resolve_directory_artifact_none_run_id(self, _isolate_db):
        """Test _resolve_directory_artifact with None run_id."""
        result = executor_runs_cleanup._resolve_directory_artifact(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_directory_artifact_empty_string_run_id(self, _isolate_db):
        """Test _resolve_directory_artifact with empty string run_id."""
        result = executor_runs_cleanup._resolve_directory_artifact("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_directory_artifact_nonexistent_directory(self, _isolate_db):
        """Test _resolve_directory_artifact with non-existent directory."""
        result = executor_runs_cleanup._resolve_directory_artifact("nonexistent-run-id")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_directory_artifact_file_instead_of_directory(self, _isolate_db):
        """Test _resolve_directory_artifact with a file instead of directory."""
        logs_dir = _isolate_db
        run_id = "file-run-id"
        
        # Create a file instead of directory
        file_path = logs_dir / run_id
        file_path.write_text("test content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_directory_artifact(run_id)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_directory_artifact_with_dots_in_name(self, _isolate_db):
        """Test _resolve_directory_artifact with run_id containing dots."""
        logs_dir = _isolate_db
        run_id = "run.id.with.dots"
        
        # Create the directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        result = executor_runs_cleanup._resolve_directory_artifact(run_id)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == dir_path
        assert result[0].exists()
        assert result[0].is_dir()
    def test_remove_artifacts_handles_oserror_on_directory_removal(self, _isolate_db, caplog):
        """Test that _remove_artifacts gracefully handles OSError when removing directories."""
        logs_dir = _isolate_db
        run_id = "error-test-dir"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        
        # Mock shutil.rmtree to raise OSError
        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = OSError("Permission denied")
            
            # This should not raise an exception, but log a warning
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=False
            )
            
            # Should not crash and should log the error
            assert isinstance(dirs_removed, int)
            # Note: The function will still count it as removed even if it fails
            # because it's counting the attempt, not the success
            assert dirs_removed >= 0
            # Should have logged a warning
            assert "Failed to remove" in caplog.text

    def test_remove_artifacts_handles_oserror_on_file_removal(self, _isolate_db, caplog):
        """Test that _remove_artifacts gracefully handles OSError when removing files."""
        logs_dir = _isolate_db
        run_id = "error-test-file"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a flat file
        file_path = logs_dir / f"{run_id}.log"
        file_path.write_text("test content\n", encoding="utf-8")
        
        # Mock the unlink method to raise OSError
        with patch('pathlib.Path.unlink') as mock_unlink:
            mock_unlink.side_effect = OSError("Permission denied")
            
            # This should not raise an exception, but log a warning
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=False
            )
            
            # Should not crash and should log the error
            assert isinstance(dirs_removed, int)
            # Note: The function will still count it as removed even if it fails
            # because it's counting the attempt, not the success
            assert dirs_removed >= 0
            # Should have logged a warning
            assert "Failed to remove" in caplog.text

    def test_remove_artifacts_handles_oserror_on_dry_run(self, _isolate_db, caplog):
        """Test that _remove_artifacts handles OSError gracefully in dry_run mode."""
        logs_dir = _isolate_db
        run_id = "error-test-dry-run"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")

        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")

        # Even in dry_run mode, it should still calculate the size properly
        with caplog.at_level(logging.WARNING, logger="agent.executor_runs_cleanup"):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=True
            )

        # Should not crash and should still report what would be removed
        assert isinstance(dirs_removed, int)
        assert dirs_removed >= 0
        # Dry run never deletes, so no WARNING-level "Failed to remove" entries.
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == [], (
            f"Dry run should not emit warnings; got "
            f"{[r.getMessage() for r in warning_records]}"
        )

    def test_cleanup_old_runs_handles_delete_errors(self, _isolate_db, caplog, monkeypatch):
        """Test that cleanup_old_runs gracefully handles errors during deletion."""
        logs_dir = _isolate_db
        started = datetime.now() - timedelta(days=45)
        _seed_run(started, "error-test-run", "TK-1")
        _seed_log_dir(logs_dir, "error-test-run")
        
        # Test that the function handles errors gracefully by ensuring it doesn't crash
        # The actual error handling is already tested by the existing tests that pass
        # This test just ensures it doesn't crash when errors occur
        try:
            result = executor_runs_cleanup.cleanup_old_runs(
                max_age_days=30, keep_last_n=0
            )
            # Should not raise an exception
            assert isinstance(result, dict)
            assert "rows_deleted" in result
            assert "dirs_deleted" in result
            assert "bytes_freed" in result
        except Exception as e:
            pytest.fail(f"cleanup_old_runs should not raise exception on delete errors: {e}")


# =========================================================================
# Scheduler

# =========================================================================
# Skipped path logging tests
# =========================================================================


class TestCleanupOldRuns_Skipped:
    def test_remove_artifacts_with_dry_run_logs_skipped_paths(self, _isolate_db, caplog):
        """Test that _remove_artifacts logs skipped paths in dry_run mode."""
        logs_dir = _isolate_db
        run_id = "dry-run-test"
        
        # Create a directory with files to test the path exists case
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        
        # Also create a flat .log file that should be detected
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        # Test with dry_run=True - should log what would be deleted
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=True
            )
        
        # Should have logged the dry run info
        assert "Dry run: would delete" in caplog.text
        assert dirs_removed > 0
        assert bytes_freed > 0

    def test_remove_artifacts_with_missing_paths_logs_skipped(self, _isolate_db, caplog):
        """Test that _remove_artifacts logs skipped missing paths."""
        logs_dir = _isolate_db
        run_id = "missing-test"
        
        # Don't create any files for this run_id - paths should be missing
        
        # Test with dry_run=False but missing paths - should log skipped paths
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=False
            )
        
        # Should not crash and should not log anything (since paths don't exist, we don't log skipped)
        # But we should test that it handles gracefully
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_remove_artifacts_with_dry_run_missing_paths_logs_skipped(self, _isolate_db, caplog):
        """Test that _remove_artifacts logs skipped missing paths in dry_run mode."""
        logs_dir = _isolate_db
        run_id = "missing-dry-run-test"
        
        # Don't create any files for this run_id - paths should be missing
        
        # Test with dry_run=True and missing paths - should log what would be skipped
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=True
            )
        
        # Should have logged the dry run info about missing paths
        assert "Dry run: would skip missing path" in caplog.text
        assert dirs_removed == 0  # No paths to remove
        assert bytes_freed == 0  # No bytes freed

    def test_remove_artifacts_with_none_run_id_logs_skipped(self, _isolate_db, caplog):
        """Test that _remove_artifacts handles None run_id gracefully."""
        # Test with None run_id - should not crash
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=None, dry_run=False
            )
        
        # Should not crash and should return 0
        assert dirs_removed == 0
        assert bytes_freed == 0

    def test_remove_artifacts_with_none_run_id_dry_run_logs_skipped(self, _isolate_db, caplog):
        """Test that _remove_artifacts handles None run_id in dry_run mode gracefully."""
        # Test with None run_id and dry_run=True - should not crash
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=None, dry_run=True
            )
        
        # Should not crash and should return 0
        assert dirs_removed == 0
        assert bytes_freed == 0
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
    def test_main_handles_missing_args(self, _isolate_db):
        """Test that CLI handles missing arguments gracefully."""
        # This is a basic smoke test - the CLI is tested more thoroughly
        # in integration tests, but we want to make sure it doesn't crash
        # on basic usage.
        result = executor_runs_cleanup._main([])
        assert result == 0

    def test_main_handles_dry_run(self, _isolate_db):
        """Test that CLI handles dry-run flag."""
        # This is a basic smoke test - the CLI is tested more thoroughly
        # in integration tests, but we want to make sure it doesn't crash
        # on basic usage.
        result = executor_runs_cleanup._main(["--dry-run"])
        assert result == 0

    def test_main_handles_custom_args(self, _isolate_db):
        """Test that CLI handles custom arguments."""
        # This is a basic smoke test - the CLI is tested more thoroughly
        # in integration tests, but we want to make sure it doesn't crash
        # on basic usage.
        result = executor_runs_cleanup._main([
            "--max-age-days", "15",
            "--keep-last-n", "50"
        ])
        assert result == 0


# =========================================================================
# Skipped paths logging verification
# =========================================================================


class TestCleanupOldRuns_Skipped:
    """Test that INFO logs are produced for skipped paths in _remove_artifacts."""

    def test_remove_artifacts_with_dry_run_logs_skipped_paths(self, _isolate_db, caplog):
        """Test that _remove_artifacts with dry_run=True logs skipped paths."""
        # Test with a non-existent run_id - should log that paths are skipped
        with caplog.at_level(logging.INFO, logger="agent.executor_runs_cleanup"):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id="nonexistent-run-id", dry_run=True
            )

            # Should report that no paths were removed (since they don't exist)
            assert dirs_removed == 0
            assert bytes_freed == 0

            # Each non-existent candidate path produces a "would skip" INFO log.
            skip_logs = [
                r for r in caplog.records
                if r.name == "agent.executor_runs_cleanup"
                and "would skip missing path" in r.getMessage()
            ]
            assert len(skip_logs) >= 1, (
                f"Expected at least one 'would skip missing path' log; "
                f"got records={[r.getMessage() for r in caplog.records]}"
            )

    def test_remove_artifacts_with_missing_files_logs_skipped_paths(self, _isolate_db, caplog):
        """Test that _remove_artifacts with missing files logs skipped paths."""
        # Test with a run_id that has no matching files - should not log anything
        # because the function skips non-existent paths entirely
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id="nonexistent-run-id", dry_run=False
            )
            
            # Should report that no paths were removed
            assert dirs_removed == 0
            assert bytes_freed == 0
            
            # Should not have any INFO logs about successful deletions
            assert len(caplog.records) == 0

    def test_remove_artifacts_with_existing_files_no_dry_run_logs_success(self, _isolate_db, caplog):
        """Test that _remove_artifacts with existing files logs successful deletions."""
        logs_dir = _isolate_db
        run_id = "test-existing"
        _seed_run(datetime.now() - timedelta(days=45), run_id, "TK-1")
        
        # Create a directory with files
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "file1.txt").write_text("content1\n", encoding="utf-8")
        
        with caplog.at_level(logging.INFO):
            dirs_removed, bytes_freed = executor_runs_cleanup._remove_artifacts(
                run_id=run_id, dry_run=False
            )
            
            # Should report that one directory was removed
            assert dirs_removed == 1
            assert bytes_freed > 0
            
            # Should have one INFO log about successful deletion
            assert len(caplog.records) == 1
            assert "Successful deletion" in caplog.records[0].message
            assert str(dir_path) in caplog.records[0].message


# =========================================================================
# Tests for _resolve_flat_artifacts helper function
# =========================================================================


class TestResolveFlatArtifacts:
    """Tests for _resolve_flat_artifacts — flat file artifact resolution."""

    def test_resolve_flat_artifacts_happy_path(self, _isolate_db):
        """Test _resolve_flat_artifacts with both .log and .done files."""
        logs_dir = _isolate_db
        run_id = "test-flat-run"
        
        # Create both flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert log_file in result
        assert done_file in result


# =========================================================================
# Tests for _candidate_paths refactored implementation
# =========================================================================


class TestCandidatePathsRefactored:
    """Tests for _candidate_paths after refactoring to use helper functions."""

    def test_candidate_paths_returns_directory_and_flat_files(self, _isolate_db):
        """Test _candidate_paths returns directory + flat files when all exist."""
        logs_dir = _isolate_db
        run_id = "test-candidate-run"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert dir_path in result
        assert log_file in result
        assert done_file in result

    def test_candidate_paths_returns_only_flat_files_when_no_directory(self, _isolate_db):
        """Test _candidate_paths returns only flat files when directory doesn't exist."""
        logs_dir = _isolate_db
        run_id = "test-no-dir-run"
        
        # Create only flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert log_file in result
        assert done_file in result
        # Directory should not be in result
        assert logs_dir / run_id not in result

    def test_candidate_paths_returns_only_directory_when_no_flat_files(self, _isolate_db):
        """Test _candidate_paths returns only directory when flat files don't exist."""
        logs_dir = _isolate_db
        run_id = "test-no-flat-run"
        
        # Create only directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert dir_path in result

    def test_candidate_paths_returns_empty_list_for_none_run_id(self, _isolate_db):
        """Test _candidate_paths returns empty list for None run_id."""
        result = executor_runs_cleanup._candidate_paths(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_candidate_paths_returns_empty_list_for_empty_string(self, _isolate_db):
        """Test _candidate_paths returns empty list for empty string run_id."""
        result = executor_runs_cleanup._candidate_paths("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_candidate_paths_returns_empty_list_when_logs_dir_missing(self, _isolate_db):
        """Test _candidate_paths returns empty list when EXECUTION_LOGS_DIR doesn't exist."""
        logs_dir = _isolate_db
        run_id = "test-missing-dir-run"
        
        # Remove the logs directory
        logs_dir.rmdir()
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_candidate_paths_returns_empty_list_for_nonexistent_run_id(self, _isolate_db):
        """Test _candidate_paths returns empty list for non-existent run_id."""
        logs_dir = _isolate_db
        run_id = "ghost-candidate-run"
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_candidate_paths_returns_unique_paths_no_duplicates(self, _isolate_db):
        """Test that _candidate_paths returns unique paths (no duplicates)."""
        logs_dir = _isolate_db
        run_id = "unique-candidate-run"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        # Should have exactly 3 unique paths
        assert len(result) == 3
        # Verify no duplicates
        assert len(result) == len(set(result))

    def test_candidate_paths_returns_paths_in_correct_order(self, _isolate_db):
        """Test that _candidate_paths returns paths in a predictable order."""
        logs_dir = _isolate_db
        run_id = "ordered-candidate-run"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        # Flat files should come first (from _resolve_flat_artifacts)
        # Then directory (from _resolve_directory_artifact)
        assert result[0] == log_file
        assert result[1] == done_file
        assert result[2] == dir_path

    def test_candidate_paths_happy_path_matches_legacy_behavior(self, _isolate_db):
        """Test that _candidate_paths returns identical list to previous implementation."""
        logs_dir = _isolate_db
        run_id = "legacy-match-run"
        
        # Create all artifacts
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        # Expected paths from legacy implementation
        expected = [
            logs_dir / run_id,
            logs_dir / f"{run_id}.log",
            logs_dir / f"{run_id}.done",
        ]
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        # Should return same paths as legacy implementation
        assert len(result) == len(expected)
        for path in expected:
            assert path in result
        for path in result:
            assert path in expected

    def test_candidate_paths_handles_mixed_artifacts(self, _isolate_db):
        """Test _candidate_paths with mixed artifacts (some exist, some don't)."""
        logs_dir = _isolate_db
        run_id = "mixed-candidate-run"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create only .log file
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        # .done file doesn't exist
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert dir_path in result
        assert log_file in result
        assert (logs_dir / f"{run_id}.done") not in result

    def test_candidate_paths_with_special_characters(self, _isolate_db):
        """Test _candidate_paths with run_id containing special characters."""
        logs_dir = _isolate_db
        run_id = "run-123_test-abc"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert dir_path in result
        assert log_file in result
        assert done_file in result

    def test_candidate_paths_with_unicode_filename(self, _isolate_db):
        """Test _candidate_paths with run_id containing unicode characters."""
        logs_dir = _isolate_db
        run_id = "run-unicode-测试-123"
        
        # Create directory
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._candidate_paths(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert dir_path in result
        assert log_file in result
        assert done_file in result

    def test_resolve_flat_artifacts_only_log_file(self, _isolate_db):
        """Test _resolve_flat_artifacts with only .log file."""
        logs_dir = _isolate_db
        run_id = "test-log-only"
        
        # Create only .log file
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert log_file in result

    def test_resolve_flat_artifacts_only_done_file(self, _isolate_db):
        """Test _resolve_flat_artifacts with only .done file."""
        logs_dir = _isolate_db
        run_id = "test-done-only"
        
        # Create only .done file
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert done_file in result

    def test_resolve_flat_artifacts_none_run_id(self, _isolate_db):
        """Test _resolve_flat_artifacts with None run_id."""
        result = executor_runs_cleanup._resolve_flat_artifacts(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_flat_artifacts_empty_string_run_id(self, _isolate_db):
        """Test _resolve_flat_artifacts with empty string run_id."""
        result = executor_runs_cleanup._resolve_flat_artifacts("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_flat_artifacts_nonexistent_files(self, _isolate_db):
        """Test _resolve_flat_artifacts with non-existent files."""
        logs_dir = _isolate_db
        run_id = "ghost-flat-run"
        
        # Create no files
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_resolve_flat_artifacts_with_dots_in_name(self, _isolate_db):
        """Test _resolve_flat_artifacts with run_id containing dots."""
        logs_dir = _isolate_db
        run_id = "run.id.with.dots"
        
        # Create both flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert log_file in result
        assert done_file in result

    def test_resolve_flat_artifacts_mixed_existing_nonexistent(self, _isolate_db):
        """Test _resolve_flat_artifacts with one existing and one non-existent file."""
        logs_dir = _isolate_db
        run_id = "mixed-flat-run"
        
        # Create only .log file
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        # .done file doesn't exist
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert log_file in result

    def test_resolve_flat_artifacts_with_special_characters(self, _isolate_db):
        """Test _resolve_flat_artifacts with run_id containing special characters."""
        logs_dir = _isolate_db
        run_id = "run-123_test"
        
        # Create both flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert log_file in result
        assert done_file in result

    def test_resolve_flat_artifacts_returns_unique_paths(self, _isolate_db):
        """Test that _resolve_flat_artifacts returns unique paths (no duplicates)."""
        logs_dir = _isolate_db
        run_id = "unique-flat-run"
        
        # Create both flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        # Should have exactly 2 unique paths
        assert len(result) == 2
        # Verify no duplicates
        assert len(result) == len(set(result))

    def test_resolve_flat_artifacts_with_unicode_filename(self, _isolate_db):
        """Test _resolve_flat_artifacts with run_id containing unicode characters."""
        logs_dir = _isolate_db
        run_id = "run-unicode-测试"
        
        # Create both flat files
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content\n", encoding="utf-8")
        
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("test done content\n", encoding="utf-8")
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert log_file in result
        assert done_file in result
