"""Tests for the _resolve_flat_artifacts helper function in executor_runs_cleanup."""

import pytest
from pathlib import Path

from agent import executor_runs_cleanup


class TestResolveFlatArtifacts:
    """Test the _resolve_flat_artifacts helper function."""

    def test_happy_path_with_existing_files(self, tmp_path):
        """Test that function returns paths for existing .log and .done files."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        # Create test files
        run_id = "test-run-id"
        log_file = logs_dir / f"{run_id}.log"
        done_file = logs_dir / f"{run_id}.done"
        
        log_file.write_text("log content\n", encoding="utf-8")
        done_file.write_text("done content\n", encoding="utf-8")
        
        # Test the function
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == log_file
        assert result[1] == done_file
        assert result[0].exists()
        assert result[1].exists()

    def test_with_only_log_file(self, tmp_path):
        """Test that function returns path for only the .log file when .done doesn't exist."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        # Create only log file
        run_id = "test-run-id"
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("log content\n", encoding="utf-8")
        
        # Test the function
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == log_file
        assert result[0].exists()

    def test_with_only_done_file(self, tmp_path):
        """Test that function returns path for only the .done file when .log doesn't exist."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        # Create only done file
        run_id = "test-run-id"
        done_file = logs_dir / f"{run_id}.done"
        done_file.write_text("done content\n", encoding="utf-8")
        
        # Test the function
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == done_file
        assert result[0].exists()

    def test_with_no_files(self, tmp_path):
        """Test that function returns empty list when no files exist."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        # Test the function with non-existent run_id
        result = executor_runs_cleanup._resolve_flat_artifacts("nonexistent-run-id")
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_none_run_id(self, tmp_path):
        """Test that function returns empty list when run_id is None."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        result = executor_runs_cleanup._resolve_flat_artifacts(None)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_empty_string_run_id(self, tmp_path):
        """Test that function returns empty list when run_id is empty string."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        result = executor_runs_cleanup._resolve_flat_artifacts("")
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_nonexistent_directory(self, tmp_path):
        """Test that function returns empty list when execution logs directory doesn't exist."""
        # Set up a non-existent directory
        logs_dir = tmp_path / "nonexistent" / "execution_logs"
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        result = executor_runs_cleanup._resolve_flat_artifacts("test-run-id")
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_directory_but_no_files(self, tmp_path):
        """Test that function returns empty list when directory exists but no files do."""
        # Set up the execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        executor_runs_cleanup.EXECUTION_LOGS_DIR = logs_dir
        
        # Create directory but no files
        run_id = "test-run-id"
        
        result = executor_runs_cleanup._resolve_flat_artifacts(run_id)
        
        assert isinstance(result, list)
        assert len(result) == 0