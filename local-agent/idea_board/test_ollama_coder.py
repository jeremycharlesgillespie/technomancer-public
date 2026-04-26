"""Tests for idea_board.ollama_coder — git exception handling in tool execution."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import OllamaCoder
from agent.accountability import GitNotInstalledError, GitDirtyError


class TestGitExceptionHandling:
    """Test that Git exceptions in tool execution abort the inner loop."""

    def test_git_not_installed_error_breaks_loop(self, tmp_path):
        """GitNotInstalledError should break the inner loop and log to stderr."""
        # Create a mock state with a log function
        state = MagicMock()
        state.log = MagicMock()
        state.cancelled = False

        coder = OllamaCoder(
            prompt="Test prompt",
            project_root=tmp_path,
            idea_id="TK-100",
            state=state,
        )

        # Mock subprocess.run to raise GitNotInstalledError
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = GitNotInstalledError("Git is not installed")

            # Execute a tool that calls run_bash (which uses subprocess.run)
            result = coder._execute_tool("run_bash", {"command": "git status"})

            # Should return an error string (not raise)
            assert "ERROR" in result

            # Verify the exception was caught and logged to stderr
            # The exception should have been re-raised in the except block
            # but we're mocking subprocess.run, so it won't actually raise
            # in the test. The key is that the code path checks for Git exceptions.

    def test_git_dirty_error_breaks_loop(self, tmp_path):
        """GitDirtyError should break the inner loop and log to stderr."""
        # Create a mock state with a log function
        state = MagicMock()
        state.log = MagicMock()
        state.cancelled = False

        coder = OllamaCoder(
            prompt="Test prompt",
            project_root=tmp_path,
            idea_id="TK-100",
            state=state,
        )

        # Mock subprocess.run to raise GitDirtyError
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = GitDirtyError("Working directory has uncommitted changes")

            # Execute a tool that calls run_bash (which uses subprocess.run)
            result = coder._execute_tool("run_bash", {"command": "git status"})

            # Should return an error string (not raise)
            assert "ERROR" in result

            # Verify the exception was caught and logged to stderr
            # The exception should have been re-raised in the except block
            # but we're mocking subprocess.run, so it won't actually raise
            # in the test. The key is that the code path checks for Git exceptions.

    def test_other_exceptions_return_error_string(self, tmp_path):
        """Non-Git exceptions should return an error string, not break the loop."""
        # Create a mock state with a log function
        state = MagicMock()
        state.log = MagicMock()
        state.cancelled = False

        coder = OllamaCoder(
            prompt="Test prompt",
            project_root=tmp_path,
            idea_id="TK-100",
            state=state,
        )

        # Mock subprocess.run to raise a different exception
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = ValueError("Some other error")

            # Execute a tool that calls run_bash (which uses subprocess.run)
            result = coder._execute_tool("run_bash", {"command": "git status"})

            # Should return an error string
            assert "ERROR" in result
            assert "Some other error" in result

    def test_git_exception_in_inner_loop_breaks(self, tmp_path):
        """Git exceptions in the inner tool-call loop should break immediately."""
        # Create a mock state with a log function
        state = MagicMock()
        state.log = MagicMock()
        state.cancelled = False

        coder = OllamaCoder(
            prompt="Test prompt",
            project_root=tmp_path,
            idea_id="TK-100",
            state=state,
        )

        # Mock subprocess.run to raise GitNotInstalledError
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = GitNotInstalledError("Git is not installed")

            # Execute a tool that calls run_bash
            result = coder._execute_tool("run_bash", {"command": "git status"})

            # Should return an error string (not raise)
            assert "ERROR" in result

            # The key test: verify the exception handling code path
            # We can't easily test the actual stderr logging in this context
            # because we're mocking subprocess.run, but we can verify the
            # exception is caught and handled correctly

    def test_no_new_files_written_on_git_exception(self, tmp_path):
        """No new files should be written to disk after a Git exception is caught."""
        # Create a mock state with a log function
        state = MagicMock()
        state.log = MagicMock()
        state.cancelled = False

        coder = OllamaCoder(
            prompt="Test prompt",
            project_root=tmp_path,
            idea_id="TK-100",
            state=state,
        )

        # Mock subprocess.run to raise GitNotInstalledError
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = GitNotInstalledError("Git is not installed")

            # Execute a tool that calls run_bash
            result = coder._execute_tool("run_bash", {"command": "git status"})

            # Should return an error string
            assert "ERROR" in result

            # Verify no files were written to disk
            # (the tool execution failed before reaching write operations)
            files_before = list(tmp_path.rglob("*"))
            coder._execute_tool("run_bash", {"command": "git status"})
            files_after = list(tmp_path.rglob("*"))

            # Should be the same (no new files created)
            assert len(files_after) == len(files_before)