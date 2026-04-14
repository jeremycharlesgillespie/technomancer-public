"""Tests for aim.worker — AI Worker state machine and execution watcher."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aim.worker import WatchResult, _check_git_clean, execute_assigned_idea, watch_execution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect state files for all tests."""
    monkeypatch.setattr("aim.state.STATE_FILE", tmp_path / ".aim_state.json")
    monkeypatch.setattr("aim.state.LOCK_FILE", tmp_path / ".aim_state.lock")
    monkeypatch.setattr("aim.state.PID_FILE", tmp_path / "aim.pid")

    from filelock import FileLock
    monkeypatch.setattr("aim.state._lock", FileLock(str(tmp_path / ".aim_state.lock"), timeout=10))

    # Initialize state
    from aim.state import AIMState, save_state
    save_state(AIMState())


class FakeExecutionState:
    """Mimics the real ExecutionState for testing."""

    def __init__(self, idea_id="idea-001", pid=1234, log_lines=None,
                 alive=True, elapsed=0.0):
        self.idea_id = idea_id
        self.pid = pid
        self.log_lines = log_lines if log_lines is not None else []
        self._alive = alive
        self._elapsed = elapsed

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def elapsed(self) -> float:
        return self._elapsed


class FakeIdea:
    def __init__(self, id="idea-001", title="Test idea", state="done",
                 execution_log="Success"):
        self.id = id
        self.title = title
        self.state = state
        self.execution_log = execution_log


# ---------------------------------------------------------------------------
# WatchResult tests
# ---------------------------------------------------------------------------

class TestWatchResult:
    def test_success(self):
        r = WatchResult(success=True, summary="All good")
        assert r.success is True
        assert r.summary == "All good"

    def test_failure(self):
        r = WatchResult(success=False, summary="Tests failed")
        assert r.success is False


# ---------------------------------------------------------------------------
# _check_git_clean tests
# ---------------------------------------------------------------------------

class TestCheckGitClean:
    @patch("aim.worker.subprocess.run")
    def test_clean_on_main(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),  # git status --porcelain
            MagicMock(stdout="main\n"),  # git rev-parse
        ]
        assert _check_git_clean() is True

    @patch("aim.worker.subprocess.run")
    def test_dirty_working_dir(self, mock_run):
        # --untracked-files=no means only tracked file changes are reported
        mock_run.side_effect = [
            MagicMock(stdout="M agent/core.py\n"),
        ]
        assert _check_git_clean() is False

    @patch("aim.worker.subprocess.run")
    def test_not_on_main(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),
            MagicMock(stdout="feature-branch\n"),
        ]
        assert _check_git_clean() is False

    @patch("aim.worker.subprocess.run")
    def test_git_error(self, mock_run):
        mock_run.side_effect = OSError("git not found")
        assert _check_git_clean() is False


# ---------------------------------------------------------------------------
# watch_execution tests
# ---------------------------------------------------------------------------

class TestWatchExecution:
    @patch("aim.worker.time.sleep")
    @patch("aim.worker.WATCH_INTERVAL", 0)
    def test_success_when_idea_done(self, mock_sleep):
        alive_calls = iter([True, True, False])

        class DyingState(FakeExecutionState):
            @property
            def is_alive(self):
                return next(alive_calls, False)

        fake_state = DyingState(
            idea_id="idea-001",
            log_lines=["Line 1", "Line 2", "Line 3"],
        )

        with patch("idea_board.executor.get_execution", return_value=fake_state), \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"), \
             patch("idea_board.models.get_idea", return_value=FakeIdea(state="done")):
            result = watch_execution("idea-001")

        assert result.success is True
        assert "successfully" in result.summary.lower()

    @patch("aim.worker.time.sleep")
    @patch("aim.worker.WATCH_INTERVAL", 0)
    def test_failure_when_idea_failed(self, mock_sleep):
        fake_state = FakeExecutionState(idea_id="idea-001")
        fake_state._alive = False  # Already dead

        with patch("idea_board.executor.get_execution", return_value=fake_state), \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"), \
             patch("idea_board.models.get_idea",
                   return_value=FakeIdea(state="failed", execution_log="Tests failed")):
            result = watch_execution("idea-001")

        assert result.success is False
        assert "failed" in result.summary.lower() or "Tests failed" in result.summary

    def test_no_execution_state(self):
        with patch("idea_board.executor.get_execution", return_value=None), \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"):
            result = watch_execution("idea-999")

        assert result.success is False
        assert "No execution state" in result.summary

    @patch("aim.worker.time.sleep")
    @patch("aim.worker.WATCH_INTERVAL", 0)
    @patch("aim.worker.STALE_THRESHOLD", 1)
    def test_stall_detection_cancels(self, mock_sleep):
        """If no new log lines for STALE_THRESHOLD seconds, cancel."""
        cancel_called = [False]

        class StallableState(FakeExecutionState):
            @property
            def is_alive(self):
                return not cancel_called[0]

        fake_state = StallableState(idea_id="idea-001", log_lines=["Initial line"])

        def fake_cancel(idea_id):
            cancel_called[0] = True
            return True

        # time.time() calls: stale_since=None -> set to 0, then 0+2 > threshold(1)
        time_values = iter([0.0, 2.0, 3.0, 4.0, 5.0])

        with patch("idea_board.executor.get_execution", return_value=fake_state), \
             patch("idea_board.executor.cancel_execution", side_effect=fake_cancel) as mock_cancel, \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.time", side_effect=time_values):
            result = watch_execution("idea-001")

        assert result.success is False
        assert "stall" in result.summary.lower() or "no output" in result.summary.lower()
        mock_cancel.assert_called_once_with("idea-001")

    @patch("aim.worker.time.sleep")
    @patch("aim.worker.WATCH_INTERVAL", 0)
    def test_timeout_detection_cancels(self, mock_sleep):
        cancel_called = [False]

        class AlwaysAliveState(FakeExecutionState):
            @property
            def is_alive(self):
                return not cancel_called[0]

        fake_state = AlwaysAliveState(
            idea_id="idea-001", log_lines=["Line"], elapsed=99999,
        )

        def fake_cancel(idea_id):
            cancel_called[0] = True
            return True

        with patch("idea_board.executor.get_execution", return_value=fake_state), \
             patch("idea_board.executor.cancel_execution", side_effect=fake_cancel) as mock_cancel, \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.aim_execution_timeout = 100
            result = watch_execution("idea-001")

        assert result.success is False
        assert "timed out" in result.summary.lower()
        mock_cancel.assert_called_once_with("idea-001")


# ---------------------------------------------------------------------------
# execute_assigned_idea tests
# ---------------------------------------------------------------------------

class TestExecuteAssignedIdea:
    @patch("aim.worker.watch_execution")
    @patch("idea_board.executor.execute_idea")
    @patch("aim.worker._check_git_clean", return_value=True)
    @patch("idea_board.executor.is_any_executing", return_value=False)
    @patch("aim.state.update_worker_status")
    @patch("aim.worker.time.sleep")
    def test_full_success(self, mock_sleep, mock_status, mock_any, mock_clean,
                          mock_execute, mock_watch):
        fake_state = FakeExecutionState()
        mock_execute.return_value = fake_state
        mock_watch.return_value = WatchResult(success=True, summary="Done")

        result = execute_assigned_idea("idea-001")

        assert result.success is True
        mock_execute.assert_called_once_with("idea-001")
        mock_watch.assert_called_once_with("idea-001")

    @patch("aim.worker._check_git_clean", return_value=True)
    @patch("idea_board.executor.is_any_executing", return_value=True)
    @patch("aim.state.update_worker_status")
    def test_rejects_if_already_executing(self, mock_status, mock_any, mock_clean):
        result = execute_assigned_idea("idea-001")
        assert result.success is False
        assert "already in progress" in result.summary.lower()

    @patch("aim.worker._check_git_clean", return_value=False)
    @patch("idea_board.executor.is_any_executing", return_value=False)
    @patch("aim.state.update_worker_status")
    def test_rejects_dirty_git(self, mock_status, mock_any, mock_clean):
        result = execute_assigned_idea("idea-001")
        assert result.success is False
        assert "not clean" in result.summary.lower() or "not on main" in result.summary.lower()

    @patch("idea_board.executor.execute_idea", return_value=None)
    @patch("aim.worker._check_git_clean", return_value=True)
    @patch("idea_board.executor.is_any_executing", return_value=False)
    @patch("aim.state.update_worker_status")
    @patch("aim.worker.time.sleep")
    def test_handles_execute_returning_none(self, mock_sleep, mock_status,
                                            mock_any, mock_clean, mock_execute):
        result = execute_assigned_idea("idea-001")
        assert result.success is False
        assert "None" in result.summary
