"""Tests for aim.worker — AI Worker state machine and execution watcher."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aim.worker import (
    RATE_LIMIT_MAX_WAIT_MINUTES,
    WatchResult,
    _check_git_clean,
    _post_rate_limit_comment,
    _reconcile_with_board_state,
    _sleep_with_heartbeat,
    execute_assigned_idea,
    watch_execution,
)


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
                 alive=True, elapsed=0.0, rate_limited=False):
        self.idea_id = idea_id
        self.pid = pid
        self.log_lines = log_lines if log_lines is not None else []
        self._alive = alive
        self._elapsed = elapsed
        self.rate_limited = rate_limited

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
                          mock_execute, mock_watch, monkeypatch):
        # The worker routes to execute_idea_ab when AIW_AB_TEST is on; this
        # test pins the non-A/B path so the mock on execute_idea actually fires.
        from agent import config as _config
        monkeypatch.setattr(_config.settings, "aiw_ab_test_enabled", False, raising=False)

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
                                            mock_any, mock_clean, mock_execute,
                                            monkeypatch):
        # Pin the non-A/B path so execute_idea (not execute_idea_ab) is called.
        from agent import config as _config
        monkeypatch.setattr(_config.settings, "aiw_ab_test_enabled", False, raising=False)

        result = execute_assigned_idea("idea-001")
        assert result.success is False
        assert "None" in result.summary


# ---------------------------------------------------------------------------
# Rate-limit handling (TK-410)
# ---------------------------------------------------------------------------

class TestWatchResultRateLimited:
    """WatchResult grows a rate_limited flag."""

    def test_default_false(self):
        r = WatchResult(success=True, summary="ok")
        assert r.rate_limited is False

    def test_can_set_true(self):
        r = WatchResult(success=False, summary="Rate limited", rate_limited=True)
        assert r.rate_limited is True


class TestWatchExecutionRateLimited:
    """watch_execution surfaces the executor's rate_limited flag."""

    @patch("aim.worker.time.sleep")
    @patch("aim.worker.WATCH_INTERVAL", 0)
    def test_returns_rate_limited(self, mock_sleep):
        """When exec_state.rate_limited is True, result.rate_limited is set
        and mark_done/mark_failed must NOT have run (neither is called here;
        we just verify the watcher's return payload)."""
        fake_state = FakeExecutionState(
            idea_id="idea-001",
            log_lines=["hit rate_limit_error"],
            alive=False,
            rate_limited=True,
        )
        with patch("idea_board.executor.get_execution", return_value=fake_state), \
             patch("aim.state.update_worker_heartbeat"), \
             patch("aim.state.update_worker_status"):
            result = watch_execution("idea-001")

        assert result.success is False
        assert result.rate_limited is True
        assert "rate" in result.summary.lower()


class TestReconcileWithBoardState:
    """_reconcile_with_board_state protects shipped work from false-fail reports."""

    def test_done_on_board_promotes_to_success(self):
        provider = MagicMock()
        provider.get.return_value = FakeIdea(state="done")
        failure = WatchResult(
            success=False, summary="Stalled — no output for 10 min",
        )
        with patch("board.get_provider", return_value=provider):
            result = _reconcile_with_board_state("TK-567", failure)
        assert result.success is True
        assert "completed" in result.summary.lower()

    def test_still_executing_preserves_failure(self):
        provider = MagicMock()
        provider.get.return_value = FakeIdea(state="executing")
        failure = WatchResult(success=False, summary="Stalled")
        with patch("board.get_provider", return_value=provider):
            result = _reconcile_with_board_state("TK-567", failure)
        assert result is failure

    def test_failed_on_board_preserves_failure(self):
        provider = MagicMock()
        provider.get.return_value = FakeIdea(state="failed")
        failure = WatchResult(success=False, summary="Timed out")
        with patch("board.get_provider", return_value=provider):
            result = _reconcile_with_board_state("TK-567", failure)
        assert result is failure

    def test_missing_idea_preserves_failure(self):
        provider = MagicMock()
        provider.get.return_value = None
        failure = WatchResult(success=False, summary="Stalled")
        with patch("board.get_provider", return_value=provider):
            result = _reconcile_with_board_state("TK-567", failure)
        assert result is failure

    def test_provider_error_preserves_failure(self):
        """If Jira is down, don't pretend work shipped — keep the failure."""
        provider = MagicMock()
        provider.get.side_effect = RuntimeError("jira down")
        failure = WatchResult(success=False, summary="Stalled")
        with patch("board.get_provider", return_value=provider):
            result = _reconcile_with_board_state("TK-567", failure)
        assert result is failure


class TestSleepWithHeartbeat:
    """_sleep_with_heartbeat ticks the heartbeat periodically."""

    @patch("aim.worker.time.sleep")
    def test_calls_heartbeat_each_chunk(self, mock_sleep):
        with patch("aim.state.update_worker_heartbeat") as mock_hb:
            _sleep_with_heartbeat(duration_seconds=90, interval=30)
        # 90s / 30s = 3 heartbeats
        assert mock_hb.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("aim.worker.time.sleep")
    def test_zero_duration_noop(self, mock_sleep):
        with patch("aim.state.update_worker_heartbeat") as mock_hb:
            _sleep_with_heartbeat(duration_seconds=0)
        mock_hb.assert_not_called()
        mock_sleep.assert_not_called()

    @patch("aim.worker.time.sleep")
    def test_heartbeat_error_swallowed(self, mock_sleep):
        """A provider hiccup during heartbeat must not break the sleep."""
        with patch(
            "aim.state.update_worker_heartbeat",
            side_effect=OSError("disk full"),
        ):
            # Should not raise
            _sleep_with_heartbeat(duration_seconds=30, interval=30)


class TestPostRateLimitComment:
    """_post_rate_limit_comment writes via the board provider."""

    def test_calls_add_comment(self):
        provider = MagicMock()
        with patch("board.get_provider", return_value=provider):
            _post_rate_limit_comment("TK-410", wait_minutes=15)
        provider.add_comment.assert_called_once()
        kwargs = provider.add_comment.call_args.kwargs
        args = provider.add_comment.call_args.args
        # text is the third positional or a kwarg
        text = kwargs.get("text") or (args[2] if len(args) >= 3 else "")
        assert "Rate Limited" in text
        assert "15 minutes" in text

    def test_noop_when_provider_lacks_add_comment(self):
        """LocalProvider without add_comment should not raise."""
        provider = object()  # no add_comment attribute
        with patch("board.get_provider", return_value=provider):
            _post_rate_limit_comment("TK-410", wait_minutes=15)  # must not raise

    def test_swallows_provider_errors(self):
        provider = MagicMock()
        provider.add_comment.side_effect = RuntimeError("jira down")
        with patch("board.get_provider", return_value=provider):
            # Must not raise — retries continue even if Jira is flaky.
            _post_rate_limit_comment("TK-410", wait_minutes=15)


class TestExecuteAssignedIdeaRateLimitRetry:
    """Full retry loop: rate-limit → sleep → retry → success/exhaustion."""

    def test_retry_then_success(self):
        """Rate-limited on first attempt, succeeds on retry.

        Asserts the primary TK-410 requirement: mark_done is NOT called
        directly by the worker (the executor owns that) and mark_failed
        is NOT called on the rate-limited attempt."""
        mock_watch = MagicMock(side_effect=[
            WatchResult(success=False, summary="Rate limited", rate_limited=True),
            WatchResult(success=True, summary="Completed"),
        ])
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_mark_failed = MagicMock()
        mock_mark_done = MagicMock()

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker._sleep_with_heartbeat"), \
             patch("aim.worker._post_rate_limit_comment"), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed", mock_mark_failed), \
             patch("idea_board.executor.mark_done", mock_mark_done), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 3
            result = execute_assigned_idea("TK-410")

        assert result.success is True
        assert result.rate_limited is False
        assert mock_execute.call_count == 2
        assert mock_watch.call_count == 2
        mock_mark_failed.assert_not_called()
        mock_mark_done.assert_not_called()

    def test_retries_exhaust_marks_failed(self):
        """Every attempt hits rate-limit; final attempt triggers mark_failed
        with a message that mentions 'rate limit'."""
        mock_watch = MagicMock(return_value=WatchResult(
            success=False, summary="Rate limited", rate_limited=True,
        ))
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_mark_failed = MagicMock()

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker._sleep_with_heartbeat"), \
             patch("aim.worker._post_rate_limit_comment"), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed", mock_mark_failed), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 2
            result = execute_assigned_idea("TK-410")

        assert result.success is False
        assert "rate limit" in result.summary.lower()
        # 3 attempts total = initial + 2 retries
        assert mock_execute.call_count == 3
        mock_mark_failed.assert_called_once()
        failed_msg = mock_mark_failed.call_args.args[1]
        assert "rate limit" in failed_msg.lower()

    def test_first_attempt_succeeds_skips_sleep(self):
        """Happy path: no rate-limit → no retry, no sleep, no comment."""
        mock_watch = MagicMock(return_value=WatchResult(success=True, summary="ok"))
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_hb = MagicMock()
        mock_comment = MagicMock()

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker._sleep_with_heartbeat", mock_hb), \
             patch("aim.worker._post_rate_limit_comment", mock_comment), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed"), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 3
            result = execute_assigned_idea("TK-410")

        assert result.success is True
        assert mock_execute.call_count == 1
        mock_hb.assert_not_called()
        mock_comment.assert_not_called()

    def test_wait_doubles_and_caps_at_max(self):
        """Wait schedule follows 15 → 30 → 60 → 60 (capped at max)."""
        waits: list[int] = []

        def capture_sleep(duration_seconds, interval=30):
            waits.append(duration_seconds // 60)

        mock_watch = MagicMock(side_effect=[
            WatchResult(success=False, summary="r", rate_limited=True),
            WatchResult(success=False, summary="r", rate_limited=True),
            WatchResult(success=False, summary="r", rate_limited=True),
            WatchResult(success=True, summary="ok"),
        ])
        mock_execute = MagicMock(return_value=FakeExecutionState())

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker._sleep_with_heartbeat", side_effect=capture_sleep), \
             patch("aim.worker._post_rate_limit_comment"), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed"), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 5
            result = execute_assigned_idea("TK-410")

        assert result.success is True
        # Three sleeps before success on the 4th attempt
        assert waits == [15, 30, RATE_LIMIT_MAX_WAIT_MINUTES]

    def test_stalled_but_shipped_reported_as_success(self):
        """TK-567: watcher says Stalled but executor already marked idea done.

        Simulates the race on FA-94 (2026-04-17): the claude -p subprocess
        goes quiet past STALE_THRESHOLD while the executor thread is
        committing+pushing+publishing, watcher cancels and returns
        success=False, but the executor completed mark_done first.
        The final WatchResult from execute_assigned_idea must be success
        so the caller doesn't clobber a shipped story with mark_failed.
        """
        stalled = WatchResult(
            success=False,
            summary="Stalled — no output for 10 min",
            rate_limited=False,
        )
        mock_watch = MagicMock(return_value=stalled)
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_mark_failed = MagicMock()
        mock_mark_done = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get.return_value = FakeIdea(state="done")

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed", mock_mark_failed), \
             patch("idea_board.executor.mark_done", mock_mark_done), \
             patch("board.get_provider", return_value=mock_provider), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 3
            result = execute_assigned_idea("TK-567")

        assert result.success is True
        assert result.rate_limited is False
        mock_mark_failed.assert_not_called()
        mock_provider.get.assert_called_with("TK-567")

    def test_genuine_stall_with_no_commits_still_fails(self):
        """A real stall where executor never reached mark_done must stay failed.

        Complements test_stalled_but_shipped_reported_as_success — ensures
        the reconciliation only overrides the failure when the board
        actually shows done. A story still in 'executing' or 'failed'
        must surface as a failure so AIM's retry/split logic runs.
        """
        stalled = WatchResult(
            success=False,
            summary="Stalled — no output for 10 min",
            rate_limited=False,
        )
        mock_watch = MagicMock(return_value=stalled)
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_provider = MagicMock()
        mock_provider.get.return_value = FakeIdea(state="executing")

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status"), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed"), \
             patch("board.get_provider", return_value=mock_provider), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 3
            result = execute_assigned_idea("TK-567")

        assert result.success is False
        assert "stall" in result.summary.lower()

    def test_sets_rate_limited_worker_status_during_wait(self):
        """Worker status must be 'rate_limited' while we're sleeping so AIM
        doesn't restart us and so the dashboard can surface the state."""
        mock_watch = MagicMock(side_effect=[
            WatchResult(success=False, summary="r", rate_limited=True),
            WatchResult(success=True, summary="ok"),
        ])
        mock_execute = MagicMock(return_value=FakeExecutionState())
        mock_status = MagicMock()

        with patch("aim.worker._check_git_clean", return_value=True), \
             patch("aim.state.update_worker_status", mock_status), \
             patch("aim.worker.time.sleep"), \
             patch("idea_board.executor.is_any_executing", return_value=False), \
             patch("aim.worker._sleep_with_heartbeat"), \
             patch("aim.worker._post_rate_limit_comment"), \
             patch("aim.worker.watch_execution", mock_watch), \
             patch("idea_board.executor.execute_idea", mock_execute), \
             patch("idea_board.executor.mark_failed"), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.rate_limit_wait_minutes = 15
            mock_settings.aiw_ab_test_enabled = False
            mock_settings.rate_limit_max_retries = 3
            execute_assigned_idea("TK-410")

        statuses = [c.args[0] for c in mock_status.call_args_list if c.args]
        assert "rate_limited" in statuses
