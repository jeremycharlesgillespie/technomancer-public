"""Tests for aim.manager._cleanup_stale_worktrees — stale-worktree
housekeeping loop."""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from aim.state import AIMState, WorkerState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    """A minimal AIMState whose worker has no live PID, so worktrees
    aren't preserved by the worker-claim heuristic."""
    return AIMState(
        manager_pid=12345,
        worker=WorkerState(pid=None, status="idle"),
    )


@pytest.fixture
def mock_worktree_module(monkeypatch):
    """Inject a stub agent.worktree_manager into sys.modules.

    The real module doesn't exist yet — TK-645 ships the cleanup loop
    ahead of the worktree-manager rollout. Tests mount a stub so the
    ``from agent import worktree_manager`` inside the cleanup function
    succeeds and we can drive list_worktrees() / remove_worktree().
    """
    module = types.ModuleType("agent.worktree_manager")
    module.list_worktrees = MagicMock(return_value=[])
    module.remove_worktree = MagicMock()
    monkeypatch.setitem(sys.modules, "agent.worktree_manager", module)
    return module


@pytest.fixture(autouse=True)
def _reset_cleanup_clock(monkeypatch):
    """Reset the module-level monotonic gate so each test starts at zero."""
    import aim.manager as manager
    monkeypatch.setattr(manager, "_LAST_WORKTREE_CLEANUP_MONOTONIC", 0.0)


@pytest.fixture(autouse=True)
def _quiet_event_log(monkeypatch):
    """Silence event_log writes — the housekeeping loop emits to it on
    every removal and we don't want test pollution."""
    from aim import event_log
    monkeypatch.setattr(event_log, "append_event", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestCleanupStaleWorktrees:
    def test_stale_worktree_removed(self, state, mock_worktree_module, tmp_path):
        """A worktree older than the threshold with no live PID is removed."""
        from aim.manager import _cleanup_stale_worktrees

        wt_path = tmp_path / "slot-1"
        wt_path.mkdir()
        # 5 hours old vs 2-hour default threshold
        old_mtime = time.time() - 5 * 3600
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_path), "slot_id": 1, "mtime": old_mtime, "pid": None},
        ]

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 1
        mock_worktree_module.remove_worktree.assert_called_once_with(1)

    def test_fresh_worktree_preserved(self, state, mock_worktree_module, tmp_path):
        """A worktree younger than the threshold is preserved."""
        from aim.manager import _cleanup_stale_worktrees

        wt_path = tmp_path / "slot-2"
        wt_path.mkdir()
        # 30 minutes old vs 2-hour default threshold
        fresh_mtime = time.time() - 30 * 60
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_path), "slot_id": 2, "mtime": fresh_mtime, "pid": None},
        ]

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 0
        mock_worktree_module.remove_worktree.assert_not_called()

    def test_live_pid_preserves_old_worktree(
        self, state, mock_worktree_module, tmp_path
    ):
        """A worktree with a live PID claim is preserved even when stale."""
        from aim.manager import _cleanup_stale_worktrees

        wt_path = tmp_path / "slot-3"
        wt_path.mkdir()
        old_mtime = time.time() - 10 * 3600
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_path), "slot_id": 3, "mtime": old_mtime, "pid": 4242},
        ]

        from agent.config import settings
        # is_process_alive is imported inside the function from aim.state —
        # patch the source binding so the in-function import sees True.
        with patch("aim.state.is_process_alive", return_value=True), \
             patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 0
        mock_worktree_module.remove_worktree.assert_not_called()

    def test_dry_run_logs_but_does_not_remove(
        self, state, mock_worktree_module, tmp_path, caplog
    ):
        """Dry-run mode counts removals and logs them without calling
        remove_worktree."""
        import logging
        from aim.manager import _cleanup_stale_worktrees

        wt_path = tmp_path / "slot-4"
        wt_path.mkdir()
        old_mtime = time.time() - 5 * 3600
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_path), "slot_id": 4, "mtime": old_mtime, "pid": None},
        ]

        from agent.config import settings
        with caplog.at_level(logging.INFO, logger="aim.manager"):
            with patch.object(settings, "stale_worktree_hours", 2.0):
                removed = _cleanup_stale_worktrees(state, dry_run=True)

        assert removed == 1
        mock_worktree_module.remove_worktree.assert_not_called()
        # Some line should mention dry-run for slot-4.
        msgs = " ".join(r.message for r in caplog.records)
        assert "dry-run" in msgs.lower()
        assert "slot-4" in msgs or "slot=4" in msgs

    def test_mixed_fresh_and_stale(
        self, state, mock_worktree_module, tmp_path
    ):
        """Only the stale worktree is removed; the fresh one survives."""
        from aim.manager import _cleanup_stale_worktrees

        fresh = tmp_path / "slot-fresh"
        fresh.mkdir()
        stale = tmp_path / "slot-stale"
        stale.mkdir()

        now = time.time()
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(fresh), "slot_id": "fresh", "mtime": now - 60, "pid": None},
            {"path": str(stale), "slot_id": "stale", "mtime": now - 6 * 3600, "pid": None},
        ]

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 1
        mock_worktree_module.remove_worktree.assert_called_once_with("stale")

    def test_missing_worktree_manager_is_noop(self, state, monkeypatch):
        """If agent.worktree_manager isn't installed, the function returns 0
        without raising — lets us land the loop ahead of the manager rollout."""
        import aim.manager as manager
        from agent.config import settings

        # Make sure the import inside the function fails cleanly. Setting
        # the cached entry to None makes Python raise ImportError on the
        # next ``from agent import worktree_manager``.
        monkeypatch.setitem(sys.modules, "agent.worktree_manager", None)

        with patch.object(settings, "stale_worktree_hours", 2.0):
            removed = manager._cleanup_stale_worktrees(state)

        assert removed == 0

    def test_age_falls_back_to_filesystem_mtime(
        self, state, mock_worktree_module, tmp_path
    ):
        """When the entry has no ``mtime``, the function reads it from
        the worktree path on disk."""
        from aim.manager import _cleanup_stale_worktrees
        import os

        wt_path = tmp_path / "slot-fs"
        wt_path.mkdir()
        # Backdate the directory so its filesystem mtime is well past the
        # threshold.
        backdated = time.time() - 6 * 3600
        os.utime(wt_path, (backdated, backdated))

        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_path), "slot_id": "fs"},  # no mtime, no pid
        ]

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 1
        mock_worktree_module.remove_worktree.assert_called_once_with("fs")

    def test_remove_worktree_failure_is_swallowed(
        self, state, mock_worktree_module, tmp_path
    ):
        """A failure in remove_worktree is logged but doesn't crash the
        loop or block subsequent entries."""
        from aim.manager import _cleanup_stale_worktrees

        wt_a = tmp_path / "slot-a"
        wt_a.mkdir()
        wt_b = tmp_path / "slot-b"
        wt_b.mkdir()

        old = time.time() - 6 * 3600
        mock_worktree_module.list_worktrees.return_value = [
            {"path": str(wt_a), "slot_id": "a", "mtime": old, "pid": None},
            {"path": str(wt_b), "slot_id": "b", "mtime": old, "pid": None},
        ]
        mock_worktree_module.remove_worktree.side_effect = [
            RuntimeError("git worktree remove failed"),
            None,
        ]

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0), \
             patch.object(settings, "aim_worktree_cleanup_dry_run", False):
            removed = _cleanup_stale_worktrees(state)

        # Only the second remove succeeded.
        assert removed == 1
        assert mock_worktree_module.remove_worktree.call_count == 2

    def test_list_worktrees_failure_returns_zero(
        self, state, mock_worktree_module
    ):
        """A failure in list_worktrees is swallowed — the cleanup loop
        must never bring AIM down."""
        from aim.manager import _cleanup_stale_worktrees

        mock_worktree_module.list_worktrees.side_effect = RuntimeError("boom")

        from agent.config import settings
        with patch.object(settings, "stale_worktree_hours", 2.0):
            removed = _cleanup_stale_worktrees(state)

        assert removed == 0
        mock_worktree_module.remove_worktree.assert_not_called()


# ---------------------------------------------------------------------------
# Hourly throttle wrapper
# ---------------------------------------------------------------------------

class TestMaybeCleanupStaleWorktrees:
    def test_first_call_runs(self, state, monkeypatch):
        """The first call after AIM start always runs the cleanup."""
        from aim import manager

        monkeypatch.setattr(manager, "_LAST_WORKTREE_CLEANUP_MONOTONIC", 0.0)
        with patch.object(manager, "_cleanup_stale_worktrees") as mock_cleanup:
            manager._maybe_cleanup_stale_worktrees(state)
        mock_cleanup.assert_called_once_with(state)

    def test_second_call_within_interval_skips(self, state, monkeypatch):
        """Back-to-back calls inside the interval window only run once."""
        from aim import manager

        monkeypatch.setattr(manager, "_LAST_WORKTREE_CLEANUP_MONOTONIC", 0.0)
        with patch.object(manager, "_cleanup_stale_worktrees") as mock_cleanup:
            manager._maybe_cleanup_stale_worktrees(state)
            manager._maybe_cleanup_stale_worktrees(state)
        assert mock_cleanup.call_count == 1

    def test_call_after_interval_runs_again(self, state, monkeypatch):
        """Once the interval elapses, the next call fires again."""
        from aim import manager

        monkeypatch.setattr(manager, "_LAST_WORKTREE_CLEANUP_MONOTONIC", 0.0)
        with patch.object(manager, "_cleanup_stale_worktrees") as mock_cleanup:
            manager._maybe_cleanup_stale_worktrees(state)
            # Pretend an hour and change has passed.
            monkeypatch.setattr(
                manager,
                "_LAST_WORKTREE_CLEANUP_MONOTONIC",
                time.monotonic() - manager.WORKTREE_CLEANUP_INTERVAL_SECONDS - 1,
            )
            manager._maybe_cleanup_stale_worktrees(state)
        assert mock_cleanup.call_count == 2

    def test_cleanup_exception_is_swallowed(self, state, monkeypatch):
        """An unhandled exception inside _cleanup_stale_worktrees must not
        propagate out of the throttle wrapper — AIM's main loop relies on it."""
        from aim import manager

        monkeypatch.setattr(manager, "_LAST_WORKTREE_CLEANUP_MONOTONIC", 0.0)
        with patch.object(
            manager, "_cleanup_stale_worktrees", side_effect=RuntimeError("nope")
        ):
            # Should not raise.
            manager._maybe_cleanup_stale_worktrees(state)
