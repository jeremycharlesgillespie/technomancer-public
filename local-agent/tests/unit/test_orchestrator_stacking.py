"""Tests for the orchestrator-stacking fix.

Background
----------

Before this fix, ``aim.worker._cleanup_stale_executions`` only killed
entries whose ``ExecutionState.pid`` was set.  The A/B orchestrator
runs as a daemon thread (``pid is None``), so the kill path was
skipped.  ``_active.clear()`` then removed the bookkeeping entry, but
the daemon thread kept running.  The worker promptly assigned a new
story and started a second orchestrator on top of the first — both
fought over the shared worktree and the shared ``_active`` slot.

The fix:

  * ``executor.is_any_executing`` now also covers
    ``_ab_orchestrator_active`` so a live A/B thread is actually
    visible.
  * ``executor.force_cancel_all_active`` flags every state
    ``cancelled=True`` AND joins the threads with a timeout.  Only
    entries whose threads actually died are removed from the dicts.
  * ``aim.worker._cleanup_stale_executions`` returns False when any
    thread refused to die — the worker then refuses to assign new
    work instead of stacking.

These tests exercise that contract end-to-end with synthetic
``ExecutionState`` objects so the suite stays fast and doesn't need a
live Ollama or git tree.
"""

from __future__ import annotations

import threading
import time

import pytest

from idea_board import executor as executor_mod
from idea_board.executor import (
    ExecutionState,
    force_cancel_all_active,
    is_any_executing,
)


# ---------------------------------------------------------------------------
# Helpers — build fake ExecutionStates whose threads observe the cancel flag.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_active_dicts():
    """Make sure no test leaks ExecutionState entries into module state."""
    executor_mod._active.clear()
    executor_mod._ab_orchestrator_active.clear()
    yield
    executor_mod._active.clear()
    executor_mod._ab_orchestrator_active.clear()


def _start_polite_thread(state: ExecutionState, *, max_seconds: float = 5.0) -> None:
    """Start a thread that exits as soon as ``state.cancelled`` flips."""

    def _worker() -> None:
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            if state.cancelled:
                return
            time.sleep(0.05)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    state.thread = t


def _start_rude_thread(state: ExecutionState, *, lifetime: float = 60.0) -> None:
    """Start a thread that ignores the cancel flag for ``lifetime`` seconds.

    Used to simulate a wedged Ollama HTTP call — the thread is alive
    long after the cancel was raised.
    """

    def _worker() -> None:
        end = time.time() + lifetime
        while time.time() < end:
            time.sleep(0.05)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    state.thread = t


# ---------------------------------------------------------------------------
# is_any_executing — must consider both dicts
# ---------------------------------------------------------------------------


class TestIsAnyExecuting:
    def test_empty_dicts_returns_false(self):
        assert is_any_executing() is False

    def test_live_thread_in_active_returns_true(self):
        state = ExecutionState(idea_id="TK-1")
        _start_polite_thread(state, max_seconds=5.0)
        executor_mod._active["TK-1"] = state
        try:
            assert is_any_executing() is True
        finally:
            state.cancelled = True
            state.thread.join(timeout=2.0) if state.thread else None

    def test_live_thread_in_ab_orchestrator_returns_true(self):
        """Regression: pre-fix ``is_any_executing`` only checked ``_active``,
        so an A/B orchestrator thread was invisible and the worker stacked."""
        state = ExecutionState(idea_id="TK-2")
        _start_polite_thread(state, max_seconds=5.0)
        executor_mod._ab_orchestrator_active["TK-2"] = state
        try:
            assert is_any_executing() is True
        finally:
            state.cancelled = True
            state.thread.join(timeout=2.0) if state.thread else None

    def test_dead_thread_in_dict_returns_false(self):
        state = ExecutionState(idea_id="TK-3")
        _start_polite_thread(state, max_seconds=0.05)
        if state.thread:
            state.thread.join(timeout=1.0)
        executor_mod._active["TK-3"] = state
        assert is_any_executing() is False


# ---------------------------------------------------------------------------
# force_cancel_all_active — set cancel flag, join threads, only clear when dead
# ---------------------------------------------------------------------------


class TestForceCancelAllActive:
    def test_no_active_executions_returns_true(self):
        assert force_cancel_all_active(timeout=1.0) is True

    def test_polite_thread_in_active_dies_and_dict_clears(self):
        state = ExecutionState(idea_id="TK-10")
        _start_polite_thread(state, max_seconds=10.0)
        executor_mod._active["TK-10"] = state

        ok = force_cancel_all_active(timeout=2.0)

        assert ok is True
        assert state.cancelled is True
        assert state.thread is not None
        assert state.thread.is_alive() is False
        assert "TK-10" not in executor_mod._active

    def test_polite_thread_in_ab_orchestrator_dies_and_dict_clears(self):
        state = ExecutionState(idea_id="TK-11")
        _start_polite_thread(state, max_seconds=10.0)
        executor_mod._ab_orchestrator_active["TK-11"] = state

        ok = force_cancel_all_active(timeout=2.0)

        assert ok is True
        assert state.cancelled is True
        assert "TK-11" not in executor_mod._ab_orchestrator_active

    def test_rude_thread_returns_false_and_stays_in_dict(self):
        """A thread that ignores the cancel flag must NOT be erased from
        the dict — leaving it visible is what makes the worker refuse to
        start a stacked second orchestrator."""
        state = ExecutionState(idea_id="TK-12")
        _start_rude_thread(state, lifetime=10.0)
        executor_mod._ab_orchestrator_active["TK-12"] = state

        ok = force_cancel_all_active(timeout=0.5)

        assert ok is False
        assert state.cancelled is True
        # Critical: the entry MUST remain so is_any_executing() keeps
        # reporting True and the worker skips its next assignment.
        assert "TK-12" in executor_mod._ab_orchestrator_active
        assert is_any_executing() is True

    def test_mixed_polite_and_rude_clears_only_polite(self):
        polite = ExecutionState(idea_id="TK-20")
        _start_polite_thread(polite, max_seconds=10.0)
        executor_mod._active["TK-20"] = polite

        rude = ExecutionState(idea_id="TK-21")
        _start_rude_thread(rude, lifetime=10.0)
        executor_mod._ab_orchestrator_active["TK-21"] = rude

        ok = force_cancel_all_active(timeout=1.5)

        assert ok is False
        assert "TK-20" not in executor_mod._active
        assert "TK-21" in executor_mod._ab_orchestrator_active

    def test_dead_thread_is_cleared_even_with_no_thread_attr(self):
        """A state whose thread already exited (or was never set) is
        treated as dead — its bookkeeping should be removed."""
        state = ExecutionState(idea_id="TK-30")
        # No thread attached.
        executor_mod._active["TK-30"] = state

        ok = force_cancel_all_active(timeout=0.5)

        assert ok is True
        assert "TK-30" not in executor_mod._active


# ---------------------------------------------------------------------------
# aim.worker._cleanup_stale_executions — wraps the helper, honors return val
# ---------------------------------------------------------------------------


class TestCleanupStaleExecutions:
    def test_returns_true_when_no_active(self):
        from aim.worker import _cleanup_stale_executions

        assert _cleanup_stale_executions(timeout=0.5) is True

    def test_returns_false_when_thread_wont_die(self):
        from aim.worker import _cleanup_stale_executions

        rude = ExecutionState(idea_id="TK-40")
        _start_rude_thread(rude, lifetime=10.0)
        executor_mod._ab_orchestrator_active["TK-40"] = rude

        assert _cleanup_stale_executions(timeout=0.3) is False

    def test_returns_true_after_polite_thread_exits(self):
        from aim.worker import _cleanup_stale_executions

        polite = ExecutionState(idea_id="TK-41")
        _start_polite_thread(polite, max_seconds=10.0)
        executor_mod._active["TK-41"] = polite

        assert _cleanup_stale_executions(timeout=2.0) is True


# ---------------------------------------------------------------------------
# execute_assigned_idea integration — refuses to stack
# ---------------------------------------------------------------------------


class TestExecuteAssignedIdeaRefusesStacking:
    """The only behavioural symptom users actually saw is the worker
    cheerfully starting a second story while the prior orchestrator
    thread was still alive.  This test pins that behaviour shut."""

    def test_refuses_to_stack_when_prior_thread_alive(self, monkeypatch):
        from aim import worker as worker_mod

        # Pre-populate the orchestrator dict with a wedged thread.  The
        # cleanup helper will fail to kill it.
        rude = ExecutionState(idea_id="TK-50")
        _start_rude_thread(rude, lifetime=10.0)
        executor_mod._ab_orchestrator_active["TK-50"] = rude

        # Make the cleanup timeout tiny so the test runs fast.
        monkeypatch.setattr(
            worker_mod,
            "_cleanup_stale_executions",
            lambda timeout=0.2: False,
        )

        # If we ever reach execute_idea / execute_idea_ab the test
        # should fail loudly — those calls would stack on top of the
        # rude thread.
        def _boom(*_a, **_kw):
            raise AssertionError(
                "worker tried to start a new execution while a prior "
                "thread was still alive — orchestrator-stacking guard failed"
            )

        monkeypatch.setattr("idea_board.executor.execute_idea", _boom)
        monkeypatch.setattr("idea_board.executor.is_any_executing", lambda: True)

        # Stub out heartbeat / status writes so we don't need aim_state
        # to exist on disk.
        monkeypatch.setattr(
            "aim.state.update_worker_status",
            lambda *_a, **_kw: None,
        )
        monkeypatch.setattr(
            "aim.state.update_worker_heartbeat",
            lambda *_a, **_kw: None,
        )

        result = worker_mod.execute_assigned_idea("TK-99")

        assert result.success is False
        assert "refusing to stack" in result.summary
