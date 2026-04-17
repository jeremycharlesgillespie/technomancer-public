"""Tests for TK-468 — kill endpoint and ``executor_runs_db.kill_run``.

Covers the three acceptance scenarios spelled out in the story:
  (a) kill on a live process transitions the row to ``killed``
  (b) 409 when the run is already in a terminal state
  (c) SIGKILL fires after the SIGTERM grace period when the process ignores
      SIGTERM (simulated by patching ``_is_process_alive`` to always report
      the process alive; that avoids spawning a real POSIX-only signal-
      ignoring child on Windows)

Plus the auxiliary checks that protect the endpoint contract: 404 on unknown
run, 202 body shape, reason persistence, schema migration.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from agent import executor_runs_db
from idea_board.web import app


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point executor_runs_db at a temporary SQLite DB for each test."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()
    yield
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Schema — TK-468 added killed_at and kill_reason columns
# ---------------------------------------------------------------------------


class TestKillColumns:
    def test_killed_at_and_kill_reason_columns_present(self):
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
        }
        assert "killed_at" in cols
        assert "kill_reason" in cols

    def test_migration_idempotent(self):
        """Calling init_db twice must not fail when the new columns already
        exist — the ALTER TABLE guard is what keeps migrations re-runnable."""
        executor_runs_db.init_db()
        executor_runs_db.init_db()
        # Still usable afterwards
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
        }
        assert "killed_at" in cols
        assert "kill_reason" in cols


# ---------------------------------------------------------------------------
# kill_run() — behavioural contract
# ---------------------------------------------------------------------------


class TestKillRunTransitionsToKilled:
    """(a) Killing a live run transitions its row to ``killed`` with
    ``killed_at`` set and ``kill_reason`` preserved."""

    def test_live_run_transitions_to_killed(self, monkeypatch):
        # Simulate a live process that exits on SIGTERM: alive before signal,
        # dead after one poll.
        alive_state = {"alive": True}

        def fake_alive(pid):
            return alive_state["alive"]

        def fake_sigterm(pid):
            alive_state["alive"] = False

        sigkill = MagicMock()
        monkeypatch.setattr(executor_runs_db, "_is_process_alive", fake_alive)
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", fake_sigterm)
        monkeypatch.setattr(executor_runs_db, "_send_sigkill", sigkill)

        run_id = executor_runs_db.record_run(
            jira_key="TK-468",
            status="running",
            pid=99999,  # fake but non-zero so kill_run actually signals
        )

        result = executor_runs_db.kill_run(
            run_id, reason="runaway", sigterm_timeout=0.2, poll_interval=0.02,
        )

        assert result["status"] == "killed"
        assert result["pid"] == 99999
        assert result["escalated"] is False
        sigkill.assert_not_called()

        row = executor_runs_db.get_run(run_id)
        assert row is not None
        assert row["status"] == "killed"
        assert row["killed_at"] is not None
        assert row["kill_reason"] == "runaway"
        assert row["ended_at"] is not None

    def test_kill_without_pid_still_marks_row(self, monkeypatch):
        """A run with no recorded pid still transitions — we just skip the
        signal dance since there's nothing to signal."""
        sigterm = MagicMock()
        sigkill = MagicMock()
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", sigterm)
        monkeypatch.setattr(executor_runs_db, "_send_sigkill", sigkill)

        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running",
        )
        # No pid column for this row
        result = executor_runs_db.kill_run(run_id, reason="no-pid")

        assert result["status"] == "killed"
        assert result["escalated"] is False
        sigterm.assert_not_called()
        sigkill.assert_not_called()
        row = executor_runs_db.get_run(run_id)
        assert row["status"] == "killed"
        assert row["kill_reason"] == "no-pid"

    def test_reason_defaults_to_null(self, monkeypatch):
        monkeypatch.setattr(
            executor_runs_db, "_is_process_alive", lambda pid: False,
        )
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", lambda pid: None)
        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running", pid=123,
        )
        executor_runs_db.kill_run(run_id, sigterm_timeout=0.05)
        row = executor_runs_db.get_run(run_id)
        assert row["kill_reason"] is None


class TestKillRunAlreadyTerminal:
    """(b) Refuses to re-kill a run that's already terminal — callers see
    :class:`RunAlreadyTerminalError` which the HTTP layer maps to 409."""

    @pytest.mark.parametrize(
        "terminal_status",
        ["success", "failed", "error", "timeout", "killed", "done"],
    )
    def test_terminal_statuses_raise(self, terminal_status, monkeypatch):
        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status=terminal_status, pid=123,
        )
        sigterm = MagicMock()
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", sigterm)

        with pytest.raises(executor_runs_db.RunAlreadyTerminalError) as excinfo:
            executor_runs_db.kill_run(run_id)

        assert excinfo.value.status == terminal_status
        assert excinfo.value.run_id == run_id
        sigterm.assert_not_called()

        # Row must be untouched — no killed_at / kill_reason written
        row = executor_runs_db.get_run(run_id)
        assert row["status"] == terminal_status
        assert row["killed_at"] is None
        assert row["kill_reason"] is None


class TestKillRunNotFound:
    def test_missing_run_raises(self):
        with pytest.raises(executor_runs_db.RunNotFoundError):
            executor_runs_db.kill_run(99999)


class TestSigkillEscalation:
    """(c) A process that ignores SIGTERM is escalated to SIGKILL after the
    grace period expires."""

    def test_sigkill_fires_when_sigterm_ignored(self, monkeypatch):
        # Fake a process that never dies — is_alive always returns True.
        monkeypatch.setattr(
            executor_runs_db, "_is_process_alive", lambda pid: True,
        )
        sigterm = MagicMock()
        sigkill = MagicMock()
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", sigterm)
        monkeypatch.setattr(executor_runs_db, "_send_sigkill", sigkill)

        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running", pid=4242,
        )

        # Short timeout so the test finishes in <1s but still exercises the
        # polling loop (at least one sleep tick before escalation).
        start = time.monotonic()
        result = executor_runs_db.kill_run(
            run_id,
            reason="stuck process",
            sigterm_timeout=0.2,
            poll_interval=0.05,
        )
        elapsed = time.monotonic() - start

        sigterm.assert_called_once_with(4242)
        sigkill.assert_called_once_with(4242)
        assert result["escalated"] is True
        assert result["status"] == "killed"

        # Must have waited approximately the grace period before escalating —
        # this is the load-bearing assertion for "after 10s" in the spec.
        assert elapsed >= 0.2

        row = executor_runs_db.get_run(run_id)
        assert row["status"] == "killed"
        assert row["killed_at"] is not None
        assert row["kill_reason"] == "stuck process"

    def test_default_sigterm_timeout_is_10_seconds(self):
        """The module-level default must match the spec's 10-second grace.
        Not a runtime test — guards against accidental edits to the constant."""
        assert executor_runs_db.KILL_SIGTERM_TIMEOUT_SECONDS == 10.0


# ---------------------------------------------------------------------------
# POST /api/executor/run/<id>/kill — HTTP contract
# ---------------------------------------------------------------------------


def _wait_for_status(run_id: int, expected: str, timeout: float = 2.0) -> dict:
    """Poll ``get_run`` until the row's status is ``expected`` or the
    timeout expires. Used to synchronize with the background kill thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = executor_runs_db.get_run(run_id)
        if row and (row.get("status") or "").lower() == expected:
            return row
        time.sleep(0.02)
    return executor_runs_db.get_run(run_id) or {}


class TestKillEndpoint:
    def test_returns_404_when_run_missing(self, client):
        resp = client.post("/api/executor/run/99999/kill")
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    def test_returns_409_when_run_already_terminal(self, client):
        """Acceptance (b): 409 on already-done run."""
        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="success", pid=123,
        )
        resp = client.post(f"/api/executor/run/{run_id}/kill")
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["status"] == "success"
        assert "terminal" in body["error"].lower()

    @pytest.mark.parametrize(
        "terminal_status", ["failed", "error", "killed", "timeout"],
    )
    def test_409_for_various_terminal_states(self, client, terminal_status):
        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status=terminal_status, pid=123,
        )
        resp = client.post(f"/api/executor/run/{run_id}/kill")
        assert resp.status_code == 409

    def test_returns_202_and_marks_killed(self, client, monkeypatch):
        """202 on a killable run, and the background thread eventually marks
        the row killed."""
        monkeypatch.setattr(
            executor_runs_db, "_is_process_alive", lambda pid: False,
        )
        sigterm = MagicMock()
        sigkill = MagicMock()
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", sigterm)
        monkeypatch.setattr(executor_runs_db, "_send_sigkill", sigkill)
        monkeypatch.setattr(
            executor_runs_db, "KILL_SIGTERM_TIMEOUT_SECONDS", 0.1,
        )
        monkeypatch.setattr(
            executor_runs_db, "KILL_POLL_INTERVAL_SECONDS", 0.02,
        )

        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running", pid=123,
        )
        resp = client.post(
            f"/api/executor/run/{run_id}/kill",
            json={"reason": "operator asked"},
        )
        assert resp.status_code == 202
        body = resp.get_json()
        assert body["status"] == "killing"
        assert body["run_id"] == run_id
        assert body["pid"] == 123

        row = _wait_for_status(run_id, "killed")
        assert row["status"] == "killed"
        assert row["kill_reason"] == "operator asked"
        sigterm.assert_called_once()

    def test_202_without_body(self, client, monkeypatch):
        """Endpoint accepts POSTs with no JSON body; reason defaults to NULL."""
        monkeypatch.setattr(
            executor_runs_db, "_is_process_alive", lambda pid: False,
        )
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", MagicMock())
        monkeypatch.setattr(
            executor_runs_db, "KILL_SIGTERM_TIMEOUT_SECONDS", 0.1,
        )

        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running", pid=123,
        )
        resp = client.post(f"/api/executor/run/{run_id}/kill")
        assert resp.status_code == 202

        row = _wait_for_status(run_id, "killed")
        assert row["status"] == "killed"
        assert row["kill_reason"] is None

    def test_blank_reason_treated_as_null(self, client, monkeypatch):
        monkeypatch.setattr(
            executor_runs_db, "_is_process_alive", lambda pid: False,
        )
        monkeypatch.setattr(executor_runs_db, "_send_sigterm", MagicMock())
        monkeypatch.setattr(
            executor_runs_db, "KILL_SIGTERM_TIMEOUT_SECONDS", 0.1,
        )

        run_id = executor_runs_db.record_run(
            jira_key="TK-468", status="running", pid=123,
        )
        resp = client.post(
            f"/api/executor/run/{run_id}/kill", json={"reason": "   "},
        )
        assert resp.status_code == 202

        row = _wait_for_status(run_id, "killed")
        assert row["kill_reason"] is None


# ---------------------------------------------------------------------------
# Process-liveness helper — cheap smoke test of the Windows/Unix dispatch
# ---------------------------------------------------------------------------


class TestIsProcessAlive:
    def test_returns_false_for_none_or_invalid(self):
        assert executor_runs_db._is_process_alive(None) is False
        assert executor_runs_db._is_process_alive(0) is False
        assert executor_runs_db._is_process_alive(-1) is False

    def test_returns_true_for_current_process(self):
        import os
        assert executor_runs_db._is_process_alive(os.getpid()) is True
