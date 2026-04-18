"""Tests for aiv.main — AIV daemon loop + one-cycle processor."""

from __future__ import annotations

import json
import signal
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from filelock import FileLock

from agent import aiv_schema
from aiv import main as aiv_main
from aiv import state as aiv_state
from aiv.scorer import StoryQualityScores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect aiv_schema at a temp SQLite DB per-test."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect AIV state files to a temp directory."""
    state_dir = tmp_path / "aiv_state"
    state_dir.mkdir()
    monkeypatch.setattr(aiv_state, "STATE_DIR", state_dir)
    monkeypatch.setattr(aiv_state, "STATE_FILE", state_dir / ".aiv_state.json")
    monkeypatch.setattr(aiv_state, "LOCK_FILE", state_dir / ".aiv_state.lock")
    monkeypatch.setattr(aiv_state, "PID_FILE", state_dir / "aiv.pid")
    monkeypatch.setattr(
        aiv_state,
        "_lock",
        FileLock(str(state_dir / ".aiv_state.lock"), timeout=10),
    )


@pytest.fixture(autouse=True)
def _reset_shutdown_flag():
    """Ensure the module-level shutdown flag starts False each test."""
    aiv_main._shutdown_requested = False
    yield
    aiv_main._shutdown_requested = False


def _seed_pending(story_key: str, paths=None, enqueued_at="2026-04-18T05:30:00") -> None:
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    conn.execute(
        "INSERT INTO aiv_pending "
        "(story_key, merged_at, diff_paths_json, enqueued_at) "
        "VALUES (?, ?, ?, ?)",
        (story_key, "2026-04-18T05:29:00", json.dumps(list(paths or [])), enqueued_at),
    )
    conn.commit()


def _happy_scores(**overrides) -> StoryQualityScores:
    base = dict(
        meets_requirements=9,
        code_quality=8,
        test_quality=7,
        security_safety=10,
        scope_discipline=9,
        edge_cases=6,
        product_impact=8,
        reasoning_map={"meets_requirements": "Covers all AC."},
        red_flags=[],
        error="",
    )
    base.update(overrides)
    return StoryQualityScores(**base)


# ---------------------------------------------------------------------------
# fetch_pending_rows + queue_depth
# ---------------------------------------------------------------------------

class TestFetchPendingRows:
    def test_empty_queue_returns_empty_list(self):
        assert aiv_main.fetch_pending_rows() == []

    def test_returns_rows_ordered_by_enqueued_at(self):
        _seed_pending("TK-2", enqueued_at="2026-04-18T06:00:00")
        _seed_pending("TK-1", enqueued_at="2026-04-18T05:00:00")

        rows = aiv_main.fetch_pending_rows()
        assert [r.story_key for r in rows] == ["TK-1", "TK-2"]

    def test_decodes_diff_paths_json(self):
        _seed_pending("TK-1", paths=["agent/a.py", "tests/b.py"])

        rows = aiv_main.fetch_pending_rows()
        assert rows[0].diff_paths == ["agent/a.py", "tests/b.py"]

    def test_malformed_diff_paths_json_decodes_to_empty_list(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        conn.execute(
            "INSERT INTO aiv_pending "
            "(story_key, merged_at, diff_paths_json, enqueued_at) "
            "VALUES (?, ?, ?, ?)",
            ("TK-1", "2026-04-18T05:29:00", "{not-json", "2026-04-18T05:30:00"),
        )
        conn.commit()

        rows = aiv_main.fetch_pending_rows()
        assert rows[0].diff_paths == []


class TestQueueDepth:
    def test_empty_returns_zero(self):
        assert aiv_main.queue_depth() == 0

    def test_counts_rows(self):
        _seed_pending("TK-1")
        _seed_pending("TK-2")
        _seed_pending("TK-3")
        assert aiv_main.queue_depth() == 3


# ---------------------------------------------------------------------------
# process_row — single-row happy + failure paths
# ---------------------------------------------------------------------------

class TestProcessRow:
    def test_happy_path_persists_story_quality_row(self, monkeypatch):
        _seed_pending("TK-1", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_happy_scores()) as mock_score:
            result = aiv_main.process_row(row)

        assert result is not None
        mock_score.assert_called_once()

        conn = aiv_schema._get_conn()
        quality_row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert quality_row is not None
        assert quality_row["meets_requirements"] == 9
        # aiv_pending must be drained by persist.record().
        pending = conn.execute(
            "SELECT * FROM aiv_pending WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert pending is None

    def test_scorer_exception_swallowed_row_remains(self):
        _seed_pending("TK-1", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", side_effect=RuntimeError("boom")):
            result = aiv_main.process_row(row)

        assert result is None
        # Row stays in the queue so a later cycle can retry.
        conn = aiv_schema._get_conn()
        pending = conn.execute(
            "SELECT * FROM aiv_pending WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert pending is not None

    def test_classifier_drives_verification_method(self):
        _seed_pending("TK-1", paths=["idea_board/web.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_happy_scores()):
            aiv_main.process_row(row)

        conn = aiv_schema._get_conn()
        quality = conn.execute(
            "SELECT verification_method FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert quality["verification_method"] == "web-render"


# ---------------------------------------------------------------------------
# run_one_cycle — acceptance criterion
# ---------------------------------------------------------------------------

class TestRunOneCycle:
    def test_processes_two_pending_rows_end_to_end(self):
        """Acceptance: one-cycle version processes 2 pending rows end-to-end
        against mocked scorer + in-memory SQLite — 2 story_quality rows
        appear, aiv_pending empties."""
        _seed_pending("TK-1", paths=["agent/a.py"])
        _seed_pending("TK-2", paths=["agent/b.py"])

        with patch("aiv.main.scorer.score", return_value=_happy_scores()) as mock_score:
            processed = aiv_main.run_one_cycle()

        assert processed == 2
        assert mock_score.call_count == 2

        conn = aiv_schema._get_conn()
        quality_keys = {
            r["story_key"]
            for r in conn.execute("SELECT story_key FROM story_quality").fetchall()
        }
        assert quality_keys == {"TK-1", "TK-2"}

        remaining = conn.execute("SELECT story_key FROM aiv_pending").fetchall()
        assert remaining == []

    def test_updates_state_after_cycle(self):
        _seed_pending("TK-1", paths=["agent/a.py"])

        with patch("aiv.main.scorer.score", return_value=_happy_scores()):
            aiv_main.run_one_cycle()

        s = aiv_state.load_state()
        assert s.cycle_count == 1
        assert s.last_cycle_at is not None
        assert s.validated_today == 1
        assert s.queue_depth == 0

    def test_empty_queue_still_bumps_cycle_count(self):
        processed = aiv_main.run_one_cycle()
        assert processed == 0
        assert aiv_state.load_state().cycle_count == 1

    def test_one_bad_row_does_not_stop_the_others(self):
        """Per-row try/except keeps the loop alive."""
        _seed_pending("TK-bad", paths=["agent/a.py"])
        _seed_pending("TK-good", paths=["agent/b.py"])

        def flaky_score(story, diff, verification_output, **_):
            if story["key"] == "TK-bad":
                raise RuntimeError("exploded")
            return _happy_scores()

        with patch("aiv.main.scorer.score", side_effect=flaky_score):
            processed = aiv_main.run_one_cycle()

        assert processed == 1

        conn = aiv_schema._get_conn()
        pending_keys = {
            r["story_key"]
            for r in conn.execute("SELECT story_key FROM aiv_pending").fetchall()
        }
        quality_keys = {
            r["story_key"]
            for r in conn.execute("SELECT story_key FROM story_quality").fetchall()
        }
        # Bad row survives in the queue; good row is scored + dequeued.
        assert pending_keys == {"TK-bad"}
        assert quality_keys == {"TK-good"}


# ---------------------------------------------------------------------------
# SIGTERM → clean exit + state saved
# ---------------------------------------------------------------------------

class TestSignalShutdown:
    def test_request_shutdown_sets_flag(self):
        assert aiv_main._shutdown_requested is False
        aiv_main._request_shutdown(signal.SIGTERM, None)
        assert aiv_main._shutdown_requested is True

    def test_run_exits_cleanly_and_saves_state_when_shutdown_requested(self):
        """Flag-before-start + mocked run_one_cycle → run() falls straight
        through the loop, reaches the post-loop cleanup, clears the PID
        file, and leaves a fresh state entry."""
        aiv_main._shutdown_requested = True

        # Stub the cycle (not executed under this flag) and sleep.
        with patch("aiv.main.run_one_cycle", return_value=0) as mock_cycle, \
             patch("aiv.main._install_signal_handlers"), \
             patch("aiv.main._setup_logging"):
            aiv_main.run()

        mock_cycle.assert_not_called()

        # PID file removed, state persisted.
        assert not aiv_state.PID_FILE.exists()
        s = aiv_state.load_state()
        assert s.manager_pid is None
        assert s.manager_started_at is not None  # set before the loop started

    def test_run_saves_state_after_shutdown_flag_set_mid_loop(self):
        """One cycle runs, then shutdown flag trips — run() exits cleanly
        and the final cycle_count survives."""
        call_log = []

        def stub_cycle():
            call_log.append(1)
            aiv_main._shutdown_requested = True
            return 0

        with patch("aiv.main.run_one_cycle", side_effect=stub_cycle), \
             patch("aiv.main._install_signal_handlers"), \
             patch("aiv.main._setup_logging"), \
             patch("aiv.main._interruptible_sleep"):
            aiv_main.run()

        assert call_log == [1]
        # Final state write completed.
        assert not aiv_state.PID_FILE.exists()
        assert aiv_state.load_state().manager_pid is None


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

class TestShowStatus:
    def test_reports_manager_pid_last_cycle_and_queue_depth(self, capsys):
        s = aiv_state.AivState(
            last_cycle_at="2026-04-18T10:00:00",
            cycle_count=5,
            validated_today=3,
            queue_depth=2,
            manager_pid=99999,
            manager_started_at="2026-04-18T09:55:00",
        )
        aiv_state.save_state(s)
        _seed_pending("TK-1")
        _seed_pending("TK-2")

        with patch("aiv.state.is_process_alive", return_value=False):
            aiv_main.show_status()

        out = capsys.readouterr().out
        assert "AIV Manager PID: 99999" in out
        assert "Last cycle:  2026-04-18T10:00:00" in out
        assert "Queue depth: 2" in out

    def test_no_state_prints_not_running(self, capsys):
        aiv_main.show_status()
        out = capsys.readouterr().out
        assert "AIV Manager PID: None" in out
        assert "not running" in out
        assert "Queue depth: 0" in out

    def test_alive_pid_reports_running(self, capsys):
        s = aiv_state.AivState(manager_pid=12345)
        aiv_state.save_state(s)

        with patch("aiv.state.is_process_alive", return_value=True):
            aiv_main.show_status()

        out = capsys.readouterr().out
        assert "(running)" in out


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------

class TestStopDaemon:
    def test_no_pid_file_prints_not_running(self, capsys):
        aiv_main.stop_daemon()
        out = capsys.readouterr().out
        assert "not running" in out.lower()

    def test_dead_pid_prints_not_running(self, capsys):
        aiv_state.PID_FILE.write_text("54321", encoding="utf-8")
        with patch("aiv.state.is_process_alive", return_value=False):
            aiv_main.stop_daemon()
        out = capsys.readouterr().out
        assert "not running" in out.lower()

    def test_live_pid_sends_kill_signal(self, capsys):
        aiv_state.PID_FILE.write_text("4242", encoding="utf-8")
        with patch("aiv.state.is_process_alive", return_value=True), \
             patch("sys.platform", "linux"), \
             patch("os.kill") as mock_kill:
            aiv_main.stop_daemon()

        mock_kill.assert_called_once_with(4242, signal.SIGTERM)
        out = capsys.readouterr().out
        assert "Sent termination signal" in out


# ---------------------------------------------------------------------------
# verify_tests_only — MVP stub
# ---------------------------------------------------------------------------

class TestVerifyTestsOnly:
    def test_returns_empty_string(self):
        row = aiv_main.PendingRow(
            story_key="TK-1",
            merged_at="2026-04-18T05:29:00",
            diff_paths=["agent/a.py"],
            enqueued_at="2026-04-18T05:30:00",
        )
        assert aiv_main.verify_tests_only(row) == ""
