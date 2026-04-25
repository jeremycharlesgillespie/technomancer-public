"""Tests for aim.state — shared state model and persistence."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import aim.state as aim_state_mod
from aim.state import (
    AIMState,
    WorkerState,
    assign_idea_to_worker,
    clear_pid,
    clear_worker_assignment,
    get_project_name,
    is_process_alive,
    load_state,
    read_pid,
    record_completion,
    record_worker_failure,
    reset_worker_failures,
    save_state,
    set_state_dir,
    update_worker_heartbeat,
    update_worker_status,
    write_pid,
)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect all state files to a temp directory."""
    monkeypatch.setattr("aim.state.STATE_DIR", tmp_path)
    monkeypatch.setattr("aim.state.STATE_FILE", tmp_path / ".aim_state.json")
    monkeypatch.setattr("aim.state.LOCK_FILE", tmp_path / ".aim_state.lock")
    monkeypatch.setattr("aim.state.PID_FILE", tmp_path / "aim.pid")

    # Recreate the lock with the new path
    from filelock import FileLock
    monkeypatch.setattr("aim.state._lock", FileLock(str(tmp_path / ".aim_state.lock"), timeout=10))


# ---------------------------------------------------------------------------
# WorkerState tests
# ---------------------------------------------------------------------------

class TestWorkerState:
    def test_default_values(self):
        ws = WorkerState()
        assert ws.pid is None
        assert ws.status == "idle"
        assert ws.current_idea_id is None
        assert ws.consecutive_failures == 0

    def test_round_trip(self):
        ws = WorkerState(
            pid=1234,
            status="executing",
            current_idea_id="idea-042",
            started_at="2026-04-14T10:00:00",
            last_heartbeat="2026-04-14T10:01:00",
            last_observation="Working on it",
            consecutive_failures=2,
        )
        d = ws.to_dict()
        restored = WorkerState.from_dict(d)
        assert restored.pid == 1234
        assert restored.status == "executing"
        assert restored.current_idea_id == "idea-042"
        assert restored.consecutive_failures == 2

    def test_from_dict_missing_keys(self):
        ws = WorkerState.from_dict({})
        assert ws.pid is None
        assert ws.status == "idle"
        assert ws.consecutive_failures == 0

    def test_from_dict_extra_keys_ignored(self):
        ws = WorkerState.from_dict({"pid": 99, "unknown_field": "hello"})
        assert ws.pid == 99


# ---------------------------------------------------------------------------
# AIMState tests
# ---------------------------------------------------------------------------

class TestAIMState:
    def test_default_values(self):
        state = AIMState()
        assert state.manager_pid is None
        assert state.cycle_count == 0
        assert state.completions_today == 0
        assert isinstance(state.worker, WorkerState)
        assert isinstance(state.board_snapshot, dict)

    def test_round_trip(self):
        state = AIMState(
            manager_pid=5678,
            manager_started_at="2026-04-14T09:00:00",
            last_cycle="2026-04-14T10:00:00",
            last_completion="2026-04-14T09:30:00",
            completions_today=5,
            completions_today_date="2026-04-14",
            cycle_count=42,
            board_snapshot={"todo": 15, "in_progress": 2},
            worker=WorkerState(pid=1234, status="watching"),
            last_error="",
            last_discord_notify={"worker_failures": "2026-04-14T09:00:00"},
        )
        d = state.to_dict()
        restored = AIMState.from_dict(d)
        assert restored.manager_pid == 5678
        assert restored.cycle_count == 42
        assert restored.completions_today == 5
        assert restored.worker.pid == 1234
        assert restored.worker.status == "watching"
        assert restored.last_discord_notify["worker_failures"] == "2026-04-14T09:00:00"

    def test_from_dict_empty(self):
        state = AIMState.from_dict({})
        assert state.manager_pid is None
        assert state.cycle_count == 0
        assert state.worker.status == "idle"

    def test_clean_shutdown_default_false(self):
        state = AIMState()
        assert state.clean_shutdown is False

    def test_clean_shutdown_round_trip(self):
        state = AIMState(clean_shutdown=True)
        restored = AIMState.from_dict(state.to_dict())
        assert restored.clean_shutdown is True

    def test_clean_shutdown_missing_key_defaults_false(self):
        state = AIMState.from_dict({"manager_pid": 1, "cycle_count": 7})
        assert state.clean_shutdown is False


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self):
        state = AIMState(
            manager_pid=os.getpid(),
            cycle_count=10,
            completions_today=3,
        )
        save_state(state)
        loaded = load_state()
        assert loaded.manager_pid == os.getpid()
        assert loaded.cycle_count == 10
        assert loaded.completions_today == 3

    def test_load_nonexistent_returns_default(self):
        state = load_state()
        assert state.manager_pid is None
        assert state.cycle_count == 0

    def test_load_corrupt_json_returns_default(self, tmp_path, monkeypatch):
        state_file = tmp_path / ".aim_state.json"
        monkeypatch.setattr("aim.state.STATE_FILE", state_file)
        state_file.write_text("{not valid json", encoding="utf-8")
        state = load_state()
        assert state.manager_pid is None

    def test_atomic_write(self, tmp_path, monkeypatch):
        """Verify the .tmp file is used for atomic writes."""
        state_file = tmp_path / ".aim_state.json"
        monkeypatch.setattr("aim.state.STATE_FILE", state_file)

        state = AIMState(cycle_count=99)
        save_state(state)

        # The .tmp file should not exist after save
        assert not (state_file.with_suffix(".tmp")).exists()
        # The state file should exist
        assert state_file.exists()
        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert loaded["cycle_count"] == 99

    def test_concurrent_access(self):
        """Multiple threads writing should not corrupt state."""
        errors = []

        def writer(n):
            try:
                for _ in range(10):
                    state = load_state()
                    state.cycle_count += 1
                    save_state(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access errors: {errors}"
        # Final count should be > 0 (exact value depends on race)
        final = load_state()
        assert final.cycle_count > 0


# ---------------------------------------------------------------------------
# Convenience mutator tests
# ---------------------------------------------------------------------------

class TestMutators:
    def test_update_worker_heartbeat(self):
        state = AIMState()
        save_state(state)

        update_worker_heartbeat()

        loaded = load_state()
        assert loaded.worker.last_heartbeat is not None
        assert "T" in loaded.worker.last_heartbeat  # ISO format

    def test_update_worker_status(self):
        state = AIMState()
        save_state(state)

        update_worker_status("executing", observation="Starting idea-042")

        loaded = load_state()
        assert loaded.worker.status == "executing"
        assert loaded.worker.last_observation == "Starting idea-042"
        assert loaded.worker.last_heartbeat is not None

    def test_update_worker_status_with_idea_id(self):
        state = AIMState()
        save_state(state)

        update_worker_status("assigned", idea_id="idea-042")

        loaded = load_state()
        assert loaded.worker.current_idea_id == "idea-042"

    def test_update_worker_status_clear_idea(self):
        state = AIMState(worker=WorkerState(current_idea_id="idea-042"))
        save_state(state)

        update_worker_status("idle", idea_id=None)

        loaded = load_state()
        assert loaded.worker.current_idea_id is None

    def test_update_worker_status_preserve_idea(self):
        """When idea_id is omitted (sentinel ...), preserve existing value."""
        state = AIMState(worker=WorkerState(current_idea_id="idea-042"))
        save_state(state)

        update_worker_status("watching")  # No idea_id arg

        loaded = load_state()
        assert loaded.worker.current_idea_id == "idea-042"

    def test_assign_idea_to_worker(self):
        state = AIMState()
        save_state(state)

        assign_idea_to_worker("idea-099")

        loaded = load_state()
        assert loaded.worker.current_idea_id == "idea-099"
        assert loaded.worker.status == "assigned"
        assert loaded.worker.started_at is not None

    def test_clear_worker_assignment(self):
        state = AIMState(worker=WorkerState(
            current_idea_id="idea-099",
            status="executing",
            started_at="2026-04-14T10:00:00",
        ))
        save_state(state)

        clear_worker_assignment()

        loaded = load_state()
        assert loaded.worker.current_idea_id is None
        assert loaded.worker.status == "idle"
        assert loaded.worker.started_at is None

    def test_record_completion(self):
        state = AIMState(completions_today=2, completions_today_date="2026-04-14")
        save_state(state)

        with patch("aim.state.datetime") as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2026-04-14T11:00:00"
            mock_dt.now.return_value.strftime.return_value = "2026-04-14"
            record_completion()

        loaded = load_state()
        assert loaded.completions_today == 3
        assert loaded.last_completion is not None

    def test_record_completion_new_day_resets(self):
        state = AIMState(completions_today=10, completions_today_date="2026-04-13")
        save_state(state)

        with patch("aim.state.datetime") as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2026-04-14T08:00:00"
            mock_dt.now.return_value.strftime.return_value = "2026-04-14"
            record_completion()

        loaded = load_state()
        assert loaded.completions_today == 1  # Reset to 1 for new day

    def test_record_worker_failure(self):
        state = AIMState(worker=WorkerState(consecutive_failures=1))
        save_state(state)

        record_worker_failure()

        loaded = load_state()
        assert loaded.worker.consecutive_failures == 2

    def test_reset_worker_failures(self):
        state = AIMState(worker=WorkerState(consecutive_failures=5))
        save_state(state)

        reset_worker_failures()

        loaded = load_state()
        assert loaded.worker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

class TestPIDHelpers:
    def test_write_and_read_pid(self):
        write_pid()
        pid = read_pid()
        assert pid == os.getpid()

    def test_read_pid_nonexistent(self):
        assert read_pid() is None

    def test_clear_pid(self):
        write_pid()
        clear_pid()
        assert read_pid() is None

    def test_is_process_alive_current(self):
        assert is_process_alive(os.getpid()) is True

    def test_is_process_alive_nonexistent(self):
        # PID 99999999 should not exist
        assert is_process_alive(99999999) is False

    def test_is_process_alive_zero(self):
        assert is_process_alive(0) is False

    def test_is_process_alive_negative(self):
        assert is_process_alive(-1) is False


# ---------------------------------------------------------------------------
# set_state_dir tests
# ---------------------------------------------------------------------------

class TestSetStateDir:
    def test_writes_land_in_new_dir(self, tmp_path):
        target = tmp_path / "custom"
        set_state_dir(target)

        save_state(AIMState(cycle_count=9))

        expected = target / ".aim_state.json"
        assert expected.exists()
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert data["cycle_count"] == 9

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "nested" / "does" / "not" / "exist"
        assert not target.exists()

        set_state_dir(target)

        assert target.is_dir()

    def test_load_reads_from_new_dir(self, tmp_path):
        target = tmp_path / "project_a"
        set_state_dir(target)

        save_state(AIMState(cycle_count=11, completions_today=4))
        loaded = load_state()

        assert loaded.cycle_count == 11
        assert loaded.completions_today == 4

    def test_accepts_string_path(self, tmp_path):
        target = tmp_path / "as_string"
        set_state_dir(str(target))

        save_state(AIMState(cycle_count=1))

        assert (target / ".aim_state.json").exists()

    def test_missing_state_file_in_new_dir_returns_default(self, tmp_path):
        target = tmp_path / "fresh_project"
        set_state_dir(target)

        loaded = load_state()

        assert loaded.manager_pid is None
        assert loaded.cycle_count == 0

    def test_sequential_calls_dont_corrupt_state(self, tmp_path):
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"

        set_state_dir(dir_a)
        save_state(AIMState(cycle_count=10))

        set_state_dir(dir_b)
        save_state(AIMState(cycle_count=20))

        # Switch back — dir_a state is intact
        set_state_dir(dir_a)
        assert load_state().cycle_count == 10

        # Switch to dir_b — separate state
        set_state_dir(dir_b)
        assert load_state().cycle_count == 20

    def test_pid_file_in_new_dir(self, tmp_path):
        target = tmp_path / "project_pid"
        set_state_dir(target)

        write_pid()
        pid = read_pid()

        assert pid == os.getpid()
        assert (target / "aim.pid").exists()


# ---------------------------------------------------------------------------
# get_project_name tests
# ---------------------------------------------------------------------------

class TestGetProjectName:
    def test_returns_name_from_project_dir(self, tmp_path, monkeypatch):
        target = tmp_path / "projects" / "TK"
        monkeypatch.setattr("aim.state.STATE_DIR", target)

        assert get_project_name() == "TK"

    def test_returns_none_for_default_package_dir(self, monkeypatch):
        from pathlib import Path
        default = Path(aim_state_mod.__file__).parent
        monkeypatch.setattr("aim.state.STATE_DIR", default)

        assert get_project_name() is None

    def test_sequential_set_state_dir_updates_name(self, tmp_path):
        dir_a = tmp_path / "projects" / "40acres"
        dir_b = tmp_path / "projects" / "TK"

        set_state_dir(dir_a)
        assert get_project_name() == "40acres"

        set_state_dir(dir_b)
        assert get_project_name() == "TK"
