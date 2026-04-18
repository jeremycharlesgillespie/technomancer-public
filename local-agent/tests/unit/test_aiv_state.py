"""Tests for aiv.state — AIV daemon state model and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from filelock import FileLock

from aiv.state import (
    AivState,
    load_state,
    save_state,
    set_state_dir,
)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect all state files to a temp directory."""
    monkeypatch.setattr("aiv.state.STATE_DIR", tmp_path)
    monkeypatch.setattr("aiv.state.STATE_FILE", tmp_path / ".aiv_state.json")
    monkeypatch.setattr("aiv.state.LOCK_FILE", tmp_path / ".aiv_state.lock")
    monkeypatch.setattr(
        "aiv.state._lock",
        FileLock(str(tmp_path / ".aiv_state.lock"), timeout=10),
    )


# ---------------------------------------------------------------------------
# AivState dataclass
# ---------------------------------------------------------------------------

class TestAivState:
    def test_default_values(self):
        state = AivState()
        assert state.last_cycle_at is None
        assert state.cycle_count == 0
        assert state.validated_today == 0
        assert state.queue_depth == 0

    def test_round_trip_in_memory(self):
        state = AivState(
            last_cycle_at="2026-04-18T10:00:00",
            cycle_count=7,
            validated_today=3,
            queue_depth=12,
        )
        restored = AivState.from_dict(state.to_dict())
        assert restored.last_cycle_at == "2026-04-18T10:00:00"
        assert restored.cycle_count == 7
        assert restored.validated_today == 3
        assert restored.queue_depth == 12

    def test_from_dict_missing_keys(self):
        state = AivState.from_dict({})
        assert state.last_cycle_at is None
        assert state.cycle_count == 0
        assert state.validated_today == 0
        assert state.queue_depth == 0

    def test_from_dict_extra_keys_ignored(self):
        state = AivState.from_dict({"cycle_count": 5, "unknown_field": "ignored"})
        assert state.cycle_count == 5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_round_trip(self):
        state = AivState(
            last_cycle_at="2026-04-18T11:30:00",
            cycle_count=42,
            validated_today=8,
            queue_depth=4,
        )
        save_state(state)
        loaded = load_state()
        assert loaded.last_cycle_at == "2026-04-18T11:30:00"
        assert loaded.cycle_count == 42
        assert loaded.validated_today == 8
        assert loaded.queue_depth == 4

    def test_load_missing_file_returns_fresh(self):
        # No prior save_state() call — the state file does not exist.
        state = load_state()
        assert isinstance(state, AivState)
        assert state.last_cycle_at is None
        assert state.cycle_count == 0
        assert state.validated_today == 0
        assert state.queue_depth == 0

    def test_load_corrupt_json_returns_fresh(self, tmp_path, monkeypatch):
        state_file = tmp_path / ".aiv_state.json"
        monkeypatch.setattr("aiv.state.STATE_FILE", state_file)
        state_file.write_text("{not valid json", encoding="utf-8")
        state = load_state()
        assert state.cycle_count == 0

    def test_atomic_write_leaves_no_tmp(self, tmp_path, monkeypatch):
        state_file = tmp_path / ".aiv_state.json"
        monkeypatch.setattr("aiv.state.STATE_FILE", state_file)

        save_state(AivState(cycle_count=99))

        assert state_file.exists()
        assert not state_file.with_suffix(".tmp").exists()
        on_disk = json.loads(state_file.read_text(encoding="utf-8"))
        assert on_disk["cycle_count"] == 99


# ---------------------------------------------------------------------------
# set_state_dir
# ---------------------------------------------------------------------------

class TestSetStateDir:
    def test_redirects_subsequent_writes(self, tmp_path):
        new_dir = tmp_path / "project_b"
        set_state_dir(new_dir)

        state = AivState(cycle_count=3, queue_depth=7)
        save_state(state)

        # File should land under the new directory.
        expected = new_dir / ".aiv_state.json"
        assert expected.exists()

        loaded = load_state()
        assert loaded.cycle_count == 3
        assert loaded.queue_depth == 7

    def test_creates_missing_directory(self, tmp_path):
        nested = tmp_path / "nested" / "aiv_state"
        assert not nested.exists()

        set_state_dir(nested)

        assert nested.is_dir()

    def test_module_globals_updated(self, tmp_path):
        import aiv.state as state_module

        target = tmp_path / "custom"
        set_state_dir(target)

        assert state_module.STATE_DIR == target
        assert state_module.STATE_FILE == target / ".aiv_state.json"
        assert state_module.LOCK_FILE == target / ".aiv_state.lock"
