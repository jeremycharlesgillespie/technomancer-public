"""Tests for aimm.state — dataclass, persistence, and set_state_dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from filelock import FileLock

from aimm import state as aimm_state
from aimm.state import AimmState, load_state, save_state, set_state_dir


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect state + lock files to a temp directory for every test."""
    monkeypatch.setattr(aimm_state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(aimm_state, "STATE_FILE", tmp_path / ".aimm_state.json")
    monkeypatch.setattr(aimm_state, "LOCK_FILE", tmp_path / ".aimm_state.lock")
    monkeypatch.setattr(
        aimm_state, "_lock", FileLock(str(tmp_path / ".aimm_state.lock"), timeout=10)
    )


# ---------------------------------------------------------------------------
# AimmState dataclass
# ---------------------------------------------------------------------------

class TestAimmState:
    def test_default_values(self):
        s = AimmState()
        assert s.last_cycle_at is None
        assert s.cycle_count == 0
        assert s.approved_keys_today == []
        assert s.drafted_keys_today == []
        assert s.rate_limit_remaining is None

    def test_round_trip_through_dict(self):
        s = AimmState(
            last_cycle_at="2026-04-18T10:00:00",
            cycle_count=7,
            approved_keys_today=["TK-1", "TK-2"],
            drafted_keys_today=["TK-10"],
            rate_limit_remaining=42000,
        )
        restored = AimmState.from_dict(s.to_dict())
        assert restored == s

    def test_from_dict_missing_keys(self):
        s = AimmState.from_dict({})
        assert s == AimmState()

    def test_from_dict_extra_keys_ignored(self):
        s = AimmState.from_dict({"cycle_count": 3, "unknown": "x"})
        assert s.cycle_count == 3

    def test_default_lists_not_shared(self):
        """Mutating one instance's list must not affect a sibling."""
        a = AimmState()
        b = AimmState()
        a.approved_keys_today.append("TK-1")
        assert b.approved_keys_today == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_round_trip(self):
        s = AimmState(
            last_cycle_at="2026-04-18T09:30:00",
            cycle_count=12,
            approved_keys_today=["TK-100", "TK-101"],
            drafted_keys_today=["TK-200"],
            rate_limit_remaining=15000,
        )
        save_state(s)

        loaded = load_state()
        assert loaded.last_cycle_at == "2026-04-18T09:30:00"
        assert loaded.cycle_count == 12
        assert loaded.approved_keys_today == ["TK-100", "TK-101"]
        assert loaded.drafted_keys_today == ["TK-200"]
        assert loaded.rate_limit_remaining == 15000

    def test_load_nonexistent_returns_fresh_state(self):
        assert not aimm_state.STATE_FILE.exists()
        loaded = load_state()
        assert loaded == AimmState()

    def test_load_corrupt_json_returns_fresh_state(self):
        aimm_state.STATE_FILE.write_text("{not valid json", encoding="utf-8")
        loaded = load_state()
        assert loaded == AimmState()

    def test_atomic_write_removes_tmp_file(self):
        save_state(AimmState(cycle_count=3))
        assert aimm_state.STATE_FILE.exists()
        assert not aimm_state.STATE_FILE.with_suffix(".tmp").exists()

    def test_file_contents_are_valid_json(self):
        save_state(AimmState(cycle_count=5, approved_keys_today=["TK-7"]))
        data = json.loads(aimm_state.STATE_FILE.read_text(encoding="utf-8"))
        assert data["cycle_count"] == 5
        assert data["approved_keys_today"] == ["TK-7"]


# ---------------------------------------------------------------------------
# set_state_dir
# ---------------------------------------------------------------------------

class TestSetStateDir:
    def test_writes_land_in_new_dir(self, tmp_path):
        target = tmp_path / "custom"
        set_state_dir(target)

        save_state(AimmState(cycle_count=9))

        expected = target / ".aimm_state.json"
        assert expected.exists()
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert data["cycle_count"] == 9

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "nested" / "does" / "not" / "exist"
        assert not target.exists()

        set_state_dir(target)

        assert target.is_dir()

    def test_load_reads_from_new_dir(self, tmp_path):
        target = tmp_path / "custom2"
        set_state_dir(target)

        save_state(AimmState(cycle_count=11, rate_limit_remaining=99))
        loaded = load_state()

        assert loaded.cycle_count == 11
        assert loaded.rate_limit_remaining == 99

    def test_accepts_string_path(self, tmp_path):
        target = tmp_path / "as_string"
        set_state_dir(str(target))

        save_state(AimmState(cycle_count=1))

        assert (target / ".aimm_state.json").exists()

    def test_missing_state_file_in_new_dir_returns_fresh(self, tmp_path):
        target = tmp_path / "fresh"
        set_state_dir(target)

        loaded = load_state()

        assert loaded == AimmState()
