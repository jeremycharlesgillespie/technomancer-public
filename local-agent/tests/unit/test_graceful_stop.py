"""Tests for aim.graceful_stop — file-based stop flag."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def stop_module(tmp_path, monkeypatch):
    """Redirect aim.state.STATE_DIR to tmp so tests don't pollute disk."""
    monkeypatch.setattr("aim.state.STATE_DIR", tmp_path)
    from aim import graceful_stop
    return graceful_stop


class TestIsStopRequested:
    def test_false_when_flag_absent(self, stop_module):
        assert stop_module.is_stop_requested() is False

    def test_true_when_flag_present(self, stop_module, tmp_path):
        (tmp_path / ".aim_stop_requested").write_text("test\n")
        assert stop_module.is_stop_requested() is True


class TestRequestStop:
    def test_creates_flag_with_reason(self, stop_module, tmp_path):
        path = stop_module.request_stop(reason="unit test")
        assert path.exists()
        assert "unit test" in path.read_text()

    def test_idempotent_overwrites(self, stop_module, tmp_path):
        stop_module.request_stop(reason="first")
        stop_module.request_stop(reason="second")
        assert "second" in (tmp_path / ".aim_stop_requested").read_text()

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "subdir" / "deep"
        monkeypatch.setattr("aim.state.STATE_DIR", nested)
        from aim import graceful_stop
        path = graceful_stop.request_stop(reason="nested")
        assert path.exists()


class TestClearStopFlag:
    def test_returns_false_when_no_flag(self, stop_module):
        assert stop_module.clear_stop_flag() is False

    def test_removes_flag_when_present(self, stop_module, tmp_path):
        flag = tmp_path / ".aim_stop_requested"
        flag.write_text("x")
        assert stop_module.clear_stop_flag() is True
        assert not flag.exists()


class TestReadReason:
    def test_empty_string_when_no_flag(self, stop_module):
        assert stop_module.read_reason() == ""

    def test_returns_flag_contents(self, stop_module, tmp_path):
        (tmp_path / ".aim_stop_requested").write_text("custom reason\n")
        assert stop_module.read_reason() == "custom reason"


class TestStateDirRebind:
    def test_picks_up_new_state_dir(self, tmp_path, monkeypatch):
        # Initial state dir
        first_dir = tmp_path / "first"
        first_dir.mkdir()
        monkeypatch.setattr("aim.state.STATE_DIR", first_dir)
        from aim import graceful_stop
        assert not graceful_stop.is_stop_requested()

        # Switch to a different dir mid-test (multi-project simulation)
        second_dir = tmp_path / "second"
        second_dir.mkdir()
        (second_dir / ".aim_stop_requested").write_text("y")
        monkeypatch.setattr("aim.state.STATE_DIR", second_dir)
        assert graceful_stop.is_stop_requested()
