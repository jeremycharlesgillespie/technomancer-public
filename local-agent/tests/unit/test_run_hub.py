"""Unit tests for run_hub.py — the standalone hub launcher.

These tests verify the launcher imports cleanly and is wired to the
configurable board_port. They do NOT bind a port or start Flask; the
smoke test for that lives outside the test suite (Mac dev only).
"""

from __future__ import annotations

import importlib

import pytest


def test_run_hub_imports_cleanly() -> None:
    """run_hub must import without side effects (no Flask app start)."""
    module = importlib.import_module("run_hub")
    assert hasattr(module, "main")
    assert callable(module.main)


def test_board_port_default_is_8322() -> None:
    """Default port stays at 8322 so existing Tailscale URLs keep working."""
    from agent.config import settings

    assert settings.board_port == 8322


def test_web_module_uses_settings_board_port() -> None:
    """idea_board.web.BOARD_PORT must read from settings, not a hardcoded literal."""
    from agent.config import settings
    from idea_board import web

    assert web.BOARD_PORT == settings.board_port


def test_run_hub_main_blocks_on_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() should call start_idea_board() and then block until SIGTERM/SIGINT.

    We patch start_idea_board to a no-op and time.sleep to flip the stop flag,
    then verify main exits via SystemExit(0).
    """
    import run_hub

    started: dict[str, bool] = {"called": False}

    def _fake_start_idea_board() -> None:
        started["called"] = True

    # Patch the import inside main()
    monkeypatch.setattr("idea_board.web.start_idea_board", _fake_start_idea_board)

    # Make time.sleep raise after one call so the loop exits via SIGINT path
    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Simulate Ctrl+C by sending SIGINT to ourselves
        import os
        import signal as _signal

        os.kill(os.getpid(), _signal.SIGINT)

    monkeypatch.setattr(run_hub.time, "sleep", _fake_sleep)

    with pytest.raises(SystemExit) as excinfo:
        run_hub.main()

    assert excinfo.value.code == 0
    assert started["called"] is True
    assert len(sleep_calls) >= 1
