"""Integration test — ``on_ready`` wires the readiness gate correctly.

Patches ``run_readiness_checks`` to return a failing required report and
asserts that ``on_ready`` fires the owner DM, crash-log write, and
``sys.exit(1)`` path in that order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import discord_memory_bot
from agent.startup_checks import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_TIMEOUT,
    CheckResult,
    ReadinessReport,
)


@pytest.fixture
def failing_report() -> ReadinessReport:
    """A ReadinessReport with two failing required checks."""
    return ReadinessReport(
        ok=False,
        skipped=False,
        results=[
            CheckResult(
                name="ollama",
                status=STATUS_FAIL,
                required=True,
                duration_ms=120,
                error="ConnectError: Connection refused",
                attempts=2,
            ),
            CheckResult(
                name="executor_db",
                status=STATUS_TIMEOUT,
                required=True,
                duration_ms=5000,
                error="timeout after 5.0s",
                attempts=2,
            ),
        ],
    )


@pytest.fixture
def passing_report() -> ReadinessReport:
    return ReadinessReport(
        ok=True,
        skipped=False,
        results=[
            CheckResult(
                name="ollama", status=STATUS_OK, required=True, duration_ms=5, attempts=1
            )
        ],
    )


def _run(coro):
    """Run a coroutine on a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_on_ready_readiness_failure_fires_crashlog_dm_exit_in_order(
    failing_report, tmp_path, monkeypatch
):
    """Failing readiness gate: crash log, owner DM, then sys.exit(1), in order."""
    call_order: list[str] = []

    # Redirect vault to a temp dir so the real Obsidian vault is untouched.
    monkeypatch.setattr(discord_memory_bot, "VAULT_PATH", tmp_path)

    # Patch run_readiness_checks → failing report.
    monkeypatch.setattr(
        discord_memory_bot,
        "run_readiness_checks",
        lambda *a, **kw: failing_report,
    )

    # Trap the three sinks so we can assert order without touching real state.
    def _fake_crash_log(failed, vault_path):
        call_order.append("crash_log_write")
        assert vault_path == tmp_path
        assert [r.name for r in failed] == ["ollama", "executor_db"]
        return Path(vault_path) / "LLM Memory" / "Permanent" / "crash_log.md"

    monkeypatch.setattr(
        discord_memory_bot,
        "_append_readiness_failure_to_crash_log",
        _fake_crash_log,
    )

    async def _fake_dm(discord_client, failed):
        call_order.append("owner_dm")
        assert [r.name for r in failed] == ["ollama", "executor_db"]
        return True

    monkeypatch.setattr(
        discord_memory_bot,
        "_dm_owner_about_readiness_failure",
        _fake_dm,
    )

    def _fake_exit(code):
        call_order.append(f"sys_exit_{code}")
        raise SystemExit(code)

    monkeypatch.setattr(discord_memory_bot.sys, "exit", _fake_exit)

    # Stub out the bits of on_ready that run before the gate: gateway health,
    # lifecycle webhook, and the client.user attribute.
    fake_client = MagicMock()
    fake_client.user = "fake-user#0001"
    monkeypatch.setattr(discord_memory_bot, "client", fake_client)

    gateway_mock = MagicMock()
    monkeypatch.setattr(
        discord_memory_bot,
        "get_gateway_health",
        lambda: gateway_mock,
    )
    monkeypatch.setattr(
        discord_memory_bot, "send_lifecycle_notification", lambda *a, **kw: None
    )

    # on_ready should propagate SystemExit from the stubbed sys.exit.
    with pytest.raises(SystemExit) as excinfo:
        _run(discord_memory_bot.on_ready())

    assert excinfo.value.code == 1
    assert gateway_mock.record_connect.called, "record_connect must run before gate"
    assert call_order == [
        "crash_log_write",
        "owner_dm",
        "sys_exit_1",
    ], f"Expected crash_log → DM → exit, got {call_order}"


def test_on_ready_readiness_passing_skips_failure_handler(
    passing_report, monkeypatch
):
    """Passing readiness gate: failure sinks must not fire."""
    # Prevent on_ready from running the heavy init beyond the gate — raise a
    # sentinel exception from the step immediately after so we can stop
    # cleanly without instantiating memory/Claude/etc.
    class _StopAfterGate(Exception):
        pass

    monkeypatch.setattr(
        discord_memory_bot,
        "run_readiness_checks",
        lambda *a, **kw: passing_report,
    )

    def _boom(*a, **kw):
        raise _StopAfterGate

    # init_memory_system is the first call after the gate — intercept it.
    monkeypatch.setattr(discord_memory_bot, "init_memory_system", _boom)

    crash_log_mock = MagicMock()
    dm_mock = AsyncMock(return_value=True)
    exit_mock = MagicMock()
    monkeypatch.setattr(
        discord_memory_bot,
        "_append_readiness_failure_to_crash_log",
        crash_log_mock,
    )
    monkeypatch.setattr(
        discord_memory_bot,
        "_dm_owner_about_readiness_failure",
        dm_mock,
    )
    monkeypatch.setattr(discord_memory_bot.sys, "exit", exit_mock)

    fake_client = MagicMock()
    fake_client.user = "fake-user#0001"
    monkeypatch.setattr(discord_memory_bot, "client", fake_client)
    monkeypatch.setattr(
        discord_memory_bot, "get_gateway_health", lambda: MagicMock()
    )
    monkeypatch.setattr(
        discord_memory_bot, "send_lifecycle_notification", lambda *a, **kw: None
    )

    with pytest.raises(_StopAfterGate):
        _run(discord_memory_bot.on_ready())

    assert not crash_log_mock.called
    assert not dm_mock.await_count
    assert not exit_mock.called


def test_append_readiness_failure_writes_structured_entry(tmp_path, failing_report):
    """Helper appends a distinct ``# Readiness Gate Failure`` block."""
    crash_file = discord_memory_bot._append_readiness_failure_to_crash_log(
        failing_report.failed_required(), tmp_path
    )
    content = crash_file.read_text(encoding="utf-8")

    assert "# Readiness Gate Failure" in content
    assert "**Failed Checks:** 2" in content
    assert "`ollama` (fail)" in content
    assert "`executor_db` (timeout)" in content
    assert "ConnectError: Connection refused" in content
    # Ensure it is not mistakenly marked as a crash report (crash_triage.py
    # splits on that exact header and would file a duplicate Jira story).
    assert "# Bot Crash Report" not in content


def test_append_readiness_failure_appends_not_overwrites(tmp_path, failing_report):
    """Second call must preserve the first entry rather than overwrite it."""
    discord_memory_bot._append_readiness_failure_to_crash_log(
        failing_report.failed_required(), tmp_path
    )
    discord_memory_bot._append_readiness_failure_to_crash_log(
        failing_report.failed_required(), tmp_path
    )
    crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
    content = crash_file.read_text(encoding="utf-8")
    assert content.count("# Readiness Gate Failure") == 2


def test_dm_owner_sends_to_matching_member(monkeypatch, failing_report):
    """DM helper finds the owner across guilds and awaits ``member.send``."""
    monkeypatch.setattr(
        discord_memory_bot.settings, "bot_owner", "testowner"
    )

    sent: list[str] = []

    class _Member:
        def __init__(self, name):
            self.name = name
            self.display_name = name

        async def send(self, body):
            sent.append(body)

    class _Guild:
        def __init__(self, members):
            self.members = members

    guild_wrong = _Guild([_Member("someone-else")])
    guild_owner = _Guild([_Member("other"), _Member("testowner")])
    fake_client = MagicMock()
    fake_client.guilds = [guild_wrong, guild_owner]

    delivered = _run(
        discord_memory_bot._dm_owner_about_readiness_failure(
            fake_client, failing_report.failed_required()
        )
    )

    assert delivered is True
    assert len(sent) == 1
    assert "Readiness gate failed" in sent[0]
    assert "`ollama`" in sent[0]
    assert "`executor_db`" in sent[0]


def test_dm_owner_returns_false_when_owner_not_found(monkeypatch, failing_report):
    """If no guild contains the owner, the helper returns False without raising."""
    monkeypatch.setattr(
        discord_memory_bot.settings, "bot_owner", "missing-owner"
    )
    fake_client = MagicMock()
    fake_client.guilds = []

    delivered = _run(
        discord_memory_bot._dm_owner_about_readiness_failure(
            fake_client, failing_report.failed_required()
        )
    )

    assert delivered is False
