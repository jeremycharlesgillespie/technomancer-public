"""Tests for agent/startup_checks.py — readiness gate orchestrator + checks."""

from __future__ import annotations

import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import startup_checks
from agent.startup_checks import (
    Check,
    CheckResult,
    ReadinessReport,
    SKIP_REQUIRED_ENV,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
    _check_daily_stats,
    _check_executor_db,
    _check_ollama,
    _check_vault,
    _run_with_retry,
    default_checks,
    make_daily_stats_check,
    run_readiness_checks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_retry_sleep(monkeypatch):
    """Skip the retry backoff sleep so tests stay fast."""
    monkeypatch.setattr(startup_checks.time, "sleep", lambda _s: None)


@pytest.fixture
def fake_vault(tmp_path, monkeypatch):
    """Point settings at a writable temp vault with the expected sub-dirs."""
    vault = tmp_path / "vault"
    context = vault / "LLM Memory" / "Context"
    context.mkdir(parents=True)
    monkeypatch.setattr(startup_checks.settings, "vault_path", vault)
    return vault


@pytest.fixture
def fake_executor_db(tmp_path, monkeypatch):
    """Point executor_runs_db at a temporary SQLite DB for isolation."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(startup_checks.executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(startup_checks.executor_runs_db, "DB_PATH", db_path)
    startup_checks.executor_runs_db._local.__dict__.pop("conn", None)
    yield db_path
    conn = getattr(startup_checks.executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        startup_checks.executor_runs_db._local.conn = None


@pytest.fixture
def _clear_skip_env(monkeypatch):
    """Ensure the escape-hatch env var is unset unless a test opts in."""
    monkeypatch.delenv(SKIP_REQUIRED_ENV, raising=False)


@pytest.fixture
def fake_daily_stats_db(tmp_path, monkeypatch):
    """Point daily_stats at a temporary SQLite DB for isolation."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr(startup_checks.daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(startup_checks.daily_stats, "DB_PATH", db_path)
    startup_checks.daily_stats._local.__dict__.pop("conn", None)
    yield db_path
    conn = getattr(startup_checks.daily_stats._local, "conn", None)
    if conn:
        conn.close()
        startup_checks.daily_stats._local.conn = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_check_defaults(self):
        c = Check(name="x", fn=lambda: None)
        assert c.required is True
        assert c.timeout == 5.0

    def test_check_result_defaults(self):
        r = CheckResult(name="x", status=STATUS_OK, required=True, duration_ms=5)
        assert r.error is None
        assert r.attempts == 0

    def test_readiness_report_failed_required_filters(self):
        report = ReadinessReport(
            ok=False,
            skipped=False,
            results=[
                CheckResult("a", STATUS_OK, True, 1),
                CheckResult("b", STATUS_FAIL, True, 1, error="boom"),
                CheckResult("c", STATUS_FAIL, False, 1, error="ignored"),
            ],
        )
        failed = report.failed_required()
        assert [r.name for r in failed] == ["b"]


# ---------------------------------------------------------------------------
# _check_ollama — pass / fail / timeout
# ---------------------------------------------------------------------------


class TestCheckOllama:
    def test_ok_with_latest_suffix(self, monkeypatch):
        monkeypatch.setattr(startup_checks.settings, "ollama_model", "fake-model")
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "fake-model:latest"}]}
        resp.raise_for_status = MagicMock()
        with patch.object(startup_checks.httpx, "get", return_value=resp):
            _check_ollama()

    def test_ok_with_tagged_variant(self, monkeypatch):
        """A tag like 'qwen3.5:9b' matches configured 'qwen3.5' base."""
        monkeypatch.setattr(startup_checks.settings, "ollama_model", "qwen3.5")
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "qwen3.5:9b"}]}
        resp.raise_for_status = MagicMock()
        with patch.object(startup_checks.httpx, "get", return_value=resp):
            _check_ollama()

    def test_fails_when_model_missing(self, monkeypatch):
        monkeypatch.setattr(startup_checks.settings, "ollama_model", "other-model")
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "qwen3.5:9b"}]}
        resp.raise_for_status = MagicMock()
        with patch.object(startup_checks.httpx, "get", return_value=resp):
            with pytest.raises(RuntimeError, match="not in tags"):
                _check_ollama()

    def test_fails_on_connect_error(self):
        with patch.object(
            startup_checks.httpx, "get", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(httpx.ConnectError):
                _check_ollama()

    def test_fails_on_http_error_status(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch.object(startup_checks.httpx, "get", return_value=resp):
            with pytest.raises(httpx.HTTPStatusError):
                _check_ollama()

    def test_timeout_path_via_retry_wrapper(self):
        """A slow check should be abandoned by _run_with_retry and reported
        as STATUS_TIMEOUT. Use ``Event.wait`` for the block so the global
        ``time.sleep`` patch can't accidentally short-circuit it."""
        evt = threading.Event()

        def slow_get(*_args, **_kwargs):
            evt.wait(5)

        try:
            with patch.object(startup_checks.httpx, "get", side_effect=slow_get):
                result = _run_with_retry(
                    Check(name="ollama", fn=_check_ollama, timeout=0.1),
                    attempts=1,
                )
        finally:
            evt.set()
        assert result.status == STATUS_TIMEOUT
        assert "timeout" in (result.error or "")


# ---------------------------------------------------------------------------
# _check_vault — pass / fail / timeout
# ---------------------------------------------------------------------------


class TestCheckVault:
    def test_ok(self, fake_vault):
        _check_vault()

    def test_fails_when_vault_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            startup_checks.settings, "vault_path", tmp_path / "does-not-exist"
        )
        with pytest.raises(RuntimeError, match="does not exist"):
            _check_vault()

    def test_fails_when_context_missing(self, tmp_path, monkeypatch):
        vault = tmp_path / "vault"
        (vault / "LLM Memory").mkdir(parents=True)
        # Context sibling folder is intentionally not created.
        monkeypatch.setattr(startup_checks.settings, "vault_path", vault)
        with pytest.raises(RuntimeError, match="Context folder missing"):
            _check_vault()

    def test_fails_when_context_not_writable(self, fake_vault, monkeypatch):
        def _raise(*_args, **_kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(
            startup_checks.tempfile, "NamedTemporaryFile", _raise
        )
        with pytest.raises(RuntimeError, match="not writable"):
            _check_vault()

    def test_timeout_path_via_retry_wrapper(self, fake_vault, monkeypatch):
        """A vault check that hangs in tempfile should time out."""
        evt = threading.Event()

        def slow_tempfile(*_args, **_kwargs):
            evt.wait(5)
            raise RuntimeError("should never reach")

        monkeypatch.setattr(
            startup_checks.tempfile, "NamedTemporaryFile", slow_tempfile
        )
        try:
            result = _run_with_retry(
                Check(name="vault", fn=_check_vault, timeout=0.1),
                attempts=1,
            )
        finally:
            evt.set()
        assert result.status == STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# _check_executor_db — pass / fail / timeout
# ---------------------------------------------------------------------------


class TestCheckExecutorDb:
    def test_ok_after_init(self, fake_executor_db):
        _check_executor_db()

    def test_fails_when_schema_missing_columns(self, tmp_path, monkeypatch):
        """If init_db produced a schema that lacks required columns, the
        check must raise. We simulate this by writing a stripped-down table
        and replacing init_db with a no-op so it doesn't re-add them."""
        db_path = tmp_path / "stripped.db"
        monkeypatch.setattr(startup_checks.executor_runs_db, "DB_DIR", tmp_path)
        monkeypatch.setattr(startup_checks.executor_runs_db, "DB_PATH", db_path)
        startup_checks.executor_runs_db._local.__dict__.pop("conn", None)

        # Create a minimal executor_runs table missing most columns.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE executor_runs (id INTEGER PRIMARY KEY, jira_key TEXT)"
        )
        conn.commit()
        conn.close()

        # Neutralise init_db so it doesn't "heal" the stripped schema.
        monkeypatch.setattr(
            startup_checks.executor_runs_db, "init_db", lambda: None
        )

        with pytest.raises(RuntimeError, match="missing columns"):
            _check_executor_db()

        cached = getattr(startup_checks.executor_runs_db._local, "conn", None)
        if cached:
            cached.close()
            startup_checks.executor_runs_db._local.conn = None

    def test_fails_when_init_raises(self, monkeypatch):
        def _raise():
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(
            startup_checks.executor_runs_db, "init_db", _raise
        )
        with pytest.raises(sqlite3.OperationalError):
            _check_executor_db()

    def test_timeout_path_via_retry_wrapper(self, monkeypatch):
        evt = threading.Event()

        def slow_init():
            evt.wait(5)

        monkeypatch.setattr(
            startup_checks.executor_runs_db, "init_db", slow_init
        )
        try:
            result = _run_with_retry(
                Check(name="executor_db", fn=_check_executor_db, timeout=0.1),
                attempts=1,
            )
        finally:
            evt.set()
        assert result.status == STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# _check_daily_stats — pass / fail / timeout
# ---------------------------------------------------------------------------


class TestCheckDailyStats:
    def test_ok_after_init(self, fake_daily_stats_db):
        _check_daily_stats()

    def test_fails_when_db_missing(self, tmp_path, monkeypatch):
        """If the daily_stats DB file doesn't exist, the check must fail."""
        monkeypatch.setattr(
            startup_checks.daily_stats, "DB_PATH", tmp_path / "does-not-exist.db"
        )
        # Neutralize init_db so it doesn't create the file.
        monkeypatch.setattr(
            startup_checks.daily_stats, "init_db", lambda: None
        )
        with pytest.raises(RuntimeError, match="Daily stats database missing"):
            _check_daily_stats()

    def test_fails_when_table_missing(self, fake_daily_stats_db, monkeypatch):
        """If the daily_stats table is missing, the check must fail."""
        db_path = fake_daily_stats_db
        conn = sqlite3.connect(str(db_path))
        # Create a minimal table without the daily_stats table
        conn.execute("CREATE TABLE other_table (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        # Neutralise init_db so it doesn't re-add the table.
        monkeypatch.setattr(
            startup_checks.daily_stats, "init_db", lambda: None
        )

        with pytest.raises(RuntimeError, match="Daily stats database corrupted"):
            _check_daily_stats()

    def test_timeout_path_via_retry_wrapper(self, monkeypatch):
        evt = threading.Event()

        def slow_init():
            evt.wait(5)

        monkeypatch.setattr(
            startup_checks.daily_stats, "init_db", slow_init
        )
        try:
            result = _run_with_retry(
                Check(name="daily_stats", fn=_check_daily_stats, timeout=0.1),
                attempts=1,
            )
        finally:
            evt.set()
        assert result.status == STATUS_TIMEOUT

    def test_gate_fails_on_missing_daily_stats(self, monkeypatch, _no_retry_sleep, _clear_skip_env):
        """The gate should fail when daily stats check fails."""
        def fail_check():
            raise RuntimeError("daily_stats table missing")

        checks = [
            Check(name="ok", fn=lambda: None, timeout=1.0),
            Check(name="daily_stats", fn=fail_check, timeout=1.0, required=True),
        ]
        report = run_readiness_checks(checks=checks, attempts=1)
        assert report.ok is False
        assert [r.name for r in report.failed_required()] == ["daily_stats"]


# ---------------------------------------------------------------------------
# make_daily_stats_check — helper function
# ---------------------------------------------------------------------------


class TestMakeDailyStatsCheck:
    def test_creates_required_check(self):
        check = make_daily_stats_check()
        assert check.name == "daily_stats"
        assert check.required is True
        assert check.timeout == 5.0


# ---------------------------------------------------------------------------
# _run_with_retry behaviour
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    def test_ok_first_attempt_records_attempt_1(self):
        result = _run_with_retry(
            Check(name="x", fn=lambda: None, timeout=1.0), attempts=2
        )
        assert result.status == STATUS_OK
        assert result.attempts == 1

    def test_ok_second_attempt_after_first_failure(self, _no_retry_sleep):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")

        result = _run_with_retry(
            Check(name="flaky", fn=flaky, timeout=1.0), attempts=2, backoff=0.01
        )
        assert result.status == STATUS_OK
        assert result.attempts == 2
        assert calls["n"] == 2

    def test_fail_after_all_attempts(self, _no_retry_sleep):
        def always_fail():
            raise ValueError("nope")

        result = _run_with_retry(
            Check(name="x", fn=always_fail, timeout=1.0),
            attempts=2,
            backoff=0.01,
        )
        assert result.status == STATUS_FAIL
        assert result.attempts == 2
        assert "ValueError" in (result.error or "")
        assert "nope" in (result.error or "")

    def test_two_attempts_is_the_retry_cap(self, _no_retry_sleep):
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise RuntimeError("boom")

        _run_with_retry(
            Check(name="x", fn=always_fail, timeout=1.0),
            attempts=2,
            backoff=0.01,
        )
        assert calls["n"] == 2

    def test_timeout_records_status_and_attempt(self, _no_retry_sleep):
        evt = threading.Event()

        def slow():
            evt.wait(2)

        try:
            result = _run_with_retry(
                Check(name="slow", fn=slow, timeout=0.05),
                attempts=2,
                backoff=0.01,
            )
        finally:
            evt.set()
        assert result.status == STATUS_TIMEOUT
        assert result.attempts == 2
        assert "timeout" in (result.error or "")


# ---------------------------------------------------------------------------
# run_readiness_checks — orchestrator + escape hatch
# ---------------------------------------------------------------------------


class TestRunReadinessChecks:
    def test_all_checks_pass(self, _no_retry_sleep, _clear_skip_env):
        checks = [
            Check(name="a", fn=lambda: None, timeout=1.0),
            Check(name="b", fn=lambda: None, timeout=1.0),
        ]
        report = run_readiness_checks(checks=checks, attempts=1)
        assert report.ok is True
        assert report.skipped is False
        assert len(report.results) == 2
        assert all(r.status == STATUS_OK for r in report.results)

    def test_required_failure_sets_ok_false(self, _no_retry_sleep, _clear_skip_env):
        def bad():
            raise RuntimeError("bad")

        checks = [
            Check(name="ok", fn=lambda: None, timeout=1.0),
            Check(name="bad", fn=bad, timeout=1.0, required=True),
        ]
        report = run_readiness_checks(checks=checks, attempts=1)
        assert report.ok is False
        assert [r.name for r in report.failed_required()] == ["bad"]

    def test_non_required_failure_does_not_block(self, _no_retry_sleep, _clear_skip_env):
        def bad():
            raise RuntimeError("bad")

        checks = [
            Check(name="ok", fn=lambda: None, timeout=1.0),
            Check(name="optional_bad", fn=bad, timeout=1.0, required=False),
        ]
        report = run_readiness_checks(checks=checks, attempts=1)
        assert report.ok is True
        assert report.failed_required() == []

    def test_escape_hatch_marks_all_skipped(self, monkeypatch, _no_retry_sleep):
        monkeypatch.setenv(SKIP_REQUIRED_ENV, "1")

        # Sentinel — the check fn must NOT run when the gate is skipped.
        ran = {"n": 0}

        def should_not_run():
            ran["n"] += 1

        checks = [
            Check(name="a", fn=should_not_run, timeout=1.0),
            Check(name="b", fn=should_not_run, timeout=1.0),
        ]
        report = run_readiness_checks(checks=checks)
        assert report.ok is True
        assert report.skipped is True
        assert ran["n"] == 0
        assert all(r.status == STATUS_SKIPPED for r in report.results)

    def test_escape_hatch_requires_exactly_1(self, monkeypatch, _no_retry_sleep, _clear_skip_env):
        """Any other truthy value does NOT activate the escape hatch."""
        monkeypatch.setenv(SKIP_REQUIRED_ENV, "true")

        def bad():
            raise RuntimeError("bad")

        checks = [Check(name="bad", fn=bad, timeout=1.0, required=True)]
        report = run_readiness_checks(checks=checks, attempts=1)
        assert report.skipped is False
        assert report.ok is False

    def test_default_checks_returns_four_required(self):
        """default_checks now includes the daily stats check."""
        checks = default_checks()
        assert len(checks) == 4
        names = [c.name for c in checks]
        assert names == ["ollama", "vault", "executor_db", "daily_stats"]
        assert all(c.required for c in checks)

    def test_orchestrator_uses_default_checks_when_none_supplied(
        self, monkeypatch, _no_retry_sleep, _clear_skip_env
    ):
        called = []

        def fake_default():
            called.append(True)
            return [Check(name="fake", fn=lambda: None, timeout=1.0)]

        monkeypatch.setattr(startup_checks, "default_checks", fake_default)
        report = run_readiness_checks(attempts=1)
        assert called == [True]
        assert [r.name for r in report.results] == ["fake"]
