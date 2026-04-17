"""Unit tests for ``idea_board/health.py`` — per-check behaviour and the
aggregated ``run_checks()`` payload.

Each check function is tested in isolation so a regression in one path
pinpoints the broken check rather than the whole endpoint. ``run_checks``
is tested with the individual check functions monkeypatched so we don't
reach the real filesystem / network / Ollama server.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import executor_runs_db
from idea_board import health
from idea_board import jira_sync_dlq


@pytest.fixture(autouse=True)
def _reset_health_cache():
    """Clear the process-wide cache so each test starts fresh."""
    health.clear_cache()
    yield
    health.clear_cache()


@pytest.fixture(autouse=True)
def _isolate_executor_db(tmp_path, monkeypatch):
    """Route executor_runs_db at a temp SQLite so tests don't touch prod."""
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


@pytest.fixture(autouse=True)
def _isolate_jira_dlq_db(tmp_path, monkeypatch):
    """Route jira_sync_dlq at a temp SQLite — no real dlq writes."""
    db_path = tmp_path / "jira_sync_dlq.db"
    monkeypatch.setattr(jira_sync_dlq, "DB_PATH", db_path)
    jira_sync_dlq._local.__dict__.pop("conn", None)
    yield
    conn = getattr(jira_sync_dlq._local, "conn", None)
    if conn:
        conn.close()
        jira_sync_dlq._local.conn = None


@pytest.fixture
def tmp_bot_files(tmp_path, monkeypatch):
    """Point PID_FILE and STATE_FILE at tmp_path. Returns (pid_file, state_file)."""
    pid_file = tmp_path / "bot.pid"
    state_file = tmp_path / "service_state.json"
    monkeypatch.setattr(health, "PID_FILE", pid_file)
    monkeypatch.setattr(health, "STATE_FILE", state_file)
    return pid_file, state_file


# ---------------------------------------------------------------------------
# check_bot
# ---------------------------------------------------------------------------


class TestCheckBot:
    def test_no_pid_file_is_not_ok(self, tmp_bot_files):
        result = health.check_bot()
        assert result["ok"] is False
        assert "no pid file" in result["detail"]
        assert "latency_ms" in result

    def test_dead_pid_is_not_ok(self, tmp_bot_files):
        pid_file, _ = tmp_bot_files
        pid_file.write_text("99999")
        with patch.object(health, "_is_pid_alive", return_value=False):
            result = health.check_bot()
        assert result["ok"] is False
        assert "99999" in result["detail"]
        assert result["pid"] == 99999

    def test_alive_pid_is_ok(self, tmp_bot_files):
        pid_file, _ = tmp_bot_files
        pid_file.write_text("12345")
        with patch.object(health, "_is_pid_alive", return_value=True):
            result = health.check_bot()
        assert result["ok"] is True
        assert result["pid"] == 12345
        assert "running" in result["detail"]

    def test_state_file_fields_merged(self, tmp_bot_files):
        pid_file, state_file = tmp_bot_files
        pid_file.write_text("12345")
        state_file.write_text(
            json.dumps(
                {
                    "total_restarts": 7,
                    "consecutive_failures": 0,
                    "last_error": "boom",
                }
            )
        )
        with patch.object(health, "_is_pid_alive", return_value=True):
            result = health.check_bot()
        assert result["restarts"] == 7
        assert result["consecutive_failures"] == 0
        assert result["last_error"] == "boom"

    def test_bad_pid_file_contents(self, tmp_bot_files):
        pid_file, _ = tmp_bot_files
        pid_file.write_text("not-a-number")
        result = health.check_bot()
        assert result["ok"] is False
        assert "bad pid file" in result["detail"]


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestCheckOllama:
    def test_happy_path_lists_models(self):
        fake = _FakeHTTPResponse({"models": [{"name": "qwen3.5:27b"}, {"name": "llava"}]})
        with patch.object(health.urllib.request, "urlopen", return_value=fake):
            result = health.check_ollama()
        assert result["ok"] is True
        assert result["model_count"] == 2
        assert "qwen3.5:27b" in result["models"]
        assert "2 models" in result["detail"]

    def test_connection_refused_is_not_ok(self):
        err = urllib.error.URLError("Connection refused")
        with patch.object(health.urllib.request, "urlopen", side_effect=err):
            result = health.check_ollama()
        assert result["ok"] is False
        assert "unreachable" in result["detail"]

    def test_unexpected_error_does_not_raise(self):
        with patch.object(
            health.urllib.request, "urlopen", side_effect=RuntimeError("nope")
        ):
            result = health.check_ollama()
        assert result["ok"] is False
        assert "RuntimeError" in result["detail"]


# ---------------------------------------------------------------------------
# check_jira
# ---------------------------------------------------------------------------


class TestCheckJira:
    def test_not_configured_reports_ok(self):
        with patch.object(health, "is_jira_configured", return_value=False):
            result = health.check_jira()
        assert result["ok"] is True
        assert result["configured"] is False
        assert "not configured" in result["detail"]

    def test_empty_dlq_is_ok(self):
        with patch.object(health, "is_jira_configured", return_value=True):
            jira_sync_dlq.init_db()
            result = health.check_jira()
        assert result["ok"] is True
        assert result["dlq_depth"] == 0

    def test_non_empty_dlq_flips_to_not_ok(self):
        jira_sync_dlq.init_db()
        jira_sync_dlq.add_dlq_entry(
            idea_id="TK-1", payload={"a": 1}, error="500", attempts=3
        )
        with patch.object(health, "is_jira_configured", return_value=True):
            result = health.check_jira()
        assert result["ok"] is False
        assert result["dlq_depth"] == 1
        assert "DLQ" in result["detail"]
        assert "last_failed_at" in result

    def test_sql_error_reports_not_ok(self):
        with patch.object(health, "is_jira_configured", return_value=True), \
             patch.object(jira_sync_dlq, "init_db", side_effect=sqlite3.OperationalError("locked")):
            result = health.check_jira()
        assert result["ok"] is False
        assert "OperationalError" in result["detail"]


# ---------------------------------------------------------------------------
# check_executor
# ---------------------------------------------------------------------------


class TestCheckExecutor:
    def test_no_running_runs_is_ok(self):
        result = health.check_executor()
        assert result["ok"] is True
        assert result["running_count"] == 0

    def test_fresh_running_run_is_ok(self):
        executor_runs_db.record_run(
            run_id="abc",
            jira_key="TK-1",
            branch="br",
            started_at=datetime.now().isoformat(),
            status="running",
        )
        result = health.check_executor()
        assert result["ok"] is True
        assert result["running_count"] == 1
        assert "oldest_age_seconds" in result

    def test_stale_running_run_is_not_ok(self):
        stale = (datetime.now() - timedelta(hours=3)).isoformat()
        executor_runs_db.record_run(
            run_id="old",
            jira_key="TK-2",
            branch="br",
            started_at=stale,
            status="running",
        )
        result = health.check_executor()
        assert result["ok"] is False
        assert result["running_count"] == 1
        assert result["oldest_age_seconds"] >= 3 * 3600 - 5


# ---------------------------------------------------------------------------
# check_disk
# ---------------------------------------------------------------------------


class TestCheckDisk:
    def test_plenty_of_free_space_is_ok(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "empty_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(health, "EXECUTION_LOGS_DIR", logs_dir)
        fake_usage = MagicMock(
            free=50 * 1024 ** 3, total=500 * 1024 ** 3, used=450 * 1024 ** 3
        )
        with patch.object(health.shutil, "disk_usage", return_value=fake_usage):
            result = health.check_disk()
        assert result["ok"] is True
        assert result["free_bytes"] == 50 * 1024 ** 3
        assert result["total_bytes"] == 500 * 1024 ** 3
        assert result["execution_logs_bytes"] == 0

    def test_low_free_space_is_not_ok(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "empty_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(health, "EXECUTION_LOGS_DIR", logs_dir)
        fake_usage = MagicMock(
            free=1 * 1024 ** 3, total=100 * 1024 ** 3, used=99 * 1024 ** 3
        )
        with patch.object(health.shutil, "disk_usage", return_value=fake_usage):
            result = health.check_disk()
        assert result["ok"] is False
        assert "low disk" in result["detail"]

    def test_execution_logs_size_counted(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "execution_logs"
        log_dir.mkdir()
        (log_dir / "a.log").write_bytes(b"x" * 100)
        (log_dir / "b.log").write_bytes(b"y" * 50)
        monkeypatch.setattr(health, "EXECUTION_LOGS_DIR", log_dir)
        fake_usage = MagicMock(free=50 * 1024 ** 3, total=100 * 1024 ** 3, used=0)
        with patch.object(health.shutil, "disk_usage", return_value=fake_usage):
            result = health.check_disk()
        assert result["execution_logs_bytes"] == 150


# ---------------------------------------------------------------------------
# run_checks — aggregation & caching
# ---------------------------------------------------------------------------


def _stub_check(ok: bool, detail: str = "stub", **extra):
    def inner():
        result = {"ok": ok, "detail": detail, "latency_ms": 1}
        result.update(extra)
        return result

    return inner


class TestRunChecks:
    def test_all_ok_is_healthy(self):
        funcs = {
            "bot": _stub_check(True),
            "ollama": _stub_check(True),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            result = health.run_checks(use_cache=False)
        assert result["status"] == "healthy"
        assert set(result["checks"].keys()) == {"bot", "ollama", "jira", "executor", "disk"}
        assert "timestamp" in result

    def test_optional_fail_is_degraded(self):
        funcs = {
            "bot": _stub_check(True),
            "ollama": _stub_check(False, detail="unreachable"),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            result = health.run_checks(use_cache=False)
        assert result["status"] == "degraded"

    def test_required_fail_is_unhealthy(self):
        funcs = {
            "bot": _stub_check(False),
            "ollama": _stub_check(True),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            result = health.run_checks(use_cache=False)
        assert result["status"] == "unhealthy"

    def test_cache_hit_skips_checks(self):
        call_count = [0]

        def counting():
            call_count[0] += 1
            return {"ok": True, "detail": "", "latency_ms": 0}

        funcs = {name: counting for name in ("bot", "ollama", "jira", "executor", "disk")}
        with patch.object(health, "_CHECK_FUNCS", funcs):
            health.run_checks(use_cache=True)
            first = call_count[0]
            health.run_checks(use_cache=True)
            second = call_count[0]
        assert first == 5
        assert second == 5

    def test_use_cache_false_bypasses_cache(self):
        funcs = {
            "bot": _stub_check(True),
            "ollama": _stub_check(True),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            # Seed the cache with stale data.
            health.run_checks(use_cache=True)
            # New checks return the same values; use_cache=False re-runs them anyway.
            fresh = health.run_checks(use_cache=False)
        assert fresh["status"] == "healthy"

    def test_check_exception_does_not_crash_aggregator(self):
        def exploder():
            raise RuntimeError("kaboom")

        funcs = {
            "bot": _stub_check(True),
            "ollama": exploder,
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            result = health.run_checks(use_cache=False)
        assert result["status"] == "degraded"
        assert result["checks"]["ollama"]["ok"] is False
        assert "RuntimeError" in result["checks"]["ollama"]["detail"]

    def test_every_check_has_ok_and_latency(self):
        """Acceptance contract: each check has {ok, latency_ms, detail}."""
        with patch.object(health, "is_jira_configured", return_value=False), \
             patch.object(
                 health.urllib.request,
                 "urlopen",
                 side_effect=urllib.error.URLError("refused"),
             ), \
             patch.object(health, "_is_pid_alive", return_value=False), \
             patch.object(
                 health.shutil,
                 "disk_usage",
                 return_value=MagicMock(free=50 * 1024 ** 3, total=100 * 1024 ** 3, used=0),
             ):
            result = health.run_checks(use_cache=False)
        for name, check in result["checks"].items():
            assert "ok" in check, f"{name} missing ok"
            assert "latency_ms" in check, f"{name} missing latency_ms"
            assert "detail" in check, f"{name} missing detail"
            assert isinstance(check["latency_ms"], int)
