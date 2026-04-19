"""Aggregated health checks for external monitors.

Exposes :func:`run_checks` — a single callable that collects the current
state of five subsystems (bot, Ollama, Jira, executor, disk) and returns
a dict shaped for ``GET /api/health``:

    {
        "status": "healthy" | "degraded" | "unhealthy",
        "checks": {
            "bot":      {"ok": bool, "latency_ms": int, "detail": str, ...},
            "ollama":   {...},
            "jira":     {...},
            "executor": {...},
            "disk":     {...},
        },
        "timestamp": "<ISO-8601>",
    }

Overall ``status`` rules:

* ``unhealthy`` — any *required* check fails. Only ``bot`` is required.
  The endpoint handler translates this to HTTP 503.
* ``degraded`` — all required checks pass but at least one optional
  check (``ollama``, ``jira``, ``executor``, ``disk``) failed.
* ``healthy``  — every check reports ``ok: True``.

The result is cached process-wide for :data:`CACHE_TTL_SECONDS` so a
burst of pings from Tailscale dashboards or uptime pollers doesn't stack
up blocking subprocess/HTTP calls behind a lock.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent import executor_runs_db
from agent.config import settings

from . import jira_sync_dlq
from .executor import EXECUTION_LOGS_DIR
from .jira_sync import is_jira_configured

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30
OLLAMA_TIMEOUT_SECONDS = 2
EXECUTOR_STALE_HOURS = 2
DISK_FREE_WARN_BYTES = 5 * 1024 * 1024 * 1024
JIRA_DLQ_WINDOW_HOURS = 24

REQUIRED_CHECKS = frozenset({"bot"})
_OPTIONAL_CHECKS = ("ollama", "jira", "executor", "disk")

_SERVICE_DIR = Path(__file__).resolve().parent.parent
PID_FILE: Path = _SERVICE_DIR / "bot.pid"
STATE_FILE: Path = _SERVICE_DIR / "service_state.json"

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"value": None, "expires_at": 0.0}


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _finalize(result: dict[str, Any], start_ms: float) -> dict[str, Any]:
    """Stamp latency and guarantee ``ok``/``detail`` keys exist."""
    result["latency_ms"] = max(0, int(_now_ms() - start_ms))
    result.setdefault("ok", False)
    result.setdefault("detail", "")
    return result


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform liveness probe. ``os.kill(pid, 0)`` is unreliable on
    Windows, so use tasklist there and fall back to a signal-0 kill on
    POSIX. A ``PermissionError`` on POSIX means the pid exists but we
    don't own it — the process is alive from the caller's perspective.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=3,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def check_bot() -> dict[str, Any]:
    """Verify the Discord bot process is alive via bot.pid + service_state.json.

    Mirrors the liveness check in ``bot_service.is_bot_running`` so an
    external monitor sees the same result as ``python bot_service.py status``.
    """
    start = _now_ms()
    result: dict[str, Any] = {"ok": False, "detail": "unknown"}
    try:
        if not PID_FILE.exists():
            result.update(ok=False, detail="no pid file — bot not started")
        else:
            try:
                pid = int(PID_FILE.read_text().strip())
            except (ValueError, OSError) as e:
                result.update(ok=False, detail=f"bad pid file: {e}")
                return _finalize(result, start)
            result["pid"] = pid
            if _is_pid_alive(pid):
                result.update(ok=True, detail=f"running (pid={pid})")
            else:
                result.update(ok=False, detail=f"pid {pid} not alive")

        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                result["restarts"] = int(state.get("total_restarts", 0) or 0)
                result["consecutive_failures"] = int(
                    state.get("consecutive_failures", 0) or 0
                )
                if state.get("last_error"):
                    result["last_error"] = str(state["last_error"])[:200]
            except (ValueError, OSError) as e:
                log.debug("could not parse service_state.json: %s", e)
    except Exception as e:
        result.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return _finalize(result, start)


def check_ollama() -> dict[str, Any]:
    """GET ``<ollama_host>/api/tags`` with a short timeout."""
    start = _now_ms()
    result: dict[str, Any] = {"ok": False, "detail": "unknown"}
    url = f"{settings.ollama_host}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "?") for m in data.get("models", []) if m]
            result.update(
                ok=True,
                detail=f"{len(models)} models loaded",
                model_count=len(models),
                models=models,
            )
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        result.update(ok=False, detail=f"unreachable: {reason}")
    except Exception as e:
        result.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return _finalize(result, start)


def check_jira() -> dict[str, Any]:
    """DLQ depth + newest failure timestamp as a proxy for sync health.

    If Jira is not configured at all the system works fine standalone,
    so report ``ok=True`` with ``configured=False``.
    """
    start = _now_ms()
    result: dict[str, Any] = {"ok": False, "detail": "unknown"}
    try:
        if not is_jira_configured():
            result.update(ok=True, detail="not configured", configured=False)
            return _finalize(result, start)

        result["configured"] = True
        jira_sync_dlq.init_db()
        conn = jira_sync_dlq._get_conn()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=JIRA_DLQ_WINDOW_HOURS)
        ).isoformat(timespec="seconds")
        row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(last_failed_at) AS last_failed "
            "FROM jira_sync_dlq WHERE last_failed_at >= ?",
            (cutoff,),
        ).fetchone()
        dlq_depth = int(row["n"]) if row else 0
        last_failed_at = row["last_failed"] if row else None

        result["dlq_depth"] = dlq_depth
        result["window_hours"] = JIRA_DLQ_WINDOW_HOURS
        if last_failed_at:
            result["last_failed_at"] = last_failed_at

        detail = f"{dlq_depth} DLQ entries in last {JIRA_DLQ_WINDOW_HOURS}h"
        result.update(ok=(dlq_depth == 0), detail=detail)
    except Exception as e:
        result.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return _finalize(result, start)


def _parse_started(raw: str) -> float | None:
    """Return seconds-of-age for an ISO timestamp, or ``None`` if unparseable."""
    try:
        dt = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return (datetime.now() - dt).total_seconds()
    return (datetime.now(timezone.utc) - dt).total_seconds()


def check_executor() -> dict[str, Any]:
    """Count in-flight AIM workers from per-project state files.

    The canonical "is a story executing?" source is each project's
    ``aim/.../.aim_state.json`` — it's what the worker itself writes in
    real time and what the /live page renders. The old SQLite query for
    ``status='running'`` rows was structurally stuck at 0 because
    ``idea_board/executor.py`` never writes a 'running' row (only the
    unused claude_code_runner path does).

    A worker whose ``started_at`` is older than :data:`EXECUTOR_STALE_HOURS`
    hours is treated as stuck and flips the check to ``ok=False``.
    """
    start = _now_ms()
    result: dict[str, Any] = {"ok": False, "detail": "unknown"}
    try:
        from pathlib import Path as _Path
        aim_root = _Path(__file__).resolve().parent.parent / "aim"
        per_project = executor_runs_db.get_current_execution_per_project(
            aim_root=aim_root,
            primary_label=settings.jira_project_key or "primary",
        )
        executing = [p for p in per_project if p.get("is_executing")]
        running_count = len(executing)

        oldest_started = None
        if executing:
            starts = [p.get("started_at") for p in executing if p.get("started_at")]
            if starts:
                oldest_started = min(starts)

        result["running_count"] = running_count
        result["oldest_started_at"] = oldest_started

        if running_count == 0:
            result.update(ok=True, detail="0 running")
            return _finalize(result, start)

        age = _parse_started(oldest_started) if oldest_started else None
        if age is not None:
            result["oldest_age_seconds"] = int(age)
        if age is not None and age > EXECUTOR_STALE_HOURS * 3600:
            result.update(
                ok=False,
                detail=(
                    f"{running_count} running; oldest started {oldest_started} "
                    f"(> {EXECUTOR_STALE_HOURS}h)"
                ),
            )
        else:
            result.update(ok=True, detail=f"{running_count} running")
    except Exception as e:
        result.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return _finalize(result, start)


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive byte count. Never raises."""
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def check_disk() -> dict[str, Any]:
    """Free bytes on the vault volume + total size of execution_logs/."""
    start = _now_ms()
    result: dict[str, Any] = {"ok": False, "detail": "unknown"}
    try:
        vault = Path(settings.vault_path)
        if vault.exists():
            probe = vault
        elif vault.parent.exists():
            probe = vault.parent
        else:
            probe = Path.cwd()
        usage = shutil.disk_usage(str(probe))
        free_bytes = int(usage.free)
        total_bytes = int(usage.total)
        exec_logs_bytes = _dir_size_bytes(EXECUTION_LOGS_DIR)

        free_gb = free_bytes / (1024 ** 3)
        result.update(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            execution_logs_bytes=exec_logs_bytes,
            vault_path=str(vault),
        )
        if free_bytes >= DISK_FREE_WARN_BYTES:
            result.update(ok=True, detail=f"{free_gb:.1f} GB free")
        else:
            result.update(ok=False, detail=f"low disk — only {free_gb:.2f} GB free")
    except Exception as e:
        result.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return _finalize(result, start)


_CHECK_FUNCS: dict[str, Any] = {
    "bot": check_bot,
    "ollama": check_ollama,
    "jira": check_jira,
    "executor": check_executor,
    "disk": check_disk,
}


def run_checks(use_cache: bool = True) -> dict[str, Any]:
    """Run every check and return the aggregated payload.

    Results are memoized for :data:`CACHE_TTL_SECONDS` so a noisy poller
    doesn't pay the full subprocess/HTTP cost on every hit.
    """
    if use_cache:
        now = time.monotonic()
        with _cache_lock:
            cached = _cache.get("value")
            expires_at = _cache.get("expires_at", 0.0)
            if cached is not None and now < expires_at:
                return cached

    checks: dict[str, dict[str, Any]] = {}
    for name, fn in _CHECK_FUNCS.items():
        try:
            checks[name] = fn()
        except Exception as e:
            checks[name] = {
                "ok": False,
                "latency_ms": 0,
                "detail": f"check raised: {type(e).__name__}: {e}",
            }

    required_failed = any(
        not checks[name].get("ok", False) for name in REQUIRED_CHECKS
    )
    optional_failed = any(
        not checks[name].get("ok", False) for name in _OPTIONAL_CHECKS
    )

    if required_failed:
        overall = "unhealthy"
    elif optional_failed:
        overall = "degraded"
    else:
        overall = "healthy"

    payload: dict[str, Any] = {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }

    if use_cache:
        with _cache_lock:
            _cache["value"] = payload
            _cache["expires_at"] = time.monotonic() + CACHE_TTL_SECONDS
    return payload


def clear_cache() -> None:
    """Drop the memoized payload. Intended for tests."""
    with _cache_lock:
        _cache["value"] = None
        _cache["expires_at"] = 0.0
