"""
Startup Readiness Gate — validate external dependencies before the bot marks itself ready.

The readiness gate runs a small set of required checks during bot startup so
failures surface immediately instead of the bot coming up and dying on the
first user message. Four checks are currently enforced:

* ``_check_ollama`` — Ollama ``/api/tags`` reachable and ``settings.ollama_model`` present.
* ``_check_vault`` — vault root exists and ``LLM Memory/Context/`` is writable.
* ``_check_executor_db`` — executor_runs SQLite opens with migrations current.
* ``_check_daily_stats`` — daily_stats SQLite DB exists and is accessible.

Each check runs with a per-check timeout and up to two attempts, with a short
backoff between retries. An escape hatch — ``BOT_SKIP_REQUIRED_CHECKS=1`` in
the environment — lets the operator force the bot past the gate in recovery
scenarios. Use it only when you know why a required dependency is unavailable.

This module is intentionally decoupled from the Discord bot entry point. It
exposes :func:`run_readiness_checks` as a pure function that returns a
:class:`ReadinessReport`, and leaves wiring decisions to the caller.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from . import daily_stats, executor_runs_db
from .config import settings

log = logging.getLogger(__name__)

SKIP_REQUIRED_ENV = "BOT_SKIP_REQUIRED_CHECKS"

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"
STATUS_SKIPPED = "skipped"

# Columns that executor_runs must expose for the DB to count as "migrated".
# Kept in sync with executor_runs_db.init_db() — new columns added there must
# be mirrored here (or the check will false-fail).
_EXECUTOR_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "id",
    "jira_key",
    "branch",
    "started_at",
    "ended_at",
    "duration_ms",
    "cost_usd",
    "status",
    "exit_code",
    "tests_passed",
    "deployed",
    "run_id",
    "artifacts_path",
})


@dataclass
class Check:
    """Declarative description of a readiness check.

    ``fn`` is a zero-arg callable that raises on failure. A clean return is
    treated as success. ``required=False`` reserves space for future optional
    checks — the orchestrator never fails the gate on non-required checks.
    """

    name: str
    fn: Callable[[], None]
    required: bool = True
    timeout: float = 5.0


@dataclass
class CheckResult:
    """Outcome of a single check run.

    ``status`` is one of ``STATUS_OK``, ``STATUS_FAIL``, ``STATUS_TIMEOUT``, or
    ``STATUS_SKIPPED`` (used by the escape hatch).
    """

    name: str
    status: str
    required: bool
    duration_ms: int
    error: str | None = None
    attempts: int = 0


@dataclass
class ReadinessReport:
    """Aggregated outcome of a readiness-gate run.

    ``ok`` is ``True`` iff every required check passed (or the gate was
    skipped via the escape hatch). ``skipped`` distinguishes the
    escape-hatch path from a genuinely healthy gate.
    """

    ok: bool
    skipped: bool = False
    results: list[CheckResult] = field(default_factory=list)

    def failed_required(self) -> list[CheckResult]:
        """Return the required checks that did not reach ``STATUS_OK``."""
        return [r for r in self.results if r.required and r.status != STATUS_OK]


# ---------------------------------------------------------------------------
# Required checks
# ---------------------------------------------------------------------------


def _check_ollama() -> None:
    """Verify Ollama ``/api/tags`` is reachable and the chat model is present.

    Raises:
        httpx.HTTPError: server unreachable or returned an error status.
        RuntimeError: the configured ``ollama_model`` is not in the tag list.
    """
    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    timeout = settings.ollama_health_check_timeout
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json() or {}
    tags = {m.get("name", "") for m in payload.get("models", [])}
    wanted = settings.ollama_model
    candidates = {wanted, f"{wanted}:latest"}
    # Also accept any tag that starts with "<wanted>:" so a pinned
    # "qwen3.5:9b" matches "qwen3.5:9b" even though the ":latest" alias
    # isn't present.
    if not (tags & candidates) and not any(t.startswith(f"{wanted}:") for t in tags):
        raise RuntimeError(
            f"Ollama model {wanted!r} not in tags. Available: {sorted(tags)}"
        )


def _check_vault() -> None:
    """Verify the vault root exists and ``Context/`` is writable.

    The writability probe creates a NamedTemporaryFile under ``Context/`` and
    lets it clean itself up on exit — avoids leaving ``.readiness`` artifacts
    around if the process is killed mid-check.
    """
    vault = settings.vault_path
    if not vault.exists():
        raise RuntimeError(f"Vault path does not exist: {vault}")

    context = settings.context_path
    if not context.exists():
        raise RuntimeError(f"Context folder missing: {context}")

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".readiness-", suffix=".tmp", dir=str(context), delete=True
        ) as tmp:
            tmp.write(b"ok")
            tmp.flush()
    except OSError as exc:
        raise RuntimeError(
            f"Context folder not writable: {context} ({exc})"
        ) from exc


def _check_executor_db() -> None:
    """Verify the executor_runs SQLite DB opens and its schema is current.

    Runs ``executor_runs_db.init_db()`` (idempotent — safe to call from the
    gate) and then inspects ``PRAGMA table_info`` to confirm the expected
    columns are present. A missing column means migrations didn't run.
    """
    executor_runs_db.init_db()
    conn = executor_runs_db._get_conn()
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
    }
    missing = _EXECUTOR_REQUIRED_COLUMNS - cols
    if missing:
        raise RuntimeError(
            f"executor_runs schema missing columns: {sorted(missing)}"
        )


def _check_daily_stats() -> None:
    """Verify the daily_stats SQLite DB exists and is accessible.

    Runs ``daily_stats.init_db()`` (idempotent — safe to call from the
    gate) and then validates that the database file exists and the
    daily_stats table is present. A missing database or table means
    the daily stats system is not initialized.
    """
    daily_stats.init_db()
    daily_stats.validate_daily_stats_db()


def make_daily_stats_check(
    required: bool = True,
    timeout: float = 5.0,
) -> Check:
    """Create a Check object for the daily stats readiness check.

    Args:
        required: If True, the gate fails if this check fails. Default is True.
        timeout: Per-check timeout in seconds. Default is 5.0.

    Returns:
        A Check object configured with the daily stats check function.

    Example:
        >>> check = make_daily_stats_check(required=False, timeout=10.0)
        >>> assert check.required is False
        >>> assert check.timeout == 10.0
    """
    return Check(
        name="daily_stats",
        fn=_check_daily_stats,
        required=required,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _run_with_retry(
    check: Check,
    attempts: int = 2,
    backoff: float = 1.0,
) -> CheckResult:
    """Run ``check`` with per-check timeout and up to ``attempts`` tries.

    Each attempt runs in its own worker thread so a hung check can be
    abandoned on timeout without blocking the whole gate. Executors are
    shut down with ``wait=False`` — a thread stuck in a C extension can't
    be cancelled cleanly, and the caller would rather move on than block
    startup waiting for it.
    """
    last_error: str | None = None
    last_status = STATUS_FAIL
    start = time.monotonic()
    used = 0

    for attempt in range(1, attempts + 1):
        used = attempt
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(check.fn)
            future.result(timeout=check.timeout)
            duration_ms = int((time.monotonic() - start) * 1000)
            return CheckResult(
                name=check.name,
                status=STATUS_OK,
                required=check.required,
                duration_ms=duration_ms,
                attempts=used,
            )
        except concurrent.futures.TimeoutError:
            last_error = f"timeout after {check.timeout:.1f}s"
            last_status = STATUS_TIMEOUT
        except Exception as exc:  # noqa: BLE001 — we want every failure captured
            last_error = f"{type(exc).__name__}: {exc}"
            last_status = STATUS_FAIL
        finally:
            pool.shutdown(wait=False)

        if attempt < attempts:
            time.sleep(backoff)

    duration_ms = int((time.monotonic() - start) * 1000)
    return CheckResult(
        name=check.name,
        status=last_status,
        required=check.required,
        duration_ms=duration_ms,
        error=last_error,
        attempts=used,
    )


def default_checks() -> list[Check]:
    """Return the default set of required startup checks.

    Includes the daily stats check to ensure the daily stats database
    is available before the bot marks itself ready. This enables fast
    failure detection when daily stats data is missing.
    """
    ollama_timeout = max(settings.ollama_health_check_timeout + 2.0, 3.0)
    return [
        Check(name="ollama", fn=_check_ollama, required=True, timeout=ollama_timeout),
        Check(name="vault", fn=_check_vault, required=True, timeout=5.0),
        Check(name="executor_db", fn=_check_executor_db, required=True, timeout=5.0),
        make_daily_stats_check(),
    ]


def run_readiness_checks(
    checks: list[Check] | None = None,
    attempts: int = 2,
    backoff: float = 1.0,
) -> ReadinessReport:
    """Run the readiness gate and return a :class:`ReadinessReport`.

    Honors the ``BOT_SKIP_REQUIRED_CHECKS=1`` escape hatch — when set, every
    check is marked ``skipped`` and the report is forced to ``ok=True``. A
    warning is logged because skipping the gate is an operator override, not
    a normal state.
    """
    check_list = checks if checks is not None else default_checks()

    if os.environ.get(SKIP_REQUIRED_ENV) == "1":
        results = [
            CheckResult(
                name=c.name,
                status=STATUS_SKIPPED,
                required=c.required,
                duration_ms=0,
                error=None,
                attempts=0,
            )
            for c in check_list
        ]
        log.warning(
            "Readiness gate skipped via %s=1 — required checks not validated.",
            SKIP_REQUIRED_ENV,
        )
        return ReadinessReport(ok=True, skipped=True, results=results)

    results: list[CheckResult] = []
    for check in check_list:
        result = _run_with_retry(check, attempts=attempts, backoff=backoff)
        results.append(result)
        if result.status == STATUS_OK:
            log.info(
                "Readiness check %r OK in %d ms (attempt %d)",
                check.name,
                result.duration_ms,
                result.attempts,
            )
        else:
            log.error(
                "Readiness check %r %s after %d attempt(s): %s",
                check.name,
                result.status,
                result.attempts,
                result.error,
            )

    ok = all(r.status == STATUS_OK for r in results if r.required)
    return ReadinessReport(ok=ok, skipped=False, results=results)
