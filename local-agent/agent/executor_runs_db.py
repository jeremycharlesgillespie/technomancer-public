"""
Executor Runs DB — SQLite persistence for executor run metadata.

Records a structured trace for every executor / claude_code_runner run so
cost, duration, and outcome trends can be analyzed without scraping logs.

Typical usage — insert at start, update on completion:

    run_id = record_run(
        jira_key="TK-447",
        branch="2026-04-16-052408-TK-447",
        started_at=datetime.now().isoformat(),
        status="running",
    )
    # ... do the run ...
    record_run(
        id=run_id,
        ended_at=datetime.now().isoformat(),
        duration_ms=12345,
        cost_usd=0.42,
        status="success",
        exit_code=0,
        tests_passed=True,
        deployed=True,
    )

Database: ``local-agent/data/executor_runs.db``
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .leak_counter import leak_counter, reset
from .tracing import DEFAULT_TRACE_ID, get_trace_id

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "executor_runs.db"

# Default root for AIM state files (``local-agent/aim``). Overridable per call
# via the ``aim_root`` argument to :func:`get_current_execution_per_project`
# and monkey-patched in tests.
AIM_ROOT: Path = Path(__file__).parent.parent / "aim"

# Hard ceiling on AIM state-file size. AIM's JSON snapshots are a few KB in
# practice — anything larger is almost certainly corrupt, and reading it in
# full from a /live HTTP handler would stall the request. The /live page
# must never hang on a bad file, so we skip and log instead.
MAX_AIM_STATE_FILE_BYTES: int = 1_000_000  # 1 MB

# Worker ``status`` values that mean the worker is actively tied to
# ``current_idea_id``. Anything else (``idle``, ``dead``, ``stuck``,
# ``rate_limited``, ``error``) is treated as not executing.
_ACTIVE_WORKER_STATUSES: frozenset[str] = frozenset(
    {"assigned", "executing", "watching"}
)

# Per-run artifact archive — stdout.log, stderr.log, diff.patch — under
# ARTIFACTS_DIR/<run_id>/. Capped at MAX_ARTIFACTS most-recent runs.
ARTIFACTS_DIR = Path(__file__).parent.parent / "executor_artifacts"
MAX_ARTIFACTS = 50

_local = threading.local()

# Whitelist of legal column names — prevents SQL injection via **kwargs keys.
_COLUMNS: frozenset[str] = frozenset({
    "id",
    "run_id",
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
    "artifacts_path",
    "pid",
    "killed_at",
    "kill_reason",
    "trace_id",
})

# Status values that mean the run is finished — the kill endpoint refuses to
# re-signal these (returns 409) and the nightly purge treats them as settled.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({
    "success",
    "failed",
    "error",
    "crashed",
    "killed",
    "timeout",
    "done",
    "cancelled",
    "canceled",
})

# SIGTERM grace period before escalating to SIGKILL. Exposed as a module
# attribute so tests can patch it down to avoid a 10-second wait.
KILL_SIGTERM_TIMEOUT_SECONDS: float = 10.0
KILL_POLL_INTERVAL_SECONDS: float = 0.25

# Whitelist of legal column names for the executor_tool_calls table.
_TOOL_COLUMNS: frozenset[str] = frozenset({
    "id",
    "run_id",
    "tool_name",
    "started_at",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "ok",
    "error_message",
})


def _migrate_story_model_usage_cache_cols(conn: sqlite3.Connection) -> None:
    """Add cache_read_tokens / cache_write_tokens if the table predates them."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(story_model_usage)").fetchall()
    }
    for col, defn in (
        ("cache_read_tokens", "INTEGER DEFAULT 0"),
        ("cache_write_tokens", "INTEGER DEFAULT 0"),
    ):
        if col not in existing:
            conn.execute(
                f"ALTER TABLE story_model_usage ADD COLUMN {col} {defn}"
            )
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection (created on first use)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create the executor_runs table if it doesn't exist, and migrate
    older schemas by adding columns introduced after the original CREATE."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executor_runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            jira_key       TEXT,
            branch         TEXT,
            started_at     TEXT,
            ended_at       TEXT,
            duration_ms    INTEGER,
            cost_usd       REAL,
            status         TEXT,
            exit_code      INTEGER,
            tests_passed   INTEGER,
            deployed       INTEGER,
            run_id         TEXT,
            artifacts_path TEXT,
            pid            INTEGER,
            killed_at      TEXT,
            kill_reason    TEXT,
            trace_id       TEXT
        )
    """)
    # Migrate older databases that pre-date run_id / artifacts_path / pid /
    # killed_at / kill_reason / trace_id. ALTER TABLE raises OperationalError
    # if the column is already present — that's the expected idempotency
    # signal, so swallow it.
    for col, decl in (
        ("run_id", "TEXT"),
        ("artifacts_path", "TEXT"),
        ("pid", "INTEGER"),
        ("killed_at", "TEXT"),
        ("kill_reason", "TEXT"),
        ("trace_id", "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE executor_runs ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_started
        ON executor_runs (started_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_jira
        ON executor_runs (jira_key)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_run_id
        ON executor_runs (run_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_trace_id
        ON executor_runs (trace_id)
    """)
    # Dashboard and /api/executor/runs filter by status and order by recency —
    # a composite index avoids a full table scan as run history grows.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_runs_status_started
        ON executor_runs (status, started_at DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executor_tool_calls (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         INTEGER NOT NULL,
            tool_name      TEXT,
            started_at     TEXT,
            duration_ms    INTEGER,
            input_tokens   INTEGER,
            output_tokens  INTEGER,
            ok             INTEGER,
            error_message  TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_executor_tool_calls_run_id
        ON executor_tool_calls (run_id, started_at)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crash_signatures (
            signature   TEXT PRIMARY KEY,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            jira_key    TEXT,
            count       INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS story_model_usage (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            story_key           TEXT,
            model               TEXT,
            call_count          INTEGER,
            cost_usd            REAL,
            cache_read_tokens   INTEGER DEFAULT 0,
            cache_write_tokens  INTEGER DEFAULT 0,
            recorded_at         TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_story_model_usage_story
        ON story_model_usage (story_key, recorded_at)
    """)
    # Migration: add cache columns to existing databases that predate this schema.
    _migrate_story_model_usage_cache_cols(conn)
    conn.commit()


# ---------------------------------------------------------------------------
# AIM state file readers — surfaced here (instead of embedded in the /live
# route) so the I/O is shared, reusable, and independently testable. The
# /live page must never hang, so every read is size-capped and wrapped in
# try/except. A missing/corrupt file yields no row — never a crash.
# ---------------------------------------------------------------------------


def _read_aim_state_file(path: Path) -> dict[str, Any] | None:
    """Read one AIM state file and parse it as JSON.

    Returns ``None`` (after logging a warning) when the file is missing,
    exceeds :data:`MAX_AIM_STATE_FILE_BYTES`, or fails to decode. Callers
    treat ``None`` as "no data" and skip the project silently.
    """
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size > MAX_AIM_STATE_FILE_BYTES:
            log.warning(
                "[aim-state] %s is %d bytes (> %d cap); skipping",
                path, size, MAX_AIM_STATE_FILE_BYTES,
            )
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError on every Python version.
        log.warning("[aim-state] cannot read %s: %s", path, exc)
        return None


def get_current_execution_per_project(
    aim_root: Path | None = None,
    primary_label: str = "primary",
) -> list[dict[str, Any]]:
    """Return per-project current-execution state parsed from AIM state files.

    Scans ``<aim_root>/.aim_state.json`` plus every
    ``<aim_root>/projects/<name>/.aim_state.json`` and returns one row per
    discovered project — including ones whose worker is idle, so the /live
    page can show an explicit "idle" indicator instead of silently omitting
    dormant projects.

    This function is the shared I/O path used by the /live landing page.
    It is deliberately synchronous and sequential — AIM only produces a
    handful of state files (one per project) and each is KB-sized, so
    reading them serially stays well under the 2-second budget the
    endpoint must meet.

    Args:
        aim_root: Override the default :data:`AIM_ROOT` directory. Tests
            pass a ``tmp_path / "aim"`` here so no real AIM is touched.
        primary_label: Display name for the top-level (non-sub) project.
            Callers typically pass ``settings.jira_project_key or "primary"``.

    Returns:
        List of dicts with keys ``project``, ``status``, ``current_idea_id``,
        ``started_at``, ``last_observation``, ``is_executing``. Empty list
        when no state files are present or readable.
    """
    root = aim_root if aim_root is not None else AIM_ROOT

    targets: list[tuple[str, Path]] = []
    primary_state = root / ".aim_state.json"
    if primary_state.is_file():
        targets.append((primary_label, primary_state))

    projects_dir = root / "projects"
    if projects_dir.is_dir():
        try:
            subs = sorted(projects_dir.iterdir())
        except OSError as exc:
            log.warning(
                "[aim-state] cannot list %s: %s", projects_dir, exc
            )
            subs = []
        for sub in subs:
            if not sub.is_dir():
                continue
            sub_state = sub / ".aim_state.json"
            if sub_state.is_file():
                targets.append((sub.name, sub_state))

    rows: list[dict[str, Any]] = []
    for label, path in targets:
        data = _read_aim_state_file(path)
        if data is None:
            continue
        worker = data.get("worker") or {}
        status = (worker.get("status") or "idle").strip() or "idle"
        current_id = worker.get("current_idea_id")
        rows.append({
            "project": label,
            "status": status,
            "current_idea_id": current_id,
            "started_at": worker.get("started_at") or "",
            "last_observation": worker.get("last_observation") or "",
            "is_executing": bool(current_id) and status in _ACTIVE_WORKER_STATUSES,
        })
    return rows


# ---------------------------------------------------------------------------
# Crash signature dedup — used by agent.crash_triage to suppress repeat
# Jira stories for identical stack signatures within a time window.
# ---------------------------------------------------------------------------


def get_crash_signature(signature: str) -> dict[str, Any] | None:
    """Fetch the ``crash_signatures`` row for ``signature``, or ``None``."""
    if not signature:
        return None
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT signature, first_seen, last_seen, jira_key, count "
        "FROM crash_signatures WHERE signature = ?",
        (signature,),
    ).fetchone()
    return dict(row) if row is not None else None


def upsert_crash_signature(
    signature: str,
    jira_key: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Insert a new crash_signatures row or bump an existing one.

    On insert, ``first_seen`` and ``last_seen`` are both set to ``now`` and
    ``count`` starts at 1. On update, ``last_seen`` is set to ``now`` and
    ``count`` is incremented by one; ``jira_key`` is only overwritten when
    the caller passes a non-None value, so dedup-hit bumps (which don't
    create a new Jira story) preserve the original ``jira_key``.

    Returns the resulting row as a dict.
    """
    if not signature:
        raise ValueError("signature must be non-empty")
    init_db()
    conn = _get_conn()
    now_iso = (now or datetime.now()).isoformat()

    existing = conn.execute(
        "SELECT jira_key, count FROM crash_signatures WHERE signature = ?",
        (signature,),
    ).fetchone()

    if existing is None:
        conn.execute(
            "INSERT INTO crash_signatures "
            "(signature, first_seen, last_seen, jira_key, count) "
            "VALUES (?, ?, ?, ?, 1)",
            (signature, now_iso, now_iso, jira_key),
        )
    elif jira_key is not None:
        conn.execute(
            "UPDATE crash_signatures "
            "SET last_seen = ?, count = count + 1, jira_key = ? "
            "WHERE signature = ?",
            (now_iso, jira_key, signature),
        )
    else:
        conn.execute(
            "UPDATE crash_signatures "
            "SET last_seen = ?, count = count + 1 "
            "WHERE signature = ?",
            (now_iso, signature),
        )
    conn.commit()

    row = conn.execute(
        "SELECT signature, first_seen, last_seen, jira_key, count "
        "FROM crash_signatures WHERE signature = ?",
        (signature,),
    ).fetchone()
    return dict(row) if row is not None else {}


# ---------------------------------------------------------------------------
# Per-story, per-model usage — feeds the PPTX cost-slide so we can split
# brain (haiku) from worker (opus) spend instead of showing only the total.
# One row per (story_key, model, call) completion; the reporter aggregates.
# ---------------------------------------------------------------------------


def record_story_model_usage(
    story_key: str,
    model: str,
    call_count: int,
    cost_usd: float,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    recorded_at: str | None = None,
) -> int:
    """Insert one ``story_model_usage`` row for a finished claude -p call.

    Args:
        story_key: Jira key (e.g. ``"TK-613"``) the usage is attributed to.
        model: Model id reported by Claude Code (e.g. ``"claude-opus-4-6"``).
        call_count: Number of assistant turns attributed to this model in
            the run being recorded.
        cost_usd: Dollar cost for those turns. Callers compute this from the
            per-turn ``usage`` blocks.
        cache_read_tokens: Cumulative ``cache_read_input_tokens`` across all
            turns — tokens served from the prompt cache at 0.1× rate.
        cache_write_tokens: Cumulative ``cache_creation_input_tokens`` — tokens
            written to the cache at 1.25× rate.
        recorded_at: ISO8601 timestamp. Defaults to ``datetime.now()``.

    Returns:
        The inserted row id.
    """
    init_db()
    conn = _get_conn()
    ts = recorded_at or datetime.now().isoformat()
    cursor = conn.execute(
        "INSERT INTO story_model_usage "
        "(story_key, model, call_count, cost_usd, "
        "cache_read_tokens, cache_write_tokens, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            story_key,
            model,
            int(call_count),
            float(cost_usd),
            int(cache_read_tokens),
            int(cache_write_tokens),
            ts,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def get_story_model_usage(story_key: str) -> list[dict[str, Any]]:
    """Return every ``story_model_usage`` row for ``story_key``, oldest first."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, story_key, model, call_count, cost_usd, "
        "cache_read_tokens, cache_write_tokens, recorded_at "
        "FROM story_model_usage WHERE story_key = ? "
        "ORDER BY recorded_at ASC, id ASC",
        (story_key,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_story_model_usage() -> list[dict[str, Any]]:
    """Return all ``story_model_usage`` rows across every story, oldest first."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, story_key, model, call_count, cost_usd, "
        "cache_read_tokens, cache_write_tokens, recorded_at "
        "FROM story_model_usage "
        "ORDER BY recorded_at ASC, id ASC",
    ).fetchall()
    return [dict(r) for r in rows]


def _coerce(key: str, value: Any) -> Any:
    """Coerce booleans to 0/1 for INTEGER columns — sqlite stores either but
    queries are more predictable when we normalize."""
    if key in ("tests_passed", "deployed") and isinstance(value, bool):
        return 1 if value else 0
    return value


def record_run(**fields: Any) -> int:
    """Insert a new run row, or update an existing row if ``id`` is provided.

    All fields are optional. Unknown keys are silently ignored (whitelist-only)
    so callers can safely pass the same kwargs to start and end a run.

    Returns:
        The row id of the inserted or updated run.
    """
    init_db()
    conn = _get_conn()

    # Drop any unknown keys — prevents SQL injection via kwargs keys
    safe_fields = {
        k: _coerce(k, v) for k, v in fields.items() if k in _COLUMNS
    }
    run_id = safe_fields.pop("id", None)

    if run_id is not None:
        if safe_fields:
            set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
            params = list(safe_fields.values()) + [run_id]
            conn.execute(
                f"UPDATE executor_runs SET {set_clause} WHERE id = ?",
                params,
            )
            conn.commit()
        return int(run_id)

    # Insert path — default started_at if caller didn't supply one
    if "started_at" not in safe_fields:
        safe_fields["started_at"] = datetime.now().isoformat()

    # Default trace_id from the current ContextVar so executor runs inherit
    # the trace of the request that spawned them. The sentinel means "no
    # trace bound" — store NULL instead so queries can distinguish between
    # "untraced" and "traced with '-'".
    if "trace_id" not in safe_fields:
        current = get_trace_id()
        safe_fields["trace_id"] = None if current == DEFAULT_TRACE_ID else current

    columns = ", ".join(safe_fields.keys())
    placeholders = ", ".join("?" * len(safe_fields))
    cursor = conn.execute(
        f"INSERT INTO executor_runs ({columns}) VALUES ({placeholders})",
        list(safe_fields.values()),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def _coerce_tool(key: str, value: Any) -> Any:
    """Coerce booleans to 0/1 for INTEGER columns on the tool-call table."""
    if key == "ok" and isinstance(value, bool):
        return 1 if value else 0
    return value


def record_tool_call(**fields: Any) -> int:
    """Insert or update one row in ``executor_tool_calls``.

    Pass ``id=`` to update an existing row (for two-phase start/complete
    writes — the event stream parser currently writes a single row per
    tool_result so this is mostly used for direct inserts).

    Unknown keys are silently dropped so callers can pass extra metadata
    without crashing. Returns the row id of the inserted or updated row.
    """
    init_db()
    conn = _get_conn()

    safe_fields = {
        k: _coerce_tool(k, v) for k, v in fields.items() if k in _TOOL_COLUMNS
    }
    row_id = safe_fields.pop("id", None)

    if row_id is not None:
        if safe_fields:
            set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
            params = list(safe_fields.values()) + [row_id]
            conn.execute(
                f"UPDATE executor_tool_calls SET {set_clause} WHERE id = ?",
                params,
            )
            conn.commit()
        return int(row_id)

    if "started_at" not in safe_fields:
        safe_fields["started_at"] = datetime.now().isoformat()

    columns = ", ".join(safe_fields.keys())
    placeholders = ", ".join("?" * len(safe_fields))
    cursor = conn.execute(
        f"INSERT INTO executor_tool_calls ({columns}) VALUES ({placeholders})",
        list(safe_fields.values()),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def get_tool_calls(run_id: int) -> list[dict[str, Any]]:
    """Return all tool-call rows for a run, oldest first (by started_at).

    Args:
        run_id: ``executor_runs.id`` to look up calls for.
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, run_id, tool_name, started_at, duration_ms,
                  input_tokens, output_tokens, ok, error_message
           FROM executor_tool_calls
           WHERE run_id = ?
           ORDER BY started_at ASC, id ASC""",
        (int(run_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def executor_run_summary(run_id: int | str) -> dict[str, Any]:
    """Return a compact per-run summary suitable for dashboards and alerts.

    Accepts either the integer row id returned by :func:`record_run`, or the
    sortable artifact run_id (``"YYYYMMDD-HHMMSS-<jira_key>"``). Both forms
    are common: the executor hands out the artifact id, while internal
    callers that already have the row id can skip the second lookup.

    Args:
        run_id: Either the numeric ``executor_runs.id`` or the artifact
            ``run_id`` string.

    Returns:
        ``{"cost_usd", "duration_ms", "status", "story_key"}``. ``story_key``
        maps the DB's ``jira_key`` column to the domain name operators use
        when reading dashboards.

    Raises:
        KeyError: No matching row was found.
    """
    init_db()
    conn = _get_conn()
    row = None
    try:
        int_id = int(run_id)
    except (TypeError, ValueError):
        int_id = None
    if int_id is not None:
        row = conn.execute(
            "SELECT cost_usd, duration_ms, status, jira_key "
            "FROM executor_runs WHERE id = ?",
            (int_id,),
        ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT cost_usd, duration_ms, status, jira_key "
            "FROM executor_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    if row is None:
        raise KeyError(f"No executor run with id {run_id!r}")
    return {
        "cost_usd": row["cost_usd"],
        "duration_ms": row["duration_ms"],
        "status": row["status"],
        "story_key": row["jira_key"],
    }


def get_run_by_run_id(run_id: str) -> dict[str, Any] | None:
    """Look up a single run row by its sortable ``run_id`` string.

    Args:
        run_id: Artifact-style ID like ``YYYYMMDD-HHMMSS-<jira_key>``.

    Returns:
        The matching row as a dict, or ``None`` if no row exists. When the
        index has duplicates (shouldn't happen, but the column isn't UNIQUE)
        the most recent row wins.
    """
    init_db()
    conn = _get_conn()
    row = conn.execute(
        """SELECT id, run_id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed, artifacts_path, pid, trace_id
           FROM executor_runs
           WHERE run_id = ?
           ORDER BY id DESC
           LIMIT 1""",
        (str(run_id),),
    ).fetchone()
    return dict(row) if row is not None else None


def get_runs_by_trace_id(trace_id: str) -> list[dict[str, Any]]:
    """Return all runs tagged with the given ``trace_id``, newest first.

    Args:
        trace_id: Crockford base32 ULID from :mod:`agent.tracing`.

    Returns:
        List of run rows as dicts. Empty when no runs carry this trace.
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, run_id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed, artifacts_path, pid, trace_id
           FROM executor_runs
           WHERE trace_id = ?
           ORDER BY id DESC""",
        (str(trace_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent runs, newest first.

    Args:
        limit: Max rows to return (default 20).
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, run_id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed, artifacts_path, trace_id
           FROM executor_runs
           ORDER BY id DESC
           LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_paginated(
    limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """Return a page of runs ordered newest-first.

    Companion to :func:`count_runs` for building a paginated UI. Callers
    are expected to have already clamped ``limit`` and ``offset`` to legal
    values — this function coerces to non-negative ints but does not
    enforce an upper bound on ``limit``.

    Args:
        limit: Max rows to return per page.
        offset: Number of rows to skip (``limit * page_index``).
    """
    init_db()
    conn = _get_conn()
    safe_limit = max(int(limit), 0)
    safe_offset = max(int(offset), 0)
    rows = conn.execute(
        """SELECT id, run_id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed, artifacts_path, trace_id
           FROM executor_runs
           ORDER BY id DESC
           LIMIT ? OFFSET ?""",
        (safe_limit, safe_offset),
    ).fetchall()
    return [dict(r) for r in rows]


def count_runs() -> int:
    """Return the total number of rows in ``executor_runs``.

    Used by the paginated API so the UI can render "X of N" and decide
    when to disable the Load-more control.
    """
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) AS n FROM executor_runs").fetchone()
    return int(row["n"]) if row is not None else 0


# ---------------------------------------------------------------------------
# Kill a running executor — SIGTERM, wait, SIGKILL escalation
# ---------------------------------------------------------------------------


def _is_process_alive(pid: int | None) -> bool:
    """Return True if ``pid`` still corresponds to a running process.

    Cross-platform — uses ``OpenProcess`` + ``GetExitCodeProcess`` on Windows
    (``os.kill(pid, 0)`` is not reliable there) and ``os.kill(pid, 0)`` on
    POSIX. Never raises; missing pids, invalid pids, and permission errors
    all collapse to a boolean.

    ``PermissionError`` on POSIX means the pid exists but we don't own it —
    the process is alive from the caller's point of view, so return True.
    """
    if not pid or pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid),
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                    handle, ctypes.byref(exit_code),
                )
                if not ok:
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except OSError:
        return False


def _send_sigterm(pid: int) -> None:
    """Ask the process to terminate gracefully.

    On POSIX this is literally ``os.kill(pid, SIGTERM)``. On Windows,
    ``taskkill`` without ``/F`` sends a close event that well-behaved console
    apps can trap — the Windows analogue of SIGTERM. Never raises; all errors
    are logged and swallowed so the escalation loop can still observe
    ``_is_process_alive`` and escalate.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(int(pid))],
                capture_output=True,
                timeout=5,
            )
        else:
            os.kill(int(pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("kill_run: SIGTERM to pid %s failed: %s", pid, exc)


def _send_sigkill(pid: int) -> None:
    """Force-terminate the process tree.

    On POSIX sends ``SIGKILL``; on Windows uses ``taskkill /F /T`` which
    TerminateProcess()-es the pid and its descendants. Never raises.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(int(pid))],
                capture_output=True,
                timeout=5,
            )
        else:
            os.kill(int(pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("kill_run: SIGKILL to pid %s failed: %s", pid, exc)


def get_run(run_id: int) -> dict[str, Any] | None:
    """Fetch a single run row by integer ``executor_runs.id``.

    Returns the row as a dict (including ``pid``, ``status``, and the
    ``killed_at`` / ``kill_reason`` columns), or ``None`` when no row exists.
    """
    init_db()
    conn = _get_conn()
    row = conn.execute(
        """SELECT id, run_id, jira_key, branch, started_at, ended_at,
                  duration_ms, cost_usd, status, exit_code,
                  tests_passed, deployed, artifacts_path, pid,
                  killed_at, kill_reason, trace_id
           FROM executor_runs
           WHERE id = ?""",
        (int(run_id),),
    ).fetchone()
    return dict(row) if row is not None else None


class RunNotFoundError(LookupError):
    """Raised by :func:`kill_run` when the run id doesn't exist."""


class RunAlreadyTerminalError(RuntimeError):
    """Raised by :func:`kill_run` when the run is already in a terminal state.

    The ``status`` attribute holds the row's current status so the HTTP layer
    can surface it in a 409 response body.
    """

    def __init__(self, run_id: int, status: str) -> None:
        super().__init__(
            f"executor run {run_id} is already terminal (status={status!r})"
        )
        self.run_id = run_id
        self.status = status


def kill_run(
    run_id: int,
    reason: str | None = None,
    sigterm_timeout: float | None = None,
    poll_interval: float | None = None,
) -> dict[str, Any]:
    """Kill a running executor and mark the DB row ``killed``.

    Sends SIGTERM to the row's ``pid`` (if any), waits up to
    ``sigterm_timeout`` seconds polling for the process to exit, then
    escalates to SIGKILL if still alive. Updates the row with
    ``status='killed'``, ``killed_at=<now>``, ``kill_reason=reason``, and
    ``ended_at=<now>``.

    Args:
        run_id: ``executor_runs.id`` of the run to kill.
        reason: Optional free-text reason persisted to ``kill_reason``.
        sigterm_timeout: Grace period before SIGKILL. Defaults to
            :data:`KILL_SIGTERM_TIMEOUT_SECONDS` (10s).
        poll_interval: How often to re-check liveness during the grace
            period. Defaults to :data:`KILL_POLL_INTERVAL_SECONDS`.

    Returns:
        ``{"run_id", "pid", "escalated", "status"}`` where ``escalated`` is
        True iff SIGKILL had to be sent.

    Raises:
        RunNotFoundError: No row matches ``run_id``.
        RunAlreadyTerminalError: The row's status is already terminal.
    """
    if sigterm_timeout is None:
        sigterm_timeout = KILL_SIGTERM_TIMEOUT_SECONDS
    if poll_interval is None:
        poll_interval = KILL_POLL_INTERVAL_SECONDS

    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, pid, status FROM executor_runs WHERE id = ?",
        (int(run_id),),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"No executor run with id {run_id}")

    current_status = (row["status"] or "").lower()
    if current_status in TERMINAL_RUN_STATUSES:
        raise RunAlreadyTerminalError(int(run_id), row["status"] or "")

    pid = row["pid"]
    escalated = False
    if pid:
        _send_sigterm(int(pid))
        deadline = time.monotonic() + max(float(sigterm_timeout), 0.0)
        while time.monotonic() < deadline:
            if not _is_process_alive(pid):
                break
            time.sleep(max(float(poll_interval), 0.01))
        if _is_process_alive(pid):
            escalated = True
            _send_sigkill(int(pid))

    now_iso = datetime.now().isoformat()
    conn.execute(
        "UPDATE executor_runs "
        "SET status = ?, killed_at = ?, kill_reason = ?, ended_at = ? "
        "WHERE id = ?",
        ("killed", now_iso, reason, now_iso, int(run_id)),
    )
    conn.commit()

    log.info(
        "kill_run: run_id=%s pid=%s escalated=%s reason=%r",
        run_id, pid, escalated, reason,
    )
    return {
        "run_id": int(run_id),
        "pid": pid,
        "escalated": escalated,
        "status": "killed",
    }


# ---------------------------------------------------------------------------
# Per-run artifact archive
# ---------------------------------------------------------------------------


def _capture_branch_diff(branch_name: str | None) -> str:
    """Capture ``git diff main...<branch>`` from the repo root.

    Returns the diff text, or a short error message prefixed with ``# ERROR:``
    so the file is never empty and operators can see what went wrong. Never
    raises — failure here must not abort the run.
    """
    if not branch_name:
        return "# ERROR: no branch name supplied — diff skipped\n"
    repo_root = Path(__file__).parent.parent.parent
    try:
        result = subprocess.run(
            ["git", "diff", f"main...{branch_name}"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return f"# ERROR: git diff failed: {exc}\n"
    if result.returncode != 0:
        return (
            f"# ERROR: git diff exited {result.returncode}\n"
            f"# stderr: {result.stderr.strip()}\n"
        )
    return result.stdout


def archive_run(
    run_id: str,
    stdout: str,
    stderr: str,
    branch_name: str | None,
) -> Path:
    """Persist stdout / stderr / branch diff for one executor run.

    Writes three files under ``ARTIFACTS_DIR/<run_id>/``:

    * ``stdout.log`` — captured subprocess stdout
    * ``stderr.log`` — captured subprocess stderr
    * ``diff.patch`` — output of ``git diff main...<branch_name>`` from the
      repo root, captured before the branch is merged

    Updates the matching ``executor_runs.run_id`` row's ``artifacts_path``
    column if one exists, then runs :func:`prune_old_artifacts` so the
    archive stays bounded.

    Args:
        run_id: Sortable ID like ``YYYYMMDD-HHMMSS-TK-452``. Used as the
            directory name and the lookup key in SQLite.
        stdout: Captured stdout text from the run.
        stderr: Captured stderr text from the run.
        branch_name: Git branch the run executed on. ``None`` is allowed —
            the diff file will record an error instead of crashing.

    Returns:
        Absolute path to the created artifact directory.
    """
    target = ARTIFACTS_DIR / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "stdout.log").write_text(stdout or "", encoding="utf-8")
    (target / "stderr.log").write_text(stderr or "", encoding="utf-8")
    (target / "diff.patch").write_text(
        _capture_branch_diff(branch_name), encoding="utf-8"
    )

    # Best-effort SQLite update — instrumentation must not break the caller.
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "UPDATE executor_runs SET artifacts_path = ? WHERE run_id = ?",
            (str(target), run_id),
        )
        conn.commit()
    except sqlite3.Error:
        log.debug("archive_run failed to update artifacts_path", exc_info=True)

    try:
        prune_old_artifacts(keep=MAX_ARTIFACTS)
    except OSError:
        log.debug("prune_old_artifacts failed", exc_info=True)

    return target


def _discover_artifacts(run_id: str) -> list[Path]:
    """Discover standard artifact files for a given run ID.

    Scans the ARTIFACTS_DIR for a directory matching the run_id and
    returns a list of standard artifact files (stdout.log, stderr.log, diff.patch).

    Args:
        run_id: The run ID to discover artifacts for.

    Returns:
        List of Path objects pointing to artifact files.
    """
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    
    # Only return standard artifact files
    expected_files = {"stdout.log", "stderr.log", "diff.patch"}
    artifacts = []
    for item in run_dir.iterdir():
        if item.is_file() and item.name in expected_files:
            artifacts.append(item)
    return artifacts


def prune_old_artifacts(keep: int = MAX_ARTIFACTS) -> int:
    """Delete oldest artifact directories so at most ``keep`` remain.

    Run-id directories are named ``YYYYMMDD-HHMMSS-...`` so a lexicographic
    sort matches chronological order — newest names sort last. Anything
    that isn't a directory under ARTIFACTS_DIR is left alone.

    Args:
        keep: Number of most-recent directories to retain. ``0`` deletes all.

    Returns:
        Count of directories deleted.
    """
    if not ARTIFACTS_DIR.exists():
        return 0
    dirs = sorted(
        (p for p in ARTIFACTS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    if len(dirs) <= keep:
        return 0
    to_delete = dirs[: len(dirs) - keep]
    deleted = 0
    for path in to_delete:
        try:
            shutil.rmtree(path)
            deleted += 1
        except OSError:
            log.warning("Failed to prune artifact dir %s", path, exc_info=True)
    return deleted


# ---------------------------------------------------------------------------
# Retention / rotation policy
# ---------------------------------------------------------------------------

# Default retention window used by the nightly scheduler.
RETENTION_DAYS = 30
# Local hour (24h) at which the nightly purge fires.
PURGE_HOUR = 3


def _purge_old_artifact_files(cutoff_ts: float) -> int:
    """Unlink archived files whose mtime is older than ``cutoff_ts``.

    Walks ``ARTIFACTS_DIR/<run_id>/*`` — any file whose modification time
    predates the cutoff is deleted. Run directories left empty afterwards
    are removed too so the archive doesn't accumulate stub folders.

    Returns the number of files (not directories) that were unlinked.
    """
    if not ARTIFACTS_DIR.exists():
        return 0
    removed = 0
    for run_dir in ARTIFACTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        # Use _discover_artifacts to get the list of artifact files
        artifact_files = _discover_artifacts(run_dir.name)
        for entry in artifact_files:
            try:
                if entry.stat().st_mtime < cutoff_ts:
                    entry.unlink()
                    removed += 1
            except OSError:
                log.warning(
                    "Failed to purge artifact file %s", entry, exc_info=True
                )
        # Best-effort: drop the directory if we emptied it.
        try:
            next(run_dir.iterdir())
        except StopIteration:
            try:
                run_dir.rmdir()
            except OSError:
                log.debug(
                    "Failed to remove empty artifact dir %s", run_dir,
                    exc_info=True,
                )
        except OSError:
            pass
    return removed


def cleanup_run_artifacts(run_id: str, cutoff_ts: float) -> int:
    """Clean up artifact files for a specific run ID using _discover_artifacts helper.

    Args:
        run_id: The run ID to clean up artifacts for.
        cutoff_ts: Timestamp cutoff for artifact files.

    Returns:
        Number of artifact files that were unlinked.
    """
    # Use _discover_artifacts to get the list of artifact files for this run
    artifact_files = _discover_artifacts(run_id)
    removed = 0
    for entry in artifact_files:
        try:
            if entry.stat().st_mtime < cutoff_ts:
                entry.unlink()
                removed += 1
        except OSError:
            log.warning(
                "Failed to purge artifact file %s", entry, exc_info=True
            )
    return removed


def purge_old_runs(days: int) -> int:
    """Delete executor_runs rows and archived files older than ``days``.

    DB rows are removed when ``started_at`` predates ``datetime('now', '-N days')``
    using a parametrized SQL comparison. On-disk artifacts under
    :data:`ARTIFACTS_DIR` are also pruned by mtime so the archive stays
    bounded alongside the DB.

    Args:
        days: Retention window in days. Rows or files older than this are
            deleted. Must be non-negative — negative values are clamped to 0
            (which would wipe everything, so callers should pick deliberately).

    Returns:
        Number of database rows deleted. File deletions are logged but not
        included in the return value so callers can reason about DB state
        directly.
    """
    days = max(int(days), 0)
    init_db()
    conn = _get_conn()

    # Compute the cutoff in Python local time — record_run writes started_at
    # via datetime.now().isoformat(), which is local time, so comparing
    # against SQLite's UTC-based datetime('now') would drift by the local
    # UTC offset.
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(
        sep=" ", timespec="seconds"
    )
    # Purge child tool_call rows first — no FK cascade in SQLite, but keeping
    # orphaned tool calls around would defeat the retention goal.
    conn.execute(
        "DELETE FROM executor_tool_calls "
        "WHERE run_id IN ("
        "  SELECT id FROM executor_runs "
        "  WHERE started_at IS NOT NULL "
        "  AND datetime(started_at) < datetime(?)"
        ")",
        (cutoff,),
    )
    cursor = conn.execute(
        "DELETE FROM executor_runs "
        "WHERE started_at IS NOT NULL "
        "AND datetime(started_at) < datetime(?)",
        (cutoff,),
    )
    deleted_rows = cursor.rowcount or 0
    conn.commit()

    cutoff_ts = time.time() - (days * 86400)
    files_removed = _purge_old_artifact_files(cutoff_ts)

    log.info(
        "purge_old_runs: deleted %d row(s) and %d artifact file(s) older "
        "than %d day(s)",
        deleted_rows, files_removed, days,
    )
    return deleted_rows


def _seconds_until_purge(hour: int = PURGE_HOUR) -> float:
    """Seconds until the next local ``hour:00`` boundary."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max((target - now).total_seconds(), 60)


def start_purge_scheduler(days: int = RETENTION_DAYS, hour: int = PURGE_HOUR) -> None:
    """Schedule :func:`purge_old_runs` to run every day at ``hour:00`` local.

    Creates a monitored asyncio task that sleeps until the next boundary,
    runs the purge in a worker thread (so SQLite and filesystem I/O don't
    block the event loop), then sleeps 24h to the next run.

    Safe to call multiple times — each call registers a separate task, so
    only invoke it once from the bot startup path.
    """
    import asyncio
    from .task_manager import create_monitored_task

    async def _loop() -> None:
        log.info(
            "[purge_old_runs] scheduled daily at %02d:00 (retention: %d days)",
            hour, days,
        )
        while True:
            wait = _seconds_until_purge(hour)
            log.info(
                "[purge_old_runs] next run in %.1f hours", wait / 3600,
            )
            await asyncio.sleep(wait)
            try:
                deleted = await asyncio.to_thread(purge_old_runs, days)
                log.info("[purge_old_runs] nightly purge deleted %d row(s)", deleted)
            except Exception:
                log.exception("[purge_old_runs] nightly purge failed")
            # Sleep past the trigger boundary before recomputing the next wait.
            await asyncio.sleep(120)

    create_monitored_task(_loop(), "executor-runs-purge", critical=False)


# ---------------------------------------------------------------------------
# CLI: ``python -m agent.executor_runs_db purge <days>``
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    """CLI entry point — supports ``purge <days>`` for manual invocation."""
    if len(argv) >= 2 and argv[0] == "purge":
        try:
            days = int(argv[1])
        except ValueError:
            print(f"error: days must be an integer, got {argv[1]!r}")
            return 2
        deleted = purge_old_runs(days)
        print(f"Deleted {deleted} row(s) older than {days} day(s)")
        return 0
    print("usage: python -m agent.executor_runs_db purge <days>")
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
