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

import logging
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "executor_runs.db"

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
})

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
            artifacts_path TEXT
        )
    """)
    # Migrate older databases that pre-date run_id / artifacts_path. ALTER TABLE
    # raises OperationalError if the column is already present — that's the
    # expected idempotency signal, so swallow it.
    for col, decl in (("run_id", "TEXT"), ("artifacts_path", "TEXT")):
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
    conn.commit()


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
                  tests_passed, deployed, artifacts_path
           FROM executor_runs
           ORDER BY id DESC
           LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [dict(r) for r in rows]


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
        for entry in list(run_dir.iterdir()):
            if not entry.is_file():
                continue
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
