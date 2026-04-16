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
from datetime import datetime
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
