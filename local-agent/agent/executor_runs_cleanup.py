"""
Executor Runs Cleanup — prune old rows in ``executor_runs`` and their
companion log directories under ``idea_board/execution_logs/``.

Runs alongside the existing per-run artifact retention in
:mod:`agent.executor_runs_db` but targets a different growth surface: the
``execution_logs/`` tree that the idea board writes one file/dir per run.
That directory shows up as untracked in ``git status`` and grows unbounded
because the executor only prunes it on process restart — which rarely
happens in production.

Policy:

* Delete rows from ``executor_runs`` whose ``started_at`` is older than
  ``max_age_days`` **and** which are not among the most recent
  ``keep_last_n`` rows by ``started_at``. This preserves a rolling window
  of recent history regardless of calendar age, so a quiet week doesn't
  wipe out the dashboard.
* For every deleted row, ``shutil.rmtree`` the matching
  ``idea_board/execution_logs/<run_id>/`` (if it exists) and also unlink
  any flat ``<run_id>.log`` / ``<run_id>.done`` files that match.

The module supports two entry points:

* :func:`cleanup_old_runs` — library API, returns a result dict.
* ``python -m agent.executor_runs_cleanup`` — CLI for ad-hoc runs.

And :func:`start_cleanup_scheduler` registers a ``threading.Timer`` loop
that fires once at startup and every ``interval_seconds`` after.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

from . import executor_runs_db
from .executor_runs_db import init_db

log = logging.getLogger(__name__)

# idea_board/execution_logs/ — one artifact per idea/run. Defined at module
# level so tests can monkeypatch it.
EXECUTION_LOGS_DIR: Path = (
    Path(__file__).parent.parent / "idea_board" / "execution_logs"
)

# Default schedule: every 6 hours.
DEFAULT_INTERVAL_SECONDS: int = 6 * 60 * 60

# Module-level timer handle so stop_cleanup_scheduler can cancel it.
_timer: threading.Timer | None = None
_timer_lock = threading.Lock()
_scheduler_running = False


# ---------------------------------------------------------------------------
# Row selection + on-disk cleanup
# ---------------------------------------------------------------------------


def _select_rows_to_delete(
    conn: sqlite3.Connection,
    max_age_days: int,
    keep_last_n: int,
) -> list[sqlite3.Row]:
    """Return rows that are both older than ``max_age_days`` and outside the
    newest ``keep_last_n`` window.

    Matches the story's spec:
        DELETE FROM executor_runs
        WHERE started_at < now - N days
          AND id NOT IN (
              SELECT id FROM executor_runs
              ORDER BY started_at DESC LIMIT keep_last_n
          )
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, run_id, jira_key, started_at
        FROM executor_runs
        WHERE started_at IS NOT NULL
          AND datetime(started_at) < datetime('now', ?)
          AND id NOT IN (
              SELECT id FROM executor_runs
              ORDER BY started_at DESC
              LIMIT ?
          )
        """,
        (f"-{int(max_age_days)} days", int(keep_last_n)),
    ).fetchall()
    return list(rows)


def _path_size_bytes(path: Path) -> int:
    """Return total bytes under ``path``, or 0 if it doesn't exist.

    Walks directories recursively. Symlinks are not followed — their target
    sizes are not included. Any per-file ``stat`` error is skipped so a
    single unreadable file doesn't abort the whole sizing pass.
    """
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _resolve_flat_artifacts(run_id: str | None) -> list[Path]:
    """Safely return a list with flat artifact files for ``run_id``.

    Returns:
        - A list with Path objects for ``<run_id>.log`` and ``<run_id>.done``
          files if they exist.
        - An empty list [] otherwise.

    This helper function allows legacy code to delegate flat file handling
    without raising exceptions on missing files.
    """
    if not run_id:
        return []
    
    flat_files = [
        EXECUTION_LOGS_DIR / f"{run_id}.log",
        EXECUTION_LOGS_DIR / f"{run_id}.done",
    ]
    
    # Filter to only return files that actually exist
    return [f for f in flat_files if f.exists()]


def _resolve_directory_artifact(run_id: str | None) -> list[Path]:
    """Safely return a list with a single Path object for the directory artifact.

    Returns:
        - A list with a single Path object [EXECUTION_LOGS_DIR / run_id] if run_id
          is not None or empty, and the directory exists.
        - An empty list [] otherwise.

    This helper function allows legacy code to delegate directory handling
    without raising exceptions on missing files.
    """
    if not run_id:
        return []
    
    path = EXECUTION_LOGS_DIR / run_id
    if not path.exists() or not path.is_dir():
        return []
    
    return [path]


def _candidate_paths(run_id: str | None) -> list[Path]:
    """Return the on-disk artifacts that belong to ``run_id``.

    The story spec describes ``execution_logs/<run_id>/`` directories, but
    the current executor writes flat ``<id>.log`` / ``<id>.done`` files.
    We return both so the cleanup works regardless of which layout is in
    play when it runs.

    Returns:
        A list of Path objects for the main run directory, .log file, and .done file.
        If no artifacts exist, returns an empty list.
    """
    if not run_id:
        return []
    
    # Delegate to helper functions for flat and directory artifacts
    flat_artifacts = _resolve_flat_artifacts(run_id)
    dir_artifact = _resolve_directory_artifact(run_id)
    
    # Concatenate results and return
    return flat_artifacts + dir_artifact


def _remove_artifacts(run_id: str | None, dry_run: bool) -> tuple[int, int]:
    """Remove execution-log artifacts for ``run_id``.

    Returns ``(dirs_removed, bytes_freed)``. ``dirs_removed`` counts every
    successfully-removed path (directory or file), not only directories —
    the name preserves the story's vocabulary. When ``dry_run`` is True the
    paths are sized but not touched.
    """
    removed = 0
    freed = 0
    paths_checked = 0
    for path in _candidate_paths(run_id):
        paths_checked += 1
        if not path.exists():
            if dry_run:
                log.info("Dry run: would skip missing path %s", path)
            continue
        size = _path_size_bytes(path)
        if dry_run:
            removed += 1
            freed += size
            log.info("Dry run: would delete %s (%d bytes)", path, size)
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            removed += 1
            freed += size
            log.info("Successful deletion of %s (%d bytes)", path, size)
        except (OSError, PermissionError) as exc:
            log.warning("Failed to remove %s: %s", path, exc)
    return removed, freed


def cleanup_old_runs(
    max_age_days: int = 30,
    keep_last_n: int = 200,
    dry_run: bool = False,
) -> dict[str, int]:
    """Prune ``executor_runs`` rows and matching log artifacts.

    Args:
        max_age_days: Delete rows whose ``started_at`` is older than this
            many days. Must be non-negative.
        keep_last_n: Always keep the N most recent rows (by ``started_at``
            DESC) regardless of age. Must be non-negative.
        dry_run: When True, report what would be deleted without touching
            SQLite or the filesystem.

    Returns:
        A dict with ``rows_deleted``, ``dirs_deleted``, ``bytes_freed``,
        and ``dry_run`` keys. Counts reflect what was (or would be)
        removed.
    """
    max_age_days = max(int(max_age_days), 0)
    keep_last_n = max(int(keep_last_n), 0)

    init_db()
    # Use a dedicated connection so we don't interfere with the per-thread
    # connection cache in executor_runs_db — this module may run from a
    # timer thread, the CLI, or the asyncio scheduler. Resolve DB_PATH
    # through the module so tests that monkeypatch it take effect here.
    conn = sqlite3.connect(str(executor_runs_db.DB_PATH), timeout=5)
    try:
        rows = _select_rows_to_delete(conn, max_age_days, keep_last_n)
        row_ids = [int(r["id"]) for r in rows]
        run_ids = [r["run_id"] for r in rows]

        dirs_deleted = 0
        bytes_freed = 0
        for run_id in run_ids:
            d, b = _remove_artifacts(run_id, dry_run=dry_run)
            dirs_deleted += d
            bytes_freed += b

        if dry_run or not row_ids:
            rows_deleted = len(row_ids)
        else:
            placeholders = ",".join("?" * len(row_ids))
            with conn:
                conn.execute(
                    "DELETE FROM executor_tool_calls "
                    f"WHERE run_id IN ({placeholders})",
                    row_ids,
                )
                cursor = conn.execute(
                    f"DELETE FROM executor_runs WHERE id IN ({placeholders})",
                    row_ids,
                )
                rows_deleted = cursor.rowcount or 0
    finally:
        conn.close()

    log.info(
        "cleanup_old_runs: %srows_deleted=%d dirs_deleted=%d bytes_freed=%d "
        "(max_age_days=%d keep_last_n=%d)",
        "DRY RUN " if dry_run else "",
        rows_deleted,
        dirs_deleted,
        bytes_freed,
        max_age_days,
        keep_last_n,
    )
    return {
        "rows_deleted": rows_deleted,
        "dirs_deleted": dirs_deleted,
        "bytes_freed": bytes_freed,
        "dry_run": 1 if dry_run else 0,
    }


# ---------------------------------------------------------------------------
# Scheduler — threading.Timer loop, runs at startup + every 6h by default
# ---------------------------------------------------------------------------


def _tick(interval_seconds: int, max_age_days: int, keep_last_n: int) -> None:
    """Run one cleanup cycle and schedule the next."""
    global _timer
    try:
        cleanup_old_runs(max_age_days=max_age_days, keep_last_n=keep_last_n)
    except Exception:
        log.exception("executor_runs_cleanup tick failed")
    with _timer_lock:
        if not _scheduler_running:
            return
        timer = threading.Timer(
            interval_seconds,
            _tick,
            args=(interval_seconds, max_age_days, keep_last_n),
        )
        timer.daemon = True
        _timer = timer
        timer.start()


def start_cleanup_scheduler(
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    max_age_days: int = 30,
    keep_last_n: int = 200,
    run_at_start: bool = True,
) -> None:
    """Kick off the cleanup timer loop.

    Runs :func:`cleanup_old_runs` immediately (when ``run_at_start``), then
    again every ``interval_seconds`` until :func:`stop_cleanup_scheduler`
    is called or the process exits (daemon timer).

    Safe to call once per process. Subsequent calls are no-ops.
    """
    global _scheduler_running, _timer
    with _timer_lock:
        if _scheduler_running:
            return
        _scheduler_running = True

    if run_at_start:
        try:
            cleanup_old_runs(max_age_days=max_age_days, keep_last_n=keep_last_n)
        except Exception:
            log.exception("executor_runs_cleanup startup run failed")

    with _timer_lock:
        timer = threading.Timer(
            interval_seconds,
            _tick,
            args=(interval_seconds, max_age_days, keep_last_n),
        )
        timer.daemon = True
        _timer = timer
        timer.start()


def stop_cleanup_scheduler() -> None:
    """Cancel the pending timer. Safe to call when not running."""
    global _scheduler_running, _timer
    with _timer_lock:
        _scheduler_running = False
        if _timer is not None:
            _timer.cancel()
            _timer = None


# ---------------------------------------------------------------------------
# CLI: ``python -m agent.executor_runs_cleanup [--dry-run] [--max-age-days N]``
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.executor_runs_cleanup",
        description=(
            "Prune executor_runs rows and execution_logs artifacts older "
            "than --max-age-days, while keeping the newest --keep-last-n."
        ),
    )
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--keep-last-n", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without touching SQLite or disk.",
    )
    args = parser.parse_args(argv)

    # Ensure the INFO summary line reaches stdout when run from a shell.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = cleanup_old_runs(
        max_age_days=args.max_age_days,
        keep_last_n=args.keep_last_n,
        dry_run=args.dry_run,
    )
    label = "Would delete" if args.dry_run else "Deleted"
    print(
        f"{label} {result['rows_deleted']} row(s) and "
        f"{result['dirs_deleted']} artifact(s); "
        f"{result['bytes_freed']} bytes freed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
