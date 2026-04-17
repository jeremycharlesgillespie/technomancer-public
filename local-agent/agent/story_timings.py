"""
Story Timings DB — SQLite persistence for story-execution phase timings.

Records a start/end timestamp for every major phase of a story execution
(queue-wait, worker-pickup, plan, code, test, deploy, ...) so the breakdown
dashboard can show where time is being spent without anyone having to grep
logs.

Typical usage — wrap each phase in the :func:`phase_timer` context manager:

    from agent.story_timings import phase_timer

    with phase_timer(
        run_id="20260417-173048-TK-572",
        story_id="TK-572",
        project="TK",
        phase="plan",
        metadata={"worker": "aim-1"},
    ):
        ...  # do the phase work

One row is written per ``with`` block. If the block exits cleanly, the row
has ``success=1``; if it raises, the row has ``success=0`` — the exception
is re-raised unchanged so callers see the original traceback.

``metadata`` is JSON-stringified at write time so callers can pass dicts,
lists, or any other JSON-serializable value without thinking about the
storage format.

Database: ``local-agent/data/story_timings.db``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "story_timings.db"

_local = threading.local()


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
    """Create the ``story_phase_timings`` table and its indexes if missing.

    Idempotent — safe to call on every write. Indexed columns match the
    three query shapes operators actually run from the dashboard:

    * ``WHERE story_id = ?`` — drill into one story's phase breakdown.
    * ``WHERE phase = ?`` — aggregate one phase across all stories.
    * ``ORDER BY started_at DESC`` — recent-first timelines.
    """
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS story_phase_timings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT,
            story_id    TEXT,
            project     TEXT,
            phase       TEXT,
            started_at  TEXT,
            ended_at    TEXT,
            duration_ms INTEGER,
            success     INTEGER,
            metadata    TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_story_phase_timings_story_id
        ON story_phase_timings (story_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_story_phase_timings_phase
        ON story_phase_timings (phase)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_story_phase_timings_started_at
        ON story_phase_timings (started_at)
    """)
    conn.commit()


def _serialize_metadata(metadata: Any) -> str | None:
    """JSON-stringify ``metadata`` for storage.

    ``None`` passes through as ``None`` so the column stores NULL rather
    than the literal string ``"null"``. Non-serializable values fall back
    to ``str(metadata)`` so instrumentation never breaks the caller.
    """
    if metadata is None:
        return None
    try:
        return json.dumps(metadata, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(metadata)


def record_phase(
    run_id: str | None,
    story_id: str | None,
    project: str | None,
    phase: str | None,
    started_at: str,
    ended_at: str,
    duration_ms: int,
    success: bool,
    metadata: Any = None,
) -> int:
    """Insert one finished phase row. Returns the inserted row id.

    Callers typically use :func:`phase_timer` instead of calling this
    directly — ``phase_timer`` handles the timing math and the exception
    path. ``record_phase`` exists so other modules (or tests) can write
    synthetic rows without spinning up a context manager.
    """
    init_db()
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO story_phase_timings
           (run_id, story_id, project, phase, started_at, ended_at,
            duration_ms, success, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            story_id,
            project,
            phase,
            started_at,
            ended_at,
            int(duration_ms),
            1 if success else 0,
            _serialize_metadata(metadata),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


@contextmanager
def phase_timer(
    run_id: str | None,
    story_id: str | None,
    project: str | None,
    phase: str | None,
    metadata: Any = None,
) -> Iterator[None]:
    """Record the wall-clock duration of a story-execution phase.

    Usage::

        with phase_timer(run_id, story_id, project, "plan"):
            ...  # work

    One row is written per ``with`` block. Successful exits record
    ``success=1``; exceptions record ``success=0`` and re-raise the
    original exception. ``metadata`` is JSON-stringified at write time.

    Timing uses :func:`time.monotonic` for duration_ms so a wall-clock
    jump (NTP slew, DST transition) can't produce negative durations.
    ``started_at`` / ``ended_at`` use :func:`datetime.now` so the rows
    remain human-readable next to everything else in the data dir.

    If the final write itself fails, the exception from the wrapped
    block (if any) is still re-raised — instrumentation failure must
    never mask a real error.
    """
    started_wall = datetime.now().isoformat()
    started_mono = time.monotonic()
    success = True
    try:
        yield
    except BaseException:
        success = False
        raise
    finally:
        ended_wall = datetime.now().isoformat()
        duration_ms = int((time.monotonic() - started_mono) * 1000)
        try:
            record_phase(
                run_id=run_id,
                story_id=story_id,
                project=project,
                phase=phase,
                started_at=started_wall,
                ended_at=ended_wall,
                duration_ms=duration_ms,
                success=success,
                metadata=metadata,
            )
        except sqlite3.Error:
            log.warning(
                "phase_timer: failed to record %s for story_id=%s",
                phase, story_id, exc_info=True,
            )


def get_phases_for_story(story_id: str) -> list[dict[str, Any]]:
    """Return every recorded phase for one story, oldest first."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, run_id, story_id, project, phase, started_at,
                  ended_at, duration_ms, success, metadata
           FROM story_phase_timings
           WHERE story_id = ?
           ORDER BY started_at ASC, id ASC""",
        (story_id,),
    ).fetchall()
    return [dict(r) for r in rows]
