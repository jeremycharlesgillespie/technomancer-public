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
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
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


# =============================================================================
# AGGREGATION HELPERS — feed the /performance/breakdown dashboard.
# =============================================================================


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Inclusive linear-interpolation percentile on a pre-sorted list.

    Returns ``0`` for an empty list; returns the lone element when the list
    has exactly one value. SQLite has no ``percentile_cont`` so the dashboard
    computes percentiles in Python — this helper keeps the formula in one
    place so tests can pin its contract.
    """
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return int(sorted_values[f])
    return int(round(sorted_values[f] * (c - k) + sorted_values[c] * (k - f)))


def get_recent_runs_breakdown(limit: int = 50) -> list[dict[str, Any]]:
    """Return up to ``limit`` recent runs newest-first with their phase rows.

    Each run dict has ``run_id``, ``story_id``, ``project``, ``started_at``
    (earliest phase start), ``total_ms`` (sum of phase durations),
    ``success`` (True only if every phase succeeded), and ``phases`` — a
    list ordered by ``started_at`` ASC of
    ``{phase, duration_ms, started_at, ended_at, success}``.

    Rows without a ``run_id`` are skipped because the stacked-bar view uses
    ``run_id`` as the bar identity.
    """
    init_db()
    conn = _get_conn()
    limit = max(1, int(limit))

    run_id_rows = conn.execute(
        """SELECT run_id, MAX(started_at) AS last_ts
           FROM story_phase_timings
           WHERE run_id IS NOT NULL
           GROUP BY run_id
           ORDER BY last_ts DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    run_ids = [r["run_id"] for r in run_id_rows]
    if not run_ids:
        return []

    placeholders = ",".join("?" * len(run_ids))
    phase_rows = conn.execute(
        f"""SELECT run_id, story_id, project, phase, started_at, ended_at,
                   duration_ms, success
            FROM story_phase_timings
            WHERE run_id IN ({placeholders})
            ORDER BY started_at ASC, id ASC""",
        run_ids,
    ).fetchall()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in phase_rows:
        grouped[r["run_id"]].append(dict(r))

    out: list[dict[str, Any]] = []
    for rid in run_ids:
        phases = grouped.get(rid, [])
        if not phases:
            continue
        started_candidates = [p["started_at"] for p in phases if p["started_at"]]
        out.append({
            "run_id": rid,
            "story_id": phases[0]["story_id"],
            "project": phases[0]["project"],
            "started_at": min(started_candidates) if started_candidates else None,
            "total_ms": sum(int(p["duration_ms"] or 0) for p in phases),
            "success": all(bool(p["success"]) for p in phases),
            "phases": [
                {
                    "phase": p["phase"],
                    "duration_ms": int(p["duration_ms"] or 0),
                    "started_at": p["started_at"],
                    "ended_at": p["ended_at"],
                    "success": bool(p["success"]),
                }
                for p in phases
            ],
        })
    return out


def get_phase_percentiles(days: int = 7) -> list[dict[str, Any]]:
    """Aggregate phase durations over the last ``days`` days.

    Returns one row per phase with ``count``, ``p50_ms``, ``p95_ms``,
    ``p99_ms`` — sorted by ``p50_ms`` descending so the slowest phases
    surface first in the breakdown table.
    """
    init_db()
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT phase, duration_ms
           FROM story_phase_timings
           WHERE started_at >= ?
             AND phase IS NOT NULL
             AND duration_ms IS NOT NULL""",
        (cutoff,),
    ).fetchall()

    by_phase: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(int(r["duration_ms"]))

    out: list[dict[str, Any]] = []
    for phase, values in by_phase.items():
        values.sort()
        out.append({
            "phase": phase,
            "count": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "p99_ms": _percentile(values, 99),
        })
    out.sort(key=lambda x: x["p50_ms"], reverse=True)
    return out


def get_phase_p50(phase: str, days: int = 1) -> int:
    """Return the p50 ``duration_ms`` for one phase over the last N days.

    Returns ``0`` when no rows match. Used by the hub card to show a
    freshness badge without pulling the full percentile table.
    """
    init_db()
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT duration_ms FROM story_phase_timings
           WHERE phase = ?
             AND started_at >= ?
             AND duration_ms IS NOT NULL""",
        (phase, cutoff),
    ).fetchall()
    values = sorted(int(r["duration_ms"]) for r in rows)
    return _percentile(values, 50)


def get_idle_gaps(days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
    """Top-N biggest idle gaps between consecutive phases within a run.

    For each run_id, phases are ordered by ``started_at``; each gap is
    ``next.started_at - prev.ended_at`` in milliseconds. Negative or zero
    gaps (overlapping or back-to-back phases) are dropped. Unparseable
    timestamps are skipped rather than raising — instrumentation failure
    must never break the dashboard.

    Returned dicts include ``from_phase``, ``to_phase``, ``from_ended_at``,
    ``to_started_at``, ``gap_ms``, plus ``run_id``, ``story_id``, ``project``
    so the UI can link back to the relevant run.
    """
    init_db()
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT run_id, story_id, project, phase, started_at, ended_at
           FROM story_phase_timings
           WHERE run_id IS NOT NULL
             AND started_at >= ?
           ORDER BY run_id, started_at ASC, id ASC""",
        (cutoff,),
    ).fetchall()

    by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)

    gaps: list[dict[str, Any]] = []
    for rid, phases in by_run.items():
        for i in range(len(phases) - 1):
            prev, curr = phases[i], phases[i + 1]
            if not prev["ended_at"] or not curr["started_at"]:
                continue
            try:
                prev_end = datetime.fromisoformat(prev["ended_at"])
                curr_start = datetime.fromisoformat(curr["started_at"])
            except (TypeError, ValueError):
                continue
            gap_ms = int((curr_start - prev_end).total_seconds() * 1000)
            if gap_ms <= 0:
                continue
            gaps.append({
                "run_id": rid,
                "story_id": curr["story_id"],
                "project": curr["project"],
                "from_phase": prev["phase"],
                "to_phase": curr["phase"],
                "from_ended_at": prev["ended_at"],
                "to_started_at": curr["started_at"],
                "gap_ms": gap_ms,
            })
    gaps.sort(key=lambda g: g["gap_ms"], reverse=True)
    return gaps[: max(1, int(limit))]
