"""
Daily Rollup — compute per-day, per-project aggregate stats from
``executor_runs`` and upsert them into ``daily_stats``.

This is the primary throughput/cost rollup: one row per ``(date, project)``
covering ``shipped``, ``failed``, ``cost_usd``, ``p50_wall_s``,
``p95_wall_s``, ``phase_timings_json`` (per-phase p50/p95 from
``story_phase_timings``), plus ``loc_added`` / ``loc_removed`` summed from
``git log --numstat`` over the day's AIM commits and
``first_attempt_success`` derived from ``executor_runs``. The remaining
columns on ``daily_stats`` (splitter outcomes) belong to follow-up stories.

Entry points:

* :func:`compute_and_write` — library API, used by the scheduler and tests.
* ``python -m agent.daily_rollup --date YYYY-MM-DD --project TK`` — manual
  re-run for a specific day/project.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import daily_stats, executor_runs_db, story_timings

# Repo root containing the .git directory. Tests monkeypatch this to a
# temp path that hosts a synthetic git repo so LOC computation can be
# exercised without touching the production history.
REPO_ROOT: Path = Path(__file__).parent.parent.parent

# Timeout for git log invocations. A healthy log call on this repo returns
# in well under a second; the timeout exists to bound the tail.
GIT_LOG_TIMEOUT_SECONDS: float = 30.0

log = logging.getLogger(__name__)

# executor_runs.status values that count as a successful ship vs. a failure.
# Anything outside these sets (``running``, ``queued``, ``done``, ...) is
# ignored for throughput counting so mid-run rows don't skew the numbers.
SHIPPED_STATUSES: frozenset[str] = frozenset({"success"})
FAILED_STATUSES: frozenset[str] = frozenset(
    {"failed", "error", "crashed", "killed", "timeout"}
)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy-style).

    ``pct`` is on the 0-100 scale. Returns 0.0 for an empty list so callers
    can assign the result to a ``NOT NULL DEFAULT 0.0`` column without a
    None guard.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    position = (pct / 100.0) * (n - 1)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, n - 1)
    frac = position - lower_idx
    return float(
        sorted_vals[lower_idx]
        + frac * (sorted_vals[upper_idx] - sorted_vals[lower_idx])
    )


def _compute_phase_timings_json(date: str, project: str) -> str | None:
    """Compute per-phase p50/p95 for ``(date, project)`` from story_phase_timings.

    Groups every finished phase row matching ``project`` and
    ``date(started_at) == date`` by ``phase`` and returns a JSON blob
    mapping each phase name to ``{"count", "p50_ms", "p95_ms"}``.

    Returns ``None`` when no phase rows exist for the day so the column
    stores SQL NULL rather than an empty-object string — the AC requires
    "missing phase data ... writes null".
    """
    story_timings.init_db()
    conn = story_timings._get_conn()
    rows = conn.execute(
        """SELECT phase, duration_ms
           FROM story_phase_timings
           WHERE project = ?
             AND started_at IS NOT NULL
             AND date(started_at) = ?
             AND duration_ms IS NOT NULL
             AND phase IS NOT NULL""",
        (project, date),
    ).fetchall()

    if not rows:
        return None

    by_phase: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(int(r["duration_ms"]))

    result: dict[str, dict[str, int]] = {}
    for phase, values in by_phase.items():
        values.sort()
        result[phase] = {
            "count": len(values),
            "p50_ms": story_timings._percentile(values, 50),
            "p95_ms": story_timings._percentile(values, 95),
        }
    return json.dumps(result, sort_keys=True)


def _git_loc_counts(
    date: str, project: str, repo_root: Path | None = None
) -> tuple[int, int]:
    """Sum LOC added / removed from AIM commits on ``date`` for ``project``.

    Runs ``git log --no-merges -E --grep='^\\[{project}-[0-9]+\\]'
    --since=<date>T00:00:00 --until=<next>T00:00:00 --numstat --format=``
    from ``repo_root`` and sums the per-file numstat output. Binary files
    (numstat shows ``-`` for both columns) are skipped. Merge commits are
    excluded via ``--no-merges`` because they don't emit numstat data and
    would otherwise depend on ``-m`` for a combined diff.

    The subject-regex matches any commit whose first line starts with
    ``[PROJECT-N]`` — the convention every AIM-authored commit follows,
    whether it's a branch commit or a deploy commit.

    Args:
        date: Calendar day in ``YYYY-MM-DD`` local time.
        project: Jira project key prefix like ``"TK"`` or ``"FA"``.
        repo_root: Directory containing ``.git``. Defaults to
            :data:`REPO_ROOT`. Tests override to point at a fake repo.

    Returns:
        ``(loc_added, loc_removed)``. Both ``0`` when the repo is missing,
        git is unavailable, the command times out, or no matching commits
        exist — the LOC rollup must never be the reason the throughput
        rollup fails.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    if not root.exists():
        return (0, 0)

    try:
        start_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        log.warning("daily_rollup: invalid date %r for git LOC", date)
        return (0, 0)
    end_dt = start_dt + timedelta(days=1)

    try:
        result = subprocess.run(
            [
                "git", "log",
                "--no-merges",
                "-E",
                f"--grep=^\\[{project}-[0-9]+\\]",
                f"--since={start_dt.strftime('%Y-%m-%dT%H:%M:%S')}",
                f"--until={end_dt.strftime('%Y-%m-%dT%H:%M:%S')}",
                "--numstat",
                "--format=",
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=GIT_LOG_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("daily_rollup: git log failed (%s)", exc)
        return (0, 0)

    if result.returncode != 0:
        log.debug(
            "daily_rollup: git log exit=%d stderr=%s",
            result.returncode, result.stderr.strip(),
        )
        return (0, 0)

    added = 0
    removed = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a_str, r_str, _path = parts
        # Binary files show "-\t-\t<path>" — skip, not a line count.
        if a_str == "-" or r_str == "-":
            continue
        try:
            added += int(a_str)
            removed += int(r_str)
        except ValueError:
            continue
    return (added, removed)


def _first_attempt_success_count(date: str, project: str) -> int:
    """Count stories whose chronologically first run on ``date`` succeeded.

    Groups ``executor_runs`` rows by ``jira_key`` within the ``(date,
    project)`` scope, takes the earliest ``started_at`` per group, and
    counts those whose status is in :data:`SHIPPED_STATUSES`. Later retry
    runs for the same key are deliberately ignored — a story only counts
    once, and only if it shipped on the first try.

    The ``ORDER BY id ASC`` tiebreaker matters when two runs share the
    same ``started_at`` string (rare, but possible when two inserts happen
    in the same tick): it keeps the earlier-inserted row as the "first
    attempt" so the result is deterministic.
    """
    conn = executor_runs_db._get_conn()
    rows = conn.execute(
        """SELECT jira_key, status
           FROM executor_runs
           WHERE jira_key LIKE ?
             AND started_at IS NOT NULL
             AND date(started_at) = ?
           ORDER BY jira_key ASC, started_at ASC, id ASC""",
        (f"{project}-%", date),
    ).fetchall()

    first_status_per_key: dict[str, str | None] = {}
    for row in rows:
        jk = row["jira_key"]
        if jk in first_status_per_key:
            continue
        first_status_per_key[jk] = row["status"]

    return sum(
        1 for status in first_status_per_key.values()
        if (status or "") in SHIPPED_STATUSES
    )


def compute_and_write(date: str, project: str) -> dict[str, Any]:
    """Compute the primary daily rollup for ``(date, project)`` and upsert.

    Reads every row from ``executor_runs`` where ``jira_key`` starts with
    ``"{project}-"`` and ``date(started_at) = date``. Writes one row into
    ``daily_stats`` with ``shipped``, ``failed``, ``cost_usd``,
    ``p50_wall_s``, ``p95_wall_s``, ``phase_timings_json``, plus
    ``loc_added`` / ``loc_removed`` (summed from ``git log --numstat``
    over AIM commits on this day) and ``first_attempt_success`` (count of
    jira_keys whose first run of the day succeeded). The remaining splitter
    columns retain their zero defaults — those belong to later stories.

    Re-running for the same ``(date, project)`` updates the existing row
    in place (ON CONFLICT UPSERT), so callers can safely re-run the rollup
    after a late-arriving executor_runs row without creating duplicates.

    Args:
        date: Calendar day in ``YYYY-MM-DD`` (local time, matches the
            format ``record_run`` writes via ``datetime.now().isoformat()``).
        project: Jira project key prefix such as ``"TK"`` or ``"FA"``.

    Returns:
        A dict of the computed values — useful for logs, CLI output, and
        assertions in tests.
    """
    executor_runs_db.init_db()
    daily_stats.init_db()

    exec_conn = executor_runs_db._get_conn()
    rows = exec_conn.execute(
        """SELECT status, cost_usd, duration_ms
           FROM executor_runs
           WHERE jira_key LIKE ?
             AND started_at IS NOT NULL
             AND date(started_at) = ?""",
        (f"{project}-%", date),
    ).fetchall()

    shipped = sum(1 for r in rows if (r["status"] or "") in SHIPPED_STATUSES)
    failed = sum(1 for r in rows if (r["status"] or "") in FAILED_STATUSES)
    cost_usd = float(sum((r["cost_usd"] or 0.0) for r in rows))
    durations_s = [
        float(r["duration_ms"]) / 1000.0
        for r in rows
        if r["duration_ms"] is not None
    ]
    p50_wall_s = _percentile(durations_s, 50)
    p95_wall_s = _percentile(durations_s, 95)

    phase_timings_json = _compute_phase_timings_json(date, project)
    loc_added, loc_removed = _git_loc_counts(date, project)
    first_attempt_success = _first_attempt_success_count(date, project)

    stats_conn = daily_stats._get_conn()
    stats_conn.execute(
        """INSERT INTO daily_stats
             (date, project, shipped, failed, cost_usd,
              p50_wall_s, p95_wall_s, phase_timings_json,
              loc_added, loc_removed, first_attempt_success)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(date, project) DO UPDATE SET
             shipped = excluded.shipped,
             failed = excluded.failed,
             cost_usd = excluded.cost_usd,
             p50_wall_s = excluded.p50_wall_s,
             p95_wall_s = excluded.p95_wall_s,
             phase_timings_json = excluded.phase_timings_json,
             loc_added = excluded.loc_added,
             loc_removed = excluded.loc_removed,
             first_attempt_success = excluded.first_attempt_success""",
        (
            date, project, shipped, failed, cost_usd,
            p50_wall_s, p95_wall_s, phase_timings_json,
            loc_added, loc_removed, first_attempt_success,
        ),
    )
    stats_conn.commit()

    result = {
        "date": date,
        "project": project,
        "shipped": shipped,
        "failed": failed,
        "cost_usd": cost_usd,
        "p50_wall_s": p50_wall_s,
        "p95_wall_s": p95_wall_s,
        "loc_added": loc_added,
        "loc_removed": loc_removed,
        "first_attempt_success": first_attempt_success,
    }
    log.info(
        "daily_rollup: date=%s project=%s shipped=%d failed=%d cost=%.4f "
        "p50_s=%.2f p95_s=%.2f loc=+%d/-%d first_attempt=%d",
        date, project, shipped, failed, cost_usd, p50_wall_s, p95_wall_s,
        loc_added, loc_removed, first_attempt_success,
    )
    return result


def _main(argv: list[str]) -> int:
    """CLI entry point — ``python -m agent.daily_rollup --date ... --project ...``."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.daily_rollup",
        description=(
            "Compute the daily throughput/cost rollup for a (date, project) "
            "pair and upsert one row into daily_stats."
        ),
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Calendar day to roll up (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Jira project key prefix, e.g. TK or FA.",
    )
    ns = parser.parse_args(argv)

    result = compute_and_write(ns.date, ns.project)
    print(
        f"{result['date']} {result['project']}: "
        f"shipped={result['shipped']} failed={result['failed']} "
        f"cost=${result['cost_usd']:.4f} "
        f"p50={result['p50_wall_s']:.2f}s p95={result['p95_wall_s']:.2f}s "
        f"loc=+{result['loc_added']}/-{result['loc_removed']} "
        f"first_attempt={result['first_attempt_success']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
