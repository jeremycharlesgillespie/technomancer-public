"""
Daily Rollup — compute per-day, per-project aggregate stats from
``executor_runs`` and upsert them into ``daily_stats``.

This is the primary throughput/cost rollup: one row per ``(date, project)``
covering ``shipped``, ``failed``, ``cost_usd``, ``p50_wall_s`` and
``p95_wall_s``. The other columns on ``daily_stats`` (LOC, first-attempt
success, splitter outcomes) belong to follow-up stories so that a broken
SQL query in the LOC rollup doesn't block the throughput numbers.

Entry points:

* :func:`compute_and_write` — library API, used by the scheduler and tests.
* ``python -m agent.daily_rollup --date YYYY-MM-DD --project TK`` — manual
  re-run for a specific day/project.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import daily_stats, executor_runs_db

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


def compute_and_write(date: str, project: str) -> dict[str, Any]:
    """Compute the primary daily rollup for ``(date, project)`` and upsert.

    Reads every row from ``executor_runs`` where ``jira_key`` starts with
    ``"{project}-"`` and ``date(started_at) = date``. Writes one row into
    ``daily_stats`` with ``shipped``, ``failed``, ``cost_usd``,
    ``p50_wall_s`` and ``p95_wall_s``. Other columns retain their zero
    defaults — later stories fill them in.

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

    stats_conn = daily_stats._get_conn()
    stats_conn.execute(
        """INSERT INTO daily_stats
             (date, project, shipped, failed, cost_usd,
              p50_wall_s, p95_wall_s)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(date, project) DO UPDATE SET
             shipped = excluded.shipped,
             failed = excluded.failed,
             cost_usd = excluded.cost_usd,
             p50_wall_s = excluded.p50_wall_s,
             p95_wall_s = excluded.p95_wall_s""",
        (date, project, shipped, failed, cost_usd, p50_wall_s, p95_wall_s),
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
    }
    log.info(
        "daily_rollup: date=%s project=%s shipped=%d failed=%d cost=%.4f "
        "p50_s=%.2f p95_s=%.2f",
        date, project, shipped, failed, cost_usd, p50_wall_s, p95_wall_s,
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
        f"p50={result['p50_wall_s']:.2f}s p95={result['p95_wall_s']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
