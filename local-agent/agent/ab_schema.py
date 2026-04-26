"""A/B Schema — SQLite tables for the AIW A/B model-comparison harness.

Two tables live alongside the existing AIV tables in ``data/aiv.db``:

``ab_test_runs`` — one row per *attempt* by a single model on a single
story. Captures performance timestamps, the resulting branch + commit
SHA, the failure log on failure, and the seven AIV scoring axes once
``aiv.scorer.score`` has run on the diff.

``ab_test_pairs`` — one row per pair of attempts (always two model runs:
A and B). Records the comparison verdict, the per-axis delta, and which
run was actually merged to main.

:func:`init_ab_db` is idempotent — ``CREATE TABLE IF NOT EXISTS``. It is
called from :func:`agent.aiv_schema.init_db` so any AIV-touching path
also brings the A/B tables into existence.
"""

from __future__ import annotations

import logging
import sqlite3

from agent.aiv_schema import _get_conn

log = logging.getLogger(__name__)


# Allowed comparison verdicts. The compare module guarantees one of these
# values is written. Persisted to ab_test_pairs.comparison_winner.
COMPARISON_WINNERS: tuple[str, ...] = (
    "model_a",
    "model_b",
    "tie",
    "both_failed",
)

# Allowed run statuses. Lifecycle is "running" -> "success" | "failed".
RUN_STATUSES: tuple[str, ...] = ("running", "success", "failed")


def init_ab_db() -> None:
    """Create the A/B test tables if they don't already exist.

    Idempotent. Safe to call repeatedly. Wired into
    :func:`agent.aiv_schema.init_db` so callers that already initialise
    the AIV DB don't need a second call.
    """
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ab_test_runs (
            run_id              TEXT PRIMARY KEY,
            story_key           TEXT NOT NULL,
            model               TEXT NOT NULL,
            model_label         TEXT NOT NULL,
            branch_name         TEXT,
            commit_sha          TEXT,
            started_at          TEXT NOT NULL,
            ended_at            TEXT,
            status              TEXT NOT NULL,
            failure_log         TEXT,
            meets_requirements  INTEGER,
            code_quality        INTEGER,
            test_quality        INTEGER,
            security_safety     INTEGER,
            scope_discipline    INTEGER,
            edge_cases          INTEGER,
            product_impact      INTEGER,
            overall_score       REAL,
            red_flags_json      TEXT,
            reasoning_json      TEXT,
            scoring_error       TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ab_runs_story
        ON ab_test_runs (story_key, started_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ab_test_pairs (
            pair_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            story_key            TEXT NOT NULL,
            model_a_run_id       TEXT NOT NULL REFERENCES ab_test_runs(run_id),
            model_b_run_id       TEXT NOT NULL REFERENCES ab_test_runs(run_id),
            comparison_winner    TEXT,
            comparison_reasoning TEXT,
            delta_axes_json      TEXT,
            merged_run_id        TEXT,
            comparison_error     TEXT,
            created_at           TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ab_pairs_story
        ON ab_test_pairs (story_key, created_at DESC)
        """
    )
    # Create the aiv_enqueue_failures table for tracking failed AIV enqueues
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aiv_enqueue_failures (
            story_key           TEXT NOT NULL,
            attempted_at        TEXT NOT NULL,
            error               TEXT NOT NULL,
            merge_commit_sha    TEXT,
            PRIMARY KEY (story_key, attempted_at)
        )
        """
    )
    conn.commit()
