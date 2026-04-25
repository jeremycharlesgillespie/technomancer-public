"""Tests for agent.ab_schema — `ab_test_runs` and `ab_test_pairs` tables.

The A/B harness writes its run-level and pair-level data into the same
SQLite file as AIV (``data/aiv.db``). The schema is created idempotently
by :func:`agent.ab_schema.init_ab_db`. These tests cover:

- Both tables are created on a fresh DB.
- Calling ``init_ab_db()`` twice is a no-op (idempotent).
- Column names match the published schema (so SELECT * stays stable).
- The ``aiv_schema.init_db()`` entry point also triggers
  ``init_ab_db()`` (auto-wiring via :mod:`agent.aiv_schema`).
"""

from __future__ import annotations

import pytest

from agent import aiv_schema, ab_schema


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


def _table_columns(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def test_init_ab_db_creates_both_tables() -> None:
    ab_schema.init_ab_db()
    conn = aiv_schema._get_conn()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "ab_test_runs" in tables
    assert "ab_test_pairs" in tables


def test_init_ab_db_is_idempotent() -> None:
    ab_schema.init_ab_db()
    ab_schema.init_ab_db()  # second call must not raise
    conn = aiv_schema._get_conn()
    cols_runs = _table_columns(conn, "ab_test_runs")
    assert "run_id" in cols_runs


def test_ab_test_runs_columns() -> None:
    ab_schema.init_ab_db()
    conn = aiv_schema._get_conn()
    cols = _table_columns(conn, "ab_test_runs")
    expected = {
        "run_id",
        "story_key",
        "model",
        "model_label",
        "branch_name",
        "commit_sha",
        "started_at",
        "ended_at",
        "status",
        "failure_log",
        "meets_requirements",
        "code_quality",
        "test_quality",
        "security_safety",
        "scope_discipline",
        "edge_cases",
        "product_impact",
        "overall_score",
        "red_flags_json",
        "reasoning_json",
        "scoring_error",
    }
    missing = expected - set(cols)
    assert not missing, f"missing columns: {missing}"


def test_ab_test_pairs_columns() -> None:
    ab_schema.init_ab_db()
    conn = aiv_schema._get_conn()
    cols = _table_columns(conn, "ab_test_pairs")
    expected = {
        "pair_id",
        "story_key",
        "model_a_run_id",
        "model_b_run_id",
        "comparison_winner",
        "comparison_reasoning",
        "delta_axes_json",
        "merged_run_id",
        "comparison_error",
        "created_at",
    }
    missing = expected - set(cols)
    assert not missing, f"missing columns: {missing}"


def test_aiv_init_db_triggers_ab_schema() -> None:
    """``aiv_schema.init_db()`` should also create the A/B tables."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "ab_test_runs" in tables
    assert "ab_test_pairs" in tables
