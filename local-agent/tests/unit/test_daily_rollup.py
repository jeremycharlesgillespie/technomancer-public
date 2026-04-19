"""
Tests for agent/daily_rollup.py — verifies that:

* ``subprocess`` is importable from the module (no import-time side-effects).
* ``GIT_LOG_TIMEOUT_SECONDS`` is still 30.0 and is not overridden by pytest
  timeout settings.
* Core computation helpers (_percentile, _git_loc_counts,
  _first_attempt_success_count, _splitter_child_counts) behave correctly.
* ``compute_and_write`` round-trips through in-memory SQLite without errors.
"""

from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite connections that replace the on-disk DBs
# ---------------------------------------------------------------------------

def _make_exec_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE executor_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            jira_key    TEXT,
            branch      TEXT,
            started_at  TEXT,
            ended_at    TEXT,
            duration_ms INTEGER,
            cost_usd    REAL,
            status      TEXT,
            exit_code   INTEGER,
            tests_passed INTEGER,
            deployed    INTEGER,
            trace_id    TEXT
        )
    """)
    conn.commit()
    return conn


def _make_stats_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE daily_stats (
            date                   TEXT    NOT NULL,
            project                TEXT    NOT NULL,
            shipped                INTEGER NOT NULL DEFAULT 0,
            failed                 INTEGER NOT NULL DEFAULT 0,
            split_children         INTEGER NOT NULL DEFAULT 0,
            cost_usd               REAL    NOT NULL DEFAULT 0.0,
            p50_wall_s             REAL    NOT NULL DEFAULT 0.0,
            p95_wall_s             REAL    NOT NULL DEFAULT 0.0,
            loc_added              INTEGER NOT NULL DEFAULT 0,
            loc_removed            INTEGER NOT NULL DEFAULT 0,
            first_attempt_success  INTEGER NOT NULL DEFAULT 0,
            splitter_child_success INTEGER,
            splitter_child_fail    INTEGER,
            phase_timings_json     TEXT,
            PRIMARY KEY (date, project)
        )
    """)
    conn.commit()
    return conn


def _make_timings_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE story_phase_timings (
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
    conn.commit()
    return conn


@pytest.fixture()
def in_memory_dbs():
    """Patch all three DB _get_conn functions to use in-memory SQLite."""
    exec_conn = _make_exec_conn()
    stats_conn = _make_stats_conn()
    timings_conn = _make_timings_conn()

    with (
        patch("agent.executor_runs_db._get_conn", return_value=exec_conn),
        patch("agent.daily_stats._get_conn", return_value=stats_conn),
        patch("agent.story_timings._get_conn", return_value=timings_conn),
        patch("agent.executor_runs_db.init_db"),
        patch("agent.daily_stats.init_db"),
        patch("agent.story_timings.init_db"),
    ):
        yield {
            "exec": exec_conn,
            "stats": stats_conn,
            "timings": timings_conn,
        }


# ---------------------------------------------------------------------------
# Import and constant checks (the acceptance criteria)
# ---------------------------------------------------------------------------

def test_subprocess_importable():
    """daily_rollup must import subprocess at module level without errors."""
    import agent.daily_rollup as dr
    import sys
    assert "subprocess" in sys.modules
    # Verify we can access subprocess through the module's namespace
    assert dr.subprocess is subprocess


def test_git_log_timeout_is_30():
    """GIT_LOG_TIMEOUT_SECONDS must remain 30.0 — not affected by pytest settings."""
    import agent.daily_rollup as dr
    assert dr.GIT_LOG_TIMEOUT_SECONDS == 30.0


def test_git_log_timeout_type():
    """GIT_LOG_TIMEOUT_SECONDS must be a float, not int."""
    import agent.daily_rollup as dr
    assert isinstance(dr.GIT_LOG_TIMEOUT_SECONDS, float)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

def test_percentile_empty():
    from agent.daily_rollup import _percentile
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    from agent.daily_rollup import _percentile
    assert _percentile([42.0], 50) == 42.0


def test_percentile_p50():
    from agent.daily_rollup import _percentile
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 50) == 3.0


def test_percentile_p95():
    from agent.daily_rollup import _percentile
    vals = list(range(1, 101))  # 1..100
    result = _percentile([float(v) for v in vals], 95)
    assert 95.0 <= result <= 96.0


def test_percentile_p100():
    from agent.daily_rollup import _percentile
    assert _percentile([10.0, 20.0, 30.0], 100) == 30.0


# ---------------------------------------------------------------------------
# _git_loc_counts
# ---------------------------------------------------------------------------

def test_git_loc_counts_missing_repo(tmp_path):
    """Returns (0, 0) when repo_root does not exist."""
    from agent.daily_rollup import _git_loc_counts
    nonexistent = tmp_path / "no_such_dir"
    assert _git_loc_counts("2026-04-19", "TK", repo_root=nonexistent) == (0, 0)


def test_git_loc_counts_invalid_date(tmp_path):
    """Returns (0, 0) on a malformed date string."""
    from agent.daily_rollup import _git_loc_counts
    assert _git_loc_counts("not-a-date", "TK", repo_root=tmp_path) == (0, 0)


def test_git_loc_counts_uses_timeout(tmp_path):
    """subprocess.run must be called with timeout=GIT_LOG_TIMEOUT_SECONDS."""
    from agent.daily_rollup import _git_loc_counts, GIT_LOG_TIMEOUT_SECONDS

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("agent.daily_rollup.subprocess.run", return_value=mock_result) as mock_run:
        _git_loc_counts("2026-04-19", "TK", repo_root=tmp_path)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == GIT_LOG_TIMEOUT_SECONDS


def test_git_loc_counts_parses_numstat(tmp_path):
    """Correctly sums added/removed lines from --numstat output."""
    from agent.daily_rollup import _git_loc_counts

    numstat_output = (
        "10\t5\tagent/foo.py\n"
        "3\t2\tagent/bar.py\n"
        "-\t-\tagent/binary.bin\n"  # binary — skip
        "\n"
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = numstat_output
    mock_result.stderr = ""

    with patch("agent.daily_rollup.subprocess.run", return_value=mock_result):
        added, removed = _git_loc_counts("2026-04-19", "TK", repo_root=tmp_path)

    assert added == 13
    assert removed == 7


def test_git_loc_counts_timeout_exception(tmp_path):
    """Returns (0, 0) when subprocess times out."""
    from agent.daily_rollup import _git_loc_counts

    with patch(
        "agent.daily_rollup.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30),
    ):
        assert _git_loc_counts("2026-04-19", "TK", repo_root=tmp_path) == (0, 0)


def test_git_loc_counts_nonzero_exit(tmp_path):
    """Returns (0, 0) when git exits non-zero."""
    from agent.daily_rollup import _git_loc_counts

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "not a git repo"

    with patch("agent.daily_rollup.subprocess.run", return_value=mock_result):
        assert _git_loc_counts("2026-04-19", "TK", repo_root=tmp_path) == (0, 0)


# ---------------------------------------------------------------------------
# _first_attempt_success_count
# ---------------------------------------------------------------------------

def test_first_attempt_success_count_empty(in_memory_dbs):
    from agent.daily_rollup import _first_attempt_success_count
    result = _first_attempt_success_count("2026-04-19", "TK")
    assert result == 0


def test_first_attempt_success_count_one_success(in_memory_dbs):
    from agent.daily_rollup import _first_attempt_success_count
    conn = in_memory_dbs["exec"]
    conn.execute(
        "INSERT INTO executor_runs (jira_key, started_at, status) VALUES (?,?,?)",
        ("TK-100", "2026-04-19T10:00:00", "success"),
    )
    conn.commit()
    assert _first_attempt_success_count("2026-04-19", "TK") == 1


def test_first_attempt_success_counts_only_first_run(in_memory_dbs):
    """A story that failed first and succeeded on retry counts as 0."""
    from agent.daily_rollup import _first_attempt_success_count
    conn = in_memory_dbs["exec"]
    conn.executemany(
        "INSERT INTO executor_runs (jira_key, started_at, status) VALUES (?,?,?)",
        [
            ("TK-101", "2026-04-19T09:00:00", "failed"),
            ("TK-101", "2026-04-19T10:00:00", "success"),
        ],
    )
    conn.commit()
    assert _first_attempt_success_count("2026-04-19", "TK") == 0


def test_first_attempt_success_ignores_other_project(in_memory_dbs):
    from agent.daily_rollup import _first_attempt_success_count
    conn = in_memory_dbs["exec"]
    conn.execute(
        "INSERT INTO executor_runs (jira_key, started_at, status) VALUES (?,?,?)",
        ("FA-200", "2026-04-19T10:00:00", "success"),
    )
    conn.commit()
    assert _first_attempt_success_count("2026-04-19", "TK") == 0


# ---------------------------------------------------------------------------
# _splitter_child_counts
# ---------------------------------------------------------------------------

def test_splitter_child_counts_jira_not_configured():
    """Returns None when Jira is not configured."""
    from agent.daily_rollup import _splitter_child_counts
    with patch("agent.daily_rollup.is_jira_configured", return_value=False):
        assert _splitter_child_counts("2026-04-19", "TK") is None


def test_splitter_child_counts_invalid_date():
    from agent.daily_rollup import _splitter_child_counts
    with patch("agent.daily_rollup.is_jira_configured", return_value=True):
        assert _splitter_child_counts("bad-date", "TK") is None


def test_splitter_child_counts_query_failure():
    """Returns None when the Jira search returns (None, status)."""
    from agent.daily_rollup import _splitter_child_counts
    with (
        patch("agent.daily_rollup.is_jira_configured", return_value=True),
        patch("agent.daily_rollup._paginated_search", return_value=(None, 500)),
    ):
        assert _splitter_child_counts("2026-04-19", "TK") is None


def test_splitter_child_counts_done_and_failed():
    from agent.daily_rollup import _splitter_child_counts
    issues = [
        {"fields": {"status": {"name": "Done"}}},
        {"fields": {"status": {"name": "Done"}}},
        {"fields": {"status": {"name": "Failed"}}},
        {"fields": {"status": {"name": "In Progress"}}},  # not counted
    ]
    with (
        patch("agent.daily_rollup.is_jira_configured", return_value=True),
        patch("agent.daily_rollup._paginated_search", return_value=(issues, 200)),
    ):
        result = _splitter_child_counts("2026-04-19", "TK")
    assert result == (2, 1)


# ---------------------------------------------------------------------------
# compute_and_write — end-to-end
# ---------------------------------------------------------------------------

def test_compute_and_write_empty_day(in_memory_dbs):
    """compute_and_write returns zero stats when no executor runs exist."""
    from agent.daily_rollup import compute_and_write

    with (
        patch("agent.daily_rollup.subprocess.run", return_value=_git_ok("")),
        patch("agent.daily_rollup.is_jira_configured", return_value=False),
    ):
        result = compute_and_write("2026-04-19", "TK")

    assert result["shipped"] == 0
    assert result["failed"] == 0
    assert result["cost_usd"] == 0.0
    assert result["p50_wall_s"] == 0.0
    assert result["p95_wall_s"] == 0.0
    assert result["loc_added"] == 0
    assert result["loc_removed"] == 0
    assert result["first_attempt_success"] == 0
    assert result["splitter_child_success"] is None
    assert result["splitter_child_fail"] is None


def test_compute_and_write_with_runs(in_memory_dbs):
    """compute_and_write aggregates shipped/failed/cost/duration correctly."""
    from agent.daily_rollup import compute_and_write
    conn = in_memory_dbs["exec"]
    conn.executemany(
        "INSERT INTO executor_runs (jira_key, started_at, status, cost_usd, duration_ms) "
        "VALUES (?,?,?,?,?)",
        [
            ("TK-1", "2026-04-19T08:00:00", "success", 0.10, 60000),
            ("TK-2", "2026-04-19T09:00:00", "failed", 0.05, 30000),
            ("TK-3", "2026-04-19T10:00:00", "success", 0.20, 90000),
        ],
    )
    conn.commit()

    with (
        patch("agent.daily_rollup.subprocess.run", return_value=_git_ok("")),
        patch("agent.daily_rollup.is_jira_configured", return_value=False),
    ):
        result = compute_and_write("2026-04-19", "TK")

    assert result["shipped"] == 2
    assert result["failed"] == 1
    assert abs(result["cost_usd"] - 0.35) < 1e-9
    assert result["p50_wall_s"] == 60.0
    assert result["first_attempt_success"] == 2  # TK-1 and TK-3 first-attempt success


def test_compute_and_write_upsert(in_memory_dbs):
    """Re-running for the same (date, project) updates rather than duplicates."""
    from agent.daily_rollup import compute_and_write

    with (
        patch("agent.daily_rollup.subprocess.run", return_value=_git_ok("")),
        patch("agent.daily_rollup.is_jira_configured", return_value=False),
    ):
        compute_and_write("2026-04-19", "TK")
        compute_and_write("2026-04-19", "TK")

    rows = in_memory_dbs["stats"].execute(
        "SELECT COUNT(*) as n FROM daily_stats WHERE date='2026-04-19' AND project='TK'"
    ).fetchone()
    assert rows["n"] == 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _git_ok(stdout: str) -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = ""
    return m
