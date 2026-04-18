"""Tests for ``agent.daily_rollup`` splitter-child tracking (TK-618).

Covers:

* :func:`agent.daily_rollup._splitter_child_counts` — returns
  ``(done, failed)`` on success, ``None`` when Jira is unreachable or
  misconfigured.
* :func:`agent.daily_rollup.compute_and_write` — writes the counts into
  ``splitter_child_success`` / ``splitter_child_fail`` or NULL on Jira
  failure.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import daily_rollup, daily_stats, executor_runs_db, story_timings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_dbs(tmp_path, monkeypatch):
    """Point every SQLite DB used by daily_rollup at a fresh temp path."""
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", tmp_path / "executor_runs.db")
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", tmp_path / "daily_stats.db")
    monkeypatch.setattr(story_timings, "DB_DIR", tmp_path)
    monkeypatch.setattr(story_timings, "DB_PATH", tmp_path / "story_timings.db")
    # Non-git path so _git_loc_counts short-circuits to (0, 0) and doesn't
    # pull LOC from the real repo.
    monkeypatch.setattr(daily_rollup, "REPO_ROOT", tmp_path)
    for mod in (executor_runs_db, daily_stats, story_timings):
        mod._local.__dict__.pop("conn", None)
    yield
    for mod in (executor_runs_db, daily_stats, story_timings):
        conn = getattr(mod._local, "conn", None)
        if conn is not None:
            conn.close()
            mod._local.__dict__.pop("conn", None)


def _issue(key: str, status: str) -> dict:
    """Build the minimal Jira issue shape ``_splitter_child_counts`` reads."""
    return {
        "key": key,
        "fields": {"status": {"name": status}},
    }


# ---------------------------------------------------------------------------
# _splitter_child_counts
# ---------------------------------------------------------------------------


class TestSplitterChildCounts:
    def test_returns_none_when_jira_not_configured(self):
        with patch.object(daily_rollup, "is_jira_configured", return_value=False):
            assert daily_rollup._splitter_child_counts("2026-04-17", "TK") is None

    def test_returns_none_on_invalid_date(self):
        with patch.object(daily_rollup, "is_jira_configured", return_value=True):
            assert daily_rollup._splitter_child_counts("not-a-date", "TK") is None

    def test_acceptance_six_children_four_done_two_failed(self):
        """AC: mocked jira_provider returns 6 splitter children, 4 Done and
        2 Failed → function returns (4, 2)."""
        issues = [
            _issue("TK-601", "Done"),
            _issue("TK-602", "Done"),
            _issue("TK-603", "Done"),
            _issue("TK-604", "Done"),
            _issue("TK-605", "Failed"),
            _issue("TK-606", "Failed"),
        ]
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(issues, 200),
             ):
            result = daily_rollup._splitter_child_counts("2026-04-17", "TK")
        assert result == (4, 2)

    def test_in_progress_issues_not_counted(self):
        """Only terminal Done/Failed statuses contribute to the totals."""
        issues = [
            _issue("TK-1", "Done"),
            _issue("TK-2", "In Progress"),
            _issue("TK-3", "To Do"),
            _issue("TK-4", "Failed"),
        ]
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(issues, 200),
             ):
            result = daily_rollup._splitter_child_counts("2026-04-17", "TK")
        assert result == (1, 1)

    def test_empty_result_returns_zero_tuple(self):
        """Jira reachable but no matching issues → (0, 0), not None."""
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=([], 200),
             ):
            assert daily_rollup._splitter_child_counts("2026-04-17", "TK") == (0, 0)

    def test_search_failure_returns_none(self):
        """Jira returned None (HTTP error) → graceful NULL."""
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(None, 500),
             ):
            assert daily_rollup._splitter_child_counts("2026-04-17", "TK") is None

    def test_search_raises_returns_none(self):
        """An exception from the search path must not propagate."""
        def _boom(*args, **kwargs):
            raise ConnectionError("jira down")

        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(daily_rollup, "_paginated_search", side_effect=_boom):
            assert daily_rollup._splitter_child_counts("2026-04-17", "TK") is None

    def test_jql_mentions_label_project_and_date_window(self):
        """Sanity: the JQL sent to Jira filters by label, project, and the
        single-day created window."""
        captured: dict = {}

        def _capture(jql, limit, fields):
            captured["jql"] = jql
            captured["limit"] = limit
            return ([], 200)

        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(daily_rollup, "_paginated_search", side_effect=_capture):
            daily_rollup._splitter_child_counts("2026-04-17", "TK")

        jql = captured["jql"]
        assert 'project = "TK"' in jql
        assert 'labels = "src:splitter"' in jql
        assert 'created >= "2026-04-17"' in jql
        assert 'created < "2026-04-18"' in jql
        assert captured["limit"] == daily_rollup.SPLITTER_CHILD_QUERY_LIMIT


# ---------------------------------------------------------------------------
# compute_and_write integration
# ---------------------------------------------------------------------------


class TestComputeAndWriteSplitter:
    def test_writes_counts_when_jira_ok(self):
        """End-to-end: compute_and_write persists 4/2 from the mocked Jira."""
        issues = [
            _issue("TK-601", "Done"),
            _issue("TK-602", "Done"),
            _issue("TK-603", "Done"),
            _issue("TK-604", "Done"),
            _issue("TK-605", "Failed"),
            _issue("TK-606", "Failed"),
        ]
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(issues, 200),
             ):
            result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["splitter_child_success"] == 4
        assert result["splitter_child_fail"] == 2

        stats_conn = daily_stats._get_conn()
        row = stats_conn.execute(
            """SELECT splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["splitter_child_success"] == 4
        assert row["splitter_child_fail"] == 2

    def test_writes_null_when_jira_unreachable(self):
        """AC: Jira unreachable → both columns persist as SQL NULL."""
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(None, 503),
             ):
            result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["splitter_child_success"] is None
        assert result["splitter_child_fail"] is None

        stats_conn = daily_stats._get_conn()
        row = stats_conn.execute(
            """SELECT splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["splitter_child_success"] is None
        assert row["splitter_child_fail"] is None

    def test_writes_null_when_jira_not_configured(self):
        """Jira-disabled environments must still write a row, with NULLs."""
        with patch.object(daily_rollup, "is_jira_configured", return_value=False):
            result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["splitter_child_success"] is None
        assert result["splitter_child_fail"] is None

        stats_conn = daily_stats._get_conn()
        row = stats_conn.execute(
            """SELECT splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["splitter_child_success"] is None
        assert row["splitter_child_fail"] is None

    def test_rerun_overwrites_null_with_real_counts(self):
        """Replaying the rollup after Jira recovers must upsert the counts."""
        # First run: Jira down → NULL.
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(None, 503),
             ):
            daily_rollup.compute_and_write("2026-04-17", "TK")

        # Second run: Jira OK → counts land on top of the previous NULLs.
        issues = [_issue("TK-1", "Done"), _issue("TK-2", "Failed")]
        with patch.object(daily_rollup, "is_jira_configured", return_value=True), \
             patch.object(
                 daily_rollup, "_paginated_search",
                 return_value=(issues, 200),
             ):
            result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["splitter_child_success"] == 1
        assert result["splitter_child_fail"] == 1

        stats_conn = daily_stats._get_conn()
        row = stats_conn.execute(
            """SELECT splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["splitter_child_success"] == 1
        assert row["splitter_child_fail"] == 1


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaNullability:
    def test_fresh_schema_allows_null_splitter_columns(self):
        """A newly-created daily_stats table accepts NULL for the splitter
        columns — the linchpin that makes 'Jira unreachable' recordable."""
        daily_stats.init_db()
        info = daily_stats._get_conn().execute(
            "PRAGMA table_info(daily_stats)"
        ).fetchall()
        by_name = {row[1]: row for row in info}
        # Row format: (cid, name, type, notnull, dflt_value, pk)
        assert by_name["splitter_child_success"][3] == 0
        assert by_name["splitter_child_fail"][3] == 0

    def test_migration_rebuilds_legacy_not_null_schema(self):
        """A pre-existing table with NOT NULL splitter columns must be
        rebuilt on init_db without losing data."""
        conn = daily_stats._get_conn()
        # Recreate the pre-TK-618 schema manually.
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
                splitter_child_success INTEGER NOT NULL DEFAULT 0,
                splitter_child_fail    INTEGER NOT NULL DEFAULT 0,
                phase_timings_json     TEXT,
                PRIMARY KEY (date, project)
            )
        """)
        conn.execute(
            """INSERT INTO daily_stats
                 (date, project, shipped, failed, cost_usd,
                  splitter_child_success, splitter_child_fail)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("2026-04-10", "TK", 3, 1, 0.5, 2, 1),
        )
        conn.commit()

        daily_stats.init_db()

        info = conn.execute("PRAGMA table_info(daily_stats)").fetchall()
        by_name = {row[1]: row for row in info}
        assert by_name["splitter_child_success"][3] == 0
        assert by_name["splitter_child_fail"][3] == 0

        # Existing row was preserved through the rebuild.
        row = conn.execute(
            """SELECT shipped, failed, splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-10", "TK"),
        ).fetchone()
        assert row["shipped"] == 3
        assert row["failed"] == 1
        assert row["splitter_child_success"] == 2
        assert row["splitter_child_fail"] == 1

        # After rebuild, NULL writes now work.
        conn.execute(
            """INSERT INTO daily_stats
                 (date, project, splitter_child_success, splitter_child_fail)
               VALUES (?, ?, ?, ?)""",
            ("2026-04-11", "TK", None, None),
        )
        conn.commit()
        row2 = conn.execute(
            """SELECT splitter_child_success, splitter_child_fail
               FROM daily_stats WHERE date = ? AND project = ?""",
            ("2026-04-11", "TK"),
        ).fetchone()
        assert row2["splitter_child_success"] is None
        assert row2["splitter_child_fail"] is None
