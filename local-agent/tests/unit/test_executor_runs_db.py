"""Tests for agent.executor_runs_db — SQLite-backed executor run metadata."""

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent import executor_runs_db, tracing


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point executor_runs_db at a temporary SQLite DB for each test."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
    # Clear cached per-thread connection so we get a fresh one
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()
    yield
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


class TestInitDb:
    def test_creates_table(self):
        conn = executor_runs_db._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='executor_runs'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        executor_runs_db.init_db()
        executor_runs_db.init_db()

    def test_schema_has_required_columns(self):
        """Schema must match the spec: id, jira_key, branch, started_at,
        ended_at, duration_ms, cost_usd, status, exit_code, tests_passed,
        deployed."""
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
        }
        expected = {
            "id", "jira_key", "branch", "started_at", "ended_at",
            "duration_ms", "cost_usd", "status", "exit_code",
            "tests_passed", "deployed",
        }
        assert expected.issubset(cols)


class TestStatusStartedIndex:
    """Composite index on (status, started_at DESC) — dashboard queries filter
    by status and order by recency; without this, the table scan grows
    linearly with run history."""

    def test_index_exists(self):
        """PRAGMA index_list exposes the composite index on a fresh DB."""
        conn = executor_runs_db._get_conn()
        names = {
            r["name"]
            for r in conn.execute(
                "PRAGMA index_list('executor_runs')"
            ).fetchall()
        }
        assert "idx_executor_runs_status_started" in names

    def test_index_covers_status_and_started_at(self):
        """The composite index must cover status first, then started_at."""
        conn = executor_runs_db._get_conn()
        cols = [
            r["name"]
            for r in conn.execute(
                "PRAGMA index_info('idx_executor_runs_status_started')"
            ).fetchall()
        ]
        assert cols == ["status", "started_at"]

    def test_query_plan_uses_index(self):
        """EXPLAIN QUERY PLAN for a status-filtered recency query must use
        the composite index, not scan the whole table."""
        conn = executor_runs_db._get_conn()
        # Seed enough rows that the planner has a reason to pick the index.
        for i in range(20):
            executor_runs_db.record_run(
                jira_key=f"TK-{i}",
                status="success" if i % 2 == 0 else "running",
                started_at=f"2026-04-{(i % 28) + 1:02d}T10:00:00",
            )
        conn.execute("ANALYZE")

        plan_rows = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT id FROM executor_runs "
            "WHERE status = ? "
            "ORDER BY started_at DESC LIMIT 10",
            ("success",),
        ).fetchall()
        plan_text = " ".join(str(r["detail"]) for r in plan_rows)
        assert "idx_executor_runs_status_started" in plan_text

    def test_migration_on_preexisting_db(self, tmp_path, monkeypatch):
        """A DB created before this index existed gets the index on next
        init_db() call — idempotent migration via IF NOT EXISTS."""
        # Close the fixture's cached connection so we can build a fresh DB
        # that's missing the new index.
        existing = getattr(executor_runs_db._local, "conn", None)
        if existing:
            existing.close()
            executor_runs_db._local.conn = None

        legacy_path = tmp_path / "legacy.db"
        monkeypatch.setattr(executor_runs_db, "DB_PATH", legacy_path)

        # Simulate an older DB: table + old indexes only, no composite index.
        legacy = sqlite3.connect(str(legacy_path))
        legacy.execute("""
            CREATE TABLE executor_runs (
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
                deployed       INTEGER
            )
        """)
        legacy.commit()
        names_before = {
            r[0] for r in legacy.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='executor_runs'"
            ).fetchall()
        }
        legacy.close()
        assert "idx_executor_runs_status_started" not in names_before

        # Running init_db() against the legacy DB must add the index.
        executor_runs_db.init_db()
        conn = executor_runs_db._get_conn()
        names_after = {
            r["name"]
            for r in conn.execute(
                "PRAGMA index_list('executor_runs')"
            ).fetchall()
        }
        assert "idx_executor_runs_status_started" in names_after


class TestRecordRun:
    def test_insert_returns_row_id(self):
        run_id = executor_runs_db.record_run(
            jira_key="TK-447",
            branch="test-branch",
            started_at="2026-04-16T10:00:00",
            status="running",
        )
        assert run_id >= 1

    def test_insert_and_retrieve(self):
        """Acceptance criterion: insert a run and retrieve it."""
        run_id = executor_runs_db.record_run(
            jira_key="TK-447",
            branch="2026-04-16-052408-TK-447",
            started_at="2026-04-16T10:00:00",
            status="running",
        )
        recent = executor_runs_db.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["id"] == run_id
        assert recent[0]["jira_key"] == "TK-447"
        assert recent[0]["branch"] == "2026-04-16-052408-TK-447"
        assert recent[0]["status"] == "running"

    def test_update_existing_run(self):
        """record_run with id= updates an existing row."""
        run_id = executor_runs_db.record_run(
            started_at="2026-04-16T10:00:00",
            status="running",
        )
        updated_id = executor_runs_db.record_run(
            id=run_id,
            ended_at="2026-04-16T10:05:00",
            duration_ms=300_000,
            cost_usd=0.42,
            status="success",
            exit_code=0,
        )
        assert updated_id == run_id

        rows = executor_runs_db.get_recent()
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "success"
        assert row["duration_ms"] == 300_000
        assert row["cost_usd"] == pytest.approx(0.42)
        assert row["exit_code"] == 0
        assert row["ended_at"] == "2026-04-16T10:05:00"

    def test_update_noop_when_no_fields(self):
        run_id = executor_runs_db.record_run(status="running")
        # update with only id — nothing should change, returns same id
        assert executor_runs_db.record_run(id=run_id) == run_id

    def test_default_started_at_on_insert(self):
        """Caller can omit started_at — module fills it with now()."""
        run_id = executor_runs_db.record_run(status="running")
        rows = executor_runs_db.get_recent()
        assert rows[0]["id"] == run_id
        assert rows[0]["started_at"] is not None
        # Must be parseable ISO format
        datetime.fromisoformat(rows[0]["started_at"])

    def test_unknown_keys_ignored(self):
        """Unknown kwargs are silently dropped — protects against SQL injection
        via keys and avoids crashing when callers pass extra metadata."""
        run_id = executor_runs_db.record_run(
            jira_key="TK-1",
            not_a_column="ignored",
            __evil__="'; DROP TABLE executor_runs; --",
        )
        rows = executor_runs_db.get_recent()
        assert len(rows) == 1
        assert rows[0]["jira_key"] == "TK-1"

    def test_boolean_coercion_for_bool_columns(self):
        """tests_passed / deployed accept bool and are stored as 0/1."""
        run_id = executor_runs_db.record_run(
            status="success",
            tests_passed=True,
            deployed=False,
        )
        rows = executor_runs_db.get_recent()
        assert rows[0]["tests_passed"] == 1
        assert rows[0]["deployed"] == 0


class TestGetRecent:
    def test_empty_db_returns_empty_list(self):
        assert executor_runs_db.get_recent() == []

    def test_orders_newest_first(self):
        ids = [
            executor_runs_db.record_run(jira_key=f"TK-{i}")
            for i in range(5)
        ]
        recent = executor_runs_db.get_recent(limit=10)
        assert [r["id"] for r in recent] == list(reversed(ids))

    def test_respects_limit(self):
        for i in range(5):
            executor_runs_db.record_run(jira_key=f"TK-{i}")
        assert len(executor_runs_db.get_recent(limit=2)) == 2

    def test_default_limit_is_20(self):
        for i in range(25):
            executor_runs_db.record_run(jira_key=f"TK-{i}")
        assert len(executor_runs_db.get_recent()) == 20


class TestGetRunByRunId:
    """Lookup a run by its sortable ``run_id`` string."""

    def test_returns_row_as_dict(self):
        """Inserting via record_run and retrieving via the helper returns
        the expected fields as a dict."""
        run_id_str = "20260416-120000-TK-491"
        executor_runs_db.record_run(
            jira_key="TK-491",
            branch="2026-04-16-test",
            started_at="2026-04-16T12:00:00",
            status="running",
            run_id=run_id_str,
            pid=4242,
        )

        row = executor_runs_db.get_run_by_run_id(run_id_str)

        assert row is not None
        assert isinstance(row, dict)
        assert row["run_id"] == run_id_str
        assert row["pid"] == 4242
        assert row["status"] == "running"

    def test_returns_none_when_missing(self):
        assert executor_runs_db.get_run_by_run_id("does-not-exist") is None

    def test_returns_most_recent_on_duplicate_run_ids(self):
        """If two rows somehow share the same run_id, the newest (highest id)
        wins — the column isn't UNIQUE so we must be deterministic."""
        run_id_str = "20260416-120000-TK-dup"
        first = executor_runs_db.record_run(
            jira_key="TK-dup", status="running", run_id=run_id_str,
        )
        second = executor_runs_db.record_run(
            jira_key="TK-dup", status="success", run_id=run_id_str,
        )
        assert second > first

        row = executor_runs_db.get_run_by_run_id(run_id_str)
        assert row is not None
        assert row["id"] == second
        assert row["status"] == "success"


class TestWalMode:
    """WAL journal mode must be active so reads and writes don't serialize."""

    def test_journal_mode_is_wal(self):
        conn = executor_runs_db._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_concurrent_read_during_open_write_transaction(self):
        """With WAL, a separate connection can read while another holds an
        open write transaction — the regression signal is `database is
        locked` which is what we're eliminating."""
        executor_runs_db.record_run(jira_key="TK-seed", status="running")
        # Drop the cached per-thread connection so the writer below is the
        # only one holding the write lock when we open the reader.
        conn = getattr(executor_runs_db._local, "conn", None)
        if conn:
            conn.close()
            executor_runs_db._local.conn = None

        writer = sqlite3.connect(str(executor_runs_db.DB_PATH), timeout=5)
        reader = sqlite3.connect(str(executor_runs_db.DB_PATH), timeout=5)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO executor_runs (jira_key, status) VALUES (?, ?)",
                ("TK-concurrent", "running"),
            )
            # Read from a separate connection while writer's txn is open.
            count = reader.execute(
                "SELECT COUNT(*) FROM executor_runs"
            ).fetchone()[0]
            assert count >= 1
            writer.rollback()
        finally:
            writer.close()
            reader.close()


class TestPurgeOldRuns:
    """Retention policy: rows and artifacts older than N days are deleted."""

    def _iso(self, days_ago: float) -> str:
        return (datetime.now() - timedelta(days=days_ago)).isoformat(
            sep=" ", timespec="seconds"
        )

    def test_old_rows_deleted(self):
        """Rows older than the cutoff are removed."""
        old_id = executor_runs_db.record_run(
            jira_key="TK-old",
            started_at=self._iso(days_ago=40),
            status="success",
        )
        executor_runs_db.record_run(
            jira_key="TK-recent",
            started_at=self._iso(days_ago=1),
            status="success",
        )

        deleted = executor_runs_db.purge_old_runs(30)

        assert deleted == 1
        remaining = {r["jira_key"] for r in executor_runs_db.get_recent()}
        assert remaining == {"TK-recent"}
        # The old row really is gone — not just hidden from get_recent
        conn = executor_runs_db._get_conn()
        row = conn.execute(
            "SELECT id FROM executor_runs WHERE id = ?", (old_id,)
        ).fetchone()
        assert row is None

    def test_recent_rows_retained(self):
        """Rows inside the retention window must survive."""
        ids = [
            executor_runs_db.record_run(
                jira_key=f"TK-{i}",
                started_at=self._iso(days_ago=float(i)),
                status="success",
            )
            for i in range(5)
        ]

        deleted = executor_runs_db.purge_old_runs(30)

        assert deleted == 0
        remaining_ids = {r["id"] for r in executor_runs_db.get_recent()}
        assert remaining_ids == set(ids)

    def test_returns_deleted_count(self):
        """Purge returns the count of rows removed."""
        for i in range(3):
            executor_runs_db.record_run(
                jira_key=f"TK-old-{i}",
                started_at=self._iso(days_ago=90),
                status="success",
            )
        executor_runs_db.record_run(
            jira_key="TK-keep",
            started_at=self._iso(days_ago=5),
            status="success",
        )

        assert executor_runs_db.purge_old_runs(30) == 3

    def test_boundary_same_day_not_deleted(self):
        """A row less than N days old is retained."""
        executor_runs_db.record_run(
            jira_key="TK-boundary",
            started_at=self._iso(days_ago=29.5),
            status="success",
        )
        assert executor_runs_db.purge_old_runs(30) == 0

    def test_null_started_at_is_not_deleted(self):
        """Rows with no started_at shouldn't match the cutoff — safer default."""
        conn = executor_runs_db._get_conn()
        conn.execute(
            "INSERT INTO executor_runs (jira_key, started_at, status) "
            "VALUES (?, NULL, ?)",
            ("TK-null", "running"),
        )
        conn.commit()

        assert executor_runs_db.purge_old_runs(30) == 0
        rows = conn.execute(
            "SELECT jira_key FROM executor_runs WHERE jira_key = 'TK-null'"
        ).fetchall()
        assert len(rows) == 1

    def test_archived_files_older_than_cutoff_removed(self, tmp_path, monkeypatch):
        """Files in ARTIFACTS_DIR older than cutoff are unlinked on disk."""
        artifacts = tmp_path / "executor_artifacts"
        artifacts.mkdir()
        monkeypatch.setattr(executor_runs_db, "ARTIFACTS_DIR", artifacts)

        old_run = artifacts / "20260101-120000-TK-old"
        old_run.mkdir()
        old_stdout = old_run / "stdout.log"
        old_stderr = old_run / "stderr.log"
        old_stdout.write_text("old", encoding="utf-8")
        old_stderr.write_text("old", encoding="utf-8")

        new_run = artifacts / "20260416-120000-TK-new"
        new_run.mkdir()
        new_stdout = new_run / "stdout.log"
        new_stdout.write_text("new", encoding="utf-8")

        # Backdate the old run's files by 45 days (well past 30-day cutoff).
        ancient_ts = time.time() - (45 * 86400)
        os.utime(old_stdout, (ancient_ts, ancient_ts))
        os.utime(old_stderr, (ancient_ts, ancient_ts))

        executor_runs_db.purge_old_runs(30)

        assert not old_stdout.exists()
        assert not old_stderr.exists()
        # Empty directory is cleaned up too
        assert not old_run.exists()
        # Recent run is untouched
        assert new_stdout.exists()
        assert new_run.exists()

    def test_missing_artifacts_dir_is_safe(self, tmp_path, monkeypatch):
        """Purge must not raise if ARTIFACTS_DIR doesn't exist."""
        missing = tmp_path / "does-not-exist"
        monkeypatch.setattr(executor_runs_db, "ARTIFACTS_DIR", missing)
        # No DB rows either — just ensure no exception
        assert executor_runs_db.purge_old_runs(30) == 0

    def test_zero_days_purges_everything(self):
        """days=0 deletes every row (sanity check for boundary)."""
        executor_runs_db.record_run(
            jira_key="TK-a",
            started_at=self._iso(days_ago=0.01),
            status="success",
        )
        executor_runs_db.record_run(
            jira_key="TK-b",
            started_at=self._iso(days_ago=1),
            status="success",
        )
        # Sleep a moment so 'now' advances past the inserted timestamps
        time.sleep(1.1)
        assert executor_runs_db.purge_old_runs(0) == 2


class TestPurgeCli:
    """CLI entry point: ``python -m agent.executor_runs_db purge <days>``."""

    def test_purge_prints_deleted_count(self, capsys):
        executor_runs_db.record_run(
            jira_key="TK-old-cli",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(
                sep=" ", timespec="seconds"
            ),
            status="success",
        )

        exit_code = executor_runs_db._main(["purge", "30"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Deleted 1" in out
        assert "30 day" in out

    def test_purge_rejects_non_integer_days(self, capsys):
        assert executor_runs_db._main(["purge", "abc"]) == 2
        assert "must be an integer" in capsys.readouterr().out

    def test_usage_printed_for_unknown_command(self, capsys):
        assert executor_runs_db._main([]) == 2
        assert "usage:" in capsys.readouterr().out


class TestDiscoverArtifacts:
    """Tests for _discover_artifacts helper function."""

    def test_returns_empty_list_for_none_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns empty list for None run_id."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        result = executor_runs_db._discover_artifacts(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_empty_list_for_empty_string_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns empty list for empty string."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        result = executor_runs_db._discover_artifacts("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_empty_list_for_nonexistent_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns empty list for non-existent run_id."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        result = executor_runs_db._discover_artifacts("nonexistent-run-id")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_directory_path_for_existing_run_id_dir(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns directory path when it exists."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "test-run-dir"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert dir_path in result
        assert all(isinstance(p, Path) for p in result)

    def test_returns_log_file_path_for_existing_run_id_log(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns .log file path when it exists."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "test-run-log"
        log_path = logs_dir / f"{run_id}.log"
        log_path.write_text("test log content\n", encoding="utf-8")

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert log_path in result
        assert all(isinstance(p, Path) for p in result)

    def test_returns_done_file_path_for_existing_run_id_done(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns .done file path when it exists."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "test-run-done"
        done_path = logs_dir / f"{run_id}.done"
        done_path.write_text("test done content\n", encoding="utf-8")

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert done_path in result
        assert all(isinstance(p, Path) for p in result)

    def test_returns_multiple_paths_for_all_artifacts(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts returns all three artifact types when they exist."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "test-run-all"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.log"
        log_path.write_text("test log content\n", encoding="utf-8")
        done_path = logs_dir / f"{run_id}.done"
        done_path.write_text("test done content\n", encoding="utf-8")

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 3
        assert dir_path in result
        assert log_path in result
        assert done_path in result
        assert all(isinstance(p, Path) for p in result)

    def test_returns_only_existing_paths(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts only returns paths that actually exist."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "test-run-mixed"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.log"
        log_path.write_text("test log content\n", encoding="utf-8")
        # .done file intentionally not created
        done_path = logs_dir / f"{run_id}.done"

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 2
        assert dir_path in result
        assert log_path in result
        assert done_path not in result

    def test_returns_paths_with_special_characters_in_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts handles run_ids with special characters."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "run-123-test-abc"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.log"
        log_path.write_text("test log content\n", encoding="utf-8")

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 2
        assert dir_path in result
        assert log_path in result

    def test_returns_paths_with_dots_in_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts handles run_ids with dots."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "run.id.with.dots"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert dir_path in result

    def test_returns_paths_with_numbers_in_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts handles run_ids with numbers."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "20260415-120000-TK-1234"
        log_path = logs_dir / f"{run_id}.log"
        log_path.write_text("test log content\n", encoding="utf-8")

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert log_path in result

    def test_returns_paths_with_unicode_in_run_id(self, tmp_path, monkeypatch):
        """Test that _discover_artifacts handles run_ids with unicode characters."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr(executor_runs_db, "EXECUTION_LOGS_DIR", logs_dir)

        run_id = "run-测试-123"
        dir_path = logs_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)

        result = executor_runs_db._discover_artifacts(run_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert dir_path in result


class TestExecutorToolCalls:
    """Per-tool telemetry table: executor_tool_calls."""

    def test_table_created(self):
        conn = executor_runs_db._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='executor_tool_calls'"
        ).fetchone()
        assert row is not None

    def test_schema_has_required_columns(self):
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute(
                "PRAGMA table_info(executor_tool_calls)"
            ).fetchall()
        }
        expected = {
            "id", "run_id", "tool_name", "started_at", "duration_ms",
            "input_tokens", "output_tokens", "ok", "error_message",
        }
        assert expected.issubset(cols)

    def test_insert_and_retrieve(self):
        run_id = executor_runs_db.record_run(status="running")
        executor_runs_db.record_tool_call(
            run_id=run_id,
            tool_name="Read",
            started_at="2026-04-16T10:00:00",
            duration_ms=123,
            input_tokens=456,
            output_tokens=789,
            ok=True,
        )
        tool_calls = executor_runs_db.get_tool_calls(run_id)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "Read"
        assert tool_calls[0]["duration_ms"] == 123
        assert tool_calls[0]["input_tokens"] == 456
        assert tool_calls[0]["output_tokens"] == 789
        assert tool_calls[0]["ok"] == 1

    def test_tool_calls_deleted_with_run(self):
        """When a run is purged, its tool calls must also be deleted."""
        # Create a run with a specific old started_at to ensure it gets purged
        old_started_at = (datetime.now() - timedelta(days=31)).isoformat(
            sep=" ", timespec="seconds"
        )
        run_id = executor_runs_db.record_run(
            status="running",
            started_at=old_started_at
        )
        executor_runs_db.record_tool_call(
            run_id=run_id,
            tool_name="Read",
        )
        # Purge the run (should delete everything older than 30 days)
        deleted = executor_runs_db.purge_old_runs(30)
        assert deleted == 1
        tool_calls = executor_runs_db.get_tool_calls(run_id)
        assert tool_calls == []


class TestExecuteCleanup:
    """Tests for the private _execute_cleanup helper function."""

    def test_deletes_existing_files(self, tmp_path):
        """_execute_cleanup deletes files that exist."""
        # Create test files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("content1", encoding="utf-8")
        file2.write_text("content2", encoding="utf-8")

        # Verify files exist
        assert file1.exists()
        assert file2.exists()

        # Delete files using _execute_cleanup
        deleted = executor_runs_db._execute_cleanup([str(file1), str(file2)])

        # Verify deletion
        assert deleted == 2
        assert not file1.exists()
        assert not file2.exists()

    def test_skips_nonexistent_files(self, tmp_path):
        """_execute_cleanup skips files that don't exist."""
        # Create one existing file
        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("content", encoding="utf-8")

        # Non-existent file paths
        nonexistent1 = tmp_path / "nonexistent1.txt"
        nonexistent2 = tmp_path / "nonexistent2.txt"

        # Verify only existing file exists
        assert existing_file.exists()
        assert not nonexistent1.exists()
        assert not nonexistent2.exists()

        # Delete files using _execute_cleanup
        deleted = executor_runs_db._execute_cleanup([
            str(existing_file),
            str(nonexistent1),
            str(nonexistent2),
        ])

        # Verify only existing file was deleted
        assert deleted == 1
        assert not existing_file.exists()
        assert not nonexistent1.exists()
        assert not nonexistent2.exists()

    def test_handles_mixed_valid_invalid(self, tmp_path):
        """_execute_cleanup handles a mix of existing and non-existent files."""
        # Create multiple files
        files = [
            tmp_path / "valid1.txt",
            tmp_path / "valid2.txt",
            tmp_path / "valid3.txt",
        ]
        for f in files:
            f.write_text("content", encoding="utf-8")

        # Non-existent files
        nonexistent = [
            tmp_path / "invalid1.txt",
            tmp_path / "invalid2.txt",
        ]

        # Verify all files exist
        for f in files:
            assert f.exists()
        for f in nonexistent:
            assert not f.exists()

        # Delete files using _execute_cleanup
        all_paths = [str(f) for f in files] + [str(f) for f in nonexistent]
        deleted = executor_runs_db._execute_cleanup(all_paths)

        # Verify only valid files were deleted
        assert deleted == 3
        for f in files:
            assert not f.exists()
        for f in nonexistent:
            assert not f.exists()

    def test_handles_os_error_on_delete(self, tmp_path, monkeypatch):
        """_execute_cleanup handles OSError when deleting files."""
        # Create a file
        file1 = tmp_path / "test.txt"
        file1.write_text("content", encoding="utf-8")

        # Mock unlink to raise OSError
        original_unlink = Path.unlink

        def mock_unlink(self):
            if self == file1:
                raise OSError("Permission denied")
            return original_unlink(self)

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        # Delete files using _execute_cleanup
        deleted = executor_runs_db._execute_cleanup([str(file1)])

        # Verify OSError was handled gracefully
        assert deleted == 0
        assert file1.exists()

    def test_empty_input_list(self, tmp_path):
        """_execute_cleanup handles empty input list gracefully."""
        deleted = executor_runs_db._execute_cleanup([])
        assert deleted == 0

    def test_handles_none_values(self, tmp_path):
        """_execute_cleanup handles None values in the list."""
        # Create a file
        file1 = tmp_path / "test.txt"
        file1.write_text("content", encoding="utf-8")

        # Delete files using _execute_cleanup with None values
        deleted = executor_runs_db._execute_cleanup([str(file1), None, ""])

        # Verify only the valid file was deleted
        assert deleted == 1
        assert not file1.exists()


class TestCleanupIntegration:
    """Tests for integration between cleanup functions and executor_runs_db."""

    def test_cleanup_integration_with_artifacts(self, tmp_path, monkeypatch):
        """Test that cleanup properly integrates with artifact removal."""
        # Set up a temporary execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("agent.executor_runs_cleanup.EXECUTION_LOGS_DIR", logs_dir)
        
        # Create a run with artifacts
        run_id = "test-run-123"
        executor_runs_db.record_run(
            run_id=run_id,
            jira_key="TK-123",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(),
            status="success",
        )
        
        # Create artifact files
        run_dir = logs_dir / run_id
        run_dir.mkdir()
        (run_dir / "stdout.log").write_text("test output", encoding="utf-8")
        (run_dir / "stderr.log").write_text("test error", encoding="utf-8")
        
        # Verify files exist before cleanup
        assert run_dir.exists()
        assert (run_dir / "stdout.log").exists()
        assert (run_dir / "stderr.log").exists()
        
        # Run cleanup
        from agent import executor_runs_cleanup
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, 
            keep_last_n=0
        )
        
        # Verify cleanup worked
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert result["bytes_freed"] > 0
        
        # Verify artifacts were removed
        assert not run_dir.exists()
        
        # Verify run was removed from DB
        conn = executor_runs_db._get_conn()
        rows = conn.execute("SELECT COUNT(*) as count FROM executor_runs").fetchone()
        assert rows["count"] == 0

    def test_cleanup_integration_with_flat_files(self, tmp_path, monkeypatch):
        """Test that cleanup properly handles flat .log and .done files."""
        # Set up a temporary execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("agent.executor_runs_cleanup.EXECUTION_LOGS_DIR", logs_dir)
        
        # Create a run with flat artifact files
        run_id = "flat-test-run"
        executor_runs_db.record_run(
            run_id=run_id,
            jira_key="TK-456",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(),
            status="success",
        )
        
        # Create flat artifact files
        log_file = logs_dir / f"{run_id}.log"
        done_file = logs_dir / f"{run_id}.done"
        log_file.write_text("test log content", encoding="utf-8")
        done_file.write_text("test done content", encoding="utf-8")
        
        # Verify files exist before cleanup
        assert log_file.exists()
        assert done_file.exists()
        
        # Run cleanup
        from agent import executor_runs_cleanup
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, 
            keep_last_n=0
        )
        
        # Verify cleanup worked
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert result["bytes_freed"] > 0
        
        # Verify artifacts were removed
        assert not log_file.exists()
        assert not done_file.exists()
        
        # Verify run was removed from DB
        conn = executor_runs_db._get_conn()
        rows = conn.execute("SELECT COUNT(*) as count FROM executor_runs").fetchone()
        assert rows["count"] == 0

    def test_cleanup_integration_with_mixed_artifacts(self, tmp_path, monkeypatch):
        """Test that cleanup handles both directory and flat file artifacts."""
        # Set up a temporary execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("agent.executor_runs_cleanup.EXECUTION_LOGS_DIR", logs_dir)
        
        # Create a run with both directory and flat files
        run_id = "mixed-artifacts-run"
        executor_runs_db.record_run(
            run_id=run_id,
            jira_key="TK-789",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(),
            status="success",
        )
        
        # Create directory artifact
        run_dir = logs_dir / run_id
        run_dir.mkdir()
        (run_dir / "stdout.log").write_text("test output", encoding="utf-8")
        
        # Create flat artifact file
        log_file = logs_dir / f"{run_id}.log"
        log_file.write_text("test log content", encoding="utf-8")
        
        # Verify artifacts exist before cleanup
        assert run_dir.exists()
        assert log_file.exists()
        
        # Run cleanup
        from agent import executor_runs_cleanup
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, 
            keep_last_n=0
        )
        
        # Verify cleanup worked
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert result["bytes_freed"] > 0
        
        # Verify artifacts were removed
        assert not run_dir.exists()
        assert not log_file.exists()
        
        # Verify run was removed from DB
        conn = executor_runs_db._get_conn()
        rows = conn.execute("SELECT COUNT(*) as count FROM executor_runs").fetchone()
        assert rows["count"] == 0

    def test_cleanup_integration_with_dry_run(self, tmp_path, monkeypatch):
        """Test that cleanup works correctly in dry_run mode."""
        # Set up a temporary execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("agent.executor_runs_cleanup.EXECUTION_LOGS_DIR", logs_dir)
        
        # Create a run with artifacts
        run_id = "dry-run-test"
        executor_runs_db.record_run(
            run_id=run_id,
            jira_key="TK-999",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(),
            status="success",
        )
        
        # Create artifact files
        run_dir = logs_dir / run_id
        run_dir.mkdir()
        (run_dir / "stdout.log").write_text("test output", encoding="utf-8")
        
        # Verify files exist before cleanup
        assert run_dir.exists()
        
        # Run cleanup in dry_run mode
        from agent import executor_runs_cleanup
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, 
            keep_last_n=0,
            dry_run=True
        )
        
        # Verify dry run results
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        assert result["bytes_freed"] > 0
        assert result["dry_run"] == 1
        
        # Verify artifacts were NOT removed (dry run)
        assert run_dir.exists()
        
        # Verify run was NOT removed from DB (dry run)
        conn = executor_runs_db._get_conn()
        rows = conn.execute("SELECT COUNT(*) as count FROM executor_runs").fetchone()
        assert rows["count"] == 1  # Still exists

    def test_cleanup_integration_with_tool_calls(self, tmp_path, monkeypatch):
        """Test that cleanup properly removes tool call records along with run records."""
        # Set up a temporary execution logs directory
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        monkeypatch.setattr("agent.executor_runs_cleanup.EXECUTION_LOGS_DIR", logs_dir)
        
        # Create a run with tool calls and artifacts
        run_id = "tool-calls-test"
        db_run_id = executor_runs_db.record_run(
            run_id=run_id,
            jira_key="TK-111",
            started_at=(datetime.now() - timedelta(days=45)).isoformat(),
            status="success",
        )
        
        # Add tool calls
        executor_runs_db.record_tool_call(
            run_id=db_run_id,
            tool_name="ReadFile",
        )
        executor_runs_db.record_tool_call(
            run_id=db_run_id,
            tool_name="WriteFile",
        )
        
        # Create artifact files to ensure dirs_deleted > 0
        run_dir = logs_dir / run_id
        run_dir.mkdir()
        (run_dir / "stdout.log").write_text("test output", encoding="utf-8")
        
        # Verify tool calls exist
        tool_calls = executor_runs_db.get_tool_calls(db_run_id)
        assert len(tool_calls) == 2
        
        # Run cleanup
        from agent import executor_runs_cleanup
        result = executor_runs_cleanup.cleanup_old_runs(
            max_age_days=30, 
            keep_last_n=0
        )
        
        # Verify cleanup worked
        assert result["rows_deleted"] == 1
        assert result["dirs_deleted"] >= 1
        
        # Verify tool calls were removed
        tool_calls = executor_runs_db.get_tool_calls(db_run_id)
        assert tool_calls == []
        
        # Verify run was removed from DB
        conn = executor_runs_db._get_conn()
        rows = conn.execute("SELECT COUNT(*) as count FROM executor_runs").fetchone()
        assert rows["count"] == 0
