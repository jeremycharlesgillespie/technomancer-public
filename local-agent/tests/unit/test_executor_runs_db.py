"""Tests for agent.executor_runs_db — SQLite-backed executor run metadata."""

import sqlite3
from datetime import datetime

import pytest

from agent import executor_runs_db


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


class TestStartThenCompleteFlow:
    """The canonical lifecycle: insert on start, UPDATE on completion."""

    def test_full_lifecycle(self):
        # Phase 1: run starts
        run_id = executor_runs_db.record_run(
            jira_key="TK-447",
            branch="2026-04-16-test",
            started_at="2026-04-16T10:00:00",
            status="running",
        )

        # Phase 2: run completes — same id, new fields
        executor_runs_db.record_run(
            id=run_id,
            ended_at="2026-04-16T10:02:00",
            duration_ms=120_000,
            cost_usd=0.15,
            status="success",
            exit_code=0,
            tests_passed=True,
            deployed=True,
        )

        # Verify: one row, all fields preserved
        rows = executor_runs_db.get_recent()
        assert len(rows) == 1
        row = rows[0]
        assert row["jira_key"] == "TK-447"
        assert row["branch"] == "2026-04-16-test"
        assert row["started_at"] == "2026-04-16T10:00:00"
        assert row["ended_at"] == "2026-04-16T10:02:00"
        assert row["duration_ms"] == 120_000
        assert row["cost_usd"] == pytest.approx(0.15)
        assert row["status"] == "success"
        assert row["exit_code"] == 0
        assert row["tests_passed"] == 1
        assert row["deployed"] == 1
