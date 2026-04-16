"""Tests for agent.executor_runs_db — SQLite-backed executor run metadata."""

import os
import sqlite3
import time
from datetime import datetime, timedelta

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
        run_id = executor_runs_db.record_run(
            jira_key="TK-469", status="running",
        )
        tc_id = executor_runs_db.record_tool_call(
            run_id=run_id,
            tool_name="Bash",
            started_at="2026-04-16T10:00:00.000001",
            duration_ms=1234,
            input_tokens=100,
            output_tokens=200,
            ok=True,
        )
        assert tc_id >= 1

        rows = executor_runs_db.get_tool_calls(run_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["tool_name"] == "Bash"
        assert row["duration_ms"] == 1234
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 200
        assert row["ok"] == 1
        assert row["error_message"] is None

    def test_boolean_coercion_for_ok(self):
        run_id = executor_runs_db.record_run(status="running")
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Edit", ok=False,
            error_message="file not found",
        )
        row = executor_runs_db.get_tool_calls(run_id)[0]
        assert row["ok"] == 0
        assert row["error_message"] == "file not found"

    def test_sorted_by_started_at_ascending(self):
        run_id = executor_runs_db.record_run(status="running")
        # Insert out of order — retrieval must return them sorted by started_at
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="third",
            started_at="2026-04-16T10:00:03", ok=True,
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="first",
            started_at="2026-04-16T10:00:01", ok=True,
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="second",
            started_at="2026-04-16T10:00:02", ok=True,
        )
        rows = executor_runs_db.get_tool_calls(run_id)
        assert [r["tool_name"] for r in rows] == ["first", "second", "third"]

    def test_filter_by_run_id(self):
        """get_tool_calls only returns calls for the given run id."""
        run_a = executor_runs_db.record_run(jira_key="TK-A", status="running")
        run_b = executor_runs_db.record_run(jira_key="TK-B", status="running")
        executor_runs_db.record_tool_call(
            run_id=run_a, tool_name="ToolA",
            started_at="2026-04-16T10:00:00", ok=True,
        )
        executor_runs_db.record_tool_call(
            run_id=run_b, tool_name="ToolB",
            started_at="2026-04-16T10:00:00", ok=True,
        )

        rows_a = executor_runs_db.get_tool_calls(run_a)
        rows_b = executor_runs_db.get_tool_calls(run_b)
        assert {r["tool_name"] for r in rows_a} == {"ToolA"}
        assert {r["tool_name"] for r in rows_b} == {"ToolB"}

    def test_update_by_id(self):
        """Passing id= updates an existing row (for token back-attribution)."""
        run_id = executor_runs_db.record_run(status="running")
        tc_id = executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:00", ok=True,
        )
        executor_runs_db.record_tool_call(
            id=tc_id, input_tokens=50, output_tokens=150,
        )
        row = executor_runs_db.get_tool_calls(run_id)[0]
        assert row["input_tokens"] == 50
        assert row["output_tokens"] == 150
        assert row["tool_name"] == "Bash"  # unchanged

    def test_unknown_keys_dropped(self):
        """Unknown keys are silently ignored — SQL injection protection."""
        run_id = executor_runs_db.record_run(status="running")
        tc_id = executor_runs_db.record_tool_call(
            run_id=run_id,
            tool_name="Bash",
            started_at="2026-04-16T10:00:00",
            ok=True,
            bogus="'; DROP TABLE executor_tool_calls; --",
        )
        assert tc_id >= 1
        rows = executor_runs_db.get_tool_calls(run_id)
        assert len(rows) == 1

    def test_empty_db_returns_empty_list(self):
        assert executor_runs_db.get_tool_calls(9999) == []

    def test_purge_cascades_to_tool_calls(self):
        """When a run row is purged, its tool_call rows go too."""
        old_run = executor_runs_db.record_run(
            jira_key="TK-old",
            started_at=(datetime.now() - timedelta(days=40)).isoformat(
                sep=" ", timespec="seconds"
            ),
            status="success",
        )
        executor_runs_db.record_tool_call(
            run_id=old_run, tool_name="Bash",
            started_at="2026-02-01T10:00:00", ok=True,
        )
        assert len(executor_runs_db.get_tool_calls(old_run)) == 1

        executor_runs_db.purge_old_runs(30)
        assert executor_runs_db.get_tool_calls(old_run) == []


class TestToolCallTotalDurationMatchesRun:
    """Acceptance: sum of per-tool durations is within 5% of the run's
    wall-clock duration when tool calls are the dominant work."""

    def test_sum_of_durations_within_5pct_of_run_duration(self):
        run_id = executor_runs_db.record_run(
            jira_key="TK-469",
            started_at="2026-04-16T10:00:00",
            status="running",
        )

        # Fake four tool calls totalling 10 seconds of work.
        expected_total_ms = 10_000
        per_call = expected_total_ms // 4
        for i in range(4):
            executor_runs_db.record_tool_call(
                run_id=run_id,
                tool_name=f"Tool{i}",
                started_at=f"2026-04-16T10:00:0{i}",
                duration_ms=per_call,
                ok=True,
            )

        # The run itself wrapped those 4 tools in a 10.2s wall-clock window.
        executor_runs_db.record_run(
            id=run_id, ended_at="2026-04-16T10:00:10.200000",
            duration_ms=10_200, status="success", exit_code=0,
        )

        rows = executor_runs_db.get_tool_calls(run_id)
        summed_ms = sum(r["duration_ms"] for r in rows)
        run_ms = executor_runs_db.get_recent()[0]["duration_ms"]
        assert abs(summed_ms - run_ms) / run_ms <= 0.05


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
