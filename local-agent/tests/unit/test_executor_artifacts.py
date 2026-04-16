"""Tests for the per-run artifact archive in agent.executor_runs_db.

Covers:
* archive_run writes stdout.log, stderr.log, diff.patch with expected content
* prune_old_artifacts keeps exactly N most-recent run dirs
* schema migration is idempotent — old DBs gain run_id / artifacts_path
"""

import sqlite3
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent import executor_runs_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect DB and artifact storage to tmp dirs for each test."""
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    artifacts_dir = tmp_path / "executor_artifacts"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", db_dir)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_dir / "executor_runs.db")
    monkeypatch.setattr(executor_runs_db, "ARTIFACTS_DIR", artifacts_dir)
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()
    yield
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


def _stub_diff(text="diff --git a/x b/x\n+hi\n"):
    """Stub subprocess.run so _capture_branch_diff returns the given text."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = text
    fake.stderr = ""
    return patch("agent.executor_runs_db.subprocess.run", return_value=fake)


# ---------------------------------------------------------------------------
# archive_run
# ---------------------------------------------------------------------------


class TestArchiveRunWritesFiles:
    def test_creates_all_three_files(self):
        with _stub_diff():
            target = executor_runs_db.archive_run(
                "20260416-100000-TK-452",
                stdout="hello stdout",
                stderr="hello stderr",
                branch_name="feature-branch",
            )
        assert target.is_dir()
        assert (target / "stdout.log").read_text(encoding="utf-8") == "hello stdout"
        assert (target / "stderr.log").read_text(encoding="utf-8") == "hello stderr"
        assert "diff --git" in (target / "diff.patch").read_text(encoding="utf-8")

    def test_directory_name_matches_run_id(self):
        with _stub_diff():
            target = executor_runs_db.archive_run(
                "20260416-100001-TK-1", "out", "err", "branch"
            )
        assert target.name == "20260416-100001-TK-1"
        assert target.parent == executor_runs_db.ARTIFACTS_DIR

    def test_handles_empty_stdout_stderr(self):
        with _stub_diff():
            target = executor_runs_db.archive_run(
                "20260416-100002-TK-2", "", "", "branch"
            )
        assert (target / "stdout.log").read_text(encoding="utf-8") == ""
        assert (target / "stderr.log").read_text(encoding="utf-8") == ""

    def test_overwrites_existing_dir(self):
        """Re-archiving the same run_id replaces the old contents."""
        with _stub_diff():
            executor_runs_db.archive_run(
                "20260416-100003-TK-3", "first", "first-err", "branch"
            )
            target = executor_runs_db.archive_run(
                "20260416-100003-TK-3", "second", "second-err", "branch"
            )
        assert (target / "stdout.log").read_text(encoding="utf-8") == "second"
        assert (target / "stderr.log").read_text(encoding="utf-8") == "second-err"

    def test_records_path_in_sqlite_row(self):
        """archive_run updates artifacts_path on the matching run_id row."""
        executor_runs_db.record_run(
            run_id="20260416-100004-TK-4",
            jira_key="TK-4",
            status="running",
        )
        with _stub_diff():
            target = executor_runs_db.archive_run(
                "20260416-100004-TK-4", "out", "err", "branch"
            )
        rows = executor_runs_db.get_recent()
        assert rows[0]["artifacts_path"] == str(target)

    def test_no_matching_row_is_silent(self):
        """If no row exists for the run_id, archive_run still writes files."""
        with _stub_diff():
            target = executor_runs_db.archive_run(
                "20260416-100005-TK-5", "out", "err", "branch"
            )
        assert target.is_dir()


class TestArchiveRunDiffCapture:
    def test_calls_git_diff_with_branch(self):
        with patch("agent.executor_runs_db.subprocess.run") as run_mock:
            run_mock.return_value = MagicMock(returncode=0, stdout="patch", stderr="")
            executor_runs_db.archive_run(
                "20260416-110000-TK-6", "o", "e", "feature-branch"
            )
            call_args = run_mock.call_args.args[0]
            assert call_args[:3] == ["git", "diff", "main...feature-branch"]

    def test_diff_failure_writes_error_marker(self):
        """If git diff fails, the file gets an error marker — never empty."""
        fake = MagicMock(returncode=128, stdout="", stderr="bad object")
        with patch("agent.executor_runs_db.subprocess.run", return_value=fake):
            target = executor_runs_db.archive_run(
                "20260416-110001-TK-7", "o", "e", "missing-branch"
            )
        content = (target / "diff.patch").read_text(encoding="utf-8")
        assert "# ERROR" in content
        assert "bad object" in content

    def test_diff_subprocess_exception_is_caught(self):
        with patch(
            "agent.executor_runs_db.subprocess.run",
            side_effect=FileNotFoundError("git not on PATH"),
        ):
            target = executor_runs_db.archive_run(
                "20260416-110002-TK-8", "o", "e", "branch"
            )
        content = (target / "diff.patch").read_text(encoding="utf-8")
        assert "# ERROR" in content

    def test_no_branch_name_writes_error(self):
        target = executor_runs_db.archive_run(
            "20260416-110003-TK-9", "o", "e", None
        )
        content = (target / "diff.patch").read_text(encoding="utf-8")
        assert "# ERROR" in content

    def test_diff_timeout_is_caught(self):
        with patch(
            "agent.executor_runs_db.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30),
        ):
            target = executor_runs_db.archive_run(
                "20260416-110004-TK-10", "o", "e", "branch"
            )
        content = (target / "diff.patch").read_text(encoding="utf-8")
        assert "# ERROR" in content


# ---------------------------------------------------------------------------
# prune_old_artifacts
# ---------------------------------------------------------------------------


def _make_dirs(n: int):
    """Create n sortable artifact directories with a token file inside."""
    for i in range(n):
        d = executor_runs_db.ARTIFACTS_DIR / f"20260416-{i:06d}-TK-{i}"
        d.mkdir(parents=True)
        (d / "stdout.log").write_text(f"run {i}", encoding="utf-8")


class TestPruneOldArtifacts:
    def test_no_op_when_dir_missing(self):
        # Fixture didn't create ARTIFACTS_DIR — sanity check
        assert not executor_runs_db.ARTIFACTS_DIR.exists()
        assert executor_runs_db.prune_old_artifacts(keep=10) == 0

    def test_no_op_when_under_cap(self):
        _make_dirs(5)
        assert executor_runs_db.prune_old_artifacts(keep=10) == 0
        assert len(list(executor_runs_db.ARTIFACTS_DIR.iterdir())) == 5

    def test_keeps_exactly_n_most_recent(self):
        _make_dirs(60)
        deleted = executor_runs_db.prune_old_artifacts(keep=50)
        assert deleted == 10
        remaining = sorted(p.name for p in executor_runs_db.ARTIFACTS_DIR.iterdir())
        assert len(remaining) == 50
        # The 10 oldest (000000–000009) are gone; 000010–000059 remain
        assert remaining[0] == "20260416-000010-TK-10"
        assert remaining[-1] == "20260416-000059-TK-59"

    def test_keep_zero_deletes_all(self):
        _make_dirs(3)
        assert executor_runs_db.prune_old_artifacts(keep=0) == 3
        assert list(executor_runs_db.ARTIFACTS_DIR.iterdir()) == []

    def test_ignores_non_directory_entries(self):
        executor_runs_db.ARTIFACTS_DIR.mkdir()
        (executor_runs_db.ARTIFACTS_DIR / "stray.txt").write_text("hi", encoding="utf-8")
        _make_dirs(3)
        # 3 dirs + 1 file present; with keep=2 only one DIR is pruned, file untouched
        deleted = executor_runs_db.prune_old_artifacts(keep=2)
        assert deleted == 1
        assert (executor_runs_db.ARTIFACTS_DIR / "stray.txt").exists()

    def test_archive_run_triggers_prune_at_cap(self, monkeypatch):
        monkeypatch.setattr(executor_runs_db, "MAX_ARTIFACTS", 3)
        _make_dirs(3)  # at the cap
        with _stub_diff():
            executor_runs_db.archive_run(
                "20260416-999999-TK-new", "o", "e", "branch"
            )
        # 3 + 1 - prune(keep=3) = 3 dirs left, oldest gone, new one present
        names = sorted(p.name for p in executor_runs_db.ARTIFACTS_DIR.iterdir())
        assert len(names) == 3
        assert "20260416-999999-TK-new" in names
        assert "20260416-000000-TK-0" not in names


# ---------------------------------------------------------------------------
# Schema migration — idempotent ALTER TABLE on older DBs
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_init_db_idempotent_on_new_db(self):
        executor_runs_db.init_db()
        executor_runs_db.init_db()
        executor_runs_db.init_db()
        # Still has both new columns
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
        }
        assert "run_id" in cols
        assert "artifacts_path" in cols

    def test_migrates_legacy_db_without_new_columns(self, tmp_path, monkeypatch):
        """Build a pre-migration DB by hand and verify init_db adds the columns."""
        # Wipe the connection / db that the autouse fixture created
        conn = getattr(executor_runs_db._local, "conn", None)
        if conn:
            conn.close()
            executor_runs_db._local.conn = None

        legacy_path = tmp_path / "legacy.db"
        monkeypatch.setattr(executor_runs_db, "DB_PATH", legacy_path)

        # Hand-roll the legacy schema (no run_id, no artifacts_path)
        legacy = sqlite3.connect(str(legacy_path))
        legacy.execute("""
            CREATE TABLE executor_runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                jira_key     TEXT,
                branch       TEXT,
                started_at   TEXT,
                ended_at     TEXT,
                duration_ms  INTEGER,
                cost_usd     REAL,
                status       TEXT,
                exit_code    INTEGER,
                tests_passed INTEGER,
                deployed     INTEGER
            )
        """)
        legacy.execute(
            "INSERT INTO executor_runs (jira_key, status) VALUES ('TK-old', 'success')"
        )
        legacy.commit()
        legacy.close()

        # init_db should ALTER in the missing columns and not crash
        executor_runs_db.init_db()
        executor_runs_db.init_db()  # second call still safe

        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(executor_runs)").fetchall()
        }
        assert "run_id" in cols
        assert "artifacts_path" in cols
        # Pre-existing data is preserved
        existing = conn.execute(
            "SELECT jira_key, status FROM executor_runs"
        ).fetchone()
        assert existing["jira_key"] == "TK-old"
