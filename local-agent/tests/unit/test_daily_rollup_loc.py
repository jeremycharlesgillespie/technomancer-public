"""Tests for ``agent.daily_rollup`` LOC + first-attempt-success extension.

Covers:

* ``_git_loc_counts`` against a real synthetic git repo built in a
  temp directory — verifies the numstat sum matches a hand-computed total.
* ``_first_attempt_success_count`` — verifies retry runs for the same
  ``jira_key`` are excluded and only the chronologically first run counts.
* ``compute_and_write`` integration — writes ``loc_added``, ``loc_removed``,
  ``first_attempt_success`` to the ``daily_stats`` row.
* Resilience: missing git, non-repo path, bogus date all return ``(0, 0)``
  without raising.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent import daily_rollup, daily_stats, executor_runs_db, story_timings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_dbs(tmp_path, monkeypatch):
    """Isolate every SQLite DB and the git REPO_ROOT for each test."""
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", tmp_path / "executor_runs.db")
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", tmp_path / "daily_stats.db")
    monkeypatch.setattr(story_timings, "DB_DIR", tmp_path)
    monkeypatch.setattr(story_timings, "DB_PATH", tmp_path / "story_timings.db")
    monkeypatch.setattr(daily_rollup, "REPO_ROOT", tmp_path)
    for mod in (executor_runs_db, daily_stats, story_timings):
        mod._local.__dict__.pop("conn", None)
    yield
    for mod in (executor_runs_db, daily_stats, story_timings):
        conn = getattr(mod._local, "conn", None)
        if conn is not None:
            conn.close()
            mod._local.__dict__.pop("conn", None)


def _git_available() -> bool:
    """Return True if the ``git`` binary is runnable."""
    try:
        subprocess.run(
            ["git", "--version"], capture_output=True, timeout=5, check=False
        )
        return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


requires_git = pytest.mark.skipif(
    not _git_available(), reason="git binary not available"
)


def _init_repo(path: Path) -> None:
    """Create a minimal git repo at ``path`` with author/email configured."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=path, check=True, capture_output=True,
    )
    for key, value in (
        ("user.email", "t@t.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
    ):
        subprocess.run(
            ["git", "config", key, value],
            cwd=path, check=True, capture_output=True,
        )


def _make_commit(
    repo: Path,
    filename: str,
    contents: str,
    subject: str,
    when: str,
) -> None:
    """Stage ``filename`` with ``contents`` and commit with subject + date.

    ``when`` is an ISO timestamp like ``"2026-04-17T10:00:00"`` which is
    stamped as both author and committer date so ``git log --since/--until``
    can filter by it.
    """
    (repo / filename).parent.mkdir(parents=True, exist_ok=True)
    (repo / filename).write_text(contents, encoding="utf-8")
    subprocess.run(
        ["git", "add", filename],
        cwd=repo, check=True, capture_output=True,
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": when,
        "GIT_COMMITTER_DATE": when,
    }
    subprocess.run(
        ["git", "commit", "-q", "-m", subject],
        cwd=repo, env=env, check=True, capture_output=True,
    )


def _insert_run(**fields: Any) -> None:
    """Insert a synthetic ``executor_runs`` row for test setup."""
    executor_runs_db.init_db()
    conn = executor_runs_db._get_conn()
    defaults: dict[str, Any] = {
        "status": "success",
        "started_at": "2026-04-17T10:00:00",
    }
    defaults.update(fields)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    conn.execute(
        f"INSERT INTO executor_runs ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# _git_loc_counts
# ---------------------------------------------------------------------------


@requires_git
class TestGitLocCounts:
    def test_missing_repo_returns_zeros(self, tmp_path):
        # Point at a path that doesn't exist at all.
        result = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path / "nope"
        )
        assert result == (0, 0)

    def test_non_git_directory_returns_zeros(self, tmp_path):
        # tmp_path exists but has no .git.
        result = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert result == (0, 0)

    def test_invalid_date_returns_zeros(self, tmp_path):
        _init_repo(tmp_path)
        result = daily_rollup._git_loc_counts(
            "not-a-date", "TK", repo_root=tmp_path
        )
        assert result == (0, 0)

    def test_empty_repo_returns_zeros(self, tmp_path):
        _init_repo(tmp_path)
        result = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert result == (0, 0)

    def test_loc_sum_matches_hand_computed_total(self, tmp_path):
        """AC: LOC numbers match manual sum across several matching commits."""
        _init_repo(tmp_path)
        # Two additions and one partial deletion — all on the target day,
        # all with [TK-NNN] subjects. Manual sum: +8, -2.
        _make_commit(
            tmp_path, "a.py", "1\n2\n3\n4\n5\n",
            subject="[TK-100] add a.py",
            when="2026-04-17T09:00:00",
        )  # +5, -0
        _make_commit(
            tmp_path, "b.py", "1\n2\n3\n",
            subject="[TK-101] add b.py",
            when="2026-04-17T10:00:00",
        )  # +3, -0
        _make_commit(
            tmp_path, "a.py", "1\n3\n5\n",
            subject="[TK-102] trim a.py",
            when="2026-04-17T11:00:00",
        )  # +0, -2

        added, removed = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert added == 8
        assert removed == 2

    def test_excludes_commits_outside_date_window(self, tmp_path):
        _init_repo(tmp_path)
        # Yesterday — must not be counted.
        _make_commit(
            tmp_path, "old.py", "x\ny\n",
            subject="[TK-1] yesterday",
            when="2026-04-16T23:59:00",
        )
        # Today — counted.
        _make_commit(
            tmp_path, "new.py", "a\nb\nc\n",
            subject="[TK-2] today",
            when="2026-04-17T10:00:00",
        )
        # Tomorrow — must not be counted.
        _make_commit(
            tmp_path, "later.py", "q\n",
            subject="[TK-3] tomorrow",
            when="2026-04-18T00:30:00",
        )

        added, removed = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert added == 3
        assert removed == 0

    def test_excludes_different_project(self, tmp_path):
        _init_repo(tmp_path)
        _make_commit(
            tmp_path, "tk.py", "1\n2\n",
            subject="[TK-1] my project",
            when="2026-04-17T10:00:00",
        )
        _make_commit(
            tmp_path, "fa.py", "1\n2\n3\n4\n5\n",
            subject="[FA-1] other project",
            when="2026-04-17T11:00:00",
        )

        tk = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        fa = daily_rollup._git_loc_counts(
            "2026-04-17", "FA", repo_root=tmp_path
        )
        assert tk == (2, 0)
        assert fa == (5, 0)

    def test_excludes_unprefixed_commits(self, tmp_path):
        _init_repo(tmp_path)
        _make_commit(
            tmp_path, "good.py", "x\ny\n",
            subject="[TK-1] matched",
            when="2026-04-17T10:00:00",
        )
        _make_commit(
            tmp_path, "ignored.py", "q\nr\ns\nt\n",
            subject="random drive-by commit without prefix",
            when="2026-04-17T11:00:00",
        )

        added, removed = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert added == 2
        assert removed == 0

    def test_skips_binary_files(self, tmp_path):
        """Binary files show as ``-\\t-\\t<path>`` and must not crash the sum."""
        _init_repo(tmp_path)
        _make_commit(
            tmp_path, "text.py", "one\ntwo\n",
            subject="[TK-1] text",
            when="2026-04-17T10:00:00",
        )
        # Commit a binary blob — .gitattributes marks it so git emits `-`.
        (tmp_path / ".gitattributes").write_text(
            "*.bin binary\n", encoding="utf-8"
        )
        (tmp_path / "data.bin").write_bytes(bytes(range(128)))
        subprocess.run(
            ["git", "add", ".gitattributes", "data.bin"],
            cwd=tmp_path, check=True, capture_output=True,
        )
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-04-17T11:00:00",
            "GIT_COMMITTER_DATE": "2026-04-17T11:00:00",
        }
        subprocess.run(
            ["git", "commit", "-q", "-m", "[TK-2] add binary"],
            cwd=tmp_path, env=env, check=True, capture_output=True,
        )

        added, removed = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        # Only "text.py" contributes: 2 added lines. .gitattributes is not
        # binary (it's a text file), so its line-count is included; the
        # real assertion we care about is "didn't raise on the '-' row".
        assert added >= 2
        assert removed == 0

    def test_missing_git_binary_returns_zeros(self, tmp_path, monkeypatch):
        """If ``git`` isn't on PATH, return (0, 0) instead of raising."""
        _init_repo(tmp_path)

        def _raise_file_not_found(*args, **kwargs):
            raise FileNotFoundError("git: not found")

        monkeypatch.setattr(subprocess, "run", _raise_file_not_found)
        result = daily_rollup._git_loc_counts(
            "2026-04-17", "TK", repo_root=tmp_path
        )
        assert result == (0, 0)


# ---------------------------------------------------------------------------
# _first_attempt_success_count
# ---------------------------------------------------------------------------


class TestFirstAttemptSuccessCount:
    def test_empty_db_returns_zero(self):
        executor_runs_db.init_db()
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 0

    def test_single_success_counts(self):
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-17T10:00:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 1

    def test_retry_after_failure_does_not_count(self):
        """AC: a key whose first run failed + second run succeeded is
        NOT a first-attempt success."""
        _insert_run(jira_key="TK-1", status="failed",
                    started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-17T11:30:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 0

    def test_first_success_then_retry_still_counts(self):
        """A spurious re-run after success shouldn't revoke the credit."""
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="TK-1", status="failed",
                    started_at="2026-04-17T11:30:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 1

    def test_multiple_keys_summed_independently(self):
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-17T09:00:00")
        _insert_run(jira_key="TK-2", status="failed",
                    started_at="2026-04-17T09:30:00")
        _insert_run(jira_key="TK-2", status="success",
                    started_at="2026-04-17T10:00:00")  # retry — excluded
        _insert_run(jira_key="TK-3", status="success",
                    started_at="2026-04-17T11:00:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 2

    def test_different_project_excluded(self):
        _insert_run(jira_key="FA-1", status="success",
                    started_at="2026-04-17T10:00:00")
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-17T10:30:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 1
        assert daily_rollup._first_attempt_success_count("2026-04-17", "FA") == 1

    def test_different_date_excluded(self):
        _insert_run(jira_key="TK-1", status="success",
                    started_at="2026-04-16T23:00:00")
        _insert_run(jira_key="TK-2", status="success",
                    started_at="2026-04-17T10:00:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 1

    def test_null_started_at_excluded(self):
        _insert_run(jira_key="TK-1", status="success", started_at=None)
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 0

    def test_in_flight_status_does_not_count(self):
        """running / queued are not terminal, so they shouldn't count."""
        _insert_run(jira_key="TK-1", status="running",
                    started_at="2026-04-17T10:00:00")
        assert daily_rollup._first_attempt_success_count("2026-04-17", "TK") == 0


# ---------------------------------------------------------------------------
# compute_and_write integration — LOC + first-attempt reach daily_stats
# ---------------------------------------------------------------------------


@requires_git
class TestComputeAndWriteLoc:
    def test_loc_and_first_attempt_written_to_daily_stats(self, tmp_path):
        _init_repo(tmp_path)
        _make_commit(
            tmp_path, "foo.py", "a\nb\nc\nd\n",
            subject="[TK-50] add foo",
            when="2026-04-17T10:00:00",
        )  # +4, -0
        _insert_run(jira_key="TK-50", status="success", cost_usd=0.10,
                    duration_ms=5_000, started_at="2026-04-17T10:00:00")
        # A retry key — should ship but NOT count as first-attempt success.
        _insert_run(jira_key="TK-51", status="failed", cost_usd=0.01,
                    duration_ms=1_000, started_at="2026-04-17T10:30:00")
        _insert_run(jira_key="TK-51", status="success", cost_usd=0.02,
                    duration_ms=2_000, started_at="2026-04-17T11:00:00")

        result = daily_rollup.compute_and_write("2026-04-17", "TK")

        assert result["loc_added"] == 4
        assert result["loc_removed"] == 0
        assert result["first_attempt_success"] == 1  # TK-50 only

        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT loc_added, loc_removed, first_attempt_success "
            "FROM daily_stats WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["loc_added"] == 4
        assert row["loc_removed"] == 0
        assert row["first_attempt_success"] == 1

    def test_rerun_updates_loc_values(self, tmp_path):
        """Adding a new matching commit after the first rollup should bump
        loc_added on the next rollup — the UPSERT must overwrite LOC."""
        _init_repo(tmp_path)
        _make_commit(
            tmp_path, "foo.py", "a\nb\n",
            subject="[TK-1] first",
            when="2026-04-17T10:00:00",
        )  # +2
        first = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert first["loc_added"] == 2

        _make_commit(
            tmp_path, "bar.py", "x\ny\nz\n",
            subject="[TK-2] second",
            when="2026-04-17T11:00:00",
        )  # +3
        second = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert second["loc_added"] == 5

        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT loc_added FROM daily_stats "
            "WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()
        assert row["loc_added"] == 5


class TestComputeAndWriteWithoutGit:
    def test_loc_zero_when_repo_root_is_not_a_repo(self, tmp_path):
        """Without a git repo at REPO_ROOT (already monkeypatched to
        tmp_path), compute_and_write must still succeed with zero LOC."""
        _insert_run(jira_key="TK-1", status="success", cost_usd=0.05,
                    duration_ms=3_000, started_at="2026-04-17T10:00:00")
        result = daily_rollup.compute_and_write("2026-04-17", "TK")
        assert result["loc_added"] == 0
        assert result["loc_removed"] == 0
        assert result["first_attempt_success"] == 1
