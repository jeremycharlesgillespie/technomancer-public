"""Tests for cleanup_branches.py — prune merged safe_update branches."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# cleanup_branches.py lives in local-agent/ (sibling of agent/), not inside
# the agent package, so add it to sys.path before importing.
_SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cleanup_branches  # noqa: E402

SECONDS_PER_DAY = cleanup_branches.SECONDS_PER_DAY


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> str:
    """Run a subprocess, assert it succeeds, return stdout."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"command failed: {' '.join(cmd)}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.stdout


def _git(args: list[str], cwd: Path, env: dict | None = None) -> str:
    return _run(["git", *args], cwd=cwd, env=env)


def _commit_with_date(repo: Path, message: str, days_ago: int) -> str:
    """Create a commit backdated by `days_ago`. Returns the commit SHA."""
    ts = int(time.time() - days_ago * SECONDS_PER_DAY)
    iso = f"{ts} +0000"
    # Write a unique file so every commit is non-empty. Sanitize tag so it is
    # a legal filename on Windows (no slashes, colons, etc.).
    raw = f"{message}-{ts}"
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    (repo / f"file-{tag}.txt").write_text(tag, encoding="utf-8")
    _git(["add", "-A"], cwd=repo)
    env = {
        "GIT_AUTHOR_DATE": iso,
        "GIT_COMMITTER_DATE": iso,
    }
    _git(["commit", "-m", message], cwd=repo, env=env)
    sha = _git(["rev-parse", "HEAD"], cwd=repo).strip()
    return sha


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a fresh git repo with `main` as the default branch."""
    _git(["init", "-b", "main", str(tmp_path)], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _git(["config", "user.name", "Test"], cwd=tmp_path)
    _git(["config", "commit.gpgsign", "false"], cwd=tmp_path)
    # Initial commit on main so `--merged main` works.
    _commit_with_date(tmp_path, "init", days_ago=30)
    return tmp_path


def _make_branch(repo: Path, name: str, days_ago: int) -> str:
    """Create a branch off main with one backdated commit, return its SHA."""
    _git(["checkout", "-b", name, "main"], cwd=repo)
    sha = _commit_with_date(repo, f"commit on {name}", days_ago=days_ago)
    # Fast-forward main so the branch is "merged into main".
    _git(["checkout", "main"], cwd=repo)
    _git(["merge", "--ff-only", name], cwd=repo)
    return sha


def _make_unmerged_branch(repo: Path, name: str, days_ago: int) -> str:
    """Create a branch whose commit is NOT merged into main."""
    _git(["checkout", "-b", name, "main"], cwd=repo)
    sha = _commit_with_date(repo, f"unmerged {name}", days_ago=days_ago)
    _git(["checkout", "main"], cwd=repo)
    return sha


class TestFindCandidates:
    def test_old_merged_branch_is_candidate(self, git_repo: Path):
        _make_branch(git_repo, "2026-01-01-120000-TK-100", days_ago=30)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-100" in names

    def test_recent_branch_is_skipped(self, git_repo: Path):
        _make_branch(git_repo, "2026-04-10-120000-TK-200", days_ago=3)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "2026-04-10-120000-TK-200" not in names

    def test_non_matching_name_is_skipped(self, git_repo: Path):
        _make_branch(git_repo, "feature/some-name", days_ago=30)
        _make_branch(git_repo, "random-branch", days_ago=30)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "feature/some-name" not in names
        assert "random-branch" not in names

    def test_unmerged_branch_is_skipped(self, git_repo: Path):
        _make_unmerged_branch(git_repo, "2026-01-01-120000-TK-300", days_ago=30)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-300" not in names

    def test_main_is_never_candidate(self, git_repo: Path):
        candidates = cleanup_branches.find_candidates(cwd=git_repo)
        names = [c.name for c in candidates]
        assert "main" not in names

    def test_current_branch_is_skipped(self, git_repo: Path):
        _make_branch(git_repo, "2026-01-01-120000-TK-400", days_ago=30)
        # Check the branch out — should now be excluded.
        _git(["checkout", "2026-01-01-120000-TK-400"], cwd=git_repo)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-400" not in names

    def test_active_safe_update_branch_is_skipped(
        self, git_repo: Path, tmp_path: Path, monkeypatch
    ):
        _make_branch(git_repo, "2026-01-01-120000-TK-500", days_ago=30)
        _make_branch(git_repo, "2026-01-02-120000-TK-501", days_ago=30)

        state_file = tmp_path / ".safe_update_state"
        state_file.write_text("2026-01-01-120000-TK-500", encoding="utf-8")
        monkeypatch.setattr(cleanup_branches, "STATE_FILE", state_file)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-500" not in names
        assert "2026-01-02-120000-TK-501" in names

    def test_exactly_seven_days_old_is_skipped(self, git_repo: Path):
        # Boundary: age == 7 days should NOT be a candidate ("older than 7").
        # Pin `now` to the commit time + exactly 7 days so wall-clock drift
        # between commit creation and find_candidates doesn't push us over.
        _make_branch(git_repo, "2026-01-01-120000-TK-600", days_ago=7)
        tip_ts = cleanup_branches.get_tip_timestamp(
            "2026-01-01-120000-TK-600", cwd=git_repo
        )
        frozen_now = tip_ts + 7 * SECONDS_PER_DAY

        candidates = cleanup_branches.find_candidates(
            cwd=git_repo, now=frozen_now
        )

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-600" not in names

    def test_just_over_seven_days_old_is_candidate(self, git_repo: Path):
        # One second past 7 days should be pruneable.
        _make_branch(git_repo, "2026-01-01-120000-TK-601", days_ago=7)
        tip_ts = cleanup_branches.get_tip_timestamp(
            "2026-01-01-120000-TK-601", cwd=git_repo
        )
        frozen_now = tip_ts + 7 * SECONDS_PER_DAY + 1

        candidates = cleanup_branches.find_candidates(
            cwd=git_repo, now=frozen_now
        )

        names = [c.name for c in candidates]
        assert "2026-01-01-120000-TK-601" in names

    def test_candidate_fields_populated(self, git_repo: Path):
        sha = _make_branch(git_repo, "2026-01-01-120000-TK-700", days_ago=30)

        candidates = cleanup_branches.find_candidates(cwd=git_repo)
        match = [c for c in candidates if c.name == "2026-01-01-120000-TK-700"]
        assert len(match) == 1
        c = match[0]
        assert c.age_days >= 7
        # The stored SHA is short (git log --format=%h); compare prefix.
        assert sha.startswith(c.sha) or c.sha.startswith(sha[:7])


class TestFormatTable:
    def test_empty_candidates(self):
        output = cleanup_branches.format_table([])
        assert "branch | age_days | last_commit_sha" in output
        assert "(no candidates)" in output

    def test_renders_rows(self):
        rows = [
            cleanup_branches.Candidate("2026-01-01-120000-TK-1", 30, "abc1234"),
            cleanup_branches.Candidate("2026-01-02-120000-TK-2", 15, "def5678"),
        ]
        output = cleanup_branches.format_table(rows)
        assert "2026-01-01-120000-TK-1 | 30 | abc1234" in output
        assert "2026-01-02-120000-TK-2 | 15 | def5678" in output


class TestDeleteBranch:
    def test_deletes_merged_branch(self, git_repo: Path):
        _make_branch(git_repo, "2026-01-01-120000-TK-800", days_ago=30)
        # Reset HEAD away from the branch so it can be deleted.
        _git(["checkout", "main"], cwd=git_repo)

        ok, message = cleanup_branches.delete_branch(
            "2026-01-01-120000-TK-800", cwd=git_repo
        )
        assert ok, message

        remaining = cleanup_branches.get_merged_branches(cwd=git_repo)
        assert "2026-01-01-120000-TK-800" not in remaining

    def test_refuses_unmerged_branch(self, git_repo: Path):
        # `git branch -d` (safe form) refuses unmerged branches.
        _make_unmerged_branch(git_repo, "2026-01-01-120000-TK-900", days_ago=30)

        ok, message = cleanup_branches.delete_branch(
            "2026-01-01-120000-TK-900", cwd=git_repo
        )
        assert not ok
        assert message  # error text surfaced


class TestMainCLI:
    def test_dry_run_default_does_not_delete(
        self, git_repo: Path, monkeypatch, capsys
    ):
        _make_branch(git_repo, "2026-01-01-120000-TK-A00", days_ago=30)

        monkeypatch.setattr(cleanup_branches, "REPO_ROOT", git_repo)
        # Ensure no leftover state file from the real repo interferes.
        monkeypatch.setattr(
            cleanup_branches, "STATE_FILE", git_repo / ".safe_update_state"
        )

        rc = cleanup_branches.main([])
        assert rc == 0

        captured = capsys.readouterr().out
        assert "2026-01-01-120000-TK-A00" in captured

        # Still present — dry-run.
        remaining = cleanup_branches.get_merged_branches(cwd=git_repo)
        assert "2026-01-01-120000-TK-A00" in remaining

    def test_delete_flag_removes_candidates(
        self, git_repo: Path, monkeypatch, capsys
    ):
        _make_branch(git_repo, "2026-01-01-120000-TK-B00", days_ago=30)
        _git(["checkout", "main"], cwd=git_repo)

        monkeypatch.setattr(cleanup_branches, "REPO_ROOT", git_repo)
        monkeypatch.setattr(
            cleanup_branches, "STATE_FILE", git_repo / ".safe_update_state"
        )

        rc = cleanup_branches.main(["--delete"])
        assert rc == 0

        remaining = cleanup_branches.get_merged_branches(cwd=git_repo)
        assert "2026-01-01-120000-TK-B00" not in remaining
