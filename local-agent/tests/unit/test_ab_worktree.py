"""Tests for ``idea_board.ab_worktree``.

Covers:
- ``short_uuid`` returns 8-char hex slugs.
- ``worktree_path_for`` returns sibling directory.
- ``create_worktree`` runs ``git worktree add --detach <path> main`` and
  returns the path; raises on failure / clobber / missing target.
- ``remove_worktree`` runs ``git worktree remove --force`` then prunes;
  falls back to ``shutil.rmtree`` when git fails; never raises.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board import ab_worktree


class TestShortUuid:
    def test_returns_8_lowercase_hex_chars(self):
        for _ in range(20):
            slug = ab_worktree.short_uuid()
            assert len(slug) == 8
            assert re.fullmatch(r"[0-9a-f]{8}", slug)

    def test_each_call_is_unique(self):
        """Collisions are theoretically possible (1/4 billion) but
        we should never see one in a small sample."""
        slugs = {ab_worktree.short_uuid() for _ in range(1000)}
        assert len(slugs) == 1000


class TestWorktreePathFor:
    def test_returns_sibling_with_correct_name(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        result = ab_worktree.worktree_path_for(repo, "abc12345")
        assert result == tmp_path / "technomancer-aiw-abc12345"

    def test_sibling_not_child(self, tmp_path):
        """``git worktree add`` rejects paths inside the repo, so the
        result must NOT be a child of repo_root."""
        repo = tmp_path / "myrepo"
        repo.mkdir()
        result = ab_worktree.worktree_path_for(repo, "feedf00d")
        assert repo not in result.parents


class TestCreateWorktree:
    def test_invokes_git_worktree_add_with_correct_args(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        target = tmp_path / "technomancer-aiw-abc12345"

        captured: dict = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            target.mkdir()  # simulate git creating the worktree dir
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(ab_worktree.subprocess, "run", fake_run)

        result = ab_worktree.create_worktree(repo, "abc12345")
        assert result == target
        assert captured["cmd"][:3] == ["git", "-C", str(repo)]
        assert "worktree" in captured["cmd"]
        assert "add" in captured["cmd"]
        assert "--detach" in captured["cmd"]
        assert str(target) in captured["cmd"]
        assert "main" in captured["cmd"]

    def test_raises_when_target_already_exists(self, tmp_path, monkeypatch):
        """A pre-existing path is a stale-from-crash signal — surface it
        instead of silently clobbering."""
        repo = tmp_path / "repo"
        repo.mkdir()
        target = tmp_path / "technomancer-aiw-deadbeef"
        target.mkdir()  # stale dir

        with pytest.raises(RuntimeError, match="already exists"):
            ab_worktree.create_worktree(repo, "deadbeef")

    def test_raises_on_git_nonzero_exit(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()

        monkeypatch.setattr(
            ab_worktree.subprocess, "run",
            lambda cmd, **kw: MagicMock(returncode=128, stdout="", stderr="fatal: bad ref"),
        )

        with pytest.raises(RuntimeError, match="exit=128"):
            ab_worktree.create_worktree(repo, "abc12345")

    def test_raises_on_subprocess_oserror(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()

        def boom(cmd, **kw):
            raise OSError("git not found")

        monkeypatch.setattr(ab_worktree.subprocess, "run", boom)

        with pytest.raises(RuntimeError, match="git worktree add failed"):
            ab_worktree.create_worktree(repo, "abc12345")

    def test_raises_when_git_succeeds_but_target_missing(self, tmp_path, monkeypatch):
        """Defensive: if git claims rc=0 but no directory appeared, that's
        still a failure — caller shouldn't trust the path we returned."""
        repo = tmp_path / "repo"
        repo.mkdir()
        # Note: don't create target — simulate git failing silently.

        monkeypatch.setattr(
            ab_worktree.subprocess, "run",
            lambda cmd, **kw: MagicMock(returncode=0, stdout="", stderr=""),
        )

        with pytest.raises(RuntimeError, match="reported success but"):
            ab_worktree.create_worktree(repo, "abc12345")

    def test_symlinks_env_into_worktree(self, tmp_path, monkeypatch):
        """When source has local-agent/.env, create a symlink in the worktree.

        The AIW pytest harness runs in the worktree; without .env, any test
        that reads ``settings.<X>`` for an env-backed value fails (TK-1215).
        Symlink (not copy) so the worktree always sees current values.
        """
        repo = tmp_path / "repo"
        (repo / "local-agent").mkdir(parents=True)
        src_env = repo / "local-agent" / ".env"
        src_env.write_text("JIRA_PROJECT_KEY=TK\n")

        target = tmp_path / "technomancer-aiw-abc12345"

        def fake_run(cmd, **kw):
            target.mkdir()
            (target / "local-agent").mkdir()
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(ab_worktree.subprocess, "run", fake_run)

        ab_worktree.create_worktree(repo, "abc12345")

        dst_env = target / "local-agent" / ".env"
        assert dst_env.is_symlink()
        assert dst_env.resolve() == src_env.resolve()

    def test_missing_source_env_does_not_raise(self, tmp_path, monkeypatch):
        """No .env in source ⇒ no symlink, no error — best-effort path."""
        repo = tmp_path / "repo"
        (repo / "local-agent").mkdir(parents=True)
        # Note: NO .env created.

        target = tmp_path / "technomancer-aiw-abc12345"

        def fake_run(cmd, **kw):
            target.mkdir()
            (target / "local-agent").mkdir()
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(ab_worktree.subprocess, "run", fake_run)

        # Should complete normally
        result = ab_worktree.create_worktree(repo, "abc12345")
        assert result == target
        assert not (target / "local-agent" / ".env").exists()


class TestRemoveWorktree:
    def test_runs_git_worktree_remove_force_and_prune(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        wt = tmp_path / "technomancer-aiw-abc12345"
        wt.mkdir()

        commands: list = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            # Simulate git removing the directory.
            if "remove" in cmd and wt.exists():
                import shutil
                shutil.rmtree(wt)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(ab_worktree.subprocess, "run", fake_run)

        ok = ab_worktree.remove_worktree(repo, wt)
        assert ok is True
        # Two commands: remove + prune.
        assert len(commands) == 2
        assert "remove" in commands[0]
        assert "--force" in commands[0]
        assert "prune" in commands[1]

    def test_returns_true_when_path_already_gone(self, tmp_path, monkeypatch):
        """Idempotent — a missing worktree path is a no-op success."""
        repo = tmp_path / "repo"
        repo.mkdir()
        wt = tmp_path / "technomancer-aiw-abc12345"  # never created

        commands: list = []
        monkeypatch.setattr(
            ab_worktree.subprocess, "run",
            lambda cmd, **kw: commands.append(cmd) or MagicMock(
                returncode=0, stdout="", stderr="",
            ),
        )

        ok = ab_worktree.remove_worktree(repo, wt)
        assert ok is True
        # Only prune ran (git worktree remove is skipped when path missing).
        assert any("prune" in c for c in commands)

    def test_falls_back_to_rmtree_when_git_fails(self, tmp_path, monkeypatch):
        """When ``git worktree remove`` errors but the directory still
        exists, the fallback nukes it directly."""
        repo = tmp_path / "repo"
        repo.mkdir()
        wt = tmp_path / "technomancer-aiw-abc12345"
        wt.mkdir()
        (wt / "stuck-file").write_text("uncommitted change")

        def fake_run(cmd, **kw):
            # git remove returns nonzero; doesn't delete dir.
            return MagicMock(returncode=1, stdout="", stderr="dirty tree")

        monkeypatch.setattr(ab_worktree.subprocess, "run", fake_run)

        ok = ab_worktree.remove_worktree(repo, wt)
        assert ok is True
        assert not wt.exists(), "rmtree fallback should have removed the dir"

    def test_returns_false_when_rmtree_also_fails(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        wt = tmp_path / "technomancer-aiw-abc12345"
        wt.mkdir()

        monkeypatch.setattr(
            ab_worktree.subprocess, "run",
            lambda cmd, **kw: MagicMock(returncode=1, stdout="", stderr="fail"),
        )
        # Force rmtree to fail.
        def boom(p):
            raise OSError("permission denied")

        monkeypatch.setattr(ab_worktree.shutil, "rmtree", boom)

        ok = ab_worktree.remove_worktree(repo, wt)
        assert ok is False

    def test_swallows_subprocess_oserror(self, tmp_path, monkeypatch):
        """OSError from subprocess.run (e.g. git binary missing) must not
        raise — the orchestrator's finally block depends on this."""
        repo = tmp_path / "repo"
        repo.mkdir()
        wt = tmp_path / "technomancer-aiw-abc12345"
        wt.mkdir()

        def boom(cmd, **kw):
            raise OSError("git missing")

        monkeypatch.setattr(ab_worktree.subprocess, "run", boom)

        # Returns a bool, never raises.
        result = ab_worktree.remove_worktree(repo, wt)
        assert result in (True, False)
