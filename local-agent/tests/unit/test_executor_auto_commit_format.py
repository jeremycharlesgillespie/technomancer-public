"""Tests for the safety-net auto-commit message format.

Pins the format so it stays in sync with ``OllamaCoder._build_commit_message``.
Per-round commits look like:

    ``[<idea>] <title> <model> r<N>: <verb> <file> (+M more)``

Safety-net auto-commits look like:

    ``[<idea>] <title> <model> auto-commit: edit <file> (+M more)``

The ``auto-commit`` sentinel replaces ``r<N>`` so the safety-net commits are
visually distinguishable in ``git log``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.executor import (
    _auto_commit_uncommitted,
    _build_auto_commit_message,
    _parse_status_porcelain_paths,
)


# ---------------------------------------------------------------------------
# _build_auto_commit_message — pure formatting
# ---------------------------------------------------------------------------

class TestBuildAutoCommitMessage:
    """Pure formatting — no subprocess, no filesystem."""

    def test_full_format_with_title_and_model(self):
        msg = _build_auto_commit_message(
            idea_id="TK-1234",
            idea_title="Add retry logic to webhook delivery",
            model="qwen3-coder:30b-a3b-q4_K_M",
            changed_paths=["agent/foo.py"],
        )
        assert msg == (
            "[TK-1234] Add retry logic to webhook delivery "
            "qwen3-coder:30b-a3b-q4_K_M auto-commit: edit agent/foo.py"
        )

    def test_multiple_files_uses_plus_more_suffix(self):
        msg = _build_auto_commit_message(
            idea_id="TK-99",
            idea_title="Fix tests",
            model="claude-sonnet",
            changed_paths=["a.py", "b.py", "c.py"],
        )
        assert msg.endswith("auto-commit: edit a.py (+2 more)")

    def test_empty_title_omitted_cleanly(self):
        msg = _build_auto_commit_message(
            idea_id="TK-1",
            idea_title="",
            model="some-model",
            changed_paths=["x.py"],
        )
        # Header should not have a stray double-space where the title was.
        assert msg == "[TK-1] some-model auto-commit: edit x.py"

    def test_empty_model_omitted_cleanly(self):
        msg = _build_auto_commit_message(
            idea_id="TK-2",
            idea_title="A title",
            model="",
            changed_paths=["x.py"],
        )
        assert msg == "[TK-2] A title auto-commit: edit x.py"

    def test_only_idea_id_when_title_and_model_blank(self):
        msg = _build_auto_commit_message(
            idea_id="TK-7",
            idea_title="",
            model="",
            changed_paths=["dir/file.py"],
        )
        assert msg == "[TK-7] auto-commit: edit dir/file.py"

    def test_whitespace_only_title_treated_as_empty(self):
        msg = _build_auto_commit_message(
            idea_id="TK-3",
            idea_title="   ",
            model="m",
            changed_paths=["a.py"],
        )
        assert msg == "[TK-3] m auto-commit: edit a.py"

    def test_uses_auto_commit_sentinel_not_round_number(self):
        """Regression — must NOT look like a per-round commit."""
        msg = _build_auto_commit_message(
            idea_id="TK-1",
            idea_title="x",
            model="y",
            changed_paths=["a.py"],
        )
        assert "auto-commit:" in msg
        assert " r0:" not in msg
        assert " r1:" not in msg

    def test_no_files_keeps_message_coherent(self):
        """Defensive — caller already filters for dirty status, but if the
        paths list is empty for any reason the message must still parse."""
        msg = _build_auto_commit_message(
            idea_id="TK-1",
            idea_title="t",
            model="m",
            changed_paths=[],
        )
        assert "auto-commit:" in msg
        assert msg.startswith("[TK-1] t m")


# ---------------------------------------------------------------------------
# _parse_status_porcelain_paths
# ---------------------------------------------------------------------------

class TestParseStatusPorcelainPaths:
    def test_modified_file(self):
        assert _parse_status_porcelain_paths(" M agent/foo.py") == ["agent/foo.py"]

    def test_untracked_file(self):
        assert _parse_status_porcelain_paths("?? new_file.py") == ["new_file.py"]

    def test_staged_modified(self):
        assert _parse_status_porcelain_paths("M  staged.py") == ["staged.py"]

    def test_multiple_lines(self):
        porcelain = " M a.py\n M b.py\n?? c.py"
        assert _parse_status_porcelain_paths(porcelain) == ["a.py", "b.py", "c.py"]

    def test_rename_uses_post_rename_path(self):
        # Renames look like: "R  old.py -> new.py"
        assert _parse_status_porcelain_paths("R  old.py -> new.py") == ["new.py"]

    def test_blank_lines_ignored(self):
        assert _parse_status_porcelain_paths(" M a.py\n\n M b.py\n") == ["a.py", "b.py"]

    def test_empty_string(self):
        assert _parse_status_porcelain_paths("") == []


# ---------------------------------------------------------------------------
# _auto_commit_uncommitted — wires the format end-to-end
# ---------------------------------------------------------------------------

class TestAutoCommitUncommittedIntegration:
    """Verify the public entry point uses the new format and threads
    ``model`` through correctly."""

    def _make_state(self):
        state = MagicMock()
        state.log = MagicMock()
        return state

    def test_uses_new_format_when_model_provided(self, tmp_path):
        state = self._make_state()
        commit_calls = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            if cmd[:2] == ["git", "status"]:
                result.stdout = " M agent/foo.py\n"
            elif cmd[:2] == ["git", "commit"]:
                # Capture the message argument for assertion.
                idx = cmd.index("-m") + 1
                commit_calls.append(cmd[idx])
            elif cmd[:3] == ["git", "rev-parse", "--short"]:
                result.stdout = "abc1234"
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            landed = _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-1234",
                state=state,
                idea_title="Test story",
                model="qwen3-coder:30b-a3b-q4_K_M",
            )

        assert landed is True
        assert len(commit_calls) == 1
        msg = commit_calls[0]
        assert msg == (
            "[TK-1234] Test story qwen3-coder:30b-a3b-q4_K_M "
            "auto-commit: edit agent/foo.py"
        )

    def test_omits_model_when_not_provided(self, tmp_path):
        """Claude-path call sites don't pass ``model`` — message must still
        be valid and not have a hanging space."""
        state = self._make_state()
        commit_calls = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:2] == ["git", "status"]:
                result.stdout = " M README.md\n"
            elif cmd[:2] == ["git", "commit"]:
                idx = cmd.index("-m") + 1
                commit_calls.append(cmd[idx])
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-9",
                state=state,
                idea_title="Fix readme",
                # model intentionally omitted — covers the Claude path
            )

        assert commit_calls == ["[TK-9] Fix readme auto-commit: edit README.md"]

    def test_returns_false_when_clean(self, tmp_path):
        state = self._make_state()

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""  # clean working tree
            result.stderr = ""
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            landed = _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-1",
                state=state,
                idea_title="x",
            )

        assert landed is False

    def test_multi_file_message(self, tmp_path):
        """Two paths dirty — message should use ``(+1 more)``."""
        state = self._make_state()
        commit_calls = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:2] == ["git", "status"]:
                result.stdout = " M a.py\n M b.py\n"
            elif cmd[:2] == ["git", "commit"]:
                idx = cmd.index("-m") + 1
                commit_calls.append(cmd[idx])
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-1",
                state=state,
                idea_title="t",
                model="m",
            )

        assert commit_calls == [
            "[TK-1] t m auto-commit: edit a.py (+1 more)"
        ]


# ---------------------------------------------------------------------------
# Regression: ``git add -A`` would balloon commits when the model touched
# ``.gitignore``. Auto-commit must stage exactly the paths git status
# reports, never sweep newly-unignored files. Observed on TK-1215 (model
# emptied .gitignore, then auto-commit committed 13 freshly-unignored
# .db files in one commit).
# ---------------------------------------------------------------------------

class TestAutoCommitStagingScope:
    def _make_state(self):
        state = MagicMock()
        state.log = MagicMock()
        return state

    def test_stages_only_reported_paths_not_dash_A(self, tmp_path):
        """The git add invocation must use explicit paths, never ``-A``."""
        state = self._make_state()
        add_invocations = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:2] == ["git", "status"]:
                # Two dirty files reported by git status.
                result.stdout = " M agent/foo.py\n M tests/test_foo.py\n"
            elif cmd[:2] == ["git", "add"]:
                add_invocations.append(list(cmd))
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-1",
                state=state,
                idea_title="t",
                model="m",
            )

        assert len(add_invocations) == 1
        cmd = add_invocations[0]
        # Never -A (the bug we're regressing against).
        assert "-A" not in cmd, f"git add -A is forbidden, got: {cmd}"
        # Exactly the reported paths land on the command line.
        assert "agent/foo.py" in cmd
        assert "tests/test_foo.py" in cmd

    def test_dash_dash_separator_for_path_safety(self, tmp_path):
        """Use ``--`` separator so paths starting with ``-`` aren't parsed
        as flags. Defensive — git status shouldn't emit such paths, but
        the separator costs nothing and protects against weird filenames."""
        state = self._make_state()
        add_invocations = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:2] == ["git", "status"]:
                result.stdout = " M weird.py\n"
            elif cmd[:2] == ["git", "add"]:
                add_invocations.append(list(cmd))
            return result

        with patch("idea_board.executor.subprocess.run", side_effect=fake_run):
            _auto_commit_uncommitted(
                project_root=tmp_path,
                idea_id="TK-1",
                state=state,
                idea_title="t",
            )

        cmd = add_invocations[0]
        assert "--" in cmd, f"missing -- separator: {cmd}"
        # The path appears AFTER --, never before.
        sep_idx = cmd.index("--")
        assert "weird.py" in cmd[sep_idx + 1:]
