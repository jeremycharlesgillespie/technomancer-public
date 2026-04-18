"""Integration tests for the executor → AIV validation hook.

Covers the wire-up between the executor's post-merge success path and
:func:`agent.aiv_hook.enqueue_for_validation`. The hand-off lives in
:mod:`idea_board.aiv_post_merge`; these tests exercise that helper
directly with both the success and failure branches of
``git merge --no-ff``.

Story TK-720:
- Successful merge path calls ``enqueue_for_validation()`` with the
  correct ``story_key`` and a non-empty ``diff_paths`` list.
- Failed merge path does NOT call ``enqueue_for_validation()``.
- Existing executor tests still pass (guarded by the full suite).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import aiv_hook
from idea_board import aiv_post_merge
from idea_board.aiv_post_merge import enqueue_merged_story


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_success(stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a ``subprocess.CompletedProcess``-shaped mock for a good merge."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


def _merge_failure(stderr: str = "merge conflict") -> MagicMock:
    """Build a ``subprocess.CompletedProcess``-shaped mock for a failed merge."""
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# Successful merge → enqueue_for_validation called with correct args
# ---------------------------------------------------------------------------


class TestSuccessfulMergeEnqueues:
    """Acceptance: a successful merge path calls ``enqueue_for_validation``
    with the idea_id as ``story_key`` and a non-empty ``diff_paths`` list."""

    def test_successful_merge_calls_enqueue_for_validation(self, tmp_path):
        """Happy path: merge returncode=0 → hook fires with idea_id + paths."""
        project_root = tmp_path

        diff_stdout = (
            "local-agent/agent/foo.py\n"
            "local-agent/tests/unit/test_foo.py\n"
        )

        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq, \
             patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=diff_stdout)

            enqueue_merged_story(
                "TK-720", _merge_success(), project_root
            )

        mock_enq.assert_called_once()

    def test_successful_merge_passes_correct_jira_key(self, tmp_path):
        """``story_key`` handed to the hook matches the idea_id exactly."""
        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq, \
             patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="local-agent/agent/foo.py\n",
            )

            enqueue_merged_story("TK-720", _merge_success(), tmp_path)

        args, kwargs = mock_enq.call_args
        if "story_key" in kwargs:
            assert kwargs["story_key"] == "TK-720"
        else:
            assert args[0] == "TK-720"

    def test_successful_merge_passes_non_empty_diff_paths(self, tmp_path):
        """The hook receives a list of the files changed by the merge commit."""
        diff_stdout = (
            "local-agent/agent/foo.py\n"
            "local-agent/idea_board/executor.py\n"
            "local-agent/tests/unit/test_executor_aiv_hook.py\n"
        )

        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq, \
             patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=diff_stdout)

            enqueue_merged_story("TK-720", _merge_success(), tmp_path)

        args, kwargs = mock_enq.call_args
        diff_paths = kwargs.get("diff_paths", args[1] if len(args) > 1 else None)

        assert isinstance(diff_paths, list), f"Expected list, got {type(diff_paths)}"
        assert len(diff_paths) > 0, "diff_paths must be non-empty on a successful merge"
        assert "local-agent/agent/foo.py" in diff_paths
        assert "local-agent/idea_board/executor.py" in diff_paths

    def test_successful_merge_invokes_git_diff_with_merge_range(self, tmp_path):
        """The helper uses ``git diff --name-only`` with an HEAD~1..HEAD range
        so it captures exactly the merge commit's file list."""
        with patch.object(aiv_post_merge, "enqueue_for_validation"), \
             patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="file.py\n")

            enqueue_merged_story("TK-720", _merge_success(), tmp_path)

        assert mock_run.called, "subprocess.run should be invoked to compute the diff"
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["git", "diff", "--name-only"], (
            f"Expected git diff --name-only ..., got {cmd[:3]}"
        )
        # The range argument should reference HEAD so we get the merge commit
        assert any("HEAD" in arg for arg in cmd[3:]), (
            f"Expected a HEAD-based range in {cmd[3:]}"
        )


# ---------------------------------------------------------------------------
# Failed merge → enqueue_for_validation NOT called
# ---------------------------------------------------------------------------


class TestFailedMergeDoesNotEnqueue:
    """Acceptance: a failed merge path must NOT call
    ``enqueue_for_validation``. We never want the AIV daemon scoring a
    story that didn't actually ship."""

    def test_failed_merge_does_not_call_enqueue(self, tmp_path):
        """merge_result.returncode=1 → no call to the hook, no git diff ran."""
        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq, \
             patch.object(aiv_post_merge.subprocess, "run") as mock_run:

            enqueue_merged_story("TK-720", _merge_failure(), tmp_path)

        mock_enq.assert_not_called()
        mock_run.assert_not_called()

    def test_non_zero_returncode_treated_as_failure(self, tmp_path):
        """Any non-zero return code (not just 1) short-circuits."""
        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq:
            for rc in (2, 128, -1):
                mock_enq.reset_mock()
                enqueue_merged_story(
                    "TK-720",
                    MagicMock(returncode=rc, stdout="", stderr="fail"),
                    tmp_path,
                )
                mock_enq.assert_not_called()

    def test_missing_returncode_attribute_treated_as_failure(self, tmp_path):
        """Defensive: if merge_result lacks returncode, we skip the hook
        rather than crashing the deploy."""
        bad_result = object()  # no ``returncode`` attribute

        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq:
            enqueue_merged_story("TK-720", bad_result, tmp_path)

        mock_enq.assert_not_called()


# ---------------------------------------------------------------------------
# Executor wire-up — the call site in idea_board.executor
# ---------------------------------------------------------------------------


class TestExecutorImportsHelper:
    """Guard that the executor actually imports ``enqueue_merged_story`` so
    a future refactor can't silently drop the hook."""

    def test_executor_module_exposes_enqueue_merged_story(self):
        """The executor must bind the helper at module level so tests can
        patch ``idea_board.executor.enqueue_merged_story`` directly."""
        from idea_board import executor

        assert hasattr(executor, "enqueue_merged_story"), (
            "idea_board.executor must import enqueue_merged_story from "
            "idea_board.aiv_post_merge so the post-merge hook remains wired up."
        )
        assert executor.enqueue_merged_story is enqueue_merged_story


# ---------------------------------------------------------------------------
# Resilience — hook errors never fail the deploy
# ---------------------------------------------------------------------------


class TestHookNeverRaises:
    """Deployment is the critical path; the validation hook must never
    raise out and break the merge pipeline."""

    def test_subprocess_failure_swallowed(self, tmp_path):
        """``git diff`` blowing up → helper returns cleanly, no exception."""
        with patch.object(aiv_post_merge, "enqueue_for_validation") as mock_enq, \
             patch.object(
                 aiv_post_merge.subprocess, "run",
                 side_effect=OSError("git missing"),
             ):
            # Must not raise
            enqueue_merged_story("TK-720", _merge_success(), tmp_path)

        # get_merged_diff_paths returns [] on error → hook still called with
        # an empty list (the AIV daemon is tolerant of empty diffs).
        mock_enq.assert_called_once()
        args, kwargs = mock_enq.call_args
        diff_paths = kwargs.get("diff_paths", args[1] if len(args) > 1 else None)
        assert diff_paths == []

    def test_enqueue_hook_exception_swallowed(self, tmp_path):
        """A SQLite error inside ``enqueue_for_validation`` is swallowed."""
        with patch.object(
            aiv_post_merge, "enqueue_for_validation",
            side_effect=RuntimeError("db locked"),
        ), patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="f.py\n")

            # Must not raise
            enqueue_merged_story("TK-720", _merge_success(), tmp_path)


# ---------------------------------------------------------------------------
# get_merged_diff_paths — the diff-computation helper
# ---------------------------------------------------------------------------


class TestGetMergedDiffPaths:
    """The diff helper is exercised by the success-path tests above; these
    cases pin down edge behaviour so regressions are loud."""

    def test_strips_blank_lines(self, tmp_path):
        with patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="a.py\n\nb.py\n   \n",
            )
            result = aiv_post_merge.get_merged_diff_paths(tmp_path)

        assert result == ["a.py", "b.py"]

    def test_returns_empty_on_nonzero_returncode(self, tmp_path):
        with patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=128, stdout="", stderr="not a git repo"
            )
            result = aiv_post_merge.get_merged_diff_paths(tmp_path)

        assert result == []

    def test_returns_empty_when_no_changes(self, tmp_path):
        with patch.object(aiv_post_merge.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="\n")
            result = aiv_post_merge.get_merged_diff_paths(tmp_path)

        assert result == []


# ---------------------------------------------------------------------------
# agent.aiv_hook.get_merged_diff_paths — standalone diff helper (TK-718)
# ---------------------------------------------------------------------------


class TestAgentAivHookGetMergedDiffPaths:
    """Tests for :func:`agent.aiv_hook.get_merged_diff_paths`.

    The helper wraps ``git diff --name-only {merge_base_ref}..{head_ref}``
    so the validation hook can compute its input list without reaching
    into the executor.
    """

    def test_parses_stdout_into_list_of_paths(self):
        """Typical output: two newline-separated paths → two-element list."""
        with patch.object(aiv_hook.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="src/a.py\nsrc/b.py\n", stderr=""
            )

            result = aiv_hook.get_merged_diff_paths("main")

        assert result == ["src/a.py", "src/b.py"]

    def test_returns_empty_and_logs_on_subprocess_exception(self, caplog):
        """If ``subprocess.run`` raises, return ``[]`` and log the error."""
        with patch.object(
            aiv_hook.subprocess, "run", side_effect=OSError("git missing")
        ):
            with caplog.at_level("ERROR", logger="agent.aiv_hook"):
                result = aiv_hook.get_merged_diff_paths("main", "HEAD")

        assert result == []
        assert any(
            "get_merged_diff_paths" in rec.message and rec.levelname == "ERROR"
            for rec in caplog.records
        ), f"Expected an ERROR log from agent.aiv_hook, got: {caplog.records}"

    def test_git_command_uses_both_refs(self):
        """The range arg to ``git diff`` must embed both merge_base_ref and
        head_ref in ``{base}..{head}`` form."""
        with patch.object(aiv_hook.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            aiv_hook.get_merged_diff_paths("origin/main", "feature-branch")

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["git", "diff", "--name-only"], (
            f"Expected git diff --name-only ..., got {cmd[:3]}"
        )
        assert cmd[3] == "origin/main..feature-branch", (
            f"Expected range 'origin/main..feature-branch', got {cmd[3]!r}"
        )
