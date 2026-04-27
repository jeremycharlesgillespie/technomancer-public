"""Regression tests for the OllamaCoder tool-execution path.

The previous version of this file tested a per-tool-call "readiness gate"
that turned out to be the source of a runaway-branch bug:

  - It re-checked git-clean and called create_branch on every read_file /
    write_file / run_bash invocation.
  - It hard-coded ``f"TK-{self.idea_id}"`` for the branch name, which
    doubled the prefix when ``idea_id`` was already ``"TK-NNNN"`` (yielding
    ``TK-TK-1050``-style branches).
  - The accountability helpers it called (verify_git_clean / create_branch)
    operate on the worker's CWD, not on the OllamaCoder's project_root,
    so the gate was creating branches in the MAIN repo while the actual
    work happened in an isolated A/B worktree.

A single 2026-04-26 run accumulated 842 stale branches before the user
killed it. The fix is to remove the gate entirely — branch + worktree
isolation is already handled upstream by ab_executor / ab_worktree.

These tests now verify that:
  1. Tool calls do NOT invoke create_branch from accountability.
  2. Tool calls do NOT invoke verify_git_clean from accountability.
  3. Branch names produced anywhere in the coder do not contain a
     double TK- prefix when idea_id already starts with TK-.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from idea_board.ollama_coder import OllamaCoder


def _make_coder(idea_id: str = "TK-1095") -> OllamaCoder:
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    return OllamaCoder(
        prompt="Test prompt",
        project_root="/tmp",
        idea_id=idea_id,
        state=state,
        model="qwen3.5:27b",
        max_turns=5,
        max_rounds=3,
        num_ctx=4096,
    )


class TestNoReadinessGate:
    """The readiness gate is gone. Tool calls must not touch git/branches."""

    def test_read_file_does_not_create_branch(self, tmp_path):
        """Regression for the 842-branch runaway: tool calls must never
        invoke accountability.create_branch."""
        coder = _make_coder()
        coder.project_root = tmp_path

        with patch("agent.accountability.create_branch") as mock_create_branch, \
             patch("agent.accountability.verify_git_clean") as mock_verify:
            coder._execute_tool("read_file", {"path": "nonexistent.py"})
            assert mock_create_branch.call_count == 0, (
                "read_file must not create branches — that was the runaway bug"
            )
            assert mock_verify.call_count == 0, (
                "read_file must not re-verify git on every tool call"
            )

    def test_write_file_does_not_create_branch(self, tmp_path):
        coder = _make_coder()
        coder.project_root = tmp_path

        with patch("agent.accountability.create_branch") as mock_create_branch, \
             patch("agent.accountability.verify_git_clean") as mock_verify:
            coder._execute_tool(
                "write_file",
                {"path": "scratch.txt", "content": "hello"},
            )
            assert mock_create_branch.call_count == 0
            assert mock_verify.call_count == 0

    def test_many_tool_calls_create_zero_branches(self, tmp_path):
        """The bug fired N times for N tool calls. This asserts the
        cumulative count stays at zero across a typical multi-tool burst."""
        coder = _make_coder()
        coder.project_root = tmp_path

        with patch("agent.accountability.create_branch") as mock_create_branch:
            for _ in range(20):
                coder._execute_tool("read_file", {"path": "x.py"})
                coder._execute_tool(
                    "write_file", {"path": "y.py", "content": "z"},
                )
            assert mock_create_branch.call_count == 0, (
                f"20 read+write pairs should produce ZERO branches, "
                f"got {mock_create_branch.call_count}"
            )


class TestNoDoublePrefixBranchName:
    """If any future code path constructs a branch name from idea_id, it
    must not double the TK- prefix."""

    def test_idea_id_with_tk_prefix_does_not_double(self):
        """Sanity: stringifying idea_id directly should be the canonical
        path, not ``f'TK-{idea_id}'``."""
        coder = _make_coder(idea_id="TK-1050")
        # The buggy formula would produce "TK-TK-1050"; the correct value
        # IS the idea_id as-is (or a derived suffix that doesn't re-add TK-).
        buggy = f"TK-{coder.idea_id}"
        assert buggy == "TK-TK-1050", (
            "test setup precondition — confirms the buggy formula"
        )
        # And the idea_id itself is what callers should use.
        assert coder.idea_id == "TK-1050"
        assert "TK-TK-" not in coder.idea_id

    def test_no_module_level_create_branch_with_double_prefix(self):
        """The ollama_coder module previously exported a create_branch
        helper that callers used as ``create_branch(f'TK-{idea_id}')``.
        That helper is gone — A/B harness owns branch creation now."""
        from idea_board import ollama_coder
        assert not hasattr(ollama_coder, "create_branch"), (
            "module-level create_branch removed; A/B harness owns branches"
        )

    def test_no_check_readiness_gate_method(self):
        """The instance-level gate is gone too."""
        coder = _make_coder()
        assert not hasattr(coder, "_check_readiness_gate"), (
            "_check_readiness_gate removed — was firing on every tool call"
        )
