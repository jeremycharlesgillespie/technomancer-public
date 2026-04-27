"""Regression tests for the AIW mechanical / cognitive split.

The OllamaCoder used to instruct the model to run `git add`, `git commit`,
and `pytest` itself via a `run_bash` tool. The Python outer loop ALSO ran
those commands, so the work was duplicated and any drift between the two
paths was a fresh class of bug — including the 842-stale-branch runaway
that hit on 2026-04-26 (commit 22f4cbe).

The split puts mechanical work (branch / test / commit / merge) entirely in
Python and gives the LLM only cognitive tools (read, write, edit, list,
search, finish). These tests pin that boundary so the next refactor can't
re-introduce the failure mode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from idea_board.ollama_coder import OllamaCoder, _TOOLS


def _make_coder(
    project_root: Path | str = "/tmp",
    idea_id: str = "TK-1234",
    story_title: str = "Add retry logic to webhook delivery",
    model: str = "qwen3-coder:30b-a3b-q4_K_M",
) -> OllamaCoder:
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    return OllamaCoder(
        prompt="Test prompt",
        project_root=project_root,
        idea_id=idea_id,
        state=state,
        model=model,
        max_turns=5,
        max_rounds=3,
        num_ctx=4096,
        story_title=story_title,
    )


class TestSystemPromptIsContextHandoff:
    """The system prompt must NOT instruct the model to run mechanical work."""

    def test_system_prompt_has_no_mechanical_instructions(self):
        coder = _make_coder()
        prompt = coder._build_system_prompt()

        forbidden_substrings = [
            "git add",
            "git commit",
            "run pytest",
            "run_bash",
            "git status",
        ]
        for forbidden in forbidden_substrings:
            assert forbidden not in prompt, (
                f"system prompt must not tell the model to {forbidden!r} — "
                f"the harness owns that work"
            )

    def test_system_prompt_directs_model_to_finish(self):
        coder = _make_coder()
        prompt = coder._build_system_prompt()
        assert "finish(" in prompt, (
            "system prompt must explicitly direct the model to call finish(...)"
        )

    def test_system_prompt_includes_project_root_and_story_id(self):
        coder = _make_coder(project_root="/tmp/foo-bar", idea_id="TK-9999")
        prompt = coder._build_system_prompt()
        assert "/tmp/foo-bar" in prompt
        assert "TK-9999" in prompt


class TestToolboxHasNoShell:
    """Six tools, no shell. Past versions exposed run_bash + verify_git_clean
    + create_branch — all gone."""

    def test_run_bash_tool_not_advertised(self):
        names = [t["function"]["name"] for t in _TOOLS]
        assert "run_bash" not in names, (
            "run_bash removed — model has no shell"
        )

    def test_legacy_git_helper_tools_not_advertised(self):
        names = [t["function"]["name"] for t in _TOOLS]
        assert "verify_git_clean" not in names
        assert "create_branch" not in names

    def test_six_cognitive_tools_present(self):
        names = {t["function"]["name"] for t in _TOOLS}
        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "search_code",
            "finish",
        }
        assert expected.issubset(names), (
            f"missing tools: {expected - names}"
        )
        # And no extras (this catches a future PR adding shell-ish tools).
        assert names == expected, (
            f"unexpected tools advertised: {names - expected}"
        )


class TestExecuteToolRejectsShell:
    """If the model still tries to call run_bash, the dispatcher returns a
    clear error and never invokes subprocess.run."""

    def test_execute_tool_rejects_run_bash_cleanly(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)

        with patch("subprocess.run") as mock_subprocess:
            result = coder._execute_tool("run_bash", {"command": "git status"})

        assert "ERROR" in result, f"expected an ERROR result, got: {result!r}"
        assert "unknown tool" in result, (
            f"expected the unknown-tool message, got: {result!r}"
        )
        assert mock_subprocess.call_count == 0, (
            "_execute_tool must not invoke subprocess.run for run_bash — "
            "that was the path the readiness-gate runaway used"
        )

    def test_execute_tool_rejects_create_branch_cleanly(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)

        with patch("subprocess.run") as mock_subprocess:
            result = coder._execute_tool(
                "create_branch", {"branch_name": "anything"}
            )

        assert "unknown tool" in result
        assert mock_subprocess.call_count == 0


class TestOuterLoopOwnsTestAndCommit:
    """The Python round loop runs pytest and commits — the model does not."""

    def test_outer_loop_runs_pytest_and_commits_on_pass(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)

        # Make _run_inner_loop a no-op that returns True ("model called finish").
        with patch.object(coder, "_run_inner_loop", return_value=True), \
             patch.object(coder, "_run_pytest", return_value={
                 "passed": True, "failing": [], "output": "",
             }) as mock_pytest, \
             patch.object(coder, "_commit_changes") as mock_commit, \
             patch.object(coder, "_get_changed_files",
                          return_value=[str(tmp_path / "agent/foo.py")]), \
             patch.object(coder, "_files_to_stage",
                          return_value=[str(tmp_path / "agent/foo.py")]), \
             patch.object(coder, "_git_branch_files_or_empty", return_value=[]), \
             patch.object(coder, "_tag_round_commits"), \
             patch.object(coder, "_count_branch_commits", return_value=0):
            coder._run_rounds()

        assert mock_pytest.call_count == 1, (
            "outer loop must run pytest exactly once for a passing round"
        )
        assert mock_commit.call_count == 1, (
            "outer loop must commit exactly once for a passing round"
        )

    def test_outer_loop_does_not_commit_on_fail(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)

        with patch.object(coder, "_run_inner_loop", return_value=True), \
             patch.object(coder, "_run_pytest", return_value={
                 "passed": False, "failing": ["t.py::a"],
                 "output": "FAILED t.py::a",
             }), \
             patch.object(coder, "_commit_changes") as mock_commit, \
             patch.object(coder, "_get_changed_files",
                          return_value=[str(tmp_path / "x.py")]), \
             patch.object(coder, "_files_to_stage",
                          return_value=[str(tmp_path / "x.py")]), \
             patch.object(coder, "_git_branch_files_or_empty", return_value=[]), \
             patch.object(coder, "_tag_round_commits"), \
             patch.object(coder, "_count_branch_commits", return_value=0):
            coder._run_rounds()

        assert mock_commit.call_count == 0, (
            "outer loop must NOT commit when tests fail — only on a passing round"
        )


class TestCommitMessageFormat:
    """Commit messages must be deterministic: ``[<id>] <title> <model> r<N>: <verb> <file> (+M more)``."""

    def test_message_format_single_file_edit(self, tmp_path):
        coder = _make_coder(
            project_root=tmp_path,
            idea_id="TK-1050",
            story_title="Refactor jira retry",
            model="qwen3-coder:30b",
        )
        files = [str(tmp_path / "agent/foo.py")]
        existing = {str(tmp_path / "agent/foo.py")}

        msg = coder._build_commit_message(
            round_num=2, files=files, pre_round_existing=existing
        )

        assert msg == "[TK-1050] Refactor jira retry qwen3-coder:30b r2: edit agent/foo.py", (
            f"unexpected commit message: {msg!r}"
        )

    def test_message_format_multi_file_with_create(self, tmp_path):
        coder = _make_coder(
            project_root=tmp_path,
            idea_id="TK-1050",
            story_title="Add retry",
            model="modelB",
        )
        files = [
            str(tmp_path / "agent/a.py"),
            str(tmp_path / "agent/b.py"),
            str(tmp_path / "agent/c.py"),
        ]
        # First file did NOT exist → verb should be "create"
        existing: set[str] = set()

        msg = coder._build_commit_message(
            round_num=0, files=files, pre_round_existing=existing
        )

        # Format: "[TK-1050] Add retry modelB r0: create agent/a.py (+2 more)"
        assert msg.startswith("[TK-1050] Add retry modelB r0: create agent/a.py")
        assert msg.endswith("(+2 more)")

    def test_message_format_omits_title_when_blank(self, tmp_path):
        coder = _make_coder(
            project_root=tmp_path, idea_id="TK-1", story_title="", model="m"
        )
        files = [str(tmp_path / "x.py")]
        msg = coder._build_commit_message(
            round_num=0, files=files, pre_round_existing=set()
        )
        # No title should mean no double-space — header is just "[TK-1] m r0: ..."
        assert msg == "[TK-1] m r0: create x.py"

    def test_commit_changes_uses_explicit_file_list(self, tmp_path):
        """When called with files=, the function must stage exactly those
        files and never recompute the set itself."""
        coder = _make_coder(project_root=tmp_path, idea_id="TK-7")
        files = [str(tmp_path / "a.py"), str(tmp_path / "b.py")]

        # _get_changed_files would return something different — confirm the
        # function ignores it when files= is passed.
        with patch.object(coder, "_get_changed_files",
                          return_value=["should_be_ignored.py"]), \
             patch("subprocess.run") as mock_subprocess:
            coder._commit_changes(round_num=1, files=files,
                                  pre_round_existing=set(files))

        # Inspect the git add call — it should contain exactly our files.
        add_calls = [
            c for c in mock_subprocess.call_args_list
            if c.args and c.args[0][:2] == ["git", "add"]
        ]
        assert len(add_calls) == 1, "expected exactly one `git add` call"
        staged = add_calls[0].args[0][2:]
        assert staged == files, (
            f"expected to stage {files}, staged {staged}"
        )


class TestFixPromptShowsUncommittedDiff:
    """When a round fails, the model's edits stay uncommitted in the working
    tree. The next round's fix prompt MUST show them so the model doesn't
    'forget' its work."""

    def test_fix_prompt_includes_uncommitted_diff_section(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)

        # Stub the two diff helpers so we don't depend on a real git repo.
        with patch.object(coder, "_git_diff_stat",
                          return_value="(no committed changes)"), \
             patch.object(coder, "_git_diff_stat_uncommitted",
                          return_value=" agent/foo.py | 12 ++++++++++++"), \
             patch.object(coder, "_tool_read_file", return_value="(content)"):
            prompt = coder._build_fix_prompt(
                round_num=1,
                test_output="FAILED t::a",
                changed_files=[],
                failing_tests=["t::a"],
            )

        assert "Uncommitted" in prompt, (
            "fix prompt must label the uncommitted-diff section so the model "
            "knows those edits are still in the working tree"
        )
        assert "agent/foo.py | 12" in prompt, (
            "fix prompt must include the actual git diff --stat HEAD output"
        )

    def test_fix_prompt_does_not_tell_model_to_commit(self, tmp_path):
        coder = _make_coder(project_root=tmp_path)
        with patch.object(coder, "_git_diff_stat", return_value=""), \
             patch.object(coder, "_git_diff_stat_uncommitted", return_value=""), \
             patch.object(coder, "_tool_read_file", return_value=""):
            prompt = coder._build_fix_prompt(
                round_num=1, test_output="", changed_files=[], failing_tests=[],
            )

        # The OLD fix prompt ended with "Commit when done, then call finish()."
        # The new one must not — the harness commits.
        assert "Commit when done" not in prompt
        assert "git commit" not in prompt
