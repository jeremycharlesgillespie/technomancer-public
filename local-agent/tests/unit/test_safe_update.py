"""Tests for safe_update.py post-deploy health check and auto-rollback."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import subprocess

import pytest

# Import the module under test
import safe_update


# ---------------------------------------------------------------------------
# _extract_story_id
# ---------------------------------------------------------------------------


class TestExtractStoryId:
    """Tests for extracting story ID from branch names."""

    def test_extracts_tk_id_from_executor_branch(self):
        assert safe_update._extract_story_id("2026-04-15-221820-TK-409") == "TK-409"

    def test_extracts_tk_id_with_large_number(self):
        assert safe_update._extract_story_id("2026-04-15-143022-TK-42") == "TK-42"

    def test_returns_empty_for_non_story_branch(self):
        assert safe_update._extract_story_id("2026-04-15-143022-fix-bug") == ""

    def test_case_insensitive(self):
        result = safe_update._extract_story_id("2026-04-15-221820-tk-100")
        assert result == "tk-100"

    def test_returns_first_match_when_multiple(self):
        result = safe_update._extract_story_id("2026-04-15-TK-100-rebased-TK-200")
        assert result == "TK-100"

    def test_returns_empty_for_empty_string(self):
        assert safe_update._extract_story_id("") == ""

    def test_extracts_from_branch_with_extra_suffix(self):
        assert safe_update._extract_story_id("2026-04-15-221820-TK-409-retry") == "TK-409"


# ---------------------------------------------------------------------------
# check_bot_running
# ---------------------------------------------------------------------------


class TestCheckBotRunning:
    """Tests for the check_bot_running helper (PID file based)."""

    def test_returns_true_when_pid_alive(self, tmp_path):
        pid_file = tmp_path / "bot.pid"
        pid_file.write_text(str(os.getpid()))
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.check_bot_running() is True

    def test_returns_false_when_no_pid_file(self, tmp_path):
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.check_bot_running() is False

    def test_returns_false_when_pid_dead(self, tmp_path):
        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("999999999")
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.check_bot_running() is False

    def test_returns_false_on_invalid_pid(self, tmp_path):
        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("not-a-number")
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.check_bot_running() is False


# ---------------------------------------------------------------------------
# restart_bot — timeout-tolerant liveness verification
# ---------------------------------------------------------------------------


class TestRestartBot:
    """Tests for restart_bot()'s PID-polling verification.

    The regression these protect against: safe_update used to trust the
    subprocess return code alone. A slow-but-successful restart (Ollama
    unload + warm-start that takes >30s) was indistinguishable from a
    real failure, and triggered spurious rollbacks. Now we verify
    liveness via the PID file after the subprocess returns.
    """

    @patch("safe_update.time.sleep")
    @patch("safe_update.subprocess.run")
    @patch("safe_update.check_bot_running")
    def test_returns_true_when_subprocess_ok_and_bot_alive(
        self, mock_check, mock_run, _mock_sleep, tmp_path
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_check.return_value = True
        (tmp_path / "bot_service.py").write_text("# stub")
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.restart_bot() is True

    @patch("safe_update.time.sleep")
    @patch("safe_update.subprocess.run")
    @patch("safe_update.check_bot_running")
    def test_returns_true_when_subprocess_times_out_but_bot_alive(
        self, mock_check, mock_run, _mock_sleep, tmp_path
    ):
        # The regression case — subprocess hit its 120s timeout, but the
        # bot is already running. We must not false-positive a rollback.
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="bot_service.py start", timeout=120,
        )
        mock_check.return_value = True
        (tmp_path / "bot_service.py").write_text("# stub")
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.restart_bot() is True

    @patch("safe_update.time.sleep")
    @patch("safe_update.subprocess.run")
    @patch("safe_update.check_bot_running")
    def test_returns_false_when_subprocess_ok_but_bot_dead(
        self, mock_check, mock_run, _mock_sleep, tmp_path
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_check.return_value = False  # bot never comes up
        (tmp_path / "bot_service.py").write_text("# stub")
        with patch.object(safe_update, "SCRIPT_DIR", tmp_path):
            assert safe_update.restart_bot() is False


# ---------------------------------------------------------------------------
# post_deploy_health_check
# ---------------------------------------------------------------------------


class TestPostDeployHealthCheck:
    """Tests for the post_deploy_health_check function."""

    @patch("safe_update.check_bot_running")
    @patch("safe_update.time.sleep")
    def test_passes_when_bot_stays_alive(self, mock_sleep, mock_check):
        mock_check.return_value = True
        result = safe_update.post_deploy_health_check(
            startup_grace=0, check_interval=5, total_duration=30
        )
        assert result is True
        # 1 grace check + 6 monitoring checks = 7
        assert mock_check.call_count == 7

    @patch("safe_update.check_bot_running")
    @patch("safe_update.time.sleep")
    def test_fails_when_bot_dead_after_grace(self, mock_sleep, mock_check):
        mock_check.return_value = False
        result = safe_update.post_deploy_health_check(
            startup_grace=0, check_interval=5, total_duration=30
        )
        assert result is False
        assert mock_check.call_count == 1

    @patch("safe_update.check_bot_running")
    @patch("safe_update.time.sleep")
    def test_fails_when_bot_crashes_midway(self, mock_sleep, mock_check):
        mock_check.side_effect = [True, True, True, True, False]
        result = safe_update.post_deploy_health_check(
            startup_grace=0, check_interval=5, total_duration=30
        )
        assert result is False
        assert mock_check.call_count == 5

    @patch("safe_update.check_bot_running")
    @patch("safe_update.time.sleep")
    def test_grace_period_waits_before_check(self, mock_sleep, mock_check):
        mock_check.return_value = True
        result = safe_update.post_deploy_health_check(
            startup_grace=20, check_interval=5, total_duration=10
        )
        assert result is True
        assert mock_sleep.call_args_list[0] == call(20)


# ---------------------------------------------------------------------------
# rollback_deploy
# ---------------------------------------------------------------------------


class TestRollbackDeploy:
    """Tests for the rollback_deploy function."""

    @patch("safe_update._send_rollback_notification")
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.run_git")
    def test_successful_rollback(self, mock_git, mock_restart, mock_notify):
        # rev-parse returns short hash
        mock_git.side_effect = [
            MagicMock(stdout="abc1234\n"),  # rev-parse --short HEAD
            MagicMock(),                     # revert HEAD
            MagicMock(),                     # push origin main
        ]

        ok, msg = safe_update.rollback_deploy(reverted_branch="test-branch")

        assert ok is True
        assert "abc1234" in msg
        assert "test-branch" in msg
        # Verify git revert was called with correct args
        revert_call = mock_git.call_args_list[1]
        assert revert_call == call(["revert", "HEAD", "--no-edit", "-m", "1"])
        # Verify notification sent
        mock_notify.assert_called_once_with("abc1234", "test-branch", success=True, bot_ok=True)

    @patch("safe_update._send_rollback_notification")
    @patch("safe_update.restart_bot", return_value=False)
    @patch("safe_update.run_git")
    def test_rollback_with_bot_restart_failure(self, mock_git, mock_restart, mock_notify):
        mock_git.side_effect = [
            MagicMock(stdout="abc1234\n"),
            MagicMock(),
            MagicMock(),
        ]

        ok, msg = safe_update.rollback_deploy()

        assert ok is True
        assert "FAILED" in msg  # bot restart failed
        mock_notify.assert_called_once()
        notify_kwargs = mock_notify.call_args
        assert notify_kwargs[1]["bot_ok"] is False

    @patch("safe_update._send_rollback_notification")
    @patch("safe_update.run_git")
    def test_rollback_revert_fails(self, mock_git, mock_notify):
        mock_git.side_effect = [
            MagicMock(stdout="abc1234\n"),  # rev-parse
            safe_update.SafeUpdateError("merge conflict"),  # revert fails
        ]

        ok, msg = safe_update.rollback_deploy(reverted_branch="bad-branch")

        assert ok is False
        assert "ROLLBACK FAILED" in msg
        mock_notify.assert_called_once_with(
            "abc1234", "bad-branch", success=False, error="merge conflict"
        )

    @patch("safe_update._send_rollback_notification")
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.run_git")
    def test_rollback_push_fails_continues(self, mock_git, mock_restart, mock_notify):
        """Push failure shouldn't stop the rollback — it's a warning."""
        mock_git.side_effect = [
            MagicMock(stdout="abc1234\n"),
            MagicMock(),  # revert succeeds
            safe_update.SafeUpdateError("push rejected"),  # push fails
        ]

        ok, msg = safe_update.rollback_deploy()

        assert ok is True
        mock_restart.assert_called_once()


# ---------------------------------------------------------------------------
# _send_rollback_notification
# ---------------------------------------------------------------------------


class TestSendRollbackNotification:
    """Tests for the Discord notification helper."""

    @patch("safe_update.os.environ", {"DISCORD_WEBHOOK_URL": ""})
    @patch("safe_update.log")
    def test_skips_when_no_webhook(self, mock_log):
        """Should warn and return when no webhook is configured."""
        with patch.dict("sys.modules", {"agent.config": MagicMock(settings=MagicMock(discord_webhook_url=""))}):
            safe_update._send_rollback_notification("abc", "branch", success=True, bot_ok=True)
        # Verify it logged a warning about missing webhook
        mock_log.assert_called()

    def test_sends_success_notification(self):
        mock_settings = MagicMock()
        mock_settings.discord_webhook_url = "https://webhook.test/abc"
        mock_config = MagicMock(settings=mock_settings)
        with patch.dict("sys.modules", {"agent.config": mock_config}):
            with patch("requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=204)
                safe_update._send_rollback_notification(
                    "abc1234", "my-branch", success=True, bot_ok=True
                )
                mock_post.assert_called_once()
                payload = mock_post.call_args[1]["json"]
                assert "Auto-Rollback Triggered" in payload["content"]
                assert "abc1234" in payload["content"]
                assert "my-branch" in payload["content"]
                assert "Running" in payload["content"]

    def test_sends_failure_notification(self):
        mock_settings = MagicMock()
        mock_settings.discord_webhook_url = "https://webhook.test/abc"
        mock_config = MagicMock(settings=mock_settings)
        with patch.dict("sys.modules", {"agent.config": mock_config}):
            with patch("requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=204)
                safe_update._send_rollback_notification(
                    "abc1234", "my-branch", success=False, error="merge conflict"
                )
                mock_post.assert_called_once()
                payload = mock_post.call_args[1]["json"]
                assert "FAILED" in payload["content"]
                assert "merge conflict" in payload["content"]


# ---------------------------------------------------------------------------
# Integration: continue_workflow with health check
# ---------------------------------------------------------------------------


class TestContinueWorkflowHealthCheck:
    """Test that continue_workflow triggers rollback on post-deploy crash."""

    @patch("safe_update.run_quality_tests", return_value=True)
    @patch("safe_update.push_to_remote", return_value=True)
    @patch("safe_update.rollback_deploy", return_value=(True, "Rolled back"))
    @patch("safe_update.post_deploy_health_check", return_value=False)
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "all passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-13-test-branch")
    @patch("safe_update.load_state", return_value="2026-04-13-test-branch")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_rollback_on_health_check_failure(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_rollback,
        mock_push,
        mock_qa,
    ):
        with pytest.raises(SystemExit) as exc_info:
            safe_update.continue_workflow()
        assert exc_info.value.code == 1
        mock_rollback.assert_called_once_with(reverted_branch="2026-04-13-test-branch")

    @patch("safe_update.subprocess.run")
    @patch("safe_update.run_quality_tests", return_value=True)
    @patch("safe_update.push_to_remote", return_value=True)
    @patch("safe_update.rollback_deploy")
    @patch("safe_update.post_deploy_health_check", return_value=True)
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "all passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-13-test-branch")
    @patch("safe_update.load_state", return_value="2026-04-13-test-branch")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_no_rollback_when_healthy(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_rollback,
        mock_push,
        mock_qa,
        mock_subprocess,
    ):
        """When health check passes, rollback should NOT be called."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
        # continue_workflow will proceed through readme/publish steps; subprocess.run
        # is mocked so those won't fail. It should complete without sys.exit.
        safe_update.continue_workflow()
        mock_rollback.assert_not_called()

    @patch("safe_update.rollback_deploy", return_value=(True, "Rolled back"))
    @patch("safe_update.post_deploy_health_check")
    @patch("safe_update.restart_bot", return_value=False)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "all passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-13-test-branch")
    @patch("safe_update.load_state", return_value="2026-04-13-test-branch")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_skips_health_check_when_restart_fails(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_rollback,
    ):
        """When bot restart fails entirely, skip health check and trigger rollback."""
        with pytest.raises(SystemExit) as exc_info:
            safe_update.continue_workflow()
        assert exc_info.value.code == 1
        mock_health.assert_not_called()
        mock_rollback.assert_called_once_with(reverted_branch="2026-04-13-test-branch")

    @patch("safe_update.rollback_deploy", return_value=(True, "Rolled back"))
    @patch("safe_update.post_deploy_health_check")
    @patch("safe_update.restart_bot", return_value=False)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "all passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-13-test-branch")
    @patch("safe_update.load_state", return_value="2026-04-13-test-branch")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_rollback_when_restart_fails(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_rollback,
    ):
        """When restart_bot returns False, rollback_deploy is called and workflow exits."""
        with pytest.raises(SystemExit) as exc_info:
            safe_update.continue_workflow()
        assert exc_info.value.code == 1
        # Health check should NOT be called — nothing to monitor
        mock_health.assert_not_called()
        # Rollback MUST be triggered
        mock_rollback.assert_called_once_with(reverted_branch="2026-04-13-test-branch")


# ---------------------------------------------------------------------------
# README commit message includes story ID
# ---------------------------------------------------------------------------


class TestReadmeCommitMessage:
    """Test that README auto-commit uses story ID from branch name."""

    @patch("safe_update.subprocess.run")
    @patch("safe_update.run_quality_tests", return_value=True)
    @patch("safe_update.push_to_remote", return_value=True)
    @patch("safe_update.post_deploy_health_check", return_value=True)
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "10 passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-15-221820-TK-409")
    @patch("safe_update.load_state", return_value="2026-04-15-221820-TK-409")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_readme_commit_includes_story_id(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_push,
        mock_qa,
        mock_subprocess,
    ):
        """README commit message should contain [TK-409] when branch has story ID."""
        # generate_readme.py exists, and git diff says READMEs changed
        def subprocess_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            result = MagicMock(returncode=0, stdout="", stderr="")
            # git diff --quiet returns 1 = file changed
            if isinstance(cmd, list) and "diff" in cmd and "--quiet" in cmd:
                result.returncode = 1
            return result

        mock_subprocess.side_effect = subprocess_side_effect

        safe_update.continue_workflow()

        # Find the git commit call and verify message contains story ID
        commit_calls = [
            c for c in mock_subprocess.call_args_list
            if isinstance(c[0][0], list) and "commit" in c[0][0]
        ]
        # Should have at least one commit call with the story ID
        commit_messages = [
            c[0][0][c[0][0].index("-m") + 1]
            for c in commit_calls
            if "-m" in c[0][0]
        ]
        assert any("[TK-409]" in msg for msg in commit_messages), (
            f"Expected a commit message with [TK-409], got: {commit_messages}"
        )
        # The generic message should NOT appear
        assert not any(
            "Update README with latest stats [auto]" in msg for msg in commit_messages
        ), "Generic README commit message should not appear for story branches"

    @patch("safe_update.subprocess.run")
    @patch("safe_update.run_quality_tests", return_value=True)
    @patch("safe_update.push_to_remote", return_value=True)
    @patch("safe_update.post_deploy_health_check", return_value=True)
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "10 passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-13-fix-typo")
    @patch("safe_update.load_state", return_value="2026-04-13-fix-typo")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_readme_commit_falls_back_for_non_story_branch(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_push,
        mock_qa,
        mock_subprocess,
    ):
        """Non-story branches fall back to generic README message."""
        def subprocess_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            result = MagicMock(returncode=0, stdout="", stderr="")
            if isinstance(cmd, list) and "diff" in cmd and "--quiet" in cmd:
                result.returncode = 1
            return result

        mock_subprocess.side_effect = subprocess_side_effect

        safe_update.continue_workflow()

        commit_calls = [
            c for c in mock_subprocess.call_args_list
            if isinstance(c[0][0], list) and "commit" in c[0][0]
        ]
        commit_messages = [
            c[0][0][c[0][0].index("-m") + 1]
            for c in commit_calls
            if "-m" in c[0][0]
        ]
        assert any(
            "Update README with latest stats [auto]" in msg for msg in commit_messages
        ), f"Expected generic fallback message, got: {commit_messages}"

    @patch("safe_update.subprocess.run")
    @patch("safe_update.run_quality_tests", return_value=True)
    @patch("safe_update.push_to_remote", return_value=True)
    @patch("safe_update.post_deploy_health_check", return_value=True)
    @patch("safe_update.restart_bot", return_value=True)
    @patch("safe_update.delete_branch")
    @patch("safe_update.merge_to_main", return_value=True)
    @patch("safe_update.run_tests", return_value=(True, "10 passed"))
    @patch("safe_update.verify_clean_state", return_value=True)
    @patch("safe_update.get_current_branch", return_value="2026-04-15-221820-TK-409")
    @patch("safe_update.load_state", return_value="2026-04-15-221820-TK-409")
    @patch("safe_update.clear_state")
    @patch("safe_update.os.environ", {})
    def test_readme_skips_commit_when_unchanged(
        self,
        mock_clear,
        mock_load,
        mock_branch,
        mock_clean,
        mock_tests,
        mock_merge,
        mock_delete,
        mock_restart,
        mock_health,
        mock_push,
        mock_qa,
        mock_subprocess,
    ):
        """When README is unchanged, no commit is created (idempotency)."""
        # git diff --quiet returns 0 = no changes
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")

        safe_update.continue_workflow()

        # No git commit calls should have been made for README
        commit_calls = [
            c for c in mock_subprocess.call_args_list
            if isinstance(c[0][0], list) and "commit" in c[0][0]
        ]
        # There should be zero README commit calls
        readme_commits = [
            c for c in commit_calls
            if "-m" in c[0][0]
            and any(
                kw in c[0][0][c[0][0].index("-m") + 1]
                for kw in ["Deploy + update stats", "Update README"]
            )
        ]
        assert readme_commits == [], (
            f"Expected no README commit when unchanged, got: {readme_commits}"
        )


# ---------------------------------------------------------------------------
# _parse_failed_nodeids
# ---------------------------------------------------------------------------


def _write_junit(path: Path, testcases_xml: str) -> Path:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites>\n'
        '<testsuite name="pytest" tests="3" failures="1" errors="1">\n'
        f'{testcases_xml}\n'
        '</testsuite>\n'
        '</testsuites>\n',
        encoding="utf-8",
    )
    return path


class TestParseFailedNodeids:
    """Tests for JUnit-XML failure parsing."""

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert safe_update._parse_failed_nodeids(tmp_path / "nope.xml") == []

    def test_returns_empty_for_none_path(self):
        assert safe_update._parse_failed_nodeids(None) == []

    def test_returns_empty_for_malformed_xml(self, tmp_path):
        bad = tmp_path / "junit.xml"
        bad.write_text("not xml at all", encoding="utf-8")
        assert safe_update._parse_failed_nodeids(bad) == []

    def test_parses_failed_method_in_class(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_method" file="tests/unit/test_foo.py">'
            '<failure message="boom">AssertionError</failure></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == ["tests/unit/test_foo.py::TestThing::test_method"]

    def test_parses_failed_free_function(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo" name="test_bar" '
            'file="tests/unit/test_foo.py">'
            '<failure message="nope">AssertionError</failure></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == ["tests/unit/test_foo.py::test_bar"]

    def test_parses_error_testcase(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_broken" file="tests/unit/test_foo.py">'
            '<error message="setup failed">RuntimeError</error></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == ["tests/unit/test_foo.py::TestThing::test_broken"]

    def test_skips_passing_testcases(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_pass" file="tests/unit/test_foo.py"/>'
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_fail" file="tests/unit/test_foo.py">'
            '<failure>oops</failure></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == ["tests/unit/test_foo.py::TestThing::test_fail"]

    def test_normalizes_windows_path_separator(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_win" file="tests\\unit\\test_foo.py">'
            '<failure>err</failure></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == ["tests/unit/test_foo.py::TestThing::test_win"]

    def test_preserves_parametrize_name(self, tmp_path):
        junit = _write_junit(
            tmp_path / "junit.xml",
            '<testcase classname="tests.unit.test_foo.TestThing" '
            'name="test_param[case1-42]" file="tests/unit/test_foo.py">'
            '<failure>err</failure></testcase>',
        )
        result = safe_update._parse_failed_nodeids(junit)
        assert result == [
            "tests/unit/test_foo.py::TestThing::test_param[case1-42]"
        ]


# ---------------------------------------------------------------------------
# run_tests_with_retry
# ---------------------------------------------------------------------------


class FakePytestRunner:
    """Callable that mimics _run_pytest_subprocess with scripted responses.

    Each call pops the next (returncode, output, junit_xml_content) tuple
    from ``responses``. When ``junit_xml`` is provided, the fake writes the
    tuple's XML into it so the real ``_parse_failed_nodeids`` can consume it.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, nodeids=None, junit_xml=None):
        self.calls.append({"nodeids": nodeids, "junit_xml": junit_xml})
        rc, output, xml_content = self.responses.pop(0)
        if junit_xml is not None and xml_content:
            Path(junit_xml).write_text(xml_content, encoding="utf-8")
        return rc, output


_FAIL_XML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<testsuites><testsuite name="pytest" tests="1" failures="1">'
    '<testcase classname="tests.unit.test_foo.TestThing" '
    'name="test_flaky" file="tests/unit/test_foo.py">'
    '<failure message="timing">AssertionError</failure>'
    '</testcase></testsuite></testsuites>\n'
)


class TestRunTestsWithRetry:
    """Tests for the flaky-retry orchestrator."""

    def test_first_run_passes_returns_success_no_retry(self):
        runner = FakePytestRunner([(0, "3 passed", "")])
        logger = MagicMock()
        success, output, flaky = safe_update.run_tests_with_retry(
            runner=runner, flaky_logger=logger,
        )
        assert success is True
        assert flaky == []
        assert "3 passed" in output
        assert len(runner.calls) == 1
        logger.assert_not_called()

    def test_first_fail_then_retry_passes_returns_success(self):
        runner = FakePytestRunner([
            (1, "FAILED tests/unit/test_foo.py::TestThing::test_flaky", _FAIL_XML),
            (0, "1 passed", ""),
        ])
        logger = MagicMock()
        success, output, flaky = safe_update.run_tests_with_retry(
            runner=runner, flaky_logger=logger,
        )
        assert success is True
        assert flaky == ["tests/unit/test_foo.py::TestThing::test_flaky"]
        assert "RETRY OUTPUT" in output
        assert len(runner.calls) == 2
        assert runner.calls[0]["nodeids"] is None
        assert runner.calls[1]["nodeids"] == [
            "tests/unit/test_foo.py::TestThing::test_flaky"
        ]
        logger.assert_called_once_with(
            ["tests/unit/test_foo.py::TestThing::test_flaky"]
        )

    def test_both_runs_fail_returns_failure(self):
        runner = FakePytestRunner([
            (1, "FAILED first run", _FAIL_XML),
            (1, "FAILED retry run", ""),
        ])
        logger = MagicMock()
        success, output, flaky = safe_update.run_tests_with_retry(
            runner=runner, flaky_logger=logger,
        )
        assert success is False
        assert flaky == []
        assert "RETRY OUTPUT" in output
        assert len(runner.calls) == 2
        logger.assert_not_called()

    def test_no_parseable_failures_no_retry(self):
        """Collection/import errors have no testcases — we shouldn't retry."""
        runner = FakePytestRunner([(2, "ERROR collecting tests", "")])
        logger = MagicMock()
        success, output, flaky = safe_update.run_tests_with_retry(
            runner=runner, flaky_logger=logger,
        )
        assert success is False
        assert flaky == []
        # Only one runner call — we bailed out before the retry
        assert len(runner.calls) == 1
        logger.assert_not_called()

    def test_flaky_logger_exception_does_not_break_success(self):
        """If the flaky logger raises, we still report success."""
        runner = FakePytestRunner([
            (1, "FAILED first", _FAIL_XML),
            (0, "passed", ""),
        ])
        logger = MagicMock(side_effect=OSError("vault unreachable"))
        success, _output, flaky = safe_update.run_tests_with_retry(
            runner=runner, flaky_logger=logger,
        )
        assert success is True
        assert flaky == ["tests/unit/test_foo.py::TestThing::test_flaky"]
        logger.assert_called_once()


# ---------------------------------------------------------------------------
# _log_flaky_tests
# ---------------------------------------------------------------------------


class TestLogFlakyTests:
    """Tests for appending flaky-test notes to crash_log.md."""

    def test_appends_flaky_entry_to_crash_log(self, tmp_path):
        permanent = tmp_path / "Permanent"
        mock_settings = MagicMock(permanent_path=permanent)
        with patch.dict(
            "sys.modules",
            {"agent.config": MagicMock(settings=mock_settings)},
        ):
            result = safe_update._log_flaky_tests(
                ["tests/unit/test_foo.py::TestThing::test_flaky"]
            )
        assert result == permanent / "crash_log.md"
        body = result.read_text(encoding="utf-8")
        assert "# Flaky Test Recovery" in body
        assert "**Recovered Tests:** 1" in body
        assert "tests/unit/test_foo.py::TestThing::test_flaky" in body

    def test_preserves_existing_content_on_append(self, tmp_path):
        permanent = tmp_path / "Permanent"
        permanent.mkdir()
        crash_file = permanent / "crash_log.md"
        crash_file.write_text("# Bot Crash Report\n\nOld crash here\n", encoding="utf-8")

        mock_settings = MagicMock(permanent_path=permanent)
        with patch.dict(
            "sys.modules",
            {"agent.config": MagicMock(settings=mock_settings)},
        ):
            safe_update._log_flaky_tests(["tests/unit/test_a.py::test_b"])

        body = crash_file.read_text(encoding="utf-8")
        assert "Old crash here" in body
        assert "# Flaky Test Recovery" in body
        # Flaky header must NOT match crash_triage's split marker
        assert "# Bot Crash Report" in body  # original preserved
        assert body.count("# Bot Crash Report") == 1

    def test_returns_none_for_empty_nodeids(self):
        assert safe_update._log_flaky_tests([]) is None

    def test_swallows_settings_import_error(self):
        """Missing/broken agent.config should not crash the deploy."""
        with patch.dict(
            "sys.modules",
            {"agent.config": MagicMock(
                settings=MagicMock(
                    permanent_path=MagicMock(
                        __truediv__=MagicMock(
                            side_effect=OSError("broken"),
                        ),
                    ),
                ),
            )},
        ):
            # Should return None gracefully instead of propagating
            result = safe_update._log_flaky_tests(["some::test"])
        assert result is None

    def test_header_does_not_trigger_crash_triage(self, tmp_path):
        """crash_triage.py splits on '# Bot Crash Report'. Our header must differ."""
        permanent = tmp_path / "Permanent"
        mock_settings = MagicMock(permanent_path=permanent)
        with patch.dict(
            "sys.modules",
            {"agent.config": MagicMock(settings=mock_settings)},
        ):
            crash_file = safe_update._log_flaky_tests(
                ["tests/unit/test_x.py::test_y"]
            )
        body = crash_file.read_text(encoding="utf-8")
        assert "# Bot Crash Report" not in body
        assert "**Exception Type:**" not in body
