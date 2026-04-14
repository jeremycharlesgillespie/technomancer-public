"""Tests for safe_update.py post-deploy health check and auto-rollback."""

import os
from unittest.mock import MagicMock, patch, call
import subprocess

import pytest

# Import the module under test
import safe_update


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
