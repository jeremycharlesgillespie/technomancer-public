"""Tests for aim.manager — AI Manager daemon."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from aim.state import AIMState, WorkerState, save_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect state files for all tests."""
    monkeypatch.setattr("aim.state.STATE_FILE", tmp_path / ".aim_state.json")
    monkeypatch.setattr("aim.state.LOCK_FILE", tmp_path / ".aim_state.lock")
    monkeypatch.setattr("aim.state.PID_FILE", tmp_path / "aim.pid")

    from filelock import FileLock
    monkeypatch.setattr("aim.state._lock", FileLock(str(tmp_path / ".aim_state.lock"), timeout=10))


@pytest.fixture
def state():
    """Provide a fresh AIMState and persist it."""
    s = AIMState(
        manager_pid=os.getpid(),
        manager_started_at="2026-04-14T09:00:00",
        worker=WorkerState(
            pid=99999,
            status="idle",
            last_heartbeat=datetime.now().isoformat(timespec="seconds"),
        ),
    )
    save_state(s)
    return s


# ---------------------------------------------------------------------------
# Worker health checks
# ---------------------------------------------------------------------------

class TestCheckWorkerHealth:
    def test_no_worker_spawns_one(self, state):
        from aim.manager import check_worker_health

        state.worker.pid = None
        save_state(state)

        with patch("aim.manager.spawn_worker") as mock_spawn:
            result = check_worker_health(state)

        assert result is True  # Give it a cycle
        mock_spawn.assert_called_once()

    def test_dead_worker_returns_false(self, state):
        from aim.manager import check_worker_health

        with patch("aim.state.is_process_alive", return_value=False):
            result = check_worker_health(state)

        assert result is False
        assert state.worker.status == "dead"

    def test_healthy_worker_returns_true(self, state):
        from aim.manager import check_worker_health

        with patch("aim.state.is_process_alive", return_value=True):
            result = check_worker_health(state)

        assert result is True

    def test_stale_heartbeat_returns_stuck(self, state):
        from aim.manager import check_worker_health

        state.worker.last_heartbeat = "2020-01-01T00:00:00"  # Very stale
        save_state(state)

        with patch("aim.state.is_process_alive", return_value=True):
            result = check_worker_health(state)

        assert result is False
        assert state.worker.status == "stuck"


# ---------------------------------------------------------------------------
# Worker failure handling
# ---------------------------------------------------------------------------

class TestHandleWorkerFailure:
    def test_increments_failures_and_respawns(self, state):
        from aim.manager import handle_worker_failure

        state.worker.consecutive_failures = 0
        save_state(state)

        with patch("aim.manager.kill_worker"), \
             patch("aim.manager.spawn_worker") as mock_spawn:
            handle_worker_failure(state)

        assert state.worker.consecutive_failures == 1
        mock_spawn.assert_called_once()

    def test_clears_assignment(self, state):
        from aim.manager import handle_worker_failure

        state.worker.current_idea_id = "idea-042"
        state.worker.started_at = "2026-04-14T10:00:00"
        save_state(state)

        with patch("aim.manager.kill_worker"), \
             patch("aim.manager.spawn_worker"):
            handle_worker_failure(state)

        assert state.worker.current_idea_id is None
        assert state.worker.started_at is None

    def test_cooldown_on_max_failures(self, state):
        from aim.manager import handle_worker_failure

        state.worker.consecutive_failures = 2  # Will become 3 (= max)
        save_state(state)

        with patch("aim.manager.kill_worker"), \
             patch("aim.manager.spawn_worker"), \
             patch("aim.manager._notify_discord_throttled") as mock_notify, \
             patch("aim.manager.time.sleep") as mock_sleep, \
             patch("agent.config.settings") as mock_settings:
            mock_settings.aim_max_worker_failures = 3
            handle_worker_failure(state)

        mock_notify.assert_called_once()
        mock_sleep.assert_called()  # Cooldown sleep
        # Failures reset after cooldown
        assert state.worker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Board assessment
# ---------------------------------------------------------------------------

class TestAssessBoard:
    @patch("aim.jira_reader.get_board_summary")
    def test_jira_available(self, mock_jira, state):
        from aim.manager import assess_board

        mock_jira.return_value = {
            "todo": 20,
            "in_progress": 2,
            "done_total": 50,
            "done_last_24h": 5,
            "recent_completions": [],
            "all_counts": {"To Do": 20, "In Progress": 2, "Done": 50},
            "jira_available": True,
        }

        @dataclass
        class FakeIdea:
            id: str = "idea-001"
            title: str = "Test"
            state: str = "approved"
            category: str = "quality"
            created: str = "2026-04-14T10:00:00"

        with patch("idea_board.models.load_ideas", return_value=[FakeIdea()]):
            board = assess_board(state)

        assert board["todo"] == 20
        assert board["jira_available"] is True
        assert len(board["approved_ideas"]) == 1

    def test_jira_unavailable_uses_local(self, state):
        from aim.manager import assess_board

        @dataclass
        class FakeIdea:
            id: str
            title: str
            state: str
            category: str = "quality"
            created: str = "2026-04-14T10:00:00"

        ideas = [
            FakeIdea(id="idea-001", title="A", state="proposed"),
            FakeIdea(id="idea-002", title="B", state="approved"),
            FakeIdea(id="idea-003", title="C", state="approved"),
            FakeIdea(id="idea-004", title="D", state="executing"),
        ]

        with patch("aim.jira_reader.get_board_summary", side_effect=Exception("No Jira")), \
             patch("idea_board.models.load_ideas", return_value=ideas):
            board = assess_board(state)

        assert board["jira_available"] is False
        assert board["todo"] == 3  # 1 proposed + 2 approved
        assert board["in_progress"] == 1
        assert len(board["approved_ideas"]) == 2


# ---------------------------------------------------------------------------
# Decision execution
# ---------------------------------------------------------------------------

class TestExecuteDecision:
    def test_assign_valid_idea(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        @dataclass
        class FakeIdea:
            id: str = "idea-042"
            title: str = "Fix perf"
            state: str = "approved"

        decision = Decision(action="ASSIGN", target="idea-042", reason="Best pick")

        with patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker") as mock_assign, \
             patch("aim.manager._notify_discord"):
            execute_decision(state, decision, {"todo": 20})

        mock_assign.assert_called_once_with("idea-042")

    def test_assign_nonexistent_idea(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="ASSIGN", target="idea-999", reason="?")

        with patch("idea_board.models.get_idea", return_value=None), \
             patch("aim.state.assign_idea_to_worker") as mock_assign:
            execute_decision(state, decision, {})

        mock_assign.assert_not_called()

    def test_assign_non_approved_idea(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        @dataclass
        class FakeIdea:
            id: str = "idea-042"
            title: str = "Already done"
            state: str = "done"

        decision = Decision(action="ASSIGN", target="idea-042", reason="?")

        with patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker") as mock_assign:
            execute_decision(state, decision, {})

        mock_assign.assert_not_called()

    def test_create_work(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="CREATE_WORK", reason="Board low")

        with patch("aim.manager._create_new_work") as mock_create:
            execute_decision(state, decision, {"todo": 5})

        mock_create.assert_called_once()

    def test_escalate(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="ESCALATE", target="3h without progress", reason="stuck")

        with patch("aim.manager._notify_discord_throttled") as mock_notify:
            execute_decision(state, decision, {})

        mock_notify.assert_called_once()

    def test_restart_worker(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="RESTART_WORKER", reason="dead")

        with patch("aim.manager.handle_worker_failure") as mock_handle:
            execute_decision(state, decision, {})

        mock_handle.assert_called_once()

    def test_wait_is_noop(self, state):
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="WAIT", reason="nothing to do")
        # Should not raise
        execute_decision(state, decision, {})


# ---------------------------------------------------------------------------
# Work creation
# ---------------------------------------------------------------------------

class TestCreateNewWork:
    def test_creates_and_auto_approves_safe_categories(self, state):
        from aim.manager import _create_new_work

        ideas = [
            {"title": "Perf fix", "description": "D", "category": "performance", "idea_type": "story"},
            {"title": "New feature", "description": "D", "category": "feature", "idea_type": "story"},
        ]

        @dataclass
        class FakeIdea:
            id: str
            title: str

        created_ideas = [
            FakeIdea(id="idea-100", title="Perf fix"),
            FakeIdea(id="idea-101", title="New feature"),
        ]
        call_idx = [0]

        def mock_add_idea(title, description, source, category):
            idx = call_idx[0]
            call_idx[0] += 1
            return created_ideas[idx]

        with patch("aim.brain.generate_work_ideas", return_value=ideas), \
             patch("idea_board.models.load_ideas", return_value=[]), \
             patch("idea_board.models.add_idea", side_effect=mock_add_idea) as mock_add, \
             patch("idea_board.models.vote") as mock_vote, \
             patch("aim.manager._notify_discord"), \
             patch("agent.config.settings") as mock_settings:
            mock_settings.aim_board_high_threshold = 100
            mock_settings.aim_auto_approve_categories = "quality,performance,test"
            _create_new_work(state, {"todo": 5})

        assert mock_add.call_count == 2
        # Performance idea should be auto-approved, feature should not
        mock_vote.assert_called_once_with("idea-100", "owner", "approve")

    def test_skips_when_board_full(self, state):
        from aim.manager import _create_new_work

        with patch("aim.brain.generate_work_ideas") as mock_gen, \
             patch("agent.config.settings") as mock_settings:
            mock_settings.aim_board_high_threshold = 100
            _create_new_work(state, {"todo": 150})

        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

class TestStatusReport:
    def test_sends_discord_message(self, state):
        from aim.manager import send_status_report

        state.board_snapshot = {"todo": 15, "in_progress": 1, "done_last_24h": 3}
        state.completions_today = 5
        state.last_completion = "2026-04-14T10:00:00"

        with patch("aim.manager._notify_discord") as mock_notify:
            send_status_report(state)

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "15 To Do" in msg
        assert "5" in msg  # completions_today


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

class TestCLI:
    def test_show_status(self, state, capsys):
        from aim.manager import show_status

        save_state(state)

        with patch("aim.state.is_process_alive", return_value=False):
            show_status()

        captured = capsys.readouterr()
        assert "AIM Manager PID" in captured.out
        assert "Worker PID" in captured.out

    def test_stop_aim_not_running(self, capsys):
        from aim.manager import stop_aim

        with patch("aim.state.read_pid", return_value=None):
            stop_aim()

        captured = capsys.readouterr()
        assert "not running" in captured.out.lower()


# ---------------------------------------------------------------------------
# Discord notification throttling
# ---------------------------------------------------------------------------

class TestNotifyDiscordThrottled:
    def test_first_notification_sends(self, state):
        from aim.manager import _notify_discord_throttled

        with patch("aim.manager._notify_discord") as mock_send:
            _notify_discord_throttled(state, "test_cat", "Hello", cooldown_seconds=300)

        mock_send.assert_called_once()
        assert "test_cat" in state.last_discord_notify

    def test_second_notification_within_cooldown_blocked(self, state):
        from aim.manager import _notify_discord_throttled

        state.last_discord_notify["test_cat"] = datetime.now().isoformat(timespec="seconds")

        with patch("aim.manager._notify_discord") as mock_send:
            _notify_discord_throttled(state, "test_cat", "Hello again", cooldown_seconds=300)

        mock_send.assert_not_called()

    def test_notification_after_cooldown_sends(self, state):
        from aim.manager import _notify_discord_throttled

        state.last_discord_notify["test_cat"] = "2020-01-01T00:00:00"  # Long ago

        with patch("aim.manager._notify_discord") as mock_send:
            _notify_discord_throttled(state, "test_cat", "Hello", cooldown_seconds=300)

        mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# Spawn worker
# ---------------------------------------------------------------------------

class TestSpawnWorker:
    def test_spawns_subprocess(self, state):
        from aim.manager import spawn_worker

        state.worker.pid = None

        with patch("aim.state.is_process_alive", return_value=False), \
             patch("subprocess.Popen") as mock_popen, \
             patch("aim.manager._notify_discord"):
            mock_popen.return_value.pid = 12345
            spawn_worker(state)

        mock_popen.assert_called_once()
        assert state.worker.pid == 12345

    def test_skips_if_already_alive(self, state):
        from aim.manager import spawn_worker

        state.worker.pid = 99999

        with patch("aim.state.is_process_alive", return_value=True), \
             patch("subprocess.Popen") as mock_popen:
            spawn_worker(state)

        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Kill worker
# ---------------------------------------------------------------------------

class TestKillWorker:
    def test_kills_and_clears_pid(self, state):
        from aim.manager import kill_worker

        state.worker.pid = 12345

        with patch("subprocess.run") as mock_run:
            kill_worker(state)

        assert state.worker.pid is None
        assert state.worker.status == "dead"

    def test_no_pid_is_noop(self, state):
        from aim.manager import kill_worker

        state.worker.pid = None
        kill_worker(state)  # Should not raise
