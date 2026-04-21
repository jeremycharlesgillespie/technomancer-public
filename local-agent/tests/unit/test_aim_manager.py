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


@pytest.fixture(autouse=True)
def _isolated_event_log(tmp_path, monkeypatch):
    """Redirect the event log so tests don't pollute the real aim/events.jsonl."""
    from aim import event_log

    monkeypatch.setattr(event_log, "LOG_DIR", tmp_path)
    monkeypatch.setattr(event_log, "LOG_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(event_log, "BACKUP_FILE", tmp_path / "events.1.jsonl")


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

    def test_rate_limited_worker_is_healthy(self, state):
        """TK-410: A Worker with status='rate_limited' must be considered
        healthy as long as heartbeats stay fresh — restarting it would
        abort the back-off retry loop."""
        from aim.manager import check_worker_health

        state.worker.status = "rate_limited"
        state.worker.last_heartbeat = datetime.now().isoformat(timespec="seconds")
        save_state(state)

        with patch("aim.state.is_process_alive", return_value=True):
            result = check_worker_health(state)

        assert result is True
        # Status is informational — health check must not overwrite it.
        assert state.worker.status == "rate_limited"


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

    def test_vetoed_items_excluded_from_approved_ideas(self, state):
        from aim.manager import assess_board

        @dataclass
        class FakeIdea:
            id: str
            title: str
            state: str
            category: str = "quality"
            created: str = "2026-04-14T10:00:00"

        ideas = [
            FakeIdea(id="TK-1", title="Good", state="approved"),
            FakeIdea(id="TK-2", title="Vetoed", state="vetoed"),
            FakeIdea(id="TK-3", title="Failed", state="failed"),
        ]

        with patch("aim.jira_reader.get_board_summary", side_effect=Exception("No Jira")), \
             patch("idea_board.models.load_ideas", return_value=ideas):
            board = assess_board(state)

        assert len(board["approved_ideas"]) == 1
        assert board["approved_ideas"][0]["id"] == "TK-1"


# ---------------------------------------------------------------------------
# Decision execution
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_dedup_llm")
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
             patch("aim.manager._notify_discord"), \
             patch("aim.manager._is_peak_hour_pt", return_value=False):
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

    def test_is_peak_hour_pt_inside_window(self):
        """Hour 9 PT is inside the default 5-11 window."""
        from zoneinfo import ZoneInfo
        from datetime import datetime
        from aim.manager import _is_peak_hour_pt

        # 9am PT on an arbitrary date.
        now_pt = datetime(2026, 4, 19, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert _is_peak_hour_pt(now_pt) is True

    def test_is_peak_hour_pt_outside_window(self):
        from zoneinfo import ZoneInfo
        from datetime import datetime
        from aim.manager import _is_peak_hour_pt

        now_pt = datetime(2026, 4, 19, 15, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert _is_peak_hour_pt(now_pt) is False

    def test_is_peak_hour_pt_respects_disable_flag(self, monkeypatch):
        """Setting aim_peak_hour_pause_enabled=False disables the guard."""
        from zoneinfo import ZoneInfo
        from datetime import datetime
        from agent.config import settings
        from aim.manager import _is_peak_hour_pt

        monkeypatch.setattr(settings, "aim_peak_hour_pause_enabled", False)
        now_pt = datetime(2026, 4, 19, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert _is_peak_hour_pt(now_pt) is False

    def test_assign_blocked_during_peak_hour(self, state):
        """Peak-hour window (5-11am PT) holds the assignment."""
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="ASSIGN", target="TK-100", reason="top")

        with patch("aim.manager._is_peak_hour_pt", return_value=True), \
             patch("aim.state.assign_idea_to_worker") as mock_assign:
            execute_decision(state, decision, {"todo": 20})

        mock_assign.assert_not_called()

    def test_assign_blocked_when_cooldown_active(self, state):
        """ASSIGN is held if last_assigned_at is within the cooldown window."""
        from aim.brain import Decision
        from aim.manager import execute_decision
        from datetime import datetime, timedelta

        # 30 min ago — inside a 1-hour cooldown.
        state.last_assigned_at = (
            datetime.now() - timedelta(minutes=30)
        ).isoformat(timespec="seconds")

        decision = Decision(action="ASSIGN", target="TK-101", reason="top")

        with patch("aim.manager._is_peak_hour_pt", return_value=False), \
             patch("aim.state.assign_idea_to_worker") as mock_assign:
            execute_decision(state, decision, {"todo": 20})

        mock_assign.assert_not_called()

    def test_assign_allowed_after_cooldown_elapses(self, state):
        """ASSIGN proceeds when last_assigned_at is older than the cooldown."""
        from aim.brain import Decision
        from aim.manager import execute_decision
        from datetime import datetime, timedelta

        @dataclass
        class FakeIdea:
            id: str = "TK-102"
            title: str = "Ready"
            state: str = "approved"

        state.last_assigned_at = (
            datetime.now() - timedelta(hours=2)
        ).isoformat(timespec="seconds")

        decision = Decision(action="ASSIGN", target="TK-102", reason="top")

        with patch("aim.manager._is_peak_hour_pt", return_value=False), \
             patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker") as mock_assign, \
             patch("aim.manager._notify_discord"), \
             patch("aim.state.save_state"):
            execute_decision(state, decision, {"todo": 20})

        mock_assign.assert_called_once_with("TK-102")

    def test_assign_stamps_last_assigned_at(self, state):
        """After a successful ASSIGN, state.last_assigned_at is updated."""
        from aim.brain import Decision
        from aim.manager import execute_decision

        @dataclass
        class FakeIdea:
            id: str = "TK-103"
            title: str = "x"
            state: str = "approved"

        state.last_assigned_at = ""

        decision = Decision(action="ASSIGN", target="TK-103", reason="top")
        with patch("aim.manager._is_peak_hour_pt", return_value=False), \
             patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker"), \
             patch("aim.manager._notify_discord"), \
             patch("aim.state.save_state"):
            execute_decision(state, decision, {"todo": 20})

        assert state.last_assigned_at != ""

    def test_assign_blocked_when_jira_in_progress(self, state):
        """If recovery can't clear In Progress items, assignment is still blocked."""
        from aim.brain import Decision
        from aim.manager import execute_decision

        @dataclass
        class FakeIdea:
            id: str = "TK-100"
            title: str = "new work"
            state: str = "approved"

        decision = Decision(action="ASSIGN", target="TK-100", reason="?")
        # Still shows in_progress=1 after recovery attempt
        board = {"todo": 20, "in_progress": 1}

        with patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker") as mock_assign, \
             patch("aim.manager._recover_orphan_in_progress"), \
             patch("aim.manager.assess_board", return_value={"todo": 20, "in_progress": 1}):
            execute_decision(state, decision, board)

        mock_assign.assert_not_called()

    def test_assign_proceeds_after_orphan_recovery(self, state):
        """Auto-recover clears the In Progress item; assignment then proceeds."""
        from aim.brain import Decision
        from aim.manager import execute_decision

        @dataclass
        class FakeIdea:
            id: str = "TK-100"
            title: str = "new work"
            state: str = "approved"

        decision = Decision(action="ASSIGN", target="TK-100", reason="?")
        board = {"todo": 20, "in_progress": 1}

        call_count = [0]
        def mock_assess(s):
            call_count[0] += 1
            return {"todo": 20, "in_progress": 0, "approved_ideas": []}

        with patch("idea_board.models.get_idea", return_value=FakeIdea()), \
             patch("aim.state.assign_idea_to_worker") as mock_assign, \
             patch("aim.manager._recover_orphan_in_progress"), \
             patch("aim.manager.assess_board", side_effect=mock_assess), \
             patch("board.get_provider") as mock_provider:
            mock_provider.return_value.get.return_value = FakeIdea()
            execute_decision(state, decision, board)

        mock_assign.assert_called_once_with("TK-100")

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

        def mock_add_idea(title, description, source, category, **kwargs):
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
# Queue review — vetoed exclusion
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_dedup_llm")
class TestReviewQueueVetoedExclusion:
    def test_vetoed_items_excluded_from_active_and_failed(self, state):
        """review_queue must never process vetoed items as active or failed."""
        from aim.manager import review_queue

        @dataclass
        class FakeIdea:
            id: str
            title: str
            state: str
            description: str = ""
            category: str = "quality"
            created: str = "2026-04-14T10:00:00"
            execution_log: str = ""

        ideas = [
            FakeIdea(id="TK-1", title="Good idea", state="approved"),
            FakeIdea(id="TK-2", title="Vetoed idea", state="vetoed"),
            FakeIdea(id="TK-3", title="Failed idea", state="failed",
                     execution_log="test failure"),
        ]

        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas

        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"):
            review_queue(state)

        # The vetoed item should never be voted on or touched
        for c in mock_provider.vote.call_args_list:
            assert c[0][0] != "TK-2", "vetoed item must not be re-voted"

    def test_review_queue_veto_does_not_move_to_todo(self, state):
        """When review_queue auto-vetoes, it calls provider.vote(id, 'owner', 'veto')
        which should transition to Failed (not To Do)."""
        from aim.manager import review_queue

        @dataclass
        class FakeIdea:
            id: str
            title: str
            state: str
            description: str = ""
            category: str = "quality"
            created: str = "2026-04-14T10:00:00"
            execution_log: str = ""
            source: str = "llm_analysis"
            parent_id: str | None = None
            labels: list = None

        ideas = [
            # This idea matches 2 failures — should be auto-vetoed
            FakeIdea(id="TK-10", title="Add caching layer", state="proposed"),
            FakeIdea(id="TK-11", title="Add caching layer v1", state="failed",
                     execution_log="test failure"),
            FakeIdea(id="TK-12", title="Add caching layer v2", state="failed",
                     execution_log="test failure"),
        ]

        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas

        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"):
            review_queue(state)

        # The matching proposed idea should be vetoed (not moved to To Do)
        mock_provider.vote.assert_called_once_with("TK-10", "owner", "veto")


@pytest.mark.usefixtures("mock_dedup_llm")
class TestReviewQueueExemptions:
    """Human-intent exemptions prevent auto-veto of legitimately planned work."""

    @dataclass
    class FakeIdea:
        id: str
        title: str
        state: str
        description: str = ""
        category: str = "quality"
        created: str = "2026-04-14T10:00:00"
        execution_log: str = ""
        source: str = "llm_analysis"
        parent_id: str | None = None
        labels: list = None

        def __post_init__(self):
            if self.labels is None:
                self.labels = []

    def _run_review(self, ideas, state):
        from aim.manager import review_queue
        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas
        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"):
            review_queue(state)
        return mock_provider

    def test_src_planning_exempt_from_dedup_veto(self, state):
        ideas = [
            self.FakeIdea(id="TK-101", title="Filter commits publish script",
                          description="filter generic commits from publish",
                          state="approved", source="planning"),
            self.FakeIdea(id="TK-100", title="Filter auto generic commits",
                          description="filter generic commits in publish step",
                          state="done"),
        ]
        provider = self._run_review(ideas, state)
        for call in provider.vote.call_args_list:
            assert call[0][0] != "TK-101", "planning-source idea must not be vetoed"

    def test_protect_label_exempt_from_dedup_veto(self, state):
        ideas = [
            self.FakeIdea(id="TK-201", title="Add caching layer",
                          description="caching layer stuff",
                          state="approved", source="llm_analysis",
                          labels=["protect:no-veto"]),
            self.FakeIdea(id="TK-200", title="Add caching layer v1",
                          description="caching layer stuff",
                          state="done"),
        ]
        provider = self._run_review(ideas, state)
        for call in provider.vote.call_args_list:
            assert call[0][0] != "TK-201", "protect-labeled idea must not be vetoed"

    def test_idea_with_active_parent_epic_exempt(self, state):
        ideas = [
            self.FakeIdea(id="TK-300", title="Dashboard epic", state="approved",
                          idea_type="epic") if False else self.FakeIdea(
                id="TK-300", title="Dashboard epic", state="approved",
            ),
            self.FakeIdea(id="TK-301", title="Add caching layer",
                          description="caching layer stuff",
                          state="approved", parent_id="TK-300"),
            self.FakeIdea(id="TK-302", title="Add caching layer v1",
                          description="caching layer stuff",
                          state="done"),
        ]
        provider = self._run_review(ideas, state)
        for call in provider.vote.call_args_list:
            assert call[0][0] != "TK-301", (
                "story under a non-vetoed parent epic must not be vetoed"
            )

    def test_orphan_story_not_vetoed_for_dup_but_flagged(self, state, mock_dedup_llm):
        """TK-742: Step 2 no longer vetoes dups-of-done — only flags a comment.

        Previously an orphan story matching a done/failed idea was auto-vetoed;
        that killed legitimate follow-up stories sharing a topic with shipped
        work. The new behavior leaves the state untouched and drops an owner-
        review comment instead.

        Uses ``mock_dedup_llm`` so the test drives the Step 2 path via the
        dedup seam directly. That keeps the assertion ("advisory-only when
        dedup says match") independent of whichever judge is wired in —
        the current word-overlap check or the LLM near-exact judge that
        replaces it.
        """
        mock_dedup_llm.return_value = True
        ideas = [
            self.FakeIdea(id="TK-400", title="Add caching layer",
                          description="caching layer stuff",
                          state="approved"),
            self.FakeIdea(id="TK-401", title="Add caching layer v1",
                          description="caching layer stuff",
                          state="done"),
        ]
        provider = self._run_review(ideas, state)

        # No veto for the topic-overlap case — that's the TK-742 behavior change.
        for c in provider.vote.call_args_list:
            assert c[0][0] != "TK-400", "dup-of-done must not auto-veto"

        # But it should leave an advisory comment flagging the possible dup
        # so the owner can review and veto manually if it really is one.
        # Lock in the full marker text + state qualifier + manual-review hint
        # so a regression that drops any of these signals (and would leave
        # the operator without enough context to act) fails this test.
        flagging_calls = [
            c for c in provider.add_comment.call_args_list
            if c[0][0] == "TK-400" and "High overlap with TK-401" in c[0][2]
        ]
        assert flagging_calls, "dup-of-done should leave an advisory comment"
        comment_text = flagging_calls[0][0][2]
        assert "High overlap with" in comment_text, (
            "comment must carry the High-overlap marker so it groups with "
            "other queue-hygiene activity"
        )
        assert "(Done)" in comment_text, (
            "comment must surface the matched idea's state so the operator "
            "knows whether the dup ships or was abandoned"
        )
        assert "Consider revising scope or closing as duplicate" in comment_text, (
            "comment must direct the operator toward manual action — Step 2 "
            "is advisory, not a queued auto-action"
        )

        # And no state-mutating method should fire against the orphan beyond
        # the (legitimate) add_comment / get_comments calls. Auto-vetoes used
        # to ride on vote(), but a future regression could route through
        # update_state or set_state instead — guard both paths.
        for forbidden in ("update_state", "set_state", "transition"):
            calls = getattr(provider, forbidden).call_args_list
            assert not any(c[0] and c[0][0] == "TK-400" for c in calls), (
                f"orphan dup-of-done must not be mutated via {forbidden}"
            )


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


# ---------------------------------------------------------------------------
# Orphaned process cleanup
# ---------------------------------------------------------------------------


class TestCleanupOrphanedProcesses:
    def _make_proc(self, pid, name, cmdline=None, environ=None):
        """Create a mock psutil process info entry."""
        m = MagicMock()
        m.info = {
            "pid": pid,
            "name": name,
            "cmdline": cmdline or [],
        }
        m.environ.return_value = environ or {}
        return m

    def test_kills_orphan_claude_exe(self, state):
        from aim.manager import _cleanup_orphaned_processes

        # Must be a headless -p process to be considered an orphan
        proc = self._make_proc(55555, "claude.exe", ["claude.exe", "-p", "--flag"])

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run") as mock_run,
            patch("aim.manager._notify_discord_throttled"),
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            killed = _cleanup_orphaned_processes(state)

        assert killed == 1
        args = mock_run.call_args[0][0]
        assert "/T" in args
        assert "55555" in args

    def test_protects_claudecode_session(self, state):
        from aim.manager import _cleanup_orphaned_processes

        proc = self._make_proc(55555, "claude.exe", ["claude.exe"])
        proc.environ.return_value = {"CLAUDECODE": "1"}

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run") as mock_run,
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            killed = _cleanup_orphaned_processes(state)

        assert killed == 0
        mock_run.assert_not_called()

    def test_protects_remote_control(self, state):
        """remote-control process must never be killed."""
        from aim.manager import _cleanup_orphaned_processes

        proc = self._make_proc(55555, "claude.exe", ["claude.exe", "remote-control", "--name", "test"])

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run") as mock_run,
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            killed = _cleanup_orphaned_processes(state)

        assert killed == 0
        mock_run.assert_not_called()

    def test_protects_worker_pid(self, state):
        from aim.manager import _cleanup_orphaned_processes

        state.worker.pid = 55555
        proc = self._make_proc(55555, "python.exe", ["python", "-m", "aim.worker"])

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run") as mock_run,
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            killed = _cleanup_orphaned_processes(state)

        assert killed == 0
        mock_run.assert_not_called()

    def test_ignores_unrelated_python(self, state):
        from aim.manager import _cleanup_orphaned_processes

        proc = self._make_proc(55555, "python.exe", ["python", "my_script.py"])

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run") as mock_run,
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            killed = _cleanup_orphaned_processes(state)

        assert killed == 0
        mock_run.assert_not_called()

    def test_calls_orphan_cleanup_on_worker_failure(self, state):
        from aim.manager import handle_worker_failure

        with (
            patch("aim.manager.kill_worker"),
            patch("aim.manager.spawn_worker"),
            patch("aim.manager._cleanup_orphaned_processes") as mock_cleanup,
        ):
            handle_worker_failure(state)

        mock_cleanup.assert_called_once_with(state)


# ---------------------------------------------------------------------------
# Event log emits (TK-372)
# ---------------------------------------------------------------------------


class TestEventLogEmits:
    def test_worker_spawned_event(self, state):
        from aim import event_log
        from aim.manager import spawn_worker

        state.worker.pid = None

        with patch("aim.state.is_process_alive", return_value=False), \
             patch("subprocess.Popen") as mock_popen, \
             patch("aim.manager._notify_discord"):
            mock_popen.return_value.pid = 42424
            spawn_worker(state)

        events = event_log.read_events()
        spawn_events = [e for e in events if e["type"] == "worker_spawned"]
        assert len(spawn_events) == 1
        assert spawn_events[0]["data"] == {"pid": 42424}

    def test_worker_died_event_pid_dead(self, state):
        from aim import event_log
        from aim.manager import check_worker_health

        with patch("aim.state.is_process_alive", return_value=False):
            check_worker_health(state)

        events = event_log.read_events()
        died_events = [e for e in events if e["type"] == "worker_died"]
        assert len(died_events) == 1
        assert died_events[0]["data"]["pid"] == state.worker.pid
        assert died_events[0]["data"]["cause"] == "pid_dead"

    def test_worker_died_event_stale_heartbeat(self, state):
        from aim import event_log
        from aim.manager import check_worker_health

        state.worker.last_heartbeat = "2020-01-01T00:00:00"
        save_state(state)

        with patch("aim.state.is_process_alive", return_value=True):
            check_worker_health(state)

        events = event_log.read_events()
        died_events = [e for e in events if e["type"] == "worker_died"]
        assert len(died_events) == 1
        assert died_events[0]["data"]["cause"] == "heartbeat_stale"
        assert "elapsed_seconds" in died_events[0]["data"]

    def test_escalation_event(self, state):
        from aim import event_log
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(
            action="ESCALATE",
            target="3h without progress",
            reason="worker stuck",
        )

        with patch("aim.manager._notify_discord_throttled"):
            execute_decision(state, decision, {})

        events = event_log.read_events()
        esc_events = [e for e in events if e["type"] == "escalation"]
        assert len(esc_events) == 1
        assert esc_events[0]["data"]["reason"] == "worker stuck"
        assert esc_events[0]["data"]["target"] == "3h without progress"

    def test_orphan_cleanup_event(self, state):
        from aim import event_log
        from aim.manager import _cleanup_orphaned_processes

        proc = MagicMock()
        proc.info = {
            "pid": 55555,
            "name": "claude.exe",
            "cmdline": ["claude.exe", "-p", "--flag"],
        }
        proc.environ.return_value = {}

        with (
            patch("psutil.process_iter", return_value=[proc]),
            patch("psutil.Process") as mock_ps,
            patch("subprocess.run"),
            patch("aim.manager._notify_discord_throttled"),
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            _cleanup_orphaned_processes(state)

        events = event_log.read_events()
        cleanup_events = [e for e in events if e["type"] == "orphan_cleanup"]
        assert len(cleanup_events) == 1
        assert cleanup_events[0]["data"]["killed"] == 1
        assert 55555 in cleanup_events[0]["data"]["pids"]

    def test_orphan_cleanup_silent_when_nothing_killed(self, state):
        from aim import event_log
        from aim.manager import _cleanup_orphaned_processes

        with (
            patch("psutil.process_iter", return_value=[]),
            patch("psutil.Process") as mock_ps,
        ):
            mock_self = MagicMock()
            mock_self.pid = os.getpid()
            mock_self.parent.return_value = None
            mock_ps.return_value = mock_self

            _cleanup_orphaned_processes(state)

        events = event_log.read_events()
        assert not [e for e in events if e["type"] == "orphan_cleanup"]

    def test_event_log_failure_does_not_break_flow(self, state):
        """event_log write failures must not prevent the action from completing."""
        from aim.brain import Decision
        from aim.manager import execute_decision

        decision = Decision(action="ESCALATE", target="x", reason="y")

        with patch("aim.event_log.append_event", side_effect=OSError("disk full")), \
             patch("aim.manager._notify_discord_throttled") as mock_notify:
            # Must not raise
            execute_decision(state, decision, {})

        mock_notify.assert_called_once()


class TestRecoverOrphanInProgress:
    """Startup orphan recovery: stale 'In Progress' items on Jira get moved back."""

    def test_moves_orphan_in_progress_to_todo(self, state):
        from aim.manager import _recover_orphan_in_progress

        state.worker.current_idea_id = None  # No active assignment
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            'issues': [{'key': 'TK-393', 'fields': {'summary': 'orphan'}}]
        }

        with patch('idea_board.jira_sync.is_jira_configured', return_value=True),              patch('idea_board.jira_sync._api', return_value=resp),              patch('idea_board.jira_sync.transition_jira_issue', return_value=True) as mock_transition,              patch('aim.manager._notify_discord'):
            _recover_orphan_in_progress(state)

        mock_transition.assert_called_once_with('TK-393', 'To Do')

    def test_skips_workers_current_assignment(self, state):
        from aim.manager import _recover_orphan_in_progress

        state.worker.current_idea_id = 'TK-396'  # Worker was on this when we shut down
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            'issues': [{'key': 'TK-396', 'fields': {'summary': 'resumable'}}]
        }

        with patch('idea_board.jira_sync.is_jira_configured', return_value=True),              patch('idea_board.jira_sync._api', return_value=resp),              patch('idea_board.jira_sync.transition_jira_issue') as mock_transition:
            _recover_orphan_in_progress(state)

        mock_transition.assert_not_called()

    def test_skips_vetoed_issue_during_recovery(self, state):
        """Orphan recovery must never move a vetoed issue back to To Do."""
        from aim.manager import _recover_orphan_in_progress

        state.worker.current_idea_id = None
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            'issues': [
                {
                    'key': 'TK-500',
                    'fields': {
                        'summary': 'vetoed idea',
                        'labels': ['vetoed', 'cat:quality'],
                    },
                },
                {
                    'key': 'TK-501',
                    'fields': {
                        'summary': 'real orphan',
                        'labels': ['cat:feature'],
                    },
                },
            ]
        }

        with patch('idea_board.jira_sync.is_jira_configured', return_value=True), \
             patch('idea_board.jira_sync._api', return_value=resp), \
             patch('idea_board.jira_sync.transition_jira_issue', return_value=True) as mock_transition, \
             patch('aim.manager._notify_discord'):
            _recover_orphan_in_progress(state)

        # Only the non-vetoed issue should be transitioned
        mock_transition.assert_called_once_with('TK-501', 'To Do')

    def test_no_op_when_jira_not_configured(self, state):
        from aim.manager import _recover_orphan_in_progress

        with patch('idea_board.jira_sync.is_jira_configured', return_value=False),              patch('idea_board.jira_sync._api') as mock_api:
            _recover_orphan_in_progress(state)

        mock_api.assert_not_called()
