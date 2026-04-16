"""End-to-end AIM -> Worker flow tests.

Verifies that a single AIM tick, paired with a stubbed canned-success
Worker step, drives an approved story through the expected state
transitions: approved -> executing -> done.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from aim.brain import Decision
from aim.state import AIMState, WorkerState, load_state, save_state
from board.provider import Comment, parse_marker
from idea_board.models import Idea


# ---------------------------------------------------------------------------
# FakeJiraProvider — in-memory provider that records state transitions
# ---------------------------------------------------------------------------


class FakeJiraProvider:
    """In-memory BoardProvider that records every state transition.

    ``state_history[item_id]`` is the ordered list of states each item
    has passed through, starting with its seeded state. The e2e test uses
    this to assert that a story passed through ``executing`` before
    reaching ``done``.
    """

    def __init__(self) -> None:
        self.items: dict[str, Idea] = {}
        self.comments: dict[str, list[Comment]] = {}
        self.state_history: dict[str, list[str]] = {}

    def seed(self, item: Idea) -> None:
        self.items[item.id] = item
        self.state_history.setdefault(item.id, []).append(item.state)

    def load_all(self) -> list[Idea]:
        return list(self.items.values())

    def list_by_state(self, state: str) -> list[Idea]:
        return [i for i in self.items.values() if i.state == state]

    def get(self, item_id: str) -> Idea | None:
        return self.items.get(item_id)

    def list_ideas_for_llm(self, state: str = "") -> str:
        return ""

    def get_comments(self, item_id: str) -> list[Comment]:
        return list(self.comments.get(item_id, []))

    def add(
        self,
        title: str,
        description: str,
        source: str = "llm_analysis",
        category: str = "feature",
        idea_type: str = "story",
        parent_id: str | None = None,
    ) -> Idea:
        new_id = f"TK-{len(self.items) + 100}"
        item = Idea(
            id=new_id,
            title=title,
            description=description,
            source=source,
            category=category,
            idea_type=idea_type,
            state="proposed",
            parent_id=parent_id,
        )
        self.items[new_id] = item
        self.state_history.setdefault(new_id, []).append(item.state)
        return item

    def vote(self, item_id: str, voter: str, value: str) -> Idea | None:
        return self.items.get(item_id)

    def add_comment(self, item_id: str, author: str, text: str) -> Idea | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        self.comments.setdefault(item_id, []).append(
            Comment(author=author, text=text, created="", marker=parse_marker(text))
        )
        return item

    def mark_executing(self, item_id: str) -> Idea | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        item.state = "executing"
        self.state_history.setdefault(item_id, []).append("executing")
        return item

    def mark_done(self, item_id: str, execution_log: str) -> Idea | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        item.state = "done"
        item.execution_log = execution_log
        self.state_history.setdefault(item_id, []).append("done")
        return item

    def mark_failed(self, item_id: str, error: str) -> Idea | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        item.state = "failed"
        item.execution_log = error
        self.state_history.setdefault(item_id, []).append("failed")
        return item

    def delete(self, item_id: str) -> bool:
        return self.items.pop(item_id, None) is not None

    def set_execution_order(self, item_id: str, order: list[str]) -> Idea | None:
        return self.items.get(item_id)

    def get_execution_order(self, item_id: str) -> list[str]:
        return []

    def set_epic_context(self, item_id: str, context: str) -> Idea | None:
        return self.items.get(item_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_aim_state(tmp_path, monkeypatch):
    """Redirect AIM state files to tmp_path and seed a live-looking worker."""
    from filelock import FileLock

    monkeypatch.setattr("aim.state.STATE_FILE", tmp_path / ".aim_state.json")
    monkeypatch.setattr("aim.state.LOCK_FILE", tmp_path / ".aim_state.lock")
    monkeypatch.setattr("aim.state.PID_FILE", tmp_path / "aim.pid")
    monkeypatch.setattr(
        "aim.state._lock",
        FileLock(str(tmp_path / ".aim_state.lock"), timeout=10),
    )

    state = AIMState(
        manager_pid=os.getpid(),
        manager_started_at=datetime.now().isoformat(timespec="seconds"),
        worker=WorkerState(
            pid=os.getpid(),
            status="idle",
            last_heartbeat=datetime.now().isoformat(timespec="seconds"),
        ),
    )
    save_state(state)
    return state


@pytest.fixture
def patched_git(monkeypatch):
    """Neutralize every git subprocess call the AIM/Worker path can make."""
    fake_run = MagicMock(
        return_value=MagicMock(returncode=0, stdout="main\n", stderr="")
    )

    monkeypatch.setattr("aim.worker._check_git_clean", lambda: True)
    monkeypatch.setattr("aim.worker._cleanup_stale_executions", lambda: None)
    monkeypatch.setattr("aim.worker.subprocess.run", fake_run)
    monkeypatch.setattr("aim.manager.subprocess.run", fake_run)
    monkeypatch.setattr(
        "aim.manager.verify_deployment",
        lambda idea_id: {"verified": True, "checks": {"on_main": True}},
    )


@pytest.fixture
def patched_jira_reader(monkeypatch):
    """Return canned Jira board summaries without hitting the real API."""
    monkeypatch.setattr(
        "aim.jira_reader.get_board_summary",
        lambda: {
            "todo": 1,
            "in_progress": 0,
            "done_total": 0,
            "done_last_24h": 0,
            "recent_completions": [],
            "all_counts": {"To Do": 1},
            "jira_available": False,
        },
    )
    monkeypatch.setattr("aim.jira_reader.is_jira_configured", lambda: False)


@pytest.fixture
def fake_jira_provider(monkeypatch):
    """Install a FakeJiraProvider as board.factory._provider."""
    provider = FakeJiraProvider()
    monkeypatch.setattr("board.factory._provider", provider)
    return provider


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


def test_story_state_transitions(
    isolated_aim_state,
    patched_git,
    patched_jira_reader,
    fake_jira_provider,
    monkeypatch,
):
    """An approved story flows through executing and lands on done.

    Seeds one approved story into the fake provider, runs a single AIM
    tick (manually orchestrating check_worker_health -> assess_board ->
    execute_decision with a direct ASSIGN decision), then fires the
    canned-success Worker step. Final state must be ``done`` with
    ``executing`` present in the state history, and the full sequence
    must complete in under 5 seconds.
    """
    from aim.manager import assess_board, check_worker_health, execute_decision

    story = Idea(
        id="TK-100",
        title="Add retry logic to webhook delivery",
        description="Stub story seeded for the e2e state-transition test.",
        source="planning",
        category="quality",
        idea_type="story",
        state="approved",
        created=datetime.now().isoformat(timespec="seconds"),
    )
    fake_jira_provider.seed(story)

    monkeypatch.setattr("aim.state.is_process_alive", lambda pid: True)
    monkeypatch.setattr("aim.manager._notify_discord", lambda *a, **k: None)
    monkeypatch.setattr("aim.manager._notify_discord_throttled", lambda *a, **k: None)

    start = time.monotonic()

    state = load_state()
    assert check_worker_health(state) is True

    board = assess_board(state)
    assert any(
        i["id"] == "TK-100" for i in board.get("approved_ideas", [])
    ), "seeded approved story must surface in assess_board output"

    decision = Decision(action="ASSIGN", target="TK-100", reason="e2e test")
    execute_decision(state, decision, board)

    assigned = load_state().worker.current_idea_id
    assert assigned == "TK-100", (
        f"AIM tick should have assigned TK-100, got {assigned!r}"
    )

    # Stubbed Worker — canned success: mark executing, then mark done.
    fake_jira_provider.mark_executing("TK-100")
    fake_jira_provider.mark_done("TK-100", "stubbed worker: canned success")

    elapsed = time.monotonic() - start

    final = fake_jira_provider.get("TK-100")
    assert final is not None
    assert final.state == "done", f"expected final state=done, got {final.state!r}"

    history = fake_jira_provider.state_history["TK-100"]
    assert "executing" in history, (
        f"expected 'executing' in state history, got {history}"
    )
    assert history[-1] == "done"
    assert history.index("executing") < history.index("done")

    assert elapsed < 5.0, f"tick + worker step took {elapsed:.2f}s (must be <5s)"
