"""Tests for TK-742: review_queue dup handling is advisory, not a veto.

The TK-742 change removed the auto-veto from Step 2 of ``review_queue``
(the "duplicate cleanup against done/failed" block). The cost of a
false-veto — silently killing legitimate follow-up work — was higher
than the cost of a false-keep (one redundant story shipping), so topic
overlap with shipped/abandoned ideas now yields an advisory comment
instead of a state transition.

These tests lock in the new behavior:

* ``_is_duplicate`` itself is unchanged; it still flags true
  near-duplicates (same title + same body) and still passes distinct
  follow-up stories even when they share a topic.
* ``review_queue`` leaves dup-of-done active-state stories alone and
  drops an owner-review comment instead.
* Step 1 (failure pattern detection — 2+ failed with matching title)
  still auto-vetoes, so the queue doesn't drown in stories that have
  already failed repeatedly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aim.state import AIMState, WorkerState


# ---------------------------------------------------------------------------
# Fixtures — minimal, self-contained AIMState so the module runs independently
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect state files so review_queue doesn't touch real AIM state."""
    monkeypatch.setattr("aim.state.STATE_FILE", tmp_path / ".aim_state.json")
    monkeypatch.setattr("aim.state.LOCK_FILE", tmp_path / ".aim_state.lock")
    monkeypatch.setattr("aim.state.PID_FILE", tmp_path / "aim.pid")
    from filelock import FileLock
    monkeypatch.setattr(
        "aim.state._lock",
        FileLock(str(tmp_path / ".aim_state.lock"), timeout=10),
    )


@pytest.fixture(autouse=True)
def _isolated_event_log(tmp_path, monkeypatch):
    from aim import event_log
    monkeypatch.setattr(event_log, "LOG_DIR", tmp_path)
    monkeypatch.setattr(event_log, "LOG_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(event_log, "BACKUP_FILE", tmp_path / "events.1.jsonl")


@pytest.fixture
def state():
    return AIMState(
        manager_pid=os.getpid(),
        manager_started_at="2026-04-18T09:00:00",
        worker=WorkerState(
            pid=99999,
            status="idle",
            last_heartbeat=datetime.now().isoformat(timespec="seconds"),
        ),
    )


@dataclass
class FakeIdea:
    """Duck-typed idea matching what review_queue reads via getattr."""

    id: str
    title: str
    state: str
    description: str = ""
    category: str = "quality"
    created: str = "2026-04-18T10:00:00"
    execution_log: str = ""
    source: str = "llm_analysis"
    parent_id: str | None = None
    labels: list = field(default_factory=list)


def _run_review(state, ideas):
    """Invoke review_queue with a MagicMock provider and return it.

    ``get_comments`` is patched to return an empty list so the
    "already-flagged" dedup check inside Step 2 doesn't short-circuit.
    """
    from aim.manager import review_queue

    mock_provider = MagicMock()
    mock_provider.load_all.return_value = ideas
    mock_provider.get_comments.return_value = []

    with patch("board.get_provider", return_value=mock_provider), \
         patch("aim.manager._notify_discord"):
        review_queue(state)
    return mock_provider


# ---------------------------------------------------------------------------
# _is_duplicate — pure function behavior
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    """_is_duplicate still catches true restated dups AND passes real follow-ups."""

    def test_tk571_tk321_distinct_descriptions_not_duplicate(self):
        """TK-571 coverage-lift story is NOT a dup of TK-321 abstract dittoed story.

        This is the exact false-positive that motivated TK-742. Titles share
        {unit, tests, capability_request} (roughly 50% title overlap — not
        above the > 0.5 threshold), and the descriptions diverge on scope
        (concrete coverage target vs. abstract "add core logic tests"), so
        combined overlap falls below the 0.4 threshold too. Result: False.
        """
        from idea_board.models import Idea, _is_duplicate

        existing = Idea(
            id="TK-321",
            title="[idea-197] Add Unit Tests for capability_request.py Core Logic",
            description=(
                "WHAT: Add unit tests covering the core capability evaluation "
                "logic. WHY: No coverage today. HOW: Write tests against the "
                "Claude API evaluation path."
            ),
        )

        assert not _is_duplicate(
            new_title="Unit tests for capability_request.py (50% -> 75%)",
            new_desc=(
                "WHAT: Raise coverage from 50 percent to 75 percent. "
                "WHY: Gaps exist in the error retry and rate-limit branches. "
                "HOW: Parametrize failure modes in the circuit-breaker helper."
            ),
            existing=existing,
        )

    def test_near_identical_title_and_body_is_duplicate(self):
        """A restated dup (same title, same body) must still be caught.

        Creation-time dedup inside ``add_idea`` is the last line of defense
        against the classic "idea generator restates the same idea" failure
        mode. TK-742 must not weaken that.
        """
        from idea_board.models import Idea, _is_duplicate

        existing = Idea(
            id="TK-1",
            title="Cache Ollama responses to improve performance",
            description=(
                "WHY: Ollama inference repeats work for identical prompts. "
                "HOW: Cache responses keyed by prompt hash to skip recompute."
            ),
        )

        assert _is_duplicate(
            new_title="Cache Ollama responses for better performance",
            new_desc=(
                "WHY: Ollama inference repeats work for identical prompts. "
                "HOW: Cache responses keyed by prompt hash to skip recompute."
            ),
            existing=existing,
        )


# ---------------------------------------------------------------------------
# Step 2 is advisory — no veto, just a comment
# ---------------------------------------------------------------------------


class TestReviewQueueStep2Advisory:
    """Dup-of-done stories get a flag comment, never an auto-veto (TK-742)."""

    def test_moderate_title_overlap_with_done_does_not_veto(self, state):
        """30–50% title overlap with a done story is NOT a dup signal."""
        ideas = [
            FakeIdea(
                id="TK-500",
                title="Retry webhook delivery on transient failures",
                description=(
                    "Add exponential backoff to webhook posting so network "
                    "blips don't drop events. Configurable max attempts."
                ),
                state="approved",
            ),
            FakeIdea(
                id="TK-400",
                title="Add webhook delivery endpoint",
                description="Create the initial webhook delivery mechanism.",
                state="done",
            ),
        ]

        provider = _run_review(state, ideas)

        # No veto — that's the TK-742 behavior change.
        for c in provider.vote.call_args_list:
            assert c[0][0] != "TK-500", (
                "moderate overlap with done story must not trigger auto-veto"
            )

    def test_high_overlap_with_done_is_flagged_with_comment(self, state):
        """Real dup-of-done shape → add_comment fires with the marker."""
        ideas = [
            FakeIdea(
                id="TK-600",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-501",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
        ]

        provider = _run_review(state, ideas)

        # Still no veto.
        for c in provider.vote.call_args_list:
            assert c[0][0] != "TK-600"

        # But a comment carrying the "Possible dup of ..." marker is added.
        flagged = any(
            c[0][0] == "TK-600" and "Possible dup of TK-501" in c[0][2]
            for c in provider.add_comment.call_args_list
        )
        assert flagged, "dup-of-done must leave an advisory comment"

    def test_already_flagged_is_not_recommented(self, state):
        """Don't spam the same advisory on every review tick.

        If a prior review already dropped the ``[Queue Review] Possible dup
        of TK-X`` marker on this idea, skip it — otherwise each 30s tick
        would add another identical comment and bury the real discussion.
        """
        from board.provider import Comment

        ideas = [
            FakeIdea(
                id="TK-700",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-701",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
        ]

        prior_comment = Comment(
            author="llm",
            text="[Queue Review] Possible dup of TK-701 (done). Review and "
                 "mark vetoed manually if this is a true dup.",
            created="2026-04-18T09:30:00",
            marker=None,
        )

        from aim.manager import review_queue

        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas
        mock_provider.get_comments.return_value = [prior_comment]

        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"):
            review_queue(state)

        for c in mock_provider.add_comment.call_args_list:
            assert c[0][0] != "TK-700", (
                "idea already flagged should not be flagged again"
            )


# ---------------------------------------------------------------------------
# Step 1 still vetoes the repeated-failure pattern
# ---------------------------------------------------------------------------


class TestReviewQueueStep1StillVetoes:
    """Matches 2+ failed stories → auto-veto. Not touched by TK-742."""

    def test_two_failed_matches_triggers_veto(self, state):
        ideas = [
            FakeIdea(id="TK-800", title="Rewrite dedup in Rust", state="approved"),
            FakeIdea(
                id="TK-801", title="Rewrite dedup in Rust v1",
                state="failed", execution_log="test failure",
            ),
            FakeIdea(
                id="TK-802", title="Rewrite dedup Rust port",
                state="failed", execution_log="test failure",
            ),
        ]

        provider = _run_review(state, ideas)

        # Step 1 still vetoes on repeated failure pattern.
        provider.vote.assert_called_once_with("TK-800", "owner", "veto")

    def test_single_failure_match_does_not_veto(self, state):
        """One matching failed story is not enough — needs 2+."""
        ideas = [
            FakeIdea(id="TK-900", title="Rewrite dedup in Rust", state="approved"),
            FakeIdea(
                id="TK-901", title="Rewrite dedup in Rust v1",
                state="failed", execution_log="test failure",
            ),
        ]

        provider = _run_review(state, ideas)

        for c in provider.vote.call_args_list:
            assert c[0][0] != "TK-900", (
                "a single failed match must not be enough to veto"
            )
