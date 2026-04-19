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

# Capture the genuine ``_jira_sync_background`` before the autouse
# ``_block_jira_sync`` fixture in tests/conftest.py replaces it with a
# no-op. Tests that need to exercise the real sync fan-out restore this
# reference via ``monkeypatch.setattr``. Module-scope assignment runs at
# collection time, ahead of any fixture setup.
import idea_board.models as _idea_models  # noqa: E402
_REAL_JIRA_SYNC_BACKGROUND = _idea_models._jira_sync_background


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

        This is the exact false-positive that motivated TK-742. After TK-764
        the high-overlap pair reaches the LLM judge; the mock returns
        DIFFERENT (the right verdict for a follow-up vs an abstract original)
        and ``_is_duplicate`` propagates that to False. The LLM seam is
        patched so this test does not spawn a real ``claude -p`` subprocess.
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

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(False, "scopes diverge: coverage-lift vs abstract"),
        ):
            is_dup, _reason = _is_duplicate(
                new_title="Unit tests for capability_request.py (50% -> 75%)",
                new_desc=(
                    "WHAT: Raise coverage from 50 percent to 75 percent. "
                    "WHY: Gaps exist in the error retry and rate-limit branches. "
                    "HOW: Parametrize failure modes in the circuit-breaker helper."
                ),
                existing=existing,
            )
        assert is_dup is False

    def test_near_identical_title_and_body_is_duplicate(self):
        """A restated dup (same title, same body) must still be caught.

        Creation-time dedup inside ``add_idea`` is the last line of defense
        against the classic "idea generator restates the same idea" failure
        mode. TK-742 must not weaken that. Post TK-764 the gate routes
        through the LLM judge for high-overlap pairs; the mock returns SAME
        (the right verdict for a near-identical rephrase) and the verdict
        propagates to True.
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

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(True, "near-exact match: same files, same outcome"),
        ):
            is_dup, _reason = _is_duplicate(
                new_title="Cache Ollama responses for better performance",
                new_desc=(
                    "WHY: Ollama inference repeats work for identical prompts. "
                    "HOW: Cache responses keyed by prompt hash to skip recompute."
                ),
                existing=existing,
            )
        assert is_dup is True


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

    def test_high_overlap_with_done_is_flagged_with_comment(
        self, state, mock_dedup_llm
    ):
        """Real dup-of-done shape → add_comment fires with the marker.

        Drives Step 2 via the ``mock_dedup_llm`` dedup seam so the
        assertion is independent of which judge is live. The legacy
        word-overlap heuristic returned True for this pair; the LLM
        near-exact judge (TK-743) returns ``(False, "no_binary")`` when
        no Claude binary is present — which is the normal case in CI —
        so the test would otherwise flip from green to red the moment
        the seam is rewired. Mocking the seam pins the outcome to the
        advisory-comment behavior this test is actually meant to lock
        in, regardless of whichever judge ships underneath.
        """
        mock_dedup_llm.return_value = True

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

        # But a comment carrying the "High overlap with ..." marker is added.
        flagged = any(
            c[0][0] == "TK-600" and "High overlap with TK-501" in c[0][2]
            for c in provider.add_comment.call_args_list
        )
        assert flagged, "dup-of-done must leave an advisory comment"

    def test_already_flagged_is_not_recommented(self, state):
        """Don't spam the same advisory on every review tick.

        If a prior review already dropped the ``High overlap with TK-X:``
        marker on this idea, skip it — otherwise each 30s tick would add
        another identical comment and bury the real discussion.
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
            text="High overlap with TK-701: Add caching layer v1 (Done). "
                 "Consider revising scope or closing as duplicate.",
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


# ---------------------------------------------------------------------------
# TK-761 — add_done_duplicate_flag_comment helper
# ---------------------------------------------------------------------------


class TestAddDoneDuplicateFlagComment:
    """TK-761: extracted helper for the Done-dup advisory comment path.

    ``add_done_duplicate_flag_comment(story, text)`` owns two steps:

      1. Append a ``llm``-authored ``Comment`` with the caller's text
         directly onto ``story.comments``. The Comment dataclass fills
         in the timestamp in ``__post_init__``.
      2. Fire the same background Jira sync every other state change
         on the story goes through, so the comment ends up in the Jira
         comment thread too — but only when Jira is actually
         configured.

    The helper exists so the append + sync glue is unit-testable in
    isolation from ``review_queue``; these tests pin both halves.
    """

    def test_appends_llm_comment_with_exact_text(self):
        """Comment lands on ``story.comments`` with author ``"llm"`` and the text the caller passed."""
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        story = Idea(id="idea-100", title="t", description="d")
        marker = (
            "High overlap with TK-501: Add caching layer v1 (Done). "
            "Consider revising scope or closing as duplicate."
        )

        appended = add_done_duplicate_flag_comment(story, marker)

        assert len(story.comments) == 1
        assert story.comments[0] is appended
        assert appended.author == "llm"
        assert appended.text == marker

    def test_timestamp_is_set_automatically(self):
        """The Comment dataclass fills in an ISO timestamp; the helper must not pass empty."""
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        story = Idea(id="idea-101", title="t", description="d")

        appended = add_done_duplicate_flag_comment(story, "marker text")

        assert appended.timestamp, "helper must leave the auto-fill path intact"
        # Round-trip through fromisoformat — if this raises, the format drifted.
        datetime.fromisoformat(appended.timestamp)

    def test_helper_invokes_jira_sync_background(self, monkeypatch):
        """Helper must route through ``_jira_sync_background`` so future wiring changes can't drop it silently."""
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        sync_spy = MagicMock()
        monkeypatch.setattr("idea_board.models._jira_sync_background", sync_spy)

        story = Idea(id="idea-102", title="t", description="d")
        add_done_duplicate_flag_comment(story, "marker text")

        sync_spy.assert_called_once_with(story)

    def test_jira_sync_fires_when_configured(self, monkeypatch):
        """If ``is_jira_configured()`` returns True, ``sync_idea_to_jira`` is called with the story.

        Unblocks the autouse ``_block_jira_sync`` fixture by restoring
        the real ``_jira_sync_background``, swaps ``threading.Thread``
        for a synchronous stand-in so the assertion doesn't need to
        race a real thread start, and mocks the Jira-layer seams.
        """
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        monkeypatch.setattr(
            "idea_board.models._jira_sync_background", _REAL_JIRA_SYNC_BACKGROUND,
        )

        class _SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr("idea_board.models.threading.Thread", _SyncThread)

        import idea_board.jira_sync as js
        sync_mock = MagicMock()
        monkeypatch.setattr(js, "is_jira_configured", lambda: True)
        monkeypatch.setattr(js, "sync_idea_to_jira", sync_mock)

        story = Idea(id="idea-103", title="t", description="d")
        add_done_duplicate_flag_comment(story, "marker text")

        sync_mock.assert_called_once_with(story)

    def test_jira_sync_skipped_when_unconfigured(self, monkeypatch):
        """If Jira is not configured, ``sync_idea_to_jira`` is never called."""
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        monkeypatch.setattr(
            "idea_board.models._jira_sync_background", _REAL_JIRA_SYNC_BACKGROUND,
        )

        class _SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr("idea_board.models.threading.Thread", _SyncThread)

        import idea_board.jira_sync as js
        sync_mock = MagicMock()
        monkeypatch.setattr(js, "is_jira_configured", lambda: False)
        monkeypatch.setattr(js, "sync_idea_to_jira", sync_mock)

        story = Idea(id="idea-104", title="t", description="d")
        add_done_duplicate_flag_comment(story, "marker text")

        sync_mock.assert_not_called()


# ---------------------------------------------------------------------------
# TK-765 — reason string from _is_duplicate is surfaced for observability
# ---------------------------------------------------------------------------


class TestReviewQueueReasonLogging:
    """Step 2 must use the tuple's reason for logs so outages are visible."""

    def test_llm_fell_open_is_logged_as_warning(self, state, caplog, monkeypatch):
        """When the judge returns (False, "llm_timeout"), emit a WARNING.

        Every LLM failure code (``no_binary``, ``llm_timeout``, ``llm_exit_*``,
        ``no_json``, ``parse_failure``) falls open to ``is_dup=False``.
        Without a log, a flaking Haiku silently disables Step 2 and true
        dups leak through. This test pins the visibility.
        """
        import logging

        def _fake_dedup(new_title, new_desc, existing):
            return (False, "llm_timeout")

        monkeypatch.setattr("idea_board.models._is_duplicate", _fake_dedup)

        ideas = [
            FakeIdea(
                id="TK-950",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-951",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
        ]

        with caplog.at_level(logging.WARNING, logger="aim.manager"):
            _run_review(state, ideas)

        fell_open = [
            rec for rec in caplog.records
            if "fell open" in rec.getMessage() and "llm_timeout" in rec.getMessage()
        ]
        assert fell_open, "LLM fall-open should emit a WARNING with the reason code"

    def test_flagged_dup_log_includes_reason(self, state, caplog, monkeypatch):
        """Info log on a flagged dup carries the LLM verdict reason."""
        import logging

        def _fake_dedup(new_title, new_desc, existing):
            return (True, "near-exact: same files, same outcome")

        monkeypatch.setattr("idea_board.models._is_duplicate", _fake_dedup)

        ideas = [
            FakeIdea(
                id="TK-960",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-961",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
        ]

        with caplog.at_level(logging.INFO, logger="aim.manager"):
            _run_review(state, ideas)

        flagged = [
            rec for rec in caplog.records
            if "flagged" in rec.getMessage() and "near-exact" in rec.getMessage()
        ]
        assert flagged, "flagged-dup INFO log must include the LLM reason"


# ---------------------------------------------------------------------------
# TK-762 — Step 2 helper pipeline integration
# ---------------------------------------------------------------------------


class TestFindDuplicateTargetStory:
    """``find_duplicate_target_story`` is the lookup half of Step 2."""

    def test_returns_first_match_with_reason(self):
        """First candidate that ``_is_duplicate`` flags wins — order matters.

        Locks the "first match" contract. A later test is the only way
        to notice if a future refactor switches to "best match" silently.
        """
        from idea_board.models import Idea, find_duplicate_target_story

        c1 = Idea(id="TK-A", title="unrelated story", description="alpha", state="done")
        c2 = Idea(id="TK-B", title="winner", description="beta", state="done")
        c3 = Idea(id="TK-C", title="also matches", description="gamma", state="done")

        def _fake_is_dup(new_title, new_desc, ref):
            return (ref.id in ("TK-B", "TK-C"), f"match={ref.id}")

        with patch("idea_board.models._is_duplicate", _fake_is_dup):
            ref, reason = find_duplicate_target_story("t", "d", [c1, c2, c3])

        assert ref is c2, "helper must return the first matching candidate"
        assert reason == "match=TK-B"

    def test_returns_none_and_last_reason_when_no_match(self):
        """When nothing matches, the last non-match reason is surfaced.

        Callers use the reason to detect LLM fall-open outages
        (``no_binary``, ``llm_timeout``, etc.) and emit a WARNING.
        Dropping the reason on a no-match would silently disable that
        observability path.
        """
        from idea_board.models import Idea, find_duplicate_target_story

        c1 = Idea(id="TK-A", title="x", description="y", state="done")
        c2 = Idea(id="TK-B", title="x", description="y", state="done")

        def _fake_is_dup(new_title, new_desc, ref):
            return (False, f"llm_timeout on {ref.id}")

        with patch("idea_board.models._is_duplicate", _fake_is_dup):
            ref, reason = find_duplicate_target_story("t", "d", [c1, c2])

        assert ref is None
        assert reason == "llm_timeout on TK-B", (
            "no-match path must surface the last reason so LLM outages are loggable"
        )

    def test_empty_candidates_returns_none_and_empty_reason(self):
        """No candidates → no LLM calls, no reason."""
        from idea_board.models import find_duplicate_target_story

        with patch("idea_board.models._is_duplicate") as mock_dedup:
            ref, reason = find_duplicate_target_story("t", "d", [])

        assert ref is None
        assert reason == ""
        mock_dedup.assert_not_called()


class TestIsDuplicateOfDoneStory:
    """``is_duplicate_of_done_story`` narrows Step 2 to Done refs only."""

    def test_done_ref_returns_true(self):
        from idea_board.models import Idea, is_duplicate_of_done_story

        ref = Idea(id="TK-1", title="t", description="d", state="done")
        assert is_duplicate_of_done_story(ref) is True

    def test_failed_ref_returns_false(self):
        """Dups of Failed stories are out of scope for Step 2 (TK-762).

        Step 1 already handles the repeated-failure pattern with an
        auto-veto, and a one-off Failed dup is often a legitimate
        retry. The advisory flag would just add noise.
        """
        from idea_board.models import Idea, is_duplicate_of_done_story

        ref = Idea(id="TK-1", title="t", description="d", state="failed")
        assert is_duplicate_of_done_story(ref) is False

    def test_non_terminal_state_returns_false(self):
        """Only ``done`` is flagged — no other state qualifies."""
        from idea_board.models import Idea, is_duplicate_of_done_story

        for state in ("proposed", "approved", "refining", "executing", "vetoed"):
            ref = Idea(id="TK-1", title="t", description="d", state=state)
            assert is_duplicate_of_done_story(ref) is False, (
                f"state={state!r} must not qualify as a done-duplicate"
            )


class TestDoneDuplicatePipelineTK759:
    """TK-759 pins the ``find_duplicate_target_story`` → ``is_duplicate_of_done_story``
    composition contract.

    The two helpers were split in TK-762 so each stage of Step 2 could
    be tested without standing up the full review_queue, but the
    pipeline only works if the first helper's return shape is the
    second helper's input shape. These tests exercise that boundary
    directly — a future refactor that drops the ``Idea`` out of the
    tuple, or changes the state-check field, will break here instead of
    silently disabling the advisory flow in production.
    """

    def test_done_target_from_pipeline_is_flagged(self):
        """End-to-end: dup against a Done ref returns True.

        Wires the real ``find_duplicate_target_story`` to a fake
        ``_is_duplicate`` that matches one candidate, then feeds the
        resulting ``ref`` straight into ``is_duplicate_of_done_story``.
        Done target → True, which is the signal Step 2 uses to emit
        the advisory comment.
        """
        from idea_board.models import (
            Idea,
            find_duplicate_target_story,
            is_duplicate_of_done_story,
        )

        done_ref = Idea(id="TK-900", title="shipped", description="d", state="done")
        candidates = [done_ref]

        def _fake_is_dup(new_title, new_desc, ref):
            return (True, "match=done")

        with patch("idea_board.models._is_duplicate", _fake_is_dup):
            ref, _reason = find_duplicate_target_story("t", "d", candidates)

        assert ref is done_ref
        assert is_duplicate_of_done_story(ref) is True

    def test_non_done_target_from_pipeline_is_not_flagged(self):
        """End-to-end: dup against a Failed ref returns False.

        The target state filter is the whole reason TK-759 exists — a
        Failed-state match is out of scope for the Step 2 advisory
        (Step 1's 2+-failure veto already covers it). Composition test
        confirms the filter rejects the Failed case even when
        ``_is_duplicate`` says it's a dup.
        """
        from idea_board.models import (
            Idea,
            find_duplicate_target_story,
            is_duplicate_of_done_story,
        )

        failed_ref = Idea(id="TK-901", title="tried", description="d", state="failed")
        candidates = [failed_ref]

        def _fake_is_dup(new_title, new_desc, ref):
            return (True, "match=failed")

        with patch("idea_board.models._is_duplicate", _fake_is_dup):
            ref, _reason = find_duplicate_target_story("t", "d", candidates)

        assert ref is failed_ref
        assert is_duplicate_of_done_story(ref) is False

    def test_no_match_short_circuits_state_check(self):
        """When the pipeline finds no dup, ``ref`` is None.

        Passing ``None`` into ``is_duplicate_of_done_story`` would
        raise ``AttributeError`` — Step 2 guards this with an ``if
        ref is None: return`` before the state check. Pin that
        guard-order contract so a future refactor can't swap the check
        order and crash the queue review on a non-match.
        """
        from idea_board.models import (
            find_duplicate_target_story,
            is_duplicate_of_done_story,
        )

        with patch("idea_board.models._is_duplicate", return_value=(False, "no")):
            ref, _reason = find_duplicate_target_story("t", "d", [])

        assert ref is None
        with pytest.raises(AttributeError):
            is_duplicate_of_done_story(ref)  # type: ignore[arg-type]


class TestFormatDoneDuplicateComment:
    """``format_done_duplicate_comment`` owns the advisory-comment shape.

    TK-760 pinned the exact wire-format so a future observability
    change (reason code, dedup score, etc.) has to update this one
    function — every caller either renders the pre-built string or
    matches against the ``"High overlap with {key}:"`` prefix, and
    any wording drift here would silently break the already-flagged
    dedup in ``review_queue``.
    """

    def test_exact_format_with_known_input(self):
        """Known ref → exact string. Locks every character of the contract."""
        from idea_board.models import Idea, format_done_duplicate_comment

        ref = Idea(
            id="TK-501",
            title="Add caching layer v1",
            description="caching layer for requests",
            state="done",
        )

        text = format_done_duplicate_comment(ref)

        assert text == (
            "High overlap with TK-501: Add caching layer v1 (Done). "
            "Consider revising scope or closing as duplicate."
        )

    def test_contains_marker_key_title_state_and_directive(self):
        """Operator context is locked into the advisory text.

        Four signals — the ``"High overlap with"`` grouping marker,
        the target ``key`` so the comment points at the right story,
        the ``title`` so the owner doesn't have to click through, the
        ``(Done)`` state qualifier, and the ``"Consider revising scope
        or closing as duplicate"`` directive making clear Step 2 is
        advisory — all have to land in the single string. Missing any
        of these degrades the operator's ability to act on the
        comment without opening the referenced story.
        """
        from idea_board.models import Idea, format_done_duplicate_comment

        ref = Idea(
            id="TK-501",
            title="Add caching layer v1",
            description="d",
            state="done",
        )
        text = format_done_duplicate_comment(ref)

        assert "High overlap with" in text
        assert "TK-501" in text
        assert "Add caching layer v1" in text
        assert "(Done)" in text
        assert "Consider revising scope or closing as duplicate" in text

    def test_missing_key_falls_back_to_unknown(self):
        """Empty ``id`` degrades gracefully — no ``KeyError`` or blank gap.

        The comment still has to be readable even when the ref is
        malformed; an empty key in the middle of the string would
        produce ``"High overlap with :"`` which is useless to the
        operator. Pin the fallback so a future refactor that breaks
        id population (e.g. a provider that returns partial rows)
        still yields actionable output.
        """
        from idea_board.models import Idea, format_done_duplicate_comment

        ref = Idea(id="", title="t", description="d", state="done")
        text = format_done_duplicate_comment(ref)

        assert text == (
            "High overlap with unknown: t (Done). "
            "Consider revising scope or closing as duplicate."
        )

    def test_missing_title_falls_back_to_untitled(self):
        """Empty ``title`` degrades to ``(untitled)``.

        Same rationale as the missing-key fallback — a blank title
        would collapse the format into ``"TK-501:  (Done)"`` which
        reads like a typo rather than degraded data.
        """
        from idea_board.models import Idea, format_done_duplicate_comment

        ref = Idea(id="TK-501", title="", description="d", state="done")
        text = format_done_duplicate_comment(ref)

        assert text == (
            "High overlap with TK-501: (untitled) (Done). "
            "Consider revising scope or closing as duplicate."
        )


class TestReviewQueueStep2DoneOnly:
    """TK-762: Step 2 flags Done dups only; Failed dups skip the flow."""

    def test_failed_duplicate_is_not_flagged(self, state, mock_dedup_llm):
        """A Failed-state duplicate match must not emit the advisory comment.

        The legacy code iterated ``done + failed`` and flagged both.
        TK-762 narrows Step 2 to Done refs — a Failed dup is either
        already vetoed (Step 1 pattern match) or a legitimate retry,
        and the advisory just adds noise. This test pins the new
        behavior so a future refactor can't silently widen it back.
        """
        mock_dedup_llm.return_value = True

        ideas = [
            FakeIdea(
                id="TK-810",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-811",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="failed",
                execution_log="test failure",
            ),
        ]

        provider = _run_review(state, ideas)

        # No advisory comment was added for the Failed-dup match.
        for c in provider.add_comment.call_args_list:
            assert c[0][0] != "TK-810", (
                "Failed-state duplicates must not trigger the Step 2 advisory"
            )
        # Also no veto (a single failure doesn't meet Step 1's 2+ bar).
        for c in provider.vote.call_args_list:
            assert c[0][0] != "TK-810"

    def test_done_ref_preferred_over_failed_when_both_match(
        self, state, mock_dedup_llm
    ):
        """When the first match is Failed, the pipeline short-circuits on it.

        ``find_duplicate_target_story`` returns the first match in
        iteration order (``done + failed``) and
        ``is_duplicate_of_done_story`` gates the advisory. Putting the
        Done ref first in the ordered candidates and asserting it's
        the one the marker names locks in both halves of that
        contract.
        """
        mock_dedup_llm.return_value = True

        ideas = [
            FakeIdea(
                id="TK-820",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-821",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
            FakeIdea(
                id="TK-822",
                title="Add caching layer v2",
                description="caching layer for requests",
                state="failed",
                execution_log="test failure",
            ),
        ]

        provider = _run_review(state, ideas)

        # Comment names the Done ref, not the Failed one.
        flagged_done = any(
            c[0][0] == "TK-820" and "High overlap with TK-821" in c[0][2]
            for c in provider.add_comment.call_args_list
        )
        assert flagged_done

        for c in provider.add_comment.call_args_list:
            assert "High overlap with TK-822" not in c[0][2], (
                "Failed ref must not be named in a Step 2 advisory comment"
            )


class TestReviewQueueHelperPipelineOrder:
    """``review_queue`` must call the four helpers in the documented order."""

    def test_helpers_called_in_sequence_for_done_duplicate(
        self, state, mock_dedup_llm
    ):
        """Spy every helper; verify the call order matches the contract.

        The order — find → is_done → format → add_comment — is the
        invariant this story (TK-762) wires in. If a future refactor
        flips the sequence (e.g. formatting before the Done gate) it
        silently changes when the Jira sync fires and which refs can
        leak through; this test is the tripwire.
        """
        import idea_board.models as models

        mock_dedup_llm.return_value = True

        ideas = [
            FakeIdea(
                id="TK-830",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-831",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="done",
            ),
        ]

        call_log: list[str] = []

        real_find = models.find_duplicate_target_story
        real_is_done = models.is_duplicate_of_done_story
        real_format = models.format_done_duplicate_comment
        real_add = models.add_done_duplicate_flag_comment

        def _spy_find(*args, **kwargs):
            call_log.append("find")
            return real_find(*args, **kwargs)

        def _spy_is_done(*args, **kwargs):
            call_log.append("is_done")
            return real_is_done(*args, **kwargs)

        def _spy_format(*args, **kwargs):
            call_log.append("format")
            return real_format(*args, **kwargs)

        def _spy_add(*args, **kwargs):
            call_log.append("add")
            return real_add(*args, **kwargs)

        from aim.manager import review_queue

        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas
        mock_provider.get_comments.return_value = []

        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"), \
             patch("idea_board.models.find_duplicate_target_story", _spy_find), \
             patch("idea_board.models.is_duplicate_of_done_story", _spy_is_done), \
             patch("idea_board.models.format_done_duplicate_comment", _spy_format), \
             patch("idea_board.models.add_done_duplicate_flag_comment", _spy_add):
            review_queue(state)

        # Only the Done-dup idea drives the pipeline; the Done ref
        # itself has no active-state entry in still_active. So the
        # sequence fires exactly once, in order.
        assert call_log == ["find", "is_done", "format", "add"], (
            f"helpers must fire in documented order; got {call_log}"
        )

    def test_non_done_match_skips_format_and_add(self, state, mock_dedup_llm):
        """When the dup is Failed, the pipeline stops after is_done.

        ``format_done_duplicate_comment`` and
        ``add_done_duplicate_flag_comment`` are never reached. That's
        the whole point of the Done-only gate — work that doesn't
        need flagging doesn't pay the formatting cost or emit a
        comment.
        """
        import idea_board.models as models

        mock_dedup_llm.return_value = True

        ideas = [
            FakeIdea(
                id="TK-840",
                title="Add caching layer",
                description="caching layer for requests",
                state="approved",
            ),
            FakeIdea(
                id="TK-841",
                title="Add caching layer v1",
                description="caching layer for requests",
                state="failed",
                execution_log="test failure",
            ),
        ]

        real_find = models.find_duplicate_target_story
        real_is_done = models.is_duplicate_of_done_story

        from aim.manager import review_queue

        mock_provider = MagicMock()
        mock_provider.load_all.return_value = ideas
        mock_provider.get_comments.return_value = []

        with patch("board.get_provider", return_value=mock_provider), \
             patch("aim.manager._notify_discord"), \
             patch(
                 "idea_board.models.find_duplicate_target_story",
                 side_effect=real_find,
             ) as find_spy, \
             patch(
                 "idea_board.models.is_duplicate_of_done_story",
                 side_effect=real_is_done,
             ) as is_done_spy, \
             patch(
                 "idea_board.models.format_done_duplicate_comment",
             ) as format_spy, \
             patch(
                 "idea_board.models.add_done_duplicate_flag_comment",
             ) as add_spy:
            review_queue(state)

        assert find_spy.called, "find must always run for an active idea"
        assert is_done_spy.called, "is_done gate must always run when a match is found"
        format_spy.assert_not_called()
        add_spy.assert_not_called()


class TestAddDoneDuplicateFlagCommentProviderMode:
    """TK-762: helper gains a provider arg so review_queue can wire it in.

    The unit-test path (no provider) is covered by
    ``TestAddDoneDuplicateFlagComment`` above — these tests pin the new
    provider-aware branch.
    """

    def test_provider_add_comment_called_with_story_id_author_text(self):
        """When a provider is supplied, persistence goes through it."""
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        story = Idea(id="idea-200", title="t", description="d")
        provider = MagicMock()

        add_done_duplicate_flag_comment(story, "marker text", provider=provider)

        provider.add_comment.assert_called_once_with("idea-200", "llm", "marker text")

    def test_provider_mode_does_not_mutate_story_comments(self):
        """Provider owns persistence; the in-memory list stays untouched.

        Avoids a double-append once the provider's own write lands
        in ``story.comments`` on the next load.
        """
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        story = Idea(id="idea-201", title="t", description="d")
        provider = MagicMock()

        add_done_duplicate_flag_comment(story, "marker text", provider=provider)

        assert story.comments == [], (
            "with provider supplied, the helper must not mutate story.comments"
        )

    def test_provider_mode_skips_direct_jira_sync(self, monkeypatch):
        """When provider is given, the helper doesn't fire its own Jira sync.

        The provider's ``add_comment`` already runs Jira sync for the
        story. Firing it again here would double-sync — harmless but
        wasteful, and obscures which write actually triggered a Jira
        update when debugging.
        """
        from idea_board.models import Idea, add_done_duplicate_flag_comment

        sync_spy = MagicMock()
        monkeypatch.setattr("idea_board.models._jira_sync_background", sync_spy)

        story = Idea(id="idea-202", title="t", description="d")
        provider = MagicMock()

        add_done_duplicate_flag_comment(story, "marker text", provider=provider)

        sync_spy.assert_not_called()
