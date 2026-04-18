"""Isolation tests for aimm.observe_cycle — observer + suggester.

Covers the two concerns the per-scorer unit tests can't:

  1. Both loops run in a single cycle. The observer produces an
     observation entry and the suggester produces a suggestion marker,
     each tagged with the matching ``Finding type:`` value in
     ``raw_findings.md``. Ensures state from one loop doesn't leak into
     the other.

  2. With no pending-approval stories, the suggester loop is a no-op —
     ``suggestions_logged=0`` and the observer loop still produces its
     summary dict with the full expected key set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aimm import observe_cycle
from aimm.observer import Observation
from aimm.suggester import Suggestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_RUBRIC = (
    "# Paper-Worthiness Rubric\n\n"
    "Criteria: concrete, reproducible, narrative-forward, observable metric."
)


@pytest.fixture
def rubric_file(tmp_path: Path) -> Path:
    path = tmp_path / "paper_rubric.md"
    path.write_text(SAMPLE_RUBRIC, encoding="utf-8")
    return path


@pytest.fixture
def findings_file(tmp_path: Path) -> Path:
    return tmp_path / "raw_findings.md"


@pytest.fixture
def shipped_story() -> dict[str, Any]:
    return {
        "key": "TK-660",
        "id": "TK-660",
        "title": "Add observer scoring",
        "description": "Scores shipped stories for paper-worthiness.",
        "category": "feature",
        "state": "done",
        "labels": ["cat:feature"],
    }


@pytest.fixture
def pending_story() -> dict[str, Any]:
    return {
        "key": "TK-661",
        "id": "TK-661",
        "title": "Log approval suggestions",
        "description": "Read-only suggester that logs approval verdicts.",
        "category": "feature",
        "state": "proposed",
        "labels": ["pending-approval", "cat:feature"],
    }


def _make_provider(
    done: list[dict[str, Any]] | None = None,
    proposed: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a MagicMock BoardProvider.

    ``list_by_state("done")`` returns ``done``;
    ``list_by_state("proposed")`` returns ``proposed``. Any other state
    returns an empty list. This matches the contract that
    ``_list_pending_approval_stories`` filters by the pending-approval
    label itself, not via state name.
    """
    provider = MagicMock()

    def list_by_state(state: str) -> list[dict[str, Any]]:
        if state == "done":
            return list(done or [])
        if state == "proposed":
            return list(proposed or [])
        return []

    provider.list_by_state.side_effect = list_by_state
    # Optional enrichment hooks default to returning empty/safe values.
    provider.get_commit_for_story.return_value = {
        "sha": "abc1234",
        "message": "shipped",
        "diff": "",
    }
    provider.get_test_output_for_story.return_value = ""
    return provider


# ---------------------------------------------------------------------------
# Test 1 — observer and suggester both run in a single cycle
# ---------------------------------------------------------------------------


class TestBothLoopsRunInOneCycle:
    def test_observation_and_suggestion_both_logged(
        self,
        rubric_file: Path,
        findings_file: Path,
        shipped_story: dict[str, Any],
        pending_story: dict[str, Any],
    ) -> None:
        """Observer and suggester run sequentially in one cycle.

        Acceptance: raw_findings.md has exactly 1 ``Finding type:
        observation`` line and exactly 1 ``Finding type: suggestion``
        line; the summary dict counts both.
        """
        provider = _make_provider(done=[shipped_story], proposed=[pending_story])

        # Observer returns a paper-worthy observation for the shipped story.
        worthy = Observation(
            finding_worthy=True,
            headline="Race condition in worker",
            why_it_matters="Regression would silently deadlock the queue.",
            evidence_pointer="abc1234 / aim/worker.py:120 / latency=5s",
            theme="failure-mode discoveries",
            reason="ok",
        )
        # Suggester returns a recommend-approve verdict for the pending story.
        recommended = Suggestion(
            story_key=pending_story["key"],
            recommend=True,
            reasoning="Story adds a concrete, observable metric.",
            reason="ok",
        )

        with patch(
            "aimm.observe_cycle.observer.score_shipped_story",
            return_value=worthy,
        ) as score_mock, patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            return_value=recommended,
        ) as suggest_mock:
            summary = observe_cycle.run(
                provider,
                "TK",
                state={},
                rubric_path=rubric_file,
                findings_path=findings_file,
                cycle_id="test-cycle-1",
            )

        # Both scorers were invoked exactly once with the right story.
        assert score_mock.call_count == 1
        assert suggest_mock.call_count == 1
        assert score_mock.call_args.args[0]["key"] == shipped_story["key"]
        assert suggest_mock.call_args.args[0]["key"] == pending_story["key"]

        # Summary reflects one of each.
        assert summary == {
            "observed": 1,
            "findings_logged": 1,
            "suggestions_logged": 1,
        }

        # raw_findings.md contains exactly one of each finding_type line.
        body = findings_file.read_text(encoding="utf-8")
        assert body.count("Finding type: observation") == 1
        assert body.count("Finding type: suggestion") == 1
        # And the entries are keyed to the right stories.
        assert shipped_story["key"] in body
        assert pending_story["key"] in body

    def test_suggester_failure_does_not_block_observation(
        self,
        rubric_file: Path,
        findings_file: Path,
        shipped_story: dict[str, Any],
        pending_story: dict[str, Any],
    ) -> None:
        """The two loops don't share failure state.

        If the suggester reports a non-``ok`` reason, the observation
        still lands in raw_findings.md and ``findings_logged`` still
        reflects it. ``suggestions_logged`` is zero because only
        ``recommend=True`` verdicts with ``reason='ok'`` count.
        """
        provider = _make_provider(done=[shipped_story], proposed=[pending_story])

        worthy = Observation(
            finding_worthy=True,
            headline="Measurable latency drop",
            why_it_matters="Shows a 40% p95 improvement.",
            evidence_pointer="abc1234 / metrics.py:44",
            theme="measurement + benchmarks",
            reason="ok",
        )
        failed_suggestion = Suggestion(
            story_key=pending_story["key"],
            recommend=False,
            reasoning="",
            reason="llm_error",
        )

        with patch(
            "aimm.observe_cycle.observer.score_shipped_story",
            return_value=worthy,
        ), patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            return_value=failed_suggestion,
        ):
            summary = observe_cycle.run(
                provider,
                "TK",
                state={},
                rubric_path=rubric_file,
                findings_path=findings_file,
            )

        assert summary["observed"] == 1
        assert summary["findings_logged"] == 1
        assert summary["suggestions_logged"] == 0

        body = findings_file.read_text(encoding="utf-8")
        assert body.count("Finding type: observation") == 1
        assert "Finding type: suggestion" not in body


# ---------------------------------------------------------------------------
# Test 2 — suggester returns 0 when no pending-approval stories exist
# ---------------------------------------------------------------------------


class TestEmptyPendingList:
    def test_no_pending_stories_yields_zero_suggestions(
        self,
        rubric_file: Path,
        findings_file: Path,
        shipped_story: dict[str, Any],
    ) -> None:
        """Empty pending list → suggester is never invoked, suggestions_logged=0.

        Acceptance: the summary dict is still fully shaped
        ``{observed, findings_logged, suggestions_logged}``.
        """
        provider = _make_provider(done=[shipped_story], proposed=[])

        worthy = Observation(
            finding_worthy=True,
            headline="Finding headline",
            why_it_matters="Why it matters.",
            evidence_pointer="abc1234",
            theme="novel autonomy mechanisms",
            reason="ok",
        )

        with patch(
            "aimm.observe_cycle.observer.score_shipped_story",
            return_value=worthy,
        ), patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            return_value=Suggestion(reason="ok"),
        ) as suggest_mock:
            summary = observe_cycle.run(
                provider,
                "TK",
                state={},
                rubric_path=rubric_file,
                findings_path=findings_file,
            )

        # Suggester must not be invoked when the pending list is empty.
        suggest_mock.assert_not_called()

        # Summary is fully shaped and suggestions_logged is zero.
        assert set(summary.keys()) == {
            "observed",
            "findings_logged",
            "suggestions_logged",
        }
        assert summary["suggestions_logged"] == 0
        assert summary["observed"] == 1
        assert summary["findings_logged"] == 1

    def test_pending_stories_without_label_are_ignored(
        self,
        rubric_file: Path,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Proposed stories missing the pending-approval label don't reach the suggester.

        Keeps the two-loops-are-independent contract honest: the
        suggester's input list is filtered by label, so unlabeled
        proposed items look the same as "no pending stories" to it.
        """
        unlabeled = dict(pending_story)
        unlabeled["labels"] = ["cat:feature"]  # no pending-approval
        provider = _make_provider(done=[], proposed=[unlabeled])

        with patch(
            "aimm.observe_cycle.observer.score_shipped_story"
        ) as score_mock, patch(
            "aimm.observe_cycle.suggester.suggest_approval"
        ) as suggest_mock:
            summary = observe_cycle.run(
                provider,
                "TK",
                state={},
                rubric_path=rubric_file,
                findings_path=findings_file,
            )

        score_mock.assert_not_called()
        suggest_mock.assert_not_called()
        assert summary == {
            "observed": 0,
            "findings_logged": 0,
            "suggestions_logged": 0,
        }
        # raw_findings.md is never created for a fully-empty cycle.
        assert not findings_file.exists()


# ---------------------------------------------------------------------------
# Test 3 — _append_suggestions_to_findings writes one entry per suggestion
# ---------------------------------------------------------------------------


class TestAppendSuggestionsToFindings:
    """Direct tests for the ``_append_suggestions_to_findings`` helper.

    The helper is the single write-path for suggestion markers, so tests
    here don't touch the provider or observer at all — they verify the
    appender alone produces the expected ``Finding type: suggestion``
    markers with ``story_key`` and ``reasoning`` in the content.
    """

    def test_single_suggestion_writes_one_finding_type_line(
        self,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """One suggestion in → exactly one ``Finding type: suggestion`` line out."""
        suggestion = Suggestion(
            story_key=pending_story["key"],
            recommend=True,
            reasoning="Concrete, observable metric.",
            reason="ok",
        )

        count = observe_cycle._append_suggestions_to_findings(
            [(suggestion, pending_story)], findings_file
        )

        assert count == 1
        body = findings_file.read_text(encoding="utf-8")
        assert body.count("Finding type: suggestion") == 1
        assert body.count("Finding type: observation") == 0

    def test_entry_includes_story_key_and_reasoning(
        self,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Each appended entry carries the story key and the suggester reasoning."""
        reasoning = "Story adds a reproducible latency benchmark."
        suggestion = Suggestion(
            story_key=pending_story["key"],
            recommend=True,
            reasoning=reasoning,
            reason="ok",
        )

        observe_cycle._append_suggestions_to_findings(
            [(suggestion, pending_story)], findings_file
        )

        body = findings_file.read_text(encoding="utf-8")
        assert pending_story["key"] in body
        assert reasoning in body

    def test_multiple_suggestions_append_one_line_each(
        self,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Iterating N suggestions produces N ``Finding type: suggestion`` lines."""
        second_story = dict(pending_story)
        second_story["key"] = "TK-999"
        second_story["title"] = "Second pending story"

        suggestions = [
            (
                Suggestion(
                    story_key=pending_story["key"],
                    recommend=True,
                    reasoning="First reasoning.",
                    reason="ok",
                ),
                pending_story,
            ),
            (
                Suggestion(
                    story_key=second_story["key"],
                    recommend=True,
                    reasoning="Second reasoning.",
                    reason="ok",
                ),
                second_story,
            ),
        ]

        count = observe_cycle._append_suggestions_to_findings(
            suggestions, findings_file
        )

        assert count == 2
        body = findings_file.read_text(encoding="utf-8")
        assert body.count("Finding type: suggestion") == 2
        assert pending_story["key"] in body
        assert second_story["key"] in body
        assert "First reasoning." in body
        assert "Second reasoning." in body

    def test_empty_list_is_a_noop(
        self,
        findings_file: Path,
    ) -> None:
        """Empty input list → no file created, count is zero."""
        count = observe_cycle._append_suggestions_to_findings([], findings_file)

        assert count == 0
        assert not findings_file.exists()


# ---------------------------------------------------------------------------
# Test 4 — _score_pending_suggestions filters to recommend+ok verdicts
# ---------------------------------------------------------------------------


class TestScorePendingSuggestions:
    """Direct tests for the ``_score_pending_suggestions`` helper.

    The helper owns the suggester loop: it iterates the pending-approval
    queue, delegates to :func:`aimm.suggester.suggest_approval`, and
    returns only the ``reason='ok'`` + ``recommend=True`` verdicts. Skips
    and error verdicts are dropped (the suggester has already persisted
    its own record).
    """

    def test_only_recommend_ok_verdicts_returned(
        self,
        rubric_file: Path,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Mixed verdicts in → only ``recommend=True`` + ``reason='ok'`` out."""
        second = dict(pending_story)
        second["key"] = "TK-900"
        third = dict(pending_story)
        third["key"] = "TK-901"
        provider = _make_provider(proposed=[pending_story, second, third])

        verdicts = {
            pending_story["key"]: Suggestion(
                story_key=pending_story["key"],
                recommend=True,
                reasoning="Good metric.",
                reason="ok",
            ),
            second["key"]: Suggestion(
                story_key=second["key"],
                recommend=False,
                reasoning="",
                reason="ok",
            ),
            third["key"]: Suggestion(
                story_key=third["key"],
                recommend=True,
                reasoning="",
                reason="llm_error",
            ),
        }

        def fake_suggest(story, rubric, *, findings_path, cycle_id):
            return verdicts[story["key"]]

        with patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            side_effect=fake_suggest,
        ) as suggest_mock:
            scored = observe_cycle._score_pending_suggestions(
                provider,
                "TK",
                SAMPLE_RUBRIC,
                findings_path=findings_file,
                cycle_id="test-cycle",
            )

        assert suggest_mock.call_count == 3
        assert len(scored) == 1
        assert scored[0][0].story_key == pending_story["key"]
        assert scored[0][1]["key"] == pending_story["key"]

    def test_no_pending_stories_returns_empty(
        self,
        rubric_file: Path,
        findings_file: Path,
    ) -> None:
        """Empty pending queue → suggester never invoked, empty list returned."""
        provider = _make_provider(proposed=[])

        with patch(
            "aimm.observe_cycle.suggester.suggest_approval"
        ) as suggest_mock:
            scored = observe_cycle._score_pending_suggestions(
                provider,
                "TK",
                SAMPLE_RUBRIC,
                findings_path=findings_file,
            )

        suggest_mock.assert_not_called()
        assert scored == []

    def test_jira_fetch_failure_returns_empty_and_logs(
        self,
        rubric_file: Path,
        findings_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Provider fetch raises → empty list returned + warning logged.

        ``_list_pending_approval_stories`` wraps the provider call in a
        try/except so a Jira outage can't abort the whole cycle. The
        suggester is never invoked and the return list is empty.
        """
        provider = MagicMock()
        provider.list_by_state.side_effect = RuntimeError("Jira API down")

        with caplog.at_level("WARNING", logger="aimm.observe_cycle"):
            with patch(
                "aimm.observe_cycle.suggester.suggest_approval"
            ) as suggest_mock:
                scored = observe_cycle._score_pending_suggestions(
                    provider,
                    "TK",
                    SAMPLE_RUBRIC,
                    findings_path=findings_file,
                )

        suggest_mock.assert_not_called()
        assert scored == []
        assert any(
            "list_by_state('proposed') failed" in rec.getMessage()
            for rec in caplog.records
        )

    def test_suggester_failure_on_one_story_skips_it_scores_others(
        self,
        rubric_file: Path,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """One story returns a non-``ok`` verdict → it's filtered; the other remains.

        The suggester never raises (it turns every failure path into a
        ``Suggestion`` with a non-``ok`` ``reason``). The score helper
        iterates the full pending list and drops failed verdicts on the
        ``reason='ok' and recommend`` guard.
        """
        good_story = dict(pending_story)
        good_story["key"] = "TK-800"
        bad_story = dict(pending_story)
        bad_story["key"] = "TK-801"
        provider = _make_provider(proposed=[good_story, bad_story])

        verdicts = {
            good_story["key"]: Suggestion(
                story_key=good_story["key"],
                recommend=True,
                reasoning="Great story.",
                reason="ok",
            ),
            bad_story["key"]: Suggestion(
                story_key=bad_story["key"],
                recommend=False,
                reasoning="",
                reason="llm_error",
            ),
        }

        def fake_suggest(story, rubric, *, findings_path, cycle_id):
            return verdicts[story["key"]]

        with patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            side_effect=fake_suggest,
        ) as suggest_mock:
            scored = observe_cycle._score_pending_suggestions(
                provider,
                "TK",
                SAMPLE_RUBRIC,
                findings_path=findings_file,
            )

        # Both stories were attempted.
        assert suggest_mock.call_count == 2
        # Only the ok + recommend story survived.
        assert len(scored) == 1
        assert scored[0][0].story_key == good_story["key"]

    def test_scored_result_has_proper_dict_format(
        self,
        rubric_file: Path,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Return list is ``[(Suggestion, story-dict), ...]`` with source fields intact.

        The story dict preserves the original key/title/description
        fields so downstream callers can pass it straight to the
        findings appender without another provider round-trip.
        """
        provider = _make_provider(proposed=[pending_story])
        recommended = Suggestion(
            story_key=pending_story["key"],
            recommend=True,
            reasoning="Measurable benchmark.",
            reason="ok",
        )

        with patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            return_value=recommended,
        ):
            scored = observe_cycle._score_pending_suggestions(
                provider,
                "TK",
                SAMPLE_RUBRIC,
                findings_path=findings_file,
                cycle_id="test-cycle",
            )

        assert len(scored) == 1
        sug, story = scored[0]
        assert isinstance(sug, Suggestion)
        assert sug.story_key == pending_story["key"]
        assert sug.recommend is True
        assert sug.reason == "ok"
        assert isinstance(story, dict)
        assert story["key"] == pending_story["key"]
        assert story["title"] == pending_story["title"]
        assert story["description"] == pending_story["description"]


# ---------------------------------------------------------------------------
# Test 5 — run() returns the full summary dict with multi-observation input
# ---------------------------------------------------------------------------


class TestRunSummaryShape:
    """Verify the ``run()`` summary dict on a mixed-input cycle."""

    def test_two_observations_one_suggestion_summary(
        self,
        rubric_file: Path,
        findings_file: Path,
        pending_story: dict[str, Any],
    ) -> None:
        """Two shipped + one pending-approval → summary counts each stream.

        Both shipped stories score as finding-worthy observations; one
        pending story recommends approval. The summary dict keeps
        observations and suggestions as separate counters so a caller
        can drive ``research_log.md`` without reparsing markdown.
        """
        shipped_a = {
            "key": "TK-700",
            "id": "TK-700",
            "title": "First shipped",
            "description": "desc a",
            "state": "done",
            "labels": ["cat:feature"],
        }
        shipped_b = {
            "key": "TK-701",
            "id": "TK-701",
            "title": "Second shipped",
            "description": "desc b",
            "state": "done",
            "labels": ["cat:quality"],
        }
        provider = _make_provider(
            done=[shipped_a, shipped_b],
            proposed=[pending_story],
        )

        worthy = Observation(
            finding_worthy=True,
            headline="Headline",
            why_it_matters="Why.",
            evidence_pointer="abc",
            theme="theme",
            reason="ok",
        )
        recommended = Suggestion(
            story_key=pending_story["key"],
            recommend=True,
            reasoning="Good.",
            reason="ok",
        )

        with patch(
            "aimm.observe_cycle.observer.score_shipped_story",
            return_value=worthy,
        ), patch(
            "aimm.observe_cycle.suggester.suggest_approval",
            return_value=recommended,
        ):
            summary = observe_cycle.run(
                provider,
                "TK",
                state={},
                rubric_path=rubric_file,
                findings_path=findings_file,
            )

        assert set(summary.keys()) == {
            "observed",
            "findings_logged",
            "suggestions_logged",
        }
        assert summary["observed"] == 2
        assert summary["findings_logged"] == 2
        assert summary["suggestions_logged"] == 1

        body = findings_file.read_text(encoding="utf-8")
        assert body.count("Finding type: observation") == 2
        assert body.count("Finding type: suggestion") == 1
