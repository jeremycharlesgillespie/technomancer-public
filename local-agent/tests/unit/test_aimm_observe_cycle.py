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
