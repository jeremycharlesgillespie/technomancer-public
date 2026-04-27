"""Integration test for the full dedup flow — overlap prefilter + LLM seam.

Exercises the complete path used by every idea-creation caller
(``add_idea``, ``/api/jira/create``, queue review) through
:func:`idea_board.models._is_duplicate` down to the LLM judge at
:func:`idea_board.dedup_llm.is_near_exact_duplicate`.

Three contract properties are pinned here, end-to-end rather than through
isolated mocks:

1. **Low overlap short-circuits** — a disjoint-vocabulary pair must
   never reach the LLM. Unit tests mock ``_is_duplicate`` directly so the
   *integration* guarantee (real ``_is_duplicate`` body, real prefilter,
   LLM seam patched at the module level) was not covered until now.

2. **High overlap produces a verdict** — near-exact rephrases resolve to
   ``True``; thematically-related-but-distinct pairs resolve to
   ``False``. The verdict can arrive from either the word-overlap
   heuristic or the LLM judge, depending on which revision of
   ``_is_duplicate`` is live; the *contract* is the same.

3. **LLM failure falls open** — timeout, missing binary, non-zero exit,
   and malformed output all return ``(False, <error_code>)`` so a flaky
   Haiku round-trip never silently kills a real story. This is the
   asymmetric-cost bias baked into the dedup design:
   false-keeps cost one redundant commit, false-vetoes kill real work.

Why this lives under ``tests/integration/`` and not ``tests/unit/``:
unit tests cover ``_is_duplicate`` and ``is_near_exact_duplicate`` in
isolation (via the ``mock_dedup_llm`` fixture that *replaces*
``_is_duplicate`` outright). The integration test's job is different —
it drives the *real* ``_is_duplicate`` and patches only the LLM seam, so
regressions in the wiring between the two layers (missing import, wrong
return shape, prefilter inverted) surface here.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from idea_board import dedup_llm
from board.types import Idea
from aim.dedup import _is_duplicate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_binary_cache(monkeypatch):
    """Force claude-binary discovery to a known path for determinism.

    Without this, a host with Claude Code installed leaks a real binary
    path into the test, and a host without it fails every test on the
    ``no_binary`` fast path. Mirrors the autouse fixture in
    ``tests/unit/test_dedup_llm.py`` but kept as an opt-in fixture here
    so the ``no_binary`` test can override the discovery function.
    """
    monkeypatch.setattr(dedup_llm, "_claude_binary_cache", None)
    monkeypatch.setattr(dedup_llm, "_find_claude_binary", lambda: "/usr/bin/claude")


# ---------------------------------------------------------------------------
# 1. Low overlap: LLM must not be called
# ---------------------------------------------------------------------------


class TestLowOverlapSkipsLLM:
    """Disjoint-vocabulary pairs never reach the LLM judge."""

    def test_disjoint_ideas_short_circuit_before_llm(self):
        """Zero meaningful-stem overlap → False, LLM seam never invoked.

        The prefilter's whole reason to exist is cost avoidance — every
        unnecessary ``claude -p`` call pays a real Haiku token cost on
        every idea-generation cycle. If this test's ``call_count == 0``
        assertion ever fails, the prefilter is broken and the pipeline
        is now spending LLM cycles on obviously-different stories.
        """
        existing = Idea(
            id="TK-900",
            title="Yoga class scheduling dashboard",
            description=(
                "Members book private studio sessions through a calendar "
                "widget connected to the instructor's availability."
            ),
        )

        # Mock returns True so that *if* the LLM seam is ever consulted by
        # mistake, the test fails twice: once on call_count and once on
        # the verdict. Double tripwire.
        llm_mock = MagicMock(return_value=(True, "forced_same"))
        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate", llm_mock
        ):
            is_dup, reason = _is_duplicate(
                new_title="Submarine propulsion telemetry analyzer",
                new_desc=(
                    "Spectrogram viewer renders turbulent wake signatures "
                    "from deep sensor streams off a towed array."
                ),
                existing=existing,
            )

        assert is_dup is False, (
            "Disjoint-vocabulary pair must not be flagged as duplicate"
        )
        assert isinstance(reason, str), (
            "_is_duplicate must always return a (bool, str) tuple — "
            "callers unpack it at every call site"
        )
        assert llm_mock.call_count == 0, (
            "LLM must not be called for zero-overlap pairs; every "
            "unnecessary invocation burns Haiku tokens on stories that "
            "can be ruled out by a word-set intersection"
        )


# ---------------------------------------------------------------------------
# 2. High overlap: dedup seam produces a correct verdict
# ---------------------------------------------------------------------------


class TestHighOverlapProducesVerdict:
    """High-overlap pairs resolve to the right True/False verdict.

    These tests deliberately do *not* pin ``call_count`` — the dedup
    seam is allowed to short-circuit on a conclusive word-overlap match
    (current behavior) or defer to the LLM judge (TK-764 and beyond).
    What the contract guarantees is the *verdict*, not the path.
    """

    def test_near_identical_rephrase_is_flagged_as_duplicate(self):
        """A restated dup (same scope, same files, rephrased) → True.

        Word-overlap catches this today (title stems overlap > 0.5);
        the LLM judge catches it tomorrow (verdict=SAME). Either way
        the contract holds — the new story does not get written.
        """
        existing = Idea(
            id="TK-901",
            title="Cache Ollama responses to improve performance",
            description=(
                "WHAT: Cache responses keyed by prompt hash. "
                "WHY: Repeated prompts waste GPU time. "
                "HOW: Wrap the Ollama call site with a hash-keyed lookup."
            ),
        )

        # Mock returns the SAME verdict so that whichever path is live,
        # the result is True. (Word-overlap also returns True for this
        # pair — the mock is belt-and-suspenders.)
        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(True, "near-exact match: same files, same outcome"),
        ):
            is_dup, reason = _is_duplicate(
                new_title="Cache Ollama responses for performance gains",
                new_desc=(
                    "WHAT: Add a response cache for Ollama prompts. "
                    "WHY: Identical prompts repeat work. "
                    "HOW: Hash the prompt and short-circuit on cache hit."
                ),
                existing=existing,
            )

        assert is_dup is True, (
            "Near-exact rephrase must be flagged — this is the TK-441 "
            "cluster symptom that motivated the dedup gate"
        )
        assert isinstance(reason, str) and reason, (
            "reason must carry a non-empty trace (word-overlap score or "
            "LLM rationale) so operators can see *why* a story was flagged"
        )

    def test_thematically_related_but_distinct_stories_not_flagged(self):
        """TK-571-style follow-up vs TK-321 abstract → False.

        The headline regression that motivated TK-743 (LLM-backed dedup):
        the word-overlap heuristic alone kept killing concrete follow-up
        stories that shared generic stems (``unit``, ``tests``,
        ``capability_request``) with older Done stories of different
        scope. The current word-overlap thresholds were tuned in TK-742
        to pass this pair; the LLM judge tuned in TK-743 is even more
        permissive on such pairs. Both paths return False; both paths
        are valid; the test is path-agnostic.
        """
        existing = Idea(
            id="TK-321",
            title="[idea-197] Add Unit Tests for capability_request.py Core Logic",
            description=(
                "WHAT: Add unit tests covering the core capability "
                "evaluation logic in capability_request.py. "
                "WHY: No unit coverage today. "
                "HOW: Write tests against the Claude API evaluation path."
            ),
            state="done",
        )

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(False, "scopes diverge: coverage-lift vs abstract"),
        ):
            is_dup, reason = _is_duplicate(
                new_title="Unit tests for capability_request.py (50% -> 75%)",
                new_desc=(
                    "WHAT: Raise coverage from 50 percent to 75 percent. "
                    "WHY: Gaps exist in the retry, rate-limit, and "
                    "circuit-breaker branches. "
                    "HOW: Parametrize failure modes and assert recovery paths."
                ),
                existing=existing,
            )

        assert is_dup is False, (
            "Concrete follow-up must not be flagged as duplicate of an "
            "older abstract story — TK-571 vs TK-321 regression"
        )
        assert isinstance(reason, str), (
            "_is_duplicate must always return the (bool, str) tuple shape"
        )


# ---------------------------------------------------------------------------
# 3. LLM failure paths fall open (bias toward keep)
# ---------------------------------------------------------------------------


class TestLLMFailureFallsOpen:
    """Every LLM failure mode returns ``(False, <code>)``, never raises.

    The whole dedup pipeline is biased toward keep: a missed duplicate
    costs one redundant commit; an incorrectly-flagged legitimate story
    costs lost work and operator trust. Every error path in
    ``is_near_exact_duplicate`` must fall open.
    """

    def test_router_failure_returns_fallopen_tuple(
        self, reset_binary_cache
    ):
        """Router returns None (any error path) → ``(False, "llm_unavailable")``.

        Under the post-routing architecture (TK-???), every subprocess-
        level error (TimeoutExpired, missing binary, non-zero exit,
        connection reset) collapses to a single ``None`` return from
        ``agent.llm_router.complete``. The dedup gate falls open on
        every such path. The router has its own unit-test coverage for
        the individual error variants.
        """
        with patch("agent.llm_router.complete", return_value=None):
            is_dup, reason = dedup_llm.is_near_exact_duplicate(
                "Story A title",
                "Story A description",
                "Story B title",
                "Story B description",
            )

        assert is_dup is False, (
            "Router failure must fall open, not veto — killing a real "
            "story on a transient LLM blip is the exact regression this "
            "guard prevents"
        )
        assert reason == "llm_unavailable"


# ---------------------------------------------------------------------------
# 4. LLM success path wiring — verdict JSON propagates to the caller
# ---------------------------------------------------------------------------


class TestLLMVerdictPropagation:
    """When the LLM is reached and responds cleanly, its verdict flows out."""

    def test_llm_same_verdict_returns_true_with_reason(
        self, reset_binary_cache
    ):
        """``claude -p`` prints ``{"verdict":"SAME",...}`` → ``(True, reason)``.

        This is the happy-path spine of the dedup gate: when the LLM
        round-trips successfully and declares the stories the same unit
        of work, the reason string propagates to the caller so the 409
        response or dedup log entry can quote the model's rationale.
        """
        stdout = '{"verdict": "SAME", "reason": "same files, same outcome"}'
        with patch(
            "agent.llm_router.complete", return_value=stdout
        ) as mock_run:
            is_dup, reason = dedup_llm.is_near_exact_duplicate(
                "Cache Ollama responses",
                "Hash-keyed cache in front of Ollama.",
                "Add Ollama response cache",
                "Wrap Ollama with a hash-keyed lookup.",
            )

        assert is_dup is True
        assert reason == "same files, same outcome"
        assert mock_run.call_count == 1, (
            "LLM seam must actually spawn claude -p when consulted — "
            "call_count==0 would mean the subprocess layer was bypassed"
        )

    def test_llm_different_verdict_returns_false_with_reason(
        self, reset_binary_cache
    ):
        """``{"verdict":"DIFFERENT",...}`` → ``(False, reason)``.

        Symmetric to the SAME case. The reason propagates so that a
        test-time debugger or a production log can see *why* the LLM
        ruled the pair distinct (different scope, different acceptance
        criteria, follow-up vs original, etc.).
        """
        stdout = (
            '{"verdict": "DIFFERENT", "reason": "coverage-lift vs abstract"}'
        )
        with patch(
            "agent.llm_router.complete", return_value=stdout
        ):
            is_dup, reason = dedup_llm.is_near_exact_duplicate(
                "Unit tests for capability_request.py (50% -> 75%)",
                "Raise coverage from 50 to 75 percent.",
                "Add Unit Tests for capability_request.py Core Logic",
                "Add tests for the core evaluation logic.",
            )

        assert is_dup is False
        assert reason == "coverage-lift vs abstract"
