"""Unit tests for ``idea_board.models._is_duplicate`` — TK-764 contract.

The dedup gate is a two-stage pipeline:

1. **Word-overlap prefilter** (cheap) — pairs with combined meaningful-stem
   overlap below ``OVERLAP_PREFILTER_THRESHOLD`` (20%) short-circuit to
   ``(False, "low_overlap=…")`` without consulting the LLM judge.
2. **LLM near-exact judge** (accurate) — pairs at or above the threshold
   defer to :func:`idea_board.models._call_dedup_llm`, which is a thin
   wrapper over :func:`idea_board.dedup_llm.is_near_exact_duplicate`.

These tests pin both halves of the split:

* Below the threshold, the LLM seam must never be invoked (cost guard).
* At or above the threshold, the LLM seam must be invoked exactly once
  with the correct positional arguments
  ``(new_title, new_desc, existing.title, existing.description)``, and
  its return value must propagate to the caller unchanged.

Why this matters: the prefilter exists to avoid burning a Haiku
round-trip on obviously-different stories on every idea-generation
cycle. If a future refactor drops the prefilter, every dedup call starts
paying LLM cost — the ``call_count == 0`` assertion below is the
tripwire. Symmetrically, if the LLM call gets replaced with a stale
cache or a mis-wired argument list, the propagation tests catch it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from idea_board.models import (
    OVERLAP_PREFILTER_THRESHOLD,
    Idea,
    _is_duplicate,
)


# ---------------------------------------------------------------------------
# Sanity check on the threshold itself
# ---------------------------------------------------------------------------


def test_overlap_prefilter_threshold_is_twenty_percent():
    """The prefilter sits at 20% — this is the contract TK-764 ships.

    A future tweak to the threshold may be defensible, but it must be
    deliberate (and documented in the function docstring). Pinning the
    value here makes accidental changes show up as a test failure
    rather than as a silent shift in dedup cost/accuracy tradeoff.
    """
    assert OVERLAP_PREFILTER_THRESHOLD == 0.20


# ---------------------------------------------------------------------------
# Below-threshold: prefilter blocks, LLM never invoked
# ---------------------------------------------------------------------------


class TestPrefilterBlocksLowOverlap:
    """Pairs below the threshold short-circuit before reaching the LLM."""

    def test_disjoint_vocab_returns_false_without_llm_call(self):
        """Zero-overlap pair → ``(False, "low_overlap=…")``, no LLM call.

        The LLM mock returns a duplicate verdict so that *if* the
        prefilter is ever broken and the judge is consulted, the test
        fails twice: once on ``call_count`` and once on the verdict.
        Double tripwire.
        """
        existing = Idea(
            id="TK-100",
            title="Yoga class scheduling dashboard",
            description=(
                "Members book private studio sessions through a calendar "
                "widget connected to instructor availability."
            ),
        )

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(True, "forced_same"),
        ) as llm_mock:
            is_dup, reason = _is_duplicate(
                new_title="Submarine propulsion telemetry analyzer",
                new_desc=(
                    "Spectrogram viewer renders turbulent wake signatures "
                    "from deep sensor streams off a towed array."
                ),
                existing=existing,
            )

        assert is_dup is False
        assert reason.startswith("low_overlap="), (
            f"reason must encode the prefilter verdict; got {reason!r}"
        )
        assert llm_mock.call_count == 0, (
            "LLM judge must not be invoked when overlap is below the "
            "prefilter threshold — every unnecessary call burns Haiku "
            "tokens on stories that can be ruled out by a word-set "
            "intersection"
        )

    def test_empty_words_returns_false_without_llm_call(self):
        """Stopword-only inputs → ``(False, "empty_words")``, no LLM call.

        ``_meaningful_words`` strips stopwords and short tokens, so a
        title made entirely of stopwords/short tokens produces an empty
        set. The prefilter must catch this before reaching the LLM —
        sending an empty pair to the judge would waste a round-trip on
        a verdict the model can't meaningfully produce.
        """
        existing = Idea(
            id="TK-101",
            title="a b c",
            description="to of in on at",
        )

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(True, "forced_same"),
        ) as llm_mock:
            is_dup, reason = _is_duplicate(
                new_title="the and or",
                new_desc="is are was were be",
                existing=existing,
            )

        assert is_dup is False
        assert reason == "empty_words"
        assert llm_mock.call_count == 0


# ---------------------------------------------------------------------------
# At-or-above threshold: LLM invoked exactly once with the right arguments
# ---------------------------------------------------------------------------


class TestHighOverlapDelegatesToLLM:
    """Pairs at or above the threshold defer to the LLM judge."""

    def test_high_overlap_invokes_llm_exactly_once(self):
        """Heavy-overlap pair → LLM called once, verdict propagates.

        The mock returns a SAME verdict; the function must surface that
        verbatim — no transformation, no caching, no second call.
        """
        existing = Idea(
            id="TK-200",
            title="Cache Ollama responses to improve performance",
            description=(
                "WHAT: Cache responses keyed by prompt hash. "
                "WHY: Repeated prompts waste GPU time. "
                "HOW: Wrap the Ollama call site with a hash-keyed lookup."
            ),
        )

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(True, "near-exact match: same files, same outcome"),
        ) as llm_mock:
            is_dup, reason = _is_duplicate(
                new_title="Cache Ollama responses for performance gains",
                new_desc=(
                    "WHAT: Add a response cache for Ollama prompts. "
                    "WHY: Identical prompts repeat work. "
                    "HOW: Hash the prompt and short-circuit on cache hit."
                ),
                existing=existing,
            )

        assert is_dup is True
        assert reason == "near-exact match: same files, same outcome"
        assert llm_mock.call_count == 1, (
            "LLM judge must be invoked exactly once for a high-overlap "
            "pair — call_count==0 means the prefilter mis-fired and "
            "blocked a borderline case; call_count>1 means a redundant "
            "round-trip is being paid on every check"
        )

    def test_llm_called_with_correct_positional_arguments(self):
        """LLM receives ``(new_title, new_desc, existing.title, existing.desc)``.

        The argument order matters: the LLM prompt labels them
        "Story A (new)" and "Story B (existing)". A swapped pair would
        make the model judge the wrong direction (a confusing but
        non-symmetric framing in the prompt template). Pin the order
        explicitly so a future refactor can't silently flip them.
        """
        existing = Idea(
            id="TK-201",
            title="Existing story title with shared keywords cache ollama",
            description="Existing description sharing cache ollama tokens for overlap.",
        )

        new_title = "New story title sharing cache ollama keywords"
        new_desc = "New description with cache ollama tokens to clear the prefilter."

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(False, "different scope"),
        ) as llm_mock:
            _is_duplicate(new_title, new_desc, existing)

        assert llm_mock.call_count == 1
        call_args = llm_mock.call_args
        # The wrapper passes positional args; assert against args.
        assert call_args.args == (
            new_title,
            new_desc,
            existing.title,
            existing.description,
        ), (
            "LLM must receive (new_title, new_desc, existing.title, "
            f"existing.description) in that order; got {call_args.args!r}"
        )

    @pytest.mark.parametrize(
        "verdict, reason",
        [
            (True, "same files, same outcome"),
            (False, "follow-up coverage lift, not a rewording"),
            (False, "no_binary"),
            (False, "llm_timeout"),
            (False, "llm_exit_1"),
        ],
    )
    def test_llm_return_value_passes_through_unchanged(self, verdict, reason):
        """Whatever ``_call_dedup_llm`` returns, ``_is_duplicate`` returns.

        Covers both happy-path verdicts and the fall-open error codes
        (``no_binary``, ``llm_timeout``, ``llm_exit_*``). The wrapper
        must not invent its own error handling — that's the LLM
        judge's contract — and must not drop or rewrite the reason
        string, which downstream logs and the 409 response body quote
        verbatim.
        """
        existing = Idea(
            id="TK-202",
            title="Cache Ollama responses to improve performance",
            description="Hash-keyed cache in front of Ollama responses.",
        )

        with patch(
            "idea_board.dedup_llm.is_near_exact_duplicate",
            return_value=(verdict, reason),
        ):
            is_dup, out_reason = _is_duplicate(
                new_title="Cache Ollama responses for performance gains",
                new_desc="Add a response cache for Ollama prompts to recover headroom.",
                existing=existing,
            )

        assert is_dup is verdict
        assert out_reason == reason
