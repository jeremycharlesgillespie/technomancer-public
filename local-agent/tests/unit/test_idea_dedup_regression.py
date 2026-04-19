"""Regression test for TK-441 — Ollama-cluster duplicate scenario.

TK-441 traced a backlog symptom where five near-duplicate
"Ollama performance" stories had slipped past dedup and cluttered the
board. The fix landed in ``idea_board.models._is_duplicate`` (stem-based
word overlap) and ``/api/jira/create`` now surfaces a duplicate hit as
HTTP 409 instead of silently returning 201 with the pre-existing key.

This module seeds three near-duplicate Ollama-performance titles through
the ``/api/jira/create`` endpoint — the same input shape that produced
the 5-duplicate backlog symptom — and asserts:

    1. The first call returns 201 and mints a new idea.
    2. The next two calls return 409 with the **first** issue's key,
       confirming that the provider-level dedup fired and that the
       endpoint translated it into a duplicate response (not a 201).

The embedding function is stubbed to realistic high-similarity vectors
so that any future switch to embedding-based dedup still sees the
cluster as duplicates. With the current word-overlap dedup the stub is
inert; if/when the dedup path consults embeddings, the stub keeps the
regression signal intact.

Acceptance (per TK-476): this test must pass in under 5s.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from idea_board.models import Idea
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def in_memory_board():
    """In-memory idea storage that replaces the JSON-backed load/save pair.

    Yields the shared list so a test can inspect final state if needed.
    Mocks the Obsidian and Jira side effects so the LocalProvider runs
    pure in-memory — no disk I/O, no background threads.
    """
    storage: list[Idea] = []

    def fake_load_ideas() -> list[Idea]:
        # Return a shallow copy so callers that mutate the list don't
        # corrupt the source of truth mid-call — matches the JSON-reload
        # semantics of the real loader.
        return list(storage)

    def fake_save_ideas(ideas: list[Idea]) -> None:
        storage[:] = list(ideas)

    with patch("idea_board.models.load_ideas", side_effect=fake_load_ideas), \
         patch("idea_board.models.save_ideas", side_effect=fake_save_ideas), \
         patch("idea_board.models.sync_to_obsidian"), \
         patch("idea_board.models._jira_sync_background"):
        yield storage


@pytest.fixture
def stub_embeddings():
    """Stub the Ollama embedding helpers to realistic high-similarity vectors.

    A tightly-clustered group of Ollama-performance ideas embeds into
    near-parallel 768-dim vectors in production; we mimic that with a
    fixed high-magnitude vector so cosine similarity between any pair
    is ~1.0. Any dedup path that consults embeddings will see the
    cluster and dedup; paths that ignore embeddings are unaffected.
    """
    high_sim_vec = [0.95] * 768

    with patch("agent.embeddings.embed_text", return_value=high_sim_vec), \
         patch("agent.embeddings.embed_texts",
               side_effect=lambda texts: [high_sim_vec for _ in texts]):
        yield high_sim_vec


# The three near-duplicate titles that reproduce the TK-441 cluster.
# Stem-based word overlap (≥0.5 on title, ≥0.4 on title+description)
# flags each pair as a duplicate. Keeping the trio explicit here makes
# the regression intent obvious at the callsite.
OLLAMA_CLUSTER = (
    {
        "title": "Cache Ollama responses to improve performance",
        "description": (
            "WHY: Ollama inference repeats work for identical prompts. "
            "HOW: Cache responses keyed by prompt hash to skip recompute."
        ),
    },
    {
        "title": "Ollama response cache for better performance",
        "description": (
            "WHY: Response latency hurts UX. "
            "HOW: Add a cache layer in front of Ollama response calls."
        ),
    },
    {
        "title": "Add response cache for Ollama performance gains",
        "description": (
            "WHY: Duplicate Ollama calls waste GPU time. "
            "HOW: Cache Ollama responses to recover performance headroom."
        ),
    },
)


def _post_create(client, payload: dict):
    return client.post(
        "/api/jira/create",
        data=json.dumps(payload),
        content_type="application/json",
    )


def test_ollama_cluster_dedup_returns_409_on_duplicates(
    client, in_memory_board, stub_embeddings
):
    """Three near-duplicate Ollama-performance posts → 201, 409, 409.

    Reproduces the TK-441 5-duplicate backlog symptom on a smaller
    scale (3 titles is enough to prove the dedup gate holds across
    multiple repeat attempts). The second and third responses must
    carry the first issue's key so the caller can link to the canonical
    entry instead of writing a new one.
    """
    start = time.monotonic()

    first = _post_create(client, OLLAMA_CLUSTER[0])
    assert first.status_code == 201, first.get_json()
    first_key = first.get_json()["key"]
    assert first_key, "First call must mint a non-empty idea key"

    for dup_payload in OLLAMA_CLUSTER[1:]:
        resp = _post_create(client, dup_payload)
        assert resp.status_code == 409, (
            f"Expected 409 for duplicate '{dup_payload['title']}', "
            f"got {resp.status_code}: {resp.get_json()}"
        )
        body = resp.get_json()
        assert body["key"] == first_key, (
            f"Duplicate response should point at the first issue's key "
            f"({first_key}); got {body['key']}"
        )

    # Only one idea should have been persisted — the other two dedup'd
    # back to it. Guards against a regression where dedup passes through
    # the endpoint but the underlying store grows anyway.
    assert len(in_memory_board) == 1
    assert in_memory_board[0].id == first_key

    # Acceptance criterion: the full trio must complete in under 5s.
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"Regression test took {elapsed:.2f}s (budget: 5s)"


# ---------------------------------------------------------------------------
# TK-742 regression — dup check must pass legitimate follow-up stories
# ---------------------------------------------------------------------------
#
# The TK-742 incident was the mirror-image of TK-441: where TK-441 caught
# too few restated dups, TK-742 killed too many legitimate follow-ups.
# TK-571 ("Unit tests for capability_request.py (50% -> 75%)" — a concrete
# coverage-lift story) was auto-vetoed in queue review because its title
# overlapped with TK-321 ("[idea-197] Add Unit Tests for capability_request
# Core Logic" — an older abstract Done story). Both stories share
# {unit, tests, capability_request} in their stem sets, but their actual
# scope diverged: TK-571 specified a coverage target and specific test
# targets; TK-321 was generic.
#
# The queue-review auto-veto was replaced with an advisory comment, but we
# also want ``_is_duplicate`` itself to return False for this pair — if a
# future change to the dedup thresholds makes them too aggressive, this
# test fails before it reaches production.


def test_tk571_is_not_dup_of_tk321():
    """_is_duplicate(TK-571, TK-321) must be False.

    Title overlap sits at the ~0.5 boundary (shared stems {unit, tests,
    capab}) — exactly the point where the old auto-veto fired — and the
    bodies diverge on scope. The function must not flag this pair.
    """
    from idea_board.models import Idea, _is_duplicate

    tk321 = Idea(
        id="TK-321",
        title="[idea-197] Add Unit Tests for capability_request.py Core Logic",
        description=(
            "WHAT: Add unit tests covering the core capability evaluation "
            "logic in capability_request.py. "
            "WHY: No unit coverage today. "
            "HOW: Write tests against the Claude API evaluation path."
        ),
        state="done",
    )

    assert not _is_duplicate(
        new_title="Unit tests for capability_request.py (50% -> 75%)",
        new_desc=(
            "WHAT: Raise line coverage in capability_request.py from 50 "
            "percent to 75 percent. "
            "WHY: Gaps remain in the retry, rate-limit, and circuit-breaker "
            "branches. "
            "HOW: Parametrize failure modes and assert recovery paths."
        ),
        existing=tk321,
    )


# ---------------------------------------------------------------------------
# TK-750 — LLM dedup verdict (SAME) flagged as duplicate
# ---------------------------------------------------------------------------
#
# The dedup seam at ``idea_board.models._is_duplicate`` is the one place
# every caller (``add_idea``, queue review, ``/api/jira/create``) consults
# to decide whether two stories collide. Today the implementation is a
# word-overlap heuristic, but the seam is intentionally swappable for an
# LLM near-exact judge — the ``mock_dedup_llm`` fixture exists for that
# transition.
#
# This test pins down the SAME-verdict half of the contract: when the
# (mocked) LLM returns True for a near-identical pair, the dedup gate
# must fire. If a future refactor drops the call into ``_is_duplicate``
# entirely (e.g. caches the result and skips the judge on a subsequent
# call), ``call_count == 0`` will catch the regression here before it
# silently neuters the dedup signal in production.


def test_is_duplicate_true_duplicate_llm_verdict(mock_dedup_llm):
    """LLM verdict SAME → ``_is_duplicate`` returns True and judge was called.

    Fed two stories with heavy stem overlap (the kind of restated pair the
    word-overlap heuristic catches today and an LLM judge would catch
    tomorrow), the dedup seam must (a) return True and (b) actually
    invoke the judge — not short-circuit on a cached or stale verdict.
    """
    # Re-import through the module so we hit the monkeypatched attribute,
    # not the original function bound at file-import time above.
    from idea_board import models

    mock_dedup_llm.return_value = True

    existing = Idea(
        id="TK-900",
        title="Cache Ollama responses to improve performance",
        description=(
            "WHAT: Cache responses keyed by prompt hash. "
            "WHY: Repeated prompts waste GPU time. "
            "HOW: Wrap the Ollama call site with a hash-keyed lookup."
        ),
        state="proposed",
    )

    result = models._is_duplicate(
        new_title="Cache Ollama responses for performance gains",
        new_desc=(
            "WHAT: Add a response cache for Ollama prompts. "
            "WHY: Identical prompts repeat work. "
            "HOW: Hash the prompt and short-circuit on cache hit."
        ),
        existing=existing,
    )

    assert result is True, "LLM SAME verdict must produce a duplicate flag"
    assert mock_dedup_llm.call_count >= 1, (
        "Dedup judge must be invoked — a 0 call count means the seam "
        "short-circuited and the LLM verdict was never consulted"
    )


# ---------------------------------------------------------------------------
# TK-751 — LLM dedup verdict (DIFFERENT) for the TK-571 vs TK-321 headline pair
# ---------------------------------------------------------------------------
#
# TK-743 introduced the LLM-backed dedup judge specifically because the old
# word-overlap heuristic kept killing concrete follow-up stories as
# duplicates of older abstract ones. The motivating headline case: TK-571
# ("Unit tests for capability_request.py (50% -> 75%)" — a concrete
# coverage-lift story with specific branch targets) getting auto-vetoed
# against TK-321 ("[idea-197] Add Unit Tests for capability_request.py Core
# Logic" — an older, generic, already-Done story). A human reader sees two
# different stories; the stem-overlap heuristic sees {unit, tests,
# capab(ility_request)} and fires.
#
# This test pins down the DIFFERENT-verdict half of the contract at the same
# seam as ``test_is_duplicate_true_duplicate_llm_verdict`` above. When the
# (mocked) LLM judge returns False for this exact pair, ``_is_duplicate``
# must propagate that verdict — and must have actually invoked the judge,
# not short-circuited on a title-overlap pre-check that would mask the
# regression in production.


def test_is_duplicate_tk571_vs_tk321_headline(mock_dedup_llm):
    """LLM verdict DIFFERENT on the TK-571/TK-321 pair → returns False.

    The headline case for TK-743. If a future refactor reintroduces a
    pre-LLM overlap gate that rejects this pair before reaching the
    judge, ``call_count == 0`` catches it; if the gate flips polarity
    and returns True for legitimate follow-ups, the ``is False`` check
    catches it.
    """
    from idea_board import models

    mock_dedup_llm.return_value = False

    tk321 = Idea(
        id="TK-321",
        title="[idea-197] Add Unit Tests for capability_request.py Core Logic",
        description=(
            "WHAT: Add unit tests covering the core capability evaluation "
            "logic in capability_request.py. "
            "WHY: No unit coverage today. "
            "HOW: Write tests against the Claude API evaluation path."
        ),
        state="done",
    )

    result = models._is_duplicate(
        new_title="Unit tests for capability_request.py (50% -> 75%)",
        new_desc=(
            "WHAT: Raise line coverage in capability_request.py from 50 "
            "percent to 75 percent. "
            "WHY: Gaps remain in the retry, rate-limit, and circuit-breaker "
            "branches. "
            "HOW: Parametrize failure modes and assert recovery paths."
        ),
        existing=tk321,
    )

    assert result is False, (
        "LLM DIFFERENT verdict on the TK-571/TK-321 headline pair must "
        "produce a non-duplicate flag — this is the exact regression "
        "TK-743 was built to prevent"
    )
    assert mock_dedup_llm.call_count >= 1, (
        "Dedup judge must be invoked — a 0 call count means the seam "
        "short-circuited on a pre-LLM overlap gate and the headline "
        "regression case silently bypassed the judge"
    )
