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
