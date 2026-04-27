"""Idea Dedup — embedding-backed duplicate detection with TTL cache and fall-open safety.

``find_duplicate(title, description)`` consults the embedding provider to find the
existing idea most similar to the payload and returns its id (or ``None`` if
nothing is similar enough). Results are cached for ``CACHE_TTL_SECONDS`` keyed
by ``hash((title, description))`` so repeated lookups during backlog churn —
e.g. the idea generator re-checking the same proposal across a queue-review
sweep — don't re-embed the same text.

Fall-open policy: if the embedding provider raises (Ollama down, network
flake, OOM), ``find_duplicate`` logs a warning at WARNING level and returns
``None`` so idea creation is *never* blocked by a flaky embedding backend.
A transient embedding outage should not manifest as a paused idea queue.

Test design note: the cache stores entries as ``CacheEntry`` dataclasses with
a mutable ``timestamp`` attribute. Expiry tests should backdate
``entry.timestamp`` directly rather than patching ``time.time`` — patching
the clock module-wide in previous attempts produced hangs when threading
locks inside the cache interacted with the patched value.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from agent.embeddings import cosine_similarity, embed_text

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS: float = 300.0
SIMILARITY_THRESHOLD: float = 0.85


@dataclass
class CacheEntry:
    """A cached ``find_duplicate`` result.

    ``timestamp`` is intentionally a plain mutable attribute so tests can
    simulate expiry by backdating it directly, avoiding the need to patch
    ``time.time``.
    """

    result: str | None
    timestamp: float


_cache: dict[int, CacheEntry] = {}
_cache_lock = threading.Lock()


def _cache_key(title: str, description: str) -> int:
    return hash((title, description))


def clear_cache() -> None:
    """Drop all cached entries. Primarily useful for test setup/teardown."""
    with _cache_lock:
        _cache.clear()


def find_duplicate(title: str, description: str) -> str | None:
    """Return the id of the existing idea most similar to ``(title, description)``.

    Returns ``None`` when:
    - No existing idea scores above :data:`SIMILARITY_THRESHOLD`.
    - The embedding provider raises (fall-open: warn and return ``None``).
    - ``load_ideas`` raises (same fall-open policy).

    Results are cached for :data:`CACHE_TTL_SECONDS` keyed by
    ``hash((title, description))``. A cache entry is treated as expired once
    ``time.time() - entry.timestamp > CACHE_TTL_SECONDS``.
    """
    key = _cache_key(title, description)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            if time.time() - entry.timestamp <= CACHE_TTL_SECONDS:
                return entry.result
            del _cache[key]

    try:
        query_vec = embed_text(f"{title}\n{description}")
    except Exception as exc:
        logger.warning("[dedup] embedding provider raised; falling open: %s", exc)
        return None

    if not query_vec:
        # embed_text swallows some errors and returns []; treat as fall-open
        # but don't cache the null result — the provider may recover shortly.
        return None

    best_id: str | None = None
    best_score: float = 0.0
    try:
        # PR 6: ideas now live in Jira, not the local JSON store. Read
        # the active set through the provider so the existing fall-open
        # contract (returns ``None`` on any backend failure) still
        # holds when Jira is unreachable.
        from board import get_provider
        ideas = get_provider().load_all()
        for idea in ideas:
            if idea.state == "vetoed":
                continue
            idea_vec = embed_text(f"{idea.title}\n{idea.description}")
            if not idea_vec:
                continue
            score = cosine_similarity(query_vec, idea_vec)
            if score > best_score:
                best_score = score
                best_id = idea.id
    except Exception as exc:
        logger.warning("[dedup] embedding provider raised; falling open: %s", exc)
        return None

    match = best_id if best_score >= SIMILARITY_THRESHOLD else None
    with _cache_lock:
        _cache[key] = CacheEntry(result=match, timestamp=time.time())
    return match
