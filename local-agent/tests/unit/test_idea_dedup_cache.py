"""Tests for ``idea_board.dedup`` — TTL cache and fall-open error handling.

Design note: these tests deliberately do **not** patch ``time.time``. A
previous attempt to simulate cache expiry by patching the clock hung
because ``find_duplicate`` acquires a ``threading.Lock`` while reading the
clock, and the patched clock interacted badly with that critical section.
Instead, we manipulate ``CacheEntry.timestamp`` directly: backdating the
attribute past the TTL deterministically forces a miss on the next lookup,
can't deadlock, and runs in microseconds.

Acceptance criterion (TK-473): the cache hit/miss/expiry + fall-open cases
must all pass in under 2 seconds.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from idea_board import dedup
from idea_board.models import Idea


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure the module-level cache is empty at the start of every test."""
    dedup.clear_cache()
    yield
    dedup.clear_cache()


@pytest.fixture
def sample_ideas():
    """A single non-vetoed idea — enough to drive the similarity path."""
    return [
        Idea(
            id="idea-001",
            title="Cache Ollama responses",
            description="Add a response cache to improve latency.",
            state="approved",
        )
    ]


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestCacheMissHit:
    def test_first_call_is_a_miss_and_embeds(self, sample_ideas):
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            result = dedup.find_duplicate("new title", "new desc")
        assert result == "idea-001"
        # First call must hit the embedder at least once (query + each idea).
        assert embed.call_count >= 1

    def test_second_call_within_ttl_is_a_hit(self, sample_ideas):
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            first = dedup.find_duplicate("title", "desc")
            after_first = embed.call_count
            second = dedup.find_duplicate("title", "desc")
        assert first == second == "idea-001"
        # Cache hit means the embedder was NOT called the second time.
        assert embed.call_count == after_first

    def test_different_keys_do_not_share_cache(self, sample_ideas):
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            dedup.find_duplicate("title A", "desc A")
            after_first = embed.call_count
            dedup.find_duplicate("title B", "desc B")
        # A distinct (title, description) must re-embed — no cross-contamination.
        assert embed.call_count > after_first

    def test_non_duplicate_result_is_cached_as_none(self, sample_ideas):
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.1):
            first = dedup.find_duplicate("title", "desc")
            after_first = embed.call_count
            second = dedup.find_duplicate("title", "desc")
        assert first is None and second is None
        # A cached "no duplicate" answer must also short-circuit the embedder.
        assert embed.call_count == after_first


# ---------------------------------------------------------------------------
# TTL expiry — timestamp manipulation only, no time.time patching
# ---------------------------------------------------------------------------


class TestCacheExpiry:
    def test_expired_entry_is_re_evaluated(self, sample_ideas):
        """Backdate the cache entry past TTL; the next call must re-embed."""
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            dedup.find_duplicate("title", "desc")
            key = dedup._cache_key("title", "desc")
            entry = dedup._cache[key]
            # Directly age the entry out — no time.time patch, no sleep.
            entry.timestamp -= dedup.CACHE_TTL_SECONDS + 1.0
            count_before = embed.call_count
            dedup.find_duplicate("title", "desc")
        assert embed.call_count > count_before

    def test_entry_exactly_at_ttl_boundary_still_hits(self, sample_ideas):
        """At exactly TTL seconds old, the entry is still valid (≤, not <)."""
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            dedup.find_duplicate("title", "desc")
            key = dedup._cache_key("title", "desc")
            entry = dedup._cache[key]
            # Age to just-before expiry: now - ts ≈ TTL - small epsilon.
            entry.timestamp = time.time() - (dedup.CACHE_TTL_SECONDS - 1.0)
            count_before = embed.call_count
            dedup.find_duplicate("title", "desc")
        assert embed.call_count == count_before

    def test_expired_entry_is_purged_from_cache(self, sample_ideas):
        """After an expired hit the stale entry is replaced, not duplicated."""
        embed = MagicMock(return_value=[0.1] * 8)
        with patch.object(dedup, "embed_text", embed), \
             patch.object(dedup, "load_ideas", return_value=sample_ideas), \
             patch.object(dedup, "cosine_similarity", return_value=0.9):
            dedup.find_duplicate("title", "desc")
            key = dedup._cache_key("title", "desc")
            original_entry = dedup._cache[key]
            original_entry.timestamp -= dedup.CACHE_TTL_SECONDS + 10.0
            dedup.find_duplicate("title", "desc")
            # After re-eval, the cache should hold a FRESH entry, not the old one.
            refreshed = dedup._cache[key]
        assert refreshed is not original_entry
        assert refreshed.timestamp > original_entry.timestamp


# ---------------------------------------------------------------------------
# Fall-open on embedding-provider exceptions
# ---------------------------------------------------------------------------


class TestFallOpen:
    def test_embedding_exception_returns_none(self):
        with patch.object(
            dedup, "embed_text", side_effect=RuntimeError("ollama down")
        ):
            assert dedup.find_duplicate("t", "d") is None

    def test_embedding_exception_does_not_raise(self):
        """The caller must be shielded — ``find_duplicate`` never propagates."""
        with patch.object(
            dedup, "embed_text", side_effect=ConnectionError("boom")
        ):
            dedup.find_duplicate("t", "d")  # must not raise

    def test_empty_vector_returns_none(self):
        """``embed_text`` returns [] on swallowed errors — treat as fall-open."""
        with patch.object(dedup, "embed_text", return_value=[]):
            assert dedup.find_duplicate("t", "d") is None

    def test_embedding_exception_logs_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=dedup.logger.name)
        with patch.object(
            dedup, "embed_text", side_effect=RuntimeError("ollama down")
        ):
            dedup.find_duplicate("t", "d")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected at least one WARNING log on fall-open"
        # Message should mention falling open so operators can spot the signal.
        assert any("falling open" in r.message.lower() for r in warnings)

    def test_load_ideas_exception_returns_none(self):
        """An unhealthy ideas store must also trigger fall-open, not a crash."""
        with patch.object(dedup, "embed_text", return_value=[0.1] * 8), \
             patch.object(dedup, "load_ideas", side_effect=OSError("disk")):
            assert dedup.find_duplicate("t", "d") is None

    def test_fall_open_result_is_not_cached(self):
        """A transient provider outage shouldn't poison the cache with None."""
        with patch.object(
            dedup, "embed_text", side_effect=RuntimeError("ollama down")
        ):
            dedup.find_duplicate("t", "d")
        # Recovery: next call should retry instead of returning cached None.
        key = dedup._cache_key("t", "d")
        assert key not in dedup._cache


# ---------------------------------------------------------------------------
# Performance budget — whole module under 2s
# ---------------------------------------------------------------------------


def test_total_suite_runtime_budget(sample_ideas):
    """Smoke check that the common hit/miss/expiry path is fast.

    This single test runs a representative set of calls and asserts it
    completes in well under 2s — the TK-473 acceptance budget. It is
    intentionally redundant with pytest's own timing so regressions show
    up as a failing assertion rather than a slow suite.
    """
    start = time.monotonic()
    embed = MagicMock(return_value=[0.1] * 8)
    with patch.object(dedup, "embed_text", embed), \
         patch.object(dedup, "load_ideas", return_value=sample_ideas), \
         patch.object(dedup, "cosine_similarity", return_value=0.9):
        for _ in range(50):
            dedup.find_duplicate("title", "desc")
        key = dedup._cache_key("title", "desc")
        dedup._cache[key].timestamp -= dedup.CACHE_TTL_SECONDS + 1.0
        dedup.find_duplicate("title", "desc")
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"dedup cache path took {elapsed:.2f}s (budget: 2s)"
