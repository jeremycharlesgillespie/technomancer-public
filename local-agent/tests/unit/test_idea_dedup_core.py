"""Tests for idea_board.dedup — embed+compare duplicate detection.

The embedding provider is mocked in every test — no real Ollama traffic —
so the whole file runs in well under the 10-second acceptance budget.

Coverage:
    - combine_text / _content_key text normalization
    - find_duplicate match (score >= threshold)
    - find_duplicate no-match (score below threshold, excluded states,
      empty pool, stale Done, missing embedding)
    - find_duplicate tie-break ordering (state priority, recency, id)
    - Default embed_fn falls through to agent.embeddings.embed_text so
      tests can patch at the import site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from idea_board.dedup import (
    DEFAULT_THRESHOLD,
    DedupMatch,
    _content_key,
    combine_text,
    find_duplicate,
)


# Override the global autouse fixtures from tests/conftest.py for this
# file only. They exist to guard AIM/Worker git-cleanup, force the
# LocalProvider, block Jira sync, and isolate the embedding store —
# none of which is exercised here. Their setup imports ``aim.manager``,
# ``board.factory``, ``idea_board.jira_sync``, and ``agent.embedding_store``,
# which collectively cost ~20s of first-time import and blow past the 10s
# acceptance budget for this file.
@pytest.fixture(autouse=True)
def _block_git_cleanup():
    yield


@pytest.fixture(autouse=True)
def _force_local_board_provider():
    yield


@pytest.fixture(autouse=True)
def _block_jira_sync():
    yield


@pytest.fixture(autouse=True)
def _isolate_embedding_store():
    yield


@dataclass
class FakeItem:
    """Minimal stand-in for Idea with the attributes find_duplicate reads."""

    id: str
    title: str
    description: str = ""
    state: str = "proposed"
    created: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now().isoformat(timespec="seconds")


def _embed_by_keyword(mapping: dict[str, list[float]], default: list[float] | None = None):
    """Build an embed function that returns a vector based on keyword match.

    Each key is a substring; the first matching substring wins. Unmatched
    texts receive ``default`` (an orthogonal vector by default) so the
    returned embeddings never accidentally score above threshold.
    """
    default_vec = default if default is not None else [0.0, 0.0, 0.0, 1.0]

    def embed(text: str) -> list[float]:
        for needle, vec in mapping.items():
            if needle in text:
                return vec
        return default_vec

    return embed


# ---------------------------------------------------------------------------
# combine_text
# ---------------------------------------------------------------------------


class TestCombineText:
    def test_joins_title_and_description_with_blank_line(self):
        result = combine_text("Add cache", "WHY: faster")
        assert "Add cache" in result
        assert "WHY: faster" in result
        assert "\n\n" in result

    def test_title_only(self):
        assert combine_text("Only title", "") == "Only title"

    def test_description_only(self):
        assert combine_text("", "Only desc") == "Only desc"

    def test_both_empty_returns_empty_string(self):
        assert combine_text("", "") == ""
        assert combine_text("   ", "   ") == ""

    def test_strips_surrounding_whitespace(self):
        result = combine_text("  Title  ", "  Desc  ")
        assert result.startswith("Title")
        assert result.endswith("Desc")


# ---------------------------------------------------------------------------
# _content_key
# ---------------------------------------------------------------------------


class TestContentKey:
    def test_lowercases(self):
        assert _content_key("HELLO", "World") == _content_key("hello", "world")

    def test_strips_punctuation(self):
        key = _content_key("Cache, Ollama!", "WHY? because.")
        assert "," not in key
        assert "!" not in key
        assert "?" not in key
        assert "." not in key

    def test_collapses_whitespace(self):
        key = _content_key("a   b\t\tc", "")
        assert key == "a b c"

    def test_deterministic(self):
        assert _content_key("X", "Y") == _content_key("X", "Y")

    def test_not_semantic_rephrasing_differs(self):
        # _content_key is exact-match canonicalization, not semantic.
        assert _content_key("Ollama cache", "") != _content_key("Cache Ollama", "")

    def test_empty_inputs_return_empty_string(self):
        assert _content_key("", "") == ""


# ---------------------------------------------------------------------------
# find_duplicate — match cases
# ---------------------------------------------------------------------------


class TestFindDuplicateMatch:
    def test_identical_vectors_match_with_score_1(self):
        existing = FakeItem(id="idea-001", title="Cache Ollama responses", description="WHY: speed")
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})

        match = find_duplicate(
            "Cache Ollama responses for perf",
            "WHY: better latency",
            [existing],
            embed_fn=embed,
        )

        assert match is not None
        assert isinstance(match, DedupMatch)
        assert match.item is existing
        assert match.score == pytest.approx(1.0)
        assert match.score >= DEFAULT_THRESHOLD

    def test_returns_dedupmatch_with_item_and_score(self):
        existing = FakeItem(id="idea-001", title="X", description="Y")
        embed = _embed_by_keyword({"X": [1.0, 0.0, 0.0]})

        match = find_duplicate("X similar", "Y similar", [existing], embed_fn=embed)

        assert match is not None
        assert match.item.id == "idea-001"
        assert 0.0 <= match.score <= 1.0

    def test_score_at_exact_threshold_matches(self):
        """Boundary: cosine exactly == threshold should count as a match."""
        existing = FakeItem(id="idea-001", title="A", description="")

        def embed(text: str) -> list[float]:
            # Two unit vectors with dot product == DEFAULT_THRESHOLD (0.88).
            # [1,0] vs [0.88, sqrt(1-0.88^2)] gives cosine == 0.88 exactly.
            import math

            if "query" in text:
                return [1.0, 0.0]
            return [DEFAULT_THRESHOLD, math.sqrt(1 - DEFAULT_THRESHOLD ** 2)]

        match = find_duplicate("query", "", [existing], embed_fn=embed)
        assert match is not None
        assert match.score == pytest.approx(DEFAULT_THRESHOLD)


# ---------------------------------------------------------------------------
# find_duplicate — no-match cases
# ---------------------------------------------------------------------------


class TestFindDuplicateNoMatch:
    def test_empty_items_returns_none(self):
        embed = _embed_by_keyword({})
        assert find_duplicate("anything", "desc", [], embed_fn=embed) is None

    def test_empty_query_text_returns_none(self):
        existing = FakeItem(id="idea-001", title="Cache Ollama")
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})
        assert find_duplicate("", "", [existing], embed_fn=embed) is None

    def test_orthogonal_vectors_do_not_match(self):
        existing = FakeItem(id="idea-001", title="Unrelated feature", description="XYZ")
        embed = _embed_by_keyword({
            "Unrelated": [1.0, 0.0, 0.0],
            "New thing": [0.0, 1.0, 0.0],
        })
        assert find_duplicate("New thing", "ABC", [existing], embed_fn=embed) is None

    def test_score_just_below_threshold_rejected(self):
        """Cosine ~0.87 must not match when threshold is 0.88."""
        existing = FakeItem(id="idea-001", title="A", description="")

        def embed(text: str) -> list[float]:
            import math

            # cosine = 0.87 — just below 0.88 threshold.
            if "query" in text:
                return [1.0, 0.0]
            return [0.87, math.sqrt(1 - 0.87 ** 2)]

        assert find_duplicate("query", "", [existing], embed_fn=embed) is None

    def test_ignores_vetoed(self):
        vetoed = FakeItem(id="idea-001", title="Cache Ollama", state="vetoed")
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})
        assert find_duplicate("Cache Ollama", "", [vetoed], embed_fn=embed) is None

    def test_ignores_failed(self):
        failed = FakeItem(id="idea-001", title="Cache Ollama", state="failed")
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})
        assert find_duplicate("Cache Ollama", "", [failed], embed_fn=embed) is None

    def test_ignores_old_done_items(self):
        old = FakeItem(
            id="idea-001",
            title="Cache Ollama",
            state="done",
            created=(datetime.now() - timedelta(days=365)).isoformat(timespec="seconds"),
        )
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})
        assert find_duplicate("Cache Ollama", "", [old], embed_fn=embed) is None

    def test_includes_recent_done_items(self):
        recent = FakeItem(
            id="idea-001",
            title="Cache Ollama",
            state="done",
            created=(datetime.now() - timedelta(days=3)).isoformat(timespec="seconds"),
        )
        embed = _embed_by_keyword({"Cache Ollama": [1.0, 0.0, 0.0]})
        match = find_duplicate("Cache Ollama", "", [recent], embed_fn=embed)
        assert match is not None
        assert match.item is recent

    def test_empty_query_embedding_returns_none(self):
        existing = FakeItem(id="idea-001", title="Cache Ollama")

        def embed(text: str) -> list[float]:
            return []

        assert find_duplicate("Cache Ollama", "", [existing], embed_fn=embed) is None


# ---------------------------------------------------------------------------
# find_duplicate — tie-break ordering
# ---------------------------------------------------------------------------


class TestFindDuplicateTieBreak:
    def _same_vec_embed(self) -> callable:
        """Every text embeds to the same vector → every pair has cosine 1.0."""
        return lambda text: [1.0, 0.0, 0.0]

    def test_in_progress_beats_to_do_on_tie(self):
        now_iso = datetime.now().isoformat(timespec="seconds")
        todo = FakeItem(id="idea-001", title="Cache", state="proposed", created=now_iso)
        in_progress = FakeItem(id="idea-002", title="Cache", state="executing", created=now_iso)

        match = find_duplicate("Cache", "", [todo, in_progress], embed_fn=self._same_vec_embed())
        assert match is not None
        assert match.item is in_progress

    def test_in_progress_beats_to_do_regardless_of_order(self):
        now_iso = datetime.now().isoformat(timespec="seconds")
        todo = FakeItem(id="idea-001", title="Cache", state="proposed", created=now_iso)
        in_progress = FakeItem(id="idea-002", title="Cache", state="executing", created=now_iso)

        # Reversed order — result must still be in_progress.
        match = find_duplicate("Cache", "", [in_progress, todo], embed_fn=self._same_vec_embed())
        assert match is not None
        assert match.item is in_progress

    def test_to_do_beats_done_on_tie(self):
        now_iso = datetime.now().isoformat(timespec="seconds")
        done = FakeItem(id="idea-001", title="Cache", state="done", created=now_iso)
        todo = FakeItem(id="idea-002", title="Cache", state="proposed", created=now_iso)

        match = find_duplicate("Cache", "", [done, todo], embed_fn=self._same_vec_embed())
        assert match is not None
        assert match.item is todo

    def test_newer_wins_at_same_priority(self):
        older = FakeItem(
            id="idea-001",
            title="Cache",
            state="proposed",
            created=(datetime.now() - timedelta(days=5)).isoformat(timespec="seconds"),
        )
        newer = FakeItem(
            id="idea-002",
            title="Cache",
            state="proposed",
            created=datetime.now().isoformat(timespec="seconds"),
        )

        match = find_duplicate("Cache", "", [older, newer], embed_fn=self._same_vec_embed())
        assert match is not None
        assert match.item is newer

    def test_jira_state_names_equivalent_to_internal_names(self):
        """Tie-break treats Jira column names the same as internal names."""
        now_iso = datetime.now().isoformat(timespec="seconds")
        jira_todo = FakeItem(id="idea-001", title="Cache", state="To Do", created=now_iso)
        jira_in_progress = FakeItem(id="idea-002", title="Cache", state="In Progress", created=now_iso)

        match = find_duplicate(
            "Cache",
            "",
            [jira_todo, jira_in_progress],
            embed_fn=self._same_vec_embed(),
        )
        assert match is not None
        assert match.item is jira_in_progress

    def test_score_difference_beats_state_priority(self):
        """A higher cosine score on a Done item beats a lower score on In Progress."""
        now_iso = datetime.now().isoformat(timespec="seconds")
        in_progress_low = FakeItem(
            id="idea-001",
            title="Only tangential",
            state="executing",
            created=now_iso,
        )
        done_high = FakeItem(
            id="idea-002",
            title="Cache Ollama responses",
            state="done",
            created=now_iso,
        )

        # done_high embeds identically to the query → cosine 1.0.
        # in_progress_low embeds to a lower-similarity vector (~0.9).
        embed = _embed_by_keyword({
            "Cache Ollama responses": [1.0, 0.0, 0.0],
            "Only tangential": [0.9, 0.1, 0.05],
        })

        match = find_duplicate("Cache Ollama responses", "", [in_progress_low, done_high], embed_fn=embed)
        assert match is not None
        assert match.item is done_high

    def test_deterministic_with_identical_priority_and_created(self):
        """When priority, score, and created are all equal, larger id wins."""
        now_iso = datetime.now().isoformat(timespec="seconds")
        a = FakeItem(id="idea-001", title="Cache", state="proposed", created=now_iso)
        b = FakeItem(id="idea-002", title="Cache", state="proposed", created=now_iso)

        # Two passes with reversed input — must pick the same winner both times.
        embed = self._same_vec_embed()
        m1 = find_duplicate("Cache", "", [a, b], embed_fn=embed)
        m2 = find_duplicate("Cache", "", [b, a], embed_fn=embed)
        assert m1 is not None and m2 is not None
        assert m1.item.id == m2.item.id
        assert m1.item.id == "idea-002"  # larger id wins the deterministic fallback


# ---------------------------------------------------------------------------
# Default embed_fn fallback
# ---------------------------------------------------------------------------


class TestDefaultEmbedFn:
    def test_uses_agent_embed_text_when_embed_fn_none(self):
        """When embed_fn is None, find_duplicate lazy-imports agent.embeddings.embed_text.

        The lazy import keeps pytest startup fast — see the module docstring.
        Patching ``agent.embeddings.embed_text`` intercepts the call because
        the in-function ``from agent.embeddings import embed_text`` reads the
        current module attribute on each call.
        """
        import sys
        import types

        # Stand up a lightweight stub for agent.embeddings so the lazy
        # import inside find_duplicate resolves without hitting the real
        # (slow-to-import) agent package.
        stub = types.ModuleType("agent.embeddings")
        stub.embed_text = lambda text: [1.0, 0.0, 0.0]  # type: ignore[attr-defined]
        agent_stub = types.ModuleType("agent")
        agent_stub.embeddings = stub  # type: ignore[attr-defined]

        existing = FakeItem(id="idea-001", title="Cache Ollama", state="proposed")

        with patch.dict(sys.modules, {"agent": agent_stub, "agent.embeddings": stub}):
            with patch.object(stub, "embed_text", wraps=stub.embed_text) as m:
                match = find_duplicate("Cache Ollama similar", "", [existing])

        assert match is not None
        assert m.called


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


def test_find_duplicate_is_fast_with_fake_embedder():
    """Sanity: 50 candidates with a fake embedder should finish in << 1s.

    The acceptance criterion is "pytest tests/unit/test_idea_dedup_core.py
    completes in under 10 seconds" — this guards against an expensive
    regression sneaking into the comparison loop.
    """
    items = [
        FakeItem(id=f"idea-{i:03d}", title=f"Item {i}", state="proposed")
        for i in range(50)
    ]
    embed = _embed_by_keyword(
        {f"Item {i}": [1.0, float(i) * 0.0001, 0.0] for i in range(50)}
    )

    start = time.monotonic()
    find_duplicate("Item 0", "", items, embed_fn=embed)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
