"""Tests for agent/embeddings.py — cosine similarity, semantic cache."""

from unittest.mock import patch

import pytest

from agent.embeddings import (
    SIMILARITY_THRESHOLD,
    SemanticCache,
    cosine_similarity,
    embed_text,
    embed_texts,
)


class TestCosineSimilarity:
    """Pure math — no mocks needed."""

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = cosine_similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


class TestEmbedTexts:
    """Mocks the Ollama embed call."""

    def test_empty_list_returns_empty(self):
        assert embed_texts([]) == []

    def test_returns_embeddings(self):
        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("ollama.embed", return_value={"embeddings": fake_embeddings}):
            result = embed_texts(["hello", "world"])
            assert result == fake_embeddings

    def test_ollama_failure_returns_empty(self):
        with patch("ollama.embed", side_effect=Exception("connection refused")):
            result = embed_texts(["test"])
            assert result == []


class TestEmbedText:
    """Single text wrapper."""

    def test_returns_single_embedding(self):
        fake = [[0.1, 0.2, 0.3]]
        with patch("agent.embeddings.embed_texts", return_value=fake):
            result = embed_text("hello")
            assert result == [0.1, 0.2, 0.3]

    def test_failure_returns_empty(self):
        with patch("agent.embeddings.embed_texts", return_value=[]):
            result = embed_text("hello")
            assert result == []


class TestSemanticCache:
    """In-memory cache — no mocks needed."""

    def test_empty_cache(self):
        cache = SemanticCache()
        assert cache.size == 0

    def test_add_and_size(self):
        cache = SemanticCache()
        cache.add("hello", {"id": 1}, [1.0, 0.0, 0.0])
        assert cache.size == 1

    def test_add_batch(self):
        cache = SemanticCache()
        items = [("hello", {"id": 1}), ("world", {"id": 2})]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        cache.add_batch(items, embeddings)
        assert cache.size == 2

    def test_search_finds_similar(self):
        cache = SemanticCache()
        cache.add("hello", {"id": 1}, [1.0, 0.0, 0.0])
        cache.add("world", {"id": 2}, [0.0, 1.0, 0.0])
        cache.add("hi there", {"id": 3}, [0.9, 0.1, 0.0])

        results = cache.search([1.0, 0.0, 0.0], top_k=2, threshold=0.5)
        assert len(results) >= 1
        # First result should be most similar
        assert results[0][2]["id"] in (1, 3)

    def test_search_respects_threshold(self):
        cache = SemanticCache()
        cache.add("hello", {"id": 1}, [1.0, 0.0, 0.0])
        cache.add("world", {"id": 2}, [0.0, 1.0, 0.0])

        # Search with very high threshold — only exact match
        results = cache.search([1.0, 0.0, 0.0], top_k=10, threshold=0.99)
        assert len(results) == 1

    def test_search_empty_cache(self):
        cache = SemanticCache()
        results = cache.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_max_size_eviction(self):
        cache = SemanticCache(max_size=3)
        for i in range(5):
            cache.add(f"text-{i}", {"id": i}, [float(i), 0.0])
        assert cache.size == 3


class TestSimilarityThreshold:
    def test_threshold_is_reasonable(self):
        assert 0.0 < SIMILARITY_THRESHOLD < 1.0
