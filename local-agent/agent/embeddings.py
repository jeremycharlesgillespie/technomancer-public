"""
Embeddings — Ollama-powered semantic similarity using nomic-embed-text.

Provides:
    - embed_texts(): Batch embed text strings via Ollama
    - cosine_similarity(): Compare two embedding vectors
    - SemanticCache: In-memory cache of text embeddings for fast lookup

Uses the local Ollama server with nomic-embed-text (274MB, 768-dim vectors).
Batch embedding is fast (~8ms/text) so embedding 100+ conversations is cheap.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

log = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
SIMILARITY_THRESHOLD = 0.50  # Minimum cosine similarity to consider "related"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Ollama's nomic-embed-text model.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (768-dim floats), one per input text.
        Returns empty list on failure.
    """
    if not texts:
        return []

    try:
        import ollama
        result = ollama.embed(model=EMBED_MODEL, input=texts)
        return result["embeddings"]
    except Exception as e:
        log.warning(f"[Embeddings] Failed to embed {len(texts)} texts: {e}")
        return []


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns empty list on failure."""
    result = embed_texts([text])
    return result[0] if result else []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0 (higher = more similar).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """In-memory cache of text embeddings for fast semantic search.

    Stores (text, embedding) pairs. When searching for similar texts,
    computes cosine similarity against all cached embeddings — O(N) but
    fast because it's pure math (no LLM calls).

    Thread-safe via a lock.
    """

    def __init__(self, max_size: int = 1500) -> None:
        self._entries: list[tuple[str, Any, list[float]]] = []  # (text, metadata, embedding)
        self._max_size = max_size
        self._lock = threading.Lock()

    def add(self, text: str, metadata: Any, embedding: list[float]) -> None:
        """Add a text + its embedding to the cache."""
        with self._lock:
            self._entries.append((text, metadata, embedding))
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]

    def add_batch(self, items: list[tuple[str, Any]], embeddings: list[list[float]]) -> None:
        """Add multiple (text, metadata) pairs with their embeddings."""
        with self._lock:
            for (text, meta), emb in zip(items, embeddings):
                self._entries.append((text, meta, emb))
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]

    def search(
        self, query_embedding: list[float], top_k: int = 3, threshold: float = SIMILARITY_THRESHOLD
    ) -> list[tuple[float, str, Any]]:
        """Find the most similar cached texts to a query embedding.

        Args:
            query_embedding: The embedding vector to search for.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity to include.

        Returns:
            List of (similarity_score, text, metadata) tuples, sorted by score descending.
        """
        with self._lock:
            entries = list(self._entries)

        scored = []
        for text, meta, emb in entries:
            sim = cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                scored.append((sim, text, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)


# Global conversation embedding cache
_conversation_cache = SemanticCache(max_size=1500)


def get_conversation_cache() -> SemanticCache:
    """Get the global conversation embedding cache."""
    return _conversation_cache
