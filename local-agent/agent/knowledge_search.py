"""
Knowledge Search — Unified semantic search across all knowledge sources.

Embeds and indexes:
  - Obsidian vault reference articles (Permanent/References/*.md)
  - Facts DB entries (definitions, geography, timezones, conversions)
  - Permanent memories (Permanent/memories.md)
  - Recent conversations (from MemorySystem buffer)

Embeddings are persisted to SQLite (via embedding_store) so that subsequent
startups load from cache instead of re-embedding via Ollama.

One ``search_knowledge`` query returns the most relevant results regardless
of where they are stored.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from . import embedding_store
from .config import settings
from .embeddings import SemanticCache, embed_text, embed_texts

log = logging.getLogger(__name__)

# Batch size for embedding calls (balances throughput vs memory)
_EMBED_BATCH_SIZE = 64


class KnowledgeIndex:
    """Unified semantic index across all knowledge sources.

    On startup, call ``build()`` to embed vault articles, facts, and memories.
    Then use ``search()`` for semantic retrieval across everything.
    """

    def __init__(self, vault_path: Path | None = None, max_size: int = 5000) -> None:
        self._cache = SemanticCache(max_size=max_size)
        self._vault_path = vault_path or Path(settings.vault_path)
        self._stats: dict[str, int] = {}
        self._built = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self) -> dict[str, int]:
        """Index all knowledge sources. Returns counts per source."""
        t0 = time.monotonic()
        self._stats = {}

        self._stats["vault_articles"] = self._index_vault_articles()
        self._stats["facts"] = self._index_facts_db()
        self._stats["memories"] = self._index_permanent_memories()
        self._stats["conversations"] = self._index_conversations()

        elapsed = time.monotonic() - t0
        total = sum(self._stats.values())
        self._built = True
        log.info(
            "[KnowledgeIndex] Built index: %d entries in %.1fs — %s",
            total,
            elapsed,
            self._stats,
        )
        return dict(self._stats)

    def _index_vault_articles(self) -> int:
        """Embed reference articles from Permanent/References/."""
        refs_dir = self._vault_path / "LLM Memory" / "Permanent" / "References"
        if not refs_dir.exists():
            return 0

        items: list[tuple[str, str, Any]] = []
        for md_file in sorted(refs_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                # Use title from filename, strip frontmatter for embedding
                title = md_file.stem.replace("_", " ")
                # Extract body (skip YAML frontmatter)
                body = _strip_frontmatter(content)
                # Truncate to ~1500 chars for embedding (nomic-embed-text context)
                text = f"{title}: {body[:1500]}"
                meta = {
                    "source": "vault_article",
                    "title": title,
                    "path": str(md_file),
                }
                items.append((text, title, meta))
            except Exception:
                log.warning("[KnowledgeIndex] Failed to read %s", md_file, exc_info=True)

        return self._embed_cached(items, "vault_article")

    def _index_facts_db(self) -> int:
        """Embed all facts from the SQLite facts database."""
        try:
            from .facts_db import _get_conn, init_db

            init_db()
            conn = _get_conn()
            rows = conn.execute(
                "SELECT id, category, key, value, source FROM facts ORDER BY id"
            ).fetchall()
        except Exception:
            log.warning("[KnowledgeIndex] Failed to read facts_db", exc_info=True)
            return 0

        items: list[tuple[str, str, Any]] = []
        for row in rows:
            text = f"{row['key']}: {row['value']}"
            cache_key = f"{row['category']}:{row['key']}"
            meta = {
                "source": "fact",
                "category": row["category"],
                "key": row["key"],
                "fact_id": row["id"],
            }
            items.append((text, cache_key, meta))

        return self._embed_cached(items, "fact")

    def _index_permanent_memories(self) -> int:
        """Embed permanent memories from Permanent/memories.md."""
        mem_file = self._vault_path / "LLM Memory" / "Permanent" / "memories.md"
        if not mem_file.exists():
            return 0

        try:
            content = mem_file.read_text(encoding="utf-8").strip()
        except Exception:
            log.warning("[KnowledgeIndex] Failed to read memories.md", exc_info=True)
            return 0

        if not content:
            return 0

        # Split on headings or double-newlines to get individual memory chunks
        chunks = _split_memory_chunks(content)
        items: list[tuple[str, str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 10:
                continue
            meta = {
                "source": "memory",
                "chunk_index": i,
            }
            items.append((chunk[:1500], f"chunk_{i}", meta))

        return self._embed_cached(items, "memory")

    def _index_conversations(self) -> int:
        """Embed recent conversations from the MemorySystem buffer."""
        try:
            from .memory_system import get_memory_system

            mem = get_memory_system()
            entries = list(mem.recent_conversations)
        except Exception:
            log.warning("[KnowledgeIndex] Failed to read conversations", exc_info=True)
            return 0

        if not entries:
            return 0

        items: list[tuple[str, Any]] = []
        for entry in entries:
            text = f"Q: {entry.message}\nA: {entry.response}"
            # Truncate long exchanges
            text = text[:1500]
            meta = {
                "source": "conversation",
                "user": entry.user,
                "timestamp": entry.timestamp.isoformat(),
            }
            items.append((text, meta))

        return self._embed_and_add(items)

    def add_entry(self, text: str, metadata: dict[str, Any]) -> bool:
        """Incrementally add a single entry to the index.

        Call this when new knowledge is created (new conversation, new fact, etc.)
        so the index stays up to date without a full rebuild.
        """
        embedding = embed_text(text[:1500])
        if not embedding:
            return False
        self._cache.add(text[:1500], metadata, embedding)
        source = metadata.get("source", "unknown")
        with self._lock:
            self._stats[source] = self._stats.get(source, 0) + 1
        # Persist to embedding store for faster restarts
        cache_key = metadata.get("key") or metadata.get("title") or str(metadata.get("chunk_index", ""))
        if cache_key:
            try:
                h = embedding_store.content_hash(text[:1500])
                embedding_store.save_cached(source, [(cache_key, h, embedding, metadata)])
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.40,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across all indexed knowledge.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            threshold: Minimum cosine similarity (0.0-1.0).
            source_filter: Optional filter — "fact", "vault_article",
                           "memory", "conversation".

        Returns:
            List of dicts with keys: score, text, source, and source-specific
            metadata fields.
        """
        query_emb = embed_text(query)
        if not query_emb:
            return []

        # Search with extra headroom if filtering
        search_k = top_k * 3 if source_filter else top_k
        raw = self._cache.search(query_emb, top_k=search_k, threshold=threshold)

        results = []
        for score, text, meta in raw:
            if source_filter and meta.get("source") != source_filter:
                continue
            results.append({"score": round(score, 4), "text": text, **meta})
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def total_entries(self) -> int:
        return self._cache.size

    def get_stats(self) -> dict[str, Any]:
        return {
            "built": self._built,
            "total_entries": self._cache.size,
            "sources": dict(self._stats),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_and_add(self, items: list[tuple[str, Any]]) -> int:
        """Embed a list of (text, metadata) pairs in batches and add to cache."""
        if not items:
            return 0

        added = 0
        for i in range(0, len(items), _EMBED_BATCH_SIZE):
            batch = items[i : i + _EMBED_BATCH_SIZE]
            texts = [text for text, _ in batch]
            embeddings = embed_texts(texts)
            if not embeddings:
                continue
            self._cache.add_batch(batch, embeddings)
            added += len(embeddings)

        return added

    def _embed_cached(self, items: list[tuple[str, str, Any]], source: str) -> int:
        """Embed items with persistent cache — loads cached embeddings from disk
        when content hasn't changed, only calling Ollama for new/modified entries.

        Args:
            items: List of (text, cache_key, metadata) tuples.
            source: Cache source type (e.g. "vault_article", "fact", "memory").

        Returns:
            Total number of entries added to the index.
        """
        if not items:
            return 0

        # Load persistent cache for this source
        try:
            cached = embedding_store.load_cached(source)
        except Exception:
            cached = {}

        hits: list[tuple[str, Any, list[float]]] = []
        misses: list[tuple[str, str, Any]] = []

        for text, cache_key, meta in items:
            h = embedding_store.content_hash(text)
            if cache_key in cached and cached[cache_key][0] == h:
                _, emb, _ = cached[cache_key]
                hits.append((text[:1500], meta, emb))
            else:
                misses.append((text, cache_key, meta))

        # Load cached embeddings directly into SemanticCache
        for text, meta, emb in hits:
            self._cache.add(text, meta, emb)

        # Embed new/changed entries via Ollama
        newly_embedded = 0
        to_save: list[tuple[str, str, list[float], dict]] = []

        for i in range(0, len(misses), _EMBED_BATCH_SIZE):
            batch = misses[i : i + _EMBED_BATCH_SIZE]
            texts = [text[:1500] for text, _, _ in batch]
            embeddings = embed_texts(texts)
            if not embeddings:
                continue
            for (text, cache_key, meta), emb in zip(batch, embeddings):
                self._cache.add(text[:1500], meta, emb)
                h = embedding_store.content_hash(text)
                to_save.append((cache_key, h, emb, meta))
            newly_embedded += len(embeddings)

        # Persist newly embedded entries for next startup
        if to_save:
            try:
                embedding_store.save_cached(source, to_save)
            except Exception:
                log.debug("[KnowledgeIndex] Failed to persist embeddings for %s", source)

        if hits:
            log.info(
                "[KnowledgeIndex] %s: %d from cache, %d newly embedded",
                source, len(hits), newly_embedded,
            )

        return len(hits) + newly_embedded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (--- ... ---) from markdown content."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3 :].strip()
    return content


def _split_memory_chunks(content: str) -> list[str]:
    """Split memories.md into individual memory entries.

    Splits on markdown headings (##, ###) or double-newline paragraphs.
    """
    import re

    # Split on heading lines
    chunks = re.split(r"\n(?=#{1,3}\s)", content)
    # If that only produced one chunk, try double-newline split
    if len(chunks) <= 1:
        chunks = re.split(r"\n\n+", content)
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_knowledge_index: KnowledgeIndex | None = None
_index_lock = threading.Lock()


def get_knowledge_index() -> KnowledgeIndex:
    """Get or create the global KnowledgeIndex singleton."""
    global _knowledge_index
    with _index_lock:
        if _knowledge_index is None:
            _knowledge_index = KnowledgeIndex()
        return _knowledge_index


def init_knowledge_index(vault_path: Path | None = None) -> KnowledgeIndex:
    """Initialize the global KnowledgeIndex and build it."""
    global _knowledge_index
    with _index_lock:
        _knowledge_index = KnowledgeIndex(vault_path=vault_path)
    _knowledge_index.build()
    return _knowledge_index


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------


def _tool_search_knowledge(query: str, source: str = "", top_k: str = "5") -> str:
    """Search across all knowledge sources using semantic similarity."""
    idx = get_knowledge_index()
    if not idx.is_built:
        idx.build()

    k = min(int(top_k), 10)
    source_filter = source if source else None
    results = idx.search(query, top_k=k, source_filter=source_filter)

    if not results:
        return f"No results found for '{query}'."

    lines = [f"Found {len(results)} results for '{query}':\n"]
    for r in results:
        score = r["score"]
        src = r["source"]
        text_preview = r["text"][:200].replace("\n", " ")

        if src == "fact":
            lines.append(f"- [{score:.0%}] **Fact** [{r.get('category', '')}] {r.get('key', '')}: {text_preview}")
        elif src == "vault_article":
            lines.append(f"- [{score:.0%}] **Article** {r.get('title', '')}: {text_preview}")
        elif src == "memory":
            lines.append(f"- [{score:.0%}] **Memory**: {text_preview}")
        elif src == "conversation":
            ts = r.get("timestamp", "")[:10]
            lines.append(f"- [{score:.0%}] **Conversation** ({ts}): {text_preview}")
        else:
            lines.append(f"- [{score:.0%}] **{src}**: {text_preview}")

    stats = idx.get_stats()
    lines.append(f"\n_Index: {stats['total_entries']} entries across {len(stats['sources'])} sources_")
    return "\n".join(lines)


def _tool_knowledge_stats() -> str:
    """Show knowledge index statistics."""
    idx = get_knowledge_index()
    stats = idx.get_stats()
    if not stats["built"]:
        return "Knowledge index not yet built. It will be built on first search."
    lines = [
        "**Knowledge Index Stats**",
        f"Total entries: {stats['total_entries']}",
        "Sources:",
    ]
    for src, count in sorted(stats["sources"].items()):
        lines.append(f"  - {src}: {count}")
    return "\n".join(lines)


def get_knowledge_search_tools() -> list:
    """Get tools for semantic knowledge search."""
    from .core import create_tool

    return [
        create_tool(
            name="search_knowledge",
            description=(
                "Semantic search across ALL knowledge sources — facts database, "
                "vault articles, permanent memories, and conversation history. "
                "Use this when looking for information that might be stored anywhere. "
                "Returns the most relevant results ranked by similarity."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'what do we know about Kubernetes?')",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional: filter by source — 'fact', 'vault_article', 'memory', 'conversation'",
                    },
                    "top_k": {
                        "type": "string",
                        "description": "Number of results (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
            function=_tool_search_knowledge,
        ),
        create_tool(
            name="knowledge_stats",
            description="Show statistics about the semantic knowledge index — how many entries from each source.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=_tool_knowledge_stats,
        ),
    ]
