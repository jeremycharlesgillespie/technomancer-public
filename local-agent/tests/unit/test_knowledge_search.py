"""Tests for agent/knowledge_search.py — unified semantic search."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.knowledge_search import (
    KnowledgeIndex,
    _split_memory_chunks,
    _strip_frontmatter,
    get_knowledge_index,
    get_knowledge_search_tools,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class TestStripFrontmatter:
    def test_removes_yaml_frontmatter(self):
        content = "---\ntitle: Test\nsource: wiki\n---\nActual body content."
        assert _strip_frontmatter(content) == "Actual body content."

    def test_no_frontmatter(self):
        content = "Just plain content."
        assert _strip_frontmatter(content) == "Just plain content."

    def test_incomplete_frontmatter(self):
        content = "---\ntitle: Test\nno closing marker"
        assert _strip_frontmatter(content) == "---\ntitle: Test\nno closing marker"


class TestSplitMemoryChunks:
    def test_splits_on_headings(self):
        content = "## Memory 1\nSome text\n## Memory 2\nMore text"
        chunks = _split_memory_chunks(content)
        assert len(chunks) == 2
        assert "Memory 1" in chunks[0]
        assert "Memory 2" in chunks[1]

    def test_splits_on_double_newline(self):
        content = "First paragraph.\n\nSecond paragraph."
        chunks = _split_memory_chunks(content)
        assert len(chunks) == 2

    def test_empty_input(self):
        assert _split_memory_chunks("") == []

    def test_single_chunk(self):
        content = "Just one paragraph with no splits."
        chunks = _split_memory_chunks(content)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# KnowledgeIndex — core class
# ---------------------------------------------------------------------------


class TestKnowledgeIndex:
    """Tests for KnowledgeIndex with mocked embeddings."""

    def _make_fake_embeddings(self, texts):
        """Generate deterministic fake embeddings for testing."""
        result = []
        for i, text in enumerate(texts):
            # Simple hash-based fake embedding (3-dim for tests)
            h = hash(text) % 1000
            result.append([float(h % 10) / 10, float((h // 10) % 10) / 10, float((h // 100) % 10) / 10])
        return result

    def test_build_empty_vault(self, temp_vault):
        """Build with no articles, no facts, no conversations."""
        with patch("agent.knowledge_search.embed_texts", return_value=[]):
            idx = KnowledgeIndex(vault_path=temp_vault)
            stats = idx.build()
            assert stats["vault_articles"] == 0
            assert stats["memories"] == 0
            assert idx.is_built

    def test_index_vault_articles(self, temp_vault):
        """Index reference articles from vault."""
        refs_dir = temp_vault / "LLM Memory" / "Permanent" / "References"
        refs_dir.mkdir(parents=True, exist_ok=True)

        # Create a reference article
        (refs_dir / "Python.md").write_text(
            "---\ntitle: Python\nsource: wikipedia\n---\n"
            "Python is a programming language.",
            encoding="utf-8",
        )
        (refs_dir / "Kubernetes.md").write_text(
            "Kubernetes is a container orchestration platform.",
            encoding="utf-8",
        )

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("agent.knowledge_search.embed_texts", return_value=fake_embeddings):
            idx = KnowledgeIndex(vault_path=temp_vault)
            count = idx._index_vault_articles()
            assert count == 2

    def test_index_facts_db(self, temp_vault):
        """Index facts from the SQLite database."""
        fake_rows = [
            {"id": 1, "category": "definition", "key": "Python", "value": "A programming language", "source": "seed"},
            {"id": 2, "category": "geography", "key": "Japan capital", "value": "Tokyo", "source": "seed"},
        ]

        # Use proper dict-like mock rows
        mock_rows = []
        for row in fake_rows:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, k, r=row: r[k]
            mock_rows.append(mock_row)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("agent.knowledge_search.embed_texts", return_value=fake_embeddings), \
             patch("agent.facts_db.init_db"), \
             patch("agent.facts_db._get_conn", return_value=mock_conn):
            idx = KnowledgeIndex(vault_path=temp_vault)
            count = idx._index_facts_db()
            assert count == 2

    def test_index_permanent_memories(self, temp_vault):
        """Index permanent memories from vault."""
        mem_file = temp_vault / "LLM Memory" / "Permanent" / "memories.md"
        mem_file.write_text(
            "## 2026-03-14 - preferences\nUser likes dark mode.\n\n"
            "## 2026-03-14 - facts\nProject uses Python.\n",
            encoding="utf-8",
        )

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("agent.knowledge_search.embed_texts", return_value=fake_embeddings):
            idx = KnowledgeIndex(vault_path=temp_vault)
            count = idx._index_permanent_memories()
            assert count == 2

    def test_search_returns_results(self, temp_vault):
        """Search finds relevant entries."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        # Manually add entries with known embeddings
        idx._cache.add(
            "Python is a programming language",
            {"source": "fact", "category": "definition", "key": "Python", "fact_id": 1},
            [1.0, 0.0, 0.0],
        )
        idx._cache.add(
            "Tokyo is the capital of Japan",
            {"source": "fact", "category": "geography", "key": "Japan capital", "fact_id": 2},
            [0.0, 1.0, 0.0],
        )
        idx._built = True

        # Search for something similar to first entry
        with patch("agent.knowledge_search.embed_text", return_value=[0.9, 0.1, 0.0]):
            results = idx.search("programming language")
            assert len(results) >= 1
            assert results[0]["source"] == "fact"
            assert results[0]["key"] == "Python"

    def test_search_with_source_filter(self, temp_vault):
        """Source filter restricts results."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        idx._cache.add(
            "Python programming",
            {"source": "fact", "category": "definition", "key": "Python", "fact_id": 1},
            [1.0, 0.0, 0.0],
        )
        idx._cache.add(
            "Python discussion",
            {"source": "conversation", "user": "test", "timestamp": "2026-01-01"},
            [0.95, 0.05, 0.0],
        )
        idx._built = True

        with patch("agent.knowledge_search.embed_text", return_value=[1.0, 0.0, 0.0]):
            results = idx.search("python", source_filter="conversation")
            assert all(r["source"] == "conversation" for r in results)

    def test_search_empty_index(self, temp_vault):
        """Search on empty index returns no results."""
        idx = KnowledgeIndex(vault_path=temp_vault)
        with patch("agent.knowledge_search.embed_text", return_value=[1.0, 0.0, 0.0]):
            results = idx.search("anything")
            assert results == []

    def test_search_embedding_failure(self, temp_vault):
        """Search returns empty when embedding fails."""
        idx = KnowledgeIndex(vault_path=temp_vault)
        with patch("agent.knowledge_search.embed_text", return_value=[]):
            results = idx.search("anything")
            assert results == []

    def test_add_entry_incremental(self, temp_vault):
        """Incrementally add a single entry."""
        idx = KnowledgeIndex(vault_path=temp_vault)
        with patch("agent.knowledge_search.embed_text", return_value=[0.5, 0.5, 0.0]):
            ok = idx.add_entry("new fact", {"source": "fact", "key": "test"})
            assert ok
            assert idx.total_entries == 1

    def test_add_entry_embedding_failure(self, temp_vault):
        """Incremental add fails gracefully when embedding fails."""
        idx = KnowledgeIndex(vault_path=temp_vault)
        with patch("agent.knowledge_search.embed_text", return_value=[]):
            ok = idx.add_entry("new fact", {"source": "fact", "key": "test"})
            assert not ok
            assert idx.total_entries == 0

    def test_get_stats(self, temp_vault):
        """Stats reflect index state."""
        idx = KnowledgeIndex(vault_path=temp_vault)
        stats = idx.get_stats()
        assert stats["built"] is False
        assert stats["total_entries"] == 0

    def test_index_conversations(self, temp_vault, memory_system):
        """Index conversations from the MemorySystem buffer."""
        memory_system.log_conversation("user1", "Hello world", "Hi there!")
        memory_system.log_conversation("user1", "What is Python?", "A programming language.")

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("agent.knowledge_search.embed_texts", return_value=fake_embeddings), \
             patch("agent.memory_system.get_memory_system", return_value=memory_system):
            idx = KnowledgeIndex(vault_path=temp_vault)
            count = idx._index_conversations()
            assert count == 2


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------


class TestToolInterface:
    def test_tool_search_knowledge(self, temp_vault):
        """Tool wrapper formats results correctly."""
        from agent.knowledge_search import _tool_search_knowledge

        mock_idx = MagicMock()
        mock_idx.is_built = True
        mock_idx.search.return_value = [
            {"score": 0.85, "text": "Python is a language", "source": "fact", "category": "definition", "key": "Python"},
        ]
        mock_idx.get_stats.return_value = {"total_entries": 10, "sources": {"fact": 5, "conversation": 5}}

        with patch("agent.knowledge_search.get_knowledge_index", return_value=mock_idx):
            result = _tool_search_knowledge("Python")
            assert "Python" in result
            assert "85%" in result
            assert "Fact" in result

    def test_tool_search_no_results(self, temp_vault):
        """Tool reports no results."""
        from agent.knowledge_search import _tool_search_knowledge

        mock_idx = MagicMock()
        mock_idx.is_built = True
        mock_idx.search.return_value = []

        with patch("agent.knowledge_search.get_knowledge_index", return_value=mock_idx):
            result = _tool_search_knowledge("nonexistent thing")
            assert "No results" in result

    def test_tool_knowledge_stats(self):
        """Stats tool returns formatted output."""
        from agent.knowledge_search import _tool_knowledge_stats

        mock_idx = MagicMock()
        mock_idx.get_stats.return_value = {
            "built": True,
            "total_entries": 100,
            "sources": {"fact": 76, "conversation": 20, "memory": 4},
        }

        with patch("agent.knowledge_search.get_knowledge_index", return_value=mock_idx):
            result = _tool_knowledge_stats()
            assert "100" in result
            assert "fact" in result

    def test_get_tools_returns_two(self):
        """get_knowledge_search_tools returns 2 tools."""
        tools = get_knowledge_search_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "search_knowledge" in names
        assert "knowledge_stats" in names


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_knowledge_index_returns_same(self, monkeypatch):
        """Singleton returns same instance."""
        import agent.knowledge_search as ks_module
        monkeypatch.setattr(ks_module, "_knowledge_index", None)

        idx1 = get_knowledge_index()
        idx2 = get_knowledge_index()
        assert idx1 is idx2

    def test_init_knowledge_index_builds(self, temp_vault, monkeypatch):
        """init_knowledge_index creates and builds the index."""
        import agent.knowledge_search as ks_module
        monkeypatch.setattr(ks_module, "_knowledge_index", None)

        with patch("agent.knowledge_search.embed_texts", return_value=[]):
            from agent.knowledge_search import init_knowledge_index
            idx = init_knowledge_index(vault_path=temp_vault)
            assert idx.is_built


# ---------------------------------------------------------------------------
# Persistent embedding cache (_embed_cached)
# ---------------------------------------------------------------------------


class TestCachedIndexing:
    """Tests for persistent embedding cache via embedding_store."""

    def test_cache_hit_skips_embedding(self, temp_vault):
        """Second build uses cached embeddings — no embed_texts calls."""
        refs_dir = temp_vault / "LLM Memory" / "Permanent" / "References"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "Python.md").write_text(
            "Python is a programming language.", encoding="utf-8"
        )

        fake_embeddings = [[0.1, 0.2, 0.3]]

        # First build: embeds and persists to store
        with patch("agent.knowledge_search.embed_texts", return_value=fake_embeddings):
            idx1 = KnowledgeIndex(vault_path=temp_vault)
            count1 = idx1._index_vault_articles()
            assert count1 == 1

        # Second build: should load from cache, no Ollama calls
        with patch("agent.knowledge_search.embed_texts") as mock_embed:
            idx2 = KnowledgeIndex(vault_path=temp_vault)
            count2 = idx2._index_vault_articles()
            assert count2 == 1
            mock_embed.assert_not_called()

    def test_changed_content_reembeds(self, temp_vault):
        """Changed file content triggers re-embedding."""
        refs_dir = temp_vault / "LLM Memory" / "Permanent" / "References"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "Python.md").write_text(
            "Python is a programming language.", encoding="utf-8"
        )

        # First build
        with patch("agent.knowledge_search.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            idx1 = KnowledgeIndex(vault_path=temp_vault)
            idx1._index_vault_articles()

        # Change file content
        (refs_dir / "Python.md").write_text(
            "Python is an amazing programming language!", encoding="utf-8"
        )

        # Second build: changed content forces re-embedding
        with patch("agent.knowledge_search.embed_texts", return_value=[[0.4, 0.5, 0.6]]) as mock_embed:
            idx2 = KnowledgeIndex(vault_path=temp_vault)
            count2 = idx2._index_vault_articles()
            assert count2 == 1
            mock_embed.assert_called_once()

    def test_store_failure_falls_back(self, temp_vault):
        """Embedding store failure falls back to normal embedding."""
        refs_dir = temp_vault / "LLM Memory" / "Permanent" / "References"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "Test.md").write_text("Test content.", encoding="utf-8")

        with patch("agent.knowledge_search.embed_texts", return_value=[[0.1, 0.2, 0.3]]), \
             patch("agent.knowledge_search.embedding_store.load_cached", side_effect=Exception("DB error")):
            idx = KnowledgeIndex(vault_path=temp_vault)
            count = idx._index_vault_articles()
            assert count == 1

    def test_facts_cache_hit(self, temp_vault):
        """Facts DB entries use persistent cache on second build."""
        fake_rows = [
            {"id": 1, "category": "definition", "key": "Python", "value": "A programming language", "source": "seed"},
        ]
        mock_rows = []
        for row in fake_rows:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, k, r=row: r[k]
            mock_rows.append(mock_row)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        # First build: embeds and caches
        with patch("agent.knowledge_search.embed_texts", return_value=[[0.1, 0.2, 0.3]]), \
             patch("agent.facts_db.init_db"), \
             patch("agent.facts_db._get_conn", return_value=mock_conn):
            idx1 = KnowledgeIndex(vault_path=temp_vault)
            idx1._index_facts_db()

        # Second build: cache hit
        with patch("agent.knowledge_search.embed_texts") as mock_embed, \
             patch("agent.facts_db.init_db"), \
             patch("agent.facts_db._get_conn", return_value=mock_conn):
            idx2 = KnowledgeIndex(vault_path=temp_vault)
            count2 = idx2._index_facts_db()
            assert count2 == 1
            mock_embed.assert_not_called()

    def test_new_article_added_to_existing_cache(self, temp_vault):
        """Adding a new article embeds only the new one, not the cached one."""
        refs_dir = temp_vault / "LLM Memory" / "Permanent" / "References"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "Python.md").write_text(
            "Python is a programming language.", encoding="utf-8"
        )

        # First build: one article
        with patch("agent.knowledge_search.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            idx1 = KnowledgeIndex(vault_path=temp_vault)
            idx1._index_vault_articles()

        # Add a second article
        (refs_dir / "Kubernetes.md").write_text(
            "Kubernetes is a container orchestration platform.", encoding="utf-8"
        )

        # Second build: Python from cache, Kubernetes newly embedded
        with patch("agent.knowledge_search.embed_texts", return_value=[[0.4, 0.5, 0.6]]) as mock_embed:
            idx2 = KnowledgeIndex(vault_path=temp_vault)
            count2 = idx2._index_vault_articles()
            assert count2 == 2  # 1 cached + 1 new
            mock_embed.assert_called_once()  # Only Kubernetes was embedded


# ---------------------------------------------------------------------------
# Search latency metrics
# -------------------------------------


class TestSearchLatencyMetrics:
    """Test that KnowledgeIndex.search() records Prometheus metrics."""

    def test_search_records_metrics_with_source_filter(self, temp_vault):
        """Search with source_filter records metrics labeled by that filter."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        # Mock the cache search to return results
        mock_results = [
            (0.95, "Test result", {"source": "fact", "key": "test1"}),
            (0.85, "Another result", {"source": "fact", "key": "test2"}),
        ]
        idx._cache.search = MagicMock(return_value=mock_results)

        # Mock embed_text to return a valid embedding
        with patch("agent.knowledge_search.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = idx.search("test query", source_filter="fact")

        # Verify results are returned
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["source"] == "fact"

    def test_search_records_metrics_without_source_filter(self, temp_vault):
        """Search without source_filter records metrics labeled as 'all'."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        # Mock the cache search to return results
        mock_results = [
            (0.95, "Test result", {"source": "vault_article", "key": "test1"}),
            (0.85, "Another result", {"source": "memory", "key": "test2"}),
        ]
        idx._cache.search = MagicMock(return_value=mock_results)

        # Mock embed_text to return a valid embedding
        with patch("agent.knowledge_search.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = idx.search("test query")

        # Verify results are returned
        assert len(results) == 2
        assert results[0]["score"] == 0.95

    def test_search_records_metrics_on_empty_results(self, temp_vault):
        """Search that returns no results still records metrics."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        # Mock the cache search to return empty results
        idx._cache.search = MagicMock(return_value=[])

        # Mock embed_text to return a valid embedding
        with patch("agent.knowledge_search.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = idx.search("test query")

        # Verify empty results are returned
        assert results == []

    def test_search_records_metrics_with_none_source_filter(self, temp_vault):
        """Search with None source_filter records metrics labeled as 'all'."""
        idx = KnowledgeIndex(vault_path=temp_vault)

        # Mock the cache search to return results
        mock_results = [
            (0.95, "Test result", {"source": "conversation", "key": "test1"}),
        ]
        idx._cache.search = MagicMock(return_value=mock_results)

        # Mock embed_text to return a valid embedding
        with patch("agent.knowledge_search.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = idx.search("test query", source_filter=None)

        # Verify results are returned
        assert len(results) == 1
        assert results[0]["score"] == 0.95
