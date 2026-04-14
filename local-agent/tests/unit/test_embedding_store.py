"""Tests for agent/embedding_store.py — persistent embedding cache."""

from agent import embedding_store


class TestInitStore:
    def test_creates_table(self):
        embedding_store.init_store()
        conn = embedding_store._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent(self):
        embedding_store.init_store()
        embedding_store.init_store()
        conn = embedding_store._get_conn()
        tables = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sqlite_master WHERE name='embeddings'"
        ).fetchone()
        assert tables["cnt"] == 1


class TestContentHash:
    def test_consistent(self):
        h1 = embedding_store.content_hash("hello world")
        h2 = embedding_store.content_hash("hello world")
        assert h1 == h2

    def test_different_texts(self):
        h1 = embedding_store.content_hash("hello")
        h2 = embedding_store.content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = embedding_store.content_hash("test")
        assert len(h) == 16


class TestPackUnpack:
    def test_roundtrip(self):
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = embedding_store._pack_embedding(original)
        unpacked = embedding_store._unpack_embedding(packed)
        assert len(unpacked) == len(original)
        for a, b in zip(original, unpacked):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        packed = embedding_store._pack_embedding([])
        unpacked = embedding_store._unpack_embedding(packed)
        assert unpacked == []


class TestSaveAndLoad:
    def test_save_and_load(self):
        entries = [
            ("key1", "hash1", [0.1, 0.2, 0.3], {"source": "fact"}),
            ("key2", "hash2", [0.4, 0.5, 0.6], {"source": "fact"}),
        ]
        saved = embedding_store.save_cached("fact", entries)
        assert saved == 2

        cached = embedding_store.load_cached("fact")
        assert len(cached) == 2
        assert "key1" in cached
        assert "key2" in cached
        assert cached["key1"][0] == "hash1"
        assert len(cached["key1"][1]) == 3

    def test_load_empty(self):
        cached = embedding_store.load_cached("nonexistent")
        assert cached == {}

    def test_upsert_replaces(self):
        embedding_store.save_cached("fact", [("key1", "hash1", [0.1, 0.2], {"v": 1})])
        embedding_store.save_cached("fact", [("key1", "hash2", [0.3, 0.4], {"v": 2})])
        cached = embedding_store.load_cached("fact")
        assert len(cached) == 1
        assert cached["key1"][0] == "hash2"

    def test_source_isolation(self):
        embedding_store.save_cached("fact", [("k1", "h1", [0.1], {"s": "fact"})])
        embedding_store.save_cached("vault", [("k2", "h2", [0.2], {"s": "vault"})])
        assert len(embedding_store.load_cached("fact")) == 1
        assert len(embedding_store.load_cached("vault")) == 1
        assert "k1" in embedding_store.load_cached("fact")
        assert "k2" in embedding_store.load_cached("vault")

    def test_save_empty_list(self):
        assert embedding_store.save_cached("fact", []) == 0


class TestClearSource:
    def test_clear(self):
        embedding_store.save_cached("fact", [("k1", "h1", [0.1], {})])
        embedding_store.save_cached("fact", [("k2", "h2", [0.2], {})])
        deleted = embedding_store.clear_source("fact")
        assert deleted == 2
        assert embedding_store.load_cached("fact") == {}

    def test_clear_only_target_source(self):
        embedding_store.save_cached("fact", [("k1", "h1", [0.1], {})])
        embedding_store.save_cached("vault", [("k2", "h2", [0.2], {})])
        embedding_store.clear_source("fact")
        assert embedding_store.load_cached("fact") == {}
        assert len(embedding_store.load_cached("vault")) == 1


class TestGetStats:
    def test_empty(self):
        stats = embedding_store.get_stats()
        assert stats["total_embeddings"] == 0
        assert stats["sources"] == {}

    def test_with_data(self):
        embedding_store.save_cached("fact", [("k1", "h1", [0.1], {})])
        embedding_store.save_cached("vault", [("k2", "h2", [0.2], {})])
        stats = embedding_store.get_stats()
        assert stats["total_embeddings"] == 2
        assert stats["sources"]["fact"] == 1
        assert stats["sources"]["vault"] == 1
