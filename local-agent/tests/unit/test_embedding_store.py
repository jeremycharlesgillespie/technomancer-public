"""Tests for agent/embedding_store.py — persistent embedding cache."""

import json
import sqlite3

import pytest

from agent import embedding_store


class TestWalMode:
    """WAL journal mode must be active to allow concurrent readers and writers."""

    def test_journal_mode_is_wal(self):
        embedding_store.init_store()
        conn = embedding_store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_concurrent_read_during_open_write_transaction(self):
        """A reader connection must succeed while another connection holds
        an open write transaction — the contention scenario that produces
        `database is locked` without WAL."""
        embedding_store.save_cached(
            "fact", [("seed", "h", [0.1], {})]
        )
        conn = getattr(embedding_store._local, "emb_conn", None)
        if conn:
            conn.close()
            embedding_store._local.emb_conn = None

        writer = sqlite3.connect(str(embedding_store.DB_PATH), timeout=5)
        reader = sqlite3.connect(str(embedding_store.DB_PATH), timeout=5)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO embeddings (source, key, text_hash, embedding) "
                "VALUES (?, ?, ?, ?)",
                ("fact", "concurrent", "h2", b"\x00\x00\x00\x00"),
            )
            count = reader.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            assert count >= 1
            writer.rollback()
        finally:
            writer.close()
            reader.close()


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
        assert stats["stale_count"] == 0
        assert stats["orphan_count"] == 0
        assert stats["last_sweep_at"] is None

    def test_with_data(self):
        embedding_store.save_cached("fact", [("k1", "h1", [0.1], {})])
        embedding_store.save_cached("vault", [("k2", "h2", [0.2], {})])
        stats = embedding_store.get_stats()
        assert stats["total_embeddings"] == 2
        assert stats["sources"]["fact"] == 1
        assert stats["sources"]["vault"] == 1


# ---------------------------------------------------------------------------
# TK-467 — schema migration, upsert, sweep
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """init_store() must backfill TK-467 columns onto pre-existing databases."""

    def test_fresh_schema_has_new_columns(self):
        embedding_store.init_store()
        cols = embedding_store._existing_columns(
            embedding_store._get_conn(), "embeddings"
        )
        assert {"source_hash", "source_path", "stale_since", "updated_at"} <= cols

    def test_migrates_legacy_database(self, tmp_path, monkeypatch):
        """A DB created by the pre-TK-467 schema upgrades in place."""
        import sqlite3

        legacy_path = tmp_path / "legacy.db"
        monkeypatch.setattr(embedding_store, "DB_PATH", legacy_path)
        monkeypatch.setattr(embedding_store, "DB_DIR", tmp_path)
        embedding_store._local.__dict__.pop("emb_conn", None)

        raw = sqlite3.connect(str(legacy_path))
        raw.execute("""
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                key TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(source, key)
            )
        """)
        raw.execute(
            "INSERT INTO embeddings (source, key, text_hash, embedding) VALUES (?, ?, ?, ?)",
            ("fact", "legacy1", "oldhash", b"\x00\x00\x00\x00"),
        )
        raw.commit()
        raw.close()

        embedding_store.init_store()
        cols = embedding_store._existing_columns(
            embedding_store._get_conn(), "embeddings"
        )
        assert "source_hash" in cols
        assert "source_path" in cols
        assert "stale_since" in cols

        # Legacy row still present after migration.
        cached = embedding_store.load_cached("fact")
        assert "legacy1" in cached

    def test_sweep_meta_table_created(self):
        embedding_store.init_store()
        conn = embedding_store._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='embedding_sweep_meta'"
        ).fetchone()
        assert row is not None


class TestSourceHash:
    def test_matches_full_sha256(self):
        import hashlib
        text = "hello world"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert embedding_store.source_hash(text) == expected

    def test_length_64(self):
        assert len(embedding_store.source_hash("anything")) == 64

    def test_distinct_from_short_content_hash(self):
        text = "x" * 100
        assert embedding_store.source_hash(text) != embedding_store.content_hash(text)


class TestUpsert:
    def test_insert_new_row(self):
        result = embedding_store.upsert(
            "fact", "abc", "hello world", [0.1, 0.2], {"tag": "greeting"},
            source_path="/vault/hello.md",
        )
        assert result == "inserted"
        cached = embedding_store.load_cached("fact")
        assert "abc" in cached
        _, emb, meta = cached["abc"]
        assert emb[0] == pytest.approx(0.1)
        assert meta == {"tag": "greeting"}

        conn = embedding_store._get_conn()
        row = conn.execute(
            "SELECT source_hash, source_path FROM embeddings WHERE source=? AND key=?",
            ("fact", "abc"),
        ).fetchone()
        assert row["source_hash"] == embedding_store.source_hash("hello world")
        assert row["source_path"] == "/vault/hello.md"

    def test_unchanged_when_hash_matches(self):
        embedding_store.upsert("fact", "k", "text-v1", [0.1], {})
        result = embedding_store.upsert("fact", "k", "text-v1", [0.1], {})
        assert result == "unchanged"

    def test_updated_when_text_changes(self):
        embedding_store.upsert("fact", "k", "text-v1", [0.1], {})
        result = embedding_store.upsert("fact", "k", "text-v2", [0.9], {"v": 2})
        assert result == "updated"

        conn = embedding_store._get_conn()
        row = conn.execute(
            "SELECT source_hash, metadata FROM embeddings WHERE source=? AND key=?",
            ("fact", "k"),
        ).fetchone()
        assert row["source_hash"] == embedding_store.source_hash("text-v2")
        assert json.loads(row["metadata"]) == {"v": 2}

    def test_update_clears_stale_marker(self):
        embedding_store.upsert("fact", "k", "v1", [0.1], {})
        assert embedding_store.mark_stale("fact", "k") is True

        embedding_store.upsert("fact", "k", "v2", [0.2], {})
        stale_rows = embedding_store.get_stale("fact")
        assert stale_rows == []

    def test_unchanged_clears_stale_marker(self):
        """A redundant upsert with identical text still clears a stale flag."""
        embedding_store.upsert("fact", "k", "same", [0.1], {})
        embedding_store.mark_stale("fact", "k")

        embedding_store.upsert("fact", "k", "same", [0.1], {})
        assert embedding_store.get_stale("fact") == []


class TestMarkStaleAndGetStale:
    def test_mark_stale_then_get(self):
        embedding_store.upsert("fact", "k1", "text1", [0.1], {})
        assert embedding_store.mark_stale("fact", "k1") is True

        stale = embedding_store.get_stale("fact")
        assert len(stale) == 1
        assert stale[0]["source_id"] == "k1"
        assert stale[0]["stale_since"] is not None

    def test_mark_stale_unknown_row_returns_false(self):
        assert embedding_store.mark_stale("fact", "missing") is False

    def test_mark_stale_idempotent(self):
        """Re-marking an already-stale row shouldn't return True twice."""
        embedding_store.upsert("fact", "k", "x", [0.1], {})
        assert embedding_store.mark_stale("fact", "k") is True
        assert embedding_store.mark_stale("fact", "k") is False

    def test_get_stale_no_filter(self):
        embedding_store.upsert("fact", "k1", "a", [0.1], {})
        embedding_store.upsert("vault", "k2", "b", [0.2], {})
        embedding_store.mark_stale("fact", "k1")
        embedding_store.mark_stale("vault", "k2")

        stale = embedding_store.get_stale()
        assert {r["source"] for r in stale} == {"fact", "vault"}

    def test_get_stale_limit(self):
        for i in range(5):
            embedding_store.upsert("fact", f"k{i}", f"text{i}", [0.1], {})
            embedding_store.mark_stale("fact", f"k{i}")
        stale = embedding_store.get_stale("fact", limit=3)
        assert len(stale) == 3


class TestDelete:
    def test_delete_existing(self):
        embedding_store.upsert("fact", "k", "x", [0.1], {})
        assert embedding_store.delete("fact", "k") is True
        assert embedding_store.load_cached("fact") == {}

    def test_delete_missing_returns_false(self):
        assert embedding_store.delete("fact", "missing") is False


class TestSweep:
    def test_deletes_orphans(self):
        embedding_store.upsert("fact", "live", "text", [0.1], {})
        embedding_store.upsert("fact", "gone", "text2", [0.2], {})

        summary = embedding_store.sweep({"fact": lambda: {"live": "text"}})
        assert summary["orphans_deleted"] == 1
        assert summary["checked"] == 2
        assert embedding_store.load_cached("fact").keys() == {"live"}

    def test_marks_stale_on_hash_change(self):
        embedding_store.upsert("fact", "k", "original", [0.1], {})

        summary = embedding_store.sweep({"fact": lambda: {"k": "modified"}})
        assert summary["stale_marked"] == 1
        assert summary["orphans_deleted"] == 0

        stale = embedding_store.get_stale("fact")
        assert len(stale) == 1

    def test_unchanged_clears_stale_marker(self):
        embedding_store.upsert("fact", "k", "same", [0.1], {})
        embedding_store.mark_stale("fact", "k")

        summary = embedding_store.sweep({"fact": lambda: {"k": "same"}})
        assert summary["unchanged"] == 1
        assert embedding_store.get_stale("fact") == []

    def test_backfills_missing_source_hash_without_marking_stale(self):
        """Legacy rows without a source_hash get backfilled silently."""
        conn = embedding_store._get_conn()
        embedding_store.init_store()
        conn.execute(
            """INSERT INTO embeddings
                   (source, key, text_hash, embedding, metadata, source_hash)
               VALUES (?, ?, ?, ?, ?, NULL)""",
            ("fact", "legacy", "oldhash", b"\x00\x00\x00\x00", "{}"),
        )
        conn.commit()

        summary = embedding_store.sweep({"fact": lambda: {"legacy": "current text"}})
        assert summary["refreshed"] == 1
        assert summary["stale_marked"] == 0

        row = conn.execute(
            "SELECT source_hash FROM embeddings WHERE key='legacy'"
        ).fetchone()
        assert row["source_hash"] == embedding_store.source_hash("current text")

    def test_leaves_unregistered_sources_alone(self):
        embedding_store.upsert("fact", "k1", "a", [0.1], {})
        embedding_store.upsert("vault", "k2", "b", [0.2], {})

        # Only sweep "fact" — "vault" rows must be untouched.
        summary = embedding_store.sweep({"fact": lambda: {}})
        assert "vault" not in summary["sources_swept"]
        assert embedding_store.load_cached("vault").keys() == {"k2"}

    def test_provider_exception_recorded_and_other_sources_continue(self):
        embedding_store.upsert("fact", "k1", "live", [0.1], {})
        embedding_store.upsert("vault", "k2", "live2", [0.2], {})

        def bad():
            raise RuntimeError("provider blew up")

        summary = embedding_store.sweep({
            "fact": bad,
            "vault": lambda: {"k2": "live2"},
        })
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["source"] == "fact"
        # vault still processed — no orphans, one unchanged
        assert summary["unchanged"] == 1

    def test_provider_returning_non_dict_recorded_as_error(self):
        embedding_store.upsert("fact", "k", "text", [0.1], {})
        summary = embedding_store.sweep({"fact": lambda: ["not", "a", "dict"]})
        assert summary["errors"] and summary["errors"][0]["source"] == "fact"
        # No rows touched when provider misbehaves
        assert embedding_store.load_cached("fact").keys() == {"k"}

    def test_persists_summary_as_last_sweep(self):
        embedding_store.upsert("fact", "k", "v", [0.1], {})
        summary = embedding_store.sweep({"fact": lambda: {"k": "v"}})

        last = embedding_store.get_last_sweep()
        assert last is not None
        assert last["swept_at"] == summary["swept_at"]
        assert last["checked"] == summary["checked"]

    def test_stats_reflect_stale_and_last_sweep(self):
        embedding_store.upsert("fact", "live", "live", [0.1], {})
        embedding_store.upsert("fact", "changed", "v1", [0.2], {})
        embedding_store.upsert("fact", "gone", "g", [0.3], {})

        embedding_store.sweep({
            "fact": lambda: {"live": "live", "changed": "v2"},
        })

        stats = embedding_store.get_stats()
        assert stats["total_embeddings"] == 2  # gone was deleted
        assert stats["stale_count"] == 1
        assert stats["orphan_count"] == 1
        assert stats["last_sweep_at"] is not None
        assert stats["last_sweep"]["stale_marked"] == 1


class TestStatsEndpoint:
    """The Flask /api/embeddings/stats endpoint returns the store stats."""

    def test_endpoint_returns_stats(self):
        from idea_board.web import app

        embedding_store.upsert("fact", "k", "hello", [0.1], {})

        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/api/embeddings/stats")

        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["total_embeddings"] == 1
        assert payload["sources"]["fact"] == 1
        assert "stale_count" in payload
        assert "orphan_count" in payload
        assert "last_sweep_at" in payload
