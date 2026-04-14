"""
Persistent Embedding Store — SQLite cache for semantic index embeddings.

Caches (source, key, content_hash, embedding) so the KnowledgeIndex can
reload embeddings from disk on restart instead of re-embedding everything
via Ollama.  Only new or changed content triggers an Ollama embed call.

Database: ``local-agent/data/embeddings.db``
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "embeddings.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection (created on first use)."""
    conn: sqlite3.Connection | None = getattr(_local, "emb_conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.emb_conn = conn
    return conn


def init_store() -> None:
    """Create the embeddings table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT NOT NULL,
            key         TEXT NOT NULL,
            text_hash   TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source, key)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_source
        ON embeddings (source)
    """)
    conn.commit()


def content_hash(text: str) -> str:
    """SHA-256 hash (truncated to 16 chars) for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into compact bytes (4 bytes per float)."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(data: bytes) -> list[float]:
    """Unpack bytes back into a float list."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def load_cached(source: str) -> dict[str, tuple[str, list[float], dict]]:
    """Load all cached embeddings for a source type.

    Returns:
        dict mapping key -> (text_hash, embedding, metadata)
    """
    conn = _get_conn()
    init_store()
    rows = conn.execute(
        "SELECT key, text_hash, embedding, metadata FROM embeddings WHERE source = ?",
        (source,),
    ).fetchall()

    result: dict[str, tuple[str, list[float], dict]] = {}
    for row in rows:
        try:
            emb = _unpack_embedding(row["embedding"])
            meta = json.loads(row["metadata"])
            result[row["key"]] = (row["text_hash"], emb, meta)
        except Exception:
            log.warning("[EmbeddingStore] Failed to load cached entry: %s/%s", source, row["key"])
    return result


def save_cached(
    source: str,
    entries: list[tuple[str, str, list[float], dict]],
) -> int:
    """Bulk save embeddings for a source type.

    Args:
        source: Source type (e.g. "vault_article", "fact", "memory").
        entries: List of (key, text_hash, embedding, metadata) tuples.

    Returns:
        Number of entries saved.
    """
    if not entries:
        return 0
    conn = _get_conn()
    init_store()
    saved = 0
    for key, text_hash, embedding, metadata in entries:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO embeddings
                   (source, key, text_hash, embedding, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (source, key, text_hash, _pack_embedding(embedding), json.dumps(metadata)),
            )
            saved += 1
        except Exception:
            log.warning("[EmbeddingStore] Failed to save: %s/%s", source, key)
    conn.commit()
    return saved


def clear_source(source: str) -> int:
    """Remove all cached embeddings for a source type. Returns count deleted."""
    conn = _get_conn()
    init_store()
    cursor = conn.execute("DELETE FROM embeddings WHERE source = ?", (source,))
    conn.commit()
    return cursor.rowcount


def get_stats() -> dict[str, Any]:
    """Return statistics about the embedding store."""
    conn = _get_conn()
    init_store()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM embeddings").fetchone()["cnt"]
    sources = conn.execute(
        "SELECT source, COUNT(*) AS cnt FROM embeddings GROUP BY source ORDER BY source"
    ).fetchall()
    return {
        "total_embeddings": total,
        "sources": {r["source"]: r["cnt"] for r in sources},
    }
