"""
Persistent Embedding Store — SQLite cache for semantic index embeddings.

Caches (source, key, content_hash, embedding) so the KnowledgeIndex can
reload embeddings from disk on restart instead of re-embedding everything
via Ollama.  Only new or changed content triggers an Ollama embed call.

Database: ``local-agent/data/embeddings.db``

Lifecycle management (TK-467):
    Each row stores a full SHA-256 ``source_hash`` of the original text plus
    an optional ``source_path`` / ``source_id`` pointer.  ``upsert()`` is the
    preferred write path; it is idempotent and overwrites the row only when
    the hash changes.  ``sweep()`` walks caller-provided source providers,
    deletes rows whose source no longer exists, and flags rows whose source
    text has changed by setting a ``stale_since`` timestamp that callers can
    later read via ``get_stale()``.  ``get_stats()`` exposes totals, stale
    count, orphan count, and last sweep time for the ``/api/embeddings/stats``
    endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names currently defined on ``table``."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_store() -> None:
    """Create the embeddings table and apply schema migrations."""
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

    # TK-467 schema additions — applied lazily so existing databases upgrade
    # without a full rebuild.
    cols = _existing_columns(conn, "embeddings")
    if "source_hash" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN source_hash TEXT")
    if "source_path" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN source_path TEXT")
    if "stale_since" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN stale_since TEXT")
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN updated_at TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_sweep_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()


def content_hash(text: str) -> str:
    """Short SHA-256 hash (16 hex chars) used for fast-path change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def source_hash(text: str) -> str:
    """Full SHA-256 hex digest of ``text`` used as the canonical source hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    now = _now_iso()
    for key, text_hash, embedding, metadata in entries:
        try:
            conn.execute(
                """INSERT INTO embeddings
                       (source, key, text_hash, embedding, metadata, updated_at, stale_since)
                   VALUES (?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(source, key) DO UPDATE SET
                       text_hash   = excluded.text_hash,
                       embedding   = excluded.embedding,
                       metadata    = excluded.metadata,
                       updated_at  = excluded.updated_at,
                       stale_since = NULL""",
                (source, key, text_hash, _pack_embedding(embedding), json.dumps(metadata), now),
            )
            saved += 1
        except Exception:
            log.warning("[EmbeddingStore] Failed to save: %s/%s", source, key)
    conn.commit()
    return saved


def upsert(
    source: str,
    source_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
    source_path: str | None = None,
) -> str:
    """Upsert an embedding keyed on ``(source, source_id)``.

    Computes the canonical SHA-256 ``source_hash`` from ``text`` and writes
    only when the hash differs from the currently stored row.  Clears any
    ``stale_since`` marker on update.

    Args:
        source: Source type (e.g. "vault_article", "fact").
        source_id: Stable identifier within that source type.
        text: The source text from which the embedding was derived.
        embedding: The embedding vector.
        metadata: Optional metadata dict (serialized as JSON).
        source_path: Optional filesystem path for file-backed sources.

    Returns:
        ``"inserted"``, ``"updated"``, or ``"unchanged"``.
    """
    conn = _get_conn()
    init_store()
    meta = metadata or {}
    shash = source_hash(text)
    thash = content_hash(text)

    row = conn.execute(
        "SELECT source_hash FROM embeddings WHERE source = ? AND key = ?",
        (source, source_id),
    ).fetchone()

    now = _now_iso()
    if row is None:
        conn.execute(
            """INSERT INTO embeddings
                   (source, key, text_hash, embedding, metadata,
                    source_hash, source_path, updated_at, stale_since)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                source,
                source_id,
                thash,
                _pack_embedding(embedding),
                json.dumps(meta),
                shash,
                source_path,
                now,
            ),
        )
        conn.commit()
        return "inserted"

    if row["source_hash"] == shash:
        # Same content — just clear any stale marker so repeated calls converge.
        conn.execute(
            "UPDATE embeddings SET stale_since = NULL WHERE source = ? AND key = ?",
            (source, source_id),
        )
        conn.commit()
        return "unchanged"

    conn.execute(
        """UPDATE embeddings
              SET text_hash   = ?,
                  embedding   = ?,
                  metadata    = ?,
                  source_hash = ?,
                  source_path = ?,
                  updated_at  = ?,
                  stale_since = NULL
            WHERE source = ? AND key = ?""",
        (
            thash,
            _pack_embedding(embedding),
            json.dumps(meta),
            shash,
            source_path,
            now,
            source,
            source_id,
        ),
    )
    conn.commit()
    return "updated"


def mark_stale(source: str, source_id: str) -> bool:
    """Flag a row as stale (source text changed). Returns True if a row was marked."""
    conn = _get_conn()
    init_store()
    cursor = conn.execute(
        """UPDATE embeddings
              SET stale_since = ?
            WHERE source = ? AND key = ? AND stale_since IS NULL""",
        (_now_iso(), source, source_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_stale(source: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Return rows currently flagged stale, oldest first."""
    conn = _get_conn()
    init_store()
    if source:
        rows = conn.execute(
            """SELECT source, key, source_path, source_hash, stale_since, metadata
                 FROM embeddings
                WHERE source = ? AND stale_since IS NOT NULL
                ORDER BY stale_since ASC
                LIMIT ?""",
            (source, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT source, key, source_path, source_hash, stale_since, metadata
                 FROM embeddings
                WHERE stale_since IS NOT NULL
                ORDER BY stale_since ASC
                LIMIT ?""",
            (limit,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            meta = json.loads(row["metadata"])
        except Exception:
            meta = {}
        out.append(
            {
                "source": row["source"],
                "source_id": row["key"],
                "source_path": row["source_path"],
                "source_hash": row["source_hash"],
                "stale_since": row["stale_since"],
                "metadata": meta,
            }
        )
    return out


def delete(source: str, source_id: str) -> bool:
    """Delete a single embedding row. Returns True if a row was removed."""
    conn = _get_conn()
    init_store()
    cursor = conn.execute(
        "DELETE FROM embeddings WHERE source = ? AND key = ?",
        (source, source_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def clear_source(source: str) -> int:
    """Remove all cached embeddings for a source type. Returns count deleted."""
    conn = _get_conn()
    init_store()
    cursor = conn.execute("DELETE FROM embeddings WHERE source = ?", (source,))
    conn.commit()
    return cursor.rowcount


#: Type alias — a source provider is a zero-arg callable that returns the
#: current live state of a source as ``{source_id: current_text}``.
SourceProvider = Callable[[], dict[str, str]]


def sweep(source_providers: dict[str, SourceProvider]) -> dict[str, Any]:
    """Audit the store against caller-supplied providers of live source state.

    For each registered source type the provider returns the current set of
    ``{source_id: current_text}`` pairs.  Rows whose ``source_id`` is absent
    from the provider's dict are deleted as orphans; rows whose source text
    hashes differently than the stored ``source_hash`` are flagged stale by
    setting ``stale_since``.  Unchanged rows clear any existing stale marker.

    Source types not present in ``source_providers`` are left untouched so
    that partial sweepers don't delete rows managed by other subsystems.

    Args:
        source_providers: Mapping of source type -> provider callable.

    Returns:
        Summary dict with per-run counts and the sweep timestamp.
    """
    conn = _get_conn()
    init_store()
    started = time.monotonic()
    sweep_at = _now_iso()

    checked = 0
    stale_marked = 0
    orphans_deleted = 0
    unchanged = 0
    refreshed = 0  # rows that previously had no source_hash and have one now
    errors: list[dict[str, str]] = []

    for source, provider in source_providers.items():
        try:
            live = provider()
        except Exception as exc:
            log.warning("[EmbeddingStore] sweep provider failed for %s: %s", source, exc)
            errors.append({"source": source, "error": str(exc)})
            continue
        if not isinstance(live, dict):
            log.warning(
                "[EmbeddingStore] sweep provider for %s returned %s, expected dict",
                source, type(live).__name__,
            )
            errors.append({"source": source, "error": "provider did not return dict"})
            continue

        rows = conn.execute(
            "SELECT key, source_hash FROM embeddings WHERE source = ?",
            (source,),
        ).fetchall()

        db_keys = {r["key"] for r in rows}

        for row in rows:
            checked += 1
            key = row["key"]
            stored_hash = row["source_hash"]
            if key not in live:
                conn.execute(
                    "DELETE FROM embeddings WHERE source = ? AND key = ?",
                    (source, key),
                )
                orphans_deleted += 1
                continue

            current_hash = source_hash(live[key])
            if stored_hash is None:
                # Legacy row from before the source_hash column existed — backfill
                # silently without marking stale so we don't force a spurious
                # re-embed of rows we have no prior hash for.
                conn.execute(
                    "UPDATE embeddings SET source_hash = ? WHERE source = ? AND key = ?",
                    (current_hash, source, key),
                )
                refreshed += 1
            elif stored_hash != current_hash:
                conn.execute(
                    """UPDATE embeddings
                          SET stale_since = COALESCE(stale_since, ?)
                        WHERE source = ? AND key = ?""",
                    (sweep_at, source, key),
                )
                stale_marked += 1
            else:
                conn.execute(
                    """UPDATE embeddings
                          SET stale_since = NULL
                        WHERE source = ? AND key = ? AND stale_since IS NOT NULL""",
                    (source, key),
                )
                unchanged += 1

        # Providers that list a source_id we've never seen are ignored here —
        # the caller is responsible for inserting new rows via upsert().

    conn.commit()
    duration_ms = int((time.monotonic() - started) * 1000)

    summary = {
        "swept_at": sweep_at,
        "duration_ms": duration_ms,
        "sources_swept": sorted(source_providers.keys()),
        "checked": checked,
        "stale_marked": stale_marked,
        "orphans_deleted": orphans_deleted,
        "unchanged": unchanged,
        "refreshed": refreshed,
        "errors": errors,
    }

    conn.execute(
        "INSERT OR REPLACE INTO embedding_sweep_meta (key, value) VALUES (?, ?)",
        ("last_sweep", json.dumps(summary)),
    )
    conn.commit()

    return summary


def get_last_sweep() -> dict[str, Any] | None:
    """Return the most recent sweep summary, or None if no sweep has run."""
    conn = _get_conn()
    init_store()
    row = conn.execute(
        "SELECT value FROM embedding_sweep_meta WHERE key = 'last_sweep'"
    ).fetchone()
    if not row:
        return None
    try:
        data = json.loads(row["value"])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_stats() -> dict[str, Any]:
    """Return statistics about the embedding store.

    Includes per-source row counts, current stale count (rows with
    ``stale_since`` set), and the summary of the most recent sweep (for
    ``last_sweep_at`` and ``orphans_deleted`` visibility).
    """
    conn = _get_conn()
    init_store()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM embeddings").fetchone()["cnt"]
    sources = conn.execute(
        "SELECT source, COUNT(*) AS cnt FROM embeddings GROUP BY source ORDER BY source"
    ).fetchall()
    stale = conn.execute(
        "SELECT COUNT(*) AS cnt FROM embeddings WHERE stale_since IS NOT NULL"
    ).fetchone()["cnt"]

    last_sweep = get_last_sweep()
    last_sweep_at = last_sweep["swept_at"] if last_sweep else None
    orphan_count = last_sweep["orphans_deleted"] if last_sweep else 0

    return {
        "total_embeddings": total,
        "sources": {r["source"]: r["cnt"] for r in sources},
        "stale_count": stale,
        "orphan_count": orphan_count,
        "last_sweep_at": last_sweep_at,
        "last_sweep": last_sweep,
    }
