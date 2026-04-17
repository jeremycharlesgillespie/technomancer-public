"""
News Preferences DB — SQLite persistence for news digest preferences.

Replaces the flat ``idea_board/news_config.json`` file with a single-row
``news_prefs`` SQLite table. Provides the same raw dict in/out that the
JSON layer did so ``idea_board.news_config`` can keep the ``NewsConfig``
dataclass as the public API unchanged.

On first access, if the legacy ``news_config.json`` still exists, its
contents are imported into the DB and the file is renamed with a
``.migrated.bak`` suffix so the migration only runs once.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR: Path = Path(__file__).parent.parent / "data"
DB_PATH: Path = DB_DIR / "news_prefs.db"
LEGACY_JSON_PATH: Path = Path(__file__).parent.parent / "idea_board" / "news_config.json"
LEGACY_BACKUP_SUFFIX: str = ".migrated.bak"

_local = threading.local()
_migration_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (created on first use)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create the ``news_prefs`` table if it doesn't exist."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_prefs (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            data        TEXT    NOT NULL DEFAULT '{}',
            updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def load_prefs() -> dict[str, Any] | None:
    """Return the stored preferences dict, or ``None`` if no row exists yet.

    Callers (``idea_board.news_config``) are expected to fall back to
    defaults when this returns ``None``.
    """
    init_db()
    _migrate_legacy_json_if_needed()
    conn = _get_conn()
    row = conn.execute("SELECT data FROM news_prefs WHERE id = 1").fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row["data"])
    except json.JSONDecodeError as e:
        log.error("Failed to decode news_prefs JSON payload: %s", e)
        return None
    if not isinstance(data, dict):
        log.error("news_prefs payload is not a dict, got %s", type(data).__name__)
        return None
    return data


def save_prefs(data: dict[str, Any]) -> None:
    """Upsert the preferences dict into the singleton row."""
    init_db()
    conn = _get_conn()
    payload = json.dumps(data, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO news_prefs (id, data, updated_at)
        VALUES (1, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            data = excluded.data,
            updated_at = datetime('now')
        """,
        (payload,),
    )
    conn.commit()


def _migrate_legacy_json_if_needed() -> None:
    """One-time import of ``news_config.json`` into the DB.

    Runs at most once per process. If the DB already has a row or the
    legacy file is missing, this is a no-op. On successful import the
    JSON file is renamed with ``.migrated.bak`` so the next restart
    doesn't try to migrate again.
    """
    with _migration_lock:
        conn = _get_conn()
        existing = conn.execute("SELECT 1 FROM news_prefs WHERE id = 1").fetchone()
        if existing is not None:
            return
        if not LEGACY_JSON_PATH.exists():
            return
        try:
            raw = LEGACY_JSON_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                log.error(
                    "Legacy news_config.json is not a JSON object (%s) — skipping migration",
                    type(data).__name__,
                )
                return
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to read legacy news_config.json for migration: %s", e)
            return

        payload = json.dumps(data, ensure_ascii=False)
        try:
            conn.execute(
                "INSERT INTO news_prefs (id, data) VALUES (1, ?)",
                (payload,),
            )
            conn.commit()
        except sqlite3.DatabaseError as e:
            log.error("Failed to write migrated news prefs to DB: %s", e)
            return

        backup_path = LEGACY_JSON_PATH.with_name(LEGACY_JSON_PATH.name + LEGACY_BACKUP_SUFFIX)
        try:
            if backup_path.exists():
                backup_path.unlink()
            LEGACY_JSON_PATH.rename(backup_path)
            log.info(
                "Migrated news prefs from %s to SQLite; legacy file renamed to %s",
                LEGACY_JSON_PATH.name,
                backup_path.name,
            )
        except OSError as e:
            log.warning(
                "Migrated news prefs to DB but couldn't rename legacy file %s: %s",
                LEGACY_JSON_PATH,
                e,
            )
