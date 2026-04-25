"""
AIV Schema — SQLite schema for the AI Validator (AIV) pipeline.

One file owns the ``CREATE TABLE`` statements for both AIV tables so
downstream modules (``aiv/main.py``, ``aiv/persist.py``, the executor
post-merge hook, the hub ``/quality`` route) never drift. :func:`init_db`
is the single entry point — callers invoke it lazily on first use.

Tables
------

``aiv_pending`` — the post-merge validation queue. One row per shipped
story awaiting scoring; deleted by ``aiv/persist.py`` after the
companion ``story_quality`` row is written.

``story_quality`` — durable record of the seven-axis quality score for
every validated story. One row per story_key (UPSERT on re-validation).
The seven integer columns match the axis names referenced in the AIV
epic: ``meets_requirements``, ``code_quality``, ``test_quality``,
``security_safety``, ``scope_discipline``, ``edge_cases``,
``product_impact``.

Database: ``local-agent/data/aiv.db``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "aiv.db"

# Column name whitelist for the seven quality axes. Importers use this
# when building INSERT / UPDATE statements so a typo at the caller doesn't
# silently write to a non-existent column.
SCORE_COLUMNS: tuple[str, ...] = (
    "meets_requirements",
    "code_quality",
    "test_quality",
    "security_safety",
    "scope_discipline",
    "edge_cases",
    "product_impact",
)

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection (created on first use)."""
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
    """Create ``aiv_pending`` and ``story_quality`` if they don't exist.

    Idempotent — safe to call on every daemon cycle / before every insert.
    Also runs additive ``ALTER TABLE`` migrations for columns introduced
    after the original schema landed (``merge_commit_sha``,
    ``verification_output``).
    """
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aiv_pending (
            story_key            TEXT PRIMARY KEY,
            merged_at            TEXT,
            diff_paths_json      TEXT,
            enqueued_at          TEXT,
            merge_commit_sha     TEXT,
            verification_output  TEXT
        )
    """)
    # Additive migration for databases created before merge_commit_sha
    # and verification_output existed. PRAGMA table_info is the simplest
    # way to ask "does this column already exist" — sqlite has no
    # ``ADD COLUMN IF NOT EXISTS``.
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(aiv_pending)")
    }
    if "merge_commit_sha" not in existing_cols:
        conn.execute("ALTER TABLE aiv_pending ADD COLUMN merge_commit_sha TEXT")
    if "verification_output" not in existing_cols:
        conn.execute("ALTER TABLE aiv_pending ADD COLUMN verification_output TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS story_quality (
            story_key            TEXT PRIMARY KEY,
            story_title          TEXT,
            merged_at            TEXT,
            validated_at         TEXT,
            meets_requirements   INTEGER,
            code_quality         INTEGER,
            test_quality         INTEGER,
            security_safety      INTEGER,
            scope_discipline     INTEGER,
            edge_cases           INTEGER,
            product_impact       INTEGER,
            overall_score        REAL,
            red_flags_json       TEXT,
            verification_method  TEXT,
            verification_output  TEXT,
            reasoning_json       TEXT,
            error                TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_story_quality_validated_at
        ON story_quality (validated_at DESC)
    """)
    conn.commit()

    # The A/B model-comparison harness reuses this database. Importing
    # here (not at module top) avoids a circular import — agent.ab_schema
    # imports _get_conn from this module.
    from agent.ab_schema import init_ab_db
    init_ab_db()
