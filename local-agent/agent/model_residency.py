"""
Model Residency Tracker — SQLite database for tracking when Ollama models are loaded/unloaded.

This module tracks model residency events to enable cross-referencing with GPU utilization
data for visualizing VRAM thrashing patterns.

Database lives at ``local-agent/data/model_residency.db``.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Dict, Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "model_residency.db"

_local = threading.local()


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
    """Create the model residency table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_residency (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   INTEGER NOT NULL,
            model_name  TEXT    NOT NULL,
            action      TEXT    NOT NULL,  -- 'load' or 'unload'
            mem_used_mb INTEGER,
            mem_total_mb INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_residency_timestamp ON model_residency(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_residency_model ON model_residency(model_name)")
    conn.commit()


def record_model_event(model_name: str, action: str, mem_used_mb: int = 0, mem_total_mb: int = 0) -> None:
    """Record a model load/unload event."""
    if action not in ('load', 'unload'):
        raise ValueError("Action must be 'load' or 'unload'")
    
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO model_residency(timestamp, model_name, action, mem_used_mb, mem_total_mb) VALUES (?, ?, ?, ?, ?)",
            (int(time.time()), model_name, action, mem_used_mb, mem_total_mb)
        )
        conn.commit()
    except Exception as e:
        log.warning("Failed to record model event for %s: %s", model_name, e)


def get_model_timeline(start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
    """Get model residency timeline for the specified time range.
    
    Returns a list of events with timestamps, model names, and actions.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT timestamp, model_name, action, mem_used_mb, mem_total_mb
            FROM model_residency
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (start_ts, end_ts)).fetchall()
        
        return [
            {
                "timestamp": row["timestamp"],
                "model_name": row["model_name"],
                "action": row["action"],
                "mem_used_mb": row["mem_used_mb"],
                "mem_total_mb": row["mem_total_mb"]
            }
            for row in rows
        ]
    except Exception as e:
        log.warning("Failed to fetch model timeline: %s", e)
        return []


def get_resident_models_at_timestamp(timestamp: int) -> List[str]:
    """Get list of models that were resident at the given timestamp."""
    conn = _get_conn()
    try:
        # Find all models that were loaded before this timestamp and not yet unloaded
        rows = conn.execute("""
            SELECT DISTINCT model_name
            FROM model_residency
            WHERE timestamp <= ? AND model_name IN (
                SELECT model_name FROM model_residency 
                WHERE timestamp <= ? AND action = 'load'
                EXCEPT
                SELECT model_name FROM model_residency 
                WHERE timestamp <= ? AND action = 'unload'
            )
        """, (timestamp, timestamp, timestamp)).fetchall()
        
        return [row["model_name"] for row in rows]
    except Exception as e:
        log.warning("Failed to get resident models at timestamp %d: %s", timestamp, e)
        return []


def prune_old_data(retention_days: int = 30) -> int:
    """Remove old residency records to prevent database bloat."""
    cutoff = int(time.time()) - retention_days * 86400
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM model_residency WHERE timestamp < ?", (cutoff,))
        conn.commit()
        return cur.rowcount
    except Exception as e:
        log.warning("Failed to prune old model residency data: %s", e)
        return 0