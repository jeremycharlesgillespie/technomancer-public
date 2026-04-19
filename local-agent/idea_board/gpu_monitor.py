"""GPU utilization logger — polls nvidia-smi every 5s, stores in SQLite.

Samples are written to ``data/gpu_metrics.db`` by a daemon thread started
from :func:`start_gpu_monitor`. The hub exposes ``/gpu`` and
``/api/gpu/metrics`` for a zoomable Chart.js view.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = _REPO_ROOT / "data" / "gpu_metrics.db"

POLL_INTERVAL_SECONDS: int = 5

RETENTION_DAYS: int = 30

_NVIDIA_SMI_ARGS: list[str] = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gpu_samples (
                ts INTEGER PRIMARY KEY,
                utilization REAL NOT NULL,
                mem_used_mb INTEGER NOT NULL,
                mem_total_mb INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gpu_samples_ts ON gpu_samples(ts)"
        )
        conn.commit()
    finally:
        conn.close()


def sample_gpu() -> tuple[float, int, int] | None:
    """Return (utilization%, mem_used_mb, mem_total_mb) or None on failure.

    Exits non-zero or missing nvidia-smi => None; caller logs and skips.
    """
    try:
        result = subprocess.run(
            _NVIDIA_SMI_ARGS,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi unavailable: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning("nvidia-smi returned %d: %s", result.returncode, result.stderr.strip())
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        logger.warning("nvidia-smi unexpected output: %r", line)
        return None
    try:
        util = float(parts[0])
        mem_used = int(parts[1])
        mem_total = int(parts[2])
    except ValueError as exc:
        logger.warning("nvidia-smi parse failed %r: %s", line, exc)
        return None
    return util, mem_used, mem_total


def record_sample(db_path: Path, ts: int, util: float, mem_used: int, mem_total: int) -> None:
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO gpu_samples(ts, utilization, mem_used_mb, mem_total_mb) VALUES (?, ?, ?, ?)",
            (ts, util, mem_used, mem_total),
        )
        conn.commit()
    finally:
        conn.close()


def prune_old(db_path: Path, now_ts: int, retention_days: int = RETENTION_DAYS) -> int:
    cutoff = now_ts - retention_days * 86400
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        cur = conn.execute("DELETE FROM gpu_samples WHERE ts < ?", (cutoff,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_samples(
    db_path: Path,
    start_ts: int | None = None,
    end_ts: int | None = None,
    max_points: int = 5000,
) -> list[dict]:
    """Return samples in [start_ts, end_ts] downsampled to at most max_points.

    Downsampling uses modulo stride so the chart stays responsive for
    multi-day zoom-outs. The latest sample is always included.
    """
    if not db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        clauses: list[str] = []
        params: list[int] = []
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(end_ts)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        total = conn.execute(
            f"SELECT COUNT(*) FROM gpu_samples{where}", tuple(params)
        ).fetchone()[0]
        if total == 0:
            return []

        stride = max(1, total // max_points)
        rows = conn.execute(
            f"""
            SELECT ts, utilization, mem_used_mb, mem_total_mb
            FROM gpu_samples{where}
            ORDER BY ts ASC
            """,
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    for idx, (ts, util, mu, mt) in enumerate(rows):
        if idx % stride != 0 and idx != len(rows) - 1:
            continue
        out.append({
            "ts": int(ts),
            "utilization": float(util),
            "mem_used_mb": int(mu),
            "mem_total_mb": int(mt),
        })
    return out


_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _poll_loop(db_path: Path, interval: int) -> None:
    _init_db(db_path)
    last_prune = 0
    while not _stop_event.is_set():
        sample = sample_gpu()
        if sample is not None:
            util, mu, mt = sample
            ts = int(time.time())
            try:
                record_sample(db_path, ts, util, mu, mt)
            except sqlite3.Error as exc:
                logger.warning("gpu_monitor DB write failed: %s", exc)

            if ts - last_prune > 3600:
                try:
                    prune_old(db_path, ts)
                except sqlite3.Error as exc:
                    logger.warning("gpu_monitor prune failed: %s", exc)
                last_prune = ts

        _stop_event.wait(interval)


def start_gpu_monitor(db_path: Path = DB_PATH, interval: int = POLL_INTERVAL_SECONDS) -> threading.Thread:
    global _thread
    if _thread is not None and _thread.is_alive():
        return _thread
    _stop_event.clear()
    _thread = threading.Thread(
        target=_poll_loop,
        args=(db_path, interval),
        daemon=True,
        name="gpu-monitor",
    )
    _thread.start()
    logger.info("GPU monitor started (interval=%ds, db=%s)", interval, db_path)
    return _thread


def stop_gpu_monitor() -> None:
    _stop_event.set()
