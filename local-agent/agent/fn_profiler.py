"""
Function Profiler — per-function wall-clock timing collection.

Provides the `@profile_fn` decorator plus durable SQLite persistence so
per-function latency can be observed across restarts and later exposed via
/api/perf/functions for the dashboard and automated optimization stories.

Design:
- Decorator wraps sync or async functions and records duration on exit.
- An in-memory registry keyed by qualified function name accumulates
  call counts, total wall time, and the last 100 durations for percentiles.
- Access to the registry is guarded by a threading.Lock since the executor
  and AIM can call decorated functions concurrently.
- A shared sqlite3 connection (check_same_thread=False, WAL) persists the
  registry via UPSERT. `flush_stats()` writes, `load_stats()` restores.
- `start_background_flush()` spins up a daemon Timer that flushes every N
  seconds (5 minutes by default) while the process lives.

Database: ``local-agent/profiling/fn_stats.db``.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "profiling"
DB_PATH = DB_DIR / "fn_stats.db"

MAX_DURATIONS = 100
FLUSH_INTERVAL_SECONDS = 300.0  # 5 minutes

F = TypeVar("F", bound=Callable[..., Any])

_registry: dict[str, "FnStats"] = {}
_lock = threading.Lock()

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()

_flush_timer: threading.Timer | None = None
_flush_running: bool = False
_timer_lock = threading.Lock()


class FnStats:
    """Accumulated timing stats for one profiled function.

    Stores the full running call count and total wall time, plus a bounded
    deque of the last MAX_DURATIONS durations used to compute p50/p95 and
    variance without retaining every sample.
    """

    __slots__ = ("name", "call_count", "total_seconds", "last_n_durations")

    def __init__(
        self,
        name: str,
        call_count: int = 0,
        total_seconds: float = 0.0,
        last_n_durations: list[float] | None = None,
    ) -> None:
        """Initialize stats, optionally seeded from previously-flushed data."""
        self.name = name
        self.call_count = call_count
        self.total_seconds = total_seconds
        self.last_n_durations: deque[float] = deque(
            last_n_durations or [], maxlen=MAX_DURATIONS
        )

    def record(self, duration: float) -> None:
        """Add one call's duration to the stats."""
        self.call_count += 1
        self.total_seconds += duration
        self.last_n_durations.append(duration)

    def p50(self) -> float:
        """Return the 50th-percentile duration over the last-N samples."""
        return _percentile(list(self.last_n_durations), 50.0)

    def p95(self) -> float:
        """Return the 95th-percentile duration over the last-N samples."""
        return _percentile(list(self.last_n_durations), 95.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stats to a JSON-friendly dict."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_seconds": self.total_seconds,
            "p50_seconds": self.p50(),
            "p95_seconds": self.p95(),
            "last_n_durations": list(self.last_n_durations),
        }


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy.percentile default.

    numpy.percentile with the default ``linear`` interpolation computes:
      pos   = (n - 1) * pct / 100
      lower = floor(pos); upper = lower + 1
      frac  = pos - lower
      result = values[lower] + frac * (values[upper] - values[lower])
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    frac = pos - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def _get_conn() -> sqlite3.Connection:
    """Return the shared SQLite connection, opening it on first use."""
    global _conn
    with _conn_lock:
        if _conn is None:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(
                str(DB_PATH), timeout=5, check_same_thread=False
            )
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
        return _conn


def init_db() -> None:
    """Create the fn_stats table if it doesn't exist."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fn_stats (
            name              TEXT PRIMARY KEY,
            call_count        INTEGER NOT NULL DEFAULT 0,
            total_seconds     REAL    NOT NULL DEFAULT 0.0,
            p50_seconds       REAL    NOT NULL DEFAULT 0.0,
            p95_seconds       REAL    NOT NULL DEFAULT 0.0,
            last_n_durations  TEXT    NOT NULL DEFAULT '[]',
            updated_at        TEXT    NOT NULL
        )
        """
    )
    conn.commit()


def _record(name: str, duration: float) -> None:
    """Update the in-memory registry with one new sample."""
    with _lock:
        stats = _registry.get(name)
        if stats is None:
            stats = FnStats(name)
            _registry[name] = stats
        stats.record(duration)


def profile_fn(func: F) -> F:
    """Decorator that records wall-clock duration per call.

    Works on both sync and async functions. Overhead is one
    ``time.perf_counter()`` call at enter and exit plus a single dict
    update under a lock — micro relative to the bodies we instrument.
    """
    name = f"{func.__module__}.{func.__qualname__}"

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                _record(name, time.perf_counter() - start)

        async_wrapper.__wrapped_fn_name__ = name  # type: ignore[attr-defined]
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _record(name, time.perf_counter() - start)

    wrapper.__wrapped_fn_name__ = name  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def get_stats(name: str) -> FnStats | None:
    """Return stats for a given qualified function name, or None."""
    with _lock:
        return _registry.get(name)


def all_stats() -> dict[str, FnStats]:
    """Return a shallow copy of the in-memory registry."""
    with _lock:
        return dict(_registry)


def reset_registry() -> None:
    """Clear the in-memory registry. Primarily for tests."""
    with _lock:
        _registry.clear()


def flush_stats() -> int:
    """Persist the in-memory registry to SQLite via UPSERT.

    Returns the number of rows written.
    """
    init_db()
    with _lock:
        snapshot = [
            (
                name,
                s.call_count,
                s.total_seconds,
                s.p50(),
                s.p95(),
                list(s.last_n_durations),
            )
            for name, s in _registry.items()
        ]
    if not snapshot:
        return 0

    now = datetime.now().isoformat(timespec="seconds")
    rows = [
        (name, call_count, total_seconds, p50, p95, json.dumps(durations), now)
        for (name, call_count, total_seconds, p50, p95, durations) in snapshot
    ]
    conn = _get_conn()
    conn.executemany(
        """INSERT INTO fn_stats
               (name, call_count, total_seconds, p50_seconds,
                p95_seconds, last_n_durations, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               call_count       = excluded.call_count,
               total_seconds    = excluded.total_seconds,
               p50_seconds      = excluded.p50_seconds,
               p95_seconds      = excluded.p95_seconds,
               last_n_durations = excluded.last_n_durations,
               updated_at       = excluded.updated_at""",
        rows,
    )
    conn.commit()
    return len(rows)


def load_stats() -> int:
    """Load persisted stats into the in-memory registry.

    Returns the number of rows loaded. Existing entries in the registry
    with the same name are replaced.
    """
    init_db()
    conn = _get_conn()
    cur = conn.execute(
        "SELECT name, call_count, total_seconds, last_n_durations FROM fn_stats"
    )
    rows = cur.fetchall()
    with _lock:
        for name, call_count, total_seconds, last_n_durations in rows:
            try:
                durations = json.loads(last_n_durations) if last_n_durations else []
            except (json.JSONDecodeError, TypeError):
                durations = []
            _registry[name] = FnStats(
                name=name,
                call_count=int(call_count),
                total_seconds=float(total_seconds),
                last_n_durations=list(durations),
            )
    return len(rows)


def snapshot_from_db() -> dict[str, FnStats]:
    """Read every persisted row and return fresh FnStats, untouched by the registry.

    Callers that just want to read current persisted stats (e.g. the
    ``/api/perf/functions`` endpoint) should use this instead of
    ``load_stats`` — it keeps the in-memory registry untouched so live
    counters that haven't been flushed yet aren't overwritten.
    """
    init_db()
    conn = _get_conn()
    cur = conn.execute(
        "SELECT name, call_count, total_seconds, last_n_durations FROM fn_stats"
    )
    result: dict[str, FnStats] = {}
    for name, call_count, total_seconds, last_n_durations in cur.fetchall():
        try:
            durations = json.loads(last_n_durations) if last_n_durations else []
        except (json.JSONDecodeError, TypeError):
            durations = []
        result[name] = FnStats(
            name=name,
            call_count=int(call_count),
            total_seconds=float(total_seconds),
            last_n_durations=list(durations),
        )
    return result


def get_collected_since() -> str | None:
    """Return the earliest ``updated_at`` in the fn_stats table, or None if empty.

    Acts as a proxy for "we've been collecting at least since this time"
    so consumers know how fresh / old the aggregate picture is.
    """
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT MIN(updated_at) FROM fn_stats").fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _flush_tick(interval: float) -> None:
    """Timer callback: flush once, then re-arm if still running."""
    global _flush_timer
    try:
        flush_stats()
    except Exception:
        log.exception("fn_profiler flush failed")
    with _timer_lock:
        if not _flush_running:
            return
        next_timer = threading.Timer(interval, _flush_tick, args=(interval,))
        next_timer.daemon = True
        _flush_timer = next_timer
        next_timer.start()


def start_background_flush(interval: float = FLUSH_INTERVAL_SECONDS) -> None:
    """Start a daemon Timer that flushes the registry every `interval` seconds.

    Idempotent: calling again while a timer is running is a no-op.
    """
    global _flush_running, _flush_timer
    with _timer_lock:
        if _flush_running:
            return
        _flush_running = True
        timer = threading.Timer(interval, _flush_tick, args=(interval,))
        timer.daemon = True
        _flush_timer = timer
        timer.start()


def stop_background_flush() -> None:
    """Stop the background flush timer if running."""
    global _flush_running, _flush_timer
    with _timer_lock:
        _flush_running = False
        if _flush_timer is not None:
            _flush_timer.cancel()
            _flush_timer = None


def _reset_connection() -> None:
    """Test helper: close and clear the cached SQLite connection."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
