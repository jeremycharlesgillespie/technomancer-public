"""
Metrics Aggregator — Unified observability snapshot for the idea board +
executor + Claude vault + Ollama health.

Two public entry points:

* :func:`get_snapshot` — returns a structured dict that bundles together
  executor success rate, p50/p95 latency over the last 24 hours, Claude
  vault prompt-cache hit rate, board queue depth + oldest top-ranked
  wait, and Ollama health. Each call is served from a 30-second in-memory
  cache (guarded by a :class:`threading.Lock`) so cheap polling endpoints
  can hammer it without re-hitting SQLite or Jira on every request.
* :func:`render_prometheus` — renders the same payload as a plain-text
  Prometheus exposition. We deliberately avoid importing
  ``prometheus_client`` so this module stays usable in environments where
  it isn't installed (and on first import).

Failure handling: if any underlying source raises, we fall back to the
last successful payload and add a ``stale_seconds`` field so consumers
can tell how old the data is. ``stale_seconds`` is ``0`` when the
snapshot is fresh.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Any

from . import claude_vault, executor_runs_db, ollama_health
from board import get_provider

log = logging.getLogger(__name__)

# Cache TTL — anything older than this is recomputed on the next call.
CACHE_TTL_SECONDS: float = 30.0

# Window used for executor success / latency aggregation.
EXECUTOR_WINDOW_HOURS: int = 24

# Terminal-but-counted-as-success statuses for the executor.
_SUCCESS_STATUSES: frozenset[str] = frozenset({"success", "deployed"})

# Idea states that count toward the "queue depth" metric — items the
# executor has not picked up yet but that are eligible to run.
_QUEUE_STATES: frozenset[str] = frozenset({"proposed", "approved"})

# Idea state used for "oldest top-ranked wait" — the next thing the
# executor should pick up. Approved items have made it through review,
# so the oldest one waiting tells us how badly the queue is backed up.
_TOP_RANKED_STATE: str = "approved"


# ---------------------------------------------------------------------------
# Cache state — guarded by _cache_lock
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_payload: dict[str, Any] | None = None
_cached_at: float = 0.0


def _reset_cache() -> None:
    """Drop the cached payload. Tests use this between cases."""
    global _cached_payload, _cached_at
    with _cache_lock:
        _cached_payload = None
        _cached_at = 0.0


# ---------------------------------------------------------------------------
# Source aggregators
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` using nearest-rank.

    Returns ``0.0`` for empty input so callers don't have to special-case
    a fresh database. ``pct`` is in the range ``[0, 100]``.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct <= 0:
        return float(ordered[0])
    if pct >= 100:
        return float(ordered[-1])
    # Nearest-rank: index = ceil(pct/100 * N) - 1, clamped to [0, N-1].
    rank = max(1, int((pct / 100.0) * len(ordered) + 0.999999))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def _executor_metrics() -> dict[str, Any]:
    """Aggregate executor stats over the last ``EXECUTOR_WINDOW_HOURS`` hours.

    Reads ``executor_runs_db`` directly so this module owns the SQL — the
    DB module exposes ``get_recent`` but no time-windowed aggregator and
    we don't want callers paying for an unbounded fetch.
    """
    executor_runs_db.init_db()
    conn = executor_runs_db._get_conn()
    cutoff = (
        datetime.now() - timedelta(hours=EXECUTOR_WINDOW_HOURS)
    ).isoformat(sep=" ", timespec="seconds")

    rows = conn.execute(
        """SELECT status, duration_ms
           FROM executor_runs
           WHERE started_at IS NOT NULL
             AND datetime(started_at) >= datetime(?)""",
        (cutoff,),
    ).fetchall()

    total = len(rows)
    successes = 0
    durations: list[float] = []
    for r in rows:
        status = (r["status"] or "").lower()
        if status in _SUCCESS_STATUSES:
            successes += 1
        # Only count completed runs in the latency distribution. A row
        # with no duration is still in flight or never wrote one.
        if r["duration_ms"] is not None:
            durations.append(float(r["duration_ms"]))

    success_rate = (successes / total) if total else 0.0
    return {
        "total_runs_24h": total,
        "successes_24h": successes,
        "success_rate": success_rate,
        "p50_latency_ms": _percentile(durations, 50),
        "p95_latency_ms": _percentile(durations, 95),
    }


def _claude_vault_metrics() -> dict[str, Any]:
    """Pull cumulative prompt-cache stats from claude_vault."""
    stats = claude_vault.get_cache_stats()
    return {
        "calls": int(stats.get("calls", 0)),
        "cache_hit_rate": float(stats.get("cache_hit_rate", 0.0)),
        "input_tokens": int(stats.get("input_tokens", 0)),
        "cache_read_tokens": int(stats.get("cache_read_tokens", 0)),
        "cache_creation_tokens": int(stats.get("cache_creation_tokens", 0)),
    }


def _board_metrics() -> dict[str, Any]:
    """Inspect the board for queue depth + the oldest waiting top item.

    "Oldest top-ranked wait" is reported as the age in seconds of the
    earliest-created item still in the ``approved`` state. The local
    backend has no explicit rank field so creation order is the closest
    proxy; with the Jira backend the same heuristic still surfaces the
    item that has been waiting the longest, which is what an operator
    actually wants to see.
    """
    provider = get_provider()
    items = provider.load_all()

    queue_depth = 0
    oldest_top_iso: str | None = None
    for item in items:
        state = getattr(item, "state", "")
        if state in _QUEUE_STATES:
            queue_depth += 1
        if state == _TOP_RANKED_STATE:
            created = getattr(item, "created", "") or ""
            if created and (oldest_top_iso is None or created < oldest_top_iso):
                oldest_top_iso = created

    oldest_wait_seconds: float | None = None
    if oldest_top_iso:
        try:
            oldest_wait_seconds = max(
                0.0,
                (datetime.now() - datetime.fromisoformat(oldest_top_iso)).total_seconds(),
            )
        except ValueError:
            # Bad timestamp on disk — record None rather than raising,
            # the metrics layer must not blow up on a single bad row.
            oldest_wait_seconds = None

    return {
        "queue_depth": queue_depth,
        "oldest_top_ranked_wait_seconds": oldest_wait_seconds,
    }


def _ollama_metrics() -> dict[str, Any]:
    """Pass through the Ollama health snapshot."""
    return ollama_health.get_ollama_status()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_payload() -> dict[str, Any]:
    """Gather every source. Raises if any source fails."""
    return {
        "executor": _executor_metrics(),
        "claude_vault": _claude_vault_metrics(),
        "board": _board_metrics(),
        "ollama": _ollama_metrics(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stale_seconds": 0.0,
    }


def get_snapshot(force_refresh: bool = False) -> dict[str, Any]:
    """Return the current metrics snapshot, served from a 30-second cache.

    Args:
        force_refresh: Bypass the cache and rebuild from the live sources.
            Use sparingly — the cache exists so polling endpoints don't
            slam SQLite and Jira on every request.

    Returns:
        A dict with ``executor``, ``claude_vault``, ``board``, ``ollama``,
        ``generated_at``, and ``stale_seconds`` keys. ``stale_seconds`` is
        ``0`` when the snapshot was just rebuilt, the age of the last good
        payload (in seconds) when a source failed and we fell back, or the
        TTL-bounded age when served from cache.
    """
    global _cached_payload, _cached_at

    now = _time.monotonic()
    with _cache_lock:
        if (
            not force_refresh
            and _cached_payload is not None
            and (now - _cached_at) < CACHE_TTL_SECONDS
        ):
            payload = dict(_cached_payload)
            payload["stale_seconds"] = round(now - _cached_at, 3)
            return payload

    # Refresh outside the lock so the (potentially slow) source calls
    # don't block other readers from getting the cached payload.
    try:
        fresh = _build_payload()
    except Exception as exc:
        log.warning("metrics: source refresh failed (%s) — serving stale", exc)
        with _cache_lock:
            if _cached_payload is not None:
                stale = round(_time.monotonic() - _cached_at, 3)
                payload = dict(_cached_payload)
                payload["stale_seconds"] = stale
                return payload
        # No prior good payload — return an explicitly-empty shell so
        # callers (and tests) can still introspect the failure mode.
        return {
            "executor": {},
            "claude_vault": {},
            "board": {},
            "ollama": {},
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stale_seconds": None,
            "error": str(exc),
        }

    with _cache_lock:
        _cached_payload = fresh
        _cached_at = _time.monotonic()
        return dict(fresh)


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------


def _fmt_metric(name: str, mtype: str, help_text: str, value: Any) -> list[str]:
    """Format one Prometheus metric (HELP, TYPE, sample) as text lines."""
    if value is None:
        # Prometheus exposition has no first-class "absent" — emit NaN so
        # the time series exists but visibly indicates no data.
        rendered = "NaN"
    elif isinstance(value, bool):
        rendered = "1" if value else "0"
    elif isinstance(value, (int, float)):
        rendered = repr(float(value))
    else:
        rendered = "0"
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} {mtype}",
        f"{name} {rendered}",
    ]


def render_prometheus() -> str:
    """Render :func:`get_snapshot` as plain-text Prometheus exposition.

    Doesn't require ``prometheus_client`` at import or call time — this
    is plain text so the metrics module stays useful in environments
    where the optional dependency is missing.
    """
    snap = get_snapshot()
    executor = snap.get("executor") or {}
    vault = snap.get("claude_vault") or {}
    board = snap.get("board") or {}
    ollama = snap.get("ollama") or {}

    lines: list[str] = []

    lines += _fmt_metric(
        "technomancer_executor_success_rate",
        "gauge",
        "Fraction of executor runs in the last 24h that ended in a success state.",
        executor.get("success_rate", 0.0),
    )
    lines += _fmt_metric(
        "technomancer_executor_runs_24h",
        "gauge",
        "Number of executor runs started in the last 24h.",
        executor.get("total_runs_24h", 0),
    )
    lines += _fmt_metric(
        "technomancer_executor_latency_p50_ms",
        "gauge",
        "p50 executor run duration in milliseconds over the last 24h.",
        executor.get("p50_latency_ms", 0.0),
    )
    lines += _fmt_metric(
        "technomancer_executor_latency_p95_ms",
        "gauge",
        "p95 executor run duration in milliseconds over the last 24h.",
        executor.get("p95_latency_ms", 0.0),
    )

    lines += _fmt_metric(
        "technomancer_claude_vault_cache_hit_rate",
        "gauge",
        "Cumulative cache_read / total input ratio for Claude vault calls.",
        vault.get("cache_hit_rate", 0.0),
    )
    lines += _fmt_metric(
        "technomancer_claude_vault_calls_total",
        "counter",
        "Cumulative Claude vault API calls observed since process start.",
        vault.get("calls", 0),
    )

    lines += _fmt_metric(
        "technomancer_board_queue_depth",
        "gauge",
        "Number of board items in proposed or approved states.",
        board.get("queue_depth", 0),
    )
    lines += _fmt_metric(
        "technomancer_board_oldest_top_ranked_wait_seconds",
        "gauge",
        "Age in seconds of the oldest approved-but-not-executing board item.",
        board.get("oldest_top_ranked_wait_seconds"),
    )

    # Ollama: status is a string — emit a 1/0 indicator per known status.
    status = (ollama.get("status") or "unknown").lower()
    for candidate in ("healthy", "degraded", "down", "unknown"):
        lines += _fmt_metric(
            f"technomancer_ollama_status_{candidate}",
            "gauge",
            f"1 when Ollama health is '{candidate}', 0 otherwise.",
            1 if status == candidate else 0,
        )
    lines += _fmt_metric(
        "technomancer_ollama_consecutive_failures",
        "gauge",
        "Consecutive failed Ollama health probes since the last success.",
        ollama.get("consecutive_failures", 0),
    )

    lines += _fmt_metric(
        "technomancer_metrics_stale_seconds",
        "gauge",
        "Age of the served metrics payload in seconds; 0 means freshly built.",
        snap.get("stale_seconds") or 0.0,
    )

    return "\n".join(lines) + "\n"
