"""
LLM Performance Optimizer — Cost analytics, query complexity scoring,
response caching, and context decay.

Builds on the existing profiling infrastructure (perf_monitor, metrics_db,
prometheus_metrics, profiler, claude_vault) to add the missing pieces:

1. Combined cost/usage analytics dashboard (stories idea-060, idea-107)
2. Query complexity scoring for routing decisions (stories idea-072, idea-083)
3. Lightweight response cache for repeated queries (story idea-091)
4. Context relevance decay scoring (story idea-067)
5. Latency threshold alerts (story idea-068)

Covers epic stories: idea-060, idea-064, idea-067, idea-068, idea-072,
idea-076, idea-077, idea-083, idea-091, idea-095, idea-107.
"""

import hashlib
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DB_PATH = DATA_DIR / "response_cache.db"

_local = threading.local()

# Latency alert threshold (seconds)
LATENCY_ALERT_THRESHOLD = 30.0


# ---------------------------------------------------------------------------
# Response cache (story idea-091) — hash-based, no embeddings needed
# ---------------------------------------------------------------------------

def _get_cache_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "cache_conn", None)
    if conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.cache_conn = conn
    return conn


def init_cache_db() -> None:
    conn = _get_cache_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS response_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash  TEXT NOT NULL UNIQUE,
            query_text  TEXT NOT NULL,
            response    TEXT NOT NULL,
            endpoint    TEXT NOT NULL DEFAULT 'ollama',
            model       TEXT NOT NULL DEFAULT '',
            tokens_saved INTEGER NOT NULL DEFAULT 0,
            hit_count   INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            last_hit_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_rc_hash ON response_cache (query_hash);
    """)
    conn.commit()


def _normalize_query(query: str) -> str:
    """Normalize a query for cache matching — lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", query.lower().strip())


def _hash_query(query: str) -> str:
    """Generate a cache key from a normalized query."""
    return hashlib.sha256(_normalize_query(query).encode()).hexdigest()[:32]


def cache_lookup(query: str, max_age_hours: int = 1) -> str | None:
    """Check the response cache for a matching query.

    Returns the cached response or None if not found / expired.
    """
    init_cache_db()
    conn = _get_cache_conn()
    qhash = _hash_query(query)
    cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

    row = conn.execute(
        "SELECT id, response, tokens_saved FROM response_cache WHERE query_hash = ? AND created_at >= ?",
        (qhash, cutoff),
    ).fetchone()

    if row:
        conn.execute(
            "UPDATE response_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (datetime.now().isoformat(), row["id"]),
        )
        conn.commit()
        return row["response"]
    return None


def cache_store(query: str, response: str, endpoint: str = "ollama", model: str = "", tokens: int = 0) -> None:
    """Store a response in the cache."""
    if not response or len(response) < 20:
        return  # Don't cache empty/trivial responses

    init_cache_db()
    conn = _get_cache_conn()
    qhash = _hash_query(query)
    now = datetime.now().isoformat()

    conn.execute(
        """INSERT OR REPLACE INTO response_cache
           (query_hash, query_text, response, endpoint, model, tokens_saved, hit_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
        (qhash, _normalize_query(query)[:200], response[:5000], endpoint, model, tokens, now),
    )
    conn.commit()


def get_cache_stats() -> dict[str, Any]:
    """Get response cache statistics."""
    init_cache_db()
    conn = _get_cache_conn()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM response_cache").fetchone()["cnt"]
    hits = conn.execute("SELECT SUM(hit_count) AS s FROM response_cache").fetchone()["s"] or 0
    tokens = conn.execute("SELECT SUM(tokens_saved * hit_count) AS s FROM response_cache").fetchone()["s"] or 0
    return {"cached_queries": total, "total_hits": hits, "tokens_saved": tokens}


# ---------------------------------------------------------------------------
# Query complexity scoring (stories idea-072, idea-083)
# ---------------------------------------------------------------------------

def score_query_complexity(query: str) -> dict[str, Any]:
    """Score a query's complexity to inform routing decisions.

    Returns a dict with:
        complexity: "simple" | "moderate" | "complex"
        score: 0.0 - 1.0
        suggested_model: which model tier would be appropriate
        reasoning: why this score was assigned
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)

    score = 0.0
    reasons: list[str] = []

    # Length-based scoring
    if word_count <= 5:
        score += 0.1
        reasons.append("short query")
    elif word_count <= 20:
        score += 0.3
        reasons.append("moderate length")
    else:
        score += 0.5
        reasons.append(f"long query ({word_count} words)")

    # Complexity indicators
    complex_patterns = [
        (r"\b(explain|analyze|compare|contrast|evaluate)\b", 0.2, "analytical"),
        (r"\b(step.by.step|how.to|implement|build|create)\b", 0.15, "procedural"),
        (r"\b(why|because|therefore|however|although)\b", 0.1, "reasoning"),
        (r"\b(code|function|class|algorithm|debug)\b", 0.15, "code-related"),
        (r"\b(and|also|additionally|furthermore)\b", 0.05, "multi-part"),
    ]
    for pattern, weight, reason in complex_patterns:
        if re.search(pattern, query_lower):
            score += weight
            reasons.append(reason)

    # Simple fact patterns (reduce complexity)
    simple_patterns = [
        r"^(what|where|who|when) (is|was|are|were) ",
        r"^(what time|what date|how old|how tall|how far)",
        r"^(define|meaning of|definition of)",
    ]
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            score = max(score - 0.3, 0.0)
            reasons.append("simple factual")
            break

    score = min(score, 1.0)

    if score < 0.3:
        complexity = "simple"
        suggested = "small model (fast, cheap)"
    elif score < 0.6:
        complexity = "moderate"
        suggested = "standard model"
    else:
        complexity = "complex"
        suggested = "large model or Claude escalation"

    return {
        "complexity": complexity,
        "score": round(score, 2),
        "suggested_model": suggested,
        "reasoning": ", ".join(reasons),
    }


# ---------------------------------------------------------------------------
# Context relevance decay (story idea-067)
# ---------------------------------------------------------------------------

def compute_context_decay(age_minutes: float, half_life_minutes: float = 60.0) -> float:
    """Compute a relevance decay factor for context based on age.

    Uses exponential decay: relevance = 2^(-age / half_life)

    Args:
        age_minutes: How old the context is (in minutes)
        half_life_minutes: Time for relevance to halve (default 60 min)

    Returns:
        Relevance factor between 0.0 and 1.0
    """
    if age_minutes <= 0:
        return 1.0
    return 2.0 ** (-age_minutes / half_life_minutes)


def rank_context_by_relevance(
    items: list[dict[str, Any]],
    timestamp_key: str = "timestamp",
    min_relevance: float = 0.1,
) -> list[dict[str, Any]]:
    """Rank context items by time-decayed relevance.

    Each item should have a timestamp field. Items below min_relevance
    are filtered out. Returns items sorted by relevance (highest first).
    """
    now = datetime.now()
    scored: list[tuple[float, dict[str, Any]]] = []

    for item in items:
        ts_str = item.get(timestamp_key, "")
        try:
            ts = datetime.fromisoformat(ts_str)
            age_minutes = (now - ts).total_seconds() / 60
        except (ValueError, TypeError):
            age_minutes = 24 * 60  # default to 24h old

        relevance = compute_context_decay(age_minutes)
        if relevance >= min_relevance:
            item_copy = dict(item)
            item_copy["_relevance"] = round(relevance, 3)
            scored.append((relevance, item_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


# ---------------------------------------------------------------------------
# Combined cost/usage analytics (stories idea-060, idea-107)
# ---------------------------------------------------------------------------

def get_usage_dashboard(hours: int = 24) -> str:
    """Combined LLM usage analytics dashboard.

    Pulls data from metrics_db, claude_vault, and cache stats to produce
    a comprehensive cost and performance overview.
    """
    lines = [f"**LLM Usage Analytics** (last {hours}h)", ""]

    # Metrics DB data
    try:
        from . import metrics_db

        stats = metrics_db.get_endpoint_stats(hours=hours)
        if stats.get("calls", 0) > 0:
            lines.append(f"**Total Calls:** {stats['calls']} ({stats['success_rate']}% success)")
            lines.append(f"**Tokens:** {stats.get('total_input_tokens', 0):,} in / {stats.get('total_output_tokens', 0):,} out")
            lines.append(f"**Latency:** avg {stats.get('avg_latency', 0)}s, max {stats.get('max_latency', 0)}s")
            lines.append("")

        # Per-endpoint breakdown
        for ep in ["ollama", "claude_api", "claude_cli", "ollama_vision"]:
            ep_stats = metrics_db.get_endpoint_stats(ep, hours)
            if ep_stats.get("calls", 0) > 0:
                pcts = metrics_db.get_percentiles(ep, hours)
                p95 = pcts.get("p95", "?")
                lines.append(
                    f"**{ep}:** {ep_stats['calls']} calls, "
                    f"avg {ep_stats['avg_latency']}s, p95 {p95}s, "
                    f"{ep_stats.get('total_input_tokens', 0):,}+{ep_stats.get('total_output_tokens', 0):,} tokens"
                )
        lines.append("")
    except Exception:
        lines.append("*Metrics DB unavailable*\n")

    # Claude cost data
    try:
        from .claude_vault import get_vault_session

        session = get_vault_session()
        usage = session.get_usage_stats()
        cost = session.get_total_cost()
        lines.append("**Claude API Cost (this session):**")
        lines.append(f"  Total: ${cost:.4f}")
        lines.append(f"  Tokens: {usage.get('total_input_tokens', 0):,} in / {usage.get('total_output_tokens', 0):,} out")
        cache_saved = usage.get("estimated_savings_tokens", 0)
        if cache_saved:
            lines.append(f"  Cache savings: {cache_saved:,} tokens")
        lines.append("")
    except Exception:
        pass  # Claude vault may not be initialized

    # Response cache stats
    cache = get_cache_stats()
    if cache["cached_queries"] > 0:
        lines.append("**Response Cache:**")
        lines.append(f"  Cached queries: {cache['cached_queries']}")
        lines.append(f"  Cache hits: {cache['total_hits']}")
        lines.append(f"  Tokens saved: {cache['tokens_saved']:,}")
        lines.append("")

    # Latency alerts
    try:
        from . import metrics_db

        slow = metrics_db.get_slowest_calls(3, hours)
        if slow:
            lines.append("**Slowest Calls:**")
            for s in slow:
                ts = s["timestamp"][11:19] if len(s["timestamp"]) > 11 else s["timestamp"]
                lines.append(
                    f"  [{ts}] {s['endpoint']} ({s.get('model', '?')}) — "
                    f"{s['duration']}s, {s['input_tokens'] + s['output_tokens']} tokens"
                )
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def get_llm_optimizer_tools() -> list:
    """Get LLM optimizer tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "llm_usage_dashboard",
            "Show LLM usage analytics: costs, token counts, latency, cache stats, and per-endpoint breakdown",
            parameters={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Hours of history to analyze (default 24)",
                    },
                },
                "required": [],
            },
            function=lambda hours=24: get_usage_dashboard(hours),
        ),
        create_tool(
            "query_complexity",
            "Score a query's complexity to determine optimal model routing",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to score"},
                },
                "required": ["query"],
            },
            function=lambda query: str(score_query_complexity(query)),
        ),
    ]
