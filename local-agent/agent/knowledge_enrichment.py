"""
Knowledge Base Enrichment — Scan conversations for unresolved queries and
automatically enrich facts_db and the Obsidian vault.

Runs every 6 hours.  Each run:
1. Scans recent conversation logs for uncertainty/failure signals
2. Clusters the gap queries by topic
3. For each cluster, tries to resolve via Wikipedia → web search
4. Caches successful results in facts_db
5. Writes reference articles to Obsidian vault
6. Produces a summary report and optionally posts to Discord

Builds on gap_frequency (scanning/clustering), knowledge_fallback (Wikipedia
and web search), ref_enrichment (vault article writing), and facts_db
(caching).
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
ENRICHMENT_LOG = VAULT_PATH / "Permanent" / "enrichment_log.json"

# Run every 6 hours, at minutes :30 to avoid colliding with hourly tasks
RUN_INTERVAL_HOURS = 6


# ---------------------------------------------------------------------------
# Enrichment log
# ---------------------------------------------------------------------------

def _load_enrichment_log() -> list[dict[str, Any]]:
    if not ENRICHMENT_LOG.exists():
        return []
    try:
        data = json.loads(ENRICHMENT_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_enrichment_log(entries: list[dict[str, Any]]) -> None:
    ENRICHMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Keep last 500 entries
    ENRICHMENT_LOG.write_text(
        json.dumps(entries[-500:], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_enrichment(query: str, status: str, source: str = "", title: str = "") -> None:
    entries = _load_enrichment_log()
    entries.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "query": query[:200],
        "status": status,
        "source": source,
        "title": title,
    })
    _save_enrichment_log(entries)


# ---------------------------------------------------------------------------
# Core enrichment run
# ---------------------------------------------------------------------------

def _get_already_enriched_queries() -> set[str]:
    """Return set of queries already enriched (to skip duplicates)."""
    entries = _load_enrichment_log()
    return {e["query"].lower().strip() for e in entries if e["status"] == "enriched"}


def run_enrichment(days: int = 7, max_enrichments: int = 10) -> dict[str, Any]:
    """Run one enrichment cycle.

    Scans the last ``days`` of conversation logs for knowledge gaps,
    clusters them, and tries to fill each gap from Wikipedia or web search.

    Returns a summary dict with counts and details.
    """
    from .gap_frequency import _parse_all_gaps, _scan_conversations_for_gaps, cluster_gaps

    # Gather gaps
    logged = _parse_all_gaps()
    scanned = _scan_conversations_for_gaps(days=days)

    # Deduplicate logged vs scanned
    logged_keys = {(g["query"][:80], g["timestamp"]) for g in logged}
    unique_scanned = [
        g for g in scanned
        if (g["query"][:80], g["timestamp"]) not in logged_keys
    ]
    all_gaps = logged + unique_scanned

    # Only process open/detected gaps (not already resolved)
    open_gaps = [g for g in all_gaps if g.get("status") in ("open", "detected")]
    if not open_gaps:
        return {"enriched": 0, "skipped": 0, "failed": 0, "details": []}

    clusters = cluster_gaps(open_gaps)
    if not clusters:
        return {"enriched": 0, "skipped": 0, "failed": 0, "details": []}

    already_done = _get_already_enriched_queries()
    enriched = 0
    skipped = 0
    failed = 0
    details: list[dict[str, str]] = []

    for cluster in clusters:
        if enriched >= max_enrichments:
            break

        query = cluster["queries"][0] if cluster["queries"] else ""
        if not query:
            continue

        # Skip if we already enriched this query
        if query.lower().strip() in already_done:
            skipped += 1
            continue

        result = _try_enrich_query(query, cluster.get("domain", ""))
        if result["status"] == "enriched":
            enriched += 1
            details.append(result)
            _log_enrichment(query, "enriched", result.get("source", ""), result.get("title", ""))

            # Try to resolve the gap in knowledge_gaps.md
            try:
                from .knowledge_gaps import resolve_gap
                resolve_gap(query[:50], resolution=f"Auto-enriched from {result.get('source', 'external')}")
            except Exception:
                pass
        else:
            failed += 1
            _log_enrichment(query, "no_source")

    return {
        "enriched": enriched,
        "skipped": skipped,
        "failed": failed,
        "total_gaps": len(open_gaps),
        "total_clusters": len(clusters),
        "details": details,
    }


def _try_enrich_query(query: str, domain: str) -> dict[str, str]:
    """Try to enrich a single query from external sources.

    Returns dict with keys: status, title, source, category.
    """
    from .facts_db import lookup_fact, add_fact
    from .knowledge_fallback import _query_wikipedia, _search_wikipedia, _guess_category

    # Already in facts_db? Skip.
    existing = lookup_fact(query)
    if existing:
        return {"status": "already_exists", "title": existing[0]["key"]}

    # Try Wikipedia
    wiki_text = _query_wikipedia(query)
    if not wiki_text:
        wiki_text = _search_wikipedia(query)

    if wiki_text:
        category = _guess_category(query, wiki_text)
        add_fact(category, query.lower().strip(), wiki_text, source="auto_enrichment")

        # Write vault article via ref_enrichment if available
        _write_vault_article(query, wiki_text, domain)

        return {
            "status": "enriched",
            "title": query,
            "source": "wikipedia",
            "category": category,
        }

    # Try web search
    from .knowledge_fallback import _web_search_summary

    web_text = _web_search_summary(query)
    if web_text:
        category = _guess_category(query, web_text)
        add_fact(category, query.lower().strip(), web_text, source="auto_enrichment")
        return {
            "status": "enriched",
            "title": query,
            "source": "web_search",
            "category": category,
        }

    return {"status": "no_source", "title": ""}


def _write_vault_article(query: str, extract: str, domain: str) -> None:
    """Best-effort write of a vault reference article."""
    try:
        from .ref_enrichment import write_reference_article

        write_reference_article(
            title=query,
            extract=extract,
            url="",
            source="auto_enrichment",
            domain=domain or "general",
            original_query=query,
        )
    except Exception:
        log.debug("Could not write vault article for %r", query, exc_info=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_enrichment_report() -> str:
    """Human-readable enrichment report for Discord."""
    entries = _load_enrichment_log()
    if not entries:
        return "No enrichment runs recorded yet. The system runs every 6 hours."

    enriched = [e for e in entries if e["status"] == "enriched"]
    no_source = [e for e in entries if e["status"] == "no_source"]

    lines = ["**Knowledge Base Enrichment Report**", ""]
    lines.append(f"Total enrichments: **{len(enriched)}**")
    lines.append(f"Failed lookups: {len(no_source)}")
    if enriched:
        rate = len(enriched) / (len(enriched) + len(no_source)) * 100
        lines.append(f"Success rate: {rate:.0f}%")
    lines.append("")

    # Recent enrichments
    recent = enriched[-5:]
    if recent:
        lines.append("**Recent enrichments:**")
        for e in reversed(recent):
            lines.append(
                f"  - [{e.get('timestamp', '?')}] **{e.get('title', '?')}** "
                f"({e.get('source', '?')})"
            )
    else:
        lines.append("No successful enrichments yet.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def _tool_run_enrichment(days: int = 7) -> str:
    """Tool wrapper for on-demand enrichment."""
    try:
        result = run_enrichment(days=days)
        if result["enriched"] == 0 and result["failed"] == 0:
            return "No open knowledge gaps found to enrich."
        lines = [f"**Enrichment complete:** {result['enriched']} enriched, {result['failed']} failed"]
        for d in result["details"]:
            lines.append(f"  - **{d.get('title', '?')}** ({d.get('source', '?')})")
        return "\n".join(lines)
    except Exception as e:
        log.exception("Enrichment run failed")
        return f"Enrichment error: {e}"


def get_knowledge_enrichment_tools() -> list:
    """Get knowledge enrichment tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "enrich_knowledge_base",
            (
                "Scan recent conversations for knowledge gaps and automatically "
                "fill them from Wikipedia and web search. Results are cached in "
                "the facts database and written to the Obsidian vault."
            ),
            {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Days of conversation history to scan (default 7)",
                    },
                },
                "required": [],
            },
            _tool_run_enrichment,
        ),
        create_tool(
            "enrichment_report",
            "Show knowledge base enrichment stats and recent additions",
            {"type": "object", "properties": {}, "required": []},
            lambda: get_enrichment_report(),
        ),
    ]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _seconds_until_next_run() -> float:
    """Calculate seconds until the next 6-hour run slot (:30 past the hour)."""
    now = datetime.now()
    # Run at hours 0:30, 6:30, 12:30, 18:30
    current_hour = now.hour
    next_hours = [h for h in (0, 6, 12, 18) if h > current_hour]
    if not next_hours:
        next_hour = 24  # wrap to midnight tomorrow
    else:
        next_hour = next_hours[0]

    next_run = now.replace(hour=next_hour % 24, minute=30, second=0, microsecond=0)
    if next_hour >= 24:
        next_run += timedelta(days=1)

    delta = (next_run - now).total_seconds()
    return max(delta, 60)  # at least 1 minute


async def enrichment_loop(client: Any, channel_name: str) -> None:
    """Background loop that enriches the knowledge base every 6 hours."""
    log.info("[KnowledgeEnrichment] Started — runs every %d hours", RUN_INTERVAL_HOURS)

    while True:
        try:
            wait = _seconds_until_next_run()
            log.info("[KnowledgeEnrichment] Next run in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            result = run_enrichment(days=7, max_enrichments=5)
            enriched = result["enriched"]

            if enriched > 0 and client and channel_name:
                titles = [d.get("title", "?") for d in result["details"]]
                lines = [f"**Knowledge Enrichment** — {enriched} new fact(s) added"]
                for t in titles[:5]:
                    lines.append(f"  - {t}")
                msg = "\n".join(lines)

                for guild in client.guilds:
                    channel = next(
                        (c for c in guild.text_channels if c.name == channel_name),
                        None,
                    )
                    if channel:
                        await channel.send(msg)
                        break

            # Sleep a minute to avoid re-triggering in the same slot
            await asyncio.sleep(60)

        except Exception:
            log.exception("[KnowledgeEnrichment] Loop error")
            await asyncio.sleep(300)


def start_knowledge_enrichment(client: Any, channel_name: str) -> None:
    """Start the knowledge enrichment background task."""
    from .task_manager import create_monitored_task

    create_monitored_task(enrichment_loop(client, channel_name), "knowledge-enrichment", critical=True)
    log.info("[KnowledgeEnrichment] Background task started")
