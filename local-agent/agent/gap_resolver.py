"""
Knowledge Gap Resolution Workflow - Automated pipeline to resolve gaps.

Detects open knowledge gaps, fetches Wikipedia suggestions for each,
stores structured resolution candidates in the vault, and provides
tools to accept suggestions (auto-adding to facts_db and marking the
gap resolved).

Lifecycle:  detected → suggested → accepted/dismissed → resolved
Runs as a daily background task and exposes agent tools for on-demand use.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from .config import settings
from .domain_coverage import classify_to_primary_domain
from .facts_db import add_fact, lookup_fact
from .knowledge_gaps import GAPS_FILE, resolve_gap

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
RESOLUTIONS_FILE = VAULT_PATH / "Permanent" / "gap_resolutions.json"

# Schedule: daily at 5 AM
REPORT_HOUR = 5

_WIKI_HEADERS = {"User-Agent": "TechnomancerBot/1.0"}
_WIKI_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Gap parsing (open gaps only — these are what need resolving)
# ---------------------------------------------------------------------------


def _parse_open_gaps() -> list[dict[str, str]]:
    """Parse open gap entries from knowledge_gaps.md."""
    if not GAPS_FILE.exists():
        return []

    content = GAPS_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"## Open Gaps\s*\n(.*?)(?=## Resolved|$)", content, re.DOTALL
    )
    if not match:
        return []

    entries = re.split(r"(?=- \*\*\[)", match.group(1))
    gaps: list[dict[str, str]] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        type_match = re.search(r"\*\*\[(\w+)\]\*\*", entry)
        ts_match = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})", entry)
        query_match = re.search(r"\*\*Query:\*\*\s*(.+)", entry)

        gaps.append({
            "gap_type": type_match.group(1).lower() if type_match else "unknown",
            "timestamp": ts_match.group(1) if ts_match else "",
            "query": query_match.group(1).strip() if query_match else "",
        })

    return [g for g in gaps if g["query"]]


# ---------------------------------------------------------------------------
# Keyword extraction for Wikipedia lookups
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "what", "where", "when", "which", "that", "this", "does", "have",
    "with", "from", "about", "your", "the", "and", "for", "are", "how",
    "can", "tell", "know", "please", "would", "could", "should", "mean",
    "means", "there", "been", "being", "will", "just", "some", "than",
    "very", "much", "more", "most", "also", "only", "like", "into",
    "them", "they", "their", "these", "those", "then", "here", "who",
    "why", "isn", "don", "didn", "wasn", "explain", "describe",
})


def _extract_search_terms(query: str) -> list[str]:
    """
    Extract meaningful search terms from a gap query.

    Returns a list of candidate search phrases, from most specific
    (multi-word) to least specific (single keywords).
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]

    terms: list[str] = []

    # Try the full cleaned query first (most specific)
    full = " ".join(keywords)
    if len(keywords) >= 2:
        terms.append(full)

    # Try 2-3 word combinations from the start
    if len(keywords) >= 3:
        terms.append(" ".join(keywords[:3]))
    if len(keywords) >= 2:
        terms.append(" ".join(keywords[:2]))

    # Individual keywords (most common nouns tend to be last)
    for kw in keywords:
        if kw not in terms:
            terms.append(kw)

    return terms[:5]


# ---------------------------------------------------------------------------
# Wikipedia lookup
# ---------------------------------------------------------------------------


def fetch_wikipedia_suggestion(query: str) -> dict[str, str] | None:
    """
    Try to find a relevant Wikipedia article for a gap query.

    Returns {title, extract, url, search_term} or None if nothing found.
    """
    terms = _extract_search_terms(query)

    for term in terms:
        result = _try_wikipedia_lookup(term)
        if result:
            result["search_term"] = term
            return result

    return None


def _try_wikipedia_lookup(term: str) -> dict[str, str] | None:
    """Attempt a single Wikipedia lookup for a term."""
    try:
        # Direct page lookup
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + term.replace(" ", "_")
        )
        resp = requests.get(url, timeout=_WIKI_TIMEOUT, headers=_WIKI_HEADERS)

        if resp.status_code == 404:
            # Fallback to search API
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "format": "json",
                "srlimit": 1,
            }
            search_resp = requests.get(
                search_url, params=params, timeout=_WIKI_TIMEOUT,
                headers=_WIKI_HEADERS,
            )
            results = search_resp.json().get("query", {}).get("search", [])
            if not results:
                return None

            title = results[0]["title"]
            url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + title.replace(" ", "_")
            )
            resp = requests.get(
                url, timeout=_WIKI_TIMEOUT, headers=_WIKI_HEADERS
            )

        if resp.status_code != 200:
            return None

        data = resp.json()
        extract = data.get("extract", "")
        if not extract or len(extract) < 20:
            return None

        return {
            "title": data.get("title", term),
            "extract": extract[:500],
            "url": (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            ),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Resolution state management (JSON file in vault)
# ---------------------------------------------------------------------------


def _load_resolutions() -> list[dict[str, Any]]:
    """Load the resolutions state file."""
    if not RESOLUTIONS_FILE.exists():
        return []
    try:
        data = json.loads(RESOLUTIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_resolutions(resolutions: list[dict[str, Any]]) -> None:
    """Save the resolutions state file."""
    RESOLUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESOLUTIONS_FILE.write_text(
        json.dumps(resolutions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _find_resolution(
    resolutions: list[dict[str, Any]], query_fragment: str
) -> tuple[int, dict[str, Any] | None]:
    """Find a resolution entry matching a query fragment."""
    fragment_lower = query_fragment.lower()
    for i, r in enumerate(resolutions):
        if fragment_lower in r.get("query", "").lower():
            return i, r
    return -1, None


# ---------------------------------------------------------------------------
# Core workflow functions
# ---------------------------------------------------------------------------


def generate_suggestions() -> list[dict[str, Any]]:
    """
    Scan open gaps and generate Wikipedia-based resolution suggestions
    for any that don't already have a pending suggestion.

    Returns list of newly created suggestions.
    """
    open_gaps = _parse_open_gaps()
    if not open_gaps:
        return []

    resolutions = _load_resolutions()
    existing_queries = {r["query"].lower() for r in resolutions}

    new_suggestions: list[dict[str, Any]] = []

    for gap in open_gaps:
        query = gap["query"]
        if query.lower() in existing_queries:
            continue

        # Check if facts DB already covers this
        keywords = _extract_search_terms(query)
        already_covered = False
        for term in keywords[:3]:
            if lookup_fact(term):
                already_covered = True
                break

        if already_covered:
            # Auto-mark as covered — facts DB has relevant content
            entry = {
                "query": query,
                "gap_type": gap["gap_type"],
                "domain": classify_to_primary_domain(query),
                "detected": gap["timestamp"],
                "status": "auto_covered",
                "suggestion": None,
                "resolved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "resolution_note": "Facts DB already contains relevant entries",
            }
            resolutions.append(entry)
            new_suggestions.append(entry)
            continue

        # Fetch Wikipedia suggestion
        wiki = fetch_wikipedia_suggestion(query)
        entry: dict[str, Any] = {
            "query": query,
            "gap_type": gap["gap_type"],
            "domain": classify_to_primary_domain(query),
            "detected": gap["timestamp"],
            "status": "suggested" if wiki else "no_source",
            "suggestion": wiki,
            "resolved_at": None,
            "resolution_note": None,
        }
        resolutions.append(entry)
        new_suggestions.append(entry)

    _save_resolutions(resolutions)
    return new_suggestions


def accept_suggestion(query_fragment: str) -> str:
    """
    Accept a suggestion: add to facts_db and resolve the gap.

    Args:
        query_fragment: Part of the original query to match.

    Returns:
        Confirmation or error message.
    """
    resolutions = _load_resolutions()
    idx, entry = _find_resolution(resolutions, query_fragment)

    if entry is None:
        return f"No resolution found matching '{query_fragment}'."

    if entry["status"] not in ("suggested",):
        return (
            f"Resolution for '{entry['query'][:60]}' has status "
            f"'{entry['status']}' — can only accept 'suggested' entries."
        )

    suggestion = entry.get("suggestion")
    if not suggestion:
        return "No Wikipedia suggestion available for this entry."

    # Add to facts database
    domain = entry.get("domain", "uncategorized")
    category = _domain_to_facts_category(domain)
    key = suggestion["title"]
    value = suggestion["extract"]

    add_fact(category, key, value, source="wikipedia")

    # Resolve the gap in knowledge_gaps.md
    query = entry["query"]
    resolve_gap(
        query[:80],
        f"Auto-resolved: added '{key}' from Wikipedia to facts DB",
    )

    # Update resolution entry
    entry["status"] = "accepted"
    entry["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry["resolution_note"] = f"Added to facts DB as [{category}] {key}"
    resolutions[idx] = entry
    _save_resolutions(resolutions)

    return (
        f"Accepted: added **{key}** to facts DB [{category}] "
        f"and resolved gap for '{query[:60]}'."
    )


def dismiss_suggestion(query_fragment: str, reason: str = "") -> str:
    """
    Dismiss a suggestion without adding to facts_db.

    The gap remains open for manual resolution.
    """
    resolutions = _load_resolutions()
    idx, entry = _find_resolution(resolutions, query_fragment)

    if entry is None:
        return f"No resolution found matching '{query_fragment}'."

    entry["status"] = "dismissed"
    entry["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry["resolution_note"] = reason or "Dismissed by user"
    resolutions[idx] = entry
    _save_resolutions(resolutions)

    return f"Dismissed suggestion for '{entry['query'][:60]}'."


def get_pending_suggestions() -> str:
    """Get all pending (suggested) resolution candidates."""
    resolutions = _load_resolutions()
    pending = [r for r in resolutions if r["status"] == "suggested"]

    if not pending:
        return "No pending resolution suggestions."

    lines = [f"**Pending Resolution Suggestions ({len(pending)}):**\n"]
    for r in pending:
        domain = r.get("domain", "?").replace("_", " ").title()
        suggestion = r.get("suggestion", {})
        title = suggestion.get("title", "N/A") if suggestion else "N/A"
        lines.append(f"- **Query:** {r['query'][:100]}")
        lines.append(f"  - Domain: {domain} | Type: {r.get('gap_type', '?')}")
        lines.append(f"  - Suggested: **{title}** (Wikipedia)")
        if suggestion and suggestion.get("extract"):
            lines.append(f"  - Preview: {suggestion['extract'][:150]}...")
        lines.append("")

    return "\n".join(lines)


def get_resolution_stats() -> str:
    """Get resolution workflow statistics."""
    resolutions = _load_resolutions()
    if not resolutions:
        return "No resolution data yet."

    counts: dict[str, int] = {}
    for r in resolutions:
        status = r.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    total = len(resolutions)
    lines = ["**Gap Resolution Statistics:**\n"]
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    for status in ["suggested", "accepted", "dismissed", "auto_covered", "no_source"]:
        if status in counts:
            lines.append(f"| {status.replace('_', ' ').title()} | {counts[status]} |")
    lines.append(f"| **Total** | **{total}** |")

    # Resolution rate
    resolved = counts.get("accepted", 0) + counts.get("auto_covered", 0)
    if total > 0:
        rate = resolved / total * 100
        lines.append(f"\nResolution rate: **{rate:.0f}%** ({resolved}/{total})")

    return "\n".join(lines)


def _domain_to_facts_category(domain: str) -> str:
    """Map a domain classification to a facts_db category."""
    mapping = {
        "food_cooking": "definition",
        "science": "definition",
        "technology": "definition",
        "geography": "geography",
        "history": "definition",
        "math": "definition",
        "language": "definition",
        "health_medicine": "definition",
        "business_finance": "definition",
        "arts_entertainment": "definition",
        "conversions_units": "conversion",
        "time_dates": "timezone",
    }
    return mapping.get(domain, "definition")


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def get_gap_resolver_tools() -> list:
    """Get agent tools for the gap resolution workflow."""
    from .core import create_tool

    return [
        create_tool(
            "get_pending_resolutions",
            (
                "Get the list of pending knowledge gap resolution suggestions. "
                "Each suggestion includes a Wikipedia article that could fill "
                "the gap. Use when reviewing what knowledge gaps can be resolved."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_pending_suggestions(),
        ),
        create_tool(
            "accept_resolution",
            (
                "Accept a resolution suggestion: adds the Wikipedia content to "
                "the facts database and marks the knowledge gap as resolved. "
                "Provide part of the original query to match the suggestion."
            ),
            {
                "type": "object",
                "properties": {
                    "query_fragment": {
                        "type": "string",
                        "description": "Part of the original gap query to match",
                    },
                },
                "required": ["query_fragment"],
            },
            lambda query_fragment: accept_suggestion(query_fragment),
        ),
        create_tool(
            "dismiss_resolution",
            (
                "Dismiss a resolution suggestion without applying it. The "
                "knowledge gap remains open for manual resolution. Provide "
                "part of the original query and optionally a reason."
            ),
            {
                "type": "object",
                "properties": {
                    "query_fragment": {
                        "type": "string",
                        "description": "Part of the original gap query to match",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the suggestion was dismissed",
                    },
                },
                "required": ["query_fragment"],
            },
            lambda query_fragment, reason="": dismiss_suggestion(
                query_fragment, reason
            ),
        ),
        create_tool(
            "get_resolution_stats",
            (
                "Get statistics on the knowledge gap resolution workflow — "
                "how many gaps have been suggested, accepted, dismissed, etc."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_resolution_stats(),
        ),
    ]


# ---------------------------------------------------------------------------
# Scheduled background task
# ---------------------------------------------------------------------------


def _seconds_until_next_run() -> float:
    """Calculate seconds until next daily run at REPORT_HOUR."""
    now = datetime.now()
    next_run = now.replace(
        hour=REPORT_HOUR, minute=0, second=0, microsecond=0
    )
    if now.hour >= REPORT_HOUR:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


async def gap_resolver_loop(client: Any, channel_name: str) -> None:
    """Background loop that generates resolution suggestions daily."""
    log.info("[GapResolver] Started — runs daily at %d:00", REPORT_HOUR)

    while True:
        try:
            wait = _seconds_until_next_run()
            log.info("[GapResolver] Next run in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            new = generate_suggestions()
            if new:
                suggested = [s for s in new if s["status"] == "suggested"]
                auto = [s for s in new if s["status"] == "auto_covered"]

                log.info(
                    "[GapResolver] Generated %d suggestions, %d auto-covered",
                    len(suggested), len(auto),
                )

                # Post summary to Discord
                if (suggested or auto) and client and channel_name:
                    lines = ["**Gap Resolver Update**"]
                    if suggested:
                        lines.append(
                            f"New suggestions: **{len(suggested)}** "
                            f"(Wikipedia matches for open gaps)"
                        )
                        for s in suggested[:3]:
                            title = (
                                s.get("suggestion", {}).get("title", "?")
                                if s.get("suggestion")
                                else "?"
                            )
                            lines.append(
                                f"  - {s['query'][:60]} → **{title}**"
                            )
                    if auto:
                        lines.append(
                            f"Auto-covered: **{len(auto)}** "
                            f"(facts DB already has content)"
                        )
                    lines.append(
                        "Use `get_pending_resolutions` to review."
                    )
                    msg = "\n".join(lines)

                    for guild in client.guilds:
                        channel = next(
                            (
                                c for c in guild.text_channels
                                if c.name == channel_name
                            ),
                            None,
                        )
                        if channel:
                            await channel.send(msg)
                            break
            else:
                log.info("[GapResolver] No new gaps to process")

            await asyncio.sleep(60)

        except Exception:
            log.exception("[GapResolver] Loop error")
            await asyncio.sleep(300)


def start_gap_resolver(client: Any, channel_name: str) -> None:
    """Start the daily gap resolver background task."""
    asyncio.create_task(gap_resolver_loop(client, channel_name))
    log.info("[GapResolver] Background task started")
