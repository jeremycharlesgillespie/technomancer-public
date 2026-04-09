"""
Knowledge Fallback — Tiered lookup for factual queries.

Checks facts_db first, then falls back to Wikipedia API, then DuckDuckGo
web search.  Successful results are cached in facts_db for future lookups
and gaps are logged to knowledge_gaps.py for tracking.

Exposes a single ``knowledge_lookup`` tool for the agent.
"""

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Wikipedia API endpoint (no auth required)
_WIKI_API = "https://en.wikipedia.org/w/api.php"


# ---------------------------------------------------------------------------
# Wikipedia API
# ---------------------------------------------------------------------------

def _query_wikipedia(query: str) -> str | None:
    """Fetch the opening extract from Wikipedia for a query.

    Returns the first paragraph (up to 500 chars) or None on failure.
    """
    try:
        import requests

        resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "format": "json",
                "titles": query,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "redirects": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            extract = page.get("extract", "").strip()
            if extract:
                # Return first meaningful paragraph, capped
                paragraphs = [p.strip() for p in extract.split("\n") if p.strip()]
                if paragraphs:
                    text = paragraphs[0]
                    if len(text) > 500:
                        text = text[:497] + "..."
                    return text
    except Exception:
        log.debug("Wikipedia query failed for %r", query, exc_info=True)
    return None


def _search_wikipedia(query: str) -> str | None:
    """Search Wikipedia when the exact title doesn't match.

    Uses opensearch to find the best title, then fetches that extract.
    """
    try:
        import requests

        resp = requests.get(
            _WIKI_API,
            params={
                "action": "opensearch",
                "format": "json",
                "search": query,
                "limit": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        if len(data) >= 2 and data[1]:
            best_title = data[1][0]
            return _query_wikipedia(best_title)
    except Exception:
        log.debug("Wikipedia search failed for %r", query, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Web search fallback
# ---------------------------------------------------------------------------

def _web_search_summary(query: str) -> str | None:
    """Search DuckDuckGo and return the first result's description."""
    try:
        from .web_search import web_search

        result = web_search(query + " definition", max_results=2)
        if result and "No results" not in result and "Search error" not in result:
            # Extract just the first description line
            lines = result.strip().split("\n")
            for line in lines:
                line = line.strip()
                # Skip header, URLs, and blank lines
                if line and not line.startswith(("Web search", "Source:", "http")) and not line[0].isdigit():
                    if not line.startswith("**"):
                        return line[:500]
            # Fallback: return the whole result block, trimmed
            return result[:500]
    except Exception:
        log.debug("Web search fallback failed for %r", query, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Core lookup pipeline
# ---------------------------------------------------------------------------

def knowledge_lookup(query: str) -> str:
    """Look up a factual query through a tiered fallback chain.

    1. facts_db (instant, local)
    2. Wikipedia API (authoritative, free)
    3. DuckDuckGo web search (broad coverage)

    Successful external results are cached in facts_db for next time.
    All lookups are logged for gap tracking.
    """
    query = query.strip()
    if not query:
        return "Please provide a query to look up."

    # --- Tier 1: facts_db ---
    try:
        from .facts_db import lookup_fact, add_fact

        results = lookup_fact(query)
        if results:
            best = results[0]
            return (
                f"**{best['key']}** ({best['category']}): {best['value']}\n"
                f"*(source: {best['source']})*"
            )
    except Exception:
        log.debug("facts_db lookup failed", exc_info=True)

    # --- Tier 2: Wikipedia ---
    wiki_result = _query_wikipedia(query)
    if not wiki_result:
        wiki_result = _search_wikipedia(query)

    if wiki_result:
        # Cache in facts_db for next time
        _cache_result(query, wiki_result, "wikipedia")
        _log_gap_resolved(query, "wikipedia")
        return f"**{query}**: {wiki_result}\n*(source: Wikipedia — cached for future)*"

    # --- Tier 3: Web search ---
    web_result = _web_search_summary(query)
    if web_result:
        _cache_result(query, web_result, "web_search")
        _log_gap_resolved(query, "web_search")
        return f"**{query}**: {web_result}\n*(source: web search — cached for future)*"

    # --- All tiers failed ---
    _log_gap_unresolved(query)
    return f"Could not find information about '{query}' in local facts, Wikipedia, or web search."


# ---------------------------------------------------------------------------
# Cache and gap logging helpers
# ---------------------------------------------------------------------------

def _cache_result(query: str, value: str, source: str) -> None:
    """Cache a successful lookup result in facts_db."""
    try:
        from .facts_db import add_fact

        # Determine a reasonable category from the content
        category = _guess_category(query, value)
        add_fact(category, query.lower(), value, source=source)
    except Exception:
        log.debug("Failed to cache result in facts_db", exc_info=True)


def _guess_category(query: str, value: str) -> str:
    """Heuristic category assignment for cached facts."""
    text = (query + " " + value).lower()
    if any(w in text for w in ("capital", "country", "city", "river", "continent", "ocean", "mountain")):
        return "geography"
    if any(w in text for w in ("born", "died", "president", "king", "queen", "leader")):
        return "history"
    if any(w in text for w in ("python", "javascript", "api", "software", "algorithm", "programming")):
        return "technology"
    if any(w in text for w in ("atom", "molecule", "cell", "planet", "species", "energy")):
        return "science"
    return "definition"


def _log_gap_resolved(query: str, source: str) -> None:
    """Log that a knowledge gap was filled by an external source."""
    try:
        from .knowledge_gaps import resolve_gap

        resolve_gap(query[:50], resolution=f"Resolved via {source} fallback")
    except Exception:
        pass  # best-effort


def _log_gap_unresolved(query: str) -> None:
    """Log an unresolved knowledge gap."""
    try:
        from .knowledge_gaps import log_knowledge_gap

        log_knowledge_gap({
            "query": query[:500],
            "response_snippet": "External fallback (Wikipedia + web search) found no results",
            "gap_type": "failure",
            "was_escalated": False,
            "timestamp": _now_str(),
        })
    except Exception:
        pass  # best-effort


def _now_str() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------

def get_knowledge_fallback_tools() -> list:
    """Get the knowledge fallback tool for agent registration."""
    from .core import create_tool

    return [
        create_tool(
            name="knowledge_lookup",
            description=(
                "Look up factual information through a tiered fallback: "
                "local facts database → Wikipedia → web search. "
                "Use this for definitions, geography, science, history, "
                "and other common knowledge queries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The factual query to look up (e.g. 'spaghetti', 'France capital', 'photosynthesis')",
                    },
                },
                "required": ["query"],
            },
            function=knowledge_lookup,
        ),
    ]
