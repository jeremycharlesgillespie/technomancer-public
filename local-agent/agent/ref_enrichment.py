"""
Reference Source Enhancement System - Write reference articles to Obsidian vault.

Complements gap_resolver (which adds facts to the SQLite DB) by writing
full, human-readable reference articles as Obsidian .md files.  Tracks
import metrics over time and supports multiple reference sources.

Flow:
1. Reads accepted resolutions and high-frequency gap clusters
2. Fetches reference content from configured sources (Wikipedia, etc.)
3. Writes Markdown articles to Permanent/References/ in the vault
4. Logs import status and metrics to import_log.json

Runs as a daily background task (4 AM) and exposes agent tools.
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

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
REFERENCES_DIR = VAULT_PATH / "Permanent" / "References"
IMPORT_LOG_FILE = VAULT_PATH / "Permanent" / "import_log.json"

REPORT_HOUR = 4  # 4 AM daily

_WIKI_HEADERS = {"User-Agent": "TechnomancerBot/1.0"}
_WIKI_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Reference sources — extensible configuration
# ---------------------------------------------------------------------------

SOURCES: list[dict[str, str]] = [
    {
        "name": "wikipedia",
        "label": "Wikipedia",
        "type": "api",
    },
]


# ---------------------------------------------------------------------------
# Wikipedia fetcher
# ---------------------------------------------------------------------------


def _fetch_wikipedia_article(term: str) -> dict[str, str] | None:
    """
    Fetch a full Wikipedia summary for a topic.

    Returns {title, extract, url, description} or None.
    """
    try:
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + term.replace(" ", "_")
        )
        resp = requests.get(url, timeout=_WIKI_TIMEOUT, headers=_WIKI_HEADERS)

        if resp.status_code == 404:
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
                url, timeout=_WIKI_TIMEOUT, headers=_WIKI_HEADERS,
            )

        if resp.status_code != 200:
            return None

        data = resp.json()
        extract = data.get("extract", "")
        if not extract or len(extract) < 20:
            return None

        return {
            "title": data.get("title", term),
            "extract": extract,
            "description": data.get("description", ""),
            "url": (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            ),
        }

    except Exception:
        return None


def fetch_reference(term: str, source: str = "wikipedia") -> dict[str, str] | None:
    """
    Fetch a reference article from the given source.

    Currently supports: wikipedia.
    Returns {title, extract, url, description, source} or None.
    """
    if source == "wikipedia":
        result = _fetch_wikipedia_article(term)
        if result:
            result["source"] = "wikipedia"
        return result
    return None


# ---------------------------------------------------------------------------
# Obsidian article writing
# ---------------------------------------------------------------------------


def _sanitize_filename(title: str) -> str:
    """Create a safe filename from a title."""
    safe = re.sub(r'[<>:"/\\|?*]', "", title)
    safe = safe.strip().replace(" ", "_")
    return safe[:80] if safe else "untitled"


def write_reference_article(
    title: str,
    extract: str,
    url: str,
    source: str,
    domain: str,
    description: str = "",
    original_query: str = "",
) -> str:
    """
    Write a reference article as an Obsidian .md file.

    Uses YAML frontmatter for metadata (searchable by Dataview).
    Returns the file path or empty string on failure.
    """
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(title) + ".md"
    filepath = REFERENCES_DIR / filename

    # Don't overwrite existing articles
    if filepath.exists():
        return str(filepath)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    domain_display = domain.replace("_", " ").title()

    content = f"""---
title: "{title}"
source: {source}
domain: {domain}
imported: {now}
url: "{url}"
original_query: "{original_query[:200]}"
---

# {title}

{f"> {description}" + chr(10) if description else ""}
{extract}

---
*Source: [{source.title()}]({url}) | Domain: {domain_display} | Imported: {now}*
"""

    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


# ---------------------------------------------------------------------------
# Import log management
# ---------------------------------------------------------------------------


def _load_import_log() -> list[dict[str, Any]]:
    """Load the import log."""
    if not IMPORT_LOG_FILE.exists():
        return []
    try:
        data = json.loads(IMPORT_LOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_import_log(entries: list[dict[str, Any]]) -> None:
    """Save the import log."""
    IMPORT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    IMPORT_LOG_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_import(
    title: str,
    source: str,
    domain: str,
    status: str,
    query: str = "",
    filepath: str = "",
    error: str = "",
) -> None:
    """Append an import event to the log."""
    entries = _load_import_log()
    entries.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": title,
        "source": source,
        "domain": domain,
        "status": status,
        "query": query[:200],
        "filepath": filepath,
        "error": error,
    })
    _save_import_log(entries)


# ---------------------------------------------------------------------------
# Keyword extraction (shared utility)
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
    """Extract search terms from a query, most-specific first."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]

    terms: list[str] = []
    full = " ".join(keywords)
    if len(keywords) >= 2:
        terms.append(full)
    if len(keywords) >= 3:
        terms.append(" ".join(keywords[:3]))
    if len(keywords) >= 2:
        terms.append(" ".join(keywords[:2]))
    for kw in keywords:
        if kw not in terms:
            terms.append(kw)
    return terms[:5]


# ---------------------------------------------------------------------------
# Core enrichment logic
# ---------------------------------------------------------------------------


def _already_imported(title: str) -> bool:
    """Check if a reference article already exists in the vault."""
    filename = _sanitize_filename(title) + ".md"
    return (REFERENCES_DIR / filename).exists()


def enrich_from_query(query: str, source: str = "wikipedia") -> dict[str, Any]:
    """
    Attempt to import a reference article for a given query.

    Returns a result dict with keys: status, title, filepath, source, domain.
    """
    terms = _extract_search_terms(query)
    domain = classify_to_primary_domain(query)

    for term in terms:
        ref = fetch_reference(term, source=source)
        if ref is None:
            continue

        title = ref["title"]
        if _already_imported(title):
            return {
                "status": "already_exists",
                "title": title,
                "filepath": str(REFERENCES_DIR / (_sanitize_filename(title) + ".md")),
                "source": source,
                "domain": domain,
            }

        filepath = write_reference_article(
            title=title,
            extract=ref["extract"],
            url=ref.get("url", ""),
            source=source,
            domain=domain,
            description=ref.get("description", ""),
            original_query=query,
        )

        # Also add to facts_db for quick lookup
        category = _domain_to_facts_category(domain)
        add_fact(category, title, ref["extract"][:500], source=source)

        _log_import(
            title=title,
            source=source,
            domain=domain,
            status="imported",
            query=query,
            filepath=filepath,
        )

        return {
            "status": "imported",
            "title": title,
            "filepath": filepath,
            "source": source,
            "domain": domain,
        }

    # No source found
    _log_import(
        title="",
        source=source,
        domain=domain,
        status="no_source",
        query=query,
    )
    return {
        "status": "no_source",
        "title": "",
        "filepath": "",
        "source": source,
        "domain": domain,
    }


def enrich_from_gap_resolutions() -> list[dict[str, Any]]:
    """
    Read accepted resolutions from gap_resolver and write vault articles
    for any that don't already have one.
    """
    from .gap_resolver import _load_resolutions

    resolutions = _load_resolutions()
    accepted = [r for r in resolutions if r.get("status") == "accepted"]
    results: list[dict[str, Any]] = []

    for res in accepted:
        suggestion = res.get("suggestion")
        if not suggestion:
            continue

        title = suggestion.get("title", "")
        if not title or _already_imported(title):
            continue

        domain = res.get("domain", "uncategorized")
        filepath = write_reference_article(
            title=title,
            extract=suggestion.get("extract", ""),
            url=suggestion.get("url", ""),
            source="wikipedia",
            domain=domain,
            description="",
            original_query=res.get("query", ""),
        )

        _log_import(
            title=title,
            source="wikipedia",
            domain=domain,
            status="imported",
            query=res.get("query", ""),
            filepath=filepath,
        )

        results.append({
            "status": "imported",
            "title": title,
            "filepath": filepath,
            "domain": domain,
        })

    return results


def enrich_from_frequent_gaps(top_n: int = 5) -> list[dict[str, Any]]:
    """
    Read top gap clusters from gap_frequency and try to import
    references for the most recurring topics.
    """
    from .gap_frequency import _parse_all_gaps, cluster_gaps

    all_gaps = _parse_all_gaps()
    if not all_gaps:
        return []

    clusters = cluster_gaps(all_gaps)
    results: list[dict[str, Any]] = []

    for cluster in clusters[:top_n]:
        if cluster["count"] < 2:
            break  # Only enrich recurring topics

        # Use the first query as the representative
        query = cluster["queries"][0] if cluster["queries"] else ""
        if not query:
            continue

        result = enrich_from_query(query)
        results.append(result)

    return results


def _domain_to_facts_category(domain: str) -> str:
    """Map a domain classification to a facts_db category."""
    mapping = {
        "geography": "geography",
        "conversions_units": "conversion",
        "time_dates": "timezone",
    }
    return mapping.get(domain, "definition")


# ---------------------------------------------------------------------------
# Import dashboard / metrics
# ---------------------------------------------------------------------------


def get_import_metrics() -> str:
    """Get import metrics as a formatted summary."""
    entries = _load_import_log()
    if not entries:
        return "No imports recorded yet."

    total = len(entries)
    imported = sum(1 for e in entries if e["status"] == "imported")
    no_source = sum(1 for e in entries if e["status"] == "no_source")
    failed = sum(1 for e in entries if e["status"] == "failed")

    # Domain breakdown
    domain_counts: dict[str, int] = {}
    for e in entries:
        if e["status"] == "imported":
            d = e.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

    # Source breakdown
    source_counts: dict[str, int] = {}
    for e in entries:
        if e["status"] == "imported":
            s = e.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

    # Recent imports
    recent = [e for e in entries if e["status"] == "imported"][-5:]

    lines = ["**Reference Import Metrics**\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total attempts | {total} |")
    lines.append(f"| Successfully imported | {imported} |")
    lines.append(f"| No source found | {no_source} |")
    if failed:
        lines.append(f"| Failed | {failed} |")
    if total > 0:
        rate = imported / total * 100
        lines.append(f"| Success rate | {rate:.0f}% |")
    lines.append("")

    if domain_counts:
        lines.append("**By Domain:**")
        for d, c in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            display = d.replace("_", " ").title()
            lines.append(f"- {display}: {c}")
        lines.append("")

    if source_counts:
        lines.append("**By Source:**")
        for s, c in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {s.title()}: {c}")
        lines.append("")

    if recent:
        lines.append("**Recent Imports:**")
        for e in reversed(recent):
            lines.append(f"- [{e.get('timestamp', '?')}] **{e.get('title', '?')}** ({e.get('source', '?')})")
        lines.append("")

    # Count vault articles
    if REFERENCES_DIR.exists():
        article_count = len(list(REFERENCES_DIR.glob("*.md")))
        lines.append(f"Vault articles: **{article_count}** in `References/`")

    return "\n".join(lines)


def list_reference_articles() -> str:
    """List all reference articles in the vault."""
    if not REFERENCES_DIR.exists():
        return "No reference articles yet."

    articles = sorted(REFERENCES_DIR.glob("*.md"))
    if not articles:
        return "No reference articles yet."

    lines = [f"**Reference Articles ({len(articles)}):**\n"]
    for a in articles[:20]:
        name = a.stem.replace("_", " ")
        lines.append(f"- {name}")
    if len(articles) > 20:
        lines.append(f"- ... and {len(articles) - 20} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def get_ref_enrichment_tools() -> list:
    """Get agent tools for the reference enrichment system."""
    from .core import create_tool

    return [
        create_tool(
            "import_reference",
            (
                "Import a reference article from Wikipedia into the Obsidian "
                "vault. Provide a topic or query to search for. The article "
                "will be saved as a .md file in Permanent/References/ and "
                "also added to the facts database."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or question to find a reference for",
                    },
                },
                "required": ["query"],
            },
            lambda query: _tool_import_reference(query),
        ),
        create_tool(
            "get_import_metrics",
            (
                "Get metrics on reference imports — success rate, counts by "
                "domain and source, recent imports. Use when asked about "
                "knowledge enrichment status."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_import_metrics(),
        ),
        create_tool(
            "list_references",
            (
                "List all reference articles currently in the Obsidian vault. "
                "Use when the user asks what references are available."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: list_reference_articles(),
        ),
    ]


def _tool_import_reference(query: str) -> str:
    """Tool wrapper for on-demand reference import."""
    result = enrich_from_query(query)
    status = result["status"]
    title = result.get("title", "")

    if status == "imported":
        return (
            f"Imported **{title}** from {result['source']} "
            f"to vault and facts DB."
        )
    elif status == "already_exists":
        return f"Reference for **{title}** already exists in the vault."
    else:
        return f"No reference source found for '{query}'."


# ---------------------------------------------------------------------------
# Scheduled background task
# ---------------------------------------------------------------------------


def _seconds_until_next_run() -> float:
    """Seconds until next daily run at REPORT_HOUR."""
    now = datetime.now()
    next_run = now.replace(
        hour=REPORT_HOUR, minute=0, second=0, microsecond=0,
    )
    if now.hour >= REPORT_HOUR:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


async def ref_enrichment_loop(client: Any, channel_name: str) -> None:
    """Background loop that enriches the vault with reference articles."""
    log.info("[RefEnrichment] Started — runs daily at %d:00", REPORT_HOUR)

    while True:
        try:
            wait = _seconds_until_next_run()
            log.info("[RefEnrichment] Next run in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            # Phase 1: Write articles for accepted resolutions
            from_resolutions = enrich_from_gap_resolutions()

            # Phase 2: Proactively enrich top recurring gaps
            from_frequent = enrich_from_frequent_gaps(top_n=3)

            imported_count = sum(
                1 for r in from_resolutions + from_frequent
                if r.get("status") == "imported"
            )

            if imported_count > 0:
                log.info(
                    "[RefEnrichment] Imported %d new reference articles",
                    imported_count,
                )

                if client and channel_name:
                    titles = [
                        r["title"] for r in from_resolutions + from_frequent
                        if r.get("status") == "imported" and r.get("title")
                    ]
                    lines = [f"**Reference Enrichment** — {imported_count} new article(s)"]
                    for t in titles[:5]:
                        lines.append(f"  - {t}")
                    lines.append("Saved to vault: `Permanent/References/`")
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
                log.info("[RefEnrichment] No new articles to import")

            await asyncio.sleep(60)

        except Exception:
            log.exception("[RefEnrichment] Loop error")
            await asyncio.sleep(300)


def start_ref_enrichment(client: Any, channel_name: str) -> None:
    """Start the daily reference enrichment background task."""
    from .task_manager import create_monitored_task

    create_monitored_task(ref_enrichment_loop(client, channel_name), "ref-enrichment", critical=True)
    log.info("[RefEnrichment] Background task started")
