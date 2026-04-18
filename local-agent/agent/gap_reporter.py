"""
Knowledge Gap Reporter - Weekly automated reports on unresolved knowledge gaps.

Scans the knowledge gaps log, cross-references with the facts database,
groups gaps by topic and type, and generates structured weekly summary
reports in the Obsidian vault.  Runs as a scheduled background task
(every Sunday at midnight) and posts a brief summary to Discord.
"""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings
from .facts_db import lookup_fact
from .knowledge_gaps import GAPS_FILE, get_gap_summary, get_open_gaps

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
REPORTS_DIR = VAULT_PATH / "Permanent" / "gap_reports"

# Day of week to generate report (0=Monday, 6=Sunday)
REPORT_DAY = 6  # Sunday
REPORT_HOUR = 0  # Midnight


# ---------------------------------------------------------------------------
# Gap parsing
# ---------------------------------------------------------------------------


def _parse_open_gaps() -> list[dict[str, str]]:
    """
    Parse individual gap entries from the knowledge_gaps.md file.

    Returns a list of dicts with keys: gap_type, timestamp, escalated,
    query, response_snippet.
    """
    if not GAPS_FILE.exists():
        return []

    content = GAPS_FILE.read_text(encoding="utf-8")
    open_match = re.search(
        r"## Open Gaps\s*\n(.*?)(?=## Resolved|$)", content, re.DOTALL
    )
    if not open_match:
        return []

    open_section = open_match.group(1)
    entries = re.split(r"(?=- \*\*\[)", open_section)

    gaps: list[dict[str, str]] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Parse type: **[UNCERTAINTY]** or **[FAILURE]**
        type_match = re.search(r"\*\*\[(\w+)\]\*\*", entry)
        gap_type = type_match.group(1).lower() if type_match else "unknown"

        # Parse timestamp
        ts_match = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})", entry)
        timestamp = ts_match.group(1) if ts_match else ""

        # Check if escalated
        escalated = "[escalated]" in entry.lower()

        # Parse query
        query_match = re.search(r"\*\*Query:\*\*\s*(.+)", entry)
        query = query_match.group(1).strip() if query_match else ""

        # Parse response snippet
        resp_match = re.search(r"\*\*Response:\*\*\s*(.+)", entry)
        snippet = resp_match.group(1).strip() if resp_match else ""

        gaps.append({
            "gap_type": gap_type,
            "timestamp": timestamp,
            "escalated": str(escalated),
            "query": query,
            "response_snippet": snippet,
        })

    return gaps


# ---------------------------------------------------------------------------
# Cross-reference with facts DB
# ---------------------------------------------------------------------------


def _cross_reference_gaps(gaps: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    For each gap, check if the facts DB already has relevant entries.

    Returns the gaps list augmented with a 'facts_matches' key containing
    any matching facts (which means the gap might be resolvable).
    """
    enriched: list[dict[str, Any]] = []
    for gap in gaps:
        query = gap.get("query", "")
        matches: list[dict[str, Any]] = []
        if query:
            # Extract key terms (words > 3 chars, skip common words)
            stop_words = {
                "what", "where", "when", "which", "that", "this",
                "does", "have", "with", "from", "about", "your",
                "the", "and", "for", "are", "how", "can", "tell",
                "know", "please", "would", "could", "should",
            }
            words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())
            key_terms = [w for w in words if w not in stop_words]

            for term in key_terms[:5]:  # Limit lookups per gap
                results = lookup_fact(term)
                for r in results:
                    fact_key = r.get("key", "")
                    if fact_key and not any(
                        m.get("key") == fact_key for m in matches
                    ):
                        matches.append(r)

        enriched_gap: dict[str, Any] = dict(gap)
        enriched_gap["facts_matches"] = matches
        enriched.append(enriched_gap)

    return enriched


# ---------------------------------------------------------------------------
# Topic grouping
# ---------------------------------------------------------------------------


def _extract_topic_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query for topic grouping."""
    stop_words = {
        "what", "where", "when", "which", "that", "this", "does",
        "have", "with", "from", "about", "your", "the", "and", "for",
        "are", "how", "can", "tell", "know", "please", "would",
        "could", "should", "mean", "means", "there", "been", "being",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    return [w for w in words if w not in stop_words]


def _group_by_topic(gaps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group gaps by recurring topic keywords.

    Gaps that share keywords get grouped together.  Gaps with no clear
    topic go into "uncategorized".
    """
    # Count keyword frequency across all gaps
    keyword_counter: Counter[str] = Counter()
    gap_keywords: list[list[str]] = []
    for gap in gaps:
        kws = _extract_topic_keywords(gap.get("query", ""))
        gap_keywords.append(kws)
        keyword_counter.update(set(kws))  # Count each keyword once per gap

    # Use keywords that appear in 2+ gaps as topic labels
    recurring = {k for k, v in keyword_counter.items() if v >= 2}

    groups: dict[str, list[dict[str, Any]]] = {}
    for gap, kws in zip(gaps, gap_keywords):
        matched_topics = [k for k in kws if k in recurring]
        if matched_topics:
            # Use the most frequent recurring keyword as topic
            topic = max(matched_topics, key=lambda k: keyword_counter[k])
            groups.setdefault(topic, []).append(gap)
        else:
            groups.setdefault("uncategorized", []).append(gap)

    return groups


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_gap_report() -> str:
    """
    Generate a structured weekly knowledge gap report as Markdown.

    Returns the full report text, or an empty string if there are no gaps.
    """
    gaps = _parse_open_gaps()
    if not gaps:
        return ""

    enriched = _cross_reference_gaps(gaps)
    topics = _group_by_topic(enriched)
    summary_text = get_gap_summary()

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_label = week_start.strftime("%Y-W%V")

    # Count stats
    total = len(gaps)
    failures = sum(1 for g in gaps if g["gap_type"] == "failure")
    uncertainties = sum(1 for g in gaps if g["gap_type"] == "uncertainty")
    escalated = sum(1 for g in gaps if g.get("escalated") == "True")
    resolvable = sum(
        1 for g in enriched if g.get("facts_matches")
    )

    # Count gaps from the last 7 days
    recent_count = 0
    cutoff = now - timedelta(days=7)
    for g in gaps:
        ts = g.get("timestamp", "")
        if ts:
            try:
                gap_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
                if gap_dt >= cutoff:
                    recent_count += 1
            except ValueError:
                pass

    # Build report
    lines: list[str] = []
    lines.append(f"# Knowledge Gap Report — {week_label}")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total open gaps | {total} |")
    lines.append(f"| Uncertainty signals | {uncertainties} |")
    lines.append(f"| Outright failures | {failures} |")
    lines.append(f"| Escalated to Claude | {escalated} |")
    lines.append(f"| New this week | {recent_count} |")
    lines.append(f"| Potentially resolvable via facts DB | {resolvable} |")
    lines.append("")

    # Resolvable gaps section
    if resolvable > 0:
        lines.append("## Potentially Resolvable Gaps\n")
        lines.append(
            "These gaps have matching entries in the facts database. "
            "The knowledge may already exist but wasn't retrieved during "
            "the original query.\n"
        )
        for gap in enriched:
            if gap.get("facts_matches"):
                query = gap.get("query", "N/A")
                lines.append(f"- **Query:** {query}")
                for fact in gap["facts_matches"][:3]:
                    lines.append(
                        f"  - Matching fact: [{fact.get('category', '?')}] "
                        f"**{fact.get('key', '?')}** — {fact.get('value', '?')[:120]}"
                    )
                lines.append("")

    # Topic groups
    lines.append("## Gaps by Topic\n")
    # Sort topics: named topics first (alphabetically), uncategorized last
    sorted_topics = sorted(
        topics.keys(), key=lambda t: (t == "uncategorized", t)
    )
    for topic in sorted_topics:
        topic_gaps = topics[topic]
        label = topic.capitalize() if topic != "uncategorized" else "Uncategorized"
        lines.append(f"### {label} ({len(topic_gaps)} gaps)\n")
        for gap in topic_gaps:
            gap_type = gap.get("gap_type", "?").upper()
            query = gap.get("query", "N/A")
            ts = gap.get("timestamp", "")
            esc = " [escalated]" if gap.get("escalated") == "True" else ""
            lines.append(f"- **[{gap_type}]** ({ts}{esc}) {query}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")
    if failures > 0:
        lines.append(
            f"- **{failures} failure(s)** indicate topics where the system "
            f"has no useful response. Prioritize adding facts or training "
            f"data for these queries."
        )
    if resolvable > 0:
        lines.append(
            f"- **{resolvable} gap(s)** may be resolvable by improving "
            f"fact retrieval. The facts DB already contains relevant entries."
        )
    if escalated > 0:
        lines.append(
            f"- **{escalated} gap(s)** required Claude escalation. "
            f"Consider adding these topics to the local knowledge base "
            f"to reduce API costs."
        )
    recurring_topics = [
        t for t in sorted_topics
        if t != "uncategorized" and len(topics[t]) >= 3
    ]
    if recurring_topics:
        topic_list = ", ".join(recurring_topics)
        lines.append(
            f"- **Recurring themes:** {topic_list} — these topics appear "
            f"in 3+ gaps and should be prioritized for knowledge acquisition."
        )
    if not any([failures, resolvable, escalated, recurring_topics]):
        lines.append("- No high-priority recommendations at this time.")
    lines.append("")

    return "\n".join(lines)


def write_weekly_report() -> str:
    """
    Generate and write the weekly report to the vault.

    Returns the file path of the written report, or an empty string if
    there were no gaps to report.
    """
    report = generate_gap_report()
    if not report:
        return ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    filename = f"{week_start.strftime('%Y-W%V')}.md"
    report_path = REPORTS_DIR / filename

    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote gap report: %s", report_path)
    return str(report_path)


def format_discord_summary(report_path: str) -> str:
    """
    Create a short Discord-friendly summary for posting to a channel.
    """
    gaps = _parse_open_gaps()
    if not gaps:
        return ""

    total = len(gaps)
    failures = sum(1 for g in gaps if g["gap_type"] == "failure")
    uncertainties = sum(1 for g in gaps if g["gap_type"] == "uncertainty")

    enriched = _cross_reference_gaps(gaps)
    resolvable = sum(1 for g in enriched if g.get("facts_matches"))

    now = datetime.now()
    cutoff = now - timedelta(days=7)
    recent = 0
    for g in gaps:
        ts = g.get("timestamp", "")
        if ts:
            try:
                if datetime.strptime(ts, "%Y-%m-%d %H:%M") >= cutoff:
                    recent += 1
            except ValueError:
                pass

    lines = [
        f"**Weekly Knowledge Gap Report**",
        f"Open gaps: **{total}** ({uncertainties} uncertainty, {failures} failure)",
        f"New this week: **{recent}**",
    ]
    if resolvable:
        lines.append(f"Potentially resolvable via facts DB: **{resolvable}**")
    lines.append(f"Full report saved to vault: `gap_reports/`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduled background task
# ---------------------------------------------------------------------------


def _seconds_until_next_report() -> float:
    """Calculate seconds until the next report time (Sunday midnight)."""
    now = datetime.now()
    # Find next Sunday (or today if it's Sunday and before report hour)
    days_ahead = REPORT_DAY - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= REPORT_HOUR):
        days_ahead += 7

    next_report = now.replace(
        hour=REPORT_HOUR, minute=0, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    return (next_report - now).total_seconds()


async def gap_reporter_loop(client: Any, channel_name: str) -> None:
    """Background loop that generates weekly gap reports."""
    log.info(
        "[GapReporter] Started — reports every Sunday at %d:00",
        REPORT_HOUR,
    )

    while True:
        try:
            wait = _seconds_until_next_report()
            log.info(
                "[GapReporter] Next report in %.1f hours",
                wait / 3600,
            )
            await asyncio.sleep(wait)

            report_path = write_weekly_report()
            if report_path:
                log.info("[GapReporter] Report written: %s", report_path)

                # Post summary to Discord
                summary = format_discord_summary(report_path)
                if summary and client and channel_name:
                    for guild in client.guilds:
                        channel = next(
                            (
                                c
                                for c in guild.text_channels
                                if c.name == channel_name
                            ),
                            None,
                        )
                        if channel:
                            await channel.send(summary)
                            break
            else:
                log.info("[GapReporter] No open gaps — skipping report")

            # Sleep a bit to avoid double-firing
            await asyncio.sleep(60)

        except Exception:
            log.exception("[GapReporter] Loop error")
            await asyncio.sleep(300)


def start_gap_reporter(client: Any, channel_name: str) -> None:
    """Start the weekly gap reporter background task."""
    from .task_manager import create_monitored_task

    create_monitored_task(gap_reporter_loop(client, channel_name), "gap-reporter", critical=True)
    log.info("[GapReporter] Background task started")
