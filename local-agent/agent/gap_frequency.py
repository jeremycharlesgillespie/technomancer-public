"""
Gap Frequency Tracker - Measure how often knowledge gaps recur by topic.

Parses the full gap history (open + resolved), scans conversation logs for
additional uncertainty signals, clusters similar queries using keyword
overlap, and produces a frequency-ranked priority list showing which
topics need content the most urgently.

Exposes agent tools for on-demand reports and a scheduled weekly task.
"""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings
from .domain_coverage import classify_to_primary_domain
from .knowledge_gaps import GAPS_FILE, UNCERTAINTY_PHRASES, FAILURE_PHRASES

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
CONVERSATIONS_DIR = VAULT_PATH / "Conversations"
REPORTS_DIR = VAULT_PATH / "Permanent" / "gap_frequency"

# Schedule: Wednesday 6 AM (mid-week, after domain coverage on Monday)
REPORT_DAY = 2  # Wednesday
REPORT_HOUR = 6

# ---------------------------------------------------------------------------
# Stop words shared by keyword extraction functions
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "what", "where", "when", "which", "that", "this", "does", "have",
    "with", "from", "about", "your", "the", "and", "for", "are", "how",
    "can", "tell", "know", "please", "would", "could", "should", "mean",
    "means", "there", "been", "being", "will", "just", "some", "than",
    "very", "much", "more", "most", "also", "only", "like", "into",
    "them", "they", "their", "these", "those", "then", "here",
})


# ---------------------------------------------------------------------------
# Gap parsing — all gaps (open + resolved)
# ---------------------------------------------------------------------------


def _parse_all_gaps() -> list[dict[str, str]]:
    """
    Parse ALL gap entries from knowledge_gaps.md (both open and resolved).

    Returns list of dicts with keys: gap_type, timestamp, query,
    response_snippet, status (open/resolved).
    """
    if not GAPS_FILE.exists():
        return []

    content = GAPS_FILE.read_text(encoding="utf-8")
    gaps: list[dict[str, str]] = []

    for section, status in [("Open Gaps", "open"), ("Resolved", "resolved")]:
        pattern = rf"## {section}\s*\n(.*?)(?=## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue

        entries = re.split(r"(?=- \*\*\[)", match.group(1))
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            type_match = re.search(r"\*\*\[(\w+)\]\*\*", entry)
            gap_type = type_match.group(1).lower() if type_match else "unknown"

            ts_match = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})", entry)
            timestamp = ts_match.group(1) if ts_match else ""

            query_match = re.search(r"\*\*Query:\*\*\s*(.+)", entry)
            query = query_match.group(1).strip() if query_match else ""

            resp_match = re.search(r"\*\*Response:\*\*\s*(.+)", entry)
            snippet = resp_match.group(1).strip() if resp_match else ""

            if query:
                gaps.append({
                    "gap_type": gap_type,
                    "timestamp": timestamp,
                    "query": query,
                    "response_snippet": snippet,
                    "status": status,
                    "source": "gap_log",
                })

    return gaps


# ---------------------------------------------------------------------------
# Conversation log scanning for uncertainty signals
# ---------------------------------------------------------------------------


def _scan_conversations_for_gaps(days: int = 30) -> list[dict[str, str]]:
    """
    Retroactively scan conversation logs for responses containing
    uncertainty or failure phrases.

    This catches gaps that may not have been logged by the real-time
    detector (e.g., before it was installed, or edge cases).
    """
    all_phrases = UNCERTAINTY_PHRASES + FAILURE_PHRASES
    found: list[dict[str, str]] = []
    today = datetime.now().date()

    for i in range(days):
        day = today - timedelta(days=i)
        log_file = CONVERSATIONS_DIR / f"{day.strftime('%Y-%m-%d')}.md"
        if not log_file.exists():
            continue

        content = log_file.read_text(encoding="utf-8")
        entries = re.findall(
            r"### (\d{2}:\d{2}:\d{2}) - (.+?)\n\*\*Q:\*\* (.+?)\n\*\*A:\*\* (.+?)(?=\n###|\n---|\Z)",
            content,
            re.DOTALL,
        )

        for time_str, _user, query, response in entries:
            response_lower = response.lower()
            matched = False

            for phrase in FAILURE_PHRASES:
                if phrase in response_lower:
                    found.append({
                        "gap_type": "failure",
                        "timestamp": f"{day.strftime('%Y-%m-%d')} {time_str[:5]}",
                        "query": query.strip(),
                        "response_snippet": response[:200].strip(),
                        "status": "detected",
                        "source": "conversation_scan",
                    })
                    matched = True
                    break

            if not matched:
                for phrase in UNCERTAINTY_PHRASES:
                    if phrase in response_lower:
                        found.append({
                            "gap_type": "uncertainty",
                            "timestamp": f"{day.strftime('%Y-%m-%d')} {time_str[:5]}",
                            "query": query.strip(),
                            "response_snippet": response[:200].strip(),
                            "status": "detected",
                            "source": "conversation_scan",
                        })
                        break

    return found


# ---------------------------------------------------------------------------
# Keyword extraction & similarity clustering
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _query_similarity(kw_a: set[str], kw_b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not kw_a or not kw_b:
        return 0.0
    intersection = kw_a & kw_b
    union = kw_a | kw_b
    return len(intersection) / len(union)


def cluster_gaps(
    gaps: list[dict[str, str]], similarity_threshold: float = 0.3
) -> list[dict[str, Any]]:
    """
    Cluster similar gap queries together.

    Uses keyword overlap (Jaccard similarity) to group queries that are
    about the same topic.  Each cluster gets a representative label,
    a frequency count, and the domain classification.

    Returns list of cluster dicts sorted by frequency (descending):
        {label, domain, count, gap_types, queries, first_seen, last_seen}
    """
    if not gaps:
        return []

    # Pre-compute keywords for each gap
    gap_keywords = [_extract_keywords(g["query"]) for g in gaps]

    # Greedy clustering: assign each gap to the first cluster it matches
    clusters: list[dict[str, Any]] = []
    assigned = [False] * len(gaps)

    for i, gap in enumerate(gaps):
        if assigned[i]:
            continue

        # Start new cluster with this gap
        cluster_indices = [i]
        assigned[i] = True

        for j in range(i + 1, len(gaps)):
            if assigned[j]:
                continue
            sim = _query_similarity(gap_keywords[i], gap_keywords[j])
            if sim >= similarity_threshold:
                cluster_indices.append(j)
                assigned[j] = True

        # Build cluster summary
        cluster_gaps_list = [gaps[idx] for idx in cluster_indices]
        all_keywords: Counter[str] = Counter()
        for idx in cluster_indices:
            all_keywords.update(gap_keywords[idx])

        # Label = top 3 most common keywords
        top_kws = [kw for kw, _ in all_keywords.most_common(3)]
        label = " + ".join(top_kws) if top_kws else "unknown"

        # Domain from the most representative query (first one)
        domain = classify_to_primary_domain(gaps[cluster_indices[0]]["query"])

        # Timestamps
        timestamps = []
        for g in cluster_gaps_list:
            ts = g.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(datetime.strptime(ts, "%Y-%m-%d %H:%M"))
                except ValueError:
                    pass

        gap_types: Counter[str] = Counter()
        for g in cluster_gaps_list:
            gap_types[g.get("gap_type", "unknown")] += 1

        clusters.append({
            "label": label,
            "domain": domain,
            "count": len(cluster_indices),
            "gap_types": dict(gap_types),
            "queries": [gaps[idx]["query"] for idx in cluster_indices],
            "first_seen": min(timestamps).strftime("%Y-%m-%d") if timestamps else "",
            "last_seen": max(timestamps).strftime("%Y-%m-%d") if timestamps else "",
        })

    # Sort by frequency (descending)
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_frequency_report(days: int = 30) -> str:
    """
    Generate a gap frequency report as Markdown.

    Combines gap log entries with retroactive conversation scanning,
    clusters by similarity, and produces a ranked priority list.
    """
    logged_gaps = _parse_all_gaps()
    scanned_gaps = _scan_conversations_for_gaps(days)

    # Deduplicate: if a scanned gap's query+timestamp matches a logged gap, skip it
    logged_keys = {
        (g["query"][:80], g["timestamp"]) for g in logged_gaps
    }
    unique_scanned = [
        g for g in scanned_gaps
        if (g["query"][:80], g["timestamp"]) not in logged_keys
    ]

    all_gaps = logged_gaps + unique_scanned
    if not all_gaps:
        return ""

    clusters = cluster_gaps(all_gaps)
    if not clusters:
        return ""

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_label = week_start.strftime("%Y-W%V")

    lines: list[str] = []
    lines.append(f"# Gap Frequency Report — {week_label}")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Analysis window: {days} days\n")

    # Summary
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total gap occurrences | {len(all_gaps)} |")
    lines.append(f"| From gap log | {len(logged_gaps)} |")
    lines.append(f"| From conversation scan | {len(unique_scanned)} |")
    lines.append(f"| Distinct topic clusters | {len(clusters)} |")
    recurring = sum(1 for c in clusters if c["count"] >= 2)
    lines.append(f"| Recurring clusters (2+ occurrences) | {recurring} |")
    lines.append("")

    # Priority ranking table
    lines.append("## Priority Ranking\n")
    lines.append(
        "Topics ranked by gap frequency. Higher frequency = more urgent "
        "need for content.\n"
    )
    lines.append("| Rank | Topic | Domain | Frequency | Type | First Seen | Last Seen |")
    lines.append("|:----:|-------|--------|:---------:|------|:----------:|:---------:|")

    for rank, cluster in enumerate(clusters, 1):
        domain_display = cluster["domain"].replace("_", " ").title()
        types = ", ".join(
            f"{t}({n})" for t, n in sorted(cluster["gap_types"].items())
        )
        lines.append(
            f"| {rank} | {cluster['label']} | {domain_display} | "
            f"**{cluster['count']}** | {types} | "
            f"{cluster['first_seen']} | {cluster['last_seen']} |"
        )

    lines.append("")

    # Detailed clusters (top 10)
    top_clusters = clusters[:10]
    if top_clusters:
        lines.append("## Top Clusters — Detail\n")
        for rank, cluster in enumerate(top_clusters, 1):
            domain_display = cluster["domain"].replace("_", " ").title()
            lines.append(
                f"### {rank}. {cluster['label']} "
                f"({cluster['count']}x, {domain_display})\n"
            )
            for q in cluster["queries"][:5]:
                lines.append(f"- {q[:120]}")
            if len(cluster["queries"]) > 5:
                lines.append(f"- ... and {len(cluster['queries']) - 5} more")
            lines.append("")

    # Domain frequency aggregation
    domain_freq: Counter[str] = Counter()
    for cluster in clusters:
        domain_freq[cluster["domain"]] += cluster["count"]

    if domain_freq:
        lines.append("## Gap Frequency by Domain\n")
        lines.append("| Domain | Total Occurrences | Clusters |")
        lines.append("|--------|:-----------------:|:--------:|")
        domain_clusters: Counter[str] = Counter()
        for cluster in clusters:
            domain_clusters[cluster["domain"]] += 1
        for domain, freq in domain_freq.most_common():
            display = domain.replace("_", " ").title()
            lines.append(f"| {display} | {freq} | {domain_clusters[domain]} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")
    if recurring > 0:
        top3 = [c["label"] for c in clusters[:3] if c["count"] >= 2]
        if top3:
            lines.append(
                f"- **Highest priority:** {', '.join(top3)} — "
                f"these topics recur most frequently and should be "
                f"added to the facts database first."
            )
    single_occurrence = sum(1 for c in clusters if c["count"] == 1)
    if single_occurrence > 0:
        lines.append(
            f"- **{single_occurrence} one-off gap(s)** — these may not "
            f"need dedicated content unless they recur."
        )
    if len(unique_scanned) > len(logged_gaps) * 0.5 and len(logged_gaps) > 0:
        lines.append(
            f"- **{len(unique_scanned)} gaps found via conversation scan** "
            f"that weren't in the gap log. The real-time gap detector may "
            f"be missing some uncertainty signals."
        )
    if not clusters:
        lines.append("- No knowledge gaps detected. The system is well-covered.")
    lines.append("")

    return "\n".join(lines)


def write_frequency_report(days: int = 30) -> str:
    """Write the frequency report to the vault. Returns file path or ''."""
    report = generate_frequency_report(days)
    if not report:
        return ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    filename = f"{week_start.strftime('%Y-W%V')}.md"
    report_path = REPORTS_DIR / filename

    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote gap frequency report: %s", report_path)
    return str(report_path)


def format_discord_summary(clusters: list[dict[str, Any]] | None = None, days: int = 30) -> str:
    """Create a short Discord-friendly frequency summary."""
    if clusters is None:
        all_gaps = _parse_all_gaps() + _scan_conversations_for_gaps(days)
        if not all_gaps:
            return ""
        clusters = cluster_gaps(all_gaps)

    if not clusters:
        return ""

    total = sum(c["count"] for c in clusters)
    recurring = [c for c in clusters if c["count"] >= 2]

    lines = ["**Gap Frequency Report**"]
    lines.append(f"Total gap occurrences: **{total}** across **{len(clusters)}** topics")

    if recurring:
        lines.append(f"Recurring topics (**{len(recurring)}**):")
        for c in recurring[:5]:
            domain = c["domain"].replace("_", " ").title()
            lines.append(f"  - {c['label']} — **{c['count']}x** ({domain})")

    lines.append("Full report: `gap_frequency/`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def _tool_gap_frequency(days: int = 30) -> str:
    """Tool: get gap frequency ranking."""
    logged = _parse_all_gaps()
    scanned = _scan_conversations_for_gaps(days)
    logged_keys = {(g["query"][:80], g["timestamp"]) for g in logged}
    unique_scanned = [
        g for g in scanned if (g["query"][:80], g["timestamp"]) not in logged_keys
    ]
    all_gaps = logged + unique_scanned

    if not all_gaps:
        return "No knowledge gaps found."

    clusters = cluster_gaps(all_gaps)
    if not clusters:
        return "No clusters formed."

    lines = [f"**Gap Frequency — Top Topics (last {days} days)**\n"]
    for rank, c in enumerate(clusters[:15], 1):
        domain = c["domain"].replace("_", " ").title()
        lines.append(f"{rank}. **{c['label']}** — {c['count']}x ({domain})")
        if c["count"] >= 2:
            lines.append(f"   First: {c['first_seen']} | Last: {c['last_seen']}")

    recurring = sum(1 for c in clusters if c["count"] >= 2)
    lines.append(f"\n{len(all_gaps)} total occurrences, {len(clusters)} topics, {recurring} recurring")
    return "\n".join(lines)


def get_gap_frequency_tools() -> list:
    """Get agent tools for gap frequency analysis."""
    from .core import create_tool

    return [
        create_tool(
            "get_gap_frequency",
            (
                "Get a ranked list of the most frequent knowledge gap topics. "
                "Shows which topics the system fails on most often, helping "
                "prioritize what content to add. Use when asked about recurring "
                "gaps, common failures, or knowledge priorities."
            ),
            {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default 30)",
                    },
                },
                "required": [],
            },
            lambda days=30: _tool_gap_frequency(days),
        ),
    ]


# ---------------------------------------------------------------------------
# Scheduled background task
# ---------------------------------------------------------------------------


def _seconds_until_next_report() -> float:
    """Calculate seconds until the next report time (Wednesday 6 AM)."""
    now = datetime.now()
    days_ahead = REPORT_DAY - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= REPORT_HOUR):
        days_ahead += 7

    next_report = now.replace(
        hour=REPORT_HOUR, minute=0, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    return (next_report - now).total_seconds()


async def gap_frequency_loop(client: Any, channel_name: str) -> None:
    """Background loop that generates weekly gap frequency reports."""
    log.info(
        "[GapFrequency] Started — reports every Wednesday at %d:00",
        REPORT_HOUR,
    )

    while True:
        try:
            wait = _seconds_until_next_report()
            log.info("[GapFrequency] Next report in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            report_path = write_frequency_report()
            if report_path:
                log.info("[GapFrequency] Report written: %s", report_path)

                summary = format_discord_summary()
                if summary and client and channel_name:
                    for guild in client.guilds:
                        channel = next(
                            (
                                c for c in guild.text_channels
                                if c.name == channel_name
                            ),
                            None,
                        )
                        if channel:
                            await channel.send(summary)
                            break
            else:
                log.info("[GapFrequency] No gaps found — skipping report")

            await asyncio.sleep(60)

        except Exception:
            log.exception("[GapFrequency] Loop error")
            await asyncio.sleep(300)


def start_gap_frequency(client: Any, channel_name: str) -> None:
    """Start the weekly gap frequency background task."""
    from .task_manager import create_monitored_task

    create_monitored_task(gap_frequency_loop(client, channel_name), "gap-frequency", critical=True)
    log.info("[GapFrequency] Background task started")
