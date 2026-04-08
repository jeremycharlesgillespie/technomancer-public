"""
Domain Coverage Tracker - Monitor knowledge coverage across topic domains.

Scans conversation logs, the facts database, and knowledge gaps to build
a coverage matrix showing which knowledge domains are well-covered and
which have gaps.  Generates weekly reports with actionable suggestions
for filling coverage holes.

Runs as a scheduled background task (Monday 6 AM) and posts a summary
to Discord.
"""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings
from .facts_db import get_stats as get_facts_stats, lookup_fact
from .knowledge_gaps import GAPS_FILE

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
CONVERSATIONS_DIR = VAULT_PATH / "Conversations"
REPORTS_DIR = VAULT_PATH / "Permanent" / "domain_coverage"

# Schedule: Monday at 6 AM
REPORT_DAY = 0  # Monday
REPORT_HOUR = 6

# ---------------------------------------------------------------------------
# Domain taxonomy - keyword lists that map queries to domains
# ---------------------------------------------------------------------------

DOMAIN_TAXONOMY: dict[str, list[str]] = {
    "food_cooking": [
        "food", "recipe", "cook", "bake", "ingredient", "meal", "dish",
        "cuisine", "restaurant", "pasta", "spaghetti", "bread", "pizza",
        "sushi", "chocolate", "coffee", "tea", "rice", "vegetable",
        "fruit", "meat", "fish", "dessert", "snack", "nutrition",
        "calorie", "diet", "spice", "sauce", "soup", "salad",
    ],
    "science": [
        "science", "physics", "chemistry", "biology", "experiment",
        "hypothesis", "theory", "molecule", "atom", "cell", "gene",
        "dna", "evolution", "photosynthesis", "gravity", "quantum",
        "particle", "energy", "force", "wave", "electron", "proton",
        "neutron", "element", "compound", "reaction", "organism",
    ],
    "technology": [
        "technology", "computer", "software", "hardware", "programming",
        "code", "algorithm", "database", "api", "server", "cloud",
        "python", "javascript", "html", "css", "machine learning",
        "artificial intelligence", "llm", "neural", "blockchain",
        "cybersecurity", "encryption", "network", "linux", "windows",
        "docker", "kubernetes", "devops", "framework", "library",
    ],
    "geography": [
        "geography", "country", "capital", "continent", "ocean", "river",
        "mountain", "island", "city", "population", "border", "climate",
        "latitude", "longitude", "hemisphere", "equator", "desert",
        "forest", "tundra", "prairie", "valley", "peninsula",
    ],
    "history": [
        "history", "historical", "ancient", "medieval", "renaissance",
        "revolution", "war", "empire", "dynasty", "civilization",
        "century", "era", "king", "queen", "president", "treaty",
        "battle", "colony", "independence", "constitution",
    ],
    "math": [
        "math", "mathematics", "calculate", "equation", "formula",
        "algebra", "geometry", "calculus", "statistics", "probability",
        "number", "fraction", "decimal", "percent", "average", "median",
        "graph", "function", "variable", "theorem", "proof",
    ],
    "language": [
        "language", "grammar", "vocabulary", "translate", "translation",
        "synonym", "antonym", "definition", "spell", "pronunciation",
        "verb", "noun", "adjective", "adverb", "sentence", "paragraph",
        "essay", "writing", "reading", "literature", "poetry",
    ],
    "health_medicine": [
        "health", "medical", "medicine", "doctor", "symptom", "disease",
        "treatment", "therapy", "diagnosis", "vaccine", "virus",
        "bacteria", "infection", "surgery", "prescription", "vitamin",
        "exercise", "fitness", "mental health", "anxiety", "depression",
    ],
    "business_finance": [
        "business", "finance", "money", "investment", "stock", "market",
        "economy", "budget", "tax", "salary", "profit", "revenue",
        "startup", "entrepreneur", "management", "marketing", "sales",
        "accounting", "bank", "loan", "interest", "inflation",
    ],
    "arts_entertainment": [
        "art", "music", "movie", "film", "book", "novel", "painting",
        "sculpture", "theater", "drama", "comedy", "dance", "song",
        "album", "artist", "musician", "actor", "director", "genre",
        "photography", "design", "animation", "game", "gaming",
    ],
    "conversions_units": [
        "convert", "conversion", "unit", "meter", "kilometer", "mile",
        "pound", "kilogram", "fahrenheit", "celsius", "gallon", "liter",
        "inch", "centimeter", "foot", "yard", "ounce", "gram",
        "acre", "hectare", "mph", "knot",
    ],
    "time_dates": [
        "time", "timezone", "date", "calendar", "clock", "hour",
        "minute", "second", "schedule", "deadline", "utc", "gmt",
        "est", "pst", "cst", "daylight saving",
    ],
}


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------


def classify_text(text: str) -> dict[str, int]:
    """
    Classify a text string against the domain taxonomy.

    Returns a dict of {domain: match_count} for all domains with at
    least one keyword match.
    """
    text_lower = text.lower()
    # Tokenize once for word-boundary matching
    words = set(re.findall(r"\b[a-z]+\b", text_lower))

    hits: dict[str, int] = {}
    for domain, keywords in DOMAIN_TAXONOMY.items():
        count = 0
        for kw in keywords:
            # Multi-word keywords: check substring
            if " " in kw:
                if kw in text_lower:
                    count += 1
            else:
                if kw in words:
                    count += 1
        if count > 0:
            hits[domain] = count
    return hits


def classify_to_primary_domain(text: str) -> str:
    """Return the single best-matching domain, or 'uncategorized'."""
    hits = classify_text(text)
    if not hits:
        return "uncategorized"
    return max(hits, key=hits.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Conversation log scanning
# ---------------------------------------------------------------------------


def _parse_conversation_log(filepath: Path) -> list[dict[str, str]]:
    """Parse a daily conversation log file into entries."""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    entries = re.findall(
        r"### (\d{2}:\d{2}:\d{2}) - (.+?)\n\*\*Q:\*\* (.+?)\n\*\*A:\*\* (.+?)(?=\n###|\n---|\Z)",
        content,
        re.DOTALL,
    )
    return [
        {"time": t, "user": u.strip(), "query": q.strip(), "response": r.strip()}
        for t, u, q, r in entries
    ]


def scan_conversations(days: int = 7) -> dict[str, int]:
    """
    Scan recent conversation logs and count domain mentions.

    Returns {domain: conversation_count} across the last N days.
    """
    domain_counts: Counter[str] = Counter()
    today = datetime.now().date()

    for i in range(days):
        day = today - timedelta(days=i)
        log_file = CONVERSATIONS_DIR / f"{day.strftime('%Y-%m-%d')}.md"
        entries = _parse_conversation_log(log_file)

        for entry in entries:
            # Classify the user query (primary signal)
            text = entry["query"]
            domain = classify_to_primary_domain(text)
            domain_counts[domain] += 1

    return dict(domain_counts)


# ---------------------------------------------------------------------------
# Facts DB coverage
# ---------------------------------------------------------------------------


# Map facts_db categories to our domain taxonomy
_FACTS_CATEGORY_TO_DOMAIN: dict[str, str] = {
    "definition": "",  # Definitions span multiple domains, handled specially
    "geography": "geography",
    "timezone": "time_dates",
    "conversion": "conversions_units",
}


def scan_facts_db() -> dict[str, int]:
    """
    Count facts database entries per domain.

    Returns {domain: fact_count}.
    """
    domain_counts: Counter[str] = Counter()

    stats = get_facts_stats()
    categories = stats.get("categories", {})

    for cat, count in categories.items():
        mapped = _FACTS_CATEGORY_TO_DOMAIN.get(cat)
        if mapped:
            domain_counts[mapped] += count
        elif cat == "definition":
            # Definitions span domains - classify each by its key/value
            # Use a heuristic: look up all facts in this category and classify
            _classify_definition_facts(domain_counts)
        else:
            # Custom categories added by user - classify by category name
            domain = classify_to_primary_domain(cat)
            domain_counts[domain] += count

    return dict(domain_counts)


def _classify_definition_facts(domain_counts: Counter) -> None:
    """Classify definition-category facts into specific domains."""
    # Look up known definition keys from seed data to classify them
    definition_keywords = [
        "spaghetti", "pasta", "pizza", "sushi", "bread", "rice",
        "chocolate", "coffee", "tea",
        "algorithm", "api", "database", "machine learning", "blockchain",
        "cloud computing", "llm", "python", "javascript", "sql",
        "photosynthesis", "gravity", "dna", "atom", "evolution",
    ]
    for kw in definition_keywords:
        results = lookup_fact(kw, category="definition")
        if results:
            domain = classify_to_primary_domain(kw)
            domain_counts[domain] += len(results)


# ---------------------------------------------------------------------------
# Knowledge gap scanning
# ---------------------------------------------------------------------------


def scan_knowledge_gaps() -> dict[str, int]:
    """
    Count knowledge gaps per domain from the gaps file.

    Returns {domain: gap_count}.
    """
    domain_counts: Counter[str] = Counter()

    if not GAPS_FILE.exists():
        return {}

    content = GAPS_FILE.read_text(encoding="utf-8")
    open_match = re.search(
        r"## Open Gaps\s*\n(.*?)(?=## Resolved|$)", content, re.DOTALL
    )
    if not open_match:
        return {}

    # Extract queries from gap entries
    queries = re.findall(r"\*\*Query:\*\*\s*(.+)", open_match.group(1))
    for query in queries:
        domain = classify_to_primary_domain(query)
        domain_counts[domain] += 1

    return dict(domain_counts)


# ---------------------------------------------------------------------------
# Coverage matrix and report
# ---------------------------------------------------------------------------


def build_coverage_matrix(days: int = 7) -> dict[str, dict[str, int]]:
    """
    Build a complete domain coverage matrix.

    Returns {domain: {conversations: N, facts: N, gaps: N}} for every
    domain in the taxonomy plus 'uncategorized'.
    """
    conv = scan_conversations(days)
    facts = scan_facts_db()
    gaps = scan_knowledge_gaps()

    # Collect all domains seen
    all_domains = set(DOMAIN_TAXONOMY.keys())
    all_domains.update(conv.keys())
    all_domains.update(facts.keys())
    all_domains.update(gaps.keys())

    matrix: dict[str, dict[str, int]] = {}
    for domain in sorted(all_domains):
        matrix[domain] = {
            "conversations": conv.get(domain, 0),
            "facts": facts.get(domain, 0),
            "gaps": gaps.get(domain, 0),
        }

    return matrix


def _coverage_level(facts: int, gaps: int) -> str:
    """Classify coverage level for a domain."""
    if facts == 0 and gaps > 0:
        return "NONE"
    if facts == 0:
        return "EMPTY"
    if gaps > facts:
        return "LOW"
    if gaps > 0:
        return "PARTIAL"
    return "GOOD"


def generate_coverage_report(days: int = 7) -> str:
    """
    Generate a domain coverage report as Markdown.

    Returns the full report text, or empty string if no data.
    """
    matrix = build_coverage_matrix(days)
    if not matrix:
        return ""

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_label = week_start.strftime("%Y-W%V")

    lines: list[str] = []
    lines.append(f"# Domain Coverage Report — {week_label}")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Analysis window: last {days} days\n")

    # Coverage matrix table
    lines.append("## Coverage Matrix\n")
    lines.append("| Domain | Conversations | Facts | Gaps | Coverage |")
    lines.append("|--------|:------------:|:-----:|:----:|----------|")

    flagged_domains: list[tuple[str, str, dict[str, int]]] = []

    for domain, counts in sorted(matrix.items()):
        conv = counts["conversations"]
        facts = counts["facts"]
        gaps = counts["gaps"]
        level = _coverage_level(facts, gaps)

        # Format domain name nicely
        display = domain.replace("_", " ").title()

        flag = ""
        if level == "NONE":
            flag = " :red_circle:"
            flagged_domains.append((domain, level, counts))
        elif level == "LOW":
            flag = " :orange_circle:"
            flagged_domains.append((domain, level, counts))
        elif level == "EMPTY":
            flag = " :white_circle:"

        lines.append(f"| {display} | {conv} | {facts} | {gaps} | {level}{flag} |")

    lines.append("")

    # Summary stats
    total_conv = sum(c["conversations"] for c in matrix.values())
    total_facts = sum(c["facts"] for c in matrix.values())
    total_gaps = sum(c["gaps"] for c in matrix.values())
    domains_with_data = sum(
        1 for c in matrix.values() if c["conversations"] > 0 or c["facts"] > 0
    )
    taxonomy_size = len(DOMAIN_TAXONOMY)

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Domains tracked | {taxonomy_size} |")
    lines.append(f"| Domains with activity | {domains_with_data} |")
    lines.append(f"| Total conversations ({days}d) | {total_conv} |")
    lines.append(f"| Total facts in DB | {total_facts} |")
    lines.append(f"| Total open knowledge gaps | {total_gaps} |")
    lines.append("")

    # Flagged domains - detailed
    if flagged_domains:
        lines.append("## Flagged Domains\n")
        lines.append(
            "These domains have knowledge gaps but insufficient facts "
            "database coverage.\n"
        )
        for domain, level, counts in flagged_domains:
            display = domain.replace("_", " ").title()
            lines.append(f"### {display} — {level}\n")
            lines.append(f"- Conversations: {counts['conversations']}")
            lines.append(f"- Facts: {counts['facts']}")
            lines.append(f"- Gaps: {counts['gaps']}")

            # Generate suggestions
            suggestions = _suggest_actions(domain, counts)
            if suggestions:
                lines.append("- **Suggested actions:**")
                for s in suggestions:
                    lines.append(f"  - {s}")
            lines.append("")
    else:
        lines.append("## Flagged Domains\n")
        lines.append("No domains flagged — all active domains have adequate coverage.\n")

    # Top queried domains
    conv_domains = sorted(
        ((d, c["conversations"]) for d, c in matrix.items() if c["conversations"] > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    if conv_domains:
        lines.append("## Most Queried Domains (Last 7 Days)\n")
        for domain, count in conv_domains[:10]:
            display = domain.replace("_", " ").title()
            lines.append(f"1. **{display}** — {count} conversations")
        lines.append("")

    return "\n".join(lines)


def _suggest_actions(domain: str, counts: dict[str, int]) -> list[str]:
    """Generate actionable suggestions for a flagged domain."""
    suggestions: list[str] = []
    display = domain.replace("_", " ")

    if counts["facts"] == 0:
        suggestions.append(
            f"Add seed facts for {display} to the facts database "
            f"using `add_fact` tool"
        )
    if counts["gaps"] > 0 and counts["facts"] > 0:
        suggestions.append(
            f"Review the {counts['gaps']} open gap(s) — the facts DB has "
            f"some {display} entries but they may not cover the queried topics"
        )
    if counts["conversations"] > 3 and counts["facts"] < 5:
        suggestions.append(
            f"{display} is frequently discussed ({counts['conversations']} "
            f"conversations) but poorly covered ({counts['facts']} facts). "
            f"Bulk-add reference material."
        )
    if counts["gaps"] > 3:
        suggestions.append(
            f"High gap count ({counts['gaps']}) suggests systematic "
            f"knowledge deficit in {display}. Consider web research or "
            f"Claude escalation to populate this domain."
        )
    return suggestions


def write_coverage_report(days: int = 7) -> str:
    """
    Generate and write the coverage report to the vault.

    Returns the file path, or empty string if no data.
    """
    report = generate_coverage_report(days)
    if not report:
        return ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    filename = f"{week_start.strftime('%Y-W%V')}.md"
    report_path = REPORTS_DIR / filename

    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote domain coverage report: %s", report_path)
    return str(report_path)


def format_discord_summary() -> str:
    """Create a short Discord-friendly coverage summary."""
    matrix = build_coverage_matrix(days=7)
    if not matrix:
        return ""

    total_domains = len(DOMAIN_TAXONOMY)
    active = sum(
        1 for c in matrix.values() if c["conversations"] > 0 or c["facts"] > 0
    )
    total_gaps = sum(c["gaps"] for c in matrix.values())

    flagged = []
    for domain, counts in matrix.items():
        level = _coverage_level(counts["facts"], counts["gaps"])
        if level in ("NONE", "LOW"):
            flagged.append(domain.replace("_", " ").title())

    lines = [
        "**Weekly Domain Coverage Report**",
        f"Domains tracked: **{total_domains}** | Active: **{active}**",
        f"Open knowledge gaps: **{total_gaps}**",
    ]
    if flagged:
        lines.append(f"Flagged (low/no coverage): **{', '.join(flagged[:5])}**")
    else:
        lines.append("All active domains have adequate coverage.")
    lines.append("Full report saved to vault: `domain_coverage/`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduled background task
# ---------------------------------------------------------------------------


def _seconds_until_next_report() -> float:
    """Calculate seconds until the next report time (Monday 6 AM)."""
    now = datetime.now()
    days_ahead = REPORT_DAY - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= REPORT_HOUR):
        days_ahead += 7

    next_report = now.replace(
        hour=REPORT_HOUR, minute=0, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    return (next_report - now).total_seconds()


async def domain_coverage_loop(client: Any, channel_name: str) -> None:
    """Background loop that generates weekly domain coverage reports."""
    log.info(
        "[DomainCoverage] Started — reports every Monday at %d:00",
        REPORT_HOUR,
    )

    while True:
        try:
            wait = _seconds_until_next_report()
            log.info(
                "[DomainCoverage] Next report in %.1f hours",
                wait / 3600,
            )
            await asyncio.sleep(wait)

            report_path = write_coverage_report()
            if report_path:
                log.info("[DomainCoverage] Report written: %s", report_path)

                summary = format_discord_summary()
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
                log.info("[DomainCoverage] No data — skipping report")

            await asyncio.sleep(60)

        except Exception:
            log.exception("[DomainCoverage] Loop error")
            await asyncio.sleep(300)


def start_domain_coverage(client: Any, channel_name: str) -> None:
    """Start the weekly domain coverage background task."""
    asyncio.create_task(domain_coverage_loop(client, channel_name))
    log.info("[DomainCoverage] Background task started")
