"""
Knowledge Gap Tracking - Detect and log queries where the LLM is uncertain.

Intercepts responses after generation, evaluates for uncertainty signals,
and logs gaps to the Obsidian vault for weekly review. This creates a
feedback loop for discovering what knowledge the system is missing.

Single source of truth: Permanent/knowledge_gaps.md in the vault.
"""

import re
from datetime import datetime
from pathlib import Path

from .config import settings

# Obsidian vault location - SINGLE SOURCE OF TRUTH
VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
GAPS_FILE = VAULT_PATH / "Permanent" / "knowledge_gaps.md"

# Phrases that signal the LLM is uncertain or lacks knowledge
UNCERTAINTY_PHRASES = [
    "i'm not sure",
    "i'm not certain",
    "i don't know",
    "i cannot verify",
    "i can't verify",
    "i don't have",
    "i'm unable to",
    "i can't find",
    "i couldn't find",
    "beyond my knowledge",
    "outside my training",
    "i may be wrong",
    "take this with a grain",
    "i'd recommend checking",
    "you might want to check",
    "you should check",
    "check the official",
    "i don't have access to",
    "i'm not aware of",
    "i haven't been trained on",
    "no reliable information",
    "i can't confirm",
    "i cannot confirm",
]

# Phrases that indicate outright failure to answer
FAILURE_PHRASES = [
    "i don't have enough information",
    "i cannot answer",
    "i can't answer",
    "unable to provide",
    "i lack the knowledge",
    "this is outside my",
    "i have no data on",
]


def detect_knowledge_gap(
    user_query: str,
    response: str,
    was_escalated: bool = False,
) -> dict | None:
    """
    Analyze a response for signs of uncertainty or knowledge gaps.

    Args:
        user_query: The original user message
        response: The LLM's response text
        was_escalated: Whether this response was escalated to Claude API

    Returns:
        A gap dict with keys (query, response_snippet, gap_type, timestamp)
        or None if no gap detected.
    """
    if not response or not user_query:
        return None

    response_lower = response.lower()

    # Check for failure phrases first (stronger signal)
    for phrase in FAILURE_PHRASES:
        if phrase in response_lower:
            return _build_gap(
                user_query, response, "failure", was_escalated
            )

    # Count uncertainty signals — require at least one match
    matches = [p for p in UNCERTAINTY_PHRASES if p in response_lower]
    if matches:
        return _build_gap(
            user_query, response, "uncertainty", was_escalated
        )

    return None


def _build_gap(
    user_query: str,
    response: str,
    gap_type: str,
    was_escalated: bool,
) -> dict:
    """Build a structured gap entry."""
    # Take a snippet of the response showing the uncertainty
    snippet = response[:300].replace("\n", " ").strip()
    if len(response) > 300:
        snippet += "..."

    return {
        "query": user_query[:500],
        "response_snippet": snippet,
        "gap_type": gap_type,
        "was_escalated": was_escalated,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def log_knowledge_gap(gap: dict) -> str:
    """
    Write a knowledge gap entry to the vault file.
    Always appends to keep a running log.

    Args:
        gap: Dict from detect_knowledge_gap()

    Returns:
        Confirmation message
    """
    entry = _format_gap_entry(gap)

    if not GAPS_FILE.exists():
        GAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        content = f"""# Knowledge Gap Log

Queries where the LLM was uncertain or failed to answer.
Review weekly to identify missing knowledge and update the facts database.

---

## Open Gaps

{entry}
## Resolved

"""
        GAPS_FILE.write_text(content, encoding="utf-8")
    else:
        content = GAPS_FILE.read_text(encoding="utf-8")

        if "## Open Gaps" in content:
            content = content.replace(
                "## Open Gaps\n", f"## Open Gaps\n\n{entry}"
            )
        else:
            content += f"\n{entry}"

        GAPS_FILE.write_text(content, encoding="utf-8")

    return f"Logged knowledge gap: {gap['gap_type']} — {gap['query'][:80]}"


def _format_gap_entry(gap: dict) -> str:
    """Format a gap dict as a markdown list item."""
    escalated = " [escalated]" if gap.get("was_escalated") else ""
    return (
        f"- **[{gap['gap_type'].upper()}]** ({gap['timestamp']}{escalated})\n"
        f"  - **Query:** {gap['query']}\n"
        f"  - **Response:** {gap['response_snippet']}\n\n"
    )


def get_open_gaps() -> str:
    """
    Get all open (unresolved) knowledge gaps from the vault.
    Always reads fresh from disk.

    Returns:
        Formatted list of open gaps
    """
    if not GAPS_FILE.exists():
        return "No knowledge gaps logged yet."

    content = GAPS_FILE.read_text(encoding="utf-8")

    open_match = re.search(
        r"## Open Gaps\s*\n(.*?)(?=## Resolved|$)", content, re.DOTALL
    )

    if open_match:
        open_section = open_match.group(1).strip()
        if open_section:
            items = re.findall(r"- \*\*\[", open_section)
            count = len(items)
            return f"**Open Knowledge Gaps ({count}):**\n\n{open_section}"

    return "No open knowledge gaps."


def resolve_gap(query_fragment: str, resolution: str = "") -> str:
    """
    Move a gap from Open to Resolved by matching a fragment of the query text.

    Args:
        query_fragment: Part of the original query to match
        resolution: Optional note about how the gap was resolved

    Returns:
        Confirmation or error message
    """
    if not GAPS_FILE.exists():
        return "No knowledge gaps file found."

    content = GAPS_FILE.read_text(encoding="utf-8")

    # Find the entry in Open Gaps that matches the fragment
    open_match = re.search(
        r"(## Open Gaps\s*\n)(.*?)(## Resolved)", content, re.DOTALL
    )

    if not open_match:
        return "Could not find Open Gaps / Resolved sections."

    open_section = open_match.group(2)

    # Find the specific gap entry — each entry is a block starting with "- **["
    entries = re.split(r"(?=- \*\*\[)", open_section)
    moved = None
    remaining = []

    for entry in entries:
        if not entry.strip():
            continue
        if query_fragment.lower() in entry.lower() and moved is None:
            moved = entry.strip()
        else:
            remaining.append(entry)

    if not moved:
        return f"No open gap matching '{query_fragment}' found."

    # Rebuild content
    new_open = "\n".join(remaining) + "\n" if remaining else "\n"

    resolved_note = f"\n  - **Resolution:** {resolution}" if resolution else ""
    resolved_date = datetime.now().strftime("%Y-%m-%d")
    resolved_entry = f"{moved}{resolved_note}\n  - **Resolved:** {resolved_date}\n\n"

    # Insert into Resolved section
    new_content = content[: open_match.start()]
    new_content += f"## Open Gaps\n\n{new_open}"
    new_content += f"## Resolved\n\n{resolved_entry}"

    # Preserve anything after the Resolved header
    after_resolved = re.search(r"## Resolved\s*\n(.*)", content, re.DOTALL)
    if after_resolved:
        existing_resolved = after_resolved.group(1).strip()
        if existing_resolved:
            new_content += existing_resolved + "\n"

    GAPS_FILE.write_text(new_content, encoding="utf-8")
    return f"Resolved gap: {query_fragment}"


def get_gap_summary() -> str:
    """
    Get a summary of knowledge gaps for weekly review.

    Returns:
        Summary with counts by type and most recent gaps
    """
    if not GAPS_FILE.exists():
        return "No knowledge gaps logged yet."

    content = GAPS_FILE.read_text(encoding="utf-8")

    # Count open gaps by type
    open_match = re.search(
        r"## Open Gaps\s*\n(.*?)(?=## Resolved|$)", content, re.DOTALL
    )

    open_section = open_match.group(1) if open_match else ""
    uncertainty_count = len(re.findall(r"\*\*\[UNCERTAINTY\]\*\*", open_section))
    failure_count = len(re.findall(r"\*\*\[FAILURE\]\*\*", open_section))
    escalated_count = len(re.findall(r"\[escalated\]", open_section))
    total = uncertainty_count + failure_count

    # Count resolved
    resolved_match = re.search(r"## Resolved\s*\n(.*)", content, re.DOTALL)
    resolved_count = 0
    if resolved_match:
        resolved_count = len(re.findall(r"\*\*\[", resolved_match.group(1)))

    if total == 0 and resolved_count == 0:
        return "No open knowledge gaps."

    return (
        f"**Knowledge Gap Summary:**\n"
        f"- Open: {total} ({uncertainty_count} uncertainty, {failure_count} failure)\n"
        f"- Escalated to Claude: {escalated_count}\n"
        f"- Resolved: {resolved_count}\n"
    )


def get_knowledge_gap_tools() -> list:
    """Get knowledge gap tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "get_knowledge_gaps",
            (
                "Get the list of open knowledge gaps — queries where you were "
                "uncertain or couldn't answer. Use when asked about gaps, "
                "weaknesses, or what you don't know."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_open_gaps(),
        ),
        create_tool(
            "get_knowledge_gap_summary",
            (
                "Get a summary of knowledge gap statistics — counts by type, "
                "escalation rate, and resolution progress. Use for weekly reviews."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_gap_summary(),
        ),
        create_tool(
            "resolve_knowledge_gap",
            (
                "Mark a knowledge gap as resolved after the missing knowledge "
                "has been added or the issue addressed. Provide part of the "
                "original query to match it."
            ),
            {
                "type": "object",
                "properties": {
                    "query_fragment": {
                        "type": "string",
                        "description": "Part of the original query to identify the gap",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "How the gap was resolved (e.g., 'added to memories', 'web search tool covers this')",
                    },
                },
                "required": ["query_fragment"],
            },
            lambda query_fragment, resolution="": resolve_gap(query_fragment, resolution),
        ),
    ]
