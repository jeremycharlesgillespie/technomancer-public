"""
Conversation Context Persistence - Stores structured conversation summaries
to the Obsidian memory system for long-term context retention.

After interactions, the LLM analyzes conversation batches and generates
structured summaries capturing: topic, key decisions, and user preferences.
These persist in Permanent/conversation_summaries.md and are available for
future context injection and tool-based retrieval.
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Vault paths — patchable in tests via monkeypatch
from .config import settings
VAULT_PATH = settings.llm_memory_path
SUMMARIES_FILE = VAULT_PATH / "Permanent" / "conversation_summaries.md"

# Batch size before generating summaries
SUMMARY_BATCH_SIZE = 3
# Minimum chars in a conversation to be worth summarizing
MIN_CONVERSATION_LENGTH = 80

# Module-level buffer
_summary_buffer: list[dict[str, str]] = []
_summary_running = False


def _ensure_summaries_file() -> None:
    """Create summaries file with header if it doesn't exist."""
    SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SUMMARIES_FILE.exists():
        header = (
            "# Conversation Summaries\n\n"
            "Structured summaries of past conversations for long-term context.\n\n---\n"
        )
        SUMMARIES_FILE.write_text(header, encoding="utf-8")


def build_summary_prompt(conversations: list[dict[str, str]]) -> str:
    """Build prompt for the LLM to generate conversation summaries."""
    conv_text = ""
    for conv in conversations:
        conv_text += f"User ({conv['user']}): {conv['message']}\n"
        conv_text += f"Bot: {conv['response'][:500]}\n\n"

    return (
        "Analyze these conversations and generate a structured summary for each one.\n"
        "For each conversation, extract:\n"
        "1. **Topic** - What was discussed (1 sentence)\n"
        "2. **Key Decisions** - Any decisions made or conclusions reached (or \"None\")\n"
        "3. **User Preferences** - Any preferences, opinions, or style choices revealed (or \"None\")\n\n"
        f"CONVERSATIONS:\n{conv_text}\n"
        "Output as a JSON array:\n"
        "```json\n"
        "[\n"
        '  {"user": "username", "topic": "Brief topic description", '
        '"decisions": "Key decisions or None", '
        '"preferences": "User preferences revealed or None"}\n'
        "]\n"
        "```\n\n"
        "Rules:\n"
        "- Be concise — each field should be 1-2 sentences max\n"
        "- Focus on substantive content, skip greetings/small talk\n"
        "- If a conversation is trivial (just a greeting), set topic to the greeting "
        'and decisions/preferences to "None"\n'
        "- Return ONLY the JSON array"
    )


def parse_summary_response(response: str) -> list[dict[str, str]]:
    """Parse LLM summary response into structured data."""
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"\[.*?\]", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return []

    try:
        summaries = json.loads(raw)
        if not isinstance(summaries, list):
            return []
        valid = []
        for s in summaries:
            if isinstance(s, dict) and "topic" in s:
                valid.append(
                    {
                        "user": s.get("user", "unknown"),
                        "topic": s.get("topic", "Unknown topic"),
                        "decisions": s.get("decisions", "None"),
                        "preferences": s.get("preferences", "None"),
                    }
                )
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


def save_summaries(summaries: list[dict[str, str]]) -> int:
    """Save conversation summaries to vault file. Returns count saved."""
    if not summaries:
        return 0

    _ensure_summaries_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    entries = []
    for s in summaries:
        entry = (
            f"\n## {timestamp} - {s['user']}\n"
            f"**Topic:** {s['topic']}\n"
            f"**Decisions:** {s['decisions']}\n"
            f"**Preferences:** {s['preferences']}\n"
        )
        entries.append(entry)

    with open(SUMMARIES_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(entries))

    return len(summaries)


def fallback_summaries(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Generate simple summaries without LLM."""
    summaries = []
    for conv in conversations:
        words = conv["message"].split()[:8]
        topic = " ".join(words)
        if len(conv["message"]) > len(topic):
            topic += "..."

        summaries.append(
            {
                "user": conv["user"],
                "topic": topic,
                "decisions": "None",
                "preferences": "None",
            }
        )
    return summaries


def get_recent_summaries(count: int = 20) -> str:
    """Get the most recent conversation summaries."""
    if not SUMMARIES_FILE.exists():
        return "No conversation summaries yet."

    content = SUMMARIES_FILE.read_text(encoding="utf-8")

    entries = re.findall(
        r"## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) - (.+?)\n"
        r"\*\*Topic:\*\* (.+?)\n"
        r"\*\*Decisions:\*\* (.+?)\n"
        r"\*\*Preferences:\*\* (.+?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )

    if not entries:
        return "No conversation summaries yet."

    recent = entries[-count:]

    lines = [f"## Recent Conversation Summaries ({len(recent)} entries)\n"]
    for ts, user, topic, decisions, preferences in recent:
        lines.append(f"**{ts} - {user}**")
        lines.append(f"  Topic: {topic.strip()}")
        if decisions.strip() != "None":
            lines.append(f"  Decisions: {decisions.strip()}")
        if preferences.strip() != "None":
            lines.append(f"  Preferences: {preferences.strip()}")
        lines.append("")

    return "\n".join(lines)


def search_summaries(query: str) -> str:
    """Search conversation summaries for a query."""
    if not SUMMARIES_FILE.exists():
        return f"No conversation summaries to search."

    content = SUMMARIES_FILE.read_text(encoding="utf-8")
    query_lower = query.lower()

    entries = re.findall(
        r"## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) - (.+?)\n"
        r"\*\*Topic:\*\* (.+?)\n"
        r"\*\*Decisions:\*\* (.+?)\n"
        r"\*\*Preferences:\*\* (.+?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )

    matches = []
    for ts, user, topic, decisions, preferences in entries:
        combined = f"{topic} {decisions} {preferences}".lower()
        if query_lower in combined:
            matches.append((ts, user, topic, decisions, preferences))

    if not matches:
        return f"No conversation summaries matching '{query}'."

    lines = [f"Found {len(matches)} summaries matching '{query}':\n"]
    for ts, user, topic, decisions, preferences in matches[-10:]:
        lines.append(f"**{ts} - {user}**")
        lines.append(f"  Topic: {topic.strip()}")
        if decisions.strip() != "None":
            lines.append(f"  Decisions: {decisions.strip()}")
        if preferences.strip() != "None":
            lines.append(f"  Preferences: {preferences.strip()}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# BUFFER AND ASYNC GENERATION
# =============================================================================


def buffer_for_summary(user: str, message: str, response: str) -> None:
    """Add a conversation to the summary buffer."""
    if len(message) + len(response) < MIN_CONVERSATION_LENGTH:
        return
    _summary_buffer.append(
        {"user": user, "message": message, "response": response}
    )


async def maybe_generate_summaries(agent: Any) -> dict[str, Any] | None:
    """Generate summaries if the buffer has enough conversations.

    Call this after each conversation. It will only actually generate
    when SUMMARY_BATCH_SIZE conversations have accumulated.
    """
    global _summary_running

    if len(_summary_buffer) < SUMMARY_BATCH_SIZE:
        return None

    if _summary_running:
        return None

    _summary_running = True
    try:
        batch = _summary_buffer.copy()
        _summary_buffer.clear()
        return await _generate_summaries(agent, batch)
    finally:
        _summary_running = False


async def _generate_summaries(
    agent: Any, conversations: list[dict[str, str]]
) -> dict[str, Any]:
    """Generate and save conversation summaries using the LLM."""
    start = time.time()

    prompt = build_summary_prompt(conversations)

    try:
        response = await asyncio.to_thread(agent.run, prompt)
        summaries = parse_summary_response(response)
    except Exception as e:
        print(f"[ConvContext] LLM summary generation failed: {e}")
        summaries = fallback_summaries(conversations)

    if not summaries:
        summaries = fallback_summaries(conversations)

    saved = save_summaries(summaries)
    duration = time.time() - start

    print(f"[ConvContext] Saved {saved} conversation summaries ({duration:.1f}s)")

    return {
        "summaries_saved": saved,
        "duration_seconds": round(duration, 1),
    }


# =============================================================================
# TOOLS FOR AGENT
# =============================================================================


def get_conversation_context_tools() -> list:
    """Get tools for conversation context persistence."""
    from .core import create_tool

    return [
        create_tool(
            "get_conversation_summaries",
            "Get recent conversation summaries showing topics discussed, decisions made, "
            "and preferences revealed in past conversations",
            {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent summaries to return (default 20)",
                    }
                },
                "required": [],
            },
            lambda count=20: get_recent_summaries(count),
        ),
        create_tool(
            "search_conversation_summaries",
            "Search through past conversation summaries by keyword. Find what was "
            "discussed, decided, or preferred in previous sessions",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find in conversation summaries",
                    }
                },
                "required": ["query"],
            },
            search_summaries,
        ),
    ]
