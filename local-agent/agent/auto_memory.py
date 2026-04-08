"""
Auto Memory Extraction - Builds a persistent "database of self" in Obsidian.

After each conversation, the LLM analyzes the exchange and extracts durable
facts about the user: preferences, opinions, skills, experiences, goals, etc.

These are stored as structured markdown files in the Obsidian vault under
Permanent/identity/, building a rich understanding of who the user is over time.

The extraction runs asynchronously after the bot responds, so it never slows
down the conversation.
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Obsidian vault location
from .config import settings

VAULT_PATH = settings.llm_memory_path
IDENTITY_DIR = VAULT_PATH / "Permanent" / "identity"
EXTRACTION_LOG = VAULT_PATH / "Permanent" / "extraction_log.md"

# Memory categories — each maps to a file in the identity directory
MEMORY_CATEGORIES = {
    "preferences": "What the user likes, dislikes, and cares about",
    "opinions": "Stances on tech, tools, practices, and industry topics",
    "skills": "Technical skills, expertise levels, and experience areas",
    "experiences": "Work history, projects, achievements, and stories shared",
    "goals": "What the user is working toward, learning, or planning",
    "habits": "Work patterns, routines, and approaches to problem-solving",
    "personality": "Communication style, humor, values, and character traits",
    "relationships": "People mentioned, team dynamics, professional network",
}

# How many recent conversations to batch before extracting
BATCH_SIZE = 3
# Minimum chars in a conversation to be worth extracting from
MIN_CONVERSATION_LENGTH = 100


def _ensure_identity_dir() -> None:
    """Create identity directory if it doesn't exist."""
    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)


def _load_identity_file(category: str) -> str:
    """Load an identity file, or return empty template."""
    path = IDENTITY_DIR / f"{category}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"# {category.title()}\n\n{MEMORY_CATEGORIES[category]}\n\n---\n"


def _save_identity_file(category: str, content: str) -> None:
    """Save content to an identity file."""
    _ensure_identity_dir()
    path = IDENTITY_DIR / f"{category}.md"
    path.write_text(content, encoding="utf-8")


def _log_extraction(facts_extracted: int, categories_updated: list[str], duration: float) -> None:
    """Append to extraction log for tracking."""
    _ensure_identity_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (
        f"| {timestamp} | {facts_extracted} | {', '.join(categories_updated) or 'none'} "
        f"| {duration:.1f}s |\n"
    )

    if not EXTRACTION_LOG.exists():
        header = (
            "# Memory Extraction Log\n\n"
            "Tracks automatic memory extraction runs.\n\n"
            "| Timestamp | Facts | Categories | Duration |\n"
            "|-----------|-------|------------|----------|\n"
        )
        EXTRACTION_LOG.write_text(header + entry, encoding="utf-8")
    else:
        with open(EXTRACTION_LOG, "a", encoding="utf-8") as f:
            f.write(entry)


def build_extraction_prompt(
    conversations: list[dict[str, str]], existing_facts: dict[str, str]
) -> str:
    """Build the prompt that asks the LLM to extract identity facts.

    Args:
        conversations: List of {"user": str, "message": str, "response": str}
        existing_facts: {category: current_file_content} so we don't duplicate
    """
    conv_text = ""
    for conv in conversations:
        conv_text += f"User ({conv['user']}): {conv['message']}\n"
        conv_text += f"Bot: {conv['response'][:500]}\n\n"

    existing_summary = ""
    for cat, content in existing_facts.items():
        # Just show bullet points, not headers
        bullets = [line for line in content.split("\n") if line.startswith("- ")]
        if bullets:
            existing_summary += f"\n**{cat}**: {'; '.join(b[2:] for b in bullets[:5])}"

    return f"""You are a memory extraction system. Analyze these conversations and extract
durable facts about the USER (not the bot). Focus on things that reveal who they are
as a person and professional.

CONVERSATIONS:
{conv_text}

ALREADY KNOWN (do not duplicate these):
{existing_summary or "Nothing yet — this is a fresh start."}

For each new fact you find, output it in this exact JSON format:
```json
[
  {{"category": "CATEGORY", "fact": "Concise fact about the user", "confidence": "high|medium"}},
  ...
]
```

Valid categories: {", ".join(MEMORY_CATEGORIES.keys())}

Rules:
- Only extract facts about the USER, not the bot
- Skip trivial/obvious things ("user asked about weather")
- Focus on durable identity facts, not ephemeral conversation topics
- If the user corrects the bot, that reveals a preference — extract it
- "high" confidence = user explicitly stated it; "medium" = inferred from behavior
- If nothing meaningful can be extracted, return an empty array: []
- Return ONLY the JSON array, nothing else"""


def parse_extraction_response(response: str) -> list[dict[str, str]]:
    """Parse the LLM's extraction response into structured facts."""
    # Try to find JSON in the response
    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try to find bare JSON array
        json_match = re.search(r"\[.*?\]", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return []

    try:
        facts = json.loads(raw)
        if not isinstance(facts, list):
            return []
        # Validate each fact
        valid = []
        for fact in facts:
            if (
                isinstance(fact, dict)
                and "category" in fact
                and "fact" in fact
                and fact["category"] in MEMORY_CATEGORIES
            ):
                valid.append(fact)
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


def merge_facts_into_files(facts: list[dict[str, str]]) -> list[str]:
    """Merge extracted facts into identity files, avoiding duplicates.

    Returns list of categories that were updated.
    """
    updated_categories = []

    # Group facts by category
    by_category: dict[str, list[dict[str, str]]] = {}
    for fact in facts:
        cat = fact["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fact)

    for category, cat_facts in by_category.items():
        content = _load_identity_file(category)
        existing_lower = content.lower()
        new_entries = []

        for fact in cat_facts:
            # Simple duplicate check — see if the core of this fact is already there
            fact_text = fact["fact"]
            # Check for rough duplicates (fuzzy match on key phrases)
            words = set(fact_text.lower().split())
            # If >70% of words are already in the file, skip
            if existing_lower:
                matching = sum(1 for w in words if w in existing_lower and len(w) > 3)
                if len(words) > 0 and matching / len(words) > 0.7:
                    continue

            confidence = fact.get("confidence", "medium")
            timestamp = datetime.now().strftime("%Y-%m-%d")
            new_entries.append(f"- {fact_text} [{confidence}, {timestamp}]")

        if new_entries:
            content += "\n".join(new_entries) + "\n"
            _save_identity_file(category, content)
            updated_categories.append(category)

    return updated_categories


async def extract_memories(
    agent: Any, conversations: list[dict[str, str]]
) -> dict[str, Any]:
    """Run memory extraction on a batch of conversations.

    Args:
        agent: The LLM agent for analysis
        conversations: List of {"user", "message", "response"} dicts

    Returns:
        Stats dict with extraction results
    """
    start = time.time()
    _ensure_identity_dir()

    # Load existing identity for dedup
    existing = {}
    for cat in MEMORY_CATEGORIES:
        existing[cat] = _load_identity_file(cat)

    # Build and run extraction prompt
    prompt = build_extraction_prompt(conversations, existing)

    try:
        response = await asyncio.to_thread(agent.run, prompt)
    except Exception as e:
        print(f"[AutoMemory] Extraction failed: {e}")
        return {"error": str(e), "facts_extracted": 0}

    # Parse and merge
    facts = parse_extraction_response(response)
    updated = merge_facts_into_files(facts)

    duration = time.time() - start
    _log_extraction(len(facts), updated, duration)

    print(
        f"[AutoMemory] Extracted {len(facts)} facts into {len(updated)} categories "
        f"({duration:.1f}s)"
    )

    return {
        "facts_extracted": len(facts),
        "categories_updated": updated,
        "duration_seconds": round(duration, 1),
    }


# =============================================================================
# CONVERSATION BUFFER — Batches conversations for extraction
# =============================================================================

# Module-level buffer for accumulating conversations
_conversation_buffer: list[dict[str, str]] = []
_extraction_running = False


def buffer_conversation(user: str, message: str, response: str) -> None:
    """Add a conversation to the extraction buffer."""
    if len(message) + len(response) < MIN_CONVERSATION_LENGTH:
        return  # Too short to be interesting
    _conversation_buffer.append(
        {"user": user, "message": message, "response": response}
    )


async def maybe_extract(agent: Any) -> dict[str, Any] | None:
    """Run extraction if the buffer has enough conversations.

    Call this after each conversation. It will only actually extract
    when BATCH_SIZE conversations have accumulated.
    """
    global _extraction_running

    if len(_conversation_buffer) < BATCH_SIZE:
        return None

    if _extraction_running:
        return None  # Don't overlap extractions

    _extraction_running = True
    try:
        # Drain the buffer
        batch = _conversation_buffer.copy()
        _conversation_buffer.clear()
        return await extract_memories(agent, batch)
    finally:
        _extraction_running = False


def get_identity_summary() -> str:
    """Get a combined summary of all identity facts for context injection.

    Returns a condensed version suitable for system prompts.
    """
    _ensure_identity_dir()
    parts = []
    for category in MEMORY_CATEGORIES:
        content = _load_identity_file(category)
        bullets = [line for line in content.split("\n") if line.startswith("- ")]
        if bullets:
            parts.append(f"**{category.title()}**")
            # Show most recent 10 facts per category
            for bullet in bullets[-10:]:
                parts.append(bullet)
            parts.append("")

    return "\n".join(parts) if parts else "No identity data yet."
