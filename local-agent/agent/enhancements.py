"""
Enhancement Queue - Track feature ideas for Claude to implement.

Stores enhancements in Obsidian vault for persistence and visibility.
ALWAYS reads from the vault file - single source of truth.
"""

import re
from datetime import datetime
from pathlib import Path

# Obsidian vault location - loaded from settings
from .config import settings
VAULT_PATH = settings.llm_memory_path
ENHANCEMENTS_FILE = VAULT_PATH / "Permanent" / "enhancements.md"


def _get_next_number() -> int:
    """Get the next enhancement number by scanning existing entries."""
    if not ENHANCEMENTS_FILE.exists():
        return 1

    content = ENHANCEMENTS_FILE.read_text(encoding="utf-8")

    # Find all existing numbers like "#1:", "#2:", etc.
    numbers = re.findall(r"#(\d+):", content)
    if numbers:
        return max(int(n) for n in numbers) + 1
    return 1


def add_enhancement(idea: str) -> str:
    """
    Add a new enhancement idea to the queue with auto-numbering.

    Args:
        idea: The enhancement/feature idea to add

    Returns:
        Confirmation message with the assigned number
    """
    date = datetime.now().strftime("%Y-%m-%d")
    next_num = _get_next_number()
    new_item = f"- [ ] **#{next_num}:** {idea} ({date})\n"

    if not ENHANCEMENTS_FILE.exists():
        # Create the file with structure
        content = f"""# Enhancement Queue

Ideas and feature requests for Claude to implement.

---

## Pending

{new_item}
## In Progress

## Completed

"""
        ENHANCEMENTS_FILE.write_text(content, encoding="utf-8")
    else:
        content = ENHANCEMENTS_FILE.read_text(encoding="utf-8")

        # Insert after "## Pending" line
        if "## Pending" in content:
            content = content.replace("## Pending\n", f"## Pending\n\n{new_item}")
        else:
            # Fallback: append to end
            content += f"\n{new_item}"

        ENHANCEMENTS_FILE.write_text(content, encoding="utf-8")

    return f"Added enhancement #{next_num}: {idea}"


def get_pending_enhancements() -> str:
    """
    Get all pending enhancements from the vault file.
    ALWAYS reads fresh from disk - never relies on memory.

    Returns:
        Formatted list of pending enhancements with numbers
    """
    if not ENHANCEMENTS_FILE.exists():
        return "No enhancements file found at vault location."

    # Always read fresh from disk
    content = ENHANCEMENTS_FILE.read_text(encoding="utf-8")

    # Extract pending section
    pending_match = re.search(
        r"## Pending\s*\n(.*?)(?=## In Progress|## Completed|$)", content, re.DOTALL
    )

    if pending_match:
        pending = pending_match.group(1).strip()
        if pending:
            # Count items
            items = [line for line in pending.split("\n") if line.strip().startswith("- [ ]")]
            count = len(items)
            return f"**Pending Enhancements ({count}):**\n{pending}"

    return "No pending enhancements in the queue."


def detect_enhancement_idea(text: str) -> str | None:
    """
    Detect if the user is suggesting a feature/enhancement.

    Returns the enhancement idea if detected, None otherwise.
    """
    text_lower = text.lower()

    # Patterns that suggest feature requests
    patterns = [
        r"it would be cool if (.+)",
        r"we should add (.+)",
        r"can you add (.+)",
        r"add an enhancement[:\s]+(.+)",
        r"feature request[:\s]+(.+)",
        r"idea[:\s]+(.+)",
        r"wouldn't it be nice if (.+)",
        r"i wish (?:you|the bot|it) could (.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract the idea and clean it up
            idea = match.group(1).strip()
            # Capitalize first letter
            idea = idea[0].upper() + idea[1:] if idea else idea
            return idea

    return None


def save_context_for_claude(context: str, enhancement_ref: str = "") -> str:
    """
    Save context/documents for Claude Code to use when implementing enhancements.
    This creates a handoff file that Claude Code will read.

    Args:
        context: The context, error message, code snippet, or document content
        enhancement_ref: Optional reference to which enhancement this relates to (e.g., "#1")

    Returns:
        Confirmation message
    """
    handoff_file = VAULT_PATH / "Permanent" / "claude_handoff.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = (
        f"## Context for Enhancement {enhancement_ref}"
        if enhancement_ref
        else "## Additional Context"
    )

    entry = f"""
{header}
*Added: {timestamp}*

{context}

---
"""

    if handoff_file.exists():
        existing = handoff_file.read_text(encoding="utf-8")
        content = existing + entry
    else:
        content = f"""# Claude Handoff

Context and documents passed from the Discord bot for Claude Code to use.

---
{entry}"""

    handoff_file.write_text(content, encoding="utf-8")
    return (
        f"Saved context for Claude. Reference: {enhancement_ref if enhancement_ref else 'general'}"
    )


def get_enhancement_tools() -> list:
    """Get enhancement tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "add_enhancement",
            (
                "Add a feature idea or enhancement request to the queue. "
                "Automatically assigns a number for easy reference. "
                "Claude Code will see these and implement them later."
            ),
            {
                "type": "object",
                "properties": {
                    "idea": {
                        "type": "string",
                        "description": "The enhancement or feature idea to add",
                    }
                },
                "required": ["idea"],
            },
            add_enhancement,
        ),
        create_tool(
            "get_enhancements",
            (
                "Get the list of pending enhancements from the Obsidian vault. "
                "Always reads fresh from the vault file - single source of truth. "
                "Use when the user asks about what's in the queue or says 'start working on enhancements'."
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda: get_pending_enhancements(),
        ),
        create_tool(
            "save_context_for_claude",
            (
                "Save a document, code snippet, error message, or other context for Claude Code. "
                "Use this when the user shares files or information related to an enhancement. "
                "Claude Code will read this handoff file when implementing features."
            ),
            {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The document content, code snippet, error message, or context to save",
                    },
                    "enhancement_ref": {
                        "type": "string",
                        "description": "Optional reference like '#1' to link this context to a specific enhancement",
                    },
                },
                "required": ["context"],
            },
            lambda context, enhancement_ref="": save_context_for_claude(context, enhancement_ref),
        ),
    ]
