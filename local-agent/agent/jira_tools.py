"""LLM tools for querying Jira via the board provider abstraction.

Replaces the old ``idea_board.models.get_idea_board_tools()`` registration.
The Discord bot exposes two tools to the model: ``list_ideas`` (board summary)
and ``get_idea`` (full detail of one issue). Both round-trip through
``board.get_provider()`` so they work against Jira (the only configured
backend after PR 7).

Tool names are kept as ``list_ideas`` / ``get_idea`` for backwards
compatibility with prompts and conversation history that reference them by
name. The "idea" wording is a holdover; the underlying calls hit Jira issues.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.core import create_tool

logger = logging.getLogger(__name__)


def _list_ideas_impl(state: str = "") -> str:
    """Return a formatted board summary, optionally filtered by state."""
    try:
        from board import get_provider

        return get_provider().list_ideas_for_llm(state=state)
    except Exception as exc:
        logger.warning("[jira_tools] list_ideas failed: %s", exc)
        return f"Board lookup failed: {exc}"


def _get_idea_impl(idea_id: str) -> str:
    """Return formatted detail for a single Jira issue (key like ``TK-42``)."""
    if not idea_id or not idea_id.strip():
        return "idea_id is required."

    try:
        from board import get_provider

        item = get_provider().get(idea_id.strip())
    except Exception as exc:
        logger.warning("[jira_tools] get_idea(%s) failed: %s", idea_id, exc)
        return f"Lookup failed: {exc}"

    if not item:
        return f"Issue '{idea_id}' not found."

    lines = [
        f"# {item.id}: {item.title}",
        f"State: {item.state} | Type: {item.idea_type} | Category: {item.category}",
        f"Source: {item.source}",
    ]
    if item.parent_id:
        lines.append(f"Parent: {item.parent_id}")
    description = (item.description or "").strip()
    if description:
        lines.append("")
        lines.append("## Description")
        lines.append(description)
    return "\n".join(lines)


def get_jira_tools() -> list[Any]:
    """Return the LLM tools for browsing Jira from the Discord bot."""
    return [
        create_tool(
            "list_ideas",
            "List stories on the project board. Returns active stories by "
            "default, or filter by state.",
            {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": (
                            "Filter by state: proposed, approved, vetoed, "
                            "refining, executing, done, failed. Empty for "
                            "active stories."
                        ),
                    },
                },
                "required": [],
            },
            lambda state="": _list_ideas_impl(state),
        ),
        create_tool(
            "get_idea",
            "Get full details of a specific story by its Jira key (e.g. "
            "'TK-42'). Returns title, state, type, category, parent, and "
            "description.",
            {
                "type": "object",
                "properties": {
                    "idea_id": {
                        "type": "string",
                        "description": "The Jira key, e.g. 'TK-42'.",
                    },
                },
                "required": ["idea_id"],
            },
            lambda idea_id: _get_idea_impl(idea_id),
        ),
    ]
