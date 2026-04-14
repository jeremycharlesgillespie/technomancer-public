"""Jira integration for the idea board.

Syncs ideas to Jira when they change state. Each idea maps to a Jira issue:
  - epic -> Epic
  - story/task -> Story

State mapping:
  - proposed/refining/approved -> To Do
  - executing -> In Progress
  - done -> Done

Usage:
    from idea_board.jira_sync import sync_idea_to_jira, is_jira_configured

    if is_jira_configured():
        sync_idea_to_jira(idea)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from agent.config import settings

logger = logging.getLogger(__name__)

# Jira issue type mapping
TYPE_MAP = {
    "epic": "Epic",
    "story": "Story",
    "task": "Story",
}

# State mapping: idea board -> Jira status name
STATE_MAP = {
    "proposed": "To Do",
    "refining": "To Do",
    "approved": "To Do",
    "executing": "In Progress",
    "done": "Done",
}


def is_jira_configured() -> bool:
    """Check if Jira credentials are configured."""
    return bool(
        settings.jira_url
        and settings.jira_email
        and settings.jira_api_token
        and settings.jira_project_key
    )


def _auth() -> tuple[str, str]:
    return (settings.jira_email, settings.jira_api_token)


def _api(method: str, path: str, **kwargs) -> requests.Response:
    """Make a Jira API request."""
    url = f"{settings.jira_url}/rest/api/3{path}"
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("auth", _auth())
    return getattr(requests, method)(url, **kwargs)


def _build_description_adf(text: str) -> dict:
    """Convert plain text to Atlassian Document Format."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text[:30000]}],
            }
        ],
    }


def find_jira_issue(idea_id: str) -> str | None:
    """Find an existing Jira issue by idea ID in the summary.

    Returns the Jira key (e.g., TK-42) or None.
    """
    if not is_jira_configured():
        return None

    try:
        resp = _api(
            "post",
            "/search/jql",
            json={
                "jql": (
                    f'project = {settings.jira_project_key} '
                    f'AND summary ~ "{idea_id}"'
                ),
                "maxResults": 1,
                "fields": ["summary"],
            },
        )
        if resp.status_code == 200:
            issues = resp.json().get("issues", [])
            if issues:
                return issues[0]["key"]
    except Exception as e:
        logger.warning("[JiraSync] Search failed: %s", e)

    return None


def create_jira_issue(
    idea_id: str,
    title: str,
    description: str,
    idea_type: str = "story",
    parent_idea_id: str | None = None,
    labels: list[str] | None = None,
) -> str | None:
    """Create a Jira issue for an idea. Returns the Jira key or None."""
    if not is_jira_configured():
        return None

    jira_type = TYPE_MAP.get(idea_type, "Story")
    summary = f"[{idea_id}] {title}"[:255]

    fields: dict[str, Any] = {
        "project": {"key": settings.jira_project_key},
        "summary": summary,
        "description": _build_description_adf(description or title),
        "issuetype": {"name": jira_type},
    }

    if labels:
        fields["labels"] = [lb for lb in labels if lb]

    # Link story to parent epic in Jira
    if parent_idea_id and jira_type != "Epic":
        parent_key = find_jira_issue(parent_idea_id)
        if parent_key:
            fields["parent"] = {"key": parent_key}

    try:
        resp = _api("post", "/issue", json={"fields": fields})
        if resp.status_code == 201:
            key = resp.json()["key"]
            logger.info("[JiraSync] Created %s for %s: %s", key, idea_id, title[:50])
            _add_execute_comment(key, idea_id)
            return key
        else:
            logger.warning(
                "[JiraSync] Create failed (%d): %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logger.warning("[JiraSync] Create error: %s", e)

    return None


def _add_execute_comment(jira_key: str, idea_id: str) -> None:
    """Add a comment to the Jira issue with execution links.

    Includes links to the idea board page, execute API endpoint,
    and the idea detail endpoint for quick access from Jira.
    """
    hub_host = getattr(settings, "server_host", "localhost")
    hub_port = 8322

    execute_url = f"http://{hub_host}:{hub_port}/execute/{idea_id}"
    view_url = f"http://{hub_host}:{hub_port}/ideas#{idea_id}"
    log_url = f"http://{hub_host}:{hub_port}/api/ideas/{idea_id}/log/stream"

    comment_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Technomancer", "marks": [{"type": "strong"}]},
                ],
            },
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [{"type": "paragraph", "content": [
                            {"type": "text", "text": "Execute",
                             "marks": [{"type": "link", "attrs": {"href": execute_url}}]},
                            {"type": "text", "text": " — click to run Claude Code on this idea"},
                        ]}],
                    },
                    {
                        "type": "listItem",
                        "content": [{"type": "paragraph", "content": [
                            {"type": "text", "text": "View on Idea Board",
                             "marks": [{"type": "link", "attrs": {"href": view_url}}]},
                        ]}],
                    },
                    {
                        "type": "listItem",
                        "content": [{"type": "paragraph", "content": [
                            {"type": "text", "text": "Live Log Stream",
                             "marks": [{"type": "link", "attrs": {"href": log_url}}]},
                        ]}],
                    },
                ],
            },
        ],
    }

    try:
        _api(
            "post",
            f"/issue/{jira_key}/comment",
            json={"body": comment_adf},
        )
    except Exception as e:
        logger.warning("[JiraSync] Failed to add execute comment to %s: %s", jira_key, e)


def transition_jira_issue(jira_key: str, target_status: str) -> bool:
    """Transition a Jira issue to a target status.

    Handles the To Do -> In Progress -> Done chain automatically.
    """
    if not is_jira_configured():
        return False

    try:
        resp = _api("get", f"/issue/{jira_key}/transitions")
        if resp.status_code != 200:
            return False

        transitions = resp.json().get("transitions", [])
        available = {t["name"]: t["id"] for t in transitions}

        if target_status in available:
            _api(
                "post",
                f"/issue/{jira_key}/transitions",
                json={"transition": {"id": available[target_status]}},
            )
            return True

        # If Done isn't directly available, go through In Progress first
        if target_status == "Done" and "In Progress" in available:
            _api(
                "post",
                f"/issue/{jira_key}/transitions",
                json={"transition": {"id": available["In Progress"]}},
            )
            # Re-fetch transitions from In Progress
            resp = _api("get", f"/issue/{jira_key}/transitions")
            transitions = resp.json().get("transitions", [])
            available = {t["name"]: t["id"] for t in transitions}
            if "Done" in available:
                _api(
                    "post",
                    f"/issue/{jira_key}/transitions",
                    json={"transition": {"id": available["Done"]}},
                )
                return True

    except Exception as e:
        logger.warning("[JiraSync] Transition error for %s: %s", jira_key, e)

    return False


def sync_idea_to_jira(idea: Any) -> str | None:
    """Sync an idea to Jira: create if new, transition if state changed.

    Args:
        idea: An Idea object with id, title, description, idea_type, state,
              parent_id, and category attributes.

    Returns:
        The Jira key if synced, None if Jira not configured or sync failed.
    """
    if not is_jira_configured():
        return None

    idea_id = idea.id if hasattr(idea, "id") else idea.get("id", "")
    title = idea.title if hasattr(idea, "title") else idea.get("title", "")
    description = idea.description if hasattr(idea, "description") else idea.get("description", "")
    idea_type = idea.idea_type if hasattr(idea, "idea_type") else idea.get("idea_type", "story")
    state = idea.state if hasattr(idea, "state") else idea.get("state", "proposed")
    parent_id = idea.parent_id if hasattr(idea, "parent_id") else idea.get("parent_id")
    category = idea.category if hasattr(idea, "category") else idea.get("category", "")

    # Skip vetoed/failed
    if state in ("vetoed", "failed"):
        return None

    # Find or create
    jira_key = find_jira_issue(idea_id)
    if not jira_key:
        jira_key = create_jira_issue(
            idea_id=idea_id,
            title=title,
            description=description,
            idea_type=idea_type,
            parent_idea_id=parent_id,
            labels=[state, category] if category else [state],
        )

    if not jira_key:
        return None

    # Transition to correct status
    target_status = STATE_MAP.get(state, "To Do")
    transition_jira_issue(jira_key, target_status)

    return jira_key
