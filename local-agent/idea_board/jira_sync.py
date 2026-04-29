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

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from agent.config import settings
from agent.jira_retry import jira_request

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
    "failed": "Failed",
}

# Retry tuning for Jira writes (429 / 5xx).
MAX_RETRY_ATTEMPTS = 5
BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Dead-letter file for sync writes that fail all retries.
DEADLETTER_PATH: Path = (
    Path(__file__).parent.parent / "logs" / "jira_sync_deadletter.jsonl"
)


class JiraRetryExhausted(Exception):
    """Raised when all retry attempts for a Jira POST are exhausted."""

    def __init__(
        self,
        status: int | None,
        body: str,
        attempts: int,
        path: str = "",
    ) -> None:
        self.status = status
        self.body = body or ""
        self.attempts = attempts
        self.path = path
        super().__init__(
            f"Jira POST {path or '?'} failed after {attempts} attempts "
            f"(last status={status})"
        )


def _backoff_for(attempt: int) -> float:
    """Return backoff seconds for ``attempt`` (1-indexed)."""
    idx = min(max(attempt, 1) - 1, len(BACKOFF_SECONDS) - 1)
    return BACKOFF_SECONDS[idx]


def _retry_after_seconds(resp: requests.Response) -> float | None:
    """Parse the ``Retry-After`` header (integer-seconds form only)."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except (TypeError, ValueError):
        return None


def _post_with_retry(
    path: str,
    payload: dict,
    *,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
) -> requests.Response:
    """POST to Jira with exponential backoff on 429 / 5xx.

    Honours ``Retry-After`` on 429 when present. Any other error response
    or a successful response is returned to the caller as-is. Raises
    :class:`JiraRetryExhausted` when all attempts return a retryable
    status or raise a network-level ``requests`` exception.
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    last_status: int | None = None
    last_body: str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            resp = _api("post", path, json=payload)
        except requests.RequestException as exc:
            last_status = None
            last_body = repr(exc)
            if attempt >= max_attempts:
                break
            delay = _backoff_for(attempt)
            logger.warning(
                "[JiraSync] POST %s network error %s, retry in %.1fs "
                "(attempt %d/%d)",
                path, exc, delay, attempt, max_attempts,
            )
            sleep_fn(delay)
            continue

        # jira_request returns None once its own retries exhaust a
        # ConnectionError / ReadTimeout / 5xx run; treat that as a
        # retryable failure at this layer as well.
        if resp is None:
            last_status = None
            last_body = "jira_request exhausted retries"
            if attempt >= max_attempts:
                break
            sleep_fn(_backoff_for(attempt))
            continue

        if resp.status_code not in RETRY_STATUS_CODES:
            return resp

        last_status = resp.status_code
        last_body = resp.text or ""
        if attempt >= max_attempts:
            break

        delay = _retry_after_seconds(resp) if resp.status_code == 429 else None
        if delay is None:
            delay = _backoff_for(attempt)

        logger.warning(
            "[JiraSync] POST %s -> %d, retry in %.1fs (attempt %d/%d)",
            path, resp.status_code, delay, attempt, max_attempts,
        )
        sleep_fn(delay)

    raise JiraRetryExhausted(last_status, last_body, max_attempts, path=path)


def _remove_artifacts(idea_id: str) -> None:
    """Remove artifacts for an idea when it's done or failed.

    Cleans up temporary files and directories in the executor_artifacts
    directory for the given idea_id.
    """
    try:
        from idea_board.settings import settings
        from idea_board.utils import get_executor_artifacts_path

        artifacts_path = get_executor_artifacts_path()
        if not artifacts_path:
            return

        idea_dir = artifacts_path / idea_id
        if idea_dir.exists():
            # Remove the entire idea directory
            import shutil
            shutil.rmtree(idea_dir, ignore_errors=True)
            logger.info("[JiraSync] Removed artifacts for idea %s", idea_id)
    except Exception as exc:
        logger.warning("[JiraSync] Failed to remove artifacts for %s: %s", idea_id, exc)


def _write_deadletter(
    idea_id: str,
    target_state: str,
    last_error: str,
    *,
    path: Path | None = None,
) -> None:
    """Append a failed-sync payload to the dead-letter JSONL file."""
    dest = path or DEADLETTER_PATH
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "idea_id": idea_id,
            "target_state": target_state,
            "last_error": (last_error or "")[:2000],
        }
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "[JiraSync] Failed to write dead-letter for %s: %s", idea_id, exc
        )


def _dlq_add(
    idea_id: str,
    payload: dict,
    error: str,
    attempts: int,
) -> None:
    """Record a failed Jira write to the SQLite dead-letter queue.

    Imported lazily to break the circular dependency with
    :mod:`idea_board.jira_sync_dlq`, which itself imports
    :func:`sync_idea_to_jira`. Any failure here is swallowed so a broken
    DLQ never blocks the caller's normal control flow.
    """
    try:
        from idea_board.jira_sync_dlq import add_dlq_entry
        add_dlq_entry(idea_id, payload, error, attempts)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "[JiraSync] Failed to add DLQ entry for %s: %s", idea_id, exc
        )


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


def _api(
    method: str,
    path: str,
    *,
    idea_id: str | None = None,
    jira_key: str | None = None,
    **kwargs,
) -> requests.Response | None:
    """Make a Jira API request via ``jira_request``.

    Returns the :class:`requests.Response` on success or non-retryable
    failure, or ``None`` when all bounded retries were exhausted.
    Transient failures (ConnectionError / ReadTimeout / 5xx) and their
    recording to ``jira_sync_failures`` are handled in ``jira_request``.
    """
    url = f"{settings.jira_url}/rest/api/3{path}"
    kwargs.setdefault("auth", _auth())
    return jira_request(
        method,
        url,
        idea_id=idea_id,
        jira_key=jira_key,
        **kwargs,
    )


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
            idea_id=idea_id,
            json={
                "jql": (
                    f'project = {settings.jira_project_key} '
                    f'AND summary ~ "{idea_id}" '
                    f'ORDER BY created ASC'
                ),
                "maxResults": 1,
                "fields": ["summary"],
            },
        )
        if resp is not None and resp.status_code == 200:
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
    summary = title[:255]

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
        resp = _post_with_retry("/issue", {"fields": fields})
        if resp.status_code == 201:
            key = resp.json()["key"]
            logger.info("[JiraSync] Created %s for %s: %s", key, idea_id, title[:50])
            _add_execute_comment(key, idea_id)
            return key
        logger.warning(
            "[JiraSync] Create failed (%d): %s",
            resp.status_code,
            resp.text[:200],
        )
        _dlq_add(
            idea_id=idea_id,
            payload={
                "endpoint": "/issue",
                "body": {"fields": fields},
                "target_state": STATE_MAP.get("proposed", "To Do"),
                "idea_type": idea_type,
            },
            error=f"HTTP {resp.status_code}: {(resp.text or '')[:500]}",
            attempts=1,
        )
    except JiraRetryExhausted:
        raise
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
    log_url = f"http://{hub_host}:{hub_port}/live/{idea_id}"

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
        resp = _api("get", f"/issue/{jira_key}/transitions", jira_key=jira_key)
        if resp is None or resp.status_code != 200:
            return False

        transitions = resp.json().get("transitions", [])
        available = {t["name"]: t["id"] for t in transitions}

        if target_status in available:
            _post_with_retry(
                f"/issue/{jira_key}/transitions",
                {"transition": {"id": available[target_status]}},
            )
            return True

        # If Done isn't directly available, go through In Progress first
        if target_status == "Done" and "In Progress" in available:
            _post_with_retry(
                f"/issue/{jira_key}/transitions",
                {"transition": {"id": available["In Progress"]}},
            )
            # Re-fetch transitions from In Progress
            resp = _api("get", f"/issue/{jira_key}/transitions", jira_key=jira_key)
            if resp is None:
                return False
            transitions = resp.json().get("transitions", [])
            available = {t["name"]: t["id"] for t in transitions}
            if "Done" in available:
                _post_with_retry(
                    f"/issue/{jira_key}/transitions",
                    {"transition": {"id": available["Done"]}},
                )
                return True

    except JiraRetryExhausted:
        raise
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

    # Skip vetoed — but still sync failed so Jira reflects actual state
    if state == "vetoed":
        return None

    target_status = STATE_MAP.get(state, "To Do")

    try:
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

        transition_jira_issue(jira_key, target_status)
        return jira_key
    except JiraRetryExhausted as exc:
        logger.error(
            "[JiraSync] Retries exhausted for %s -> %s: %s",
            idea_id, target_status, exc,
        )
        _write_deadletter(
            idea_id=idea_id,
            target_state=target_status,
            last_error=str(exc),
        )
        _dlq_add(
            idea_id=idea_id,
            payload={
                "endpoint": exc.path or "",
                "body": {
                    "title": title,
                    "state": state,
                    "idea_type": idea_type,
                },
                "target_state": target_status,
                "idea_type": idea_type,
            },
            error=str(exc),
            attempts=exc.attempts,
        )
        return None
