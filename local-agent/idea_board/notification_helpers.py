"""
Notification Helpers — Query recent healthy-bot Discord notifications.

Thin wrapper around :func:`agent.healthy_notifications.query_healthy_notifications`
that reshapes rows for the ``/errors`` empty-state block. Keeps the notification
concern out of ``idea_board/web.py`` so the web module stays under its line cap.

Each returned row is ``{"timestamp": str, "message_id": str|None,
"discord_url": str|None}``. ``message_id`` is the trailing path segment of the
stored Discord message URL (Discord URLs have the form
``https://discord.com/channels/{guild_id}/{channel_id}/{message_id}``); it is
``None`` when the webhook was posted without ``?wait=true`` and no URL was
persisted.
"""

from __future__ import annotations

from typing import Optional

from agent.healthy_notifications import (
    query_healthy_notifications as _query_healthy_notifications,
)


def _extract_message_id(url: Optional[str]) -> Optional[str]:
    """Return the trailing segment of a Discord message URL, or ``None``."""
    if not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def query_healthy_notifications(
    limit: int = 5, max_age_hours: Optional[int] = 24
) -> list[dict[str, Optional[str]]]:
    """Return recent healthy-bot notifications shaped for the ``/errors`` page.

    Delegates the DB read to :mod:`agent.healthy_notifications` and adds a
    derived ``message_id`` field so callers can render per-message anchors
    without re-parsing URLs.
    """
    rows = _query_healthy_notifications(limit=limit, max_age_hours=max_age_hours)
    return [
        {
            "timestamp": row.get("timestamp"),
            "message_id": _extract_message_id(row.get("message_url")),
            "discord_url": row.get("message_url"),
        }
        for row in rows
    ]
