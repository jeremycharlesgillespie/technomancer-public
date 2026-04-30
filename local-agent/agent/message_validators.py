"""
Message validation utilities for Discord API calls.

Prevents 400 Bad Request errors from empty or oversized messages
by validating content before it hits the Discord API.
Includes retry logic with content enrichment for empty message errors,
and contextual suggestions based on recent conversation patterns.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime
from typing import Any, Callable, Coroutine

import discord

from .discord_rate_limit import async_retry_on_rate_limit

logger = logging.getLogger(__name__)

#: Maximum retry attempts for empty message errors.
MAX_EMPTY_RETRIES: int = 3

#: Discord's hard limit for message content.
DISCORD_MAX_LENGTH: int = 2000

#: Truncation suffix appended when a message is cut to fit the limit.
TRUNCATION_SUFFIX: str = "..."

# ---------------------------------------------------------------------------
# Recent message tracking for contextual suggestions
# ---------------------------------------------------------------------------

#: Tracks the last 5 user messages for contextual suggestion generation.
_recent_messages: deque[str] = deque(maxlen=5)


def buffer_user_message(content: str) -> None:
    """Record a user message for contextual suggestion generation.

    Called from the Discord message handler before processing so that
    empty-response fallbacks can suggest actions relevant to the conversation.
    """
    stripped = content.strip()
    if stripped:
        _recent_messages.append(stripped)


def get_contextual_suggestion() -> str:
    """Generate a contextual response suggestion based on recent messages.

    Analyzes the last few user messages to suggest a relevant follow-up
    action, making empty-response fallbacks more helpful than a generic
    "I couldn't generate a response" message.
    """
    if not _recent_messages:
        return _default_suggestion()

    recent = list(_recent_messages)
    last = recent[-1].lower()

    # Pattern matching on recent conversation topics
    if _mentions_pattern(recent, r"\b(code|python|bug|error|debug|fix|script)\b"):
        return (
            "I had trouble generating a response. You could try:\n"
            "- Paste the specific error message\n"
            "- Ask me to explain a concept, e.g. *'What does this error mean?'*\n"
            "- Try `betterDev python` for a quick learning article"
        )

    if _mentions_pattern(recent, r"\b(news|headline|tech|update|latest)\b"):
        return (
            "I had trouble generating a response. You could try:\n"
            "- `technews` — get the latest tech news digest\n"
            "- Ask about a specific topic, e.g. *'What's new with AI this week?'*"
        )

    if _mentions_pattern(recent, r"\b(time|date|day|when|schedule|calendar)\b"):
        now = datetime.now()
        return (
            f"I had trouble generating a response, but the time is "
            f"**{now.strftime('%I:%M %p')}** on **{now.strftime('%A, %B %d, %Y')}**.\n"
            f"You can also ask things like *'What day is Friday?'* or "
            f"*'itinerary for my trip'*."
        )

    if _mentions_pattern(recent, r"\b(learn|study|tutorial|explain|how does|how do)\b"):
        return (
            "I had trouble generating a response. You could try:\n"
            "- `betterDev` — generate a learning article on a random topic\n"
            "- `betterDev <topic>` — learn about a specific subject\n"
            "- `learningHistory` — review past articles"
        )

    if _mentions_pattern(recent, r"\b(remember|memory|know about me|profile)\b"):
        return (
            "I had trouble generating a response. You could try:\n"
            "- `think` — see what I remember about you\n"
            "- Ask *'What do you know about me?'*"
        )

    if _mentions_pattern(recent, r"\b(video|youtube|watch|download)\b"):
        return (
            "I had trouble generating a response. You could try:\n"
            "- `listVideos` — see downloaded videos\n"
            "- `searchVideos <query>` — search your library\n"
            "- `downloadVideo <url>` — download a YouTube video"
        )

    return _default_suggestion()


def _mentions_pattern(messages: list[str], pattern: str) -> bool:
    """Check if any recent message matches the given regex pattern.

    Returns False for empty patterns (no matches possible) and None/empty
    message lists (no matches possible).
    """
    if not pattern or not messages:
        return False
    compiled = re.compile(pattern, re.IGNORECASE)
    return any(compiled.search(msg) for msg in messages)


def _default_suggestion() -> str:
    """Fallback suggestion when no specific topic is detected."""
    return (
        "I had trouble generating a response. Here are some things you can try:\n"
        "- Ask a question, e.g. *'What time is it?'*\n"
        "- `technews` — latest tech headlines\n"
        "- `betterDev` — a quick learning article\n"
        "- `showCommands` — see all available commands"
    )


def validate_discord_message(content: str | None) -> str | None:
    """Validate and sanitize a message before sending to Discord.

    Returns a safe-to-send string, or None if the content is empty/whitespace.

    Rules:
        - None or whitespace-only -> returns None
        - Content longer than 2000 chars -> truncated with "..." suffix
        - Otherwise -> returns stripped content as-is

    Args:
        content: The raw message content.

    Returns:
        Sanitized content ready for Discord, or None if empty.
    """
    if content is None:
        logger.warning("[MsgPipeline] validate_discord_message received None content")
        return None

    stripped = content.strip()
    if not stripped:
        logger.warning(
            "[MsgPipeline] validate_discord_message received empty/whitespace content "
            "(original length=%d, repr=%.100r)",
            len(content),
            content,
        )
        return None

    if len(stripped) > DISCORD_MAX_LENGTH:
        truncated = stripped[: DISCORD_MAX_LENGTH - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
        logger.warning(
            "Message truncated from %d to %d chars", len(stripped), len(truncated)
        )
        return truncated

    return stripped


def safe_send_content(content: str | None, fallback: str = "") -> str:
    """Return validated content, substituting a contextual suggestion when empty.

    Unlike ``validate_discord_message`` this never returns None — callers that
    must always send *something* can rely on the fallback.  When content is
    empty, a contextual suggestion based on recent conversation is used instead
    of the generic fallback (unless the fallback is explicitly non-empty and
    no recent messages are available).

    Args:
        content: The raw message content.
        fallback: Text to use when content is empty and no contextual
            suggestion is available.

    Returns:
        Non-empty validated string, contextual suggestion, or the fallback.
    """
    validated = validate_discord_message(content)
    if validated is not None:
        return validated

    # Use contextual suggestion based on recent conversation
    suggestion = get_contextual_suggestion()
    logger.warning(
        "[MsgPipeline] safe_send_content using contextual suggestion "
        "(content was empty, suggestion=%.200r)",
        suggestion,
    )
    return suggestion if suggestion else fallback


def _is_empty_message_error(exc: discord.HTTPException) -> bool:
    """Check if a Discord HTTPException is a 400 empty message error."""
    return exc.status == 400 and "empty message" in str(exc).lower()


def _enrich_empty_content(attempt: int) -> str:
    """Build enriched fallback content for an empty message retry.

    Uses context-aware recovery on the first retry (what was the user
    asking about?), then contextual command suggestions, then a generic
    timestamp fallback.
    """
    if attempt == 1:
        # Try context recovery from the message buffer first
        try:
            from .discord_errors import suggest_recovery_content, get_recent_context
            # Only use context recovery if there are actually buffered messages
            if get_recent_context(1):
                recovery = suggest_recovery_content()
                if recovery and len(recovery) > 20:
                    return recovery
        except Exception:
            pass
        return get_contextual_suggestion()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"I processed your request but my response was empty. "
        f"[retry {attempt}/{MAX_EMPTY_RETRIES}, {timestamp}]"
    )


async def resilient_send(
    send_func: Callable[..., Coroutine[Any, Any, Any]],
    content: str,
    **kwargs: Any,
) -> Any:
    """Send a Discord message, retrying with enriched content on empty message errors.

    If the Discord API returns a 400 "Cannot send an empty message" error, the
    content is replaced with a timestamped fallback and the send is retried up
    to ``MAX_EMPTY_RETRIES`` times.

    Args:
        send_func: A bound async method such as ``message.reply`` or
            ``channel.send``.
        content: The message text to send.
        **kwargs: Additional keyword arguments forwarded to *send_func*
            (e.g. ``files``).

    Returns:
        The result of the successful send call.

    Raises:
        discord.HTTPException: If all retry attempts are exhausted or the
            error is not an empty-message 400.
    """
    last_exc: discord.HTTPException | None = None

    for attempt in range(MAX_EMPTY_RETRIES + 1):
        try:
            # Smooth outbound bursts to stay under Discord rate limits.
            from .discord_rate_limit import get_outbound_limiter
            await get_outbound_limiter().acquire()

            # Wrap the actual send in rate limit retry so 429s are
            # handled transparently with backoff + jitter.
            return await async_retry_on_rate_limit(send_func, content, **kwargs)
        except discord.HTTPException as exc:
            if not _is_empty_message_error(exc) or attempt >= MAX_EMPTY_RETRIES:
                raise
            last_exc = exc
            attempt_num = attempt + 1
            logger.warning(
                "[MsgPipeline] Empty message error on send (attempt %d/%d), "
                "enriching content and retrying",
                attempt_num,
                MAX_EMPTY_RETRIES,
            )
            content = _enrich_empty_content(attempt_num)
            # Drop files on retry — only the text fallback matters now
            kwargs.pop("files", None)
            kwargs.pop("file", None)

    # Should not reach here, but satisfy type checker
    if last_exc is not None:
        raise last_exc
