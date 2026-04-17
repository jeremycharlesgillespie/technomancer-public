"""
Discord Delivery — Centralized message pipeline with chunking, retry, and spool.

Callers across the bot (``discord_memory_bot``, ``news_digest``, ``daily_briefing``,
``bot_utils``) have independently hand-rolled message splitting and error handling,
which means long LLM replies get cut mid code-fence or mid-sentence and transient
5xx/429 responses are handled inconsistently. This module owns the whole send
pipeline so every caller benefits from the same guarantees:

  1. **Splitting** — :func:`split_for_discord` respects the 2000-char Discord limit
     while preferring to break on fenced code blocks, then bullet groups, then
     paragraphs, then line boundaries. A fenced block that crosses a chunk
     boundary is re-wrapped with its original language tag so syntax
     highlighting survives the split.

  2. **Rate limiting** — sends go through the existing
     :class:`~agent.discord_rate_limit.OutboundRateLimiter` (token bucket sized
     to Discord's ~5-per-5-seconds-per-channel budget).

  3. **Retry policy** — :class:`DiscordDelivery` retries on 429 and 5xx up to
     :data:`MAX_ATTEMPTS` times, honoring ``Retry-After`` from the response when
     present. ``403`` / channel-deleted is treated as permanent and dropped
     after a single warning. ``401`` (token revoked) escalates to
     :mod:`agent.alerts`.

  4. **Spool** — messages that exhaust their retries are written to
     :data:`SPOOL_PATH` as JSONL and replayed on the next startup via
     :func:`replay_spool`. Spool corruption is recovered by renaming the file
     to ``.corrupt`` and starting fresh, so a poisoned record can never stop
     delivery permanently.

  5. **Ordering** — each channel gets its own asyncio queue and worker so a
     slow channel cannot starve another, while messages within the same
     channel preserve the order they were submitted in.

The module is intentionally import-safe when discord.py is missing (for tests
that don't want to pull in the whole bot): HTTP exceptions are matched by
duck-typing on ``.status`` / ``.code`` attributes rather than isinstance checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from . import alerts
from .discord_rate_limit import (
    JITTER_FACTOR,
    MAX_DELAY,
    calculate_backoff,
    get_outbound_limiter,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard cap from Discord's API. We leave a few chars of headroom for the
#: re-wrapping of code fences and the continuation marker.
DISCORD_HARD_LIMIT: int = 2000

#: Default soft limit used by the splitter. Keeps room for the ``...`` marker
#: or a re-opened code fence without blowing the hard limit.
DEFAULT_SOFT_LIMIT: int = 1900

#: Maximum retry attempts before a message is spooled or dropped.
MAX_ATTEMPTS: int = 5

#: On-disk spool location. Created lazily on first failed send.
SPOOL_DIR: Path = Path(__file__).resolve().parent.parent / "state"
SPOOL_PATH: Path = SPOOL_DIR / "delivery_spool.jsonl"

#: Status codes that are retryable transient failures.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Status codes that mean "give up immediately, don't spool".
_PERMANENT_DROP_STATUSES: frozenset[int] = frozenset({403, 404})


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```([A-Za-z0-9_+\-]*)\s*$")


def _split_fenced_block(
    lang: str, body: str, soft_limit: int
) -> list[str]:
    """Split an oversized fenced block, re-wrapping each chunk with ``lang``."""
    fence_open = f"```{lang}\n" if lang else "```\n"
    fence_close = "\n```"
    # Budget inside the fence = soft_limit minus fences themselves.
    inner_budget = max(200, soft_limit - len(fence_open) - len(fence_close))
    chunks: list[str] = []
    lines = body.split("\n")
    current: list[str] = []
    current_len = 0
    for line in lines:
        added = len(line) + (1 if current else 0)
        if current_len + added > inner_budget and current:
            chunks.append(fence_open + "\n".join(current) + fence_close)
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += added
    if current:
        chunks.append(fence_open + "\n".join(current) + fence_close)

    # Any chunk that's still oversized (one absurdly long line) gets a hard
    # character split as a last resort.
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= DISCORD_HARD_LIMIT:
            out.append(chunk)
        else:
            out.extend(_hard_split(chunk, soft_limit))
    return out


def _hard_split(text: str, soft_limit: int) -> list[str]:
    """Last-resort char-count split for content with no natural boundaries."""
    return [text[i : i + soft_limit] for i in range(0, len(text), soft_limit)]


def _iter_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Yield ``(kind, content)`` tuples where kind is ``'fence'`` or ``'text'``.

    Fences are emitted as complete blocks (opening fence + body + closing
    fence) so the splitter never breaks inside them unintentionally. Unclosed
    fences are treated as plain text — that way a stray triple-backtick in the
    middle of prose doesn't cause us to swallow the rest of the message.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if m:
            # Look ahead for the matching closing fence.
            lang = m.group(1)
            start = i
            j = i + 1
            closed = False
            while j < len(lines):
                if _FENCE_RE.match(lines[j]):
                    closed = True
                    break
                j += 1
            if closed:
                body = "\n".join(lines[start + 1 : j])
                yield ("fence", f"```{lang}\n{body}\n```")
                i = j + 1
                continue
            # Unclosed fence — treat the fence line itself plus the rest of
            # the buffer as prose. Yielding it all at once lets the text
            # splitter handle paragraph/line boundaries normally.
            yield ("text", "\n".join(lines[i:]))
            return
        # Plain text line. Accumulate until the next fence.
        start = i
        while i < len(lines) and not _FENCE_RE.match(lines[i]):
            i += 1
        yield ("text", "\n".join(lines[start:i]))


def _split_text_block(text: str, soft_limit: int) -> list[str]:
    """Split a non-fenced block, preferring paragraph then line boundaries."""
    if len(text) <= soft_limit:
        return [text] if text else []

    chunks: list[str] = []
    # Prefer double-newline (paragraph) boundaries, then single-newline.
    for separator in ("\n\n", "\n"):
        if separator not in text:
            continue
        parts = text.split(separator)
        current = ""
        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) > soft_limit and current:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)
        # If any chunk is still too long, fall through to a finer-grained split.
        if all(len(c) <= soft_limit for c in chunks):
            return chunks
        chunks = []

    # Last resort: character boundary.
    return _hard_split(text, soft_limit)


def split_for_discord(
    content: str, soft_limit: int = DEFAULT_SOFT_LIMIT
) -> list[str]:
    """Split *content* into chunks that fit Discord's 2000-char limit.

    Preference order: fenced code blocks stay intact (or are re-wrapped with
    their language tag when they themselves exceed the limit), then paragraph
    boundaries, then line boundaries, then — as a last resort — a hard
    character split.

    Empty input returns an empty list, which the delivery worker treats as a
    no-op rather than sending a blank message.
    """
    if not content:
        return []
    if len(content) <= soft_limit:
        return [content]

    rendered_blocks: list[str] = []
    for kind, block in _iter_blocks(content):
        if kind == "fence":
            if len(block) <= soft_limit:
                rendered_blocks.append(block)
                continue
            # Re-wrap oversized fence with its original language tag.
            m = _FENCE_RE.match(block.split("\n", 1)[0])
            lang = m.group(1) if m else ""
            body = "\n".join(block.split("\n")[1:-1])
            rendered_blocks.extend(_split_fenced_block(lang, body, soft_limit))
        else:
            rendered_blocks.extend(_split_text_block(block, soft_limit))

    # Merge adjacent small chunks so we don't send more messages than needed.
    merged: list[str] = []
    for chunk in rendered_blocks:
        if not chunk:
            continue
        if merged and len(merged[-1]) + len(chunk) + 2 <= soft_limit:
            # Don't merge across fence boundaries — if either side is a fence,
            # keep them separate to preserve rendering.
            if not (merged[-1].startswith("```") or chunk.startswith("```")):
                merged[-1] = merged[-1] + "\n\n" + chunk
                continue
        merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# Spool
# ---------------------------------------------------------------------------


@dataclass
class SpooledMessage:
    """One message that failed delivery and is awaiting replay."""

    channel_id: int
    content: str
    reply_to: Optional[int] = None
    spooled_at: float = field(default_factory=time.time)
    attempts: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "channel_id": self.channel_id,
                "content": self.content,
                "reply_to": self.reply_to,
                "spooled_at": self.spooled_at,
                "attempts": self.attempts,
            }
        )

    @classmethod
    def from_json(cls, line: str) -> "SpooledMessage":
        data = json.loads(line)
        return cls(
            channel_id=int(data["channel_id"]),
            content=str(data["content"]),
            reply_to=data.get("reply_to"),
            spooled_at=float(data.get("spooled_at", time.time())),
            attempts=int(data.get("attempts", 0)),
        )


def _ensure_spool_dir() -> None:
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)


def spool_append(msg: SpooledMessage, path: Path = SPOOL_PATH) -> None:
    """Append *msg* to the on-disk spool. Never raises — spool failures are
    logged and swallowed because losing a retry is strictly better than
    crashing the delivery loop."""
    try:
        _ensure_spool_dir()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(msg.to_json() + "\n")
    except Exception:
        logger.exception("discord_delivery: failed to append to spool %s", path)


def spool_load(path: Path = SPOOL_PATH) -> list[SpooledMessage]:
    """Load the spool, quarantining a corrupt file to ``<path>.corrupt``.

    Returns an empty list if the spool is absent, empty, or unreadable. The
    file is truncated as part of loading so replay is exactly-once — callers
    that need at-least-once semantics should re-spool on failure.
    """
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("discord_delivery: could not read spool %s", path)
        return []

    messages: list[SpooledMessage] = []
    bad_lines = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(SpooledMessage.from_json(line))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            bad_lines += 1

    if bad_lines and not messages:
        # The whole file is garbage — quarantine it and start fresh.
        corrupt = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, corrupt)
            logger.warning(
                "discord_delivery: spool %s was corrupt (%d bad lines), "
                "renamed to %s",
                path,
                bad_lines,
                corrupt,
            )
        except OSError:
            logger.exception("discord_delivery: could not quarantine spool %s", path)
        return []

    if bad_lines:
        logger.warning(
            "discord_delivery: dropped %d malformed lines from spool %s",
            bad_lines,
            path,
        )

    # Truncate after successful load so replay doesn't re-deliver.
    try:
        path.unlink()
    except OSError:
        logger.exception("discord_delivery: could not clear spool %s", path)
    return messages


# ---------------------------------------------------------------------------
# Delivery engine
# ---------------------------------------------------------------------------


def _status_of(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status from a Discord exception."""
    status = getattr(exc, "status", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    return None


def _retry_after_of(exc: BaseException) -> Optional[float]:
    """Extract retry-after from either the exception or its response headers."""
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, (int, float)) and direct > 0:
        return float(direct)
    response = getattr(exc, "response", None)
    if response is not None:
        return parse_retry_after(response)
    return None


@dataclass
class DeliveryResult:
    """Outcome of a single delivery attempt, returned for tests/telemetry."""

    delivered: int = 0
    spooled: int = 0
    dropped: int = 0
    attempts: int = 0


SendCallable = Callable[[int, str, Optional[int]], Awaitable[Any]]
"""Signature of the function that actually talks to Discord: ``(channel_id,
content, reply_to) -> Awaitable``. Injected so the engine can be tested
without a live Discord client."""


class DiscordDelivery:
    """Per-channel ordered delivery with retry, rate limiting, and spool.

    Each ``send(channel_id, content, reply_to)`` call enqueues work on the
    channel's dedicated worker. Callers get back immediately; the worker
    handles splitting, rate limiting, retries, and spooling in the background.

    The actual Discord call is abstracted through *send_func* so tests and
    non-Discord transports (the bridge, a webhook) can plug in.
    """

    def __init__(
        self,
        send_func: SendCallable,
        *,
        soft_limit: int = DEFAULT_SOFT_LIMIT,
        max_attempts: int = MAX_ATTEMPTS,
        spool_path: Path = SPOOL_PATH,
    ) -> None:
        self._send_func = send_func
        self._soft_limit = soft_limit
        self._max_attempts = max_attempts
        self._spool_path = spool_path
        self._queues: dict[int, asyncio.Queue[Optional[SpooledMessage]]] = {}
        self._workers: dict[int, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._limiter = get_outbound_limiter()
        self._token_alert_sent = False

    # ---- public API -------------------------------------------------------

    async def send(
        self,
        channel_id: int,
        content: str,
        reply_to: Optional[int] = None,
    ) -> int:
        """Enqueue *content* for delivery. Returns the number of chunks queued.

        Never blocks on the network — the send happens in a background worker
        so the caller (for example a Discord ``on_message`` handler) stays
        responsive. If *content* is empty, returns 0 immediately.
        """
        chunks = split_for_discord(content, soft_limit=self._soft_limit)
        if not chunks:
            return 0
        queue = await self._get_queue(channel_id)
        for chunk in chunks:
            await queue.put(
                SpooledMessage(
                    channel_id=channel_id,
                    content=chunk,
                    reply_to=reply_to,
                )
            )
        return len(chunks)

    async def shutdown(self) -> None:
        """Stop all worker tasks cleanly. Used at bot shutdown and in tests."""
        async with self._lock:
            for q in self._queues.values():
                await q.put(None)
            workers = list(self._workers.values())
            self._workers.clear()
            self._queues.clear()
        for task in workers:
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ---- internals --------------------------------------------------------

    async def _get_queue(
        self, channel_id: int
    ) -> asyncio.Queue[Optional[SpooledMessage]]:
        async with self._lock:
            q = self._queues.get(channel_id)
            if q is None:
                q = asyncio.Queue()
                self._queues[channel_id] = q
                self._workers[channel_id] = asyncio.create_task(
                    self._worker(channel_id, q),
                    name=f"discord-delivery-{channel_id}",
                )
            return q

    async def _worker(
        self,
        channel_id: int,
        queue: asyncio.Queue[Optional[SpooledMessage]],
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            try:
                await self._deliver_with_retry(item)
            except Exception:
                logger.exception(
                    "discord_delivery: unhandled error delivering to %s",
                    channel_id,
                )

    async def _deliver_with_retry(self, msg: SpooledMessage) -> DeliveryResult:
        result = DeliveryResult()
        attempt = msg.attempts
        while attempt < self._max_attempts:
            result.attempts = attempt + 1
            await self._limiter.acquire()
            try:
                await self._send_func(msg.channel_id, msg.content, msg.reply_to)
                result.delivered = 1
                return result
            except Exception as exc:
                status = _status_of(exc)
                if status in _PERMANENT_DROP_STATUSES:
                    logger.warning(
                        "discord_delivery: dropping message to %s permanently "
                        "(status=%s, %s)",
                        msg.channel_id,
                        status,
                        exc,
                    )
                    result.dropped = 1
                    return result
                if status == 401:
                    self._notify_token_revoked(exc)
                    result.dropped = 1
                    return result
                if status is not None and status not in _RETRYABLE_STATUSES:
                    logger.warning(
                        "discord_delivery: non-retryable status %s, dropping "
                        "(channel=%s): %s",
                        status,
                        msg.channel_id,
                        exc,
                    )
                    result.dropped = 1
                    return result

                attempt += 1
                if attempt >= self._max_attempts:
                    logger.error(
                        "discord_delivery: exhausted %d retries for channel %s, "
                        "spooling (last error: %s)",
                        self._max_attempts,
                        msg.channel_id,
                        exc,
                    )
                    msg.attempts = attempt
                    spool_append(msg, self._spool_path)
                    result.spooled = 1
                    return result

                wait = self._compute_wait(exc, attempt - 1)
                logger.warning(
                    "discord_delivery: transient failure on channel %s "
                    "(status=%s, attempt %d/%d), retrying in %.2fs",
                    msg.channel_id,
                    status,
                    attempt,
                    self._max_attempts,
                    wait,
                )
                await asyncio.sleep(wait)
        return result

    def _compute_wait(self, exc: BaseException, attempt: int) -> float:
        header_wait = _retry_after_of(exc)
        if header_wait is not None:
            # Pad with small jitter so 100 sends don't all wake at once when
            # Discord hands out the same Retry-After.
            import random

            jitter = random.uniform(0, header_wait * JITTER_FACTOR)
            return min(header_wait + jitter, MAX_DELAY)
        return calculate_backoff(attempt)

    def _notify_token_revoked(self, exc: BaseException) -> None:
        if self._token_alert_sent:
            return
        self._token_alert_sent = True
        logger.critical("discord_delivery: Discord token rejected (401): %s", exc)
        try:
            alerts.send_alert(
                f"Discord rejected our token with 401: {exc}. "
                "The bot cannot send messages until the token is rotated.",
                title="Discord token revoked",
                level="critical",
                category="auth",
            )
        except Exception:
            logger.exception("discord_delivery: failed to send token-revoked alert")


# ---------------------------------------------------------------------------
# Spool replay
# ---------------------------------------------------------------------------


async def replay_spool(
    delivery: DiscordDelivery, path: Path = SPOOL_PATH
) -> int:
    """Re-enqueue any messages parked in the spool. Called once at startup.

    Returns the number of messages re-submitted. A corrupt spool file is
    quarantined to ``<path>.corrupt`` and the function returns 0 so startup
    cannot wedge on bad state.
    """
    messages = spool_load(path)
    replayed = 0
    for msg in messages:
        # Reset the attempt counter so spooled messages get a fresh retry
        # budget; they were already spooled *after* exhausting attempts.
        await delivery.send(msg.channel_id, msg.content, msg.reply_to)
        replayed += 1
    if replayed:
        logger.info("discord_delivery: replayed %d spooled messages", replayed)
    return replayed
