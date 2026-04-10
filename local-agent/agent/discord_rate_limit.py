"""
Discord rate limit handling with exponential backoff, jitter, and
outbound message buffering.

Provides:
1. Retry wrappers for synchronous (requests) and async (discord.py) calls
2. Outbound rate limiter that smooths message bursts to stay under Discord limits
3. Header parsing for Retry-After and X-RateLimit-Reset
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable, Coroutine

import discord

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of retry attempts for rate-limited requests.
MAX_RETRIES: int = 5

#: Base delay in seconds for exponential backoff when headers are missing.
BASE_DELAY: float = 1.0

#: Maximum delay cap in seconds to prevent excessively long waits.
MAX_DELAY: float = 60.0

#: Jitter factor — actual jitter is uniform in [0, delay * JITTER_FACTOR].
JITTER_FACTOR: float = 0.5


# ---------------------------------------------------------------------------
# Backoff calculation
# ---------------------------------------------------------------------------


def calculate_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and random jitter.

    Args:
        attempt: Zero-based retry attempt number.

    Returns:
        Delay in seconds before the next retry.
    """
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, delay * JITTER_FACTOR)
    return delay + jitter


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def parse_retry_after(response: Any) -> float | None:
    """Extract wait time from a 429 response's headers.

    Checks ``Retry-After`` (seconds) and ``X-RateLimit-Reset`` (epoch
    timestamp) headers. Returns the number of seconds to wait, or None
    if neither header is present or parseable.
    """
    # Retry-After header (seconds to wait)
    retry_after = _get_header(response, "Retry-After")
    if retry_after is not None:
        try:
            wait = float(retry_after)
            if wait > 0:
                logger.debug("Retry-After header: %.2f seconds", wait)
                return wait
        except (ValueError, TypeError):
            pass

    # X-RateLimit-Reset header (epoch timestamp)
    reset_at = _get_header(response, "X-RateLimit-Reset")
    if reset_at is not None:
        try:
            reset_epoch = float(reset_at)
            wait = reset_epoch - time.time()
            if wait > 0:
                logger.debug("X-RateLimit-Reset header: %.2f seconds from now", wait)
                return wait
        except (ValueError, TypeError):
            pass

    return None


def _get_header(response: Any, name: str) -> str | None:
    """Safely retrieve a header value from various response types."""
    if hasattr(response, "headers"):
        return response.headers.get(name)
    return None


# ---------------------------------------------------------------------------
# Synchronous retry wrapper (for requests library)
# ---------------------------------------------------------------------------


def retry_request(
    method: Callable[..., Any],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    """Execute a requests call with automatic retry on 429 responses.

    Parses rate limit headers to determine precise wait times. Falls
    back to exponential backoff with jitter when headers are absent.

    Args:
        method: The ``requests`` method to call (e.g. ``requests.post``).
        *args: Positional arguments forwarded to *method*.
        max_retries: Maximum number of retry attempts.
        **kwargs: Keyword arguments forwarded to *method*.

    Returns:
        The ``requests.Response`` from a successful (non-429) call.

    Raises:
        requests.exceptions.HTTPError: If all retries are exhausted.
    """
    last_response = None

    for attempt in range(max_retries + 1):
        response = method(*args, **kwargs)
        last_response = response

        if response.status_code != 429:
            return response

        if attempt >= max_retries:
            logger.warning(
                "Rate limit: exhausted %d retries for %s %s",
                max_retries,
                getattr(response.request, "method", "?"),
                getattr(response.request, "url", "?"),
            )
            return response

        # Determine wait time from headers or backoff
        header_wait = parse_retry_after(response)
        if header_wait is not None:
            wait = header_wait + random.uniform(0, header_wait * JITTER_FACTOR)
            wait = min(wait, MAX_DELAY)
        else:
            wait = calculate_backoff(attempt)

        logger.warning(
            "Rate limited (429). Retry %d/%d in %.2f seconds",
            attempt + 1,
            max_retries,
            wait,
        )
        time.sleep(wait)

    return last_response


# ---------------------------------------------------------------------------
# Async retry wrapper (for discord.py calls)
# ---------------------------------------------------------------------------


async def async_retry_on_rate_limit(
    send_func: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    """Execute an async Discord call with retry on rate limit errors.

    Handles ``discord.HTTPException`` with status 429 by extracting the
    retry-after value from the exception and waiting before retrying.

    Args:
        send_func: An async callable (e.g. ``message.reply``).
        *args: Positional arguments forwarded to *send_func*.
        max_retries: Maximum retry attempts.
        **kwargs: Keyword arguments forwarded to *send_func*.

    Returns:
        The result of the successful call.

    Raises:
        discord.HTTPException: If all retries are exhausted or the error
            is not a rate limit (429).
    """
    last_exc: discord.HTTPException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await send_func(*args, **kwargs)
        except discord.HTTPException as exc:
            if exc.status != 429 or attempt >= max_retries:
                raise

            last_exc = exc

            # discord.py includes retry_after on the response object
            header_wait = getattr(exc, "retry_after", None)
            if header_wait is not None and header_wait > 0:
                wait = float(header_wait) + random.uniform(
                    0, float(header_wait) * JITTER_FACTOR
                )
                wait = min(wait, MAX_DELAY)
            else:
                wait = calculate_backoff(attempt)

            logger.warning(
                "Discord rate limited (429). Retry %d/%d in %.2f seconds",
                attempt + 1,
                max_retries,
                wait,
            )
            await asyncio.sleep(wait)

    # Satisfy type checker — should not reach here
    if last_exc is not None:
        raise last_exc


# ---------------------------------------------------------------------------
# Outbound rate limiter — smooth message bursts
# ---------------------------------------------------------------------------

class OutboundRateLimiter:
    """Token-bucket rate limiter for outbound Discord messages.

    Discord allows ~5 messages per 5 seconds per channel. This limiter
    enforces a minimum gap between sends so bursts of LLM output (long
    responses split into chunks, multi-part replies) don't trigger 429s.

    Usage::

        limiter = get_outbound_limiter()
        await limiter.acquire()  # waits if sending too fast
        await message.reply(text)
    """

    def __init__(self, sends_per_window: int = 5, window_seconds: float = 5.0) -> None:
        self._sends_per_window = sends_per_window
        self._window_seconds = window_seconds
        self._min_gap = window_seconds / sends_per_window  # 1.0s at defaults
        self._lock = asyncio.Lock()
        self._last_send: float = 0.0
        self._send_count: int = 0
        self._wait_count: int = 0

    async def acquire(self) -> None:
        """Wait if necessary to stay under the rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_send
            if elapsed < self._min_gap:
                wait = self._min_gap - elapsed
                self._wait_count += 1
                logger.debug("Outbound rate limiter: waiting %.2fs", wait)
                await asyncio.sleep(wait)
            self._last_send = asyncio.get_event_loop().time()
            self._send_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_sends": self._send_count,
            "throttled_sends": self._wait_count,
            "min_gap_seconds": self._min_gap,
            "throttle_rate": (
                round(self._wait_count / self._send_count * 100, 1)
                if self._send_count > 0
                else 0.0
            ),
        }


_outbound_limiter: OutboundRateLimiter | None = None


def get_outbound_limiter() -> OutboundRateLimiter:
    """Get the global outbound rate limiter (created on first use)."""
    global _outbound_limiter
    if _outbound_limiter is None:
        _outbound_limiter = OutboundRateLimiter()
    return _outbound_limiter
