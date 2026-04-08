"""
Discord rate limit handling with exponential backoff and jitter.

Provides retry wrappers for both synchronous (requests library) and
asynchronous (discord.py) Discord API calls. Parses rate limit headers
from 429 responses to calculate precise wait times before retrying.
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
