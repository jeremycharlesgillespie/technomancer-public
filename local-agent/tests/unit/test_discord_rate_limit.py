"""
Tests for agent.discord_rate_limit module.

Covers backoff calculation, header parsing, synchronous retry wrapper,
and async retry wrapper for discord.py calls.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import discord
import pytest

from agent.discord_rate_limit import (
    MAX_DELAY,
    MAX_RETRIES,
    calculate_backoff,
    parse_retry_after,
    retry_request,
    async_retry_on_rate_limit,
)


# =========================================================================
# calculate_backoff
# =========================================================================


class TestCalculateBackoff:
    """Tests for exponential backoff with jitter."""

    def test_first_attempt_bounded(self):
        """First attempt delay should be around BASE_DELAY."""
        for _ in range(50):
            delay = calculate_backoff(0)
            assert 1.0 <= delay <= 1.5  # BASE_DELAY + up to 50% jitter

    def test_increases_with_attempts(self):
        """Later attempts should produce larger average delays."""
        early = [calculate_backoff(0) for _ in range(100)]
        late = [calculate_backoff(3) for _ in range(100)]
        assert sum(late) / len(late) > sum(early) / len(early)

    def test_capped_at_max_delay(self):
        """Very high attempt numbers must not exceed MAX_DELAY + jitter."""
        for _ in range(50):
            delay = calculate_backoff(20)
            # delay = min(base*2^20, MAX_DELAY) + jitter, jitter <= MAX_DELAY*0.5
            assert delay <= MAX_DELAY * 1.5 + 0.01

    def test_always_positive(self):
        """Delay must always be positive."""
        for attempt in range(10):
            assert calculate_backoff(attempt) > 0


# =========================================================================
# parse_retry_after
# =========================================================================


class TestParseRetryAfter:
    """Tests for header parsing from 429 responses."""

    def test_retry_after_header(self):
        """Should parse Retry-After header (seconds)."""
        response = MagicMock()
        response.headers = {"Retry-After": "2.5"}
        result = parse_retry_after(response)
        assert result == 2.5

    def test_rate_limit_reset_header(self):
        """Should parse X-RateLimit-Reset (epoch timestamp)."""
        future = time.time() + 5.0
        response = MagicMock()
        response.headers = {"X-RateLimit-Reset": str(future)}
        result = parse_retry_after(response)
        assert result is not None
        assert 4.0 <= result <= 6.0  # roughly 5 seconds from now

    def test_retry_after_takes_precedence(self):
        """Retry-After should be checked before X-RateLimit-Reset."""
        response = MagicMock()
        response.headers = {
            "Retry-After": "1.0",
            "X-RateLimit-Reset": str(time.time() + 100),
        }
        result = parse_retry_after(response)
        assert result == 1.0

    def test_no_headers_returns_none(self):
        """Should return None when no rate limit headers are present."""
        response = MagicMock()
        response.headers = {}
        assert parse_retry_after(response) is None

    def test_invalid_header_value(self):
        """Should return None for non-numeric header values."""
        response = MagicMock()
        response.headers = {"Retry-After": "not-a-number"}
        assert parse_retry_after(response) is None

    def test_past_reset_time_returns_none(self):
        """Should return None if X-RateLimit-Reset is in the past."""
        response = MagicMock()
        response.headers = {"X-RateLimit-Reset": str(time.time() - 10)}
        assert parse_retry_after(response) is None

    def test_no_headers_attribute(self):
        """Should handle objects without headers attribute."""
        response = object()
        assert parse_retry_after(response) is None

    def test_zero_retry_after(self):
        """Zero or negative Retry-After should fall through to reset header."""
        response = MagicMock()
        response.headers = {"Retry-After": "0"}
        assert parse_retry_after(response) is None


# =========================================================================
# retry_request (synchronous)
# =========================================================================


class TestRetryRequest:
    """Tests for synchronous request retry wrapper."""

    def test_success_on_first_try(self):
        """Should return immediately on non-429 response."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_method = MagicMock(return_value=mock_response)

        result = retry_request(mock_method, "http://example.com", timeout=5)

        assert result.status_code == 204
        assert mock_method.call_count == 1

    def test_retry_on_429_then_success(self):
        """Should retry after 429 and return successful response."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "0.01"}
        rate_limited.request = MagicMock(method="POST", url="http://example.com")

        success = MagicMock()
        success.status_code = 204

        mock_method = MagicMock(side_effect=[rate_limited, success])

        with patch("agent.discord_rate_limit.time.sleep"):
            result = retry_request(mock_method, "http://example.com", timeout=5)

        assert result.status_code == 204
        assert mock_method.call_count == 2

    def test_exhausted_retries_returns_last_response(self):
        """Should return the 429 response after exhausting retries."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "0.01"}
        rate_limited.request = MagicMock(method="POST", url="http://example.com")

        mock_method = MagicMock(return_value=rate_limited)

        with patch("agent.discord_rate_limit.time.sleep"):
            result = retry_request(
                mock_method, "http://example.com", max_retries=2, timeout=5
            )

        assert result.status_code == 429
        # 1 initial + 2 retries = 3 calls
        assert mock_method.call_count == 3

    def test_uses_header_wait_time(self):
        """Should use Retry-After header to determine sleep duration."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "3.0"}
        rate_limited.request = MagicMock(method="POST", url="http://example.com")

        success = MagicMock()
        success.status_code = 200

        mock_method = MagicMock(side_effect=[rate_limited, success])

        with patch("agent.discord_rate_limit.time.sleep") as mock_sleep:
            retry_request(mock_method, "http://example.com", timeout=5)
            # Sleep should be called with ~3.0 + jitter
            assert mock_sleep.call_count == 1
            slept = mock_sleep.call_args[0][0]
            assert 3.0 <= slept <= 3.0 * 1.5 + 0.01

    def test_falls_back_to_backoff_without_headers(self):
        """Should use exponential backoff when no rate limit headers."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}
        rate_limited.request = MagicMock(method="POST", url="http://example.com")

        success = MagicMock()
        success.status_code = 200

        mock_method = MagicMock(side_effect=[rate_limited, success])

        with patch("agent.discord_rate_limit.time.sleep") as mock_sleep:
            retry_request(mock_method, "http://example.com", timeout=5)
            assert mock_sleep.call_count == 1

    def test_forwards_args_and_kwargs(self):
        """Should forward all arguments to the underlying method."""
        success = MagicMock()
        success.status_code = 200
        mock_method = MagicMock(return_value=success)

        retry_request(
            mock_method, "http://example.com", json={"key": "val"}, timeout=10
        )

        mock_method.assert_called_once_with(
            "http://example.com", json={"key": "val"}, timeout=10
        )

    def test_non_429_error_returned_immediately(self):
        """Non-429 status codes should be returned without retry."""
        error_response = MagicMock()
        error_response.status_code = 500
        mock_method = MagicMock(return_value=error_response)

        result = retry_request(mock_method, "http://example.com")
        assert result.status_code == 500
        assert mock_method.call_count == 1


# =========================================================================
# async_retry_on_rate_limit
# =========================================================================


class TestAsyncRetryOnRateLimit:
    """Tests for async discord.py retry wrapper."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Should return immediately when no exception."""
        send = AsyncMock(return_value="sent!")
        result = await async_retry_on_rate_limit(send, "hello")
        assert result == "sent!"
        send.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        """Should retry after 429 HTTPException."""
        exc_429 = discord.HTTPException(MagicMock(status=429), "rate limited")
        exc_429.status = 429
        exc_429.retry_after = 0.01

        send = AsyncMock(side_effect=[exc_429, "sent!"])

        with patch("agent.discord_rate_limit.asyncio.sleep", new_callable=AsyncMock):
            result = await async_retry_on_rate_limit(send, "hello")

        assert result == "sent!"
        assert send.await_count == 2

    @pytest.mark.asyncio
    async def test_non_429_raised_immediately(self):
        """Non-429 HTTPException should be raised without retry."""
        exc_400 = discord.HTTPException(MagicMock(status=400), "bad request")
        exc_400.status = 400

        send = AsyncMock(side_effect=exc_400)

        with pytest.raises(discord.HTTPException):
            await async_retry_on_rate_limit(send, "hello")

        assert send.await_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """Should raise after exhausting all retries."""
        exc_429 = discord.HTTPException(MagicMock(status=429), "rate limited")
        exc_429.status = 429
        exc_429.retry_after = 0.01

        send = AsyncMock(side_effect=exc_429)

        with patch("agent.discord_rate_limit.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(discord.HTTPException):
                await async_retry_on_rate_limit(send, "hello", max_retries=2)

        # 1 initial + 2 retries = 3 calls
        assert send.await_count == 3

    @pytest.mark.asyncio
    async def test_uses_retry_after_attribute(self):
        """Should use exc.retry_after to determine sleep duration."""
        exc_429 = discord.HTTPException(MagicMock(status=429), "rate limited")
        exc_429.status = 429
        exc_429.retry_after = 5.0

        send = AsyncMock(side_effect=[exc_429, "sent!"])

        with patch(
            "agent.discord_rate_limit.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await async_retry_on_rate_limit(send, "hello")
            assert mock_sleep.await_count == 1
            slept = mock_sleep.call_args[0][0]
            assert 5.0 <= slept <= 5.0 * 1.5 + 0.01

    @pytest.mark.asyncio
    async def test_forwards_kwargs(self):
        """Should forward all kwargs to the send function."""
        send = AsyncMock(return_value="ok")
        await async_retry_on_rate_limit(send, "hello", files=["a.txt"])
        send.assert_awaited_once_with("hello", files=["a.txt"])
