"""Tests for message validation utilities."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from agent.message_validators import (
    DISCORD_MAX_LENGTH,
    MAX_EMPTY_RETRIES,
    TRUNCATION_SUFFIX,
    _default_suggestion,
    _enrich_empty_content,
    _is_empty_message_error,
    _mentions_pattern,
    _recent_messages,
    buffer_user_message,
    get_contextual_suggestion,
    resilient_send,
    safe_send_content,
    validate_discord_message,
)


# ── validate_discord_message ─────────────────────────────────────────────


class TestValidateDiscordMessage:
    """Tests for validate_discord_message."""

    def test_none_returns_none(self):
        assert validate_discord_message(None) is None

    def test_empty_string_returns_none(self):
        assert validate_discord_message("") is None

    def test_whitespace_only_returns_none(self):
        assert validate_discord_message("   \n\t  ") is None

    def test_normal_message_returned_stripped(self):
        assert validate_discord_message("  hello  ") == "hello"

    def test_message_at_limit_not_truncated(self):
        msg = "a" * DISCORD_MAX_LENGTH
        result = validate_discord_message(msg)
        assert result == msg
        assert len(result) == DISCORD_MAX_LENGTH

    def test_message_over_limit_truncated(self):
        msg = "a" * (DISCORD_MAX_LENGTH + 500)
        result = validate_discord_message(msg)
        assert result is not None
        assert len(result) == DISCORD_MAX_LENGTH
        assert result.endswith(TRUNCATION_SUFFIX)

    def test_message_one_over_limit_truncated(self):
        msg = "a" * (DISCORD_MAX_LENGTH + 1)
        result = validate_discord_message(msg)
        assert result is not None
        assert len(result) == DISCORD_MAX_LENGTH
        assert result.endswith(TRUNCATION_SUFFIX)

    def test_short_message_unchanged(self):
        assert validate_discord_message("Hello, world!") == "Hello, world!"

    def test_multiline_message_preserved(self):
        msg = "line 1\nline 2\nline 3"
        assert validate_discord_message(msg) == msg

    def test_truncated_content_preserves_beginning(self):
        prefix = "KEEP THIS "
        msg = prefix + "x" * DISCORD_MAX_LENGTH
        result = validate_discord_message(msg)
        assert result is not None
        assert result.startswith(prefix)


# ── safe_send_content ────────────────────────────────────────────────────


class TestSafeSendContent:
    """Tests for safe_send_content."""

    def setup_method(self):
        _recent_messages.clear()

    def test_valid_content_returned(self):
        assert safe_send_content("hello") == "hello"

    def test_empty_content_uses_contextual_suggestion(self):
        # With no recent messages, should get default suggestion
        result = safe_send_content("")
        assert "trouble generating" in result

    def test_none_content_uses_contextual_suggestion(self):
        result = safe_send_content(None)
        assert "trouble generating" in result

    def test_whitespace_content_uses_contextual_suggestion(self):
        result = safe_send_content("   ")
        assert "trouble generating" in result

    def test_empty_content_with_recent_code_message(self):
        buffer_user_message("I have a python error")
        result = safe_send_content("")
        assert "error" in result.lower() or "betterDev" in result

    def test_long_content_truncated(self):
        msg = "b" * (DISCORD_MAX_LENGTH + 100)
        result = safe_send_content(msg)
        assert len(result) == DISCORD_MAX_LENGTH

    def test_fallback_not_used_when_content_valid(self):
        assert safe_send_content("real content", "fallback") == "real content"


# ── _mentions_pattern ─────────────────────────────────────────────


class TestMentionsPattern:
    """Tests for _mentions_pattern."""

    def test_matches_code_pattern(self):
        messages = ["I have a python error"]
        assert _mentions_pattern(messages, r"\b(code|python)\b") is True

    def test_no_match(self):
        messages = ["hello world"]
        assert _mentions_pattern(messages, r"\b(code|python)\b") is False

    def test_case_insensitive(self):
        messages = ["I have a PYTHON error"]
        assert _mentions_pattern(messages, r"\b(code|python)\b") is True

    def test_empty_message_list(self):
        messages = []
        assert _mentions_pattern(messages, r"\b(code|python)\b") is False

    def test_multiple_messages_one_matches(self):
        messages = ["hello", "I need help with coding"]
        assert _mentions_pattern(messages, r"\b(code|python)\b") is False

    def test_multiple_messages_none_match(self):
        messages = ["hello", "world", "test"]
        assert _mentions_pattern(messages, r"\b(code|python)\b") is False

    def test_multiple_matches(self):
        messages = ["I code in Python and JavaScript"]
        assert _mentions_pattern(messages, r"\b(code|python|javascript)\b") is True

    def test_empty_pattern(self):
        messages = ["hello"]
        assert _mentions_pattern(messages, "") is False


# ── _is_empty_message_error ─────────────────────────────────────────────


class TestIsEmptyMessageError:
    """Tests for _is_empty_message_error."""

    def _make_http_exc(self, status: int, text: str) -> discord.HTTPException:
        resp = MagicMock()
        resp.status = status
        resp.reason = text
        exc = discord.HTTPException(resp, text)
        return exc

    def test_400_empty_message_detected(self):
        exc = self._make_http_exc(400, "Cannot send an empty message")
        assert _is_empty_message_error(exc) is True

    def test_400_other_error_not_detected(self):
        exc = self._make_http_exc(400, "Invalid Form Body")
        assert _is_empty_message_error(exc) is False

    def test_non_400_not_detected(self):
        exc = self._make_http_exc(403, "Missing Permissions with empty message")
        assert _is_empty_message_error(exc) is False


# ── _enrich_empty_content ───────────────────────────────────────────────


class TestEnrichEmptyContent:
    """Tests for _enrich_empty_content."""

    def setup_method(self):
        _recent_messages.clear()

    def test_first_attempt_uses_contextual_suggestion(self):
        result = _enrich_empty_content(1)
        # First retry uses contextual suggestion instead of generic retry text
        assert "trouble generating" in result
        assert "retry" not in result

    def test_contains_retry_number_on_later_attempts(self):
        result = _enrich_empty_content(2)
        assert "retry 2/" in result

    def test_contains_timestamp_on_later_attempts(self):
        result = _enrich_empty_content(3)
        # Should contain a date-like pattern
        assert "202" in result  # year prefix

    def test_not_empty(self):
        result = _enrich_empty_content(1)
        assert len(result) > 0


# ── resilient_send ──────────────────────────────────────────────────────


class TestResilientSend:
    """Tests for resilient_send."""

    def _make_empty_exc(self) -> discord.HTTPException:
        resp = MagicMock()
        resp.status = 400
        resp.reason = "Cannot send an empty message"
        return discord.HTTPException(resp, "Cannot send an empty message")

    def test_success_on_first_try(self):
        send = AsyncMock(return_value="ok")
        result = asyncio.run(resilient_send(send, "hello"))
        assert result == "ok"
        send.assert_awaited_once_with("hello")

    def test_retries_on_empty_message_error(self):
        _recent_messages.clear()
        exc = self._make_empty_exc()
        send = AsyncMock(side_effect=[exc, "ok"])
        result = asyncio.run(resilient_send(send, ""))
        assert result == "ok"
        assert send.await_count == 2
        # Second call should use contextual suggestion (first retry)
        second_call_content = send.call_args_list[1][0][0]
        assert "trouble generating" in second_call_content

    def test_raises_after_max_retries(self):
        exc = self._make_empty_exc()
        send = AsyncMock(side_effect=exc)
        with pytest.raises(discord.HTTPException):
            asyncio.run(resilient_send(send, ""))
        assert send.await_count == MAX_EMPTY_RETRIES + 1

    def test_non_empty_400_not_retried(self):
        resp = MagicMock()
        resp.status = 400
        resp.reason = "Invalid Form Body"
        exc = discord.HTTPException(resp, "Invalid Form Body")

        send = AsyncMock(side_effect=exc)
        with pytest.raises(discord.HTTPException):
            asyncio.run(resilient_send(send, "test"))
        send.assert_awaited_once()

    def test_non_400_not_retried(self):
        resp = MagicMock()
        resp.status = 500
        resp.reason = "Internal Server Error"
        exc = discord.HTTPException(resp, "Internal Server Error")

        send = AsyncMock(side_effect=exc)
        with pytest.raises(discord.HTTPException):
            asyncio.run(resilient_send(send, "test"))
        send.assert_awaited_once()

    def test_kwargs_forwarded(self):
        send = AsyncMock(return_value="ok")
        asyncio.run(resilient_send(send, "hello", files=["f1"]))
        send.assert_awaited_once_with("hello", files=["f1"])

    def test_files_dropped_on_retry(self):
        exc = self._make_empty_exc()
        send = AsyncMock(side_effect=[exc, "ok"])
        asyncio.run(resilient_send(send, "", files=["f1"]))
        # First call has files
        assert "files" in send.call_args_list[0][1]
        # Second call (retry) should not have files
        assert "files" not in send.call_args_list[1][1]


# ── buffer_user_message ────────────────────────────────────────────────


class TestBufferUserMessage:
    """Tests for buffer_user_message."""

    def setup_method(self):
        _recent_messages.clear()

    def test_buffers_message(self):
        buffer_user_message("hello world")
        assert "hello world" in _recent_messages

    def test_ignores_empty_messages(self):
        buffer_user_message("")
        buffer_user_message("   ")
        assert len(_recent_messages) == 0

    def test_maxlen_five(self):
        for i in range(10):
            buffer_user_message(f"message {i}")
        assert len(_recent_messages) == 5
        assert "message 5" in _recent_messages
        assert "message 9" in _recent_messages

    def test_strips_whitespace(self):
        buffer_user_message("  padded  ")
        assert _recent_messages[0] == "padded"


# ── get_contextual_suggestion ──────────────────────────────────────────


class TestGetContextualSuggestion:
    """Tests for get_contextual_suggestion."""

    def setup_method(self):
        _recent_messages.clear()

    def test_default_suggestion_when_no_messages(self):
        result = get_contextual_suggestion()
        assert "trouble generating" in result
        assert "showCommands" in result

    def test_code_topic_detected(self):
        buffer_user_message("I have a python bug")
        result = get_contextual_suggestion()
        assert "error" in result.lower() or "betterDev" in result

    def test_news_topic_detected(self):
        buffer_user_message("what's the latest tech news?")
        result = get_contextual_suggestion()
        assert "technews" in result

    def test_time_topic_detected(self):
        buffer_user_message("what time is it?")
        result = get_contextual_suggestion()
        assert "time" in result.lower()

    def test_learning_topic_detected(self):
        buffer_user_message("can you explain how decorators work?")
        result = get_contextual_suggestion()
        assert "betterDev" in result

    def test_memory_topic_detected(self):
        buffer_user_message("what do you know about me?")
        result = get_contextual_suggestion()
        assert "think" in result

    def test_video_topic_detected(self):
        buffer_user_message("download this youtube video")
        result = get_contextual_suggestion()
        assert "Video" in result or "video" in result

    def test_matches_across_recent_messages(self):
        # Topic appears in earlier message, not just the latest
        buffer_user_message("show me the latest news")
        buffer_user_message("thanks")
        result = get_contextual_suggestion()
        assert "technews" in result

    def test_always_returns_non_empty(self):
        buffer_user_message("random gibberish xyzzy")
        result = get_contextual_suggestion()
        assert len(result) > 0
        assert "trouble generating" in result
