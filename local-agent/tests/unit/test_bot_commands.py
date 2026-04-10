"""Tests for bot_commands.py — Discord command handlers.

Uses mock Discord objects to test command handlers without a live Discord connection.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.bot_commands import (
    handle_add_enhancement,
    handle_learning_history,
    handle_perf,
    handle_show_commands,
    handle_show_enhancements,
    handle_show_learning,
)
from tests.conftest import MockDiscordMessage


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_msg(content="test", user="Jeremy"):
    return MockDiscordMessage(content=content, author_name=user, channel_name="llm_chat")


async def _send_response(message, response):
    """Mock send_response that stores the response on the message."""
    message._last_response = response


class TestHandleShowCommands:
    def test_shows_commands(self):
        msg = _make_msg("commands")
        _run(handle_show_commands(msg))
        assert msg.replied_to is not None
        assert "Available Commands" in msg.replied_to

    def test_includes_betterdev(self):
        msg = _make_msg("help")
        _run(handle_show_commands(msg))
        assert "betterDev" in msg.replied_to

    def test_includes_technews(self):
        msg = _make_msg("showcommands")
        _run(handle_show_commands(msg))
        assert "techNews" in msg.replied_to

    def test_includes_all_sections(self):
        msg = _make_msg()
        _run(handle_show_commands(msg))
        for section in ["Learning", "News", "Memory", "Admin", "YouTube", "Performance"]:
            assert section in msg.replied_to, f"Missing section: {section}"

    def test_includes_newsletter(self):
        msg = _make_msg()
        _run(handle_show_commands(msg))
        assert "newsletter" in msg.replied_to

    def test_includes_suggest(self):
        msg = _make_msg()
        _run(handle_show_commands(msg))
        assert "suggest" in msg.replied_to


class TestHandlePerf:
    @patch("agent.bot_commands.get_performance_summary", return_value="**Performance Summary**\nAvg: 2.5s")
    def test_returns_perf_summary(self, mock_summary):
        msg = _make_msg("perf")
        _run(handle_perf(msg, _send_response))
        assert "Performance" in msg._last_response or "2.5" in msg._last_response


class TestHandleLearningHistory:
    @patch("agent.bot_commands.handle_learning_history_command",
           return_value="**Past Learning Articles**\n1. Python Decorators")
    def test_shows_history(self, mock_cmd):
        msg = _make_msg("learningHistory")
        _run(handle_learning_history(msg, _send_response))
        assert "Learning" in msg._last_response or "Python" in msg._last_response


class TestHandleShowLearning:
    @patch("agent.bot_commands.handle_show_learning_command",
           return_value="# Python Decorators\n\nDecorators are powerful.")
    def test_shows_article(self, mock_cmd):
        msg = _make_msg("showLearning 1")
        _run(handle_show_learning(msg, msg.content, _send_response))
        assert "Decorators" in msg._last_response

    @patch("agent.bot_commands.handle_show_learning_command",
           return_value="Invalid article number: **abc**")
    def test_invalid_number(self, mock_cmd):
        msg = _make_msg("showLearning abc")
        _run(handle_show_learning(msg, msg.content, _send_response))
        assert "Invalid" in msg._last_response


class TestHandleShowEnhancements:
    @patch("agent.bot_commands.get_pending_enhancements",
           return_value="## Pending\n- #1: Add dark mode")
    def test_shows_enhancements(self, mock_get):
        msg = _make_msg("showEnhancements")
        _run(handle_show_enhancements(msg, _send_response))
        assert "dark mode" in msg._last_response or "Pending" in msg._last_response


class TestHandleAddEnhancement:
    @patch("agent.bot_commands.add_enhancement",
           return_value="Added enhancement #3: Better errors")
    def test_adds_enhancement(self, mock_add):
        msg = _make_msg("addEnhancement Better error messages")
        memory = MagicMock()
        _run(handle_add_enhancement(msg, msg.content, "Jeremy", memory))
        assert msg.replied_to is not None
        assert "#3" in msg.replied_to or "Better" in msg.replied_to


class TestHandlePublish:
    @patch("subprocess.run")
    def test_publish_owner_only(self, mock_run):
        from agent.bot_commands import handle_publish
        mock_run.return_value = MagicMock(returncode=0, stdout="Published", stderr="")
        msg = _make_msg("publish", user="Jeremy")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="Jeremy")):
            _run(handle_publish(msg, "Jeremy"))
        # Should have replied (either success or permission denied)
        assert msg.replied_to is not None or len(msg.channel.sent_messages) > 0

    def test_publish_non_owner_rejected(self):
        from agent.bot_commands import handle_publish
        msg = _make_msg("publish", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="Jeremy")):
            _run(handle_publish(msg, "Alice"))
        assert msg.replied_to is not None
        assert "owner" in msg.replied_to.lower()


class TestHandleReloadServer:
    def test_non_owner_rejected(self):
        from agent.bot_commands import handle_reload_server
        msg = _make_msg("reloadServer", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="Jeremy")):
            _run(handle_reload_server(msg, "Alice"))
        assert msg.replied_to is not None


class TestHandleEvolve:
    def test_non_owner_rejected(self):
        from agent.bot_commands import handle_evolve
        msg = _make_msg("evolve", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="Jeremy")):
            _run(handle_evolve(msg, "Alice"))
        assert msg.replied_to is not None


class TestHandleThink:
    @patch("agent.memory_system.get_full_profile", return_value="User profile data here")
    def test_shows_profile(self, mock_profile):
        from agent.bot_commands import handle_think
        msg = _make_msg("think")
        memory = MagicMock()
        _run(handle_think(msg, "think", "Jeremy", memory, _send_response))
        assert hasattr(msg, "_last_response") or msg.replied_to is not None


class TestHandleKaren:
    def test_karen_handler_exists(self):
        from agent.bot_commands import handle_karen
        assert callable(handle_karen)


class TestHandleListVideos:
    @patch("agent.bot_commands.extract_channel_videos")
    def test_lists_videos(self, mock_extract):
        from agent.bot_commands import handle_list_videos
        mock_extract.return_value = [
            {"title": "Video 1", "url": "https://youtube.com/1"},
            {"title": "Video 2", "url": "https://youtube.com/2"},
        ]
        msg = _make_msg("listVideos https://youtube.com/@channel")
        _run(handle_list_videos(msg, msg.content, "Jeremy", _send_response))

    @patch("agent.bot_commands.extract_channel_videos")
    def test_no_url_provided(self, mock_extract):
        from agent.bot_commands import handle_list_videos
        msg = _make_msg("listVideos")
        _run(handle_list_videos(msg, msg.content, "Jeremy", _send_response))
        # Should reply with usage info
        assert msg.replied_to is not None or hasattr(msg, "_last_response")


class TestHandleBetterDev:
    @patch("agent.bot_commands.handle_better_dev_command", new_callable=AsyncMock)
    def test_random_topic(self, mock_cmd):
        from agent.bot_commands import handle_better_dev
        mock_cmd.return_value = ("Article about Python decorators", "https://pages.github.io/article")
        msg = _make_msg("betterDev")
        memory = MagicMock()
        _run(handle_better_dev(msg, msg.content, "Jeremy", memory, _send_response))
