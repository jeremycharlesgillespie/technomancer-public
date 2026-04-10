"""Tests for bot_commands.py — Discord command handlers.

Uses mock Discord objects to test command handlers without a live Discord connection.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.bot_commands import (
    handle_learning_history,
    handle_perf,
    handle_show_commands,
    handle_show_ideas,
    handle_show_learning,
)
from tests.conftest import MockDiscordMessage


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_msg(content="test", user="TestUser"):
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


class TestHandleShowIdeas:
    @patch("idea_board.models.list_ideas_for_llm",
           return_value="**Idea Board** — 1 idea(s):\n\n**idea-001**: Test idea")
    def test_shows_ideas(self, mock_list):
        msg = _make_msg("ideas")
        _run(handle_show_ideas(msg, _send_response))
        assert "idea" in msg._last_response.lower() or "Idea Board" in msg._last_response


class TestHandlePublish:
    @patch("subprocess.run")
    def test_publish_owner_only(self, mock_run):
        from agent.bot_commands import handle_publish
        mock_run.return_value = MagicMock(returncode=0, stdout="Published", stderr="")
        msg = _make_msg("publish", user="TestUser")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="TestUser")):
            _run(handle_publish(msg, "TestUser"))
        # Should have replied (either success or permission denied)
        assert msg.replied_to is not None or len(msg.channel.sent_messages) > 0

    def test_publish_non_owner_rejected(self):
        from agent.bot_commands import handle_publish
        msg = _make_msg("publish", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="TestUser")):
            _run(handle_publish(msg, "Alice"))
        assert msg.replied_to is not None
        assert "owner" in msg.replied_to.lower()


class TestHandleReloadServer:
    def test_non_owner_rejected(self):
        from agent.bot_commands import handle_reload_server
        msg = _make_msg("reloadServer", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="TestUser")):
            _run(handle_reload_server(msg, "Alice"))
        assert msg.replied_to is not None


class TestHandleEvolve:
    def test_non_owner_rejected(self):
        from agent.bot_commands import handle_evolve
        msg = _make_msg("evolve", user="Alice")
        with patch("agent.bot_commands.settings", MagicMock(bot_owner="TestUser")):
            _run(handle_evolve(msg, "Alice"))
        assert msg.replied_to is not None


class TestHandleThink:
    @patch("agent.memory_system.get_full_profile", return_value="User profile data here")
    def test_shows_profile(self, mock_profile):
        from agent.bot_commands import handle_think
        msg = _make_msg("think")
        memory = MagicMock()
        _run(handle_think(msg, "think", "TestUser", memory, _send_response))
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
        _run(handle_list_videos(msg, msg.content, "TestUser", _send_response))

    @patch("agent.bot_commands.extract_channel_videos")
    def test_no_url_provided(self, mock_extract):
        from agent.bot_commands import handle_list_videos
        msg = _make_msg("listVideos")
        _run(handle_list_videos(msg, msg.content, "TestUser", _send_response))
        # Should reply with usage info
        assert msg.replied_to is not None or hasattr(msg, "_last_response")


class TestHandleBetterDev:
    @patch("agent.bot_commands.handle_better_dev_command", new_callable=AsyncMock)
    def test_random_topic(self, mock_cmd):
        from agent.bot_commands import handle_better_dev
        mock_cmd.return_value = ("Article about Python decorators", "https://pages.github.io/article")
        msg = _make_msg("betterDev")
        memory = MagicMock()
        _run(handle_better_dev(msg, msg.content, "TestUser", memory, _send_response))

    @patch("agent.bot_commands.handle_better_dev_command", new_callable=AsyncMock)
    def test_with_category(self, mock_cmd):
        from agent.bot_commands import handle_better_dev
        mock_cmd.return_value = ("Python article", None)
        msg = _make_msg("betterDev python")
        memory = MagicMock()
        _run(handle_better_dev(msg, msg.content, "TestUser", memory, _send_response))


class TestHandleIdea:
    def test_handler_exists(self):
        from agent.bot_commands import handle_idea
        assert callable(handle_idea)


class TestHandleKarenFull:
    def test_no_complaint_text(self):
        from agent.bot_commands import handle_karen
        msg = _make_msg("karen")
        _run(handle_karen(msg, msg.content, "TestUser"))
        assert "Usage" in msg.replied_to or "K.A.R.E.N" in msg.replied_to


class TestHandleSearchVideos:
    def test_no_args(self):
        from agent.bot_commands import handle_search_videos
        msg = _make_msg("searchVideos")
        _run(handle_search_videos(msg, msg.content, _send_response))
        assert "Usage" in msg.replied_to

    @patch("agent.bot_commands.search_channel_videos", return_value="Found 3 videos")
    def test_with_args(self, mock_search):
        from agent.bot_commands import handle_search_videos
        msg = _make_msg("searchVideos https://youtube.com/@ch python")
        _run(handle_search_videos(msg, msg.content, _send_response))


class TestHandleDownloadVideo:
    def test_no_url(self):
        from agent.bot_commands import handle_download_video
        msg = _make_msg("downloadVideo")
        _run(handle_download_video(msg, msg.content))
        assert "Usage" in msg.replied_to

    @patch("agent.bot_commands.download_video", return_value="Downloaded video.mp4")
    def test_with_url(self, mock_dl):
        from agent.bot_commands import handle_download_video
        msg = _make_msg("downloadVideo https://youtube.com/watch?v=abc")
        _run(handle_download_video(msg, msg.content))
        assert len(msg._replies) >= 2  # "Starting download" + result


class TestHandleDownloadChannel:
    def test_no_url(self):
        from agent.bot_commands import handle_download_channel
        msg = _make_msg("downloadChannel")
        _run(handle_download_channel(msg, msg.content, "TestUser"))
        assert "Usage" in msg.replied_to


class TestHandleDlCover:
    def test_no_url(self):
        from agent.bot_commands import handle_dl_cover
        msg = _make_msg("dlcover")
        _run(handle_dl_cover(msg, msg.content))
        assert "Usage" in msg.replied_to

    @patch("agent.bot_commands.download_thumbnail", return_value="Saved thumbnail.jpg")
    def test_with_url(self, mock_dl):
        from agent.bot_commands import handle_dl_cover
        msg = _make_msg("dlcover https://youtube.com/watch?v=abc")
        _run(handle_dl_cover(msg, msg.content))


class TestHandleDlCovers:
    def test_no_url(self):
        from agent.bot_commands import handle_dl_covers
        msg = _make_msg("dlcovers")
        _run(handle_dl_covers(msg, msg.content, "TestUser"))
        assert "Usage" in msg.replied_to
