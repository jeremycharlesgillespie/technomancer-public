"""
Bot Commands — Discord command handlers extracted from discord_memory_bot.py.

Each function handles one Discord text command. They receive the message,
content, user, agent, memory, and any other dependencies they need.

These are called from the on_message dispatcher in discord_memory_bot.py.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import discord

from .bot_utils import log
from .config import settings
from .core import Agent, AgentConfig
from .dev_learning import (
    handle_better_dev_command,
    handle_learning_history_command,
    handle_show_learning_command,
)
from .enhancements import add_enhancement, get_pending_enhancements
from .news_digest import handle_technews_command
from .metrics_db import get_summary as get_metrics_summary
from .perf_monitor import get_endpoint_summary
from .profiler import get_performance_summary
from .youtube_tools import (
    download_channel_thumbnails,
    download_channel_videos,
    download_thumbnail,
    download_video,
    extract_channel_videos,
    search_channel_videos,
)


async def handle_perf(message: Any, send_response: Any) -> None:
    """Show performance profiling data."""
    summary = get_performance_summary()
    endpoint_summary = get_endpoint_summary()
    await send_response(message, f"{summary}\n\n---\n\n{endpoint_summary}")


async def handle_metrics(message: Any, send_response: Any) -> None:
    """Show persistent LLM metrics with trend analysis from SQLite."""
    summary = get_metrics_summary(hours=24)
    await send_response(message, summary)


async def handle_idea(message: Any, send_response: Any) -> None:
    """Trigger immediate idea generation."""
    await message.reply("Generating ideas from news, conversations, and performance data...")
    async with message.channel.typing():
        from .idea_generator import generate_ideas

        idea_agent = Agent(AgentConfig(
            model=settings.ollama_model,
            verbose=False,
            system_prompt="You are an improvement analyst. Output only JSON arrays.",
        ))
        created = await generate_ideas(idea_agent)
        if created:
            titles = "\n".join(f"- {c['title']}" for c in created)
            await message.reply(
                f"Generated {len(created)} new idea(s):\n{titles}\n\n"
                f"View the board: http://localhost:8322"
            )
        else:
            await message.reply("No new ideas this cycle — everything looks good.")


async def handle_better_dev(
    message: Any, content: str, user: str, memory: Any, send_response: Any
) -> None:
    """Handle betterDev learning command."""
    async with message.channel.typing():
        parts = content.split(maxsplit=1)
        category = parts[1].lower().strip() if len(parts) > 1 else None
        response, html_url = await handle_better_dev_command(category)
        await send_response(message, response)
        if memory:
            memory.log_conversation(user, content, "[Generated developer learning content]")


async def handle_reload_server(message: Any, user: str) -> None:
    """Restart the bot (owner only)."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can reload the server.")
        return

    await message.reply("Restarting bot... I'll be back in a few seconds.")
    log(f"Server reload requested by {user}")

    subprocess.Popen(
        [sys.executable, "bot_service.py", "start"],
        cwd=Path(__file__).parent.parent,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


async def handle_evolve(message: Any, user: str) -> None:
    """Run self-improvement cycle (owner only)."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can trigger evolution.")
        return

    await message.reply("Starting self-improvement cycle... I'll post results when done.")
    log(f"Evolution cycle triggered by {user}")

    async def run_evolve():
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(Path(__file__).parent.parent / "auto_improve.py")],
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=Path(__file__).parent.parent,
            )
            if result.returncode != 0 and result.stderr:
                log(f"Evolution error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log("Evolution cycle timed out (30 min)")
        except Exception as e:
            log(f"Evolution error: {e}")

    asyncio.create_task(run_evolve())


async def handle_show_commands(message: Any) -> None:
    """Show available commands."""
    commands_list = """**Available Commands**

**Learning & Development**
`betterDev` - Random dev topic from any category
`betterDev python` - Python-specific topic
`betterDev oracle` - Oracle/database topic
`betterDev system_design` - System design topic
`betterDev best_practices` - Best practices topic
`learningHistory` - List past learning articles
`showLearning <#>` - View a saved article (e.g., `showLearning 1`)

**News**
`techNews` - Get latest tech news with analysis (3 articles)

**Memory**
`think` - Show what I know about you (permanent memories)

**Enhancements**
`showEnhancements` - Show pending enhancement queue
`addEnhancement <idea>` - Add a new enhancement to the queue

**YouTube**
`listVideos <url>` - List videos from a YouTube channel
`searchVideos <url> <term>` - Search videos in a channel
`downloadVideo <url>` - Download a YouTube video
`downloadChannel <url>` - Download all videos from a channel
`dlcover <url>` - Download video thumbnail/cover art
`dlcovers <channel_url>` - Download all thumbnails from a channel

**Admin**
`reloadServer` - Restart the bot (owner only)
`evolve` - Run self-improvement cycle (owner only)

**Performance**
`perf` - Show current session profiling data
`metrics` - Show persistent LLM latency trends (SQLite-backed)

**Info**
`showCommands` - Show this help message

**Other Features**
- Upload PDF/DOCX/TXT files for analysis
- Share images for identification (local + Claude)
- Reply to news posts to ask questions about them
- Just chat naturally - I'll remember important things!
"""
    await message.reply(commands_list)


async def handle_learning_history(message: Any, send_response: Any) -> None:
    """Show past learning articles."""
    response = handle_learning_history_command()
    await send_response(message, response)


async def handle_show_learning(message: Any, content: str, send_response: Any) -> None:
    """Show a specific learning article."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply("Usage: `showLearning <number>` (e.g., `showLearning 1`)")
        return
    response = handle_show_learning_command(parts[1])
    await send_response(message, response)


async def handle_tech_news(
    message: Any, content: str, user: str, agent: Any, memory: Any, send_response: Any
) -> None:
    """Get on-demand tech news."""
    async with message.channel.typing():
        response = await handle_technews_command(agent)
        await send_response(message, response)
        if memory:
            memory.log_conversation(user, content, "[Generated tech news digest]")


async def handle_show_enhancements(message: Any, send_response: Any) -> None:
    """Show pending enhancements."""
    response = get_pending_enhancements()
    await send_response(message, response)


async def handle_add_enhancement(
    message: Any, content: str, user: str, memory: Any
) -> None:
    """Add a new enhancement."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `addEnhancement <description>`\n"
            "Example: `addEnhancement I want the bot to check stock prices`"
        )
        return

    idea = parts[1].strip()
    response = add_enhancement(idea)
    await message.reply(response)
    if memory:
        memory.log_conversation(user, content, f"[Added enhancement: {idea[:50]}...]")


async def handle_think(
    message: Any, content: str, user: str, memory: Any, send_response: Any
) -> None:
    """Show permanent knowledge about the user."""
    from .memory_system import get_full_profile

    async with message.channel.typing():
        try:
            if memory:
                permanent_memories = memory.get_context("permanent")

                if "No permanent memories" in permanent_memories:
                    response = (
                        "**What I Know About You**\n\n"
                        "I don't have any permanent memories saved about you yet. "
                        "As we chat, I'll remember important things."
                    )
                else:
                    response = f"**What I Know About You**\n\n{permanent_memories}"

                    if len(response) > 1800:
                        log("Permanent memories too long, using file option")
                        full_profile = await asyncio.to_thread(get_full_profile, save_to_file=True)
                        await send_response(message, full_profile)
                        if memory:
                            memory.log_conversation(user, content, "[Showed permanent knowledge via file]")
                        return
            else:
                response = "Memory system not available."

            await send_response(message, response)
            if memory:
                memory.log_conversation(user, content, "[Showed permanent knowledge]")
        except Exception as e:
            log(f"Error in think command: {e}")
            await message.reply(f"Sorry, had trouble accessing my memories: {e}")


# =============================================================================
# YOUTUBE COMMANDS
# =============================================================================

async def handle_list_videos(message: Any, content: str, user: str, send_response: Any) -> None:
    """List YouTube channel videos."""
    log(f"[listvideos] Received from {user}: {content[:80]}")
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `listvideos <channel_url>`\n"
            "Example: `listvideos https://www.youtube.com/@ChannelName`"
        )
        return

    channel_url = parts[1].strip()
    async with message.channel.typing():
        try:
            result = await asyncio.to_thread(extract_channel_videos, channel_url)
            await send_response(message, result)
        except Exception as e:
            log(f"Error listing videos: {e}")
            await message.reply(f"Error listing videos: {e}")


async def handle_search_videos(message: Any, content: str, send_response: Any) -> None:
    """Search within a YouTube channel."""
    parts = content.split(maxsplit=2)
    if len(parts) < 3:
        await message.reply(
            "Usage: `searchvideos <channel_url> <search_term>`\n"
            "Example: `searchvideos https://www.youtube.com/@ChannelName python tutorial`"
        )
        return

    channel_url = parts[1].strip()
    search_term = parts[2].strip()
    async with message.channel.typing():
        try:
            result = await asyncio.to_thread(search_channel_videos, channel_url, search_term)
            await send_response(message, result)
        except Exception as e:
            log(f"Error searching videos: {e}")
            await message.reply(f"Error searching videos: {e}")


async def handle_download_video(message: Any, content: str) -> None:
    """Download a YouTube video."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `downloadVideo <url>`\n"
            "Example: `downloadVideo https://www.youtube.com/watch?v=dQw4w9WgXcQ`"
        )
        return

    video_url = parts[1].strip()
    await message.reply("Starting download... This may take a while.")
    async with message.channel.typing():
        try:
            result = await asyncio.to_thread(download_video, video_url)
            await message.reply(result)
        except Exception as e:
            log(f"Error downloading video: {e}")
            await message.reply(f"Error downloading video: {e}")


async def handle_download_channel(message: Any, content: str, user: str) -> None:
    """Download all videos from a YouTube channel."""
    log(f"[downloadchannel] Received from {user}: {content[:80]}")
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `downloadChannel <channel_url>`\n"
            "Example: `downloadChannel https://www.youtube.com/@ChannelName`"
        )
        return

    channel_url = parts[1].strip()

    async def send_progress(msg: str) -> None:
        try:
            await message.channel.send(msg)
        except Exception:
            pass

    await message.reply(f"Starting channel download: {channel_url}")
    try:
        result = await download_channel_videos(
            channel_url, max_videos=50, progress_callback=send_progress
        )
        await message.reply(result)
    except Exception as e:
        log(f"Error downloading channel: {e}")
        await message.reply(f"Error downloading channel: {e}")


async def handle_dl_cover(message: Any, content: str) -> None:
    """Download video thumbnail/cover art."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `dlcover <url>`\n"
            "Example: `dlcover https://www.youtube.com/watch?v=dQw4w9WgXcQ`"
        )
        return

    video_url = parts[1].strip()
    await message.reply("Downloading thumbnail...")
    async with message.channel.typing():
        try:
            result = await asyncio.to_thread(download_thumbnail, video_url)
            await message.reply(result)
        except Exception as e:
            log(f"Error downloading thumbnail: {e}")
            await message.reply(f"Error downloading thumbnail: {e}")


async def handle_dl_covers(message: Any, content: str, user: str) -> None:
    """Download all thumbnails from a YouTube channel (background)."""
    log(f"[dlcovers] Received from {user}: {content[:80]}")
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "Usage: `dlcovers <channel_url>`\n"
            "Example: `dlcovers https://www.youtube.com/@ChannelName`\n"
            "Downloads ALL thumbnails (1 per minute). Runs in background."
        )
        return

    channel_url = parts[1].strip()
    channel = message.channel

    async def background_download():
        try:
            log(f"[dlcovers] Background task started for {channel_url}")

            async def send_progress(msg: str) -> None:
                try:
                    await channel.send(msg)
                except Exception:
                    pass

            result = await download_channel_thumbnails(
                channel_url, max_thumbnails=10000, progress_callback=send_progress
            )
            await channel.send(result)
            log(f"[dlcovers] Background task completed for {channel_url}")
        except Exception as e:
            log(f"[dlcovers] Background task error: {e}")
            try:
                await channel.send(f"Thumbnail download failed: {e}")
            except Exception:
                pass

    asyncio.create_task(background_download())
    await message.reply(
        f"Started background thumbnail download for: {channel_url}\n"
        f"Downloading 1 per minute to avoid rate limits.\n"
        f"Progress updates will be posted here. You can close Discord."
    )
