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

from .task_manager import create_monitored_task

from .bot_utils import log
from .config import settings
from .core import Agent, AgentConfig
from .dev_learning import (
    handle_better_dev_command,
    handle_learning_history_command,
    handle_show_learning_command,
    handle_suggest_learning_command,
)
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


async def handle_publish(message: Any, user: str) -> None:
    """Sync code to technomancer-public repo and push (owner only)."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can publish.")
        return

    await message.reply("Publishing to technomancer-public...")

    async def _run_publish() -> None:
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "publish.py", "--push", "--force"],
                capture_output=True, text=True, timeout=120,
                cwd=Path(__file__).parent.parent,
            )
            output = result.stdout.strip()
            if result.returncode == 0:
                await message.channel.send(f"Published successfully!\n```\n{output[-500:]}\n```")
            else:
                error = result.stderr.strip() or output
                await message.channel.send(f"Publish failed:\n```\n{error[-500:]}\n```")
        except Exception as e:
            await message.channel.send(f"Publish error: {e}")

    create_monitored_task(_run_publish(), "publish-command")


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
                f"View the board: http://{settings.server_host}:8322"
            )
        else:
            await message.reply("No new ideas this cycle — everything looks good.")


async def handle_karen(message: Any, content: str, user: str) -> None:
    """Submit a complaint to KAREN and generate improvement ideas."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "**K.A.R.E.N.** — Kinetic Aggression Routing Enhancement Network\n"
            "Usage: `karen <your complaint>`\n"
            "Example: `karen The news digest keeps sending me articles about crypto`"
        )
        return

    complaint_text = parts[1].strip()
    await message.reply("Complaint received. Generating improvement ideas from your frustration...")

    async with message.channel.typing():
        try:
            from idea_board.karen import add_complaint, generate_ideas_from_complaint

            complaint = add_complaint(text=complaint_text, author=user)
            idea_ids = await asyncio.to_thread(generate_ideas_from_complaint, complaint)

            if idea_ids:
                from idea_board.models import load_ideas

                ideas = load_ideas()
                lines = [f"Generated {len(idea_ids)} idea(s) from your complaint:"]
                for iid in idea_ids:
                    idea = next((i for i in ideas if i.id == iid), None)
                    if idea:
                        lines.append(f"- **{idea.id}**: {idea.title}")
                lines.append(f"\nView the board: http://{settings.server_host}:8322/karen")
                await message.reply("\n".join(lines))
            else:
                await message.reply(
                    "Complaint logged but no new ideas could be generated. "
                    "It might overlap with existing ideas."
                )
        except Exception as e:
            log(f"[KAREN] Error: {e}")
            await message.reply(f"Error processing complaint: {e}")


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

    await message.reply("Starting self-improvement cycle (up to 60 min)... I'll post results when done.")
    log(f"Evolution cycle triggered by {user}")

    async def run_evolve() -> None:
        """Run auto_improve.py and update evolve status on completion."""
        import json
        status_file = Path(__file__).parent.parent / ".evolve_status.json"
        start_time = datetime.now()

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(Path(__file__).parent.parent / "auto_improve.py")],
                capture_output=True,
                text=True,
                timeout=3600,  # 60 minutes
                cwd=Path(__file__).parent.parent,
            )
            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                log(f"Evolution cycle completed in {elapsed:.0f}s")
                await message.channel.send(f"Evolution cycle completed ({elapsed:.0f}s)")
            else:
                error = result.stderr[:300] or result.stdout[-300:]
                log(f"Evolution failed (exit {result.returncode}, {elapsed:.0f}s): {error}")
                await message.channel.send(f"Evolution failed (exit {result.returncode}):\n```\n{error}\n```")

                # Update status file so dashboard shows failure
                try:
                    status = {"running": False, "phase": "failed", "progress": f"Exit code {result.returncode}: {error[:200]}"}
                    status_file.write_text(json.dumps(status), encoding="utf-8")
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            elapsed = (datetime.now() - start_time).total_seconds()
            log(f"Evolution cycle timed out after {elapsed:.0f}s (60 min limit)")
            await message.channel.send(f"Evolution timed out after {elapsed:.0f}s. Check the idea board for partial results.")

            try:
                status = {"running": False, "phase": "timeout", "progress": f"Timed out after {elapsed:.0f}s"}
                status_file.write_text(json.dumps(status), encoding="utf-8")
            except OSError:
                pass

        except Exception as e:
            log(f"Evolution error: {e}")
            await message.channel.send(f"Evolution error: {e}")

            try:
                status = {"running": False, "phase": "error", "progress": str(e)[:200]}
                status_file.write_text(json.dumps(status), encoding="utf-8")
            except OSError:
                pass

    create_monitored_task(run_evolve(), "evolve-command")


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
`suggestLearning` - Suggest topics based on recent conversations
`showLearning <#>` - View a saved article (e.g., `showLearning 1`)
`newsletter` - Get this week's learning digest

**News**
`techNews` - Get latest tech news with analysis (3 articles)

**Memory**
`think` - Show what I know about you (permanent memories)

**Idea Board**
`ideas` - Show active ideas from the idea board (or visit http://{settings.server_host}:8322/ideas)

**Feedback**
`karen <complaint>` - Submit a complaint to K.A.R.E.N. (generates improvement ideas)

**YouTube**
`listVideos <url>` - List videos from a YouTube channel
`searchVideos <url> <term>` - Search videos in a channel
`downloadVideo <url>` - Download a YouTube video
`downloadChannel <url>` - Download all videos from a channel
`dlcover <url>` - Download video thumbnail/cover art
`dlcovers <channel_url>` - Download all thumbnails from a channel

**Project Tracker**
`track <name> [url]` - Track a new project (owner only)
`untrack <name>` - Remove a tracked project (owner only)
`projects` - List all tracked projects
`project <name>` - Show project details
`blocker <name> <text>` - Add a blocker to a project (owner only)

**Admin**
`reloadServer` - Restart the bot (owner only)
`evolve` - Run self-improvement cycle (owner only)

**Performance**
`perf` - Show current session profiling data
`metrics` - Show persistent LLM latency trends (SQLite-backed)

**Info**
`showCommands` - Show this help message
`suggest` - Get command suggestions based on recent conversation context

**Other Features**
- Upload PDF/DOCX/TXT files for analysis
- Share images for identification (local + Claude)
- Reply to news posts to ask questions about them
- Just chat naturally - I'll remember important things!
"""
    await message.reply(
        commands_list.replace("{settings.server_host}", settings.server_host)
    )


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


async def handle_suggest_learning(message: Any, send_response: Any) -> None:
    """Suggest learning topics based on recent conversations."""
    response = handle_suggest_learning_command()
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


async def handle_show_ideas(message: Any, send_response: Any) -> None:
    """Show active ideas from the idea board."""
    from idea_board.models import list_ideas_for_llm

    response = list_ideas_for_llm()
    await send_response(message, response)


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

    create_monitored_task(background_download(), "dl-covers-command")
    await message.reply(
        f"Started background thumbnail download for: {channel_url}\n"
        f"Downloading 1 per minute to avoid rate limits.\n"
        f"Progress updates will be posted here. You can close Discord."
    )


# ================================================================
# Project Tracker commands
# ================================================================

async def handle_track(message: Any, content: str, user: str) -> None:
    """Track a new project: track <name> [url]."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can manage projects.")
        return
    parts = content.split(maxsplit=2)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "**Track a Project**\n"
            "Usage: `track <name> [repo_url]`\n"
            "Example: `track myapp https://github.com/user/myapp`"
        )
        return
    name = parts[1].strip()
    repo_url = parts[2].strip() if len(parts) > 2 else ""
    from .project_tracker import add_project
    result = add_project(name, repo_url)
    await message.reply(result)


async def handle_untrack(message: Any, content: str, user: str) -> None:
    """Remove a tracked project: untrack <name>."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can manage projects.")
        return
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "**Untrack a Project**\n"
            "Usage: `untrack <name>`\n"
            "Example: `untrack myapp`"
        )
        return
    name = parts[1].strip()
    from .project_tracker import remove_project
    result = remove_project(name)
    await message.reply(result)


async def handle_projects(message: Any) -> None:
    """List all tracked projects."""
    from .project_tracker import list_projects
    result = list_projects()
    await message.reply(result)


async def handle_project_detail(message: Any, content: str) -> None:
    """Show detail for a single project: project <name>."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply(
            "**Project Detail**\n"
            "Usage: `project <name>`\n"
            "Example: `project technomancer`"
        )
        return
    name = parts[1].strip()
    from .project_tracker import get_project
    result = get_project(name)
    await message.reply(result)


async def handle_blocker(message: Any, content: str, user: str) -> None:
    """Add a blocker to a project: blocker <name> <text>."""
    if user.lower() != settings.bot_owner.lower():
        await message.reply("Sorry, only the bot owner can manage projects.")
        return
    parts = content.split(maxsplit=2)
    if len(parts) < 3 or not parts[2].strip():
        await message.reply(
            "**Add a Blocker**\n"
            "Usage: `blocker <project> <description>`\n"
            "Example: `blocker myapp waiting on API key from vendor`"
        )
        return
    name = parts[1].strip()
    blocker_text = parts[2].strip()
    from .project_tracker import add_blocker
    result = add_blocker(name, blocker_text)
    await message.reply(result)
