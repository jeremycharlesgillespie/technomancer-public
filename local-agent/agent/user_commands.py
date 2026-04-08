"""
User Commands — Personal/custom Discord command handlers.

These are user-specific tools that may not be relevant for all deployments.
They are separated from bot_commands.py (core commands) so they can be
easily customized, removed, or replaced per user.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

import discord

from .bot_utils import log
from .itinerary import generate_itinerary


async def handle_itinerary(
    message: Any, content: str, user: str, memory: Any, send_response: Any
) -> None:
    """Route itinerary/travel requests to Claude API for fast generation."""
    parts = content.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply(
            "Usage: `itinerary <your request>`\n"
            "Example: `itinerary Japan April 4-23, cherry blossoms, food, anime, temples`"
        )
        return

    request = parts[1]
    await message.reply("Generating itinerary via Claude API... this should be fast.")
    async with message.channel.typing():
        markdown_response, html_content, duration = await generate_itinerary(request)

        if html_content:
            from .github_pages import deploy_html_to_pages
            from .html_generator import slugify

            filename = f"itinerary-{slugify(request[:40])}-{datetime.now().strftime('%Y%m%d')}.html"
            url = await asyncio.to_thread(deploy_html_to_pages, html_content, filename)

            if url:
                await message.reply(
                    f"Done in **{duration:.1f}s** (vs ~8min on local LLM)\n\n"
                    f"View your itinerary: {url}"
                )
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
                    f.write(html_content)
                    temp_path = f.name
                await message.reply(
                    f"Done in **{duration:.1f}s** (GitHub Pages deploy failed, file attached)",
                    file=discord.File(temp_path, filename=filename),
                )
                os.unlink(temp_path)
        else:
            await send_response(message, markdown_response)

        if memory:
            memory.log_conversation(user, content, f"[Generated itinerary in {duration:.1f}s via Claude API]")
