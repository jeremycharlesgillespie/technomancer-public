"""
Slash Commands — Discord application commands with autocomplete.

Registers slash commands on the bot's CommandTree so users get
autocomplete suggestions as they type. Each command delegates to
the existing handler functions in bot_commands.py.

Usage:
    from .slash_commands import setup_slash_commands
    await setup_slash_commands(client)  # call in on_ready
"""

from __future__ import annotations

import logging
from typing import Any

import discord
from discord import app_commands

logger = logging.getLogger(__name__)


def _build_tree(client: discord.Client) -> app_commands.CommandTree:
    """Build the CommandTree with all slash commands."""
    tree = app_commands.CommandTree(client)

    # ------------------------------------------------------------------
    # Helper: send a long response, splitting if needed
    # ------------------------------------------------------------------
    async def _respond(interaction: discord.Interaction, text: str) -> None:
        """Send a response, chunking if over Discord's 2000-char limit."""
        if len(text) <= 2000:
            await interaction.response.send_message(text)
        else:
            await interaction.response.send_message(text[:2000])
            # Send remaining chunks as follow-ups
            remaining = text[2000:]
            while remaining:
                chunk = remaining[:2000]
                remaining = remaining[2000:]
                await interaction.followup.send(chunk)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tree.command(name="commands", description="Show all available bot commands")
    async def cmd_commands(interaction: discord.Interaction) -> None:
        from .bot_commands import handle_show_commands
        # Build the help text directly since the handler expects a message object
        help_text = _get_help_text()
        await _respond(interaction, help_text)

    @tree.command(name="ideas", description="Show active ideas from the idea board")
    async def cmd_ideas(interaction: discord.Interaction) -> None:
        from board import get_provider
        result = get_provider().list_ideas_for_llm()
        await _respond(interaction, result)

    @tree.command(name="think", description="Show what the bot knows about you (permanent memories)")
    async def cmd_think(interaction: discord.Interaction) -> None:
        from .memory_system import get_memory_system
        mem = get_memory_system()
        if mem:
            result = mem.get_context("permanent")
            await _respond(interaction, result or "No permanent memories saved yet.")
        else:
            await interaction.response.send_message("Memory system not initialized.")

    @tree.command(name="technews", description="Get latest tech news with AI analysis")
    async def cmd_technews(interaction: discord.Interaction) -> None:
        await interaction.response.defer()  # News takes time
        from .news_digest import handle_technews_command
        # We need the agent for news — import the module-level reference
        from . import discord_memory_bot as _bot
        if _bot.agent:
            result = await handle_technews_command(_bot.agent)
            await interaction.followup.send(result[:2000])
        else:
            await interaction.followup.send("Bot agent not ready yet.")

    @tree.command(name="perf", description="Show current session profiling data")
    async def cmd_perf(interaction: discord.Interaction) -> None:
        from .perf_monitor import get_endpoint_summary
        result = get_endpoint_summary()
        await _respond(interaction, result)

    @tree.command(name="metrics", description="Show persistent LLM latency trends")
    async def cmd_metrics(interaction: discord.Interaction) -> None:
        from .metrics_db import get_summary
        result = get_summary()
        await _respond(interaction, result)

    @tree.command(name="newsletter", description="Get this week's learning digest")
    async def cmd_newsletter(interaction: discord.Interaction) -> None:
        from .learning_newsletter import handle_newsletter_command
        result = handle_newsletter_command()
        await _respond(interaction, result)

    @tree.command(name="learninghistory", description="List past developer learning articles")
    async def cmd_learning_history(interaction: discord.Interaction) -> None:
        from .dev_learning import handle_learning_history_command
        result = handle_learning_history_command()
        await _respond(interaction, result)

    betterdev_categories = [
        app_commands.Choice(name="Random topic", value=""),
        app_commands.Choice(name="Python", value="python"),
        app_commands.Choice(name="Oracle", value="oracle"),
        app_commands.Choice(name="System Design", value="system_design"),
        app_commands.Choice(name="Best Practices", value="best_practices"),
    ]

    @tree.command(name="betterdev", description="Generate a developer learning article")
    @app_commands.describe(category="Topic category (or random)")
    @app_commands.choices(category=betterdev_categories)
    async def cmd_betterdev(
        interaction: discord.Interaction,
        category: str = "",
    ) -> None:
        await interaction.response.defer()  # Article generation takes time
        from .dev_learning import handle_better_dev_command
        result = await handle_better_dev_command(category if category else None)
        # Returns tuple of (article_text, github_pages_url)
        text, url = result if isinstance(result, tuple) else (result, None)
        response = text[:1900] if text else "Could not generate article."
        if url:
            response += f"\n\n[Read full article]({url})"
        await interaction.followup.send(response)

    @tree.command(name="suggestlearning", description="Suggest learning topics based on your recent conversations")
    async def cmd_suggest_learning(interaction: discord.Interaction) -> None:
        from .dev_learning import handle_suggest_learning_command
        result = handle_suggest_learning_command()
        await _respond(interaction, result)

    @tree.command(name="karen", description="Submit a complaint to K.A.R.E.N. (generates improvement ideas)")
    @app_commands.describe(complaint="What's bothering you about the bot?")
    async def cmd_karen(
        interaction: discord.Interaction,
        complaint: str,
    ) -> None:
        await interaction.response.defer()
        from idea_board.karen import add_complaint
        result = add_complaint(complaint, author="discord")
        await interaction.followup.send(
            f"Complaint filed: **{result.id}** — {result.text[:100]}\n"
            f"K.A.R.E.N. will analyze this and may generate improvement ideas."
        )

    @tree.command(name="suggest", description="Get context-aware command suggestions")
    async def cmd_suggest(interaction: discord.Interaction) -> None:
        from .conversation_context import get_recent_summaries
        from .command_suggestions import suggest_commands_for_context, format_context_suggestions
        recent = get_recent_summaries(count=5)
        suggestions = suggest_commands_for_context(recent)
        if suggestions:
            await _respond(interaction, format_context_suggestions(suggestions))
        else:
            await interaction.response.send_message(
                "No contextual suggestions right now. Use `/commands` to see all available commands."
            )

    return tree


def _get_help_text() -> str:
    """Return the commands help text (same as showCommands)."""
    return """**Available Commands**

**Learning & Development**
`/betterdev [category]` - Generate a learning article
`/learninghistory` - List past articles
`/newsletter` - Weekly learning digest

**News**
`/technews` - Latest tech news with analysis

**Memory**
`/think` - Show what I know about you

**Idea Board**
`/ideas` - Show active ideas (or visit the web dashboard)

**Feedback**
`/karen <complaint>` - Submit feedback to K.A.R.E.N.

**Performance**
`/perf` - Session profiling data
`/metrics` - Persistent LLM latency trends

**Other**
`/suggest` - Context-aware command suggestions
`/commands` - This help message

*Tip: Start typing `/` to see all commands with descriptions.*
*Text commands (betterDev, techNews, etc.) also still work.*"""


async def setup_slash_commands(client: discord.Client) -> None:
    """Build the command tree and sync it with Discord.

    Call this in on_ready after the client is connected.
    """
    tree = _build_tree(client)
    client.tree = tree  # type: ignore[attr-defined]

    try:
        synced = await tree.sync()
        logger.info(f"Synced {len(synced)} slash commands with Discord")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}")
