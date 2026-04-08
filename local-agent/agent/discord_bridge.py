"""
Discord Bridge API — REST server for Claude Code ↔ Discord communication.

Runs inside the Discord bot process as an aiohttp web server on port 8321.
Gives Claude Code (and any local tool) the ability to:

  - Send messages to Discord channels
  - Read recent message history
  - Ask a question in Discord and WAIT for the user's reply
  - Get bot status and channel info

This is the bidirectional bridge that lets Claude Code interact with the
user through Discord, even when running headlessly via `claude -p`.

Architecture:
    Claude Code (subprocess)
        ↓ HTTP POST/GET
    Bridge API (port 8321, same process as Discord bot)
        ↓ discord.py
    Discord Channel
        ↓ user reads/replies
    Bridge API (captures reply via event listener)
        ↓ HTTP response
    Claude Code (gets the user's answer)

Security:
    - Listens on 127.0.0.1 only (localhost, not exposed to network)
    - Requires X-Bridge-Token header matching a configured secret
    - Only the bot owner's replies are captured for ask/wait

Usage from Claude Code or any local script:
    curl http://localhost:8321/api/send -X POST \\
        -H "Content-Type: application/json" \\
        -H "X-Bridge-Token: <token>" \\
        -d '{"message": "Hello from Claude Code!"}'

Endpoints:
    POST /api/send          — Send a message to the Discord channel
    POST /api/reply         — Reply to a specific message by ID
    GET  /api/history       — Get recent message history
    POST /api/ask           — Send a question and wait for user's reply
    GET  /api/status        — Get bot/bridge status
    GET  /api/health        — Simple health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import discord
from aiohttp import web

from .message_validators import validate_discord_message

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

#: Port the bridge API listens on. Only accessible from localhost.
BRIDGE_PORT: int = 8321

#: Maximum seconds to wait for a user reply when using /api/ask
ASK_TIMEOUT: int = 300  # 5 minutes

#: Maximum number of messages to return from /api/history
MAX_HISTORY: int = 50

#: Path to the bridge token file (auto-generated on first run)
TOKEN_FILE: Path = Path(__file__).parent.parent / ".bridge_token"


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class PendingQuestion:
    """Tracks a question sent via /api/ask that's waiting for a reply.

    Attributes:
        question_id: Unique identifier for this question
        message_id: Discord message ID of the sent question
        channel_id: Discord channel ID where the question was sent
        asked_at: Timestamp when the question was sent
        event: asyncio.Event that gets set when a reply arrives
        reply: The user's reply text (populated when event is set)
    """

    question_id: str
    message_id: int
    channel_id: int
    asked_at: float = field(default_factory=time.time)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reply: str | None = None
    reply_author: str | None = None


# ============================================================================
# BRIDGE SERVER
# ============================================================================

class DiscordBridgeAPI:
    """REST API server that bridges Claude Code to Discord.

    This runs inside the Discord bot process and has direct access to the
    discord.py Client object. It starts as an aiohttp web server on a
    background task when the bot connects.

    Attributes:
        client: The discord.py Client instance
        channel_name: Name of the Discord channel to interact with
        token: Authentication token required in X-Bridge-Token header
        pending_questions: Map of channel_id -> PendingQuestion for /api/ask
        app: The aiohttp web Application
        runner: The aiohttp AppRunner (for cleanup)
    """

    def __init__(self, client: discord.Client, channel_name: str) -> None:
        """Initialize the bridge API.

        Args:
            client: The discord.py Client instance (must be connected)
            channel_name: Name of the channel to send messages to
        """
        self.client: discord.Client = client
        self.channel_name: str = channel_name
        self.token: str = self._load_or_create_token()
        self.pending_questions: dict[int, PendingQuestion] = {}
        self.app: web.Application = self._create_app()
        self.runner: web.AppRunner | None = None
        self._start_time: float = time.time()

    # ========================================================================
    # TOKEN MANAGEMENT
    # ========================================================================

    def _load_or_create_token(self) -> str:
        """Load the bridge token from disk, or create one if it doesn't exist.

        The token is a 32-character hex string stored in .bridge_token.
        Claude Code reads this file to authenticate its requests.

        Returns:
            The bridge authentication token
        """
        if TOKEN_FILE.exists():
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token:
                return token

        token = secrets.token_hex(16)
        TOKEN_FILE.write_text(token, encoding="utf-8")
        logger.info(f"Generated new bridge token at {TOKEN_FILE}")
        return token

    # ========================================================================
    # CHANNEL LOOKUP
    # ========================================================================

    def _find_channel(self) -> discord.TextChannel | None:
        """Find the configured Discord text channel.

        Searches all guilds the bot is connected to for a channel matching
        self.channel_name.

        Returns:
            The TextChannel object, or None if not found
        """
        for guild in self.client.guilds:
            for channel in guild.text_channels:
                if channel.name == self.channel_name:
                    return channel
        return None

    # ========================================================================
    # AUTH MIDDLEWARE
    # ========================================================================

    def _check_auth(self, request: web.Request) -> bool:
        """Verify the X-Bridge-Token header matches our token.

        Args:
            request: The incoming HTTP request

        Returns:
            True if authenticated, False otherwise
        """
        provided = request.headers.get("X-Bridge-Token", "")
        return secrets.compare_digest(provided, self.token)

    # ========================================================================
    # APP SETUP
    # ========================================================================

    def _create_app(self) -> web.Application:
        """Create the aiohttp web application with all routes.

        Returns:
            Configured aiohttp Application
        """
        app = web.Application()
        app.router.add_get("/api/health", self._handle_health)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/history", self._handle_history)
        app.router.add_post("/api/send", self._handle_send)
        app.router.add_post("/api/reply", self._handle_reply)
        app.router.add_post("/api/ask", self._handle_ask)
        return app

    # ========================================================================
    # ROUTE HANDLERS
    # ========================================================================

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /api/health — Simple health check. No auth required.

        Returns:
            200 with {"status": "ok", "uptime": <seconds>}
        """
        return web.json_response({
            "status": "ok",
            "uptime": round(time.time() - self._start_time, 1),
        })

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /api/status — Bot and bridge status. Requires auth.

        Returns:
            200 with bot status, channel info, pending questions count
            401 if auth fails
        """
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        channel = self._find_channel()
        return web.json_response({
            "bot_connected": self.client.is_ready(),
            "bot_user": str(self.client.user) if self.client.user else None,
            "channel_found": channel is not None,
            "channel_name": self.channel_name,
            "channel_id": channel.id if channel else None,
            "pending_questions": len(self.pending_questions),
            "uptime": round(time.time() - self._start_time, 1),
            "bridge_port": BRIDGE_PORT,
        })

    async def _handle_history(self, request: web.Request) -> web.Response:
        """GET /api/history — Get recent message history. Requires auth.

        Query params:
            limit: Number of messages to return (default 20, max MAX_HISTORY)

        Returns:
            200 with list of {id, author, content, timestamp, is_bot}
            401 if auth fails
            404 if channel not found
        """
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        channel = self._find_channel()
        if not channel:
            return web.json_response(
                {"error": f"Channel '{self.channel_name}' not found"}, status=404
            )

        limit = min(int(request.query.get("limit", "20")), MAX_HISTORY)

        messages: list[dict[str, Any]] = []
        async for msg in channel.history(limit=limit):
            messages.append({
                "id": str(msg.id),
                "author": str(msg.author.name),
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "is_bot": msg.author.bot,
                "attachments": [a.filename for a in msg.attachments],
            })

        # Return in chronological order (oldest first)
        messages.reverse()

        return web.json_response({"messages": messages, "count": len(messages)})

    async def _handle_send(self, request: web.Request) -> web.Response:
        """POST /api/send — Send a message to the Discord channel. Requires auth.

        JSON body:
            message: str — The message text to send (required)

        Returns:
            200 with {message_id, channel, timestamp}
            400 if message is missing
            401 if auth fails
            404 if channel not found
        """
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        message_text = validate_discord_message(data.get("message", ""))
        if not message_text:
            return web.json_response({"error": "Missing or empty 'message' field"}, status=400)

        channel = self._find_channel()
        if not channel:
            return web.json_response(
                {"error": f"Channel '{self.channel_name}' not found"}, status=404
            )

        sent = await channel.send(message_text)
        logger.info(f"[Bridge] Sent message: {message_text[:80]}...")

        return web.json_response({
            "message_id": str(sent.id),
            "channel": self.channel_name,
            "timestamp": sent.created_at.isoformat(),
        })

    async def _handle_reply(self, request: web.Request) -> web.Response:
        """POST /api/reply — Reply to a specific Discord message. Requires auth.

        JSON body:
            message_id: str — The ID of the message to reply to (required)
            message: str    — The reply text (required)

        Returns:
            200 with {reply_id, in_reply_to, timestamp}
            400 if fields are missing
            401 if auth fails
            404 if channel or original message not found
        """
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        message_id = data.get("message_id", "").strip()
        reply_text = validate_discord_message(data.get("message", ""))
        if not message_id or not reply_text:
            return web.json_response(
                {"error": "Missing 'message_id' and/or 'message' fields"}, status=400
            )

        channel = self._find_channel()
        if not channel:
            return web.json_response(
                {"error": f"Channel '{self.channel_name}' not found"}, status=404
            )

        try:
            original = await channel.fetch_message(int(message_id))
        except (discord.NotFound, ValueError):
            return web.json_response(
                {"error": f"Message {message_id} not found"}, status=404
            )

        reply = await original.reply(reply_text)
        logger.info(f"[Bridge] Replied to {message_id}: {reply_text[:80]}...")

        return web.json_response({
            "reply_id": str(reply.id),
            "in_reply_to": message_id,
            "timestamp": reply.created_at.isoformat(),
        })

    async def _handle_ask(self, request: web.Request) -> web.Response:
        """POST /api/ask — Send a question and WAIT for the user's reply. Requires auth.

        This is the key endpoint for interactive Claude Code workflows.
        It sends a message to Discord, then blocks until the user replies
        (or the timeout expires).

        JSON body:
            question: str     — The question to ask (required)
            timeout: int      — Seconds to wait for reply (default ASK_TIMEOUT)

        Returns:
            200 with {question_id, reply, reply_author, wait_seconds}
            400 if question is missing
            401 if auth fails
            404 if channel not found
            408 if the user didn't reply within the timeout
        """
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        question_text = validate_discord_message(data.get("question", ""))
        if not question_text:
            return web.json_response({"error": "Missing or empty 'question' field"}, status=400)

        timeout = min(int(data.get("timeout", ASK_TIMEOUT)), ASK_TIMEOUT)

        channel = self._find_channel()
        if not channel:
            return web.json_response(
                {"error": f"Channel '{self.channel_name}' not found"}, status=404
            )

        # Send the question with a visual indicator that we're waiting
        sent = await channel.send(
            f"**[Claude Code is asking]:**\n{question_text}\n\n"
            f"*Reply to this message within {timeout // 60} minutes.*"
        )
        logger.info(f"[Bridge] Asked: {question_text[:80]}... (waiting {timeout}s)")

        # Register the pending question so on_message can capture the reply
        pending = PendingQuestion(
            question_id=secrets.token_hex(8),
            message_id=sent.id,
            channel_id=channel.id,
        )
        self.pending_questions[channel.id] = pending

        # Wait for the user to reply
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_questions.pop(channel.id, None)
            logger.info(f"[Bridge] Ask timed out after {timeout}s")
            return web.json_response(
                {"error": "Timed out waiting for reply", "timeout": timeout}, status=408
            )

        # Got a reply
        self.pending_questions.pop(channel.id, None)
        wait_seconds = round(time.time() - pending.asked_at, 1)
        logger.info(
            f"[Bridge] Got reply from {pending.reply_author} after {wait_seconds}s: "
            f"{(pending.reply or '')[:80]}..."
        )

        return web.json_response({
            "question_id": pending.question_id,
            "reply": pending.reply,
            "reply_author": pending.reply_author,
            "wait_seconds": wait_seconds,
        })

    # ========================================================================
    # MESSAGE LISTENER (called from on_message in the bot)
    # ========================================================================

    def check_for_reply(self, message: discord.Message) -> bool:
        """Check if an incoming Discord message is a reply to a pending question.

        This should be called from the bot's on_message handler for every
        non-bot message. If the message is a reply to a pending /api/ask
        question, it captures the reply and signals the waiting coroutine.

        Args:
            message: The incoming Discord message

        Returns:
            True if the message was captured as a reply (caller should skip
            normal processing), False otherwise
        """
        # Must be in a channel with a pending question
        pending = self.pending_questions.get(message.channel.id)
        if not pending:
            return False

        # Must be a reply to the question message, OR the next message in the channel
        is_direct_reply = (
            message.reference is not None
            and message.reference.message_id == pending.message_id
        )
        is_next_message = not message.author.bot

        if is_direct_reply or is_next_message:
            pending.reply = message.content
            pending.reply_author = str(message.author.name)
            pending.event.set()
            return True

        return False

    # ========================================================================
    # SERVER LIFECYCLE
    # ========================================================================

    async def start(self) -> None:
        """Start the bridge API server on localhost:BRIDGE_PORT.

        This is called from the bot's on_ready handler. The server runs
        as a background task in the same event loop as the Discord bot.
        """
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", BRIDGE_PORT)
        await site.start()
        logger.info(
            f"[Bridge] API running on http://127.0.0.1:{BRIDGE_PORT} "
            f"(token in {TOKEN_FILE})"
        )

    async def stop(self) -> None:
        """Stop the bridge API server. Called on bot shutdown."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("[Bridge] API stopped")


# ============================================================================
# MODULE-LEVEL INSTANCE (set by start_bridge, used by on_message check)
# ============================================================================

_bridge: DiscordBridgeAPI | None = None


def start_bridge(client: discord.Client, channel_name: str) -> DiscordBridgeAPI:
    """Create and start the Discord Bridge API.

    Call this from the bot's on_ready handler. It creates the bridge
    instance and schedules the server start as an asyncio task.

    Args:
        client: The connected discord.py Client
        channel_name: Name of the channel to bridge

    Returns:
        The DiscordBridgeAPI instance (also stored in module-level _bridge)
    """
    global _bridge
    _bridge = DiscordBridgeAPI(client, channel_name)
    asyncio.create_task(_bridge.start())
    return _bridge


def get_bridge() -> DiscordBridgeAPI | None:
    """Get the running bridge instance, or None if not started.

    Returns:
        The DiscordBridgeAPI instance, or None
    """
    return _bridge
