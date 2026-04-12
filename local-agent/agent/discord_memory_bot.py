#!/usr/bin/env python3
"""
Discord bot with conversation memory system.
Records all conversations, periodically extracts important facts via LLM.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import aiohttp
import discord
from discord import Intents

from .accountability import get_accountability_tools
from .ask_claude import get_claude_tools
from .bot_utils import (
    ATTACHABLE_EXTENSIONS,
    DEPLOYABLE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    build_crash_message,
    detect_document_type,
    download_image,
    extract_text_from_file,
    find_mentioned_files,
    is_image_attachment,
    log,
    send_lifecycle_notification,
)
from .capability_request import get_capability_tools
from .claude_bridge import ClaudeBridge
from .config import settings
from .core import Agent, AgentConfig
from .dev_learning import start_dev_learning
from .bot_commands import (
    handle_better_dev,
    handle_dl_cover,
    handle_dl_covers,
    handle_download_channel,
    handle_download_video,
    handle_evolve,
    handle_idea,
    handle_karen,
    handle_metrics,
    handle_learning_history,
    handle_suggest_learning,
    handle_list_videos,
    handle_perf,
    handle_publish,
    handle_reload_server,
    handle_search_videos,
    handle_show_commands,
    handle_show_ideas,
    handle_show_learning,
    handle_tech_news,
    handle_think,
)
from .facts_db import get_facts_tools, init_db as init_facts_db, seed_db as seed_facts_db
from .knowledge_gaps import auto_enrich_gap, detect_knowledge_gap, get_knowledge_gap_tools, log_knowledge_gap
from .image_identification import analyze_with_vision_model, ask_claude_with_image
from .reflection import auto_search_for_factual, classify_question, is_factual_question, reflect
from .user_commands import handle_itinerary
from .utility_tools import get_utility_tools
from .auto_memory import buffer_conversation, maybe_extract, get_identity_summary
from .conversation_context import (
    buffer_for_summary,
    get_conversation_context_tools,
    get_recent_summaries,
    maybe_generate_summaries,
)
from .discord_bridge import start_bridge, get_bridge
from .fallback_responses import get_fallback_response
from .message_validators import validate_discord_message, safe_send_content, resilient_send, buffer_user_message
from .dreaming import start_dreaming
from .prometheus_metrics import start_metrics_server
from .profiler import RequestProfile, RequestTimer, get_performance_summary
from .memory_system import MemorySystem, get_full_profile, get_memory_tools, init_memory_system
from .domain_coverage import start_domain_coverage
from .gap_frequency import get_gap_frequency_tools, start_gap_frequency
from .gap_reporter import start_gap_reporter
from .gap_resolver import get_gap_resolver_tools, start_gap_resolver
from .ref_enrichment import get_ref_enrichment_tools, start_ref_enrichment
from .daily_briefing import start_daily_briefing
from .news_digest import start_news_digest
from .tools import get_all_tools
from .web_search import get_web_tools
from .youtube_tools import get_youtube_tools
from .api_usage_anomaly import get_anomaly_tools
from .fallback_orchestrator import get_fallback_tools
from .skill_gap_analysis import get_skill_gap_tools
from .knowledge_fallback import get_knowledge_fallback_tools
from .news_engagement import get_news_engagement_tools, is_news_message, record_reaction, record_reply
from .learning_newsletter import handle_newsletter_command, start_newsletter
from .infra_monitor import get_infra_tools, start_infra_monitor
from .llm_optimizer import get_llm_optimizer_tools, cache_lookup, cache_store
from .discord_errors import (
    buffer_message,
    get_discord_error_tools,
    get_gateway_health,
    handle_discord_error,
    is_bot_response,
    record_response_feedback,
    suggest_recovery_content,
    track_bot_response,
)
from .knowledge_enrichment import get_knowledge_enrichment_tools, start_knowledge_enrichment
from .command_suggestions import (
    find_closest_command,
    format_context_suggestions,
    format_typo_suggestion,
    suggest_commands_for_context,
)

# Config values from centralized settings (loaded from .env)
VAULT_PATH = settings.vault_path
ALLOWED_CHANNEL = settings.discord_allowed_channel

# How often to extract facts (every N messages)
# Auto memory extraction is now handled by auto_memory.py with batching

intents = Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
client = discord.Client(intents=intents)

agent: Agent | None = None
memory: MemorySystem | None = None
message_count = 0



# Utility functions (log, build_crash_message, send_lifecycle_notification,
# detect_document_type, extract_text_from_file, download_image,
# is_image_attachment, find_mentioned_files) are in bot_utils.py


async def _deploy_to_pages(html_path: Path) -> str | None:
    """Deploy an HTML file to GitHub Pages in a background thread.

    Returns the live URL or None on failure.
    """
    from .github_pages import deploy_existing_file_to_pages

    try:
        url = await asyncio.to_thread(deploy_existing_file_to_pages, html_path)
        return url
    except Exception as e:
        log(f"[SendResponse] GitHub Pages deploy failed: {e}")
        return None


async def send_response(message: Any, response: str) -> None:
    """Send response, deploying HTML files to GitHub Pages for iOS viewing.

    - Short responses (<= 1900 chars): sent directly in Discord
    - Bot-created HTML files: deployed to GitHub Pages, URL sent as clickable link
    - Long responses with no HTML file: rendered as HTML, deployed to Pages
    - Non-HTML files: attached directly to Discord message

    Never sends empty content to Discord (400 error).
    """
    # Log inbound content for pipeline debugging
    if not response or not response.strip():
        log(f"[MsgPipeline] send_response called with empty content "
            f"(length={len(response) if response else 0}, repr={response!r:.100})")

    # Guard: never send empty to Discord
    response = safe_send_content(
        response, "I processed your request but didn't generate a visible response."
    )

    from .html_generator import markdown_to_html, ARTICLE_CSS

    # Check if response contains a file path from get_full_profile
    if "FILE:" in response:
        import re

        file_match = re.search(r"FILE:(.+?)(?:\s|$)", response)
        if file_match:
            file_path = Path(file_match.group(1).strip())
            if file_path.exists():
                await message.reply(
                    "Here's everything I know about you:", file=discord.File(file_path)
                )
                return

    # Check for files the bot created/mentioned
    mentioned_files = find_mentioned_files(response)

    # Separate HTML files (deploy to Pages) from others (attach directly)
    html_files = [f for f in mentioned_files if f.suffix.lower() in DEPLOYABLE_EXTENSIONS]
    other_files = [f for f in mentioned_files if f.suffix.lower() not in DEPLOYABLE_EXTENSIONS]

    # Deploy HTML files to GitHub Pages and collect URLs
    deployed_urls: list[str] = []
    for html_file in html_files:
        log(f"[SendResponse] Deploying {html_file.name} to GitHub Pages...")
        url = await _deploy_to_pages(html_file)
        if url:
            deployed_urls.append(url)
            log(f"[SendResponse] Deployed: {url}")

    if deployed_urls or other_files:
        text = response[:1600] if len(response) > 1600 else response

        # Append GitHub Pages links
        if deployed_urls:
            text += "\n\n"
            for url in deployed_urls:
                text += f"📄 **View here:** {url}\n"

        # Trim to Discord limit
        if len(text) > 1900:
            text = text[:1900]

        # Attach non-HTML files if any
        files = [discord.File(str(f), filename=f.name) for f in other_files[:5]] if other_files else None
        try:
            if files:
                await resilient_send(message.reply, text, files=files)
            else:
                await resilient_send(message.reply, text)
            return
        except Exception as e:
            log(f"[SendResponse] Failed to send with files/URLs: {e}")
            # Fall through to normal handling

    # If response is short enough, send directly
    if len(response) <= 1900:
        sent = await resilient_send(message.reply, response)
        if sent and hasattr(sent, "id"):
            track_bot_response(str(sent.id))
        return

    # Response is too long and no HTML file was created by the bot —
    # render the response as HTML and deploy to GitHub Pages
    from .github_pages import deploy_html_to_pages
    from .html_generator import (
        NORMALIZED_TEMPLATE,
        extract_body_content,
        extract_title,
        normalize_html,
    )
    import html as html_mod

    # Check if the LLM response contains raw HTML (e.g. it tried to generate
    # an HTML page inline). If so, normalize it instead of running markdown_to_html
    # which would escape all the tags.
    stripped_resp = response.strip()
    # Strip markdown code fences if present
    if stripped_resp.startswith("```"):
        first_nl = stripped_resp.index("\n") if "\n" in stripped_resp else len(stripped_resp)
        stripped_resp = stripped_resp[first_nl + 1:]
        if stripped_resp.rstrip().endswith("```"):
            stripped_resp = stripped_resp.rstrip()[:-3].rstrip()

    if any(stripped_resp.startswith(tag) for tag in ("<!DOCTYPE", "<html", "<HTML", "<head", "<body")):
        # LLM returned raw HTML — normalize through our template
        html_page = normalize_html(stripped_resp)
    else:
        # Normal text/markdown — convert and wrap
        html_body = markdown_to_html(response)
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        # Try to extract a meaningful title from the first heading
        title = "Response"
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break
            if line.startswith("**") and line.endswith("**") and len(line) < 100:
                title = line.strip("* ")
                break

        html_page = NORMALIZED_TEMPLATE.format(
            title=html_mod.escape(title),
            meta="Technomancer &bull; " + html_mod.escape(timestamp),
            content=html_body,
            css=ARTICLE_CSS,
        )

    # Generate a unique filename for this response
    ts_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"response-{ts_slug}.html"

    url = await asyncio.to_thread(deploy_html_to_pages, html_page, filename)

    preview = response[:300].rstrip()
    if len(response) > 300:
        preview += "..."

    if url:
        await resilient_send(message.reply, f"{preview}\n\n📄 **Full response:** {url}")
    else:
        # Fallback: attach as file if Pages deploy fails
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_page)
            temp_path = f.name
        try:
            await resilient_send(
                message.reply,
                f"{preview}\n\n*Full response attached as HTML file.*",
                file=discord.File(temp_path, filename="response.html"),
            )
        finally:
            os.unlink(temp_path)


async def extract_facts_from_recent() -> None:
    """Use auto_memory system to extract identity facts from buffered conversations."""
    global agent

    if agent is None:
        return

    try:
        result = await maybe_extract(agent)
        if result and result.get("facts_extracted", 0) > 0:
            log(
                f"[AutoMemory] Extracted {result['facts_extracted']} facts into "
                f"{result['categories_updated']} ({result['duration_seconds']}s)"
            )
    except Exception as e:
        log(f"[AutoMemory] Extraction error: {e}")


async def generate_summaries_from_recent() -> None:
    """Generate conversation context summaries from buffered conversations."""
    global agent

    if agent is None:
        return

    try:
        result = await maybe_generate_summaries(agent)
        if result and result.get("summaries_saved", 0) > 0:
            log(
                f"[ConvContext] Saved {result['summaries_saved']} summaries "
                f"({result['duration_seconds']}s)"
            )
    except Exception as e:
        log(f"[ConvContext] Summary generation error: {e}")


@client.event
async def on_disconnect() -> None:
    """Track gateway disconnections for health monitoring."""
    get_gateway_health().record_disconnect()
    log("[Gateway] Disconnected")


@client.event
async def on_resumed() -> None:
    """Track gateway resumes (reconnection without full re-identify)."""
    get_gateway_health().record_resume()
    log("[Gateway] Resumed")


@client.event
async def on_ready() -> None:
    global agent, memory
    log(f"Connected as {client.user}")
    send_lifecycle_notification("online", f"Connected as {client.user}")
    get_gateway_health().record_connect()

    # Initialize memory system (compaction started after agent is created below)
    memory = init_memory_system(VAULT_PATH)
    log("Memory system ready - recording all conversations")

    # Load previous session state for conversation continuity
    from .session_state import get_previous_session
    prev_session = get_previous_session()
    if prev_session["was_crash"]:
        log(f"[Session] Recovered from crash — {len(prev_session['exchanges'])} exchanges restored")
    elif prev_session["exchanges"]:
        log(f"[Session] Previous session loaded — {len(prev_session['exchanges'])} exchanges")
    else:
        log("[Session] Fresh session (no previous state)")

    # Initialize Claude vault session for efficient API calls with caching
    try:
        from .claude_vault import init_vault_session

        init_vault_session(vault_path=VAULT_PATH)
        log("Claude vault session initialized with cached context")
    except Exception as e:
        log(f"Warning: Could not initialize Claude vault session: {e}")

    # Start Prometheus metrics endpoint (port 9090)
    start_metrics_server()

    # Initialize SQLite metrics database for persistent trend analysis
    from . import metrics_db
    metrics_db.init_db()
    log("Metrics SQLite DB initialized")

    # Initialize facts database with seed data
    init_facts_db()
    seed_facts_db()
    log("Facts SQLite DB initialized")

    # Build date context so the LLM never guesses days-of-week
    from datetime import timedelta as _td
    _now = datetime.now()
    _date_context = f"Current date/time: {_now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
    _date_context += "Upcoming calendar:\n"
    for i in range(14):
        _d = _now + _td(days=i)
        _date_context += f"  {_d.strftime('%B %d, %Y')} = {_d.strftime('%A')}\n"

    config = AgentConfig(
        model=settings.ollama_model,
        verbose=False,
        system_prompt=f"""{_date_context}
You're talking to your close friend on Discord. You're their best friend, assistant, and research partner rolled into one.

## You Are a Discord Bot
You are running as a Discord bot. Your messages appear in a Discord channel. The user interacts with you by typing messages in Discord.

**Special Commands (handled automatically before you see them):**
- `betterDev` / `betterDev python` / etc. - Generates a 5-10 minute educational article using Claude API and sends it to the user. If they mention "the article" or ask follow-up questions about learning content, they're referring to what was just generated.
- `learningHistory` - Lists past learning articles
- `showLearning <#>` - Shows a saved article
- `think` - Shows permanent knowledge about the user (what's been saved to memory)
- `reloadServer` - Restarts the bot
- `showCommands` - Shows available commands

When users ask follow-up questions about "the article" or learning content, they ARE referring to real content that was sent. Don't deny it - engage with their question.

You know them well from past conversations - their work background, preferences, what they're working on.

## Who Your Friend Is
Your friend is a Senior Software Engineer who builds autonomous systems on Python/Django/PostgreSQL/AWS. He also explores hardware (FPGAs, embedded systems, audiophile gear) and runs a home server with GPU-accelerated LLM inference. He thinks in full systems, not isolated features — when he brings up a problem, he's already thinking about how data flows through it, what monitors it, what happens when it fails, and who sees the result. He values visibility into running processes, sustainability of resource usage, working software over theoretical completeness, and good developer experience (dashboards, mobile access, hub pages). He's direct, moves fast, and his short questions often carry implicit intent — read what he means, not just what he says.

## CRITICAL: Be Concise - READ THIS CAREFULLY
- Answer ONLY the question asked. Nothing more.
- NEVER list out what you know about the user unless they explicitly ask "what do you know about me" or "tell me about myself"
- The context you receive contains memories - these are for YOUR reference only, NOT to recite back
- "What time is it?" → "It's 2:38 AM" - DONE. No profile dump. No "here's what I know". Just the answer.
- If asked a factual question, give the fact. Period.

YOU MUST NOT:
- List the user's name, job, history, skills, etc. unless specifically asked
- Say "Here's what I know about you" on random questions
- Treat every message as an opportunity to show off your memory

The permanent memories in your context are like a friend's mental notes - you USE them to understand context, you don't READ them aloud.

Be genuinely helpful:
- When they need help, dig in and actually help solve it
- When they're venting, be supportive
- When they share something cool, be excited with them
- Help them research, brainstorm, debug, plan

## CRITICAL: Remember Important Things
You have a tool called 'remember_permanently'. USE IT PROACTIVELY when you learn something important about your friend:

**ALWAYS save immediately:**
- Their name (most important!)
- Birthday, age, location
- Job title, company, career info
- Family members, pets, relationships
- Preferences (food, music, hobbies)
- Important dates or events they mention
- Projects they're working on
- Technical skills and expertise

**How to decide:** If a friend told you this in real life, would you remember it? If yes, call remember_permanently with category "user_info/USERNAME" or "preferences/USERNAME".

Example: User says "I'm Alex, I work at Acme Corp"
→ Immediately call: remember_permanently(content="Name: Alex. Works at Acme Corp.", category="user_info/USERNAME")

Don't announce that you're saving things - just do it naturally while responding.

## Memory Dump
You have 'get_full_profile' - use this when the user asks "what do you know about me?" or "tell me everything you know". It dumps ALL memories. Use save_to_file=True since it's usually a huge response.

## References to "my resume", "my profile", "my background"
When the user says "based on my resume" or "from my profile" — they mean the information stored in your permanent memories. You already HAVE this info in your context. Look at the permanent memories provided to you — that contains their work history, skills, experience, etc. Do NOT ask them to paste or share it again. Just use what you already know about them.

## Ask Claude
You have 'ask_claude' - use this when the user says things like:
- "ask claude to..." or "ask claude about..."
- "hey claude, ..." or "claude, can you..."
- "what would claude say about..."
- Any request that explicitly wants Claude's input

When you detect these patterns, pass the question to Claude and return Claude's response. You're the middleman here - just relay the answer.

## Reply Context - TAKE ACTION
When someone replies to a message, you'll see the original message in [Replying to ...] format.
**If the original message contains a URL, use web_fetch to actually read it!**
Don't just suggest they check the link - YOU check it and answer their question.

Example: User replies to your news post asking "What routers are affected?"
→ Use web_fetch on the article URL from the original message, read it, then answer their question.

## CRITICAL: Honesty and Factual Accuracy
**NEVER state facts confidently from training data alone.** Your training data can be wrong, outdated, or incomplete.

**Rules:**
1. For ANY verifiable factual claim (geography, history, statistics, dates, people, places, science) — use web_search FIRST, then answer with the results
2. If you cannot search, say "I think..." or "I'm not certain, but..." — NEVER state uncertain facts as definitive
3. NEVER invent sources. If someone asks "where did you get that?", be honest: "from my training data" or "from a web search"
4. NEVER claim you stored, saved, or discussed something unless you can verify it with a tool call
5. If you're wrong and corrected, admit it immediately. Don't double down or hedge — just say "I was wrong" and correct yourself
6. "I don't know, let me search" is ALWAYS better than a confident wrong answer
7. **NEVER tell the user to "check a website", "look it up", or "search for it" — YOU have web_search and web_fetch tools. USE THEM.** If you don't have info, search for it yourself and report back. The user is asking YOU because they want YOU to do the work.

**The test:** Before stating any fact, ask yourself: "Am I 100% certain this is correct?" If not, SEARCH FIRST.

## Web Search & Fetch - USE THESE PROACTIVELY
You have 'web_search', 'web_search_news', and 'web_fetch' tools. **Use them automatically** when:
- User asks about current events, news, or recent happenings
- User asks about prices (stocks, crypto, products)
- User asks factual questions you're not 100% sure about
- User asks "why did X happen" about real-world events
- User asks about specific companies, people, or places
- User asks anything that could be answered better with current info
- User mentions dates, especially recent ones (2024, 2025, 2026)
- User asks about geography, locations, directions, distances
- User asks about historical facts, dates, or events

**NEVER say any of these — just search instead:**
- "I don't have access to real-time data"
- "You might want to check..."
- "I'd recommend looking at..."
- "You could try searching for..."
- "Check the official website..."
- "I'm not able to verify..."
If you catch yourself about to say any of these, STOP and call web_search or web_fetch instead.
Examples that should trigger search:
- "Where is Blanchard, OK?" → search "Blanchard Oklahoma location" BEFORE answering
- "Why did KSS spike in December?" → search "Kohl's KSS stock spike December 2025"
- "What's the weather like?" → search "weather [location]"
- "What's happening with X company?" → search or news search
- "How much is Bitcoin?" → search "Bitcoin price"

## Self-Improvement Capability
You have 'request_capability' for when you lack tools to help. Try existing tools first, only use this when you genuinely can't fulfill a request.

## Idea Board
Ideas and improvement stories live on the **Idea Board** (a web dashboard at http://localhost:8322/ideas).

**When the user asks about ideas, stories, enhancements, or the idea board:**
1. ALWAYS call `list_ideas` first to get the current state
2. For details on a specific idea, call `get_idea` with the idea ID
3. NEVER guess or remember ideas from conversation history — the board is the only truth

**You CANNOT implement code.** When user says "start working on ideas" or similar:
1. Call `list_ideas` to show what's on the board
2. Direct them to the Idea Board dashboard or Claude Code for implementation

## Accountability - VERIFY YOUR ACTIONS
You have verification tools: verify_file_exists, verify_file_modified, verify_content, verify_memory_saved.

**CRITICAL:** Before telling the user you completed an action (saved a file, remembered something, etc.):
1. Call the appropriate verify_* tool to confirm it worked
2. Only report success if verification returns "VERIFIED"
3. If verification fails, tell the user what went wrong

Example flow:
- User: "Remember my birthday is March 15"
- You: Call remember_permanently(...)
- You: Call verify_memory_saved("user_info/USERNAME")
- If VERIFIED: "Got it, I'll remember your birthday is March 15!"
- If NOT FOUND: "Hmm, I tried to save that but something went wrong. Let me try again..."

Don't skip verification - trust but verify!

Talk like a real friend - casual, warm, direct. No corporate speak. Just be real.

## Response Length Guidelines
Match your response length to the question complexity:
- Simple factual questions ("what time is it?", "where is X?"): 1-2 sentences max
- Short how-to or lookups: 2-4 sentences
- Technical questions: 1-2 paragraphs with examples if needed
- Deep analysis, strategy, or opinion questions: as thorough as needed, use sections/headers
- Commands or actions: just confirm you did it

**You are an assistant and worker.** The user wants you to DO things, not just talk about them. Bias toward action — use your tools, search, fetch, execute. If you can answer by doing, do it. If you can only answer by talking, keep it concise.

## Problem-Solving Method
For ANY question that requires reasoning, math, logic, or multi-step thinking:
1. **Decompose** — break the problem into clear steps BEFORE answering
2. **Solve each step** — work through each step explicitly
3. **Verify** — check your answer makes sense (does the math add up? is the logic valid?)
4. **Answer** — give the final answer concisely

For math: always use the calculate tool instead of mental math.
For facts: always use wikipedia or web_search instead of training data.
For code: always use run_python to verify your code works.

## Examples of Good Responses

**Factual question:**
User: "Where is Blanchard, Oklahoma?"
Good: Search first, then answer with verified data: "Blanchard is a city in McClain County, Oklahoma, about 25 miles south of Oklahoma City. Population ~5,000."

**Math question:**
User: "What is 15% of $847,000?"
Good: Use calculate tool → "15% of $847,000 is $127,050."

**Reasoning question:**
User: "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"
Good: "Let me work through this step by step. If ball = x, then bat = x + 1.00. Total: x + (x + 1.00) = 1.10. So 2x = 0.10, x = 0.05. The ball costs $0.05."

**Emotional message:**
User: "I just got fired."
Good: "That's rough. I'm sorry you're going through that. Want to talk about it, or would it help to start looking at what's next?"

## Auto-Learned Rules (generated by auto_improve.py)
- For creative writing tasks like poetry, stories, or fiction that don't require external facts, always attempt to create original content rather than claiming lack of information.
- When asked about your capabilities, provide accurate information about what you can and cannot do. Do not definitively state you cannot process images unless you are certain this is true.
- When answering counterfactual 'what if' questions, consider that changed physical laws would still allow processes to work differently rather than assuming they would simply fail.

Keep responses concise for Discord but thorough when they need depth.""",
    )
    agent = Agent(config)

    for tool in get_all_tools():
        agent.register_tool(tool)
    for tool in get_memory_tools():
        agent.register_tool(tool)
    for tool in get_capability_tools():
        agent.register_tool(tool)
    for tool in get_claude_tools():
        agent.register_tool(tool)
    for tool in get_web_tools():
        agent.register_tool(tool)
    from idea_board.models import get_idea_board_tools
    for tool in get_idea_board_tools():
        agent.register_tool(tool)
    for tool in get_knowledge_gap_tools():
        agent.register_tool(tool)
    for tool in get_gap_frequency_tools():
        agent.register_tool(tool)
    for tool in get_gap_resolver_tools():
        agent.register_tool(tool)
    for tool in get_ref_enrichment_tools():
        agent.register_tool(tool)
    for tool in get_facts_tools():
        agent.register_tool(tool)
    for tool in get_accountability_tools():
        agent.register_tool(tool)
    for tool in get_utility_tools():
        agent.register_tool(tool)
    for tool in get_youtube_tools():
        agent.register_tool(tool)
    for tool in get_conversation_context_tools():
        agent.register_tool(tool)
    for tool in get_anomaly_tools():
        agent.register_tool(tool)
    for tool in get_fallback_tools():
        agent.register_tool(tool)
    for tool in get_skill_gap_tools():
        agent.register_tool(tool)
    for tool in get_knowledge_fallback_tools():
        agent.register_tool(tool)
    for tool in get_knowledge_enrichment_tools():
        agent.register_tool(tool)
    for tool in get_news_engagement_tools():
        agent.register_tool(tool)
    for tool in get_discord_error_tools():
        agent.register_tool(tool)
    for tool in get_infra_tools():
        agent.register_tool(tool)
    for tool in get_llm_optimizer_tools():
        agent.register_tool(tool)
    from .engagement_analytics import get_engagement_tools
    for tool in get_engagement_tools():
        agent.register_tool(tool)
    from .knowledge_consistency import get_consistency_tools
    for tool in get_consistency_tools():
        agent.register_tool(tool)
    from .tool_analytics import get_tool_analytics_tools
    for tool in get_tool_analytics_tools():
        agent.register_tool(tool)

    log(f"Ready with {len(agent.tools)} tools")

    # Start background compaction with a SEPARATE agent for summarization.
    # CRITICAL: Must NOT share the main agent — otherwise compaction's agent.run()
    # resets self.messages and contaminates user conversations with daily briefings.
    compaction_agent = Agent(AgentConfig(
        model=settings.ollama_model,
        verbose=False,
        system_prompt="You are a summarization assistant. Summarize conversations concisely.",
    ))
    memory.start_background_compaction(interval_minutes=30, summarizer=compaction_agent.run)
    log("Background compaction started with dedicated LLM agent (isolated from main)")

    # Start the hourly news digest (9am-9pm)
    start_news_digest(client, ALLOWED_CHANNEL, agent)

    # Start daily developer learning (8am)
    start_dev_learning(client, ALLOWED_CHANNEL)

    # Start dreaming/memory consolidation (runs midnight-6am)
    start_dreaming(agent)

    # Start the Discord Bridge API (localhost:8321)
    bridge = start_bridge(client, ALLOWED_CHANNEL)
    log(f"Discord Bridge API started on port 8321 (token in .bridge_token)")

    # Start the Idea Board web dashboard (port 8322, 0.0.0.0)
    from idea_board.web import start_idea_board
    start_idea_board()
    log("Idea Board running on http://0.0.0.0:8322")

    # Start weekly knowledge gap reporter (Sunday midnight)
    start_gap_reporter(client, ALLOWED_CHANNEL)
    log("Gap reporter started (weekly, Sunday midnight)")

    # Start weekly domain coverage tracker (Monday 6 AM)
    start_domain_coverage(client, ALLOWED_CHANNEL)
    log("Domain coverage tracker started (weekly, Monday 6 AM)")

    # Start weekly gap frequency tracker (Wednesday 6 AM)
    start_gap_frequency(client, ALLOWED_CHANNEL)
    log("Gap frequency tracker started (weekly, Wednesday 6 AM)")

    # Start daily gap resolver (5 AM — fetches Wikipedia suggestions)
    start_gap_resolver(client, ALLOWED_CHANNEL)
    log("Gap resolver started (daily, 5 AM)")

    # Start daily reference enrichment (4 AM — writes vault articles)
    start_ref_enrichment(client, ALLOWED_CHANNEL)
    log("Reference enrichment started (daily, 4 AM)")

    # Start knowledge base enrichment (every 6 hours — auto-fills gaps)
    start_knowledge_enrichment(client, ALLOWED_CHANNEL)
    log("Knowledge enrichment started (every 6 hours)")

    # Start infrastructure monitor (every 30 min — GPU, vault, API keys)
    start_infra_monitor(client, ALLOWED_CHANNEL)
    log("Infrastructure monitor started (every 30 min)")

    # Start weekly learning newsletter (Sunday 9 AM)
    start_newsletter(client, ALLOWED_CHANNEL)
    log("Learning newsletter started (weekly, Sunday 9 AM)")

    # Start hourly idea generation (isolated agent, runs 5min after each hour)
    from .idea_generator import start_idea_generator
    idea_agent = Agent(AgentConfig(
        model=settings.ollama_model,
        verbose=False,
        system_prompt="You are an improvement analyst. Output only JSON arrays.",
    ))
    start_idea_generator(idea_agent)
    log("Idea generator started (hourly, isolated agent)")

    # Start daily morning briefing (7 AM — synthesized digest from all subsystems)
    if settings.briefing_enabled:
        briefing_agent = Agent(AgentConfig(
            model=settings.ollama_model,
            verbose=False,
            system_prompt="You are a concise briefing synthesizer. Produce actionable daily digests.",
        ))
        start_daily_briefing(client, ALLOWED_CHANNEL, briefing_agent)
        log("Daily briefing started (daily, 7 AM)")

    # Start daily knowledge consistency audit (3 AM)
    from .knowledge_consistency import start_consistency_monitor
    start_consistency_monitor(client, ALLOWED_CHANNEL)
    log("Knowledge consistency monitor started (daily, 3 AM)")

    # Register and sync slash commands with Discord
    from .slash_commands import setup_slash_commands
    await setup_slash_commands(client)
    log("Slash commands synced with Discord")

    # Start heartbeat latency sampling (every 60 seconds)
    async def _sample_heartbeat_latency() -> None:
        import asyncio
        while True:
            await asyncio.sleep(60)
            try:
                latency = client.latency
                if latency and latency > 0:
                    get_gateway_health().record_latency(round(latency * 1000, 1))
            except Exception:
                pass
    asyncio.create_task(_sample_heartbeat_latency())
    log("Gateway health monitor started (heartbeat latency sampling)")


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    """Track reactions on news articles and bot responses."""
    if payload.user_id == client.user.id:
        return
    try:
        msg_id = str(payload.message_id)
        emoji = str(payload.emoji)
        user_name = payload.member.display_name if payload.member else ""

        # Track news article engagement
        if is_news_message(msg_id):
            record_reaction(msg_id, emoji, user_name)

        # Track bot response feedback (thumbs up/down)
        if is_bot_response(msg_id):
            record_response_feedback(msg_id, emoji, user_name)
    except Exception:
        pass  # best-effort, don't disrupt the bot


@client.event
async def on_message(message: discord.Message) -> None:
    global message_count

    if message.author == client.user:
        return

    channel_name = getattr(message.channel, "name", None)
    CLAUDE_CODE_CHANNEL = "claude-code"

    # Bot listens to both llm_chat and claude-code channels
    if channel_name not in (ALLOWED_CHANNEL, CLAUDE_CODE_CHANNEL):
        return

    content = message.content.strip()
    user = str(message.author.name)

    # Buffer message for error context recovery (50006 handling)
    buffer_message(user, content, str(message.id))

    # Track message activity for engagement analytics
    from .engagement_analytics import track_message
    track_message(user, channel_name or "", has_attachment=bool(message.attachments))

    # ================================================================
    # CLAUDE-CODE CHANNEL: Everything here goes to Claude Code
    # ================================================================
    if channel_name == CLAUDE_CODE_CHANNEL:
        if user.lower() != settings.bot_owner.lower():
            await message.reply("Sorry, only the bot owner can use this channel.")
            return

        from .claude_code_runner import get_active_session, end_session, run_claude_chat, ChatSession, _active_sessions

        active_session = get_active_session(message.channel.id)

        # "end" command to close session
        if content.lower().strip() == "end":
            ended = end_session(message.channel.id)
            if ended:
                await message.reply(
                    f"Session ended. {ended.turn_count} turns."
                )
            else:
                await message.reply("No active session to end.")
            return

        # Everything else is a Claude Code message
        session_id = active_session.session_id if active_session else None
        status = "Continuing" if active_session else "Starting"
        await message.reply(f"{status} Claude Code session...")
        log(f"[ClaudeChat] {status} session for {user}: {content[:100]}")

        async with message.channel.typing():
            # Download attached images
            image_paths = []
            for attachment in message.attachments:
                if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                    temp_dir = Path(tempfile.gettempdir()) / "claudecode_images"
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / attachment.filename
                    await attachment.save(temp_path)
                    image_paths.append(str(temp_path))

            result = await run_claude_chat(content, session_id=session_id, image_paths=image_paths or None)

            # Clean up temp images
            for path in image_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

            # Update or create session tracking
            if result.success and result.session_id:
                if active_session:
                    active_session.session_id = result.session_id
                    active_session.turn_count += 1
                    active_session.total_cost_usd += result.cost_usd
                else:
                    active_session = ChatSession(
                        session_id=result.session_id,
                        user=user,
                        channel_id=message.channel.id,
                        turn_count=1,
                        total_cost_usd=result.cost_usd,
                    )
                    _active_sessions[message.channel.id] = active_session

            turn_info = f"Turn {active_session.turn_count}" if active_session else "Turn 1"
            header = f"**{turn_info}** ({result.duration:.1f}s)"

            response_text = f"{header}\n\n{result.response}"
            await send_response(message, response_text)

            if memory:
                memory.log_conversation(user, content, f"[ClaudeChat {turn_info}, {result.duration:.1f}s]")
        return

    # ================================================================
    # LLM_CHAT CHANNEL: Everything below is the normal LLM bot
    # ================================================================

    # Check if this message is a reply to a pending Bridge API question
    bridge_api = get_bridge()
    if bridge_api and bridge_api.check_for_reply(message):
        log(f"[Bridge] Captured reply from {user} for pending question")
        return

    # ================================================================
    # COMMAND DISPATCHER — handlers are in bot_commands.py
    # ================================================================
    lower = content.lower().strip()

    # Track command usage for engagement analytics
    def _track_cmd(cmd_name: str) -> None:
        from .engagement_analytics import track_command
        args = content[len(cmd_name):].strip() if len(content) > len(cmd_name) else ""
        track_command(cmd_name, user=user, args=args[:100])

    if lower == "perf":
        _track_cmd("perf")
        await handle_perf(message, send_response)
        return
    if lower == "publish":
        _track_cmd("publish")
        await handle_publish(message, user)
        return
    if lower == "metrics":
        _track_cmd("metrics")
        await handle_metrics(message, send_response)
        return
    if lower == "idea":
        _track_cmd("idea")
        await handle_idea(message, send_response)
        return
    if lower.startswith("karen"):
        _track_cmd("karen")
        await handle_karen(message, content, user)
        return
    if lower.startswith("itinerary"):
        _track_cmd("itinerary")
        await handle_itinerary(message, content, user, memory, send_response)
        return
    if lower.startswith("betterdev"):
        _track_cmd("betterdev")
        await handle_better_dev(message, content, user, memory, send_response)
        return
    if lower == "reloadserver":
        _track_cmd("reloadserver")
        await handle_reload_server(message, user)
        return
    if lower == "evolve":
        _track_cmd("evolve")
        await handle_evolve(message, user)
        return
    if lower in ("showcommands", "commands", "help"):
        _track_cmd("commands")
        await handle_show_commands(message)
        return
    if lower in ("learninghistory", "pastlearning", "learning history"):
        _track_cmd("learninghistory")
        await handle_learning_history(message, send_response)
        return
    if lower in ("newsletter", "weeklylearning", "learning digest"):
        _track_cmd("newsletter")
        newsletter = handle_newsletter_command()
        await send_response(message, newsletter)
        return
    if lower.startswith("showlearning"):
        _track_cmd("showlearning")
        await handle_show_learning(message, content, send_response)
        return
    if lower in ("suggestlearning", "suggest_learning", "suggest learning"):
        _track_cmd("suggestlearning")
        await handle_suggest_learning(message, send_response)
        return
    if lower == "technews":
        _track_cmd("technews")
        await handle_tech_news(message, content, user, agent, memory, send_response)
        return
    if lower == "ideas":
        _track_cmd("ideas")
        await handle_show_ideas(message, send_response)
        return
    if lower.startswith("listvideos"):
        _track_cmd("listvideos")
        await handle_list_videos(message, content, user, send_response)
        return
    if lower.startswith("searchvideos"):
        _track_cmd("searchvideos")
        await handle_search_videos(message, content, send_response)
        return
    if lower.startswith("downloadvideo"):
        _track_cmd("downloadvideo")
        await handle_download_video(message, content)
        return
    if lower.startswith("downloadchannel"):
        _track_cmd("downloadchannel")
        await handle_download_channel(message, content, user)
        return
    if lower.startswith("dlcovers"):
        _track_cmd("dlcovers")
        await handle_dl_covers(message, content, user)
        return
    if lower.startswith("dlcover"):
        _track_cmd("dlcover")
        await handle_dl_cover(message, content)
        return
    if lower == "think":
        _track_cmd("think")
        await handle_think(message, content, user, memory, send_response)
        return
    if lower in ("suggest", "suggestions", "?"):
        # Context-aware command suggestions based on recent conversation
        recent = get_recent_summaries(count=5)
        suggestions = suggest_commands_for_context(recent)
        if suggestions:
            await message.reply(format_context_suggestions(suggestions))
        else:
            await message.reply(
                "No contextual suggestions right now. Use `commands` to see all available commands."
            )
        return

    # Near-miss / typo detection — only for short messages that look like
    # they might be a command (single word or two words, no question mark)
    if len(lower.split()) <= 2 and "?" not in lower and len(lower) < 30:
        closest = find_closest_command(lower)
        if closest:
            await message.reply(format_typo_suggestion(lower, closest))
            return

    # Handle reply context - fetch the original message being replied to
    reply_context = ""
    if message.reference and message.reference.message_id:
        try:
            referenced_msg = await message.channel.fetch_message(message.reference.message_id)
            if referenced_msg:
                reply_context = f'[Replying to {referenced_msg.author.name}: "{referenced_msg.content[:2000]}"]\n\n'
                log(f"{user} replied to message from {referenced_msg.author.name}")

                # Track engagement if replying to a news article
                ref_id = str(message.reference.message_id)
                if is_news_message(ref_id):
                    record_reply(ref_id, user, content[:200])
        except Exception as e:
            log(f"Could not fetch referenced message: {e}")

    # Handle file attachments
    if message.attachments:
        async with message.channel.typing():
            # Separate images from documents
            images = [a for a in message.attachments if is_image_attachment(a)]
            documents = [a for a in message.attachments if not is_image_attachment(a)]

            # Handle images with vision
            if images:
                log(f"{user} shared {len(images)} image(s)")

                # Download all images
                image_bytes_list = []
                for img in images:
                    img_data = await download_image(img)
                    if img_data:
                        image_bytes_list.append(img_data)
                        log(f"Downloaded {img.filename} ({len(img_data)} bytes)")

                if image_bytes_list:
                    try:
                        # TWO-STAGE IMAGE IDENTIFICATION:
                        # 1. Local vision model responds first (fast)
                        # 2. Claude follows up with second opinion
                        user_question = content if content else "Who/what is this?"
                        image_bytes = image_bytes_list[0]

                        # Stage 1: Local vision model (quick response)
                        log("Stage 1: Getting local vision model response...")
                        vision_result = await asyncio.to_thread(
                            analyze_with_vision_model, [image_bytes]
                        )
                        log(f"Local vision result: {vision_result[:200]}...")

                        # Format local response through the agent
                        local_prompt = f"""Your friend shared an image and asked: "{user_question}"

Your local vision analysis:
---
{vision_result}
---

Respond naturally. If you identified who/what it is, tell them. If not, describe what you see and mention you'll ask Claude for a second opinion."""

                        local_response = await asyncio.to_thread(agent.run, local_prompt, "")
                        log(f"Local response: {len(local_response)} chars")

                        # Send local response first
                        await send_response(message, local_response)

                        # Log the interaction
                        img_names = ", ".join(img.filename for img in images)
                        memory.log_conversation(
                            user, f"[Shared image(s): {img_names}] {content}", local_response
                        )

                        # Stage 2: Ask Claude for second opinion (in background)
                        log("Stage 2: Asking Claude for second opinion...")
                        async with message.channel.typing():
                            claude_prompt = f"""Look at this image and answer: {user_question}

If this is an anime/game character, identify them specifically (name and series).
If this is a real person, identify them if possible.
Be confident if you recognize them."""

                            claude_result = await asyncio.to_thread(
                                ask_claude_with_image, image_bytes, claude_prompt
                            )
                            log(f"Claude result: {claude_result[:200]}...")

                            # Only send Claude's response if it's useful
                            if (
                                claude_result
                                and "error" not in claude_result.lower()
                                and len(claude_result) > 20
                            ):
                                # Format Claude's response
                                claude_formatted = f"**Claude's take:** {claude_result}"
                                await message.channel.send(claude_formatted[:1900])
                                memory.log_conversation(
                                    "Claude", "[Second opinion on image]", claude_result
                                )

                    except Exception as e:
                        log(f"Vision analysis error: {e}")
                        await message.reply(f"had trouble analyzing that image: {e}")

                if not documents:
                    return

            # Handle documents (PDF, DOCX, TXT, code, spreadsheets, etc.)
            if documents:
                # Extract text from ALL documents in parallel
                file_contents: list[tuple[str, str]] = []  # (filename, text)
                failed_files: list[str] = []

                for attachment in documents:
                    log(f"{user} uploaded: {attachment.filename}")
                    text = await extract_text_from_file(attachment)
                    if text:
                        file_contents.append((attachment.filename, text))
                    else:
                        failed_files.append(attachment.filename)

                if failed_files:
                    await message.reply(
                        f"Couldn't read: {', '.join(failed_files)}. "
                        f"Supported: pdf, docx, txt, md, csv, xlsx, py, js, ts, json, yaml, html, sql, and more."
                    )

                if not file_contents:
                    # No readable documents
                    pass
                elif len(file_contents) == 1:
                    # Single file — existing behavior
                    fname, text = file_contents[0]
                    doc_type = content if content else "document"
                    analysis_prompt = f"""Your friend just shared their {doc_type} ({fname}) with you. Here it is:

---
{text[:15000]}
---

React like a friend would - you're genuinely interested. Talk about what stands out, what's impressive, ask follow-up questions if something's interesting."""

                    try:
                        response = await asyncio.to_thread(agent.run, analysis_prompt)
                        log(f"Analyzed {fname}: {len(response)} chars")
                        memory.log_conversation(user, f"[Uploaded: {fname}] {content}", response)
                        from .session_state import save_exchange
                        save_exchange(user, f"[Uploaded: {fname}] {content}", response)
                        await message.reply(response[:1900])
                    except Exception as e:
                        log(f"Analysis error: {e}")
                        await message.reply(f"had trouble reading that: {e}")
                else:
                    # MULTI-FILE — combine all files into one prompt for cross-file analysis
                    file_sections = []
                    total_chars = 0
                    for fname, text in file_contents:
                        # Budget ~5000 chars per file (max ~15000 total)
                        budget = min(5000, max(2000, 15000 // len(file_contents)))
                        section = f"### {fname}\n{text[:budget]}"
                        file_sections.append(section)
                        total_chars += len(section)

                    combined = "\n\n".join(file_sections)
                    file_list = ", ".join(f[0] for f in file_contents)
                    user_request = content if content else "Analyze these files together"

                    analysis_prompt = f"""Your friend shared {len(file_contents)} files: {file_list}

Their message: "{user_request}"

Here are the files:

{combined}

---

Analyze ALL files together. If there are relationships between them (e.g., code that references other files, data that correlates), point those out. Give a unified analysis, not separate per-file summaries."""

                    try:
                        response = await asyncio.to_thread(agent.run, analysis_prompt)
                        log(f"Multi-file analysis ({len(file_contents)} files, {total_chars} chars): {len(response)} chars")
                        memory.log_conversation(user, f"[Uploaded {len(file_contents)} files: {file_list}] {content}", response)
                        from .session_state import save_exchange
                        save_exchange(user, f"[Uploaded {len(file_contents)} files: {file_list}] {content}", response)
                        await send_response(message, response)
                    except Exception as e:
                        log(f"Multi-file analysis error: {e}")
                        await message.reply(f"had trouble analyzing those files: {e}")
        return

    if not content:
        return

    # Buffer message for contextual empty-response suggestions
    buffer_user_message(content)

    # Detect pasted documents (resumes, etc.)
    doc_type = detect_document_type(content)
    if doc_type:
        log(f"{user} pasted a {doc_type}")
        async with message.channel.typing():
            analysis_prompt = f"""Your friend just shared their {doc_type} with you. Here it is:

---
{content}
---

React like a friend would - you're genuinely interested. Talk about what stands out, what's impressive, ask follow-up questions if something's interesting. Use "you" and "your" - this is their stuff."""

            try:
                response = await asyncio.to_thread(agent.run, analysis_prompt)
                log(f"Analyzed {doc_type}: {len(response)} chars")

                # Store in permanent memory - replace old documents
                memory.save_permanent_memory(
                    f"## {user}'s {doc_type}\n\n{response}\n\n### Raw content:\n{content}",
                    f"documents/{user}",
                    replace_category=True,
                )

                memory.log_conversation(user, f"[Pasted {doc_type}]", response)
                await message.reply(response[:1900])
            except Exception as e:
                log(f"Analysis error: {e}")
                await message.reply(f"had trouble with that: {e}")
        return

    log(f"{user}: {content[:50]}")
    message_count += 1

    async with message.channel.typing():
        # Initialize profiling for this request
        profile = RequestProfile(user=user, message=content, message_length=len(content))
        timer = RequestTimer(profile)

        try:
            # Include reply context if this is a reply to another message
            full_content = reply_context + content if reply_context else content

            # ============================================================
            # FAST PATH — Simple questions skip the full agent pipeline.
            # Direct Ollama call: no tools, no context injection, no reflection.
            # Saves 3-8 seconds on questions like "what time is it?"
            # ============================================================
            from .llm_optimizer import score_query_complexity
            with timer.phase("smart_routing"):
                complexity = score_query_complexity(full_content)
                profile.question_type = complexity["complexity"]

            if complexity["complexity"] == "simple" and not message.attachments:
                log(f"[FastPath] Simple query ({complexity['reasoning']}) — direct Ollama call")
                from .core import _ollama_client
                _fast_start = time.monotonic()
                try:
                    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                    fast_resp = await asyncio.to_thread(
                        _ollama_client.chat,
                        model=settings.ollama_model,
                        messages=[
                            {"role": "system", "content": f"Current date/time: {now_str}\nYou are a helpful, concise assistant. Answer in 1-2 sentences."},
                            {"role": "user", "content": full_content},
                        ],
                        options={"temperature": 0.3, "num_predict": 200},
                    )
                    response = fast_resp.get("message", {}).get("content", "").strip()
                    _fast_duration = time.monotonic() - _fast_start

                    from .perf_monitor import record_llm_call
                    record_llm_call(
                        endpoint="ollama",
                        duration=round(_fast_duration, 3),
                        success=bool(response),
                        model=settings.ollama_model,
                        output_tokens=len(response) // 4,
                    )
                    log(f"[FastPath] Response in {_fast_duration:.1f}s ({len(response)} chars)")
                except Exception as fast_err:
                    log(f"[FastPath] Failed ({fast_err}), falling back to full pipeline")
                    response = None

                if response:
                    # Still do all post-response work (memory, engagement, send)
                    memory.log_conversation(user, content, response)
                    from .session_state import save_exchange as _save_ex
                    _save_ex(user, content, response)
                    buffer_conversation(user, content, response)
                    buffer_for_summary(user, content, response)

                    with timer.phase("discord_send"):
                        _send_start = time.monotonic()
                        _send_success = True
                        _send_error = ""
                        try:
                            await send_response(message, response)
                        except Exception as send_err:
                            _send_success = False
                            _send_error = str(send_err)[:200]
                            raise
                        finally:
                            _send_duration = time.monotonic() - _send_start
                            record_llm_call(
                                endpoint="discord_send",
                                duration=round(_send_duration, 3),
                                success=_send_success,
                                output_tokens=len(response),
                                error=_send_error,
                            )

                    timer.save()
                    log(f"[Profile] {profile.total:.1f}s total | FastPath | type={complexity['complexity']}")
                    asyncio.create_task(extract_facts_from_recent())
                    asyncio.create_task(generate_summaries_from_recent())
                    return  # Done — skip the full pipeline

            # ============================================================
            # STANDARD PATH — Full agent pipeline with tools and context
            # ============================================================

            # Smart context injection — tiered by message complexity
            with timer.phase("context_build"):
                ctx_parts = []
                lowered = content.lower().strip()
                context_tier = "standard"

                # Tier 1: Always inject identity summary (~2KB)
                identity = get_identity_summary()
                if "No identity" not in identity:
                    ctx_parts.append(f"Who this person is:\n{identity}")

                # Tier 2: Recent conversations for continuity (~2KB)
                recent = memory.get_context("hour")
                if "No conversations" not in recent:
                    ctx_parts.append(f"Recent conversations:\n{recent[:4000]}")
                else:
                    # No recent in-memory conversations — inject previous session
                    # exchanges so the bot knows what was just discussed before restart
                    from .session_state import get_previous_session, format_session_context
                    session_ctx = format_session_context(get_previous_session())
                    if session_ctx:
                        ctx_parts.append(session_ctx)

                # Tier 2b: Past conversation summaries for cross-session continuity
                # Only inject summaries from last 48 hours to avoid stale context
                past_summaries = get_recent_summaries(5, max_age_hours=48)
                if "No conversation summaries" not in past_summaries:
                    ctx_parts.append(f"Past conversation context:\n{past_summaries[:2000]}")

                # Tier 3: Full permanent memories only when the user asks about themselves
                self_query_patterns = [
                    "what do you know about me", "what do you remember",
                    "my profile", "my info", "tell me about me",
                    "who am i", "what have i told you", "my memories",
                    "think", "everything you know",
                ]
                if any(p in lowered for p in self_query_patterns):
                    permanent = memory.get_context("permanent")
                    if "No permanent" not in permanent:
                        ctx_parts.append(f"Full knowledge about your friend:\n{permanent[:40000]}")
                    context_tier = "full_memory"

                context = "\n".join(ctx_parts)
                profile.context_tier = context_tier
                profile.context_injected_chars = len(context)

            # Pre-classify to detect factual questions BEFORE the agent answers
            with timer.phase("pre_classification"):
                is_factual = is_factual_question(full_content)
                profile.is_factual = is_factual
                if is_factual:
                    log("[FactCheck] Factual question detected — auto-searching before answer...")
                    agent.set_temperature(0.2)
                    search_results = await asyncio.to_thread(auto_search_for_factual, agent, full_content)
                    if search_results:
                        full_content = (
                            f"{full_content}\n\n"
                            f"[Reference data from web search — use this to supplement your knowledge, "
                            f"especially for current events, prices, or recent changes:]\n"
                            f"{search_results}\n"
                            f"[End reference — Use both your knowledge and the above to give the best answer.]"
                        )

            # Refresh the timestamp in the system prompt so time-sensitive
            # queries always get the current time, not the bot's startup time.
            import re as _re
            agent.config.system_prompt = _re.sub(
                r"Current date/time: .+",
                f"Current date/time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
                agent.config.system_prompt,
                count=1,
            )

            # Attach profiler to agent so LLM calls and tool executions are recorded
            agent._request_timer = timer
            response = await asyncio.to_thread(agent.run, full_content, context)
            agent._request_timer = None

            # Pipeline logging: catch empty responses from agent.run()
            if not response or not response.strip():
                log(f"[MsgPipeline] agent.run() returned empty response "
                    f"(length={len(response) if response else 0}, repr={response!r:.200}) "
                    f"for message: {full_content[:100]!r}")

            # Record the num_ctx that was used
            if profile.llm_calls:
                profile.num_ctx_used = profile.llm_calls[0].num_ctx

            # Lightweight reflection — only for genuinely complex questions
            with timer.phase("reflection"):
                question_type = "skip" if response.startswith("Agent error:") else classify_question(full_content, response)
                profile.question_type = question_type
                log(f"[Reflection] Classified: {question_type}")

                if question_type == "full":
                    profile.reflection_mode = "light"
                    log("[Reflection] Starting light reflection (completeness + final)...")
                    agent._request_timer = timer
                    refined, thoughts = await asyncio.to_thread(reflect, agent, full_content, mode="light")
                    agent._request_timer = None
                    if refined:
                        log(f"[Reflection] Done. {len(response)} -> {len(refined)} chars")
                        if not refined.strip():
                            log(f"[MsgPipeline] Reflection returned empty refined response, "
                                f"keeping original ({len(response)} chars)")
                        else:
                            response = refined

            # Auto-escalate to Claude if the local model is uncertain or gave a weak answer
            uncertainty_phrases = [
                "i'm not sure", "i'm not certain", "i don't know",
                "i cannot verify", "i can't verify", "i don't have",
                "i'm unable to", "i can't find", "i couldn't find",
                "beyond my knowledge", "outside my training",
                "i may be wrong", "take this with a grain",
                "i'd recommend checking", "you might want to check",
                "you should check", "check the official",
            ]
            response_lower = response.lower()
            if (
                not response.startswith("Agent error:")
                and question_type in ("factual", "full")
                and any(phrase in response_lower for phrase in uncertainty_phrases)
                and settings.anthropic_api_key
            ):
                log("[AutoEscalate] Local model uncertain — escalating to Claude API...")
                try:
                    bridge = ClaudeBridge(api_key=settings.anthropic_api_key)
                    claude_prompt = f"""Answer this question accurately and thoroughly. You are chatting via Discord.

{f"Context:{chr(10)}{context[:8000]}" if context else ""}

Question: {full_content}

The local AI was uncertain and gave this answer: {response[:2000]}

Provide the correct, definitive answer. Be direct and concise."""

                    claude_response = await asyncio.to_thread(bridge.send, claude_prompt)
                    if claude_response and not claude_response.startswith("Error"):
                        log(f"[AutoEscalate] Claude responded: {len(claude_response)} chars")
                        if not claude_response.strip():
                            log("[MsgPipeline] AutoEscalate Claude returned empty response, "
                                "keeping original")
                        else:
                            response = claude_response
                except Exception as e:
                    log(f"[AutoEscalate] Claude escalation failed: {e}")

            # Check if Ollama failed - fall back to Claude API
            if response.startswith("Agent error:"):
                log(f"Ollama failed: {response[:100]}... Falling back to Claude API")
                try:
                    bridge = ClaudeBridge(api_key=settings.anthropic_api_key)
                    # Build a prompt for Claude with context
                    claude_prompt = f"""You are a helpful AI assistant chatting with your friend via Discord.

{f"Context:{chr(10)}{context[:8000]}" if context else ""}

User message: {full_content}

Respond naturally and helpfully. Be conversational and friendly."""

                    response = await asyncio.to_thread(bridge.send, claude_prompt)
                    log(f"Claude fallback response: {len(response)} chars")
                    if not response or not response.strip():
                        log(f"[MsgPipeline] Claude fallback returned empty response "
                            f"(repr={response!r:.200})")
                except Exception as claude_err:
                    log(f"Claude fallback also failed: {claude_err}")
                    fallback = get_fallback_response(full_content)
                    if fallback:
                        log(f"[Fallback] Using fallback response for: {full_content[:80]!r}")
                        response = fallback
                    else:
                        response = f"Both Ollama and Claude are unavailable. Ollama error: {response}"

            profile.response_length = len(response)
            profile.claude_escalated = "Claude" in response[:50] if response.startswith("Agent error:") else False
            log(f"Done: {len(response)} chars")

            # Log conversation to Obsidian
            memory.log_conversation(user, content, response)

            # Save exchange for cross-session continuity
            from .session_state import save_exchange
            save_exchange(user, content, response)

            # Buffer for auto memory extraction
            buffer_conversation(user, content, response)

            # Buffer for conversation context persistence
            buffer_for_summary(user, content, response)

            # Detect and log knowledge gaps (async, non-blocking)
            gap = detect_knowledge_gap(
                content, response,
                was_escalated=profile.claude_escalated,
            )
            if gap:
                try:
                    log_knowledge_gap(gap)
                    log(f"[KnowledgeGap] Logged {gap['gap_type']}: {content[:80]}")
                    # Immediately try to fill the gap for next time
                    enriched = auto_enrich_gap(gap)
                    if enriched:
                        log(f"[KnowledgeGap] {enriched}")
                except Exception as gap_err:
                    log(f"[KnowledgeGap] Failed to log: {gap_err}")

            with timer.phase("discord_send"):
                _send_start = time.monotonic()
                _send_success = True
                _send_error = ""
                try:
                    await send_response(message, response)
                except Exception as send_err:
                    _send_success = False
                    _send_error = str(send_err)[:200]
                    raise
                finally:
                    _send_duration = time.monotonic() - _send_start
                    from .perf_monitor import record_llm_call
                    record_llm_call(
                        endpoint="discord_send",
                        duration=round(_send_duration, 3),
                        success=_send_success,
                        output_tokens=len(response) if response else 0,
                        error=_send_error,
                    )

            # Save profiling data
            timer.save()
            llm_time = sum(c.duration for c in profile.llm_calls)
            log(f"[Profile] {profile.total:.1f}s total | {llm_time:.1f}s LLM ({len(profile.llm_calls)} calls) | ctx={profile.num_ctx_used} | type={profile.question_type}")

            # Try to extract identity facts (runs only when batch is full)
            asyncio.create_task(extract_facts_from_recent())

            # Try to generate conversation summaries (runs only when batch is full)
            asyncio.create_task(generate_summaries_from_recent())

        except Exception as e:
            log(f"ERR: {e}")

            # Categorize Discord errors for incident tracking
            import discord as _disc
            if isinstance(e, _disc.HTTPException):
                cat = handle_discord_error(e, context=content[:200])
                log(f"[DiscordError] {cat.name}/{cat.severity}: {cat.recovery}")
            else:
                handle_discord_error(e, context=content[:200])

            # Write crash log for debugging (sys is imported at module level)
            exc_type, exc_value, exc_tb = sys.exc_info()
            _crash_file = write_crash_log(exc_type, exc_value, exc_tb)

            # Send detailed crash to Discord webhook
            crash_details = build_crash_message(exc_type, exc_value, exc_tb)
            send_lifecycle_notification("crash", crash_details)
            try:
                await message.reply(f"Error: {e}\n(Crash log saved)")
            except Exception:
                pass  # Can't reply if the error IS a send failure


def write_crash_log(
    exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType | None
) -> Path:
    """
    Write detailed crash information to the vault for Claude to diagnose.
    Includes full stack trace and local variables from each frame.
    """
    crash_file = Path(VAULT_PATH) / "LLM Memory" / "Permanent" / "crash_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build detailed crash report
    lines = [
        "# Bot Crash Report",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Exception Type:** {exc_type.__name__}",
        f"**Exception Message:** {exc_value}",
        "",
        "## Full Stack Trace",
        "```python",
    ]

    # Add full traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    lines.extend([line.rstrip() for line in tb_lines])
    lines.append("```")

    # Extract local variables from each frame
    lines.append("")
    lines.append("## Local Variables by Frame")

    tb = exc_tb
    frame_num = 0
    while tb is not None:
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name

        lines.append("")
        lines.append(f"### Frame {frame_num}: {func_name} ({filename}:{lineno})")
        lines.append("```python")

        # Get local variables, filtering out large/complex objects
        for var_name, var_value in frame.f_locals.items():
            try:
                # Skip modules, classes, functions
                if isinstance(var_value, (type, type(sys))):
                    continue
                # Truncate long values
                value_str = repr(var_value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "... [truncated]"
                lines.append(f"{var_name} = {value_str}")
            except Exception:
                lines.append(f"{var_name} = <unable to repr>")

        lines.append("```")
        tb = tb.tb_next
        frame_num += 1

    # Write to file
    crash_file.parent.mkdir(parents=True, exist_ok=True)
    crash_file.write_text("\n".join(lines), encoding="utf-8")
    log(f"Crash log written to {crash_file}")

    return crash_file


def main() -> None:
    log("Starting with AUTO-REMEMBER...")
    try:
        client.run(settings.discord_bot_token)
    except KeyboardInterrupt:
        log("Shutdown requested")
        from .session_state import mark_clean_shutdown
        mark_clean_shutdown()
        send_lifecycle_notification("offline", "Graceful shutdown")
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        # Send crash notification
        send_lifecycle_notification("crash", f"```{str(e)[:200]}```")
        # Write comprehensive crash log
        exc_type, exc_value, exc_tb = sys.exc_info()
        crash_file = write_crash_log(exc_type, exc_value, exc_tb)
        log(f"Crash details saved to: {crash_file}")
        # Re-raise so process manager knows it crashed
        raise


if __name__ == "__main__":
    main()
