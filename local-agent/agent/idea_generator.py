"""
Idea Generator — Hourly analysis of news, conversations, errors, and
performance data to suggest improvement ideas for the Technomancer project.

Runs as an async background task inside the bot process. Uses its own
dedicated Agent instance — completely isolated from the main bot agent
to prevent any interaction leakage.

Inputs:
    - ALL RSS feeds (13 sources, full articles, not just the 1 sent to Discord)
    - Recent conversation logs from Obsidian vault
    - Crash log (recent errors)
    - Request profiling data
    - Existing ideas (for dedup and refinement awareness)

Schedule:
    - Every hour (5 minutes after news digest)
    - On-demand via 'idea' Discord command
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths for input data
from .config import settings
VAULT_PATH: Path = settings.llm_memory_path
CRASH_LOG: Path = VAULT_PATH / "Permanent" / "crash_log.md"
HOURLY_CONTEXT: Path = VAULT_PATH / "Context" / "hourly.md"
PROFILING_FILE: Path = Path(__file__).parent.parent / "profiling" / "requests.jsonl"

# Schedule
GENERATION_INTERVAL_MINUTES: int = 60
OFFSET_AFTER_NEWS_MINUTES: int = 5

# Prompt for the LLM
IDEA_PROMPT = """You are an improvement analyst for the Technomancer project — a Discord bot
framework with Ollama LLM, Claude API integration, memory system, news digest,
and developer learning tools. The owner is the project owner (a Senior Software Engineer).

Analyze the following inputs and suggest 1-3 concrete improvement ideas.
Write each idea like a Jira Story — detailed enough that an engineer could
pick it up and start working on it.

For each idea, output a JSON object with these fields:
- "title": Short descriptive title (under 80 chars)
- "description": A structured description using these exact section headers on separate lines:
    "WHAT: <what to build or change>
    WHY: <what problem it solves>
    HOW: <implementation approach — files, patterns, libraries>
    BENEFITS: <how it helps the project owner — saves time, improves quality, etc.>
    COST: <resource impact — CPU/GPU/disk/API costs, or 'Minimal'>
    UNLOCKS: <what new capabilities become possible>"
- "category": One of: performance, feature, quality, security, ux
- "source": Which input prompted this (news_analysis, conversation_analysis, error_analysis, performance_analysis)

RULES:
- Be specific and actionable — not vague suggestions
- Reference specific files, functions, or metrics when possible
- Focus on things that would genuinely help Jeremy as a Sr. Software Engineer
- Consider the tech stack: Python, Discord.py, Ollama, Claude API, Obsidian
- Consider the hardware: nvidia 5080 GPU, 96GB RAM, Windows 11
- Don't suggest things already in the EXISTING IDEAS list
- Don't suggest things the CODEBASE already has (check the file list below)
- Each description should be a full paragraph, NOT just 1-2 sentences

Output ONLY a JSON array of idea objects. No other text.

--- CODEBASE (files that already exist — don't suggest features we already have) ---
{codebase}

--- TECH NEWS (what's happening in the industry) ---
{news}

--- RECENT CONVERSATIONS (what the user has been working on) ---
{conversations}

--- RECENT ERRORS (things that broke) ---
{errors}

--- PERFORMANCE DATA (response times and bottlenecks) ---
{performance}

--- EXISTING IDEAS (don't duplicate these) ---
{existing}
"""


def _load_codebase_summary() -> str:
    """List all Python files in agent/ with their first docstring line.

    This tells the LLM what already exists so it doesn't suggest
    features we've already built.

    Returns:
        Formatted file listing with descriptions
    """
    agent_dir = Path(__file__).parent
    lines = []
    for f in sorted(agent_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        # Read first docstring line
        desc = ""
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if '"""' in content:
                doc_start = content.index('"""') + 3
                doc_end = content.index('"""', doc_start)
                first_line = content[doc_start:doc_end].strip().split("\n")[0]
                desc = f" — {first_line}"
        except (ValueError, OSError):
            pass
        lines.append(f"- {f.name}{desc}")

    # Also list idea_board files
    board_dir = Path(__file__).parent.parent / "idea_board"
    if board_dir.exists():
        for f in sorted(board_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            lines.append(f"- idea_board/{f.name}")

    return "\n".join(lines)


async def _load_news_articles() -> str:
    """Fetch ALL articles from ALL RSS feeds (not just the 1 picked for Discord).

    Returns:
        Formatted string of recent news articles
    """
    try:
        from .news_digest import fetch_all_news

        articles = await fetch_all_news()

        if not articles:
            return "No recent news available."

        lines = []
        for article in articles[:30]:  # Cap at 30 to keep prompt manageable
            lines.append(f"- [{article.get('source', '')}] {article.get('title', '')}")
            summary = article.get("summary", "")[:200]
            if summary:
                lines.append(f"  {summary}")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Failed to fetch news for idea generation: {e}")
        return "News unavailable."


def _load_conversations() -> str:
    """Load recent conversation context from the vault.

    Returns:
        Recent conversation summary text
    """
    if HOURLY_CONTEXT.exists():
        try:
            return HOURLY_CONTEXT.read_text(encoding="utf-8")[:3000]
        except OSError:
            pass
    return "No recent conversations."


def _load_errors() -> str:
    """Load recent crash log entries.

    Returns:
        Last 2000 chars of the crash log
    """
    if CRASH_LOG.exists():
        try:
            content = CRASH_LOG.read_text(encoding="utf-8")
            return content[-2000:] if len(content) > 2000 else content
        except OSError:
            pass
    return "No recent errors."


def _load_performance() -> str:
    """Load recent profiling data summary.

    Returns:
        Summary of last 20 request profiles
    """
    if not PROFILING_FILE.exists():
        return "No profiling data yet."

    try:
        lines = PROFILING_FILE.read_text(encoding="utf-8").strip().split("\n")
        recent = lines[-20:]
        summaries = []
        for line in recent:
            try:
                p = json.loads(line)
                summaries.append(
                    f"- {p.get('total_seconds', 0):.1f}s | "
                    f"{p.get('llm_summary', {}).get('total_calls', 0)} LLM calls | "
                    f"type={p.get('classification', {}).get('question_type', '?')} | "
                    f"\"{p.get('message', '')[:60]}\""
                )
            except json.JSONDecodeError:
                continue
        return "\n".join(summaries) if summaries else "No profiling data."
    except OSError:
        return "No profiling data."


def _load_existing_ideas() -> str:
    """Load titles of existing ideas for dedup.

    Returns:
        Bullet list of existing idea titles
    """
    try:
        from idea_board.models import load_ideas

        ideas = load_ideas()
        if not ideas:
            return "No existing ideas."
        return "\n".join(
            f"- [{i.state}] {i.title}" for i in ideas
            if i.state not in ("vetoed", "failed")
        )
    except Exception:
        return "No existing ideas."


def _parse_ideas(response: str) -> list[dict[str, str]]:
    """Parse the LLM's JSON response into idea dicts.

    Args:
        response: Raw LLM output

    Returns:
        List of validated idea dicts
    """
    # Try to find JSON array in the response
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return []

    try:
        ideas = json.loads(raw)
        if not isinstance(ideas, list):
            return []

        valid = []
        for idea in ideas:
            if isinstance(idea, dict) and "title" in idea and "description" in idea:
                valid.append({
                    "title": str(idea["title"])[:100],
                    "description": str(idea["description"])[:2000],
                    "category": str(idea.get("category", "feature")),
                    "source": str(idea.get("source", "llm_analysis")),
                })
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


async def generate_ideas(agent: Any) -> list[dict[str, str]]:
    """Run one idea generation cycle using the provided agent.

    IMPORTANT: The agent passed here MUST be a dedicated instance,
    not the main bot agent. This prevents any interaction leakage.

    Args:
        agent: An isolated Agent instance for idea generation

    Returns:
        List of idea dicts that were created
    """
    logger.info("[IdeaGen] Starting idea generation cycle...")

    # Gather all inputs
    codebase = _load_codebase_summary()
    news = await _load_news_articles()
    conversations = _load_conversations()
    errors = _load_errors()
    performance = _load_performance()
    existing = _load_existing_ideas()

    # Build prompt
    prompt = IDEA_PROMPT.format(
        codebase=codebase,
        news=news,
        conversations=conversations,
        errors=errors,
        performance=performance,
        existing=existing,
    )

    # Run through the isolated agent
    try:
        response = await asyncio.to_thread(agent.run, prompt)
    except Exception as e:
        logger.error(f"[IdeaGen] LLM call failed: {e}")
        return []

    # Parse and save ideas
    parsed = _parse_ideas(response)
    if not parsed:
        logger.info("[IdeaGen] No new ideas generated this cycle.")
        return []

    from idea_board.models import add_idea

    created = []
    for idea_data in parsed:
        idea = add_idea(
            title=idea_data["title"],
            description=idea_data["description"],
            source=idea_data["source"],
            category=idea_data["category"],
        )
        created.append(idea_data)

    logger.info(f"[IdeaGen] Generated {len(created)} new ideas.")
    return created


async def _notify_discord(ideas: list[dict[str, str]]) -> None:
    """Send a notification to Discord with idea titles.

    Args:
        ideas: List of idea dicts with "title" and "category" keys
    """
    try:
        import requests

        token_file = Path(__file__).parent.parent / ".bridge_token"
        if not token_file.exists():
            return

        lines = [f"**{len(ideas)} new idea(s) on the board:**"]
        for idea in ideas:
            cat = idea.get("category", "")
            lines.append(f"- [{cat}] {idea['title']}")
        lines.append("\nhttp://localhost:8322")

        token = token_file.read_text(encoding="utf-8").strip()
        requests.post(
            "http://127.0.0.1:8321/api/send",
            headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
            json={"message": "\n".join(lines)},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"[IdeaGen] Discord notification failed: {e}")


async def idea_generation_loop(agent: Any) -> None:
    """Background loop that generates ideas every hour.

    Runs 5 minutes after each hour (offset from the news digest which
    runs on the hour). Uses a dedicated Agent instance.

    Args:
        agent: An isolated Agent instance (NOT the main bot agent)
    """
    logger.info(f"[IdeaGen] Started — will generate ideas every {GENERATION_INTERVAL_MINUTES} min")

    # Wait 5 minutes on startup to let other services initialize
    await asyncio.sleep(300)

    while True:
        try:
            now = datetime.now()
            # Only run during waking hours (8am-11pm) to accumulate overnight
            if 8 <= now.hour <= 23:
                created = await generate_ideas(agent)
                if created:
                    await _notify_discord(created)

            # Wait until next cycle
            await asyncio.sleep(GENERATION_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"[IdeaGen] Loop error: {e}")
            await asyncio.sleep(600)


def start_idea_generator(agent: Any) -> None:
    """Start the idea generation background task.

    Args:
        agent: An isolated Agent instance (NOT the main bot agent)
    """
    asyncio.create_task(idea_generation_loop(agent))
    logger.info("[IdeaGen] Background task started")
