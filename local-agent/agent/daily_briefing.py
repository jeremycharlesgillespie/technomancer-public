"""
Daily Briefing — Synthesized morning digest from all bot subsystems.

Runs as a daily background task (default 7 AM). Collects data from engagement
analytics, idea board, knowledge consistency, memory system, performance
metrics, infrastructure monitor, tool analytics, news engagement, and
conversation context. Sends a concise LLM-synthesized digest to Discord.

Schedule: Daily at settings.briefing_hour (default 7 AM)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

VAULT_PATH: Path = settings.llm_memory_path
CRASH_LOG: Path = VAULT_PATH / "Permanent" / "crash_log.md"
KNOWLEDGE_GAPS: Path = VAULT_PATH / "Permanent" / "knowledge_gaps.md"
DAILY_CONTEXT: Path = VAULT_PATH / "Context" / "daily.md"

# Maximum chars per collector to keep LLM prompt manageable
MAX_SECTION_CHARS = 500

BRIEFING_PROMPT = """You are the daily briefing synthesizer for Technomancer, a Discord bot system.
Analyze the following subsystem data and produce a concise morning briefing for the bot owner.

RULES:
- Be concise - this should be readable in under 60 seconds
- Lead with the MOST IMPORTANT items (crashes, failures, anomalies)
- Group related insights together
- Use bullet points, not paragraphs
- Highlight actionable items (things requiring attention TODAY)
- If a section has no notable data, skip it entirely
- End with a single sentence summarizing overall system health

OUTPUT FORMAT (use EXACTLY these section headers):
**Attention Required** (only if there are crashes, failures, or critical issues)
- item

**Yesterday's Activity**
- key stats (messages, commands, active users)

**System Health**
- memory, performance, infrastructure highlights

**Project Health**
- blockers, stale repos, projects needing attention

**Ideas & Knowledge**
- pending ideas, knowledge gaps, consistency notes

**News & Engagement**
- what resonated, what didn't

**Today's Priorities**
- 1-3 suggested focus areas based on all the above data

Only include sections that have meaningful content. Skip empty ones.

--- ENGAGEMENT ANALYTICS ---
{engagement}

--- IDEA BOARD ---
{ideas}

--- KNOWLEDGE CONSISTENCY ---
{consistency}

--- RECENT CONVERSATIONS ---
{conversations}

--- PERFORMANCE METRICS ---
{perf_metrics}

--- INFRASTRUCTURE ---
{infra}

--- RECENT CRASHES ---
{crash_log}

--- TOOL USAGE ---
{tool_analytics}

--- NEWS ENGAGEMENT ---
{news_engagement}

--- KNOWLEDGE GAPS ---
{knowledge_gaps}

--- MEMORY CONTEXT ---
{memory_context}

--- PROJECT HEALTH ---
{project_health}
"""


# ---------------------------------------------------------------------------
# Data collectors — each returns str, wrapped in try/except
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    """Truncate text to limit, preserving last portion (most recent data)."""
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def _collect_engagement() -> str:
    """Collect yesterday's engagement analytics."""
    try:
        from .engagement_analytics import get_engagement_report

        report = get_engagement_report(days=1)
        return _truncate(report)
    except Exception as e:
        logger.debug(f"[Briefing] Engagement unavailable: {e}")
        return "No engagement data available."


def _collect_ideas() -> str:
    """Collect pending and executing ideas from the idea board."""
    try:
        from idea_board.models import load_ideas

        ideas = load_ideas()
        if not ideas:
            return "No ideas on the board."

        by_state: dict[str, int] = {}
        for idea in ideas:
            by_state[idea.state] = by_state.get(idea.state, 0) + 1

        lines = [f"Total ideas: {len(ideas)}"]
        for state, count in sorted(by_state.items()):
            lines.append(f"  {state}: {count}")

        # Show titles of proposed/executing ideas
        active = [i for i in ideas if i.state in ("proposed", "approved", "executing")]
        if active:
            lines.append("Active:")
            for idea in active[:5]:
                lines.append(f"  - [{idea.state}] {idea.title}")

        return _truncate("\n".join(lines), 600)
    except Exception as e:
        logger.debug(f"[Briefing] Ideas unavailable: {e}")
        return "Idea board unavailable."


def _collect_consistency() -> str:
    """Collect knowledge consistency audit results."""
    try:
        from .knowledge_consistency import get_consistency_report

        report = get_consistency_report()
        return _truncate(report)
    except Exception as e:
        logger.debug(f"[Briefing] Consistency unavailable: {e}")
        return "No consistency data."


def _collect_conversations() -> str:
    """Collect recent conversation summaries."""
    try:
        from .conversation_context import get_recent_summaries

        summaries = get_recent_summaries(count=10)
        return _truncate(summaries, 600)
    except Exception as e:
        logger.debug(f"[Briefing] Conversations unavailable: {e}")
        return "No recent conversations."


def _collect_perf_metrics() -> str:
    """Collect performance metrics summary."""
    try:
        from .profiler import get_performance_summary

        summary = get_performance_summary()
        return _truncate(summary)
    except Exception as e:
        logger.debug(f"[Briefing] Performance unavailable: {e}")
        return "No performance data."


def _collect_infra() -> str:
    """Collect infrastructure health report."""
    try:
        from .infra_monitor import get_infra_report

        report = get_infra_report()
        return _truncate(report, 600)
    except Exception as e:
        logger.debug(f"[Briefing] Infra unavailable: {e}")
        return "Infrastructure data unavailable."


def _collect_crash_log() -> str:
    """Read recent crash log entries."""
    try:
        if not CRASH_LOG.exists():
            return "No recent crashes."
        content = CRASH_LOG.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return "No recent crashes."
        return _truncate(content, 1500)
    except Exception as e:
        logger.debug(f"[Briefing] Crash log unavailable: {e}")
        return "Crash log unavailable."


def _collect_tool_analytics() -> str:
    """Collect tool usage analytics."""
    try:
        from .tool_analytics import get_tool_usage_report

        report = get_tool_usage_report(days=1)
        return _truncate(report)
    except Exception as e:
        logger.debug(f"[Briefing] Tool analytics unavailable: {e}")
        return "No tool usage data."


def _collect_news_engagement() -> str:
    """Collect news article engagement stats."""
    try:
        from .news_engagement import get_engagement_report

        report = get_engagement_report(days=1)
        return _truncate(report)
    except Exception as e:
        logger.debug(f"[Briefing] News engagement unavailable: {e}")
        return "No news engagement data."


def _collect_knowledge_gaps() -> str:
    """Read recent knowledge gap entries."""
    try:
        if not KNOWLEDGE_GAPS.exists():
            return "No knowledge gaps tracked."
        content = KNOWLEDGE_GAPS.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return "No knowledge gaps tracked."
        return _truncate(content, 1000)
    except Exception as e:
        logger.debug(f"[Briefing] Knowledge gaps unavailable: {e}")
        return "Knowledge gaps unavailable."


def _collect_memory_context() -> str:
    """Read yesterday's conversation context summary."""
    try:
        if not DAILY_CONTEXT.exists():
            return "No daily context available."
        content = DAILY_CONTEXT.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return "No daily context available."
        return _truncate(content, 600)
    except Exception as e:
        logger.debug(f"[Briefing] Memory context unavailable: {e}")
        return "Memory context unavailable."


def _collect_project_health() -> str:
    """Collect project health summary from the project tracker."""
    try:
        from .project_tracker import get_project_health_summary

        return _truncate(get_project_health_summary(), 600)
    except Exception as e:
        logger.debug(f"[Briefing] Project health unavailable: {e}")
        return "Project health data unavailable."


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_COLLECTORS = {
    "engagement": _collect_engagement,
    "ideas": _collect_ideas,
    "consistency": _collect_consistency,
    "conversations": _collect_conversations,
    "perf_metrics": _collect_perf_metrics,
    "infra": _collect_infra,
    "crash_log": _collect_crash_log,
    "tool_analytics": _collect_tool_analytics,
    "news_engagement": _collect_news_engagement,
    "knowledge_gaps": _collect_knowledge_gaps,
    "memory_context": _collect_memory_context,
    "project_health": _collect_project_health,
}


async def collect_all_data() -> dict[str, str]:
    """Collect data from all subsystems concurrently.

    Each collector runs in a thread to avoid blocking the event loop.
    Failures are isolated — one broken subsystem won't block others.

    Returns:
        Dict mapping section name to collected text.
    """
    results: dict[str, str] = {}

    async def _run_collector(name: str, fn: Any) -> tuple[str, str]:
        try:
            text = await asyncio.to_thread(fn)
            return name, text
        except Exception as e:
            logger.warning(f"[Briefing] Collector '{name}' failed: {e}")
            return name, f"{name} data unavailable."

    tasks = [_run_collector(name, fn) for name, fn in ALL_COLLECTORS.items()]
    completed = await asyncio.gather(*tasks)

    for name, text in completed:
        results[name] = text

    return results


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------


async def synthesize_briefing(agent: Any, data: dict[str, str]) -> str:
    """Use an LLM to synthesize collected data into a concise briefing.

    Args:
        agent: An isolated Agent instance (NOT the main bot agent).
        data: Dict of section name -> collected text.

    Returns:
        Synthesized briefing text.
    """
    prompt = BRIEFING_PROMPT.format(**data)

    try:
        response = await asyncio.to_thread(agent.run, prompt)
        if response and response.strip():
            return response.strip()
    except Exception as e:
        logger.error(f"[Briefing] LLM synthesis failed: {e}")

    # Fallback: return raw data sections
    return _build_fallback_briefing(data)


def _build_fallback_briefing(data: dict[str, str]) -> str:
    """Build a simple briefing without LLM synthesis."""
    lines = ["**Daily Briefing** (raw data — LLM synthesis unavailable)", ""]

    section_names = {
        "crash_log": "Recent Crashes",
        "engagement": "Engagement",
        "infra": "Infrastructure",
        "perf_metrics": "Performance",
        "project_health": "Project Health",
        "ideas": "Ideas",
        "consistency": "Knowledge Consistency",
        "tool_analytics": "Tool Usage",
        "news_engagement": "News Engagement",
        "conversations": "Conversations",
        "knowledge_gaps": "Knowledge Gaps",
        "memory_context": "Memory Context",
    }

    for key, title in section_names.items():
        text = data.get(key, "")
        if text and "unavailable" not in text.lower() and "no " not in text.lower()[:5]:
            lines.append(f"**{title}**")
            lines.append(text[:300])
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord formatting
# ---------------------------------------------------------------------------

DISCORD_MAX_LEN = 1900


def format_for_discord(synthesis: str) -> list[str]:
    """Split synthesis into Discord-safe message chunks.

    Args:
        synthesis: The full briefing text.

    Returns:
        List of strings, each under 2000 chars.
    """
    today = datetime.now().strftime("%A, %B %d")
    header = f"**Good morning! Here's your daily briefing for {today}:**\n\n"
    full_text = header + synthesis.strip()

    if len(full_text) <= DISCORD_MAX_LEN:
        return [full_text]

    # Split on double-newline (section boundaries)
    chunks: list[str] = []
    current = ""
    for section in full_text.split("\n\n"):
        candidate = current + "\n\n" + section if current else section
        if len(candidate) > DISCORD_MAX_LEN:
            if current:
                chunks.append(current.strip())
            current = section
        else:
            current = candidate
    if current:
        chunks.append(current.strip())

    return chunks if chunks else [full_text[:DISCORD_MAX_LEN]]


# ---------------------------------------------------------------------------
# Discord notification (via bridge, same pattern as idea_generator)
# ---------------------------------------------------------------------------


def _send_briefing_sync(messages: list[str]) -> None:
    """Blocking helper that sends briefing via the Discord bridge.

    Runs in a thread so it never blocks the event loop.
    """
    import requests

    token_file = Path(__file__).parent.parent / ".bridge_token"
    if not token_file.exists():
        logger.warning("[Briefing] No .bridge_token — cannot send to Discord")
        return

    token = token_file.read_text(encoding="utf-8").strip()

    for msg in messages:
        try:
            requests.post(
                "http://127.0.0.1:8321/api/send",
                headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
                json={"message": msg},
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"[Briefing] Bridge send failed: {e}")


async def _send_to_discord(messages: list[str]) -> None:
    """Send briefing messages to Discord via the bridge API."""
    try:
        await asyncio.to_thread(_send_briefing_sync, messages)
    except Exception as e:
        logger.warning(f"[Briefing] Discord notification failed: {e}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


async def send_daily_briefing(
    client: Any, channel_name: str, agent: Any
) -> None:
    """Collect data, synthesize, and send the daily briefing.

    Args:
        client: Discord client instance.
        channel_name: Target channel name.
        agent: Isolated Agent instance for LLM synthesis.
    """
    logger.info("[Briefing] Starting daily briefing generation...")

    # Step 1: Collect data from all subsystems
    data = await collect_all_data()
    logger.info(f"[Briefing] Collected data from {len(data)} subsystems")

    # Step 2: Synthesize with LLM
    synthesis = await synthesize_briefing(agent, data)
    logger.info(f"[Briefing] Synthesis complete ({len(synthesis)} chars)")

    # Step 3: Format and send
    messages = format_for_discord(synthesis)
    await _send_to_discord(messages)
    logger.info(f"[Briefing] Sent {len(messages)} message(s) to Discord")


def _seconds_until_briefing() -> float:
    """Calculate seconds until the next briefing time."""
    now = datetime.now()
    target = now.replace(
        hour=settings.briefing_hour, minute=0, second=0, microsecond=0
    )
    if target <= now:
        target += timedelta(days=1)
    return max((target - now).total_seconds(), 60)


async def briefing_loop(
    client: Any, channel_name: str, agent: Any
) -> None:
    """Background loop that sends daily briefing at the configured hour.

    Args:
        client: Discord client instance.
        channel_name: Target channel name.
        agent: Isolated Agent instance (NOT the main bot agent).
    """
    logger.info(
        f"[Briefing] Started — daily at {settings.briefing_hour}:00"
    )

    while True:
        try:
            wait = _seconds_until_briefing()
            logger.info(f"[Briefing] Next briefing in {wait / 3600:.1f} hours")
            await asyncio.sleep(wait)

            await send_daily_briefing(client, channel_name, agent)

            # Sleep past the trigger window
            await asyncio.sleep(120)

        except Exception as e:
            logger.error(f"[Briefing] Loop error: {e}")
            await asyncio.sleep(600)


def start_daily_briefing(
    client: Any, channel_name: str, agent: Any
) -> None:
    """Start the daily briefing background task.

    Args:
        client: Discord client instance.
        channel_name: Target channel name.
        agent: Isolated Agent instance (NOT the main bot agent).
    """
    asyncio.create_task(briefing_loop(client, channel_name, agent))
    logger.info("[Briefing] Background task started")
