"""
Weekly Learning Newsletter — Compiles recent dev_learning articles into a
digest and delivers it via Discord DM or channel message.

Runs weekly on Sunday at 9 AM and supports on-demand generation via the
``newsletter`` command.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings
from .dev_learning import list_learning_articles, LEARNING_ARTICLES_DIR

log = logging.getLogger(__name__)

# Schedule: Sunday 9 AM
NEWSLETTER_DAY = 6  # Sunday
NEWSLETTER_HOUR = 9


# ---------------------------------------------------------------------------
# Newsletter generation
# ---------------------------------------------------------------------------

def _get_articles_for_week(
    week_start: datetime | None = None,
) -> list[dict[str, Any]]:
    """Get learning articles published in a given week.

    If ``week_start`` is None, defaults to the current week (Monday-Sunday).
    """
    if week_start is None:
        today = datetime.now()
        # Roll back to Monday
        week_start = today - timedelta(days=today.weekday())
    week_start_date = week_start.date()
    week_end_date = week_start_date + timedelta(days=6)

    all_articles = list_learning_articles(limit=50)
    weekly: list[dict[str, Any]] = []

    for art in all_articles:
        try:
            art_date = datetime.strptime(art["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if week_start_date <= art_date <= week_end_date:
            weekly.append(art)

    # Chronological order (oldest first)
    weekly.sort(key=lambda a: a.get("date", ""))
    return weekly


def _read_article_excerpt(filename: str, max_chars: int = 300) -> str:
    """Read the first paragraph of an article (after frontmatter)."""
    filepath = LEARNING_ARTICLES_DIR / filename
    if not filepath.exists():
        return ""
    try:
        content = filepath.read_text(encoding="utf-8")
        # Strip frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:]
        # Strip heading
        content = re.sub(r"^#.*\n", "", content.strip())
        content = re.sub(r"^\*Category:.*\*\s*\n-+\s*\n?", "", content.strip())
        # Take first non-empty paragraph
        for para in content.split("\n\n"):
            text = para.strip()
            if text and not text.startswith("#") and not text.startswith("*") and len(text) > 30:
                if len(text) > max_chars:
                    text = text[:max_chars].rsplit(" ", 1)[0] + "..."
                return text
    except Exception:
        pass
    return ""


def _get_github_url(filename: str) -> str:
    """Try to get the GitHub Pages URL for an article."""
    try:
        from .github_pages import get_article_url

        html_name = filename.replace(".md", ".html")
        return get_article_url(html_name)
    except Exception:
        return ""


def generate_newsletter(week_start: datetime | None = None) -> str:
    """Generate a weekly learning newsletter as a formatted Discord message.

    Returns the newsletter text or an empty string if no articles exist
    for the week.
    """
    articles = _get_articles_for_week(week_start)

    if not articles:
        if week_start:
            label = week_start.strftime("%b %d")
        else:
            today = datetime.now()
            monday = today - timedelta(days=today.weekday())
            label = monday.strftime("%b %d")
        return f"No learning articles published the week of {label}."

    # Week label
    first_date = articles[0]["date"]
    last_date = articles[-1]["date"]
    try:
        start_label = datetime.strptime(first_date, "%Y-%m-%d").strftime("%b %d")
        end_label = datetime.strptime(last_date, "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        start_label = first_date
        end_label = last_date

    lines = [
        f"**Weekly Developer Learning Digest**",
        f"*{start_label} – {end_label}*",
        "",
        f"**{len(articles)} article(s)** this week:",
        "",
    ]

    # Category summary
    cats: dict[str, int] = {}
    for a in articles:
        c = a.get("category", "other")
        cats[c] = cats.get(c, 0) + 1

    cat_parts = [f"{c.replace('_', ' ').title()}: {n}" for c, n in sorted(cats.items())]
    lines.append(f"Categories: {', '.join(cat_parts)}")
    lines.append("")

    # Article list
    for i, art in enumerate(articles, 1):
        topic = art.get("topic", "Unknown")
        cat = art.get("category", "").replace("_", " ").title()
        date = art.get("date", "")

        lines.append(f"**{i}. {topic}**")
        lines.append(f"   *{cat} — {date}*")

        excerpt = _read_article_excerpt(art.get("filename", ""))
        if excerpt:
            lines.append(f"   {excerpt}")

        url = _get_github_url(art.get("filename", ""))
        if url:
            lines.append(f"   [Read on GitHub Pages]({url})")

        lines.append("")

    lines.append("---")
    lines.append("*Use `showLearning <#>` to read any article in full, "
                 "or `betterDev` to generate a new one.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

async def send_newsletter_to_channel(
    client: Any,
    channel_name: str,
    week_start: datetime | None = None,
) -> bool:
    """Send the newsletter to a Discord channel. Returns True if sent."""
    newsletter = generate_newsletter(week_start)
    if not newsletter:
        return False

    for guild in client.guilds:
        channel = next(
            (c for c in guild.text_channels if c.name == channel_name),
            None,
        )
        if channel:
            # Split into chunks if > 2000 chars
            chunks = _split_message(newsletter, 1900)
            for chunk in chunks:
                await channel.send(chunk)
            return True
    return False


async def send_newsletter_dm(client: Any, week_start: datetime | None = None) -> bool:
    """Send the newsletter as a DM to the bot owner. Returns True if sent."""
    newsletter = generate_newsletter(week_start)
    if not newsletter:
        return False

    owner_name = settings.bot_owner
    if not owner_name:
        return False

    for guild in client.guilds:
        member = next(
            (m for m in guild.members if m.name == owner_name or m.display_name == owner_name),
            None,
        )
        if member:
            try:
                chunks = _split_message(newsletter, 1900)
                for chunk in chunks:
                    await member.send(chunk)
                return True
            except Exception:
                log.debug("Could not DM %s", owner_name, exc_info=True)
    return False


def _split_message(text: str, max_len: int) -> list[str]:
    """Split a long message into chunks at line boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        if current_len + len(line) + 1 > max_len and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _seconds_until_next_run() -> float:
    """Seconds until next Sunday at NEWSLETTER_HOUR."""
    now = datetime.now()
    days_until = (NEWSLETTER_DAY - now.weekday()) % 7
    if days_until == 0 and now.hour >= NEWSLETTER_HOUR:
        days_until = 7
    next_run = (now + timedelta(days=days_until)).replace(
        hour=NEWSLETTER_HOUR, minute=0, second=0, microsecond=0,
    )
    return max((next_run - now).total_seconds(), 60)


async def newsletter_loop(client: Any, channel_name: str) -> None:
    """Background loop that sends the weekly newsletter on Sunday mornings."""
    log.info("[Newsletter] Started — sends weekly on Sundays at %d:00", NEWSLETTER_HOUR)

    while True:
        try:
            wait = _seconds_until_next_run()
            log.info("[Newsletter] Next send in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            sent = await send_newsletter_to_channel(client, channel_name)
            if sent:
                log.info("[Newsletter] Weekly digest sent to %s", channel_name)
                # Also try DM
                await send_newsletter_dm(client)
            else:
                log.info("[Newsletter] No articles to include in digest")

            # Sleep past the trigger window
            await asyncio.sleep(60)

        except Exception:
            log.exception("[Newsletter] Loop error")
            await asyncio.sleep(300)


def start_newsletter(client: Any, channel_name: str) -> None:
    """Start the weekly newsletter background task."""
    asyncio.create_task(newsletter_loop(client, channel_name))
    log.info("[Newsletter] Background task started")


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def handle_newsletter_command() -> str:
    """Handle the on-demand newsletter command. Returns the formatted newsletter."""
    return generate_newsletter()
