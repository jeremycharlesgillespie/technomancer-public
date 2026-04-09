"""
Tech News Digest - Hourly tech news with LLM commentary.

Fetches news from tech sources, has the LLM analyze them,
and sends to Discord with opinions on relevance to developers.
Cross-references articles with conversation memory for richer context.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import feedparser

from .message_validators import validate_discord_message

# Fallback feeds if news config is not available
_FALLBACK_FEEDS = [
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    ("Hacker News", "https://hnrss.org/frontpage"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Dev.to", "https://dev.to/feed"),
    ("InfoQ", "https://feed.infoq.com/"),
    ("The New Stack", "https://thenewstack.io/feed/"),
    ("Real Python", "https://realpython.com/atom.xml"),
    ("Python Insider", "https://blog.python.org/feeds/posts/default?alt=rss"),
    ("AWS Blog", "https://aws.amazon.com/blogs/aws/feed/"),
    ("AWS Architecture", "https://aws.amazon.com/blogs/architecture/feed/"),
    ("MIT Tech Review AI", "https://www.technologyreview.com/feed/"),
    ("OpenAI Blog", "https://openai.com/blog/rss.xml"),
]


def _get_active_feeds() -> list[tuple[str, str]]:
    """Get the list of enabled feeds from news config, falling back to defaults."""
    try:
        from idea_board.news_config import get_enabled_feeds
        feeds = get_enabled_feeds()
        return feeds if feeds else _FALLBACK_FEEDS
    except Exception:
        return _FALLBACK_FEEDS

# Track sent articles to avoid duplicates
SENT_ARTICLES_FILE = Path(__file__).parent.parent / "sent_articles.json"

# Obsidian vault location for LLM context
from .config import settings
VAULT_PATH = settings.llm_memory_path
USER_PROFILE_FILE = VAULT_PATH / "Permanent" / "profile.md"


def load_user_profile() -> dict[str, Any]:
    """Load user profile from Obsidian vault markdown file."""
    profile: dict[str, Any] = {"role": "software developer", "stack": [], "interests": []}

    if not USER_PROFILE_FILE.exists():
        return profile

    try:
        content = USER_PROFILE_FILE.read_text(encoding="utf-8")
        current_section = None

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("## Role"):
                current_section = "role"
            elif line.startswith("## Tech Stack"):
                current_section = "stack"
            elif line.startswith("## Interests"):
                current_section = "interests"
            elif line.startswith("## Currently Learning"):
                current_section = "learning"
            elif line.startswith("## Avoid"):
                current_section = "avoid"
            elif line.startswith("##"):
                current_section = None
            elif line.startswith("- ") and current_section in [
                "stack",
                "interests",
                "learning",
                "avoid",
            ]:
                item = line[2:].strip()
                if item and not item.startswith("<!--"):
                    if current_section not in profile:
                        profile[current_section] = []
                    profile[current_section].append(item)
            elif (
                line
                and current_section == "role"
                and not line.startswith("#")
                and not line.startswith("<!--")
            ):
                profile["role"] = line
    except Exception as e:
        print(f"[NewsDigest] Error loading profile: {e}")

    return profile


# Stop words to exclude from keyword extraction
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it its that this with from by as are was were "
    "be been being have has had do does did will would could should may might shall can not no "
    "how what when where who why which new more most also than just about after before into over "
    "between through during their there them they your you we our all any some many much very "
    "only other each every both few own same so such too still already even now then here well "
    "says said report reports according source sources article blog post says announced today".split()
)


def extract_article_keywords(article: dict[str, str], max_keywords: int = 8) -> list[str]:
    """Extract searchable keywords from an article title and summary.

    Focuses on meaningful technical terms, proper nouns, and multi-word phrases
    that are likely to appear in conversation history.
    """
    text = f"{article.get('title', '')} {article.get('summary', '')}"
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Extract capitalized phrases (likely proper nouns / product names)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

    # Extract individual words, lowercased
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    words_lower = [w.lower() for w in words]

    # Score keywords by relevance
    keyword_scores: dict[str, int] = {}
    for noun in proper_nouns:
        noun_lower = noun.lower()
        if noun_lower not in _STOP_WORDS and len(noun_lower) > 2:
            keyword_scores[noun_lower] = keyword_scores.get(noun_lower, 0) + 3

    for w in words_lower:
        if w not in _STOP_WORDS and len(w) > 3:
            keyword_scores[w] = keyword_scores.get(w, 0) + 1

    # Sort by score descending, return top keywords
    ranked = sorted(keyword_scores, key=lambda k: keyword_scores[k], reverse=True)
    return ranked[:max_keywords]


def find_memory_connections(article: dict[str, str], max_results: int = 3) -> list[str]:
    """Search conversation memory for entries related to article topics.

    Returns a list of formatted memory match strings, or empty list if no matches
    or memory system is unavailable.
    """
    try:
        from .memory_system import get_memory_system

        mem = get_memory_system()
    except (ValueError, Exception):
        # Memory system not initialized yet
        return []

    keywords = extract_article_keywords(article)
    if not keywords:
        return []

    seen_messages: set[str] = set()
    matches: list[str] = []

    for keyword in keywords:
        query_lower = keyword.lower()
        for entry in mem.recent_conversations:
            if entry.message in seen_messages:
                continue
            if query_lower in entry.message.lower() or query_lower in entry.response.lower():
                seen_messages.add(entry.message)
                date_str = entry.timestamp.strftime("%m/%d %H:%M")
                matches.append(
                    f"- **{date_str}** ({entry.user}): {entry.message[:80]}..."
                    if len(entry.message) > 80
                    else f"- **{date_str}** ({entry.user}): {entry.message}"
                )
                if len(matches) >= max_results:
                    return matches

    return matches


def format_memory_section(connections: list[str]) -> str:
    """Format memory connections into a Discord-friendly section."""
    if not connections:
        return ""
    header = "\n**🔗 Related from your conversations:**"
    return header + "\n" + "\n".join(connections)


# Default schedule: 9am to 9pm (overridden by news config if available)
_DEFAULT_START_HOUR = 9
_DEFAULT_END_HOUR = 21  # 9pm in 24h format


def _get_schedule() -> tuple[int, int]:
    """Get active hours from news config, falling back to defaults."""
    try:
        from idea_board.news_config import get_schedule
        return get_schedule()
    except Exception:
        return _DEFAULT_START_HOUR, _DEFAULT_END_HOUR


def load_sent_articles() -> set[str]:
    """Load the set of already-sent article hashes."""
    if SENT_ARTICLES_FILE.exists():
        try:
            data = json.loads(SENT_ARTICLES_FILE.read_text(encoding="utf-8"))
            return set(data.get("sent", []))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def save_sent_articles(sent: set[str]) -> None:
    """Save the set of sent article hashes."""
    # Keep only the last 1000 to prevent file from growing forever
    sent_list = list(sent)[-1000:]
    SENT_ARTICLES_FILE.write_text(
        json.dumps({"sent": sent_list, "updated": datetime.now().isoformat()}), encoding="utf-8"
    )


def get_article_hash(title: str, link: str) -> str:
    """Generate a unique hash for an article."""
    content = f"{title}:{link}".lower()
    return hashlib.md5(content.encode()).hexdigest()[:16]


async def fetch_feed(session: aiohttp.ClientSession, name: str, url: str) -> list[dict[str, str]]:
    """Fetch and parse an RSS feed."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return []
            text = await resp.text()
            feed = feedparser.parse(text)

            articles = []
            for entry in feed.entries[:5]:  # Top 5 from each source
                articles.append(
                    {
                        "source": name,
                        "title": entry.get("title", "No title"),
                        "link": entry.get("link", ""),
                        "summary": entry.get("summary", "")[:500],  # Truncate summary
                        "published": entry.get("published", ""),
                    }
                )
            return articles
    except Exception as e:
        print(f"[NewsDigest] Error fetching {name}: {e}")
        return []


async def fetch_all_news() -> list[dict[str, str]]:
    """Fetch news from all enabled sources (configured via news config UI)."""
    feeds = _get_active_feeds()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed(session, name, url) for name, url in feeds]
        results = await asyncio.gather(*tasks)

    # Flatten and combine
    all_articles = []
    for articles in results:
        all_articles.extend(articles)

    return all_articles


def filter_new_articles(articles: list[dict[str, str]], sent: set[str]) -> list[dict[str, str]]:
    """Filter out articles that have already been sent."""
    new_articles = []
    for article in articles:
        article_hash = get_article_hash(article["title"], article["link"])
        if article_hash not in sent:
            article["hash"] = article_hash
            new_articles.append(article)
    return new_articles


async def check_relevance(agent: Any, article: dict[str, str], profile: dict[str, Any]) -> bool:
    """Quick LLM check: is this article relevant to the user?

    Returns True if the article is at least somewhat relevant, False if
    it's completely irrelevant (e.g. celebrity gossip, sports, unrelated
    industry news). Uses topic likes/dislikes from news config for guidance.
    """
    stack = ", ".join(profile.get("stack", []))
    interests = ", ".join(profile.get("interests", []))

    # Load user topic preferences from news config
    try:
        from idea_board.news_config import get_topic_preferences
        likes, dislikes = get_topic_preferences()
    except Exception:
        likes, dislikes = [], []

    likes_section = ""
    if likes:
        likes_section = f"\n\nTopics the user ESPECIALLY wants to see: {', '.join(likes)}"
        likes_section += "\nArticles about these topics should be strongly favored as RELEVANT."

    dislikes_section = ""
    if dislikes:
        dislikes_section = f"\n\nTopics the user does NOT want to see: {', '.join(dislikes)}"
        dislikes_section += "\nArticles primarily about these topics should be marked IRRELEVANT."

    prompt = f"""You are a relevance filter. Decide if this article is relevant to an American senior software engineer.

Their tech stack: {stack}
Their interests: {interests}{likes_section}{dislikes_section}

Article title: {article['title']}
Source: {article['source']}
Summary: {article['summary'][:300]}

An article is RELEVANT if it relates to ANY of:
- Their tech stack or programming languages
- Software engineering practices, tools, or industry trends
- AI, machine learning, or LLMs
- Cloud computing, DevOps, or infrastructure
- US tech industry news, policy, or regulation that affects developers
- Cybersecurity or data privacy
- Open source projects or developer tools
- Career growth or engineering leadership

An article is IRRELEVANT if it's ONLY about:
- Celebrity news, entertainment, or gossip
- Sports or gaming (unless AI/tech related)
- Consumer product reviews (phones, gadgets) with no engineering angle
- Non-US political news with no tech connection
- Social media drama or influencer content
- Topics completely outside software engineering

Respond with ONLY one word: RELEVANT or IRRELEVANT"""

    try:
        result = await asyncio.to_thread(agent.run, prompt)
        # Parse the response - look for the keyword
        result_clean = result.strip().upper()
        return "IRRELEVANT" not in result_clean
    except Exception:
        # On error, assume relevant so we don't skip everything
        return True


async def get_llm_opinion(agent: Any, article: dict[str, str]) -> str:
    """Have the LLM analyze an article and give its opinion."""
    profile = load_user_profile()
    stack = ", ".join(profile.get("stack", []))
    interests = ", ".join(profile.get("interests", []))

    prompt = f"""You're advising a {profile.get('role', 'developer')} who works with: {stack}
Their interests: {interests}

Based on the article summary below, give a brief take in this format:

**My take:** [1-2 sentences on whether this is interesting/important]

**How it affects you:** [1 sentence on practical impact given their stack/interests, or "probably doesn't affect your work"]

DO NOT search the web - just analyze what's provided:

**{article['title']}**
Source: {article['source']}
Summary: {article['summary']}

Be direct and opinionated. No fluff."""

    try:
        opinion = await asyncio.to_thread(agent.run, prompt)
        return opinion
    except Exception as e:
        return f"Couldn't analyze this one: {e}"


async def send_news_digest(client: Any, channel_name: str, agent: Any) -> None:
    """Fetch news, get LLM opinions, and send to Discord."""
    from datetime import datetime

    print(f"[NewsDigest] {datetime.now().strftime('%H:%M')} - Fetching tech news...")

    # Find the channel
    channel = None
    for guild in client.guilds:
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

    if not channel:
        print(f"[NewsDigest] Channel '{channel_name}' not found")
        return

    # Fetch news
    articles = await fetch_all_news()
    if not articles:
        print("[NewsDigest] No articles fetched")
        return

    # Filter out already-sent articles
    sent = load_sent_articles()
    new_articles = filter_new_articles(articles, sent)

    if not new_articles:
        print("[NewsDigest] No new articles to share")
        return

    # Shuffle for variety, then screen for relevance
    import random

    random.shuffle(new_articles)

    profile = load_user_profile()
    article = None
    max_candidates = min(10, len(new_articles))

    for candidate in new_articles[:max_candidates]:
        print(f"[NewsDigest] Checking relevance: {candidate['title'][:60]}...")
        is_relevant = await check_relevance(agent, candidate, profile)
        if is_relevant:
            article = candidate
            break
        print(f"[NewsDigest] Skipped (irrelevant): {candidate['title'][:60]}")
        # Mark skipped articles as sent so we don't re-check them next hour
        sent.add(candidate.get("hash", get_article_hash(candidate["title"], candidate["link"])))

    if article is None:
        # All candidates were irrelevant — fall back to first one
        print("[NewsDigest] No relevant articles found, using best available")
        article = new_articles[0]

    print(f"[NewsDigest] Analyzing: {article['title'][:50]}...")

    # Get LLM opinion
    opinion = await get_llm_opinion(agent, article)

    # Cross-reference with conversation memory
    memory_connections = find_memory_connections(article)
    memory_section = format_memory_section(memory_connections)

    # Format the message (opinion already includes "**My take:**" formatting)
    message = f"""**Tech News Alert**

**{article['title']}**
*Source: {article['source']}*

{article['link']}

{opinion}{memory_section}"""

    # Send to Discord
    try:
        validated = validate_discord_message(message)
        if not validated:
            print(f"[NewsDigest] Skipping empty message for: {article['title'][:50]}")
            return
        sent_msg = await channel.send(validated)
        print(f"[NewsDigest] Sent article: {article['title'][:50]}...")

        # Track engagement for this article
        try:
            from .news_engagement import record_article_sent

            record_article_sent(
                message_id=str(sent_msg.id),
                article_hash=article["hash"],
                title=article["title"],
                source=article["source"],
                link=article.get("link", ""),
            )
        except Exception as eng_err:
            print(f"[NewsDigest] Engagement tracking error: {eng_err}")

        # Mark as sent
        sent.add(article["hash"])
        save_sent_articles(sent)

    except Exception as e:
        print(f"[NewsDigest] Error sending: {e}")


def is_active_hour() -> bool:
    """Check if current time is within active hours (configurable via news config UI)."""
    now = datetime.now()
    start, end = _get_schedule()
    return start <= now.hour < end


async def news_digest_loop(client: Any, channel_name: str, agent: Any) -> None:
    """Background loop that sends news digest hourly during active hours."""
    start, end = _get_schedule()
    print(f"[NewsDigest] Started - will send news hourly from {start}:00 to {end}:00")

    # Wait until the next hour boundary before first check (don't send on startup)
    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0)
    next_hour = next_hour.replace(hour=next_hour.hour + 1)
    wait_seconds = (next_hour - now).total_seconds()
    print(
        f"[NewsDigest] First check at {next_hour.strftime('%H:%M')} (waiting {int(wait_seconds/60)} min)"
    )
    await asyncio.sleep(wait_seconds)

    while True:
        try:
            if is_active_hour():
                await send_news_digest(client, channel_name, agent)
            else:
                start, end = _get_schedule()
                print(
                    f"[NewsDigest] Outside active hours ({start}:00-{end}:00), skipping"
                )

            # Wait until the next hour
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0)
            next_hour = next_hour.replace(hour=next_hour.hour + 1)
            wait_seconds = (next_hour - now).total_seconds()

            print(f"[NewsDigest] Next check at {next_hour.strftime('%H:%M')}")
            await asyncio.sleep(wait_seconds)

        except Exception as e:
            print(f"[NewsDigest] Loop error: {e}")
            await asyncio.sleep(300)  # Wait 5 min on error


def start_news_digest(client: Any, channel_name: str, agent: Any) -> None:
    """Start the news digest background task."""
    asyncio.create_task(news_digest_loop(client, channel_name, agent))
    print("[NewsDigest] Background task started")


async def handle_technews_command(agent: Any) -> str:
    """
    Handle on-demand tech news request. Returns 1 article in the same
    format as the hourly digest.

    Args:
        agent: The LLM agent for generating opinions

    Returns:
        Formatted news string (single article)
    """
    print("[NewsDigest] On-demand request")

    articles = await fetch_all_news()
    if not articles:
        return "Couldn't fetch any tech news right now. Try again in a bit."

    sent = load_sent_articles()
    new_articles = filter_new_articles(articles, sent)

    if not new_articles:
        new_articles = articles  # fall back to any article if all sent

    if not new_articles:
        return "No tech news available right now."

    import random
    random.shuffle(new_articles)

    # Screen for relevance
    profile = load_user_profile()
    article = None
    max_candidates = min(10, len(new_articles))

    for candidate in new_articles[:max_candidates]:
        print(f"[NewsDigest] Checking relevance: {candidate['title'][:60]}...")
        is_relevant = await check_relevance(agent, candidate, profile)
        if is_relevant:
            article = candidate
            break
        print(f"[NewsDigest] Skipped (irrelevant): {candidate['title'][:60]}")

    if article is None:
        article = new_articles[0]  # fallback

    print(f"[NewsDigest] Analyzing: {article['title'][:50]}...")
    opinion = await get_llm_opinion(agent, article)

    article_hash = article.get("hash", get_article_hash(article["title"], article["link"]))
    sent.add(article_hash)
    save_sent_articles(sent)

    # Cross-reference with conversation memory
    memory_connections = find_memory_connections(article)
    memory_section = format_memory_section(memory_connections)

    return f"""**Tech News Alert**

**{article['title']}**
*Source: {article['source']}*

{article['link']}

{opinion}{memory_section}"""
