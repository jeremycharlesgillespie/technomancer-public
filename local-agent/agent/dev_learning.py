"""
Daily Developer Learning - Educational content for professional growth.

Generates personalized 5-10 minute educational articles on programming,
system design, Oracle, and best practices. Delivered daily at 8 AM or on-demand.

Commands:
    better_dev              # Random topic from any category
    better_dev python       # Python-specific topic
    better_dev oracle       # Oracle/database topic
    better_dev system_design
    better_dev best_practices
"""

import asyncio
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import time as _time

from .config import settings
from .message_validators import validate_discord_message
from .perf_monitor import record_llm_call as _record_perf
from .web_search import web_search


# State tracking file
SENT_TOPICS_FILE = Path(__file__).parent.parent / "sent_topics.json"

# Learning articles storage in Obsidian vault
LEARNING_ARTICLES_DIR = settings.llm_memory_path / "Learning"

# Daily schedule - 8 AM
DAILY_HOUR = 8

# Curated topic catalog organized by category
LEARNING_TOPICS: dict[str, list[str]] = {
    "python": [
        # Core language features
        "Python context managers and the 'with' statement",
        "Python decorators: from basics to advanced patterns",
        "Python generators and lazy evaluation for memory efficiency",
        "Python asyncio fundamentals and async/await patterns",
        "Python type hints and static type checking with mypy",
        "Python dataclasses vs Pydantic models: when to use each",
        "Python metaclasses: understanding and practical uses",
        "Python descriptors and the descriptor protocol",
        "functools module: lru_cache, partial, reduce, and wraps",
        "Python's GIL and strategies for CPU-bound parallelism",
        # Standard library
        "Python collections module: Counter, defaultdict, deque",
        "Python itertools for efficient iteration patterns",
        "Python pathlib for modern file path handling",
        "Python logging module best practices",
        "Python unittest vs pytest: testing strategies",
        # Advanced patterns
        "Python dependency injection patterns",
        "Python factory pattern and abstract base classes",
        "Python singleton pattern: implementation and alternatives",
        "Python observer pattern with callbacks and events",
        "Python strategy pattern for runtime algorithm selection",
        # Performance
        "Python profiling with cProfile and line_profiler",
        "Python memory optimization techniques",
        "Python list comprehensions vs generator expressions",
        "Python slots for memory-efficient classes",
        "Python string interning and small integer caching",
        # Modern Python
        "Python match statements (structural pattern matching)",
        "Python walrus operator (:=) practical uses",
        "Python positional-only and keyword-only parameters",
        "Python f-strings: advanced formatting tricks",
        "Python __init__.py and package organization",
    ],
    "oracle": [
        # Query optimization
        "Oracle execution plans and EXPLAIN PLAN analysis",
        "Oracle index types: B-tree, Bitmap, and Function-based",
        "Oracle hints for query optimization",
        "Oracle optimizer statistics and histogram management",
        "Oracle adaptive query optimization features",
        # Performance tuning
        "Oracle partitioning strategies for large tables",
        "Oracle materialized views for query performance",
        "Oracle result cache for repeated queries",
        "Oracle parallel query execution",
        "Oracle connection pooling best practices",
        # PL/SQL
        "Oracle PL/SQL bulk operations: FORALL and BULK COLLECT",
        "Oracle PL/SQL exception handling patterns",
        "Oracle PL/SQL packages: organization and encapsulation",
        "Oracle PL/SQL cursors: implicit vs explicit",
        "Oracle PL/SQL autonomous transactions",
        # Advanced features
        "Oracle analytic functions: OVER, PARTITION BY, windowing",
        "Oracle flashback queries and point-in-time recovery",
        "Oracle JSON support and JSON_TABLE function",
        "Oracle CTEs (WITH clause) and recursive queries",
        "Oracle MERGE statement for upsert operations",
        # Administration
        "Oracle tablespace management and storage optimization",
        "Oracle redo logs and recovery concepts",
        "Oracle AWR reports for performance analysis",
        "Oracle Data Guard for high availability",
        "Oracle Resource Manager for workload management",
    ],
    "system_design": [
        # Fundamentals
        "CAP theorem: practical implications for distributed systems",
        "ACID vs BASE: choosing consistency models",
        "Database sharding patterns and trade-offs",
        "Horizontal vs vertical scaling strategies",
        "Load balancing algorithms: round-robin, least connections, IP hash",
        # Patterns
        "Circuit breaker pattern for fault tolerance",
        "CQRS: Command Query Responsibility Segregation",
        "Event sourcing: storing state as events",
        "Saga pattern for distributed transactions",
        "Bulkhead pattern for failure isolation",
        # Caching
        "Caching strategies: read-through, write-through, write-behind",
        "Cache invalidation strategies and patterns",
        "CDN architecture and edge caching",
        "Redis vs Memcached: choosing the right cache",
        "Cache warming and preloading strategies",
        # Messaging
        "Message queue patterns: pub/sub vs point-to-point",
        "Event-driven architecture fundamentals",
        "Kafka vs RabbitMQ: when to use each",
        "Dead letter queues and message retry strategies",
        "Idempotency in message processing",
        # Architecture
        "Microservices vs monolith: decision framework",
        "API gateway patterns and responsibilities",
        "Service mesh architecture with Istio/Linkerd",
        "Database per service vs shared database",
        "Strangler fig pattern for legacy migration",
        # Reliability
        "Designing for failure: chaos engineering principles",
        "Rate limiting and throttling strategies",
        "Health checks and readiness probes",
        "Graceful degradation patterns",
        "Distributed tracing with OpenTelemetry",
    ],
    "best_practices": [
        # Code quality
        "Writing effective unit tests: AAA pattern",
        "Test-driven development (TDD) workflow",
        "Code review best practices and checklist",
        "Refactoring techniques: extract method, inline, rename",
        "Technical debt: identification and management",
        # Git workflow
        "Git branching strategies: GitFlow vs trunk-based",
        "Writing good commit messages",
        "Git rebase vs merge: when to use each",
        "Git hooks for automated quality checks",
        "Semantic versioning (SemVer) best practices",
        # API design
        "RESTful API design principles",
        "API versioning strategies",
        "API pagination patterns: offset, cursor, keyset",
        "Error handling in APIs: status codes and messages",
        "API documentation with OpenAPI/Swagger",
        # Security
        "OWASP Top 10: common vulnerabilities",
        "Input validation and sanitization",
        "Secure password storage with bcrypt/argon2",
        "JWT tokens: best practices and pitfalls",
        "SQL injection prevention techniques",
        # Operations
        "Logging best practices: structured logging",
        "Application monitoring and alerting",
        "Feature flags for safe deployments",
        "Blue-green and canary deployment strategies",
        "Infrastructure as Code principles",
        # Collaboration
        "Documentation as code: keeping docs current",
        "Pair programming: techniques and benefits",
        "On-call best practices and incident response",
        "Sprint retrospectives: making them effective",
        "Knowledge sharing: tech talks and brown bags",
    ],
}


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [DevLearning] {msg}", flush=True)


def get_topic_hash(topic: str) -> str:
    """Generate a unique hash for a topic."""
    return hashlib.md5(topic.lower().encode()).hexdigest()[:16]


def load_sent_topics() -> list[dict[str, str]]:
    """Load previously sent topics from JSON file."""
    if SENT_TOPICS_FILE.exists():
        try:
            data = json.loads(SENT_TOPICS_FILE.read_text(encoding="utf-8"))
            return data.get("sent", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_sent_topic(topic: str, category: str) -> None:
    """Record a sent topic to avoid repeats."""
    sent = load_sent_topics()
    topic_hash = get_topic_hash(topic)

    sent.append(
        {
            "topic_hash": topic_hash,
            "topic": topic,
            "category": category,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    )

    # Keep last 365 entries (1 year of daily learning)
    sent = sent[-365:]

    SENT_TOPICS_FILE.write_text(
        json.dumps({"sent": sent, "updated": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )


def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    import re

    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]  # Limit length


def save_learning_article(topic: str, category: str, content: str) -> tuple[Path, str | None]:
    """
    Save a learning article to the Obsidian vault and GitHub Pages.

    Args:
        topic: The article topic
        category: The category (python, oracle, etc.)
        content: The full article content

    Returns:
        Tuple of (markdown_path, github_pages_url or None)
    """
    # Ensure directory exists
    LEARNING_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename: YYYY-MM-DD_category_topic-slug.md
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(topic)
    filename = f"{date_str}_{category}_{slug}.md"
    filepath = LEARNING_ARTICLES_DIR / filename

    # Build full article with frontmatter
    article = f"""---
topic: {topic}
category: {category}
date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
---

# {topic}

*Category: {category}*

---

{content}
"""

    filepath.write_text(article, encoding="utf-8")
    log(f"Article saved: {filename}")

    # Generate HTML and deploy to GitHub Pages
    html_url = None
    if settings.github_pages_enabled:
        try:
            from .github_pages import deploy_to_github, save_article_html

            _, html_url = save_article_html(topic, category, content)
            deploy_to_github(f"Add learning article: {topic[:50]}")
            log(f"Published to GitHub Pages: {html_url}")
        except Exception as e:
            log(f"GitHub Pages deployment failed: {e}")

    return filepath, html_url


def list_learning_articles(limit: int = 10) -> list[dict[str, str]]:
    """
    List saved learning articles, most recent first.

    Args:
        limit: Maximum number of articles to return

    Returns:
        List of dicts with article info (number, date, category, topic, filename)
    """
    if not LEARNING_ARTICLES_DIR.exists():
        return []

    # Get all markdown files, sorted by name (date prefix makes this chronological)
    files = sorted(LEARNING_ARTICLES_DIR.glob("*.md"), reverse=True)

    articles = []
    for i, filepath in enumerate(files[:limit], 1):
        # Parse filename: YYYY-MM-DD_category_topic-slug.md
        name = filepath.stem
        parts = name.split("_", 2)
        if len(parts) >= 3:
            date_str, category, slug = parts
            # Try to extract topic from file frontmatter
            try:
                content = filepath.read_text(encoding="utf-8")
                topic = slug.replace("-", " ").title()  # Default
                for line in content.split("\n"):
                    if line.startswith("topic:"):
                        topic = line.split(":", 1)[1].strip()
                        break
            except Exception:
                topic = slug.replace("-", " ").title()

            articles.append(
                {
                    "number": str(i),
                    "date": date_str,
                    "category": category,
                    "topic": topic,
                    "filename": filepath.name,
                }
            )

    return articles


def get_learning_article(number: int) -> tuple[str, str] | None:
    """
    Retrieve a specific learning article by its number.

    Args:
        number: The article number (1 = most recent)

    Returns:
        Tuple of (topic, content) or None if not found
    """
    articles = list_learning_articles(limit=100)

    if number < 1 or number > len(articles):
        return None

    article_info = articles[number - 1]
    filepath = LEARNING_ARTICLES_DIR / article_info["filename"]

    try:
        content = filepath.read_text(encoding="utf-8")
        # Remove frontmatter for display
        if content.startswith("---"):
            # Find second ---
            end_idx = content.find("---", 3)
            if end_idx != -1:
                content = content[end_idx + 3 :].strip()
        return article_info["topic"], content
    except Exception as e:
        log(f"Error reading article: {e}")
        return None


def get_unsent_topic(category: str | None = None) -> tuple[str, str] | None:
    """
    Get a topic that hasn't been sent recently.

    Args:
        category: Optional category filter (python, oracle, system_design, best_practices)

    Returns:
        Tuple of (category, topic) or None if no topics available
    """
    sent_hashes = {t["topic_hash"] for t in load_sent_topics()}

    # Build list of (category, topic) tuples
    if category and category in LEARNING_TOPICS:
        topics = [(category, t) for t in LEARNING_TOPICS[category]]
    else:
        topics = [(cat, t) for cat, topics_list in LEARNING_TOPICS.items() for t in topics_list]

    # Filter out recently sent topics
    available = [(cat, t) for cat, t in topics if get_topic_hash(t) not in sent_hashes]

    if not available:
        # All topics have been sent - reset by picking from full list
        log("All topics sent! Picking from full catalog.")
        available = topics

    return random.choice(available) if available else None


def load_user_profile() -> dict[str, Any]:
    """Load user profile from Obsidian vault for personalization."""
    # Reuse the profile loading from news_digest
    from .news_digest import load_user_profile as _load_profile

    return _load_profile()


async def generate_learning_content(topic: str, category: str) -> str:
    """
    Generate educational content using web search + Claude.

    Args:
        topic: The topic to write about
        category: The category (python, oracle, etc.)

    Returns:
        Generated article content as markdown
    """
    log(f"Generating content for: {topic}")

    # Step 1: Web search for current resources
    try:
        search_query = f"{topic} tutorial examples best practices 2026"
        search_results = web_search(search_query, max_results=3)
    except Exception as e:
        log(f"Web search failed: {e}")
        search_results = "(Web search unavailable)"

    # Step 2: Load user profile for personalization
    profile = load_user_profile()
    stack = ", ".join(profile.get("stack", ["Python", "Django", "PostgreSQL"]))
    role = profile.get("role", "software developer")
    interests = ", ".join(profile.get("interests", []))

    # Step 3: Generate with Claude
    prompt = f"""You are creating a 5-10 minute educational article for a {role} who works with: {stack}
Their interests include: {interests}

TOPIC: {topic}
CATEGORY: {category}

REFERENCE MATERIAL FROM WEB SEARCH (use for current context):
{search_results}

Write an article with these EXACT sections:

## Quick Overview
2-3 paragraphs explaining the concept clearly. Start with what it IS, then why it matters.
Include a simple code example if relevant to the topic.

## How This Makes You a Better Developer
1-2 paragraphs on the career and skill benefits. Be specific to their stack ({stack}).
What doors does this knowledge open? What problems does it solve?

## How to Implement It
Practical step-by-step guidance with code examples.
Use {stack.split(',')[0].strip()} where relevant.
Show real, working code they can adapt.

## Where to Use / Where NOT to Use

**Good use cases:**
- [Specific scenario 1]
- [Specific scenario 2]
- [Specific scenario 3]

**Avoid when:**
- [Anti-pattern 1]
- [Anti-pattern 2]

Include common pitfalls and mistakes to watch for.

## Key Takeaways
- [Memorable point 1]
- [Memorable point 2]
- [Memorable point 3]
- [Memorable point 4]
- [Memorable point 5]

---

**Requirements:**
- Make it conversational but professional
- Include at least 2 code examples
- Target reading time: 5-10 minutes (1000-2000 words)
- Be practical and actionable
- Include specific advice for someone using {stack}"""

    # Use claude -p (Pro subscription) instead of API credits
    from .claude_code_runner import run_claude_prompt

    def _call_claude() -> str:
        start = _time.perf_counter()
        result = run_claude_prompt(prompt, timeout=120, max_turns=1)
        duration = _time.perf_counter() - start

        if result["success"]:
            content = result["result"]
            _record_perf(
                "claude_pro_sub", duration, success=True,
                model="claude-code-pro",
            )
            log(f"Content generated: {len(content)} chars (Pro sub, ${result.get('cost_usd', 0):.4f})")
            return content
        else:
            _record_perf(
                "claude_pro_sub", duration, success=False,
                model="claude-code-pro", error=str(result.get("error", ""))[:200],
            )
            log(f"claude -p failed: {result.get('error')} — falling back to Ollama")
            return f"Error generating content: {result.get('error')}"

    return await asyncio.to_thread(_call_claude)


async def send_daily_learning(client: Any, channel_name: str) -> None:
    """Send daily learning content to Discord channel."""
    topic_info = get_unsent_topic()
    if not topic_info:
        log("No topics available")
        return

    category, topic = topic_info
    log(f"Daily topic: {topic} ({category})")

    # Find the Discord channel
    channel = None
    for guild in client.guilds:
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break
        if channel:
            break

    if not channel:
        log(f"Channel '{channel_name}' not found")
        return

    # Generate the content
    content = await generate_learning_content(topic, category)

    # Save article and get GitHub Pages URL
    _, html_url = save_learning_article(topic, category, content)

    # Send just the link to the HTML page
    message = "**Daily Developer Learning**\n\n"
    message += f"**Topic:** {topic}\n"
    message += f"**Category:** {category}\n\n"
    if html_url:
        message += f"{html_url}"
    else:
        message += "(HTML generation failed - article saved to vault only)"

    validated = validate_discord_message(message)
    if not validated:
        log(f"Skipping empty message for topic: {topic}")
        return
    await channel.send(validated)

    # Mark topic as sent
    save_sent_topic(topic, category)
    log(f"Sent and recorded: {topic}")


async def dev_learning_loop(client: Any, channel_name: str) -> None:
    """Background loop that sends daily learning at 8 AM."""
    log(f"Started - will send daily at {DAILY_HOUR}:00")

    while True:
        try:
            now = datetime.now()

            # Calculate time until next 8 AM
            if now.hour < DAILY_HOUR:
                # Today at DAILY_HOUR
                next_run = now.replace(hour=DAILY_HOUR, minute=0, second=0, microsecond=0)
            else:
                # Tomorrow at DAILY_HOUR
                next_run = now.replace(hour=DAILY_HOUR, minute=0, second=0, microsecond=0)
                next_run = next_run + timedelta(days=1)

            wait_seconds = (next_run - now).total_seconds()
            hours_until = int(wait_seconds / 3600)
            log(f"Next learning at {next_run.strftime('%Y-%m-%d %H:%M')} ({hours_until}h from now)")

            await asyncio.sleep(wait_seconds)

            # Send daily learning
            await send_daily_learning(client, channel_name)

            # Wait a minute before next check to avoid duplicate sends
            await asyncio.sleep(60)

        except Exception as e:
            log(f"Loop error: {e}")
            await asyncio.sleep(300)  # Wait 5 min on error


def start_dev_learning(client: Any, channel_name: str) -> None:
    """Start the daily learning background task."""
    asyncio.create_task(dev_learning_loop(client, channel_name))
    log("Background task started")


async def handle_better_dev_command(topic_or_category: str | None = None) -> tuple[str, str | None]:
    """
    Handle on-demand better_dev command.

    Args:
        topic_or_category: Optional - can be:
            - A predefined category (python, oracle, system_design, best_practices)
            - Any custom topic (e.g., "neo4j", "kubernetes", "GraphQL basics")
            - None for a random topic from curated list

    Returns:
        Tuple of (generated article content or error message, github_pages_url or None)
    """
    valid_categories = list(LEARNING_TOPICS.keys())

    # Determine if this is a category or a custom topic
    if topic_or_category:
        input_lower = topic_or_category.lower().strip()

        if input_lower in valid_categories:
            # It's a predefined category - pick a topic from it
            topic_info = get_unsent_topic(input_lower)
            if not topic_info:
                return "No topics available in this category!", None
            cat, topic = topic_info
        else:
            # It's a custom topic - use it directly as both topic and category
            topic = topic_or_category.strip()
            cat = topic_or_category.strip().lower()
            log(f"Custom topic requested: {topic}")
    else:
        # No input - pick a random topic from curated list
        topic_info = get_unsent_topic(None)
        if not topic_info:
            return "No topics available. All topics have been covered!", None
        cat, topic = topic_info

    # Generate content
    content = await generate_learning_content(topic, cat)

    # Mark as sent (only for curated topics, not custom ones)
    if cat != "custom":
        save_sent_topic(topic, cat)

    # Save to Obsidian and GitHub Pages
    _, html_url = save_learning_article(topic, cat, content)

    # Format response - just the link
    message = f"**Developer Learning: {topic}**\n"
    message += f"*Category: {cat}*\n\n"
    if html_url:
        message += f"{html_url}"
    else:
        message += "(HTML generation failed - article saved to vault only)"

    return message, html_url


def handle_learning_history_command(limit: int = 10) -> str:
    """
    Handle learning_history command to list past articles.

    Args:
        limit: Maximum number of articles to show

    Returns:
        Formatted list of past articles
    """
    articles = list_learning_articles(limit=limit)

    if not articles:
        return "No learning articles saved yet. Use `better_dev` to generate some!"

    lines = ["**Past Learning Articles**\n"]
    for article in articles:
        lines.append(
            f"`{article['number']}` | {article['date']} | **{article['category']}** | {article['topic']}"
        )

    lines.append("\nUse `show_learning <number>` to view an article (e.g., `show_learning 1`)")

    return "\n".join(lines)


def handle_show_learning_command(number_str: str) -> str:
    """
    Handle show_learning command to display a specific article.

    Args:
        number_str: The article number as a string

    Returns:
        The article content or error message
    """
    try:
        number = int(number_str.strip())
    except ValueError:
        return f"Invalid article number: **{number_str}**\n\nUse `learning_history` to see available articles."

    result = get_learning_article(number)
    if result is None:
        return f"Article #{number} not found.\n\nUse `learning_history` to see available articles."

    topic, content = result
    return f"**{topic}**\n\n{content}"
