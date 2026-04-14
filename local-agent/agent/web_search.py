"""
Web Search - Give the LLM ability to search the internet.

Uses DuckDuckGo for free web searches without API keys.
Includes context-aware search with source credibility scoring.
"""

import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from .config import settings
from .core import _ollama_client

log = logging.getLogger(__name__)

# =============================================================================
# DOMAIN CREDIBILITY SCORING
# =============================================================================
# Tier 1 (score 90-100): Official documentation, standards bodies, academic
# Tier 2 (score 70-89): Major tech platforms, curated Q&A, reputable news
# Tier 3 (score 50-69): Blogs, tutorials, general tech sites
# Tier 4 (score 30-49): Forums, user-generated content, social media
# Tier 5 (score 10-29): Unknown or low-trust domains

DOMAIN_CREDIBILITY: dict[str, int] = {
    # Tier 1 — Official docs, standards, academic
    "docs.python.org": 95,
    "docs.djangoproject.com": 95,
    "developer.mozilla.org": 95,
    "docs.microsoft.com": 95,
    "learn.microsoft.com": 95,
    "docs.aws.amazon.com": 95,
    "cloud.google.com": 93,
    "docs.docker.com": 93,
    "kubernetes.io": 93,
    "postgresql.org": 95,
    "dev.mysql.com": 93,
    "redis.io": 93,
    "nginx.org": 93,
    "rust-lang.org": 95,
    "go.dev": 95,
    "typescriptlang.org": 95,
    "nodejs.org": 95,
    "reactjs.org": 93,
    "react.dev": 93,
    "vuejs.org": 93,
    "angular.io": 93,
    "pypi.org": 90,
    "crates.io": 90,
    "npmjs.com": 90,
    "w3.org": 95,
    "ietf.org": 95,
    "rfc-editor.org": 95,
    "arxiv.org": 92,
    "ieee.org": 92,
    "acm.org": 92,
    "github.com": 85,
    "gitlab.com": 85,
    # Tier 2 — Curated Q&A, reputable tech news/tutorials
    "stackoverflow.com": 82,
    "stackexchange.com": 80,
    "serverfault.com": 80,
    "superuser.com": 78,
    "realpython.com": 80,
    "digitalocean.com": 78,
    "aws.amazon.com": 85,
    "azure.microsoft.com": 85,
    "wiki.archlinux.org": 80,
    "man7.org": 82,
    "cppreference.com": 85,
    "en.wikipedia.org": 75,
    "arstechnica.com": 75,
    "techcrunch.com": 72,
    "theverge.com": 70,
    "wired.com": 72,
    "infoq.com": 78,
    "thenewstack.io": 75,
    "martinfowler.com": 82,
    "blog.golang.org": 85,
    "blog.rust-lang.org": 85,
    "engineering.fb.com": 80,
    "netflixtechblog.com": 80,
    "uber.com/blog": 78,
    # Tier 3 — Blogs, tutorials, general tech
    "dev.to": 60,
    "medium.com": 55,
    "towardsdatascience.com": 60,
    "freecodecamp.org": 65,
    "baeldung.com": 68,
    "geeksforgeeks.org": 58,
    "tutorialspoint.com": 55,
    "w3schools.com": 55,
    "hackernoon.com": 55,
    "dzone.com": 55,
    "smashingmagazine.com": 65,
    "css-tricks.com": 65,
    # Tier 4 — Forums, user-generated, social
    "reddit.com": 40,
    "quora.com": 35,
    "news.ycombinator.com": 55,
    "twitter.com": 30,
    "x.com": 30,
    "facebook.com": 25,
    "discord.com": 25,
}

# Patterns for matching subdomains (e.g., docs.aws.amazon.com -> aws.amazon.com)
DOMAIN_PATTERNS: list[tuple[str, int]] = [
    ("docs.", 90),  # Any docs.* subdomain gets a boost
    ("developer.", 88),
    ("wiki.", 70),
    ("blog.", 60),
    ("forum.", 40),
    ("community.", 45),
]

# Credibility tier labels
CREDIBILITY_TIERS: list[tuple[int, str, str]] = [
    (90, "official", "[OFFICIAL]"),
    (70, "trusted", "[TRUSTED]"),
    (50, "informative", "[INFO]"),
    (30, "community", "[COMMUNITY]"),
    (0, "unverified", "[UNVERIFIED]"),
]

DEFAULT_CREDIBILITY = 40


def get_domain_credibility(url: str) -> tuple[int, str]:
    """Score a URL's domain credibility and return (score, tier_label).

    Checks exact domain match first, then falls back to parent domain,
    then subdomain pattern heuristics, then default.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return DEFAULT_CREDIBILITY, _tier_label(DEFAULT_CREDIBILITY)

    # Strip www.
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Exact match
    if hostname in DOMAIN_CREDIBILITY:
        score = DOMAIN_CREDIBILITY[hostname]
        return score, _tier_label(score)

    # Try parent domain (e.g., blog.python.org -> python.org)
    parts = hostname.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in DOMAIN_CREDIBILITY:
            score = DOMAIN_CREDIBILITY[parent]
            return score, _tier_label(score)

    # Subdomain pattern heuristic
    for prefix, base_score in DOMAIN_PATTERNS:
        if hostname.startswith(prefix):
            return base_score, _tier_label(base_score)

    return DEFAULT_CREDIBILITY, _tier_label(DEFAULT_CREDIBILITY)


def _tier_label(score: int) -> str:
    """Return the tier badge string for a credibility score."""
    for threshold, _name, label in CREDIBILITY_TIERS:
        if score >= threshold:
            return label
    return "[UNVERIFIED]"


# =============================================================================
# LLM QUERY REWRITING
# =============================================================================

def _rewrite_query_with_llm(user_query: str) -> str:
    """Use Ollama to rewrite a natural language question into an optimal search query.

    Falls back to the original query if the LLM call fails.
    """
    try:
        response = _ollama_client.chat(
            model=settings.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. Given a user's question, "
                        "rewrite it as a concise, effective web search query. "
                        "Rules:\n"
                        "- Output ONLY the search query, nothing else\n"
                        "- Remove filler words and conversational language\n"
                        "- Add technical terms that would match authoritative sources\n"
                        "- Keep it under 10 words\n"
                        "- Do not add quotes unless searching for an exact phrase"
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.3, "num_ctx": 2048},
            keep_alive=-1,
        )
        rewritten = (response.get("message", {}).get("content", "") or "").strip()
        # Strip thinking tags if present
        rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
        # Sanity check: if empty or absurdly long, fall back
        if not rewritten or len(rewritten) > 200:
            return user_query
        log.info(f"[SmartSearch] Query rewritten: '{user_query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        log.warning(f"[SmartSearch] Query rewrite failed, using original: {e}")
        return user_query


# =============================================================================
# SMART SEARCH — Context-aware search with credibility scoring
# =============================================================================

def web_search_smart(query: str, max_results: int = 8) -> str:
    """Context-aware web search with source credibility scoring.

    1. Rewrites the query using LLM for optimal search terms
    2. Searches DuckDuckGo for results
    3. Scores each result by domain credibility
    4. Sorts by credibility score (highest first)
    5. Presents results with credibility badges

    Args:
        query: The user's natural language question
        max_results: Maximum number of results to return (default 8)

    Returns:
        Formatted search results with credibility indicators, sorted by authority
    """
    # Step 1: Rewrite query for better search results
    optimized_query = _rewrite_query_with_llm(query)

    # Step 2: Search with DuckDuckGo (fetch more than needed to allow filtering)
    try:
        fetch_count = min(max_results + 5, 15)  # Over-fetch for better ranking
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(optimized_query, max_results=fetch_count))
    except Exception as e:
        return f"Search error: {e}"

    if not raw_results:
        return f"No results found for: {query} (searched: {optimized_query})"

    # Step 3: Score and sort by credibility
    scored_results = []
    for r in raw_results:
        url = r.get("href", "")
        score, tier = get_domain_credibility(url)
        scored_results.append({
            "title": r.get("title", "No title"),
            "body": r.get("body", "No description"),
            "url": url,
            "score": score,
            "tier": tier,
        })

    # Sort by credibility score descending, then by original rank as tiebreaker
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Trim to requested count
    scored_results = scored_results[:max_results]

    # Step 4: Format output with credibility badges
    output = [f"Smart search results for: {query}"]
    if optimized_query != query:
        output.append(f"(optimized query: {optimized_query})")
    output.append("")

    for i, r in enumerate(scored_results, 1):
        output.append(f"{i}. {r['tier']} **{r['title']}** (credibility: {r['score']}/100)")
        output.append(f"   {r['body']}")
        if r["url"]:
            output.append(f"   Source: {r['url']}")
        output.append("")

    # Add legend
    output.append("---")
    output.append(
        "Credibility: [OFFICIAL] 90+ | [TRUSTED] 70-89 | "
        "[INFO] 50-69 | [COMMUNITY] 30-49 | [UNVERIFIED] <30"
    )

    return "\n".join(output)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default 5)

    Returns:
        Formatted search results as a string
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        # Format results nicely
        output = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            url = r.get("href", "")
            output.append(f"{i}. **{title}**")
            output.append(f"   {body}")
            if url:
                output.append(f"   Source: {url}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Search error: {e}"


def web_search_news(query: str, max_results: int = 5) -> str:
    """
    Search for recent news using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted news results as a string
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        if not results:
            return f"No news found for: {query}"

        output = [f"Recent news for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            date = r.get("date", "")
            url = r.get("url", "")
            source = r.get("source", "")

            output.append(f"{i}. **{title}**")
            if source:
                output.append(f"   Source: {source} | {date}")
            output.append(f"   {body}")
            if url:
                output.append(f"   Link: {url}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"News search error: {e}"


def web_fetch(url: str) -> str:
    """
    Fetch and extract readable content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        The extracted text content from the page
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find the main article content
        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"article|content|post|entry"))
        )

        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated...]"

        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error parsing {url}: {e}"


def get_web_tools() -> list:
    """Get web search tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "web_search",
            (
                "Search the web for information. Use this when you need current information, "
                "facts you don't know, prices, news, recent events, or anything that might be "
                "found online. Examples: stock prices, weather, sports scores, recent news, "
                "product info, definitions, how-to guides, etc."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - be specific for better results",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
            web_search,
            timeout=45,
        ),
        create_tool(
            "web_search_smart",
            (
                "Smart web search with source credibility scoring. Rewrites your question "
                "into an optimal search query, ranks results by source authority, and shows "
                "credibility badges. Use this for technical questions, research, or when "
                "source quality matters. Prioritizes official docs and trusted sources over "
                "forums and user-generated content."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your question or search topic — can be natural language",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8, max 15)",
                    },
                },
                "required": ["query"],
            },
            web_search_smart,
            timeout=60,
        ),
        create_tool(
            "web_search_news",
            (
                "Search for recent news articles. Use this specifically for current events, "
                "breaking news, or recent developments about a topic."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The news search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
            web_search_news,
            timeout=45,
        ),
        create_tool(
            "web_fetch",
            (
                "Fetch and read the content of a specific URL. Use this when you have a URL "
                "and need to read its contents - for example, when someone replies to a message "
                "that contains a link and asks questions about it, or when you need to get "
                "detailed information from a specific article or webpage."
            ),
            {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (e.g., https://example.com/article)",
                    }
                },
                "required": ["url"],
            },
            web_fetch,
            timeout=30,
        ),
    ]
