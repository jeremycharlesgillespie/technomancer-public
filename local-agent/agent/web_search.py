"""
Web Search - Give the LLM ability to search the internet.

Uses DuckDuckGo for free web searches without API keys.
"""

import re

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


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
        ),
    ]
