"""
Ask Claude - Use the Anthropic SDK to get responses from Claude.

The public ``ask_claude(question, context)`` function is unchanged so
existing callers and tool wiring do not need updates. Internally this
calls ``anthropic.Anthropic().messages.create`` directly and attaches
an ephemeral prompt-cache breakpoint to the static system prompt so
repeated calls within 5 minutes read from Anthropic's prompt cache.
"""

import logging
from datetime import datetime
from pathlib import Path

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .config import settings

logger = logging.getLogger(__name__)

# Log file for tracking Claude queries
LOG_FILE = Path(__file__).parent.parent / "claude_queries.log"

# Keep in sync with agent/claude_vault.py::DEFAULT_MODEL
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0  # seconds — matches prior subprocess timeout
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


def _build_system_blocks(prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    """Return system content blocks with an ephemeral cache breakpoint.

    Anthropic caches the system prefix up to each cache_control marker.
    The prompt here is static across calls, so a single ephemeral
    breakpoint on the block lets calls within the 5-minute TTL hit the
    cache and pay the discounted cache_read rate instead of full input.
    """
    return [
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

# Lazy module-level client singleton.
_client: "anthropic.Anthropic | None" = None


def _get_client() -> "anthropic.Anthropic | None":
    """Return a lazily-initialized Anthropic client, or None if unavailable."""
    global _client
    if _client is not None:
        return _client
    if not HAS_ANTHROPIC:
        return None
    if not settings.anthropic_api_key:
        return None
    _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _log_cache_usage(response: object) -> None:
    """Debug-log prompt cache token counts so hits are verifiable.

    Never raises — if the response shape is unexpected the helper just
    returns quietly so a logging hiccup can't break a user-facing call.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        logger.debug(
            "ask_claude usage: cache_read=%d cache_write=%d input=%d output=%d",
            cache_read,
            cache_write,
            input_tokens,
            output_tokens,
        )
    except Exception:
        return


def log_query(question: str, response: str, success: bool) -> None:
    """Log Claude queries for debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        status = "SUCCESS" if success else "FAILED"
        f.write(f"\n[{timestamp}] {status}\n")
        f.write(f"Q: {question[:200]}...\n" if len(question) > 200 else f"Q: {question}\n")
        f.write(f"A: {response[:500]}...\n" if len(response) > 500 else f"A: {response}\n")
        f.write("-" * 80 + "\n")


def ask_claude(question: str, context: str = "") -> str:
    """
    Ask Claude a question via the Anthropic SDK.

    Args:
        question: The question to ask Claude
        context: Optional context to provide

    Returns:
        Claude's response as a string
    """
    if context:
        full_prompt = f"Context: {context}\n\nQuestion: {question}"
    else:
        full_prompt = question

    client = _get_client()
    if client is None:
        if not HAS_ANTHROPIC:
            msg = "Anthropic SDK not installed. Run: pip install anthropic"
        else:
            msg = "ANTHROPIC_API_KEY not set in environment/.env"
        log_query(question, msg, False)
        return msg

    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=_build_system_blocks(),
            messages=[{"role": "user", "content": full_prompt}],
            timeout=DEFAULT_TIMEOUT,
        )
        _log_cache_usage(response)
        text = response.content[0].text.strip()
        log_query(question, text, True)
        return text

    except anthropic.APITimeoutError:
        log_query(question, "Timeout", False)
        return "Claude took too long to respond (timeout)"
    except anthropic.APIConnectionError as e:
        log_query(question, f"Connection error: {e}", False)
        return f"Error calling Claude: connection error ({e})"
    except anthropic.APIStatusError as e:
        log_query(question, f"API error: {e}", False)
        return f"Claude API error: {e}"
    except Exception as e:
        log_query(question, str(e), False)
        return f"Error calling Claude: {e}"


def get_claude_tools() -> list:
    """Get the ask_claude tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="ask_claude",
            description=(
                "Ask Claude (the advanced AI) a question. Use this when you need "
                "Claude's expertise for complex reasoning, coding help, creative writing, "
                "or any question you want Claude to answer directly. "
                "The user can trigger this by saying 'ask claude to...' or 'hey claude...'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or request for Claude",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context to provide Claude (e.g., code snippets, background info)",
                    },
                },
                "required": ["question"],
            },
            function=ask_claude,
        ),
    ]
