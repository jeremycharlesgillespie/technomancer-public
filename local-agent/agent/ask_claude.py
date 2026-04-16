"""
Ask Claude - Use the Anthropic SDK to get responses from Claude.

The public ``ask_claude(question, context)`` function is unchanged so
existing callers and tool wiring do not need updates. Internally this
now calls ``anthropic.Anthropic().messages.create`` directly instead of
shelling out to the ``claude -p`` CLI, which removes subprocess startup
cost and enables future use of prompt caching.
"""

from datetime import datetime
from pathlib import Path

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .config import settings

# Log file for tracking Claude queries
LOG_FILE = Path(__file__).parent.parent / "claude_queries.log"

# Keep in sync with agent/claude_vault.py::DEFAULT_MODEL
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0  # seconds — matches prior subprocess timeout
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."

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
            system=DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_prompt}],
            timeout=DEFAULT_TIMEOUT,
        )
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
