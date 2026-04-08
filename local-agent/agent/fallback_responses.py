"""
Fallback responses for common queries when LLM calls fail.

Provides basic answers for time/date, greetings, and other frequent queries
so the bot remains functional during Ollama/Claude API outages.
"""

import re
from datetime import datetime


def get_fallback_response(message: str) -> str | None:
    """Try to match a common query pattern and return a hardcoded response.

    Returns a response string if the message matches a known pattern,
    or None if no fallback applies (caller should use the error message).
    """
    lowered = message.lower().strip()

    for pattern, handler in _FALLBACK_HANDLERS:
        if pattern.search(lowered):
            return handler()

    return None


# ---------------------------------------------------------------------------
# Response handlers
# ---------------------------------------------------------------------------

def _time_response() -> str:
    now = datetime.now()
    return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."


def _date_response() -> str:
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}."


def _greeting_response() -> str:
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    return f"{greeting}! My LLM backends are currently down, but I'm still here. What do you need?"


def _help_response() -> str:
    return (
        "I'm having trouble reaching my AI backends right now, but here are some things I can do:\n"
        "- `perf` — show performance stats\n"
        "- `showCommands` — list all commands\n"
        "- `technews` — latest tech news\n"
        "- `betterDev` — generate a learning article\n"
        "I should be back to full capability soon!"
    )


def _status_response() -> str:
    return (
        "I'm online but my LLM backends (Ollama/Claude) are currently unavailable. "
        "Basic commands still work. I'll be back to normal once the APIs recover."
    )


# ---------------------------------------------------------------------------
# Pattern → handler mapping
# ---------------------------------------------------------------------------

_FALLBACK_HANDLERS: list[tuple[re.Pattern, callable]] = [
    # Time queries
    (re.compile(r"\bwhat\s+time\b"), _time_response),
    (re.compile(r"\bwhat\'?s?\s+the\s+time\b"), _time_response),
    (re.compile(r"\bcurrent\s+time\b"), _time_response),
    (re.compile(r"\btime\s+is\s+it\b"), _time_response),

    # Date queries
    (re.compile(r"\bwhat\s+date\b"), _date_response),
    (re.compile(r"\bwhat\'?s?\s+the\s+date\b"), _date_response),
    (re.compile(r"\bwhat\s+day\s+is\s+it\b"), _date_response),
    (re.compile(r"\btoday\'?s?\s+date\b"), _date_response),
    (re.compile(r"\bcurrent\s+date\b"), _date_response),

    # Greetings
    (re.compile(r"^(hi|hello|hey|yo|sup|what\'?s?\s*up|howdy|good\s+(morning|afternoon|evening))[\s!?.]*$"), _greeting_response),

    # Help / status
    (re.compile(r"^(help|commands)[\s!?.]*$"), _help_response),
    (re.compile(r"\bare\s+you\s+(there|alive|ok|online|working)\b"), _status_response),
    (re.compile(r"\byou\s+(up|down|broken|dead)\b"), _status_response),
]
