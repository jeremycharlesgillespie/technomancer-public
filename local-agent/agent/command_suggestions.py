"""
Context-Aware Command Suggestions — surface relevant commands based on
conversation context, typo correction, and topic matching.

Maintains a structured registry of every bot command with keywords,
categories, and descriptions.  When a message doesn't match any command
exactly, this module can:

1. Suggest the closest command (typo / near-miss detection)
2. Suggest commands relevant to recent conversation topics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandInfo:
    """Metadata about a single bot command."""

    name: str  # canonical name as typed (e.g. "betterDev")
    aliases: tuple[str, ...] = ()  # alternative spellings users might try
    description: str = ""
    category: str = ""
    keywords: tuple[str, ...] = ()  # topic keywords for context matching
    usage: str = ""  # example usage string


# ---------------------------------------------------------------------------
# Command registry — single source of truth for all bot commands
# ---------------------------------------------------------------------------

COMMANDS: tuple[CommandInfo, ...] = (
    # Learning
    CommandInfo(
        "betterDev", ("betterdev", "better_dev", "dev", "learn"),
        "Generate a learning article on a topic",
        "learning",
        ("python", "oracle", "learning", "study", "article", "education",
         "system_design", "best_practices", "tutorial"),
        "betterDev [python|oracle|system_design|best_practices|<topic>]",
    ),
    CommandInfo(
        "learningHistory", ("learninghistory", "pastlearning", "learning history"),
        "List past learning articles",
        "learning",
        ("history", "articles", "past", "learning", "list"),
        "learningHistory",
    ),
    CommandInfo(
        "showLearning", ("showlearning",),
        "View a saved learning article by number",
        "learning",
        ("article", "read", "show", "view", "learning"),
        "showLearning <number>",
    ),
    CommandInfo(
        "newsletter", ("weeklylearning", "learning digest"),
        "Get this week's learning digest",
        "learning",
        ("newsletter", "digest", "weekly", "learning", "summary", "recap"),
        "newsletter",
    ),
    # News
    CommandInfo(
        "techNews", ("technews", "news"),
        "Get latest tech news with personalised analysis",
        "news",
        ("news", "tech", "articles", "trends", "industry"),
        "techNews",
    ),
    # Memory
    CommandInfo(
        "think", (),
        "Show permanent memories about you",
        "memory",
        ("memory", "remember", "know", "profile", "about"),
        "think",
    ),
    # Idea Board
    CommandInfo(
        "ideas", ("ideas",),
        "Show active ideas from the idea board",
        "ideas",
        ("ideas", "enhancements", "features", "queue", "pending", "improvements", "board", "stories"),
        "ideas",
    ),
    # Feedback
    CommandInfo(
        "karen", (),
        "Submit a complaint to K.A.R.E.N. (generates improvement ideas)",
        "feedback",
        ("complaint", "feedback", "frustration", "karen", "annoying", "broken"),
        "karen <your complaint>",
    ),
    # Performance
    CommandInfo(
        "perf", (),
        "Show current session profiling data",
        "performance",
        ("perf", "performance", "latency", "speed", "slow", "timing", "profiling"),
        "perf",
    ),
    CommandInfo(
        "metrics", (),
        "Show persistent LLM latency trends",
        "performance",
        ("metrics", "trends", "latency", "api", "calls", "statistics"),
        "metrics",
    ),
    # Ideas
    CommandInfo(
        "idea", (),
        "Trigger immediate idea generation",
        "ideas",
        ("idea", "brainstorm", "generate", "improvement", "suggest"),
        "idea",
    ),
    # YouTube
    CommandInfo(
        "listVideos", ("listvideos",),
        "List videos from a YouTube channel",
        "youtube",
        ("youtube", "videos", "list", "channel"),
        "listVideos <channel_url>",
    ),
    CommandInfo(
        "searchVideos", ("searchvideos",),
        "Search videos in a YouTube channel",
        "youtube",
        ("youtube", "search", "videos", "find"),
        "searchVideos <channel_url> <search_term>",
    ),
    CommandInfo(
        "downloadVideo", ("downloadvideo",),
        "Download a YouTube video",
        "youtube",
        ("youtube", "download", "video", "save"),
        "downloadVideo <video_url>",
    ),
    CommandInfo(
        "downloadChannel", ("downloadchannel",),
        "Download all videos from a YouTube channel",
        "youtube",
        ("youtube", "download", "channel", "all"),
        "downloadChannel <channel_url>",
    ),
    CommandInfo(
        "dlcover", (),
        "Download video thumbnail art",
        "youtube",
        ("thumbnail", "cover", "art", "image", "youtube"),
        "dlcover <video_url>",
    ),
    CommandInfo(
        "dlcovers", (),
        "Download all thumbnails from a channel",
        "youtube",
        ("thumbnails", "covers", "art", "channel", "youtube"),
        "dlcovers <channel_url>",
    ),
    # Itinerary
    CommandInfo(
        "itinerary", (),
        "Manage travel itineraries",
        "utility",
        ("itinerary", "travel", "trip", "plan", "vacation", "schedule"),
        "itinerary <details>",
    ),
    # Admin
    CommandInfo(
        "reloadServer", ("reloadserver",),
        "Restart the bot (owner only)",
        "admin",
        ("restart", "reload", "reboot", "server"),
        "reloadServer",
    ),
    CommandInfo(
        "evolve", (),
        "Run self-improvement cycle (owner only)",
        "admin",
        ("evolve", "improve", "self", "upgrade"),
        "evolve",
    ),
    CommandInfo(
        "publish", (),
        "Sync code to public repo (owner only)",
        "admin",
        ("publish", "sync", "public", "github"),
        "publish",
    ),
    # Help
    CommandInfo(
        "commands", ("showcommands", "help"),
        "Show available commands",
        "help",
        ("help", "commands", "list", "available", "how", "usage"),
        "commands",
    ),
)

# Pre-built lookup: lowered name/alias → CommandInfo
_COMMAND_LOOKUP: dict[str, CommandInfo] = {}
for _cmd in COMMANDS:
    _COMMAND_LOOKUP[_cmd.name.lower()] = _cmd
    for _alias in _cmd.aliases:
        _COMMAND_LOOKUP[_alias.lower()] = _cmd


# ---------------------------------------------------------------------------
# Typo / near-miss detection
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def find_closest_command(text: str, max_distance: int = 2) -> CommandInfo | None:
    """Find the closest command if the input looks like a near-miss.

    Returns None if no command is close enough.
    """
    text_lower = text.lower().split()[0] if text.strip() else ""
    if not text_lower or len(text_lower) < 3:
        return None

    best: CommandInfo | None = None
    best_dist = max_distance + 1

    for key, cmd in _COMMAND_LOOKUP.items():
        dist = _edit_distance(text_lower, key)
        if dist < best_dist:
            best_dist = dist
            best = cmd

    return best if best_dist <= max_distance else None


# ---------------------------------------------------------------------------
# Context-aware suggestion
# ---------------------------------------------------------------------------

def suggest_commands_for_context(recent_text: str, limit: int = 3) -> list[CommandInfo]:
    """Suggest commands relevant to recent conversation context.

    Extracts keywords from ``recent_text`` (e.g. the last few user messages
    or conversation summaries) and scores each command by keyword overlap.
    """
    if not recent_text:
        return []

    context_words = set(re.findall(r"[a-zA-Z]{3,}", recent_text.lower()))
    if not context_words:
        return []

    scored: list[tuple[int, CommandInfo]] = []
    for cmd in COMMANDS:
        overlap = len(context_words & set(cmd.keywords))
        if overlap > 0:
            scored.append((overlap, cmd))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate (a command may appear via alias)
    seen: set[str] = set()
    results: list[CommandInfo] = []
    for _, cmd in scored:
        if cmd.name not in seen:
            seen.add(cmd.name)
            results.append(cmd)
            if len(results) >= limit:
                break
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_suggestion(cmd: CommandInfo) -> str:
    """Format a single command suggestion for Discord."""
    return f"`{cmd.usage or cmd.name}` — {cmd.description}"


def format_typo_suggestion(text: str, cmd: CommandInfo) -> str:
    """Format a 'did you mean?' message."""
    return f"Did you mean `{cmd.name}`? — {cmd.description}\n*(Use `commands` to see all available commands)*"


def format_context_suggestions(commands: list[CommandInfo]) -> str:
    """Format context-aware suggestions as a Discord message."""
    if not commands:
        return ""
    lines = ["**Suggested commands based on your recent activity:**"]
    for cmd in commands:
        lines.append(f"  • `{cmd.usage or cmd.name}` — {cmd.description}")
    lines.append("*(Use `commands` to see all available commands)*")
    return "\n".join(lines)
