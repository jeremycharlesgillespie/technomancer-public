"""
Claude Vault - Efficient Claude API access to Obsidian vault with prompt caching.

Features:
- Caches static user data (profile, resume) in system prompt
- Provides tools for on-demand vault queries
- Minimizes token usage with strategic cache breakpoints
- Supports 5-minute and 1-hour cache TTL options
"""

import logging
import re
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# The module-level ``anthropic`` above is the shim. For genuine prompt
# caching we need the real SDK — loaded lazily inside ClaudeVault so
# the shim stays in effect everywhere else.
from . import real_anthropic

from .config import settings
from .logging_config import DEFAULT_REQUEST_ID, request_id_var
from .perf_monitor import record_llm_call as _record_perf


def _request_id_headers() -> dict[str, str]:
    """Return ``{"X-Request-ID": <id>}`` when a non-default rid is set."""
    rid = request_id_var.get()
    if rid and rid != DEFAULT_REQUEST_ID:
        return {"X-Request-ID": rid}
    return {}

# Constants
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_SEARCH_RESULTS = 10

# Pricing per 1M tokens (as of 2026) - Claude Sonnet
PRICING = {
    "input": 3.00,  # $3.00 per 1M input tokens
    "output": 15.00,  # $15.00 per 1M output tokens
    "cache_read": 0.30,  # $0.30 per 1M cached tokens (90% discount)
    "cache_write": 3.75,  # $3.75 per 1M cache write tokens (25% premium)
}

# =============================================================================
# PROCESS-WIDE PROMPT-CACHE STATS
# =============================================================================
# Cumulative counters for every Anthropic response processed by this module.
# Reset only on process restart. Guarded by a lock so concurrent ask() /
# ask_with_tools() callers do not drop increments.

_cache_stats_lock = threading.Lock()
_cache_stats: dict[str, int] = {
    "calls": 0,
    "input_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
}


def _record_cache_stats(usage: Any) -> None:
    """Accumulate token counts from one Anthropic ``usage`` block."""
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    with _cache_stats_lock:
        _cache_stats["calls"] += 1
        _cache_stats["input_tokens"] += int(input_tokens)
        _cache_stats["cache_read_tokens"] += int(cache_read)
        _cache_stats["cache_creation_tokens"] += int(cache_creation)


def _reset_cache_stats() -> None:
    """Zero out the process-wide counters. For tests only."""
    with _cache_stats_lock:
        _cache_stats["calls"] = 0
        _cache_stats["input_tokens"] = 0
        _cache_stats["cache_read_tokens"] = 0
        _cache_stats["cache_creation_tokens"] = 0


def get_cache_stats() -> dict[str, Any]:
    """Return cumulative prompt-cache stats for the process lifetime.

    Keys:
        calls — number of Anthropic responses observed
        input_tokens — sum of ``input_tokens``
        cache_read_tokens — sum of ``cache_read_input_tokens``
        cache_creation_tokens — sum of ``cache_creation_input_tokens``
        cache_hit_rate — cache_read_tokens / (cache_read + cache_creation +
            input_tokens); 0.0 when the denominator is 0
    """
    with _cache_stats_lock:
        calls = _cache_stats["calls"]
        input_tokens = _cache_stats["input_tokens"]
        cache_read = _cache_stats["cache_read_tokens"]
        cache_creation = _cache_stats["cache_creation_tokens"]
    denom = cache_read + cache_creation + input_tokens
    hit_rate = (cache_read / denom) if denom else 0.0
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "cache_hit_rate": hit_rate,
    }


# =============================================================================
# VAULT DATA EXTRACTION
# =============================================================================


def load_profile(vault_path: Path) -> dict[str, Any]:
    """Load user profile from profile.md."""
    profile: dict[str, Any] = {"role": "software developer", "stack": [], "interests": []}
    profile_file = vault_path / "Permanent" / "profile.md"

    if not profile_file.exists():
        return profile

    try:
        content = profile_file.read_text(encoding="utf-8")
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
            elif line.startswith("##"):
                current_section = None
            elif line.startswith("- ") and current_section in [
                "stack",
                "interests",
                "learning",
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
        print(f"[ClaudeVault] Error loading profile: {e}")

    return profile


def extract_resume_section(memories_content: str) -> str:
    """Extract just the resume section from memories.md."""
    # Look for resume-related content
    lines = memories_content.splitlines()
    resume_lines = []
    in_resume = False

    for line in lines:
        # Detect start of resume section (various markers)
        if any(
            marker in line.lower()
            for marker in ["resume", "work experience", "employment", "career"]
        ):
            in_resume = True
        # Detect end (new section starting with different category)
        elif in_resume and line.startswith("## ") and "resume" not in line.lower():
            # Check if this is a new unrelated section
            if any(marker in line.lower() for marker in ["system", "extracted_facts", "notes"]):
                in_resume = False
                continue

        if in_resume:
            resume_lines.append(line)

    if resume_lines:
        return "\n".join(resume_lines)

    # Fallback: return all permanent memories (they likely contain resume)
    return memories_content


def build_user_summary(vault_path: Path) -> str:
    """Build a compact user summary (~300 tokens) for quick reference."""
    profile = load_profile(vault_path)
    memories_file = vault_path / "Permanent" / "memories.md"

    summary_parts = ["## Quick Reference"]

    # Extract name from memories if available
    name = "User"
    current_role = ""
    if memories_file.exists():
        try:
            content = memories_file.read_text(encoding="utf-8")
            # Look for name patterns
            name_match = re.search(r"(?:Name|name|User):\s*([^\n]+)", content)
            if name_match:
                name = name_match.group(1).strip()

            # Look for current role/company
            role_match = re.search(r"(?:Current|current|Now|now)[^\n]*(?:at|@)\s*([^\n]+)", content)
            if role_match:
                current_role = role_match.group(1).strip()
        except Exception:
            pass

    summary_parts.append(f"- Name: {name}")
    if current_role:
        summary_parts.append(f"- Current: {current_role}")

    # Add profile info
    if profile.get("role"):
        summary_parts.append(f"- Role: {profile['role']}")
    if profile.get("stack"):
        summary_parts.append(f"- Stack: {', '.join(profile['stack'][:8])}")
    if profile.get("interests"):
        summary_parts.append(f"- Interests: {', '.join(profile['interests'][:6])}")
    if profile.get("learning"):
        summary_parts.append(f"- Learning: {', '.join(profile['learning'][:4])}")

    summary_parts.append("\n[Full details available via vault tools if needed]")

    return "\n".join(summary_parts)


def _finalize_cache_breakpoint(blocks: list[dict]) -> list[dict]:
    """Place a single ephemeral cache_control marker on the last static block.

    The Anthropic prompt cache keys on the prefix ending at each
    cache_control marker. We want exactly one breakpoint, on the final
    static block, so the whole system prefix is cached as a unit and
    dynamic per-call content (user query, recent context) stays OUT of
    the cached prefix — it belongs in `messages`.
    """
    if not blocks:
        return blocks
    for block in blocks[:-1]:
        block.pop("cache_control", None)
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_cached_prefix(vault_path: Path) -> list[dict]:
    """
    Build system prompt blocks with a single ephemeral cache breakpoint.

    Structure (all static — rebuilt at most once per hour):
    1. Core instructions (~100 tokens)
    2. User profile quick reference (~200 tokens)
    3. Full resume + facts (~3,500 tokens)

    The final block always carries cache_control: ephemeral so the entire
    prefix is cached together. Per-call dynamic content (user question,
    recent conversation context) is passed via `messages`, not `system`.

    Returns list of content blocks for system parameter.
    """
    blocks: list[dict] = []

    # Block 1: Core instructions
    core_instructions = """You are Claude, assisting a user whose profile and history are provided below.

You have access to tools for querying the user's Obsidian vault:
- read_vault_file: Read any file from the vault
- search_vault: Search across vault files
- list_vault_contents: List files in a directory
- get_recent_conversations: Get conversation history

Use these tools when you need specific details not in your cached context.
Be concise and helpful. Reference the user's background when relevant."""

    blocks.append({"type": "text", "text": core_instructions})

    # Block 2: User summary (quick reference)
    try:
        user_summary = build_user_summary(vault_path)
        blocks.append({"type": "text", "text": user_summary})
    except Exception as e:
        blocks.append({"type": "text", "text": f"[Profile unavailable: {e}]"})

    # Block 3+: stable long-form context from Permanent/.
    # Pulling Resume, claude_handoff, and memories together pushes the
    # cached prefix well above Anthropic's 1024-token minimum for cache
    # activation on Sonnet. Everything here is static — rebuilt hourly
    # via refresh_cache().
    permanent = vault_path / "Permanent"
    static_sources = [
        ("Resume", permanent / "Resume.md"),
        ("Handoff Notes", permanent / "claude_handoff.md"),
        ("Permanent Knowledge", permanent / "memories.md"),
    ]
    any_content = False
    for label, src in static_sources:
        if not src.exists():
            continue
        try:
            content = src.read_text(encoding="utf-8")
        except Exception as e:
            blocks.append({"type": "text", "text": f"[{label} unavailable: {e}]"})
            continue
        if len(content) > 20000:
            content = content[:20000] + "\n\n[Truncated...]"
        blocks.append({"type": "text", "text": f"## {label}\n\n{content}"})
        any_content = True

    if not any_content:
        blocks.append({"type": "text", "text": "[No permanent context available]"})

    return _finalize_cache_breakpoint(blocks)


# =============================================================================
# VAULT TOOLS
# =============================================================================


def read_vault_file(path: str, vault_path: Path | None = None) -> str:
    """Read a file from the vault."""
    if vault_path is None:
        vault_path = settings.llm_memory_path

    file_path = vault_path / path
    if not file_path.exists():
        return f"File not found: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
        # Truncate very large files
        if len(content) > 15000:
            return content[:15000] + "\n\n[Truncated - file too large]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def search_vault(query: str, max_results: int = 5, vault_path: Path | None = None) -> str:
    """Search across vault files."""
    if vault_path is None:
        vault_path = settings.llm_memory_path

    results = []
    query_lower = query.lower()

    try:
        for md_file in vault_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if query_lower in content.lower():
                    # Find matching lines
                    rel_path = md_file.relative_to(vault_path)
                    matching_lines = []
                    for i, line in enumerate(content.splitlines(), 1):
                        if query_lower in line.lower():
                            matching_lines.append(f"  L{i}: {line[:100]}")
                            if len(matching_lines) >= 3:
                                break

                    results.append(f"**{rel_path}**\n" + "\n".join(matching_lines[:3]))
                    if len(results) >= max_results:
                        break
            except Exception:
                continue

        if not results:
            return f"No results found for: {query}"

        return f"Found {len(results)} matches:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def list_vault_contents(folder: str = "", vault_path: Path | None = None) -> str:
    """List files in a vault directory."""
    if vault_path is None:
        vault_path = settings.llm_memory_path

    target = vault_path / folder if folder else vault_path

    if not target.exists():
        return f"Directory not found: {folder or 'root'}"

    try:
        items = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                items.append(f"[DIR] {item.name}/")
            else:
                size = item.stat().st_size
                items.append(f"      {item.name} ({size:,} bytes)")

        if not items:
            return f"Empty directory: {folder or 'root'}"

        return f"Contents of {folder or 'root'}:\n" + "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"


def get_recent_conversations(hours: int = 24, vault_path: Path | None = None) -> str:
    """Get recent conversation history."""
    if vault_path is None:
        vault_path = settings.llm_memory_path

    conversations_dir = vault_path / "Conversations"
    if not conversations_dir.exists():
        return "No conversations directory found"

    try:
        # Get recent conversation files
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_content = []

        for conv_file in sorted(conversations_dir.glob("*.md"), reverse=True):
            try:
                # Parse date from filename (YYYY-MM-DD.md)
                date_str = conv_file.stem
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date.date() >= cutoff.date():
                    content = conv_file.read_text(encoding="utf-8")
                    recent_content.append(f"## {date_str}\n{content}")
            except (ValueError, Exception):
                continue

            if len(recent_content) >= 3:  # Max 3 days
                break

        if not recent_content:
            return f"No conversations in the last {hours} hours"

        result = "\n\n".join(recent_content)
        # Truncate if too large
        if len(result) > 10000:
            result = result[:10000] + "\n\n[Truncated...]"

        return result
    except Exception as e:
        return f"Error retrieving conversations: {e}"


# Tool definitions for Claude API
VAULT_TOOLS = [
    {
        "name": "read_vault_file",
        "description": "Read a file from the user's Obsidian vault. Use for detailed information not in cached context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within vault (e.g., 'Permanent/memories.md', 'Context/daily.md')",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_vault",
        "description": "Search for content across all vault files. Returns matching files with context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (case-insensitive)"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_vault_contents",
        "description": "List files and folders in a vault directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Folder path (empty string for root)",
                    "default": "",
                }
            },
        },
    },
    {
        "name": "get_recent_conversations",
        "description": "Get recent conversation history with the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Hours of history to retrieve (default 24)",
                    "default": 24,
                }
            },
        },
    },
]


def get_vault_tool_functions() -> dict[str, callable]:
    """Map tool names to functions."""
    return {
        "read_vault_file": read_vault_file,
        "search_vault": search_vault,
        "list_vault_contents": list_vault_contents,
        "get_recent_conversations": get_recent_conversations,
    }


# =============================================================================
# CLAUDE VAULT SESSION
# =============================================================================


class ClaudeVaultSession:
    """Manages Claude API calls with cached vault context."""

    def __init__(
        self,
        vault_path: Path | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cache_ttl: str = "5m",
    ):
        self.vault_path = vault_path or settings.llm_memory_path
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model
        self.cache_ttl = cache_ttl

        # Prefer the real SDK so cache_control markers aren't stripped by
        # the shim. Falls back to the shim (claude -p subprocess) if the
        # real package can't be located or the user disabled real-API use
        # via CLAUDE_VAULT_USE_REAL_API=false.
        self.client = None
        self.using_real_sdk = False
        if self.api_key:
            if getattr(settings, "claude_vault_use_real_api", True):
                try:
                    real = real_anthropic.get()
                    self.client = real.Anthropic(api_key=self.api_key)
                    self.using_real_sdk = True
                except ImportError as exc:
                    logger.warning(
                        "[ClaudeVault] real_anthropic unavailable (%s); "
                        "falling back to shimmed client — cache_control will be dropped.",
                        exc,
                    )
            if self.client is None and HAS_ANTHROPIC:
                self.client = anthropic.Anthropic(api_key=self.api_key)

        # Build initial cached prefix
        self._cached_prefix = build_cached_prefix(self.vault_path)
        self._prefix_built_at = datetime.now()

        # Usage tracking (cumulative)
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Per-request tracking (last request only)
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def _should_refresh_prefix(self) -> bool:
        """Check if cached prefix should be refreshed."""
        age = datetime.now() - self._prefix_built_at
        return age > timedelta(hours=1)

    def refresh_cache(self) -> None:
        """Force refresh of the cached prefix."""
        self._cached_prefix = build_cached_prefix(self.vault_path)
        self._prefix_built_at = datetime.now()

    def _track_usage(self, usage: Any) -> None:
        """Track token usage from response."""
        # Per-request tracking
        self.last_cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
        self.last_cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
        self.last_input_tokens = getattr(usage, "input_tokens", 0)
        self.last_output_tokens = getattr(usage, "output_tokens", 0)

        # Cumulative tracking
        self.total_cache_read_tokens += self.last_cache_read_tokens
        self.total_cache_write_tokens += self.last_cache_write_tokens
        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += self.last_output_tokens

        # Process-wide cache stats (for /api/claude_vault/stats)
        _record_cache_stats(usage)

    def ask(
        self,
        question: str,
        include_recent_context: bool = False,
        context: str = "",
    ) -> str:
        """
        Send a question to Claude with cached vault context.

        Args:
            question: The question to ask
            include_recent_context: If True, includes last hour of conversations
            context: Additional context to include
        """
        if not self.client:
            return "Error: Anthropic client not available (missing API key or library)"

        if self._should_refresh_prefix():
            self.refresh_cache()

        # Build messages
        messages = []

        if include_recent_context:
            recent = get_recent_conversations(hours=1, vault_path=self.vault_path)
            if recent and "No conversations" not in recent:
                messages.append(
                    {"role": "user", "content": f"[Recent conversation context]\n{recent}"}
                )
                messages.append(
                    {"role": "assistant", "content": "I've noted the recent conversation context."}
                )

        if context:
            question = f"{context}\n\n{question}"

        messages.append({"role": "user", "content": question})

        api_start = _time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._cached_prefix,
                messages=messages,
                extra_headers=_request_id_headers(),
            )

            self._track_usage(response.usage)
            in_t = getattr(response.usage, "input_tokens", 0)
            out_t = getattr(response.usage, "output_tokens", 0)
            cache_r = getattr(response.usage, "cache_read_input_tokens", 0)
            cache_w = getattr(response.usage, "cache_creation_input_tokens", 0)
            _record_perf(
                "claude_api", _time.perf_counter() - api_start, success=True,
                model=self.model, input_tokens=in_t, output_tokens=out_t,
            )
            from .claude_bridge import _log_api_cost
            _log_api_cost("ClaudeVault.ask", self.model, in_t, out_t, cache_r, cache_w)
            return response.content[0].text

        except Exception as e:
            _record_perf(
                "claude_api", _time.perf_counter() - api_start, success=False,
                model=self.model, error=str(e),
            )
            return f"Error: {e}"

    def ask_with_tools(
        self,
        task: str,
        max_turns: int = 10,
        additional_tools: list | None = None,
    ) -> str:
        """
        Agentic loop with vault tools for complex tasks.

        Args:
            task: The task to accomplish
            max_turns: Maximum tool-calling turns
            additional_tools: Extra tools beyond vault tools
        """
        if not self.client:
            return "Error: Anthropic client not available (missing API key or library)"

        if self._should_refresh_prefix():
            self.refresh_cache()

        tools = VAULT_TOOLS.copy()
        if additional_tools:
            tools.extend(additional_tools)

        tool_functions = get_vault_tool_functions()

        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            try:
                api_start = _time.perf_counter()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._cached_prefix,
                    tools=tools,
                    messages=messages,
                    extra_headers=_request_id_headers(),
                )

                self._track_usage(response.usage)
                in_t = getattr(response.usage, "input_tokens", 0)
                out_t = getattr(response.usage, "output_tokens", 0)
                cache_r = getattr(response.usage, "cache_read_input_tokens", 0)
                cache_w = getattr(response.usage, "cache_creation_input_tokens", 0)
                _record_perf(
                    "claude_api", _time.perf_counter() - api_start, success=True,
                    model=self.model, input_tokens=in_t, output_tokens=out_t,
                )
                from .claude_bridge import _log_api_cost
                _log_api_cost(
                    f"ClaudeVault.tools[{turn}]", self.model,
                    in_t, out_t, cache_r, cache_w,
                )

                # Process response
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        # Execute tool
                        if tool_name in tool_functions:
                            try:
                                result = tool_functions[tool_name](**tool_input)
                            except Exception as e:
                                result = f"Tool error: {e}"
                        else:
                            result = f"Unknown tool: {tool_name}"

                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": tool_name,
                                "input": tool_input,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                messages.append({"role": "assistant", "content": assistant_content})

                # If no tool calls, return final response
                if not tool_results:
                    for block in response.content:
                        if block.type == "text":
                            return block.text
                    return "Task completed."

                # Add tool results and continue
                messages.append({"role": "user", "content": tool_results})

            except Exception as e:
                _record_perf(
                    "claude_api", _time.perf_counter() - api_start, success=False,
                    model=self.model, error=str(e),
                )
                return f"Error during agentic loop: {e}"

        return "Max turns reached without completion."

    def get_usage_stats(self) -> dict[str, Any]:
        """Get accumulated token usage statistics."""
        return {
            "cache_read_tokens": self.total_cache_read_tokens,
            "cache_write_tokens": self.total_cache_write_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_savings_tokens": int(self.total_cache_read_tokens * 0.9),
            "total_cost_usd": self.get_total_cost(),
        }

    def get_last_request_cost(self) -> float:
        """Calculate the cost of the last API request in USD."""
        # Non-cached input tokens (input_tokens includes everything, subtract cached)
        uncached_input = max(0, self.last_input_tokens - self.last_cache_read_tokens)

        cost = (
            (uncached_input / 1_000_000) * PRICING["input"]
            + (self.last_output_tokens / 1_000_000) * PRICING["output"]
            + (self.last_cache_read_tokens / 1_000_000) * PRICING["cache_read"]
            + (self.last_cache_write_tokens / 1_000_000) * PRICING["cache_write"]
        )
        return round(cost, 6)

    def get_total_cost(self) -> float:
        """Calculate total cost of all requests in this session in USD."""
        uncached_input = max(0, self.total_input_tokens - self.total_cache_read_tokens)

        cost = (
            (uncached_input / 1_000_000) * PRICING["input"]
            + (self.total_output_tokens / 1_000_000) * PRICING["output"]
            + (self.total_cache_read_tokens / 1_000_000) * PRICING["cache_read"]
            + (self.total_cache_write_tokens / 1_000_000) * PRICING["cache_write"]
        )
        return round(cost, 6)

    def format_last_cost(self) -> str:
        """Format the last request cost as a readable string."""
        cost = self.get_last_request_cost()
        if cost < 0.01:
            return f"${cost:.4f}"
        return f"${cost:.2f}"

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_session: ClaudeVaultSession | None = None


def get_vault_session() -> ClaudeVaultSession:
    """Get or create the global vault session."""
    global _session
    if _session is None:
        _session = ClaudeVaultSession()
    return _session


def init_vault_session(vault_path: Path | None = None, **kwargs: Any) -> ClaudeVaultSession:
    """Initialize the global vault session."""
    global _session
    _session = ClaudeVaultSession(vault_path=vault_path, **kwargs)
    return _session


def ask_claude_with_vault(question: str, **kwargs: Any) -> str:
    """Convenience function to ask Claude with vault context."""
    return get_vault_session().ask(question, **kwargs)


def ask_claude_with_vault_tools(task: str, **kwargs: Any) -> str:
    """Convenience function to ask Claude with vault tools."""
    return get_vault_session().ask_with_tools(task, **kwargs)


def get_last_request_cost() -> str:
    """Get the cost of the last Claude API request as a formatted string."""
    return get_vault_session().format_last_cost()


def ask_claude_with_cost(question: str, **kwargs: Any) -> tuple[str, str]:
    """
    Ask Claude and return both the response and the cost.

    Returns:
        Tuple of (response_text, cost_string)
    """
    session = get_vault_session()
    response = session.ask(question, **kwargs)
    cost = session.format_last_cost()
    return response, cost
