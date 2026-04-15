"""
Idea Generator — Hourly analysis of news, conversations, errors, and
performance data to suggest improvement ideas for the Technomancer project.

Runs as an async background task inside the bot process. Uses its own
dedicated Agent instance — completely isolated from the main bot agent
to prevent any interaction leakage.

Inputs:
    - ALL RSS feeds (13 sources, full articles, not just the 1 sent to Discord)
    - Recent conversation logs from Obsidian vault
    - Crash log (recent errors)
    - Request profiling data
    - Existing ideas (for dedup and refinement awareness)

Schedule:
    - Every hour (5 minutes after news digest)
    - On-demand via 'idea' Discord command
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths for input data
from .config import settings
from .memory_system import get_memory_system
from .perf_monitor import get_monitor as get_perf_monitor

try:
    from board import get_provider as _get_board_provider

    def load_ideas():  # type: ignore[no-redef]
        return _get_board_provider().load_all()

    def add_idea(  # type: ignore[no-redef]
        title,
        description,
        source="llm_analysis",
        category="feature",
        idea_type="story",
        parent_id=None,
    ):
        return _get_board_provider().add(
            title=title,
            description=description,
            source=source,
            category=category,
            idea_type=idea_type,
            parent_id=parent_id,
        )

    def set_execution_order(item_id, order):  # type: ignore[no-redef]
        return _get_board_provider().set_execution_order(item_id, order)

    def set_epic_context(item_id, context):  # type: ignore[no-redef]
        return _get_board_provider().set_epic_context(item_id, context)

except ImportError:  # board package unavailable in some contexts
    load_ideas = None  # type: ignore[assignment]
    add_idea = None  # type: ignore[assignment]
    set_execution_order = None  # type: ignore[assignment]
    set_epic_context = None  # type: ignore[assignment]

VAULT_PATH: Path = settings.llm_memory_path
CRASH_LOG: Path = VAULT_PATH / "Permanent" / "crash_log.md"
HOURLY_CONTEXT: Path = VAULT_PATH / "Context" / "hourly.md"
PROFILING_FILE: Path = Path(__file__).parent.parent / "profiling" / "requests.jsonl"
COVERAGE_FILE: Path = Path(__file__).parent.parent / "profiling" / "coverage.json"
REPO_ROOT: Path = Path(__file__).parent.parent

# Schedule — hourly, 5 minutes after each hour (after news digest)
OFFSET_AFTER_NEWS_MINUTES: int = 5
MAX_PROPOSED_IDEAS: int = 10  # skip generation if backlog exceeds this

# Prompt for the LLM
IDEA_PROMPT = """You are an improvement analyst for the Technomancer project — a Discord bot
framework with Ollama LLM, Claude API integration, memory system, news digest,
and developer learning tools. The owner is the project owner (a Senior Software Engineer).

Analyze the following inputs and suggest 1-3 concrete improvement ideas.
Think in terms of FULL LIFECYCLE — don't just suggest adding a metric or a data
collection layer.  Every idea should describe the complete value chain from data
collection through to the behaviour change or user-visible outcome.

Each idea can be an "epic" (a multi-part initiative) or a "story" (a standalone
deliverable).  Epics should list the 2-4 stories needed to deliver end-to-end value.

For each idea, output a JSON object with these fields:
- "title": Short descriptive title (under 80 chars)
- "idea_type": "epic" if it requires multiple stories to deliver value, "story" if standalone
- "description": A structured description with each section on its OWN LINE separated by blank lines.
    Use EXACTLY this format with newlines between sections:
    "WHAT: <what to build or change>\n\nWHY: <what problem it solves>\n\nHOW: <implementation approach — files, patterns, libraries>\n\nFULL LIFECYCLE: <describe the complete value chain: data collection → analysis → action → user-visible outcome. What consumes this data? What behaviour changes? How does the user see the improvement?>\n\nBENEFITS: <how it helps the project owner — saves time, improves quality, etc.>\n\nCOST: <resource impact — CPU/GPU/disk/API costs, or 'Minimal'>\n\nUNLOCKS: <what new capabilities become possible>"
    CRITICAL: Each section MUST start on a new line. Do NOT put all sections on one line.
- "stories": (only for epics) A JSON array of 2-4 story titles that together deliver the full lifecycle. Each story should be independently implementable and testable. Example: ["Collect engagement data", "Build ranking algorithm from engagement", "Auto-filter low-engagement sources"]
- "category": One of: performance, feature, quality, security, ux
- "source": Which input prompted this (news_analysis, conversation_analysis, error_analysis, performance_analysis)

RULES:
- THINK END-TO-END: Don't suggest "add tracking for X" without also describing what consumes that tracking data and what changes as a result.  Layer 1 (data collection) is useless without Layer 2 (analysis) and Layer 3 (action).
- Be specific and actionable — not vague suggestions
- Reference specific files, functions, or metrics when possible
- Focus on things that would genuinely help {owner_name} as a Sr. Software Engineer
- Consider the tech stack: Python, Discord.py, Ollama, Claude API, Obsidian
- Consider the hardware: nvidia 5080 GPU, 96GB RAM, Windows 11
- Don't suggest things already in the EXISTING IDEAS list
- Don't suggest things the CODEBASE already has (check the file list below)
- Each description should be a full paragraph, NOT just 1-2 sentences

Output ONLY a JSON array of idea objects. No other text.

--- CODEBASE (files that already exist — don't suggest features we already have) ---
{codebase}

--- TECH NEWS (what's happening in the industry) ---
{news}

--- RECENT CONVERSATIONS (what the user has been working on) ---
{conversations}

--- RECENT ERRORS (things that broke) ---
{errors}

--- PERFORMANCE DATA (response times and bottlenecks) ---
{performance}

--- EXISTING IDEAS (don't duplicate these) ---
{existing}
"""


# Prompt for signal-driven epic synthesis (idea-194)
SYNTHESIS_PROMPT = """You are an improvement analyst for the Technomancer project — a Discord bot
framework with Ollama LLM, Claude API integration, memory system, news digest,
and developer learning tools.

Analyze these system signals and synthesize exactly ONE epic with 1-3 small,
focused stories.  The epic should address the most impactful opportunity found
in the signals.

Output a JSON object (NOT an array) with these fields:
- "title": Epic title (under 80 chars)
- "epic_context": 2-3 sentences explaining the big picture — what problem this
  epic solves and what the end state looks like when all stories are done.
- "category": One of: performance, feature, quality, security, ux
- "source": Which signal most influenced this — one of: error_analysis,
  performance_analysis, conversation_analysis, coverage_analysis, recent_changes
- "stories": Array of 1-3 story objects, each with:
    - "title": Story title (under 80 chars)
    - "description": Structured with WHAT/WHY/HOW sections.
      Use this format:
      "WHAT: <what to build>\\n\\nWHY: <problem it solves>\\n\\nHOW: <implementation approach>\\n\\nFiles to modify: <comma-separated file paths>"

RULES:
- Each story must be independently implementable and testable
- Stories should be small: 1-2 files changed, under 200 lines of new code
- The epic must deliver end-to-end value — not just data collection without action
- Be specific: name files, functions, and concrete changes
- Don't suggest things already in the EXISTING IDEAS list
- Don't suggest things the CODEBASE already has
- Focus on what would genuinely help the project owner (a Sr. Software Engineer)

--- COLLECTED SIGNALS ---
{signals}

--- CODEBASE (files that already exist — don't suggest features we already have) ---
{codebase}

--- EXISTING IDEAS (don't duplicate these) ---
{existing}

Output ONLY a JSON object. No other text.
"""


def _load_codebase_summary() -> str:
    """List all Python files in agent/ with their first docstring line.

    This tells the LLM what already exists so it doesn't suggest
    features we've already built.

    Returns:
        Formatted file listing with descriptions
    """
    agent_dir = Path(__file__).parent
    lines = []
    for f in sorted(agent_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        # Read first docstring line
        desc = ""
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if '"""' in content:
                doc_start = content.index('"""') + 3
                doc_end = content.index('"""', doc_start)
                first_line = content[doc_start:doc_end].strip().split("\n")[0]
                desc = f" — {first_line}"
        except (ValueError, OSError):
            pass
        lines.append(f"- {f.name}{desc}")

    # Also list idea_board files
    board_dir = Path(__file__).parent.parent / "idea_board"
    if board_dir.exists():
        for f in sorted(board_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            lines.append(f"- idea_board/{f.name}")

    return "\n".join(lines)


async def _load_news_articles() -> str:
    """Fetch ALL articles from ALL RSS feeds (not just the 1 picked for Discord).

    Returns:
        Formatted string of recent news articles
    """
    try:
        from .news_digest import fetch_all_news

        articles = await fetch_all_news()

        if not articles:
            return "No recent news available."

        lines = []
        for article in articles[:30]:  # Cap at 30 to keep prompt manageable
            lines.append(f"- [{article.get('source', '')}] {article.get('title', '')}")
            summary = article.get("summary", "")[:200]
            if summary:
                lines.append(f"  {summary}")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Failed to fetch news for idea generation: {e}")
        return "News unavailable."


def _load_conversations() -> str:
    """Load recent conversation context from the vault.

    Returns:
        Recent conversation summary text
    """
    if HOURLY_CONTEXT.exists():
        try:
            return HOURLY_CONTEXT.read_text(encoding="utf-8")[:3000]
        except OSError:
            pass
    return "No recent conversations."


def _load_errors() -> str:
    """Load recent crash log entries.

    Returns:
        Last 2000 chars of the crash log
    """
    if CRASH_LOG.exists():
        try:
            content = CRASH_LOG.read_text(encoding="utf-8")
            return content[-2000:] if len(content) > 2000 else content
        except OSError:
            pass
    return "No recent errors."


def _load_performance() -> str:
    """Load recent profiling data summary.

    Returns:
        Summary of last 20 request profiles
    """
    if not PROFILING_FILE.exists():
        return "No profiling data yet."

    try:
        lines = PROFILING_FILE.read_text(encoding="utf-8").strip().split("\n")
        recent = lines[-20:]
        summaries = []
        for line in recent:
            try:
                p = json.loads(line)
                summaries.append(
                    f"- {p.get('total_seconds', 0):.1f}s | "
                    f"{p.get('llm_summary', {}).get('total_calls', 0)} LLM calls | "
                    f"type={p.get('classification', {}).get('question_type', '?')} | "
                    f"\"{p.get('message', '')[:60]}\""
                )
            except json.JSONDecodeError:
                continue
        return "\n".join(summaries) if summaries else "No profiling data."
    except OSError:
        return "No profiling data."


def _load_existing_ideas() -> str:
    """Load titles of existing ideas for dedup.

    Returns:
        Bullet list of existing idea titles
    """
    try:
        if load_ideas is None:
            return "No existing ideas."
        ideas = load_ideas()
        if not ideas:
            return "No existing ideas."
        return "\n".join(
            f"- [{i.state}] {i.title}" for i in ideas
            if i.state not in ("vetoed", "failed")
        )
    except Exception:
        return "No existing ideas."


# ---------------------------------------------------------------------------
# Signal Collector — gathers structured improvement signals from all sources
# ---------------------------------------------------------------------------

# Timestamp pattern in crash_log.md entries: "## 2026-04-14 16:30:00"
_CRASH_TS_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _collect_recent_errors(since: datetime | None = None) -> str:
    """Extract crash log entries from the last hour.

    Parses crash_log.md for entries with timestamps, keeps only those
    within the time window.

    Args:
        since: Cutoff datetime (defaults to 1 hour ago).

    Returns:
        Formatted string of recent error entries, or a no-data message.
    """
    if since is None:
        since = datetime.now() - timedelta(hours=1)

    if not CRASH_LOG.exists():
        return "No crash log found."

    try:
        content = CRASH_LOG.read_text(encoding="utf-8")
    except OSError:
        return "Could not read crash log."

    # Split into entries by "## " heading
    entries = re.split(r"(?=^## )", content, flags=re.MULTILINE)
    recent: list[str] = []
    for entry in entries:
        match = _CRASH_TS_RE.match(entry)
        if not match:
            continue
        try:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts >= since:
            # Keep first 500 chars of each entry to stay concise
            recent.append(entry[:500].strip())

    if not recent:
        return "No errors in the last hour."
    return "\n\n".join(recent)


def _collect_slow_operations() -> str:
    """Find endpoints where p95 latency exceeds 5 seconds.

    Reads from the global PerfMonitor singleton.

    Returns:
        Formatted string listing slow endpoints and their stats.
    """
    monitor = get_perf_monitor()
    all_stats = monitor.get_endpoint_stats()
    if all_stats.get("calls", 0) == 0:
        return "No performance data recorded."

    # Check each endpoint individually
    with monitor._lock:
        endpoints = sorted(set(r.endpoint for r in monitor._records))

    slow: list[str] = []
    for ep in endpoints:
        stats = monitor.get_endpoint_stats(ep)
        p95 = stats.get("p95_latency", 0)
        if p95 > 5.0:
            slow.append(
                f"- {ep}: p95={p95:.1f}s, avg={stats['avg_latency']:.1f}s, "
                f"{stats['calls']} calls, {stats['failures']} failures"
            )

    if not slow:
        return "No slow operations (all endpoints p95 < 5s)."
    return "Slow endpoints (p95 > 5s):\n" + "\n".join(slow)


def _collect_conversation_topics(since: datetime | None = None) -> str:
    """Extract recent conversation topics from the memory system.

    Summarises what users have been asking about in the last hour.

    Args:
        since: Cutoff datetime (defaults to 1 hour ago).

    Returns:
        Formatted string of recent user messages.
    """
    if since is None:
        since = datetime.now() - timedelta(hours=1)

    try:
        mem = get_memory_system()
        recent = [
            e for e in mem.recent_conversations
            if e.timestamp >= since
        ]
    except Exception:
        # Memory system may not be initialised (e.g. in tests)
        return "Memory system not available."

    if not recent:
        return "No conversations in the last hour."

    lines: list[str] = []
    for entry in recent[-20:]:  # Cap at 20 most recent
        # Just the user message — enough for topic extraction
        lines.append(f"- [{entry.user}] {entry.message[:150]}")
    return "\n".join(lines)


def _collect_coverage_gaps() -> str:
    """Find modules with less than 50% test coverage.

    Reads from profiling/coverage.json (generated by ``make test-cov``).

    Returns:
        Formatted string listing under-covered modules.
    """
    if not COVERAGE_FILE.exists():
        return "No coverage data (run `make test-cov` to generate)."

    try:
        data = json.loads(COVERAGE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Could not parse coverage data."

    files_data = data.get("files", {})
    if not files_data:
        return "No per-file coverage data."

    gaps: list[str] = []
    for filepath, info in sorted(files_data.items()):
        summary = info.get("summary", {})
        pct = summary.get("percent_covered", 100)
        stmts = summary.get("num_statements", 0)
        if pct < 50 and stmts > 10:  # Skip tiny files
            gaps.append(f"- {filepath.replace(chr(92), '/')}: {pct:.0f}% ({stmts} statements)")

    if not gaps:
        return "All modules above 50% coverage."
    return "Low coverage modules (< 50%):\n" + "\n".join(gaps[:15])


def _collect_recent_changes() -> str:
    """List files changed in the last hour via git log.

    Returns:
        Formatted string of recently modified files and their commit messages.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--since=1 hour ago", "--name-only", "--pretty=format:%h %s"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "Could not read git history."

    if not output:
        return "No commits in the last hour."

    # Deduplicate file paths, keep commit messages
    lines = output.split("\n")
    commits: list[str] = []
    files_seen: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Commit lines start with a short hash (hex chars + space)
        if re.match(r"^[0-9a-f]+ ", line):
            commits.append(f"- {line}")
        elif line not in files_seen:
            files_seen.add(line)

    parts: list[str] = []
    if commits:
        parts.append("Commits:\n" + "\n".join(commits[:10]))
    if files_seen:
        parts.append("Changed files:\n" + "\n".join(f"- {f}" for f in sorted(files_seen)[:15]))
    return "\n".join(parts) if parts else "No recent changes."


def _collect_pending_ideas() -> str:
    """List ideas that are approved or executing but not yet done.

    These represent work the system knows about but hasn't completed.

    Returns:
        Formatted string of pending idea titles and states.
    """
    try:
        if load_ideas is None:
            return "Could not load idea board."
        ideas = load_ideas()
    except Exception:
        return "Could not load idea board."

    if not ideas:
        return "Idea board is empty."

    pending = [i for i in ideas if i.state in ("approved", "executing", "refining")]
    if not pending:
        return "No approved/executing ideas pending."

    lines: list[str] = []
    for idea in pending[:15]:
        lines.append(f"- [{idea.state}] {idea.id}: {idea.title}")
    return "\n".join(lines)


def collect_signals(since: datetime | None = None) -> str:
    """Collect improvement signals from all system sources.

    Gathers data from crash logs, performance metrics, conversations,
    test coverage, git history, and the idea board, then formats them
    into a structured string ready for LLM consumption.

    This is the main entry point used by the idea synthesis step
    (idea-194) to build context for the LLM prompt.

    Args:
        since: Cutoff datetime for time-windowed signals (defaults to 1 hour ago).

    Returns:
        A multi-section formatted string with all collected signals.
    """
    if since is None:
        since = datetime.now() - timedelta(hours=1)

    sections = [
        ("RECENT ERRORS (last hour)", _collect_recent_errors(since)),
        ("SLOW OPERATIONS", _collect_slow_operations()),
        ("CONVERSATION TOPICS (last hour)", _collect_conversation_topics(since)),
        ("TEST COVERAGE GAPS", _collect_coverage_gaps()),
        ("RECENTLY CHANGED FILES (last hour)", _collect_recent_changes()),
        ("PENDING IDEAS (approved/executing)", _collect_pending_ideas()),
    ]

    parts: list[str] = []
    for title, content in sections:
        parts.append(f"### {title}\n{content}")

    return "\n\n".join(parts)


def _parse_ideas(response: str) -> list[dict[str, str]]:
    """Parse the LLM's JSON response into idea dicts.

    Args:
        response: Raw LLM output

    Returns:
        List of validated idea dicts
    """
    # Try to find JSON array in the response
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return []

    try:
        ideas = json.loads(raw)
        if not isinstance(ideas, list):
            return []

        valid = []
        for idea in ideas:
            if isinstance(idea, dict) and "title" in idea and "description" in idea:
                parsed: dict[str, Any] = {
                    "title": str(idea["title"])[:100],
                    "description": str(idea["description"])[:2000],
                    "category": str(idea.get("category", "feature")),
                    "source": str(idea.get("source", "llm_analysis")),
                }
                # Preserve epic-related fields when present
                if "idea_type" in idea:
                    parsed["idea_type"] = str(idea["idea_type"])
                if "stories" in idea and isinstance(idea["stories"], list):
                    parsed["stories"] = idea["stories"]
                if "epic_context" in idea:
                    parsed["epic_context"] = str(idea["epic_context"])[:2000]
                valid.append(parsed)
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_epic_response(response: str) -> dict[str, Any] | None:
    """Parse the LLM's JSON response into an epic dict with stories.

    Expects a JSON *object* (not array) with title, epic_context,
    category, source, and stories fields.

    Args:
        response: Raw LLM output

    Returns:
        Validated epic dict, or None if parsing fails
    """
    # Try markdown code block first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Fall back to bare JSON object
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return None

    try:
        epic = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(epic, dict) or "title" not in epic:
        return None

    # Normalise stories — accept list of dicts or list of strings
    stories = epic.get("stories", [])
    if not isinstance(stories, list):
        stories = []

    valid_stories: list[dict[str, str]] = []
    for s in stories:
        if isinstance(s, dict) and s.get("title", "").strip():
            valid_stories.append({
                "title": str(s["title"]).strip()[:100],
                "description": str(s.get("description", f"Story under epic: {epic['title']}"))[:2000],
            })
        elif isinstance(s, str) and s.strip():
            valid_stories.append({
                "title": s.strip()[:100],
                "description": f"Story under epic: {epic['title']}",
            })
    epic["stories"] = valid_stories[:3]

    return epic


async def synthesize_epic(signals: str, agent: Any) -> dict[str, Any] | None:
    """Synthesize one epic with stories from collected signals using the LLM.

    This is the core synthesis step of the idea generation pipeline:
    collected signals in, structured epic + stories out on the idea board.

    Steps:
        1. Build prompt from signals + codebase summary + existing ideas
        2. Call Ollama via the agent
        3. Parse the structured JSON response
        4. Create epic via add_idea (dedup handled automatically)
        5. Create child stories under the epic
        6. Set execution_order and epic_context on the epic

    Args:
        signals: Formatted signal string from collect_signals()
        agent: An isolated Agent instance for LLM calls

    Returns:
        Dict with epic_id, epic_title, story_ids, category, source — or
        None if nothing was created (LLM failure, parse failure, or dedup).
    """
    codebase = _load_codebase_summary()
    existing = _load_existing_ideas()

    prompt = SYNTHESIS_PROMPT.format(
        signals=signals,
        codebase=codebase,
        existing=existing,
    )

    # Call Ollama via the isolated agent
    try:
        response = await asyncio.to_thread(agent.run, prompt)
    except Exception as e:
        logger.error("[IdeaGen] LLM synthesis failed: %s", e)
        return None

    # Parse the response
    epic_data = _parse_epic_response(response)
    if not epic_data:
        logger.info("[IdeaGen] No valid epic parsed from LLM response.")
        return None

    # Record existing IDs so we can detect dedup from add_idea
    existing_ids = set()
    if load_ideas is not None:
        try:
            existing_ids = {i.id for i in load_ideas()}
        except Exception:
            pass

    # Create the epic
    epic_desc = epic_data.get("epic_context", epic_data.get("description", ""))
    epic = add_idea(
        title=str(epic_data["title"])[:100],
        description=str(epic_desc)[:2000],
        source=str(epic_data.get("source", "signal_analysis")),
        category=str(epic_data.get("category", "feature")),
        idea_type="epic",
    )

    # If add_idea returned an existing idea (dedup), skip story creation
    if epic.id in existing_ids:
        logger.info(
            "[IdeaGen] Duplicate epic skipped: '%s' ≈ '%s'",
            epic_data["title"],
            epic.title,
        )
        return None

    # Set epic_context
    epic_context = str(epic_data.get("epic_context", ""))[:2000]
    if epic_context:
        set_epic_context(epic.id, epic_context)

    # Create child stories
    story_ids: list[str] = []
    for story_data in epic_data.get("stories", []):
        title = story_data.get("title", "").strip()
        if not title:
            continue

        story = add_idea(
            title=title[:100],
            description=str(story_data.get("description", f"Story under epic: {epic.title}"))[:2000],
            source=str(epic_data.get("source", "signal_analysis")),
            category=str(epic_data.get("category", "feature")),
            idea_type="story",
            parent_id=epic.id,
        )
        # Only track if it's a genuinely new story (not deduped)
        if story.id not in existing_ids:
            story_ids.append(story.id)

    # Set execution order on the epic
    if story_ids:
        set_execution_order(epic.id, story_ids)

    logger.info(
        "[IdeaGen] Synthesized epic %s: %s with %d stories",
        epic.id,
        epic.title,
        len(story_ids),
    )

    return {
        "epic_id": epic.id,
        "epic_title": epic.title,
        "story_ids": story_ids,
        "category": epic_data.get("category", "feature"),
        "source": epic_data.get("source", "signal_analysis"),
    }


async def generate_ideas(agent: Any) -> list[dict[str, str]]:
    """Run one idea generation cycle using the provided agent.

    Collects system signals, synthesizes an epic with stories via the
    LLM, and creates them on the idea board.

    IMPORTANT: The agent passed here MUST be a dedicated instance,
    not the main bot agent. This prevents any interaction leakage.

    Args:
        agent: An isolated Agent instance for idea generation

    Returns:
        List of idea summary dicts (one per epic created)
    """
    logger.info("[IdeaGen] Starting idea generation cycle...")

    # Collect signals from all system sources (idea-193)
    signals = collect_signals()

    # Synthesize an epic + stories from the signals (idea-194)
    result = await synthesize_epic(signals, agent)
    if not result:
        logger.info("[IdeaGen] No new ideas generated this cycle.")
        return []

    # Return a list for compatibility with _notify_discord
    created = [{
        "title": result["epic_title"],
        "category": result["category"],
        "source": result["source"],
        "story_count": len(result["story_ids"]),
    }]

    logger.info("[IdeaGen] Generated epic %s with %d stories.",
                result["epic_id"], len(result["story_ids"]))
    return created


def _count_proposed_ideas() -> int:
    """Count unreviewed proposed ideas on the board.

    Returns:
        Number of ideas with state == "proposed".  Returns 0 if the
        idea board is unavailable (import missing, file unreadable, etc.).
    """
    try:
        if load_ideas is None:
            return 0
        ideas = load_ideas()
        return sum(1 for i in ideas if i.state == "proposed")
    except Exception:
        return 0


def _seconds_until_next_run() -> float:
    """Calculate seconds until the next hourly run.

    Target: OFFSET_AFTER_NEWS_MINUTES past each hour (default :05).
    If the current time is before :05 this hour, wait until :05.
    Otherwise wait until :05 next hour.

    Returns:
        Seconds to sleep (minimum 60 to avoid tight loops).
    """
    now = datetime.now()
    target = now.replace(minute=OFFSET_AFTER_NEWS_MINUTES, second=0, microsecond=0)
    if now >= target:
        target += timedelta(hours=1)
    return max((target - now).total_seconds(), 60)


async def _notify_discord(client: Any, ideas: list[dict[str, str]]) -> None:
    """Send a notification to Discord #claude-code with generated idea summary.

    Args:
        client: Discord client instance (must be connected).
        ideas: List of idea summary dicts from generate_ideas().
    """
    try:
        channel = None
        for guild in client.guilds:
            for ch in guild.text_channels:
                if ch.name == "claude-code":
                    channel = ch
                    break
            if channel:
                break

        if not channel:
            logger.warning("[IdeaGen] #claude-code channel not found")
            return

        lines = [f"**[IdeaGen] {len(ideas)} new epic(s) generated:**"]
        for idea in ideas:
            cat = idea.get("category", "")
            story_count = idea.get("story_count", 0)
            lines.append(f"- [{cat}] {idea['title']} ({story_count} stories)")
        lines.append("\n<http://localhost:8322>")

        await channel.send("\n".join(lines))
    except Exception as e:
        logger.warning("[IdeaGen] Discord notification failed: %s", e)


async def idea_generation_loop(client: Any, agent: Any) -> None:
    """Background loop that generates ideas once per hour.

    Runs OFFSET_AFTER_NEWS_MINUTES past each hour (after the news
    digest). Skips the cycle when the idea board already has more
    than MAX_PROPOSED_IDEAS unreviewed proposed ideas.

    Args:
        client: Discord client instance for sending notifications.
        agent: An isolated Agent instance (NOT the main bot agent).
    """
    logger.info(
        "[IdeaGen] Started — hourly, %d min past each hour",
        OFFSET_AFTER_NEWS_MINUTES,
    )

    while True:
        try:
            wait = _seconds_until_next_run()
            logger.info("[IdeaGen] Next run in %.0f min", wait / 60)
            await asyncio.sleep(wait)

            # Check backlog before generating
            proposed = _count_proposed_ideas()
            if proposed > MAX_PROPOSED_IDEAS:
                logger.info(
                    "[IdeaGen] Skipping — %d proposed ideas on board (limit %d)",
                    proposed,
                    MAX_PROPOSED_IDEAS,
                )
                await asyncio.sleep(120)
                continue

            created = await generate_ideas(agent)
            if created:
                await _notify_discord(client, created)

            # Sleep past the trigger window to avoid double-fire
            await asyncio.sleep(120)

        except Exception as e:
            logger.error("[IdeaGen] Loop error: %s", e)
            await asyncio.sleep(300)


def start_idea_generator(client: Any, agent: Any) -> None:
    """Start the idea generation background task.

    Args:
        client: Discord client instance for notifications.
        agent: An isolated Agent instance (NOT the main bot agent).
    """
    from .task_manager import create_monitored_task

    create_monitored_task(idea_generation_loop(client, agent), "idea-generator", critical=True)
    logger.info("[IdeaGen] Background task started (hourly)")
