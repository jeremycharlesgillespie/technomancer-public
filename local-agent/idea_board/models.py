"""
Idea Board Data Model — JSON-backed idea storage with Obsidian sync.

Each idea tracks: title, description, source, category, state, votes,
and a threaded comment discussion. Ideas are persisted to ideas.json
and mirrored to Obsidian vault for searchability.

States: proposed -> approved / vetoed / refining / executing -> done / failed
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _jira_sync_background(idea: Any) -> None:
    """Sync idea to Jira in a background thread (non-blocking)."""
    def _sync():
        try:
            from .jira_sync import is_jira_configured, sync_idea_to_jira
            if is_jira_configured():
                sync_idea_to_jira(idea)
        except Exception as e:
            logger.warning("[JiraSync] Background sync failed: %s", e)
    threading.Thread(target=_sync, daemon=True).start()


# Storage paths
IDEAS_DIR: Path = Path(__file__).parent
IDEAS_FILE: Path = IDEAS_DIR / "ideas.json"
from agent.config import settings
VAULT_IDEAS_DIR: Path = settings.llm_memory_path / "Permanent" / "ideas"

# Thread lock for concurrent JSON access
_lock = threading.Lock()


@dataclass
class Comment:
    """A single comment in an idea's discussion thread.

    Attributes:
        author: Who wrote this — "owner", "llm", or "claude"
        text: The comment content
        timestamp: ISO format timestamp
    """

    author: str
    text: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, str]:
        return {"author": self.author, "text": self.text, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Comment:
        author = data["author"]
        if author == "jeremy":
            author = "owner"
        return cls(author=author, text=data["text"], timestamp=data.get("timestamp", ""))


@dataclass
class Idea:
    """A single improvement idea with votes and discussion.

    Attributes:
        id: Unique identifier (e.g. "idea-042")
        title: Short descriptive title
        description: 2-3 sentence rationale with technical details
        source: What prompted this idea (conversation_analysis, news_analysis, etc.)
        category: Type of improvement (performance, feature, quality, security, ux)
        idea_type: Hierarchy level — "epic", "story", or "task"
        created: ISO timestamp when the idea was created
        state: Current state (proposed, approved, vetoed, refining, executing, done, failed)
        votes: Dict of voter -> vote ("up", "down", "approve", "veto", or None)
        comments: Threaded discussion
        parent_id: For stories/tasks, links to the parent epic or story
        execution_log: Output from Claude Code execution (populated when done/failed)
    """

    id: str
    title: str
    description: str
    source: str = "llm_analysis"
    category: str = "feature"
    idea_type: str = "story"  # "epic", "story", or "task"
    created: str = ""
    state: str = "proposed"
    votes: dict[str, str | None] = field(default_factory=lambda: {"claude": None, "owner": None})
    comments: list[Comment] = field(default_factory=list)
    parent_id: str | None = None
    execution_log: str | None = None
    execution_order: list[str] = field(default_factory=list)
    epic_context: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "category": self.category,
            "idea_type": self.idea_type,
            "created": self.created,
            "state": self.state,
            "votes": self.votes,
            "comments": [c.to_dict() for c in self.comments],
            "parent_id": self.parent_id,
            "execution_log": self.execution_log,
            "execution_order": self.execution_order,
            "epic_context": self.epic_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Idea:
        comments = [Comment.from_dict(c) for c in data.get("comments", [])]
        raw_votes = data.get("votes", {"claude": None, "owner": None})
        if "jeremy" in raw_votes and "owner" not in raw_votes:
            raw_votes["owner"] = raw_votes.pop("jeremy")
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            source=data.get("source", "llm_analysis"),
            category=data.get("category", "feature"),
            idea_type=data.get("idea_type", "story"),
            created=data.get("created", ""),
            state=data.get("state", "proposed"),
            votes=raw_votes,
            comments=comments,
            parent_id=data.get("parent_id"),
            execution_log=data.get("execution_log"),
            execution_order=data.get("execution_order", []),
            epic_context=data.get("epic_context", ""),
        )


# ============================================================================
# JSON PERSISTENCE
# ============================================================================

def load_ideas() -> list[Idea]:
    """Load all ideas from ideas.json. Thread-safe.

    Returns:
        List of Idea objects, or empty list if file doesn't exist
    """
    with _lock:
        if not IDEAS_FILE.exists():
            return []
        try:
            data = json.loads(IDEAS_FILE.read_text(encoding="utf-8"))
            return [Idea.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading ideas: {e}")
            return []


def save_ideas(ideas: list[Idea]) -> None:
    """Save all ideas to ideas.json. Thread-safe.

    Args:
        ideas: List of Idea objects to persist
    """
    with _lock:
        IDEAS_FILE.write_text(
            json.dumps([i.to_dict() for i in ideas], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _next_id(ideas: list[Idea]) -> str:
    """Generate the next idea ID.

    Args:
        ideas: Current list of ideas

    Returns:
        Next ID string like "idea-042"
    """
    if not ideas:
        return "idea-001"
    max_num = max(int(i.id.split("-")[1]) for i in ideas if "-" in i.id)
    return f"idea-{max_num + 1:03d}"


# ============================================================================
# IDEA OPERATIONS
# ============================================================================

def _stopwords() -> set[str]:
    """Common words to ignore when comparing idea similarity."""
    return {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
        "it", "its", "not", "no", "so", "if", "then", "than", "when", "where",
        "how", "what", "which", "who", "whom", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "into", "over", "after",
        "before", "between", "under", "about", "up", "out", "use", "using",
        "add", "implement", "create", "make", "build", "improve", "update",
    }


def _meaningful_words(text: str) -> set[str]:
    """Extract meaningful word stems from text, ignoring stopwords.

    Uses a simple stemming approach: truncate words to 6 chars to catch
    basic inflections (cache/caching, query/queries, response/responses).

    Args:
        text: Input text

    Returns:
        Set of lowercase stemmed meaningful words
    """
    stops = _stopwords()
    words = set()
    for w in text.lower().split():
        if len(w) >= 3 and w not in stops:
            # Simple stem: truncate to 5 chars to group inflections
            # (cache/caching→cach, query/queries→query, response/responses→respo)
            stem = w[:5]
            words.add(stem)
    return words


def _is_duplicate(new_title: str, new_desc: str, existing: Idea) -> bool:
    """Check if a new idea is essentially the same as an existing one.

    Compares both title and description using meaningful word overlap.
    An idea is a duplicate if:
    - Title words overlap > 50%, OR
    - Combined (title + description) meaningful words overlap > 40%

    This catches rephrased duplicates while allowing legitimately
    different ideas about the same topic.

    Args:
        new_title: Title of the new idea
        new_desc: Description of the new idea
        existing: An existing Idea to compare against

    Returns:
        True if the new idea is a duplicate of the existing one
    """
    # Title comparison
    new_title_words = _meaningful_words(new_title)
    existing_title_words = _meaningful_words(existing.title)

    if new_title_words and existing_title_words:
        title_overlap = len(new_title_words & existing_title_words) / max(
            len(new_title_words), len(existing_title_words)
        )
        if title_overlap > 0.5:
            return True

    # Combined title + description comparison (catches rephrased ideas)
    new_combined = _meaningful_words(new_title + " " + new_desc)
    existing_combined = _meaningful_words(existing.title + " " + existing.description)

    if new_combined and existing_combined:
        combined_overlap = len(new_combined & existing_combined) / max(
            len(new_combined), len(existing_combined)
        )
        if combined_overlap > 0.4:
            return True

    return False


def _normalize_description(desc: str) -> str:
    """Ensure WHAT/WHY/HOW/BENEFITS/COST/UNLOCKS sections are on separate lines.

    If the LLM produced all sections inline (on one line), insert newlines
    before each section header so the dashboard renders them cleanly.
    """
    import re

    headers = r"(?:WHAT|WHY|HOW|BENEFITS|COST|UNLOCKS):"
    # Already well-formatted: headers on their own lines
    if re.search(r"\n\s*(?:" + headers[4:], desc):
        return desc
    # Insert double newline before each header (except the first)
    normalized = re.sub(r"\s+(" + headers + r")", r"\n\n\1", desc)
    return normalized.strip()


def add_idea(
    title: str,
    description: str,
    source: str = "llm_analysis",
    category: str = "feature",
    idea_type: str = "story",
    parent_id: str | None = None,
) -> Idea:
    """Add a new idea, deduplicating against existing ideas.

    Checks both title and description for meaningful word overlap.
    Exact-same and closely-rephrased ideas are rejected. Slightly
    different ideas (same topic, different angle) are allowed through.

    Args:
        title: Short descriptive title
        description: Technical rationale
        source: What prompted this idea
        category: Type (performance, feature, quality, security, ux)
        idea_type: Hierarchy level — "epic", "story", or "task"
        parent_id: For stories/tasks, the parent epic or story ID

    Returns:
        The created Idea (or existing one if duplicate detected)
    """
    description = _normalize_description(description)
    ideas = load_ideas()

    # Check against all active ideas (not vetoed/done/failed)
    for existing in ideas:
        if existing.state not in ("vetoed", "done", "failed"):
            if _is_duplicate(title, description, existing):
                logger.info(f"Duplicate idea detected: '{title}' ≈ '{existing.title}'")
                return existing

    idea = Idea(
        id=_next_id(ideas),
        title=title,
        description=description,
        source=source,
        category=category,
        idea_type=idea_type,
        parent_id=parent_id,
    )
    ideas.append(idea)
    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    logger.info(f"Added idea {idea.id}: {title}")
    return idea


def vote(idea_id: str, voter: str, vote_value: str) -> Idea | None:
    """Record a vote on an idea.

    Args:
        idea_id: The idea to vote on
        voter: "owner" or "claude"
        vote_value: "approve", "veto", "up", or "down"

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None

    idea.votes[voter] = vote_value

    # State transitions based on owner's vote (final say)
    if voter == "owner":
        if vote_value == "approve":
            idea.state = "approved"
            # Auto-resolve KAREN complaint when its idea is accepted
            if idea.source == "karen":
                from .karen import resolve_complaint_for_idea
                resolve_complaint_for_idea(idea.id)
        elif vote_value == "veto":
            idea.state = "vetoed"

    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    return idea


def add_comment(idea_id: str, author: str, text: str) -> Idea | None:
    """Add a comment to an idea's discussion thread.

    Args:
        idea_id: The idea to comment on
        author: "owner", "llm", or "claude"
        text: Comment content

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None

    idea.comments.append(Comment(author=author, text=text))

    # If the owner comments on a proposed idea, mark it as refining
    if author == "owner" and idea.state == "proposed":
        idea.state = "refining"

    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    return idea


def mark_executing(idea_id: str) -> Idea | None:
    """Mark an idea as currently being executed by Claude Code.

    Args:
        idea_id: The idea to mark

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None
    idea.state = "executing"
    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    return idea


def mark_done(idea_id: str, execution_log: str) -> Idea | None:
    """Mark an idea as successfully executed.

    Args:
        idea_id: The idea to mark
        execution_log: Output from Claude Code

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None
    idea.state = "done"
    idea.execution_log = execution_log
    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    # Auto-resolve KAREN complaint when its idea completes
    if idea.source == "karen":
        from .karen import resolve_complaint_for_idea
        resolve_complaint_for_idea(idea.id)
    return idea


def mark_failed(idea_id: str, error: str) -> Idea | None:
    """Mark an idea as failed execution.

    Args:
        idea_id: The idea to mark
        error: Error message or output

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None
    idea.state = "failed"
    idea.execution_log = error
    save_ideas(ideas)
    sync_to_obsidian(idea)
    _jira_sync_background(idea)
    return idea


def delete_idea(idea_id: str) -> bool:
    """Permanently remove an idea from the board.

    Args:
        idea_id: The idea to delete

    Returns:
        True if deleted, False if not found
    """
    ideas = load_ideas()
    original_count = len(ideas)
    ideas = [i for i in ideas if i.id != idea_id]
    if len(ideas) == original_count:
        return False
    save_ideas(ideas)

    # Also remove from Obsidian
    obsidian_file = VAULT_IDEAS_DIR / f"{idea_id}.md"
    try:
        obsidian_file.unlink(missing_ok=True)
    except OSError:
        pass

    logger.info(f"Deleted idea {idea_id}")
    return True


def set_execution_order(idea_id: str, order: list[str]) -> Idea | None:
    """Set the execution order for an epic's child stories.

    Args:
        idea_id: The epic idea ID
        order: List of child story IDs in execution order

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None
    idea.execution_order = order
    save_ideas(ideas)
    return idea


def set_epic_context(idea_id: str, context: str) -> Idea | None:
    """Set the epic context narrative for an epic.

    Args:
        idea_id: The epic idea ID
        context: Free-text narrative explaining the big picture

    Returns:
        Updated Idea, or None if not found
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return None
    idea.epic_context = context
    save_ideas(ideas)
    return idea


def get_execution_order(idea_id: str) -> list[str]:
    """Get the execution order for an epic, auto-populating from children if empty.

    If execution_order is not set, builds it from child story IDs in creation order.

    Args:
        idea_id: The epic idea ID

    Returns:
        List of child story IDs in execution order
    """
    ideas = load_ideas()
    idea = next((i for i in ideas if i.id == idea_id), None)
    if not idea:
        return []
    if idea.execution_order:
        return idea.execution_order
    # Auto-populate from children in creation order
    children = [i for i in ideas if i.parent_id == idea_id]
    return [c.id for c in children]


def get_idea(idea_id: str) -> Idea | None:
    """Get a single idea by ID.

    Args:
        idea_id: The idea ID to look up

    Returns:
        The Idea, or None if not found
    """
    ideas = load_ideas()
    return next((i for i in ideas if i.id == idea_id), None)


# ============================================================================
# OBSIDIAN SYNC
# ============================================================================

def sync_to_obsidian(idea: Idea) -> None:
    """Write/update an idea as a markdown file in the Obsidian vault.

    Creates `Permanent/ideas/idea-NNN.md` with status, description,
    and discussion thread. Lets the LLM and user reference ideas in
    conversations and search them in Obsidian.

    Args:
        idea: The Idea to sync
    """
    VAULT_IDEAS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {idea.id}: {idea.title}",
        "",
        f"**Status:** {idea.state}",
        f"**Category:** {idea.category}",
        f"**Source:** {idea.source}",
        f"**Created:** {idea.created[:10]}",
    ]

    if idea.parent_id:
        lines.append(f"**Refinement of:** [[{idea.parent_id}]]")

    lines.extend(["", "## Description", idea.description])

    if idea.comments:
        lines.append("")
        lines.append("## Discussion")
        for c in idea.comments:
            ts = c.timestamp[:16] if c.timestamp else ""
            lines.append(f"- **{c.author}** ({ts}): {c.text}")

    if idea.execution_log:
        lines.append("")
        lines.append("## Execution Log")
        lines.append(f"```\n{idea.execution_log[:2000]}\n```")

    file_path = VAULT_IDEAS_DIR / f"{idea.id}.md"
    file_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# LLM TOOLS — Let the bot query the idea board directly
# ============================================================================

def _format_idea_summary(idea: Idea) -> str:
    """Format a single idea as a concise text summary for the LLM."""
    votes = idea.votes
    comment_count = len(idea.comments)
    return (
        f"**{idea.id}**: {idea.title}\n"
        f"  State: {idea.state} | Type: {idea.idea_type} | Category: {idea.category}\n"
        f"  Votes — Claude: {votes.get('claude') or 'none'}, Owner: {votes.get('owner') or 'none'}\n"
        f"  Comments: {comment_count} | Created: {idea.created[:10]}"
    )


def list_ideas_for_llm(state: str = "") -> str:
    """List ideas from the idea board, optionally filtered by state.

    Args:
        state: Filter by state (proposed, approved, vetoed, refining, executing, done, failed).
               Empty string returns all active ideas.

    Returns:
        Formatted text summary of matching ideas.
    """
    ideas = load_ideas()
    if state:
        ideas = [i for i in ideas if i.state == state]
    else:
        ideas = [i for i in ideas if i.state not in ("vetoed", "done", "failed")]

    if not ideas:
        label = f"in state '{state}'" if state else "active"
        return f"No {label} ideas on the board."

    lines = [f"**Idea Board** — {len(ideas)} idea(s):\n"]
    for idea in ideas:
        lines.append(_format_idea_summary(idea))
        lines.append("")
    return "\n".join(lines)


def get_idea_detail_for_llm(idea_id: str) -> str:
    """Get full details of a specific idea including description and comments.

    Args:
        idea_id: The idea ID (e.g. "idea-042")

    Returns:
        Detailed text representation of the idea.
    """
    idea = get_idea(idea_id)
    if not idea:
        return f"Idea '{idea_id}' not found."

    lines = [
        f"# {idea.id}: {idea.title}",
        f"State: {idea.state} | Type: {idea.idea_type} | Category: {idea.category}",
        f"Source: {idea.source} | Created: {idea.created[:10]}",
        "",
        "## Description",
        idea.description,
    ]

    if idea.parent_id:
        lines.append(f"\nParent: {idea.parent_id}")

    if idea.comments:
        lines.append("\n## Discussion")
        for c in idea.comments:
            ts = c.timestamp[:16] if c.timestamp else ""
            lines.append(f"- **{c.author}** ({ts}): {c.text}")

    if idea.execution_log:
        lines.append(f"\n## Execution Log\n{idea.execution_log[:1000]}")

    return "\n".join(lines)


def get_idea_board_tools() -> list:
    """Return tools for the LLM to query the idea board."""
    from agent.core import create_tool

    return [
        create_tool(
            "list_ideas",
            "List ideas from the idea board. Returns all active ideas by default, or filter by state.",
            {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Filter by state: proposed, approved, vetoed, refining, executing, done, failed. Leave empty for all active ideas.",
                    },
                },
                "required": [],
            },
            lambda state="": list_ideas_for_llm(state),
        ),
        create_tool(
            "get_idea",
            "Get full details of a specific idea including description, votes, and discussion thread.",
            {
                "type": "object",
                "properties": {
                    "idea_id": {
                        "type": "string",
                        "description": "The idea ID, e.g. 'idea-042'",
                    },
                },
                "required": ["idea_id"],
            },
            lambda idea_id: get_idea_detail_for_llm(idea_id),
        ),
    ]
