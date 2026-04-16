"""
Idea Board Deduplication — embed + cosine dedup for new ideas.

Provides a small, dependency-injected surface the idea-board uses to decide
whether a candidate idea collides with something already on the board:

    - ``combine_text(title, description)`` — canonical embedding input.
    - ``_content_key(title, description)`` — lower/punct/whitespace-normalized
      content key. Useful as a cache key for embeddings and a fast exact-match
      pre-check.
    - ``find_duplicate(title, description, items, ...)`` — returns the best
      ``DedupMatch`` when cosine similarity is at or above the threshold
      (default 0.88) across **active** (To Do, In Progress) plus **recent
      Done** items.

The embedding call is pluggable. Tests inject a fake ``embed_fn`` so no
Ollama traffic happens. When ``embed_fn`` is ``None`` the module lazily
imports ``agent.embeddings.embed_text`` — the lazy import is deliberate:
eagerly importing ``agent.embeddings`` pulls in the whole ``agent``
package tree on test startup and blows past the 10-second acceptance
budget for this file. Tests that want to exercise the fallback patch
``agent.embeddings.embed_text`` directly.

State classification handles both internal Idea state names
(``proposed``/``refining``/``approved``/``executing``/``done``) and the
Jira column names (``To Do``/``In Progress``/``Done``) so the module
works with whichever provider is active.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Inlined (rather than imported from ``agent.embeddings``) so that merely
    importing this module does not drag in the entire ``agent`` package.
    That keeps ``pytest tests/unit/test_idea_dedup_core.py`` under the
    10-second acceptance budget.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))

DEFAULT_THRESHOLD = 0.88
DEFAULT_DONE_WINDOW_DAYS = 30

_TODO_STATES = frozenset({"proposed", "refining", "approved", "To Do"})
_IN_PROGRESS_STATES = frozenset({"executing", "In Progress"})
_DONE_STATES = frozenset({"done", "Done"})
_ACTIVE_STATES = _TODO_STATES | _IN_PROGRESS_STATES

_STATE_PRIORITY: dict[str, int] = {
    "executing": 3, "In Progress": 3,
    "proposed": 2, "refining": 2, "approved": 2, "To Do": 2,
    "done": 1, "Done": 1,
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupMatch:
    """The best existing item that duplicates the candidate.

    Attributes:
        item: The existing board item (typically ``Idea``) that matched.
        score: Cosine similarity in [0.0, 1.0] — always >= the threshold
            used for the lookup.
    """

    item: Any
    score: float


def combine_text(title: str, description: str) -> str:
    """Build the canonical embedding input for an idea.

    Joins title and description with a blank-line separator so the embedder
    treats them as one coherent chunk. Either field may be empty; if both
    are empty the result is an empty string.
    """
    t = (title or "").strip()
    d = (description or "").strip()
    if t and d:
        return f"{t}\n\n{d}"
    return t or d


def _content_key(title: str, description: str) -> str:
    """Normalize (title, description) into a stable, comparable string.

    Lowercases, drops punctuation, collapses whitespace. Used as a cheap
    cache key for embeddings and as a fast exact-duplicate pre-check —
    not as a semantic comparison.
    """
    combined = combine_text(title, description).lower()
    combined = _PUNCT_RE.sub(" ", combined)
    combined = _WHITESPACE_RE.sub(" ", combined).strip()
    return combined


def _get(item: Any, attr: str, default: str = "") -> str:
    value = getattr(item, attr, default)
    return value if isinstance(value, str) else default


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _is_candidate(item: Any, now: datetime, done_window: timedelta) -> bool:
    """Does this item qualify for the dedup comparison pool?

    True for active (To Do / In Progress) items, and for Done items created
    within the ``done_window``. Everything else (vetoed, failed, stale Done)
    is skipped.
    """
    state = _get(item, "state")
    if state in _ACTIVE_STATES:
        return True
    if state in _DONE_STATES:
        dt = _parse_iso(_get(item, "created"))
        if dt is None:
            return False
        # Compare naive-to-naive to avoid tz mismatch when callers pass
        # a naive ``now``; Idea.created is stored naive-local in models.py.
        if dt.tzinfo and not now.tzinfo:
            dt = dt.replace(tzinfo=None)
        elif now.tzinfo and not dt.tzinfo:
            dt = dt.replace(tzinfo=now.tzinfo)
        return (now - dt) <= done_window
    return False


def _tie_break_key(item: Any) -> tuple[int, str, str]:
    """Sort key for ties in cosine similarity.

    Higher-priority state first (In Progress > To Do > Done), then newer
    ``created`` wins, then lexicographically-larger id as the deterministic
    fallback. The outer sort uses ``reverse=True``, so larger tuples win.
    """
    priority = _STATE_PRIORITY.get(_get(item, "state"), 0)
    return (priority, _get(item, "created"), _get(item, "id"))


def find_duplicate(
    title: str,
    description: str,
    items: Iterable[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
    done_window_days: int = DEFAULT_DONE_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> Optional[DedupMatch]:
    """Return the best duplicate match for (title, description).

    Candidate pool is every item in ``items`` that is active (To Do / In
    Progress) or recently Done (within ``done_window_days``). For each
    candidate, embeds the candidate text and the item's text and takes the
    cosine similarity; the highest score at or above ``threshold`` wins.
    Ties break on state priority → most recent ``created`` → item id.

    Args:
        title: Candidate idea title.
        description: Candidate idea description.
        items: Existing board items to compare against. Each must expose
            ``title``, ``description``, ``state``, ``created`` (and ``id``
            for the deterministic tie-break).
        threshold: Minimum cosine similarity to count as a duplicate.
        embed_fn: Optional injected embedding function. Defaults to
            ``agent.embeddings.embed_text`` so production calls Ollama
            while tests pass in a deterministic stub.
        done_window_days: How far back to include Done items.
        now: Override for "now" (useful in tests). Defaults to
            ``datetime.now()``.

    Returns:
        ``DedupMatch`` when a qualifying existing item is found, else None.
    """
    current = now or datetime.now()
    done_window = timedelta(days=done_window_days)

    candidates = [i for i in items if _is_candidate(i, current, done_window)]
    if not candidates:
        return None

    query_text = combine_text(title, description)
    if not query_text:
        return None

    if embed_fn is None:
        # Lazy import so this module does not pull in the full ``agent``
        # package tree. The site remains patchable as
        # ``agent.embeddings.embed_text`` because ``from X import Y``
        # inside a function reads the current module attribute each call.
        from agent.embeddings import embed_text as _default_embed
        embed = _default_embed
    else:
        embed = embed_fn
    query_vec = embed(query_text)
    if not query_vec:
        return None

    scored: list[tuple[float, Any]] = []
    for item in candidates:
        other_text = combine_text(_get(item, "title"), _get(item, "description"))
        if not other_text:
            continue
        other_vec = embed(other_text)
        if not other_vec:
            continue
        score = _cosine(query_vec, other_vec)
        if score >= threshold:
            scored.append((score, item))

    if not scored:
        return None

    scored.sort(key=lambda t: (t[0], *_tie_break_key(t[1])), reverse=True)
    best_score, best_item = scored[0]
    return DedupMatch(item=best_item, score=best_score)
