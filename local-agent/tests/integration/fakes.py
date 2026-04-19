"""Test doubles for BoardProvider used by AIM tick-loop integration tests.

``FakeJiraProvider`` is an in-memory ``BoardProvider`` implementation keyed
by item id. It satisfies the ``board.provider.BoardProvider`` Protocol so it
can be dropped into any code path that calls ``board.get_provider()`` —
letting tests exercise AIM's tick loop without a real Jira round-trip or
the disk-backed ``LocalProvider``.

Typical use::

    provider = FakeJiraProvider()
    item = provider.add(title="t", description="d")
    provider.update_state(item.id, "executing")

``update_state`` is a test-only convenience that maps directly onto the
state-transition methods (``mark_executing`` / ``mark_done`` / ``mark_failed``)
so individual tests don't need to remember which one to call.
"""

from __future__ import annotations

import copy
import itertools
from datetime import datetime
from typing import Iterable

from board.provider import BoardItem, Comment, parse_marker
from idea_board.models import Comment as IdeaComment
from idea_board.models import Idea


class FakeJiraProvider:
    """In-memory BoardProvider backed by a dict keyed by item id.

    Mirrors ``LocalProvider`` behavior closely enough for tick-loop tests
    but without touching disk, Obsidian, or the real Jira API. Returned
    items are deep-copied so callers can't mutate the store by accident.
    """

    def __init__(self, items: Iterable[BoardItem] | None = None) -> None:
        self._items: dict[str, Idea] = {}
        self._id_counter = itertools.count(1)
        if items:
            for item in items:
                self._items[item.id] = copy.deepcopy(item)

    # --- Internal helpers ---------------------------------------------

    def _next_id(self) -> str:
        while True:
            candidate = f"FAKE-{next(self._id_counter)}"
            if candidate not in self._items:
                return candidate

    def _clone(self, item: Idea | None) -> Idea | None:
        return copy.deepcopy(item) if item is not None else None

    # --- Reads --------------------------------------------------------

    def load_all(self) -> list[BoardItem]:
        return [copy.deepcopy(i) for i in self._items.values()]

    def load_active(self) -> list[BoardItem]:
        active = {"proposed", "refining", "approved", "executing"}
        return [copy.deepcopy(i) for i in self._items.values() if i.state in active]

    def list_by_state(self, state: str) -> list[BoardItem]:
        return [copy.deepcopy(i) for i in self._items.values() if i.state == state]

    def get(self, item_id: str) -> BoardItem | None:
        return self._clone(self._items.get(item_id))

    def list_ideas_for_llm(self, state: str = "") -> str:
        items = self._items.values()
        if state:
            items = [i for i in items if i.state == state]
        return "\n".join(f"{i.id}: {i.title} [{i.state}]" for i in items)

    # --- Writes -------------------------------------------------------

    def add(
        self,
        title: str,
        description: str,
        source: str = "llm_analysis",
        category: str = "feature",
        idea_type: str = "story",
        parent_id: str | None = None,
    ) -> BoardItem:
        item_id = self._next_id()
        idea = Idea(
            id=item_id,
            title=title,
            description=description,
            source=source,
            category=category,
            idea_type=idea_type,
            parent_id=parent_id,
        )
        self._items[item_id] = idea
        return copy.deepcopy(idea)

    def vote(self, item_id: str, voter: str, value: str) -> BoardItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.votes[voter] = value
        return copy.deepcopy(item)

    def add_comment(self, item_id: str, author: str, text: str) -> BoardItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.comments.append(
            IdeaComment(
                author=author,
                text=text,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )
        return copy.deepcopy(item)

    def get_comments(self, item_id: str) -> list[Comment]:
        item = self._items.get(item_id)
        if item is None:
            return []
        return [
            Comment(
                author=c.author,
                text=c.text,
                created=c.timestamp,
                marker=parse_marker(c.text),
            )
            for c in item.comments
        ]

    def mark_executing(self, item_id: str) -> BoardItem | None:
        return self._set_state(item_id, "executing")

    def mark_done(self, item_id: str, execution_log: str) -> BoardItem | None:
        item = self._set_state(item_id, "done")
        if item is not None:
            self._items[item_id].execution_log = execution_log
            return copy.deepcopy(self._items[item_id])
        return None

    def mark_failed(self, item_id: str, error: str) -> BoardItem | None:
        item = self._set_state(item_id, "failed")
        if item is not None:
            self._items[item_id].execution_log = error
            return copy.deepcopy(self._items[item_id])
        return None

    def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    # --- Epic/ordering helpers ---------------------------------------

    def set_execution_order(self, item_id: str, order: list[str]) -> BoardItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.execution_order = list(order)
        return copy.deepcopy(item)

    def get_execution_order(self, item_id: str) -> list[str]:
        item = self._items.get(item_id)
        return list(item.execution_order) if item is not None else []

    def set_epic_context(self, item_id: str, context: str) -> BoardItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.epic_context = context
        return copy.deepcopy(item)

    # --- Test helpers -------------------------------------------------

    def update_state(self, item_id: str, new_state: str) -> BoardItem | None:
        """Set an item's state directly — convenience for test setup.

        Routes through the matching ``mark_*`` method when one exists so
        the fake behaves the same way real code would transition the
        state; falls back to a direct assignment for states without a
        dedicated transition helper (``proposed``, ``refining``,
        ``approved``, ``vetoed``).
        """
        if new_state == "executing":
            return self.mark_executing(item_id)
        if new_state == "done":
            return self.mark_done(item_id, execution_log="")
        if new_state == "failed":
            return self.mark_failed(item_id, error="")
        return self._set_state(item_id, new_state)

    def _set_state(self, item_id: str, new_state: str) -> BoardItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.state = new_state
        return copy.deepcopy(item)


# ---------------------------------------------------------------------
# Self-test (runs under pytest)
# ---------------------------------------------------------------------


def test_fake_jira_provider_update_state_transitions_executing() -> None:
    provider = FakeJiraProvider()
    item = provider.add(title="seeded", description="body")

    provider.update_state(item.id, "executing")

    stored = provider.get(item.id)
    assert stored is not None
    assert stored.state == "executing"
