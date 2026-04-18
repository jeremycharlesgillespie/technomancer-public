"""AIV Hook — enqueue a shipped story for post-merge validation.

After a successful merge lands on ``main``, the executor calls
:func:`enqueue_for_validation` with the Jira key and the list of files
changed by the branch. One row per call is written to the
``aiv_pending`` table defined in :mod:`agent.aiv_schema`; the AIV daemon
drains that queue on its own cycle.

The function is best-effort and non-blocking — any SQLite error is
logged and swallowed so a validation-queue hiccup can never fail a
deploy.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from . import aiv_schema

log = logging.getLogger(__name__)


def enqueue_for_validation(story_key: str, diff_paths: list[str]) -> None:
    """Insert an ``aiv_pending`` row for ``story_key``.

    Args:
        story_key: The Jira issue key (e.g. ``"TK-123"``) that just merged.
        diff_paths: Files changed by the branch, as returned by
            ``git diff --name-only {merge_base}..HEAD``. May be empty if
            the caller couldn't compute a diff.

    Uses ``INSERT OR REPLACE`` so re-validation of the same story simply
    refreshes the pending row instead of raising on the primary-key
    constraint.
    """
    try:
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO aiv_pending "
            "(story_key, merged_at, diff_paths_json, enqueued_at) "
            "VALUES (?, ?, ?, ?)",
            (story_key, now, json.dumps(list(diff_paths or [])), now),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("enqueue_for_validation(%r) failed: %s", story_key, exc)
