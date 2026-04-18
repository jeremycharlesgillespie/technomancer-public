"""Post-merge hand-off from the executor to the AIV validation queue.

The executor calls :func:`enqueue_merged_story` immediately after
``git merge --no-ff`` returns. The helper is a no-op on a failed merge
(so the call site can stay a single line, with no extra guard), and
computes the list of changed files via ``git diff --name-only`` before
delegating to :func:`agent.aiv_hook.enqueue_for_validation`.

Any error — SQLite, subprocess, unicode — is logged and swallowed so a
validation-queue hiccup can never fail a deploy.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from agent.aiv_hook import enqueue_for_validation

log = logging.getLogger(__name__)


def get_merged_diff_paths(
    project_root: Path, base: str = "HEAD~1", head: str = "HEAD"
) -> list[str]:
    """Return the files changed by the most recent merge commit.

    After ``git merge --no-ff`` lands, ``HEAD~1..HEAD`` spans exactly the
    merge-commit range on ``main``. Returns an empty list on any
    subprocess error so the caller never has to wrap this itself.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}..{head}"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("get_merged_diff_paths failed: %s", exc)
        return []
    if result.returncode != 0:
        return []
    return [ln.strip() for ln in result.stdout.split("\n") if ln.strip()]


def enqueue_merged_story(
    idea_id: str, merge_result: Any, project_root: Path
) -> None:
    """Enqueue ``idea_id`` for AIV validation if ``merge_result`` succeeded.

    Args:
        idea_id: The Jira issue key of the just-merged story.
        merge_result: The ``subprocess.CompletedProcess`` returned by the
            executor's ``git merge`` call. Only inspected for
            ``returncode``; anything truthy non-zero means "skip".
        project_root: Repo root for the ``git diff`` invocation.

    On a failed merge this function is a deliberate no-op so the executor
    can always call it in a single line without branching.
    """
    if getattr(merge_result, "returncode", 1) != 0:
        return
    try:
        diff_paths = get_merged_diff_paths(project_root)
        enqueue_for_validation(idea_id, diff_paths)
    except Exception as exc:  # noqa: BLE001
        log.warning("enqueue_merged_story(%r) failed: %s", idea_id, exc)
