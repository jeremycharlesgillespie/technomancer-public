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
import subprocess
from datetime import datetime, timezone

from . import aiv_schema

log = logging.getLogger(__name__)


def get_merged_diff_paths(
    merge_base_ref: str, head_ref: str = "HEAD"
) -> list[str]:
    """Return the files changed between ``merge_base_ref`` and ``head_ref``.

    Runs ``git diff --name-only {merge_base_ref}..{head_ref}`` in the current
    working directory and parses the output into a list of file paths.

    Args:
        merge_base_ref: Git ref for the merge base (e.g. the pre-merge tip
            of ``main``, or ``HEAD~1`` right after a ``--no-ff`` merge).
        head_ref: Git ref for the merged tip. Defaults to ``"HEAD"``.

    Returns:
        The list of changed file paths, one per line of ``git diff`` output
        with blank lines stripped. Returns an empty list on any subprocess
        error (missing git, non-zero exit, timeout) — the caller never has
        to wrap this itself because the validation hook must never fail a
        deploy.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{merge_base_ref}..{head_ref}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "get_merged_diff_paths(%r, %r) failed: %s",
            merge_base_ref,
            head_ref,
            exc,
        )
        return []
    if result.returncode != 0:
        return []
    return [ln.strip() for ln in result.stdout.split("\n") if ln.strip()]


def enqueue_for_validation(
    story_key: str,
    diff_paths: list[str],
    merged_at: str | None = None,
    *,
    merge_commit_sha: str | None = None,
    verification_output: str | None = None,
) -> None:
    """Insert an ``aiv_pending`` row for ``story_key``.

    Args:
        story_key: The Jira issue key (e.g. ``"TK-123"``) that just merged.
        diff_paths: Files changed by the branch, as returned by
            ``git diff --name-only {merge_base}..HEAD``. May be empty if
            the caller couldn't compute a diff.
        merged_at: ISO-8601 timestamp of when the merge landed on
            ``main``. Defaults to the current UTC time if not provided.
        merge_commit_sha: Full SHA of the merge commit on ``main``. The
            AIV daemon uses this to materialise the actual unified diff
            via ``git show`` at scoring time, so the scorer evaluates
            real code rather than just file paths.
        verification_output: Captured pytest / validate.py output that
            proved the merge was safe. Stored verbatim and replayed to
            the scorer; the scorer grades ``test_quality`` and
            ``edge_cases`` against this text.

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
            "(story_key, merged_at, diff_paths_json, enqueued_at, "
            " merge_commit_sha, verification_output) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                story_key,
                merged_at if merged_at is not None else now,
                json.dumps(list(diff_paths or [])),
                now,
                merge_commit_sha,
                verification_output,
            ),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("enqueue_for_validation(%r) failed: %s", story_key, exc)
