"""Per-orchestrator git worktrees for the A/B harness.

Each :func:`idea_board.ab_executor.execute_idea_ab` run generates a
short UUID and provisions an isolated git worktree at::

    <repo_parent>/technomancer-aiw-<uuid8>

Both the model A and model B inner runs operate inside that worktree.
The orchestrator removes the worktree (forcefully) in its ``finally``
block. UUID isolation guarantees that:

- Two orchestrators racing each other (e.g. a stalled-but-not-killed
  zombie thread plus a fresh story) never write to the same checkout.
- A crash that leaves a worktree behind doesn't collide with the next
  run because the next run's UUID is different.

The worktrees are sibling directories to the main checkout, NOT inside
it — that's required by ``git worktree add`` and also keeps them out of
``find . -type f`` style searches in the main tree.

Why a separate module: single mockable seam for tests so we can verify
``execute_idea_ab`` calls create/remove without spawning real worktrees.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def short_uuid() -> str:
    """Return an 8-char lowercase hex slug, sufficient to disambiguate
    a few thousand concurrent worktrees without making paths unreadable."""
    return uuid.uuid4().hex[:8]


def worktree_path_for(repo_root: Path, slug: str) -> Path:
    """Return the sibling directory where the worktree should live.

    Sibling not child: ``git worktree add`` rejects paths inside the
    repo. Naming pattern: ``<parent>/technomancer-aiw-<slug>``.
    """
    parent = repo_root.parent
    return parent / f"technomancer-aiw-{slug}"


def create_worktree(repo_root: Path, slug: str) -> Path:
    """Create a fresh worktree pointed at ``main`` and return its path.

    Steps:
    1. Compute the sibling path.
    2. Refuse to clobber an existing directory there (extremely unlikely
       given the UUID, but a stale dir would be a real bug worth surfacing).
    3. ``git -C <repo_root> worktree add --detach <path> main`` so the
       new tree is on a detached HEAD pointed at main. The orchestrator's
       inner ``execute_idea`` will branch from there as it always does.
    4. Return the absolute path.

    Raises ``RuntimeError`` if any step fails — the orchestrator catches
    this and falls back to single-tree mode (logged as a warning).
    """
    target = worktree_path_for(repo_root, slug)
    if target.exists():
        raise RuntimeError(
            f"worktree path already exists: {target} "
            f"(stale from prior crash? remove it manually)"
        )

    cmd = ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(target), "main"]
    logger.info("[AB-Worktree] creating: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"git worktree add failed: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"git worktree add exit={result.returncode}: "
            f"stderr={(result.stderr or '').strip()[:500]}"
        )

    if not target.exists():
        raise RuntimeError(
            f"git worktree add reported success but {target} does not exist"
        )

    # Symlink local-agent/.env from the source repo into the worktree so
    # pytest runs in the worktree see the same env (Jira creds, project
    # key, model overrides, etc.) the source repo uses. Without this, any
    # test that reads `agent.config.settings.<X>` for a value that lives
    # in .env hits None and fails — and pytest is the harness's
    # success/failure gate, so a single missing setting kills every AIW
    # round. Symlink not copy: env values change frequently and we never
    # want a worktree to hold a stale token snapshot. Best-effort: if
    # the source .env is missing or the link fails for any reason, log
    # and continue — tests SHOULD mock their own env, this is a safety
    # net for tests that forget to.
    src_env = repo_root / "local-agent" / ".env"
    dst_env = target / "local-agent" / ".env"
    if src_env.is_file() and not dst_env.exists():
        try:
            dst_env.symlink_to(src_env)
            logger.info("[AB-Worktree] symlinked .env: %s -> %s", dst_env, src_env)
        except OSError as exc:
            logger.warning("[AB-Worktree] could not symlink .env: %s", exc)

    logger.info("[AB-Worktree] created: %s", target)
    return target


def remove_worktree(repo_root: Path, worktree_path: Path) -> bool:
    """Force-remove the worktree at ``worktree_path`` and prune.

    Force-remove because the inner runs may have left local changes
    (uncommitted files, unpushed branches) that ``git worktree remove``
    would otherwise refuse to delete. We don't care — the orchestrator
    has already pushed any branches it wants to keep.

    Returns True on success, False on failure. Failures are logged but
    not raised so the orchestrator's ``finally`` block always completes.
    Falls back to ``shutil.rmtree`` if the git command can't clean up.
    Always runs ``git worktree prune`` at the end so stale entries don't
    accumulate in ``.git/worktrees/`` even if the directory removal fails.
    """
    if not worktree_path.exists():
        logger.info("[AB-Worktree] already gone: %s", worktree_path)
        _prune(repo_root)
        return True

    cmd = ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[AB-Worktree] git worktree remove crashed: %s", exc)
        result = None

    git_ok = result is not None and result.returncode == 0

    # Fallback: directory may still exist if git refused. Nuke it.
    if worktree_path.exists():
        try:
            shutil.rmtree(worktree_path)
            logger.info("[AB-Worktree] rmtree fallback succeeded for %s", worktree_path)
        except OSError as exc:
            logger.warning(
                "[AB-Worktree] rmtree fallback failed for %s: %s",
                worktree_path, exc,
            )
            _prune(repo_root)
            return False

    _prune(repo_root)

    if not git_ok:
        logger.info(
            "[AB-Worktree] removed via fallback (git rc=%s)",
            result.returncode if result is not None else "crash",
        )
    else:
        logger.info("[AB-Worktree] removed: %s", worktree_path)
    return True


def _prune(repo_root: Path) -> None:
    """Run ``git worktree prune`` to clear stale entries from
    ``.git/worktrees/``. Best-effort — failures are logged and swallowed."""
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "prune"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[AB-Worktree] prune failed: %s", exc)
