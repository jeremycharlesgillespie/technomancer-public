"""Resource-lifecycle wrapper for AIW story runs.

The AIW orchestrator today scatters acquire/release pairs across multiple
files (executor.py creates branches, ab_executor.py creates worktrees,
ollama_coder.py loads models with keep_alive=-1). Cleanup lives in
hand-written ``finally`` blocks — many of which never run on a hard
process kill, leaving orphan worktrees, resident model VRAM, and
half-pushed feature branches that require manual cleanup.

This module introduces a single owner for that lifecycle:

- :class:`Resource` is the abstract base for one acquired thing. Its
  ``release()`` method is idempotent and never raises — failures are
  logged and swallowed so an exception in one teardown step doesn't
  block the next.

- :class:`StoryRun` is a context manager that tracks acquired resources
  in LIFO order. Resources are released in reverse acquisition order
  on ``__exit__``, regardless of whether the body succeeded, failed,
  or raised. The body cannot bypass cleanup.

Concrete resource classes (:class:`Branch`, :class:`Worktree`,
:class:`ModelHandle`) wrap the existing primitives but expose only the
acquire/release shape.

This first PR ships the building block. Wiring ``ab_executor`` to use
``StoryRun`` is a follow-up so the change is reviewable in isolation
and the cutover can be done with confidence.

Design notes
------------

- ``release`` must be idempotent. A resource that has been released
  already is a no-op. This lets the wrapper retry teardown safely if
  an outer cleanup pass is ever added.

- ``release`` must NEVER raise. The whole point of the wrapper is to
  guarantee cleanup runs to completion even when individual steps
  fail. Each Resource subclass is responsible for catching its own
  exceptions, logging them via ``logger.warning`` (or ``error`` for
  loud cases), and returning normally. Tests pin this contract.

- The wrapper does NOT log success cases — only failures. Successful
  teardown is the boring default; noise about it would drown out the
  signals that matter.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource base class
# ---------------------------------------------------------------------------


class Resource(ABC):
    """One acquired thing whose lifecycle is owned by a :class:`StoryRun`.

    Concrete subclasses must implement :meth:`_do_release`. The public
    :meth:`release` wraps that with idempotency + exception swallowing
    so callers (and the wrapper) can call it freely.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._released = False

    @property
    def released(self) -> bool:
        """True after :meth:`release` has been called at least once."""
        return self._released

    def release(self) -> None:
        """Release the resource. Idempotent. Never raises.

        Logs warnings on failure. The wrapper relies on this being
        safe to call from a ``finally`` block during a stack unwind.
        """
        if self._released:
            return
        # Mark released BEFORE running the actual teardown. If teardown
        # raises (it shouldn't — _do_release should swallow — but defense
        # in depth) we still don't want to retry on a partial state.
        self._released = True
        try:
            self._do_release()
        except Exception as exc:  # noqa: BLE001 — Resource.release MUST NOT raise
            logger.error(
                "[StoryRun] Resource %s release raised (swallowed): %s",
                self.label, exc,
            )

    @abstractmethod
    def _do_release(self) -> None:
        """Subclass-specific teardown. May log; should not raise.

        Implementations should catch their own subprocess / OS errors
        and log them via ``logger.warning``. The base class swallows
        anything that escapes as a defense-in-depth, but the warning
        messages are clearer when the subclass logs the original error
        with full context.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete resources
# ---------------------------------------------------------------------------


class Branch(Resource):
    """A git feature branch created for a story run.

    Today's branches survive on purpose — they're the durable artifact
    that A/B compares and the merge picks from. ``release()`` is
    therefore a no-op by default. Subclasses or future use cases that
    DO want auto-deletion (e.g. a discarded run that nobody will look
    at) can set ``delete_on_release=True``.
    """

    def __init__(
        self,
        name: str,
        repo_root: Path,
        *,
        delete_on_release: bool = False,
    ) -> None:
        super().__init__(label=f"branch:{name}")
        self.name = name
        self.repo_root = repo_root
        self.delete_on_release = delete_on_release

    def _do_release(self) -> None:
        if not self.delete_on_release:
            # Branches are durable by default. Nothing to do.
            return
        try:
            result = subprocess.run(
                ["git", "branch", "-D", self.name],
                capture_output=True, text=True, timeout=15,
                cwd=str(self.repo_root),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(
                "[StoryRun] branch -D %s crashed: %s", self.name, exc,
            )
            return
        if result.returncode != 0:
            logger.warning(
                "[StoryRun] branch -D %s exit=%d stderr=%s",
                self.name, result.returncode, (result.stderr or "")[:200],
            )


class Worktree(Resource):
    """A git worktree created for an A/B run.

    Owns the directory at ``path`` plus the corresponding entry in
    ``.git/worktrees/``. ``release()`` runs ``git worktree remove
    --force`` then falls back to ``shutil.rmtree`` if the directory
    survives, then runs ``git worktree prune`` so stale metadata never
    accumulates. All errors are logged and swallowed.
    """

    def __init__(self, repo_root: Path, path: Path) -> None:
        super().__init__(label=f"worktree:{path.name}")
        self.repo_root = repo_root
        self.path = path

    def _do_release(self) -> None:
        # Step 1: ask git to remove the worktree.
        if self.path.exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(self.repo_root),
                     "worktree", "remove", "--force", str(self.path)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    logger.warning(
                        "[StoryRun] git worktree remove %s exit=%d: %s",
                        self.path, result.returncode,
                        (result.stderr or "")[:200],
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning(
                    "[StoryRun] git worktree remove %s crashed: %s",
                    self.path, exc,
                )

        # Step 2: directory may still exist if git refused. Nuke it.
        if self.path.exists():
            try:
                shutil.rmtree(self.path)
            except OSError as exc:
                logger.warning(
                    "[StoryRun] rmtree fallback failed for %s: %s",
                    self.path, exc,
                )

        # Step 3: always prune so stale .git/worktrees/ entries don't
        # accumulate even if the directory removal failed.
        try:
            subprocess.run(
                ["git", "-C", str(self.repo_root), "worktree", "prune"],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("[StoryRun] worktree prune crashed: %s", exc)


class ModelHandle(Resource):
    """An Ollama model loaded with ``keep_alive=-1``.

    OllamaCoder pins the coder model resident for the duration of a
    round so the 6+ second reload penalty doesn't fire between rounds.
    ``release()`` POSTs ``keep_alive=0`` to ``/api/generate`` so the
    runner is evicted immediately when the run ends — without this,
    a hard kill of the worker would leave the model resident in VRAM
    until manual eviction.
    """

    def __init__(self, model_tag: str, host: str = "") -> None:
        super().__init__(label=f"model:{model_tag}@{host or 'localhost'}")
        self.model_tag = model_tag
        self.host = host

    def _do_release(self) -> None:
        if not self.model_tag:
            return
        try:
            # Late import — keeps this module importable in tests that
            # don't have requests / agent.ollama_client wired.
            import requests
            from agent.ollama_client import OLLAMA_HOST
            target_host = self.host or OLLAMA_HOST
            r = requests.post(
                f"{target_host}/api/generate",
                json={
                    "model": self.model_tag,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
                timeout=15,
            )
            if r.status_code != 200:
                logger.warning(
                    "[StoryRun] unload %s on %s: HTTP %d",
                    self.model_tag, target_host, r.status_code,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[StoryRun] unload %s on %s crashed: %s",
                self.model_tag, self.host or "localhost", exc,
            )


# ---------------------------------------------------------------------------
# StoryRun context manager
# ---------------------------------------------------------------------------


class StoryRun:
    """Context manager owning the lifecycle of one story's resources.

    Acquire resources via :meth:`acquire`. They are tracked in
    acquisition order and released in REVERSE order on ``__exit__``,
    whether the body returned normally or raised.

    The context manager itself does NOT swallow exceptions raised by
    the body — those propagate. It DOES guarantee that every resource
    sees ``release()`` exactly once.

    Example::

        with StoryRun(label="TK-1234") as run:
            worktree = run.acquire(Worktree(repo_root, wt_path))
            model = run.acquire(ModelHandle("qwen3-coder:30b", host=""))
            do_work(worktree, model)
        # On exit: model released first (LIFO), then worktree.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._resources: list[Resource] = []

    def acquire(self, resource: Resource) -> Resource:
        """Register an already-acquired resource for cleanup.

        The convention is: callers construct a :class:`Resource`
        subclass (which performs whatever setup it needs in its
        ``__init__``), then pass it here so its ``release`` is
        guaranteed on context exit.

        Returns the resource so the caller can chain
        ``thing = run.acquire(Thing(...))``.
        """
        self._resources.append(resource)
        return resource

    @property
    def resources(self) -> list[Resource]:
        """Snapshot of currently-tracked resources, in acquisition order.

        Mainly for tests. Callers should not mutate this list.
        """
        return list(self._resources)

    def release_all(self) -> None:
        """Release every tracked resource in LIFO order.

        Public so callers can release manually if they want to free
        resources before the context block ends. ``__exit__`` calls
        this automatically — calling it twice is safe because each
        resource's ``release`` is idempotent.
        """
        # Iterate over a snapshot. A pathological _do_release could
        # mutate self._resources; we don't want to skip entries.
        for resource in reversed(list(self._resources)):
            resource.release()

    def __enter__(self) -> "StoryRun":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Run every release. Resource.release is idempotent + never
        # raises, so this loop cannot itself fail. We do NOT suppress
        # the original exception (returning None / falsy from __exit__).
        self.release_all()
