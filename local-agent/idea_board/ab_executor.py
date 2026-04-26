"""A/B Executor — drive two-model comparison runs over a single story.

When ``settings.aiw_ab_test_enabled`` is True, ``aim/worker.py`` calls
:func:`execute_idea_ab` instead of :func:`idea_board.executor.execute_idea`.

The orchestrator returns an :class:`ExecutionState` immediately (so the
worker's existing ``watch_execution`` loop is unchanged) and runs the
following sequence in a background thread:

1. Run A on the configured incumbent model
   (:data:`agent.config.settings.aiw_ab_model_a`).
2. Push A's feature branch to private + public.
3. Reset to a clean main.
4. Run B on the challenger model
   (:data:`agent.config.settings.aiw_ab_model_b`). If the challenger model
   isn't pulled locally, the run is short-circuited as failed and A
   continues unaffected.
5. Push B's feature branch to private + public.
6. Score both diffs through :func:`aiv.scorer.score`.
7. Compare via :func:`aiv.ab_compare.compare`.
8. Pick the merge winner via :func:`idea_board.ab_repo.pick_winner`
   (model A wins when it succeeded; model B only wins when A failed and
   B succeeded; both-failed → no merge).
9. Merge the winner's branch to main + push private main + run
   ``publish.py --push --force`` for the public-side ``main``.
10. Mark the outer idea ``done`` (or ``failed`` when no winner exists).

Failure of the comparison call (LLM timeout, malformed JSON) is logged
into ``ab_test_pairs.comparison_error`` but does not block the merge —
the priority rule fires regardless.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.aiv_schema import SCORE_COLUMNS
from idea_board.executor import (
    ExecutionState,
    _ab_orchestrator_active,
    _active,
    execute_idea,
    mark_done,
    mark_executing,
    mark_failed,
)
from idea_board import ab_repo
from idea_board import ab_worktree

logger = logging.getLogger(__name__)


@dataclass
class _RunOutcome:
    """Internal record of one model's attempt at the story."""

    run_id: str
    model: str
    label: str
    status: str = "failed"
    branch_name: str = ""
    commit_sha: str = ""
    failure_log: str = ""
    diff: str = ""
    log_text: str = ""
    scores: Any = None  # aiv.scorer.StoryQualityScores | None

    def to_compare_dict(self) -> dict[str, Any]:
        """Build the dict consumed by :func:`aiv.ab_compare.compare`."""
        out: dict[str, Any] = {
            "model": self.model,
            "status": self.status,
            "branch_name": self.branch_name,
            "commit_sha": self.commit_sha,
            "diff": self.diff,
            "failure_log": self.failure_log,
        }
        if self.scores is not None:
            for axis in SCORE_COLUMNS:
                v = getattr(self.scores, axis, None)
                if isinstance(v, int):
                    out[axis] = v
        return out


def _unload_ollama_model(model_tag: str, host: str = "") -> bool:
    """Force-evict ``model_tag`` from Ollama's resident set on ``host``.

    Posts ``keep_alive=0`` to ``/api/generate`` with an empty prompt,
    which tells Ollama to drop the runner immediately. This is the
    Ollama-recommended way to free VRAM without restarting the server.

    Used between A and B runs so we never have BOTH coder models
    resident at once on the same GPU — on memory-constrained boxes the
    second load would page-thrash or fail outright. When A and B run
    on *different* hosts there is no contention to resolve, so the
    orchestrator skips this call.

    ``host`` is the Ollama base URL. Empty string falls back to the
    module-level OLLAMA_HOST (localhost). Returns True on HTTP 200.
    """
    if not model_tag:
        return False
    try:
        from agent.ollama_client import OLLAMA_HOST
        import requests
        target_host = host or OLLAMA_HOST
        r = requests.post(
            f"{target_host}/api/generate",
            json={
                "model": model_tag,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=15,
        )
        return r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AB] unload %s failed: %s", model_tag, exc)
        return False


def _ollama_has_model(model_tag: str, host: str = "") -> bool:
    """Return True if Ollama on ``host`` has ``model_tag`` installed.

    Queries ``GET /api/tags`` over HTTP so the same code works whether
    the target is the local Ollama or a remote one (e.g. the 5090 box
    over Tailscale/LAN). The earlier implementation used the local
    ``ollama list`` CLI — that couldn't see a remote machine.

    ``host`` is the Ollama base URL. Empty string falls back to the
    module-level OLLAMA_HOST (localhost). Failures (network, timeout,
    non-200) return False so the orchestrator short-circuits B's run
    with a clear failure_log instead of hanging on the model load.
    """
    try:
        from agent.ollama_client import OLLAMA_HOST
        import requests
        target_host = host or OLLAMA_HOST
        r = requests.get(f"{target_host}/api/tags", timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AB] /api/tags on %s failed: %s", host or "localhost", exc)
        return False
    if r.status_code != 200:
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    # /api/tags returns {"models": [{"name": "qwen2.5-coder:32b", ...}, ...]}.
    # Exact match so qwen2.5-coder:32b doesn't false-match qwen2.5-coder:7b.
    for m in data.get("models", []) or []:
        if isinstance(m, dict) and m.get("name") == model_tag:
            return True
    return False


def _diff_paths_from_text(diff_text: str) -> list[str]:
    """Extract changed file paths from a unified diff string.

    Used by the A/B AIV hand-off: we already captured the full diff for
    scoring, so re-extracting paths from it avoids a second ``git diff``
    subprocess. Looks for ``diff --git a/<path> b/<path>`` headers.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for line in (diff_text or "").splitlines():
        if not line.startswith("diff --git "):
            continue
        # Format: ``diff --git a/<path> b/<path>``. Take the ``b/`` half
        # so renames map to the new name.
        parts = line.split(" b/", 1)
        if len(parts) != 2:
            continue
        path = parts[1].strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _head_sha(project_root: Path) -> str | None:
    """Return the full SHA of HEAD on the current checkout (post-merge ``main``)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(project_root),
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    sha = (result.stdout or "").strip()
    return sha or None


def _capture_diff(project_root: Path, branch: str) -> str:
    """Return ``git diff main...<branch>`` capped at 12 KB."""
    if not branch:
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", f"main...{branch}"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"[diff capture failed: {exc}]"
    if result.returncode != 0:
        return f"[diff capture exit={result.returncode}]"
    out = result.stdout or ""
    if len(out) > 12000:
        out = out[:12000] + f"\n... [truncated {len(out) - 12000} chars]"
    return out


def _reset_to_main(project_root: Path, state: ExecutionState) -> bool:
    """Force-reset to a clean main between runs A and B.

    Returns True on success. The first run leaves us on a feature branch
    that's been merged-to-main on the next run, so this duplicates the
    sequence ``aim/worker._ensure_git_clean`` performs at the worker
    boundary.
    """
    try:
        subprocess.run(
            ["git", "checkout", "--force", "main"],
            capture_output=True, text=True, timeout=15,
            cwd=str(project_root),
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
    except Exception as exc:  # noqa: BLE001
        state.log(f"[AB] reset_to_main failed: {exc}")
        return False
    return True


def _wait_for_inner_state(
    inner_state: ExecutionState,
    state: ExecutionState,
    *,
    timeout: int,
    label: str,
) -> None:
    """Block until ``inner_state.thread`` finishes or we hit ``timeout``.

    Mirrors a tiny subset of ``aim.worker.watch_execution`` — just enough
    to let the orchestrator drive sequential runs from inside its own
    background thread without re-entering the worker's heartbeat / stall
    machinery (which is owned by the outer worker run).

    Also propagates ``state.cancelled`` to the inner state so an
    orchestrator-level cancel (worker stall, force_cancel_all_active)
    promptly aborts the inner run instead of waiting out its own
    timeout.
    """
    if inner_state.thread is None:
        return
    deadline = time.time() + timeout
    while inner_state.thread.is_alive():
        if state.cancelled and not inner_state.cancelled:
            state.log(f"[AB] orchestrator cancelled — propagating to {label}")
            inner_state.cancelled = True
        if time.time() > deadline:
            state.log(f"[AB] {label} run hit hard timeout ({timeout}s)")
            return
        time.sleep(2)


def _build_story_dict(idea: Any) -> dict[str, Any]:
    """Project an idea object into the dict shape the scorer / compare
    modules expect."""
    return {
        "key": getattr(idea, "id", "") or "",
        "title": getattr(idea, "title", "") or "",
        "description": getattr(idea, "description", "") or "",
        "category": getattr(idea, "category", "") or "",
    }


def _scores_to_dict(scores: Any) -> dict[str, Any]:
    """Project a :class:`aiv.scorer.StoryQualityScores` into the dict
    consumed by :func:`ab_repo.record_run_end`. Tolerates ``None``."""
    if scores is None:
        return {}
    try:
        d = asdict(scores)
    except TypeError:
        return {}
    # Compute overall_score as the mean of axis scores >= 0 (sentinel = -1).
    valid = [d.get(a) for a in SCORE_COLUMNS if isinstance(d.get(a), int) and d.get(a) >= 0]
    if valid:
        d["overall_score"] = sum(valid) / len(valid)
    else:
        d["overall_score"] = None
    return d


def _do_attempt(
    idea_id: str,
    *,
    model: str,
    suffix: str,
    state: ExecutionState,
    project_root: Path,
    inner_timeout: int,
    host: str = "",
) -> tuple[bool, str, str, str]:
    """Run a single inner :func:`execute_idea` with skip_merge=True.

    Returns ``(success, branch_name, commit_sha, log_tail)`` where
    success means the inner run committed to a feature branch and pushed
    it to private origin. The orchestrator turns that triplet into an
    :class:`_RunOutcome`.

    ``host`` is the Ollama base URL for this attempt. Empty string
    falls back to the default localhost. Used by the A/B harness to
    point one model at a remote Ollama instance.
    """
    state.log(
        f"[AB] starting attempt: model={model} suffix={suffix} "
        f"host={host or 'localhost'}"
    )

    # Pop any leftover _active entry so the dedup guard at
    # idea_board.executor:1819 doesn't bounce us. The previous run's
    # finally clause already pops, but we belt-and-suspenders here.
    _active.pop(idea_id, None)

    inner = execute_idea(
        idea_id,
        model_override=model,
        host_override=host or None,
        branch_suffix=suffix,
        skip_merge=True,
        project_root_override=project_root,
    )
    if inner is None or inner.thread is None:
        state.log(f"[AB] inner execute_idea returned None for model={model}")
        return False, "", "", ""

    _wait_for_inner_state(inner, state, timeout=inner_timeout, label=model)

    success = bool(inner.deploy_sha)
    branch = inner.branch_name
    sha = inner.deploy_sha
    log_tail = inner.log_text[-5000:] if inner.log_text else ""

    # The inner thread's `finally` clears _active for us.
    state.log(
        f"[AB] attempt finished: model={model} success={success} "
        f"branch={branch} sha={sha[:7] if sha else ''}"
    )
    return success, branch, sha, log_tail


def _push_branch(
    state: ExecutionState,
    branch: str,
    project_root: Path,
) -> tuple[bool, str]:
    """Push the feature branch to both repos via :mod:`ab_repo`."""
    publish_script = project_root / "local-agent" / "publish.py"
    ok, msg = ab_repo.push_branch_to_both_repos(
        branch,
        private_repo=project_root,
        publish_script=publish_script,
    )
    state.log(f"[AB] push {branch}: {msg}")
    return ok, msg


def _merge_winner(
    state: ExecutionState,
    branch: str,
    idea_id: str,
    project_root: Path,
) -> tuple[bool, str]:
    """Merge ``branch`` to main + push origin main + run publish.py.

    Mirrors the deploy block at the bottom of ``executor.execute_idea``
    but with the A/B-friendly twist that the branch we're merging may
    not be the one we're currently on (we're already on main from the
    reset between runs A/B).
    """
    try:
        # Make sure we're on main and up to date.
        subprocess.run(
            ["git", "checkout", "main"],
            capture_output=True, text=True, timeout=15,
            cwd=str(project_root),
        )
        subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        # Need the winner branch locally — fetch and create a tracking ref.
        subprocess.run(
            ["git", "fetch", "origin", branch],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        subprocess.run(
            ["git", "branch", "-f", branch, f"origin/{branch}"],
            capture_output=True, text=True, timeout=15,
            cwd=str(project_root),
        )
        merge = subprocess.run(
            ["git", "merge", "--no-ff", branch,
             "-m", f"[{idea_id}] Merge A/B winner branch '{branch}' - ab_executor"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        if merge.returncode != 0:
            state.log(f"[AB] winner merge failed: {(merge.stderr or merge.stdout)[:300]}")
            return False, "merge_failed"

        push = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True, text=True, timeout=60,
            cwd=str(project_root),
        )
        if push.returncode != 0:
            state.log(f"[AB] push origin main failed: {(push.stderr or push.stdout)[:300]}")
            return False, "push_main_failed"

        # Public-side main mirror — same call as the regular executor.
        publish_script = project_root / "local-agent" / "publish.py"
        if publish_script.exists():
            pub = subprocess.run(
                [sys.executable, str(publish_script), "--push", "--force"],
                capture_output=True, text=True, timeout=120,
                cwd=str(publish_script.parent),
            )
            if pub.returncode != 0:
                # Non-fatal — main already shipped privately. Log and continue.
                state.log(
                    f"[AB] publish main failed (non-fatal): "
                    f"{(pub.stderr or pub.stdout)[:300]}"
                )
            else:
                state.log("[AB] published main to technomancer-public")
        return True, "merged"
    except Exception as exc:  # noqa: BLE001
        state.log(f"[AB] merge_winner raised: {exc}")
        return False, str(exc)[:200]


def execute_idea_ab(idea_id: str) -> ExecutionState | None:
    """Drive two model attempts at one story and merge the winner.

    Returns an :class:`ExecutionState` immediately so the worker's
    existing ``watch_execution`` loop is happy. The state's ``thread``
    runs the full A/B flow.

    Returns ``None`` only when ``idea_id`` doesn't resolve to an idea.
    """
    from agent.config import settings
    from board import get_provider

    idea = get_provider().get(idea_id)
    if idea is None:
        return None

    # The inner per-attempt runs touch ``_active`` themselves, so we keep the
    # orchestrator's outer state in a sibling registry. ``get_execution`` falls
    # back to it when the inner slot is transiently empty between attempts.
    if idea_id in _ab_orchestrator_active and _ab_orchestrator_active[idea_id].is_alive:
        return _ab_orchestrator_active[idea_id]
    if idea_id in _active and _active[idea_id].is_alive:
        return _active[idea_id]

    state = ExecutionState(idea_id=idea_id)
    _ab_orchestrator_active[idea_id] = state
    mark_executing(idea_id)

    project_root = (
        Path(settings.project_root)
        if settings.project_root
        else Path(__file__).parent.parent.parent
    )

    model_a = settings.aiw_ab_model_a
    model_b = settings.aiw_ab_model_b
    host_a = settings.aiw_ab_model_a_host or ""
    host_b = settings.aiw_ab_model_b_host or ""
    label_a = ab_repo.model_label(model_a)
    label_b = ab_repo.model_label(model_b)
    suffix_a = ab_repo.branch_suffix(model_a)
    suffix_b = ab_repo.branch_suffix(model_b)

    inner_timeout = max(
        getattr(settings, "aim_execution_timeout", 2700),
        2700,
    )

    def _run() -> None:
        run_a_id = uuid.uuid4().hex
        run_b_id = uuid.uuid4().hex
        outcome_a = _RunOutcome(run_id=run_a_id, model=model_a, label=label_a)
        outcome_b = _RunOutcome(run_id=run_b_id, model=model_b, label=label_b)

        # Record start times for each run individually
        ab_repo.record_run_start(run_a_id, idea_id, model_a, label_a)
        ab_repo.record_run_start(run_b_id, idea_id, model_b, label_b)

        # Per-orchestrator git worktree. UUID-suffixed sibling directory
        # so two orchestrators (e.g. a stalled-but-not-killed zombie plus
        # a fresh story) can never write to the same checkout. Falls back
        # to single-tree mode (project_root) if worktree creation fails —
        # logged loudly because that's the dangerous path.
        wt_slug = ab_worktree.short_uuid()
        worktree_root: Path | None = None
        try:
            worktree_root = ab_worktree.create_worktree(project_root, wt_slug)
            work_root = worktree_root
            state.log(f"[AB] using isolated worktree: {worktree_root}")
        except Exception as exc:  # noqa: BLE001
            state.log(
                f"[AB] WORKTREE FAILED ({exc}) — falling back to shared "
                f"project_root={project_root}. CONCURRENT RUNS MAY COLLIDE."
            )
            work_root = project_root

        try:
            # ----- Run A -----
            state.log(
                f"=== A/B Run A: {model_a} @ {host_a or 'localhost'} ==="
            )
            ok_a, branch_a, sha_a, log_a = _do_attempt(
                idea_id,
                model=model_a,
                suffix=suffix_a,
                state=state,
                project_root=work_root,
                inner_timeout=inner_timeout,
                host=host_a,
            )
            outcome_a.status = "success" if ok_a else "failed"
            outcome_a.branch_name = branch_a
            outcome_a.commit_sha = sha_a
            outcome_a.log_text = log_a
            if not ok_a:
                outcome_a.failure_log = log_a

            if ok_a and branch_a:
                outcome_a.diff = _capture_diff(work_root, branch_a)
                push_ok, push_msg = _push_branch(state, branch_a, work_root)
                if not push_ok:
                    outcome_a.status = "failed"
                    outcome_a.failure_log = (
                        outcome_a.failure_log + f"\n[AB-push] {push_msg}"
                    ).strip()

            # Evict model A before B loads — both coder runs pin
            # ``keep_alive=-1`` in OllamaCoder, so without an explicit
            # unload the scheduler may try to keep both 25-30 GB models
            # resident and thrash. We accept the ~6s reload penalty for
            # the next AIM cycle in exchange for clean single-model VRAM.
            #
            # When A and B target *different* hosts there is no shared
            # VRAM to thrash — model A's runner stays loaded on its host
            # while B loads on the other. Skip the eviction in that case
            # so we don't need a no-op round-trip to the wrong host.
            if model_a != model_b and host_a == host_b:
                state.log(
                    f"[AB] unloading model A ({model_a}) on "
                    f"{host_a or 'localhost'} before B"
                )
                _unload_ollama_model(model_a, host=host_a)
            elif host_a != host_b:
                state.log(
                    f"[AB] cross-host run: A on {host_a or 'localhost'}, "
                    f"B on {host_b or 'localhost'} — skipping eviction"
                )

            # Reset the worktree to clean main before B. We reset the
            # *worktree*, not the main checkout — the main checkout is
            # never touched by the inner runs and stays clean throughout.
            if not _reset_to_main(work_root, state):
                state.log("[AB] failed to reset to main between runs — aborting")
                ab_repo.record_run_end(
                    run_a_id,
                    outcome_a.status,
                    branch_name=outcome_a.branch_name or None,
                    commit_sha=outcome_a.commit_sha or None,
                    failure_log=outcome_a.failure_log or None,
                )
                ab_repo.record_run_end(
                    run_b_id,
                    "failed",
                    failure_log="orchestrator could not reset to main between runs",
                )
                mark_failed(idea_id, state.log_text[-5000:])
                return

            # Cancellation check: if the worker force-cancelled us
            # between attempts (stall, retry, shutdown), don't start B.
            if state.cancelled:
                state.log("[AB] cancelled before Run B — aborting")
                ab_repo.record_run_end(
                    run_a_id,
                    outcome_a.status,
                    branch_name=outcome_a.branch_name or None,
                    commit_sha=outcome_a.commit_sha or None,
                    failure_log=outcome_a.failure_log or None,
                )
                ab_repo.record_run_end(
                    run_b_id,
                    "failed",
                    failure_log="orchestrator cancelled before Run B",
                )
                mark_failed(idea_id, state.log_text[-5000:])
                return

            # ----- Run B -----
            state.log(
                f"=== A/B Run B: {model_b} @ {host_b or 'localhost'} ==="
            )
            if not _ollama_has_model(model_b, host=host_b):
                state.log(
                    f"[AB] model {model_b} not pulled on "
                    f"{host_b or 'localhost'} — short-circuit failure"
                )
                outcome_b.status = "failed"
                outcome_b.failure_log = (
                    f"model not pulled on {host_b or 'localhost'}: "
                    f"run `ollama pull {model_b}` there "
                    f"before enabling AIW_AB_TEST"
                )
            else:
                ok_b, branch_b, sha_b, log_b = _do_attempt(
                    idea_id,
                    model=model_b,
                    suffix=suffix_b,
                    state=state,
                    project_root=work_root,
                    inner_timeout=inner_timeout,
                    host=host_b,
                )
                outcome_b.status = "success" if ok_b else "failed"
                outcome_b.branch_name = branch_b
                outcome_b.commit_sha = sha_b
                outcome_b.log_text = log_b
                if not ok_b:
                    outcome_b.failure_log = log_b
                if ok_b and branch_b:
                    outcome_b.diff = _capture_diff(work_root, branch_b)
                    push_ok, push_msg = _push_branch(state, branch_b, work_root)
                    if not push_ok:
                        outcome_b.status = "failed"
                        outcome_b.failure_log = (
                            outcome_b.failure_log + f"\n[AB-push] {push_msg}"
                        ).strip()

            # ----- Score both -----
            from aiv.scorer import score as _aiv_score

            story_dict = _build_story_dict(idea)
            for outcome in (outcome_a, outcome_b):
                if outcome.status != "success":
                    continue
                try:
                    outcome.scores = _aiv_score(
                        story_dict,
                        outcome.diff,
                        outcome.log_text,
                    )
                    state.log(
                        f"[AB] scored {outcome.label}: "
                        f"err={getattr(outcome.scores, 'error', '')!r}"
                    )
                except Exception as exc:  # noqa: BLE001
                    state.log(f"[AB] scoring {outcome.label} raised: {exc}")
                    outcome.scores = None

            # ----- Persist run rows with scores -----
            ab_repo.record_run_end(
                run_a_id,
                outcome_a.status,
                branch_name=outcome_a.branch_name or None,
                commit_sha=outcome_a.commit_sha or None,
                failure_log=outcome_a.failure_log or None,
                scores=_scores_to_dict(outcome_a.scores),
            )
            ab_repo.record_run_end(
                run_b_id,
                outcome_b.status,
                branch_name=outcome_b.branch_name or None,
                commit_sha=outcome_b.commit_sha or None,
                failure_log=outcome_b.failure_log or None,
                scores=_scores_to_dict(outcome_b.scores),
            )

            # ----- Compare -----
            from aiv.ab_compare import compare as _ab_compare

            try:
                comparison = _ab_compare(
                    story_dict,
                    outcome_a.to_compare_dict(),
                    outcome_b.to_compare_dict(),
                )
            except Exception as exc:  # noqa: BLE001
                state.log(f"[AB] compare raised: {exc}")
                from aiv.ab_compare import ABComparison
                comparison = ABComparison.sentinel("llm_error")

            state.log(
                f"[AB] comparison: winner={comparison.winner!r} "
                f"err={comparison.error!r}"
            )

            # Cancellation check: don't merge a winner if we were
            # cancelled — the worker will retry the story fresh.
            if state.cancelled:
                state.log("[AB] cancelled before merge — skipping winner merge")
                mark_failed(idea_id, state.log_text[-5000:])
                return

            # ----- Pick merge winner (priority rule) -----
            merge_pick = ab_repo.pick_winner(outcome_a.status, outcome_b.status)
            merged_run_id: str | None = None
            winning_outcome: _RunOutcome | None = None
            if merge_pick == "model_a":
                ok_merge, msg = _merge_winner(state, outcome_a.branch_name, idea_id, project_root)
                if ok_merge:
                    merged_run_id = run_a_id
                    winning_outcome = outcome_a
            elif merge_pick == "model_b":
                ok_merge, msg = _merge_winner(state, outcome_b.branch_name, idea_id, project_root)
                if ok_merge:
                    merged_run_id = run_b_id
                    winning_outcome = outcome_b
            else:
                state.log("[AB] both runs failed — no merge")

            # Hand the winner off to the AIV validation queue so the
            # /quality page eventually shows real scores for this story.
            # Without this, A/B merges would land on main but never get
            # graded — the regular executor's enqueue is bypassed by the
            # A/B path. We pass the winner's captured diff and pytest
            # output so the daemon doesn't need to re-derive them.
            if merged_run_id and winning_outcome is not None:
                try:
                    from agent.aiv_hook import enqueue_for_validation as _enq
                    diff_paths = _diff_paths_from_text(winning_outcome.diff)
                    merge_sha = _head_sha(project_root)
                    
                    # Retry logic for AIV enqueue with exponential backoff
                    max_retries = 3
                    backoff_times = [1, 3, 7]  # seconds
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            _enq(
                                idea_id,
                                diff_paths,
                                merge_commit_sha=merge_sha,
                                verification_output=(winning_outcome.log_text or "")[-6000:],
                            )
                            state.log(
                                f"[AB] enqueued {idea_id} for AIV validation "
                                f"(sha={merge_sha[:7] if merge_sha else 'none'})"
                            )
                            break  # Success, exit retry loop
                        except Exception as exc:  # noqa: BLE001
                            retry_count += 1
                            if retry_count < max_retries:
                                backoff_time = backoff_times[retry_count - 1]
                                state.log(
                                    f"[AB] AIV enqueue failed (attempt {retry_count}/{max_retries}), "
                                    f"retrying in {backoff_time}s: {exc}"
                                )
                                time.sleep(backoff_time)
                            else:
                                # All retries exhausted, log to AIV daemon logger and record failure
                                from agent import aiv_schema
                                from agent.aiv_hook import log as aiv_hook_log
                                aiv_hook_log.warning(
                                    "AIV enqueue failed after %d attempts for story %s: %s",
                                    max_retries, idea_id, exc
                                )
                                # Record the failure in the new table for manual retry
                                try:
                                    conn = aiv_schema._get_conn()
                                    conn.execute(
                                        "INSERT INTO aiv_enqueue_failures "
                                        "(story_key, attempted_at, error, merge_commit_sha) "
                                        "VALUES (?, ?, ?, ?)",
                                        (
                                            idea_id,
                                            datetime.now(timezone.utc).isoformat(),
                                            str(exc),
                                            merge_sha,
                                        ),
                                    )
                                    conn.commit()
                                except Exception as db_exc:  # noqa: BLE001
                                    aiv_hook_log.error(
                                        "Failed to record enqueue failure for %s in aiv_enqueue_failures: %s",
                                        idea_id, db_exc
                                    )
                                state.log(
                                    f"[AB] AIV enqueue failed after {max_retries} attempts for {idea_id}: {exc}"
                                )
                except Exception as exc:  # noqa: BLE001
                    state.log(f"[AB] AIV enqueue failed (non-fatal): {exc}")

            ab_repo.record_pair(
                idea_id,
                run_a_id,
                run_b_id,
                comparison_winner=comparison.winner or "",
                comparison_reasoning=comparison.reasoning or "",
                delta_axes=comparison.delta_axes or {},
                merged_run_id=merged_run_id,
                comparison_error=comparison.error or "",
            )

            if merged_run_id:
                mark_done(idea_id, state.log_text[-5000:])
            else:
                mark_failed(
                    idea_id,
                    "A/B run produced no winner: " + state.log_text[-5000:],
                )

        except Exception as exc:  # noqa: BLE001
            state.log(f"[AB] orchestrator raised: {exc}")
            try:
                ab_repo.record_run_end(run_a_id, outcome_a.status or "failed",
                                       failure_log=str(exc)[:1000])
                ab_repo.record_run_end(run_b_id, outcome_b.status or "failed",
                                       failure_log=str(exc)[:1000])
            except Exception:
                pass
            mark_failed(idea_id, state.log_text[-5000:])
        finally:
            # Evict model B at run end — keeps VRAM clean for the next
            # AIM cycle (which starts with model A again). Skipped when
            # A == B since there's nothing distinct to unload. Also a
            # courtesy unload on the *remote* host when B is on a
            # different machine — frees the 5090's VRAM for whatever
            # else the user is doing on it.
            if model_b and model_a != model_b:
                state.log(
                    f"[AB] unloading model B ({model_b}) on "
                    f"{host_b or 'localhost'} at run end"
                )
                _unload_ollama_model(model_b, host=host_b)

            # Tear down the per-orchestrator worktree. ``--force`` because
            # the inner runs may have left uncommitted state we don't care
            # about (any branch the orchestrator wanted to keep was already
            # pushed). Failures are logged but never raise — the worker
            # depends on this finally completing.
            if worktree_root is not None:
                try:
                    ab_worktree.remove_worktree(project_root, worktree_root)
                    state.log(f"[AB] removed worktree: {worktree_root}")
                except Exception as exc:  # noqa: BLE001
                    state.log(f"[AB] worktree removal raised (non-fatal): {exc}")

            _active.pop(idea_id, None)
            _ab_orchestrator_active.pop(idea_id, None)

    thread = threading.Thread(target=_run, daemon=True, name=f"ab-executor-{idea_id}")
    thread.start()
    state.thread = thread
    return state
