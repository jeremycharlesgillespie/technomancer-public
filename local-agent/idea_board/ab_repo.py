"""A/B Repo — DB persistence + dual-repo push helpers for the A/B harness.

Single mockable seam between :mod:`idea_board.ab_executor` and the
outside world (SQLite, git, ``publish.py``). Tests stub these functions
to drive every scenario without touching the filesystem.

Functions
---------

- :func:`record_run_start` — INSERT a row into ``ab_test_runs`` when a
  model begins its attempt.
- :func:`record_run_end` — UPDATE the row with status, branch, commit,
  failure log, and (optionally) the seven AIV axis scores.
- :func:`record_pair` — INSERT a row into ``ab_test_pairs`` after both
  runs are scored and the comparison verdict is in.
- :func:`pick_winner` — pure function applying the priority rule
  (model A wins by default; model B only wins when A failed and B
  succeeded). Returns ``"model_a"``, ``"model_b"``, or ``""`` (no
  merge).
- :func:`push_branch_to_both_repos` — push a feature branch to private
  origin, then mirror it to ``technomancer-public`` via
  ``publish.py --branch``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.aiv_schema import _get_conn, SCORE_COLUMNS
from agent.ab_schema import init_ab_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def model_label(model_tag: str) -> str:
    """Derive a short display label from an Ollama model tag.

    Examples
    --------
    >>> model_label("qwen3-coder:30b-a3b-q4_K_M")
    'qwen3-coder'
    >>> model_label("qwen2.5-coder:32b-instruct-q5_K_M")
    'qwen2.5-coder'
    """
    if not isinstance(model_tag, str) or not model_tag:
        return ""
    return model_tag.split(":", 1)[0]


def branch_suffix(model_tag: str) -> str:
    """Sanitize a model tag into a branch-name suffix.

    Branches can't contain ``:`` or repeated ``.`` — replace both with
    ``_`` so ``qwen2.5-coder:32b-instruct-q5_K_M`` becomes
    ``qwen2_5-coder_32b-instruct-q5_K_M``. We keep the full tag (not the
    label) so two A/B runs on the same day, same story, same label-pair
    don't collide.
    """
    if not isinstance(model_tag, str) or not model_tag:
        return ""
    return model_tag.replace(":", "_").replace(".", "_")


def pick_winner(
    status_a: str,
    status_b: str,
    *,
    incumbent: str = "model_a",
) -> str:
    """Return the run id label of the merge winner, or ``""`` for no merge.

    Priority rule (incumbent = ``model_a`` by default):
      - A succeeded, B succeeded -> A wins (incumbent priority).
      - A succeeded, B failed    -> A wins.
      - A failed,    B succeeded -> B wins.
      - A failed,    B failed    -> "" (no merge).

    Symmetric for the rare ``incumbent="model_b"`` override.
    """
    a_ok = status_a == "success"
    b_ok = status_b == "success"

    if not a_ok and not b_ok:
        return ""

    challenger = "model_b" if incumbent == "model_a" else "model_a"
    if incumbent == "model_a":
        return "model_a" if a_ok else "model_b"
    return "model_b" if b_ok else "model_a"


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def record_run_start(
    run_id: str,
    story_key: str,
    model: str,
    label: str,
) -> None:
    """INSERT a fresh ``ab_test_runs`` row in the ``running`` state."""
    init_ab_db()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO ab_test_runs
            (run_id, story_key, model, model_label, started_at, status)
        VALUES (?, ?, ?, ?, ?, 'running')
        """,
        (run_id, story_key, model, label, _now_iso()),
    )
    conn.commit()


def record_run_end(
    run_id: str,
    status: str,
    *,
    branch_name: str | None = None,
    commit_sha: str | None = None,
    failure_log: str | None = None,
    scores: dict[str, Any] | None = None,
) -> None:
    """UPDATE the run row with terminal state.

    ``scores`` is the dict produced by :class:`aiv.scorer.StoryQualityScores`
    (or any duck-typed equivalent); the seven axis ints, ``overall_score``,
    ``red_flags_json``, ``reasoning_json``, and ``scoring_error`` are
    written when present and ignored otherwise.
    """
    init_ab_db()
    conn = _get_conn()
    set_clauses = ["status = ?", "ended_at = ?"]
    params: list[Any] = [status, _now_iso()]

    if branch_name is not None:
        set_clauses.append("branch_name = ?")
        params.append(branch_name)
    if commit_sha is not None:
        set_clauses.append("commit_sha = ?")
        params.append(commit_sha)
    if failure_log is not None:
        # Last 5000 chars only — full logs blow up the SQLite row size.
        set_clauses.append("failure_log = ?")
        params.append(failure_log[-5000:])

    if scores:
        for axis in SCORE_COLUMNS:
            v = scores.get(axis)
            if isinstance(v, int):
                set_clauses.append(f"{axis} = ?")
                params.append(v)
        if "overall_score" in scores and scores["overall_score"] is not None:
            set_clauses.append("overall_score = ?")
            params.append(float(scores["overall_score"]))
        if "red_flags" in scores and scores["red_flags"] is not None:
            set_clauses.append("red_flags_json = ?")
            params.append(json.dumps(scores["red_flags"]))
        if "reasoning_map" in scores and scores["reasoning_map"] is not None:
            set_clauses.append("reasoning_json = ?")
            params.append(json.dumps(scores["reasoning_map"]))
        if scores.get("error"):
            set_clauses.append("scoring_error = ?")
            params.append(str(scores["error"]))

    params.append(run_id)
    conn.execute(
        f"UPDATE ab_test_runs SET {', '.join(set_clauses)} WHERE run_id = ?",
        params,
    )
    conn.commit()


def end_if_running(
    run_id: str,
    status: str,
    *,
    failure_log: str | None = None,
) -> None:
    """Same as :func:`record_run_end` but a no-op if the row is no
    longer in the ``running`` state.

    Used by orchestrator safety-net paths so they don't clobber an
    ``ended_at`` that the happy-path flow already wrote correctly.
    Status, ``ended_at``, and (optionally) ``failure_log`` only.
    """
    init_ab_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT status FROM ab_test_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None or row["status"] != "running":
        return
    set_clauses = ["status = ?", "ended_at = ?"]
    params: list[Any] = [status, _now_iso()]
    if failure_log is not None:
        set_clauses.append("failure_log = ?")
        params.append(failure_log[-5000:])
    params.append(run_id)
    conn.execute(
        f"UPDATE ab_test_runs SET {', '.join(set_clauses)} WHERE run_id = ?",
        params,
    )
    conn.commit()


def update_run_scores(
    run_id: str,
    scores: dict[str, Any] | None,
) -> None:
    """UPDATE only the score-related columns on an existing run row.

    Used when the run's terminal time was already recorded (so we
    don't want to clobber ``ended_at``) and the AIV scoring pass
    runs later — e.g. orchestrators that close out Run A before
    starting Run B and only score after both runs finish.
    """
    if not scores:
        return
    init_ab_db()
    conn = _get_conn()
    set_clauses: list[str] = []
    params: list[Any] = []
    for axis in SCORE_COLUMNS:
        v = scores.get(axis)
        if isinstance(v, int):
            set_clauses.append(f"{axis} = ?")
            params.append(v)
    if "overall_score" in scores and scores["overall_score"] is not None:
        set_clauses.append("overall_score = ?")
        params.append(float(scores["overall_score"]))
    if "red_flags" in scores and scores["red_flags"] is not None:
        set_clauses.append("red_flags_json = ?")
        params.append(json.dumps(scores["red_flags"]))
    if "reasoning_map" in scores and scores["reasoning_map"] is not None:
        set_clauses.append("reasoning_json = ?")
        params.append(json.dumps(scores["reasoning_map"]))
    if scores.get("error"):
        set_clauses.append("scoring_error = ?")
        params.append(str(scores["error"]))

    if not set_clauses:
        return
    params.append(run_id)
    conn.execute(
        f"UPDATE ab_test_runs SET {', '.join(set_clauses)} WHERE run_id = ?",
        params,
    )
    conn.commit()


def reap_stranded_runs(
    *,
    reason: str = "abandoned: process killed before completion",
) -> int:
    """Mark every ``ab_test_runs`` row stuck in ``running`` as ``failed``.

    Daemon-thread A/B orchestrators die silently when the worker process
    is killed mid-run (SIGTERM, crash, restart). The two ``running`` rows
    they wrote at start are then never closed, which leaves
    ``ab_test_pairs`` empty for that story and the ``/quality`` page never
    renders the comparison.

    This sweep is idempotent and safe to call at every worker startup —
    if no rows are stranded it returns 0 and changes nothing. Returns the
    number of rows updated so the caller can log it.
    """
    init_ab_db()
    conn = _get_conn()
    cursor = conn.execute(
        """
        UPDATE ab_test_runs
           SET status = 'failed',
               ended_at = ?,
               failure_log = COALESCE(failure_log, '') || ?
         WHERE status = 'running'
           AND ended_at IS NULL
        """,
        (_now_iso(), reason),
    )
    conn.commit()
    return cursor.rowcount or 0


def record_pair(
    story_key: str,
    run_a_id: str,
    run_b_id: str,
    *,
    comparison_winner: str,
    comparison_reasoning: str,
    delta_axes: dict[str, int],
    merged_run_id: str | None,
    comparison_error: str = "",
) -> int:
    """INSERT one row into ``ab_test_pairs``. Returns the new ``pair_id``."""
    init_ab_db()
    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO ab_test_pairs (
            story_key,
            model_a_run_id,
            model_b_run_id,
            comparison_winner,
            comparison_reasoning,
            delta_axes_json,
            merged_run_id,
            comparison_error,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            story_key,
            run_a_id,
            run_b_id,
            comparison_winner,
            comparison_reasoning,
            json.dumps(delta_axes),
            merged_run_id,
            comparison_error,
            _now_iso(),
        ),
    )
    conn.commit()
    pair_id = cursor.lastrowid
    return int(pair_id) if pair_id is not None else -1


# ---------------------------------------------------------------------------
# Git / push helpers
# ---------------------------------------------------------------------------

def push_branch_to_both_repos(
    branch_name: str,
    *,
    private_repo: Path,
    publish_script: Path,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Push ``branch_name`` to private origin, then mirror to public.

    The public-side mirror is delegated to ``publish.py --branch <name>``
    so the `local-agent/**`-only filter and the secrets gate are reused.

    Returns
    -------
    (ok, message)
        ``ok`` is True only when *both* pushes succeeded. ``message`` is
        a short human-readable summary suitable for the executor log.
    """
    # Step 1: push to private origin.
    try:
        priv = subprocess.run(
            ["git", "push", "origin", branch_name],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(private_repo),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"private push raised: {exc}"
    if priv.returncode != 0:
        return False, f"private push failed: {(priv.stderr or priv.stdout)[:500]}"

    # Step 2: mirror to public via publish.py --branch.
    if not publish_script.exists():
        return False, f"publish.py not found at {publish_script}"

    try:
        pub = subprocess.run(
            [sys.executable, str(publish_script),
             "--branch", branch_name, "--push", "--force"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(publish_script.parent),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"public push raised: {exc}"
    if pub.returncode != 0:
        return False, f"public push failed: {(pub.stderr or pub.stdout)[:500]}"

    return True, f"pushed {branch_name} to private + public"
