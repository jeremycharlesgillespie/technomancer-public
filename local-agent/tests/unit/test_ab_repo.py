"""Tests for ``idea_board.ab_repo`` — the DB + git seam for A/B runs.

Covers:

- Pure helpers: ``model_label``, ``branch_suffix``, ``pick_winner``.
- DB writers: ``record_run_start``, ``record_run_end``, ``record_pair``
  round-trip cleanly into ``ab_test_runs`` / ``ab_test_pairs``.
- ``push_branch_to_both_repos`` calls ``git push origin <branch>`` and
  invokes ``publish.py --branch`` correctly; failures in either step
  surface as ``(False, message)``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import aiv_schema, ab_schema
from idea_board import ab_repo


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_model_label_strips_tag_suffix() -> None:
    assert ab_repo.model_label("qwen3-coder:30b-a3b-q4_K_M") == "qwen3-coder"
    assert ab_repo.model_label("qwen2.5-coder:32b-instruct-q5_K_M") == "qwen2.5-coder"


def test_model_label_handles_no_tag() -> None:
    assert ab_repo.model_label("plain") == "plain"


def test_model_label_empty_input() -> None:
    assert ab_repo.model_label("") == ""
    assert ab_repo.model_label(None) == ""  # type: ignore[arg-type]


def test_branch_suffix_replaces_special_chars() -> None:
    assert (
        ab_repo.branch_suffix("qwen2.5-coder:32b-instruct-q5_K_M")
        == "qwen2_5-coder_32b-instruct-q5_K_M"
    )


def test_branch_suffix_empty() -> None:
    assert ab_repo.branch_suffix("") == ""


# pick_winner truth table -----------------------------------------------------

def test_pick_winner_both_succeed_incumbent_a_wins() -> None:
    assert ab_repo.pick_winner("success", "success") == "model_a"


def test_pick_winner_a_succeeds_b_fails() -> None:
    assert ab_repo.pick_winner("success", "failed") == "model_a"


def test_pick_winner_a_fails_b_succeeds() -> None:
    assert ab_repo.pick_winner("failed", "success") == "model_b"


def test_pick_winner_both_fail_returns_empty() -> None:
    assert ab_repo.pick_winner("failed", "failed") == ""


def test_pick_winner_incumbent_b_inverts_priority() -> None:
    assert ab_repo.pick_winner("success", "success", incumbent="model_b") == "model_b"
    assert ab_repo.pick_winner("failed", "success", incumbent="model_b") == "model_b"
    assert ab_repo.pick_winner("success", "failed", incumbent="model_b") == "model_a"


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------

def test_record_run_start_inserts_running_row() -> None:
    ab_repo.record_run_start(
        run_id="run-1",
        story_key="TK-1",
        model="qwen3-coder:30b-a3b-q4_K_M",
        label="qwen3-coder",
    )
    conn = aiv_schema._get_conn()
    row = conn.execute(
        "SELECT * FROM ab_test_runs WHERE run_id = ?", ("run-1",)
    ).fetchone()
    assert row is not None
    assert row["story_key"] == "TK-1"
    assert row["model"] == "qwen3-coder:30b-a3b-q4_K_M"
    assert row["model_label"] == "qwen3-coder"
    assert row["status"] == "running"
    assert row["started_at"]


def test_record_run_end_writes_status_and_branch() -> None:
    ab_repo.record_run_start("run-2", "TK-2", "m", "label")
    ab_repo.record_run_end(
        "run-2",
        "success",
        branch_name="2026-04-25-TK-2-x",
        commit_sha="abc1234",
    )
    conn = aiv_schema._get_conn()
    row = conn.execute(
        "SELECT * FROM ab_test_runs WHERE run_id = ?", ("run-2",)
    ).fetchone()
    assert row["status"] == "success"
    assert row["branch_name"] == "2026-04-25-TK-2-x"
    assert row["commit_sha"] == "abc1234"
    assert row["ended_at"]


def test_record_run_end_persists_scores() -> None:
    ab_repo.record_run_start("run-3", "TK-3", "m", "label")
    ab_repo.record_run_end(
        "run-3",
        "success",
        scores={
            "meets_requirements": 9,
            "code_quality": 8,
            "test_quality": 7,
            "security_safety": 10,
            "scope_discipline": 9,
            "edge_cases": 6,
            "product_impact": 8,
            "overall_score": 8.1,
            "red_flags": ["scope-creep"],
            "reasoning_map": {"code_quality": "looks fine"},
        },
    )
    conn = aiv_schema._get_conn()
    row = conn.execute(
        "SELECT * FROM ab_test_runs WHERE run_id = ?", ("run-3",)
    ).fetchone()
    assert row["meets_requirements"] == 9
    assert row["code_quality"] == 8
    assert row["overall_score"] == pytest.approx(8.1)
    assert json.loads(row["red_flags_json"]) == ["scope-creep"]
    assert json.loads(row["reasoning_json"]) == {"code_quality": "looks fine"}


def test_record_run_end_truncates_long_failure_log() -> None:
    ab_repo.record_run_start("run-4", "TK-4", "m", "label")
    huge = "x" * 10_000
    ab_repo.record_run_end("run-4", "failed", failure_log=huge)
    conn = aiv_schema._get_conn()
    row = conn.execute(
        "SELECT failure_log FROM ab_test_runs WHERE run_id = ?", ("run-4",)
    ).fetchone()
    assert row["failure_log"] is not None
    assert len(row["failure_log"]) == 5000  # last 5000 chars only


def test_record_pair_returns_pair_id_and_persists() -> None:
    ab_repo.record_run_start("ra", "TK-5", "m", "a")
    ab_repo.record_run_start("rb", "TK-5", "m", "b")
    pair_id = ab_repo.record_pair(
        "TK-5",
        "ra",
        "rb",
        comparison_winner="model_a",
        comparison_reasoning="A's tests are stronger",
        delta_axes={"code_quality": 1, "test_quality": 2},
        merged_run_id="ra",
    )
    assert pair_id > 0
    conn = aiv_schema._get_conn()
    row = conn.execute(
        "SELECT * FROM ab_test_pairs WHERE pair_id = ?", (pair_id,)
    ).fetchone()
    assert row["story_key"] == "TK-5"
    assert row["model_a_run_id"] == "ra"
    assert row["model_b_run_id"] == "rb"
    assert row["comparison_winner"] == "model_a"
    assert row["merged_run_id"] == "ra"
    assert json.loads(row["delta_axes_json"]) == {
        "code_quality": 1,
        "test_quality": 2,
    }


# ---------------------------------------------------------------------------
# push_branch_to_both_repos
# ---------------------------------------------------------------------------

def _make_completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_push_branch_to_both_repos_success(tmp_path: Path) -> None:
    publish_script = tmp_path / "publish.py"
    publish_script.write_text("# fake")

    with patch("idea_board.ab_repo.subprocess.run") as mrun:
        mrun.side_effect = [
            _make_completed(0, stdout="Everything up-to-date"),  # private push
            _make_completed(0, stdout="published"),               # publish.py
        ]
        ok, msg = ab_repo.push_branch_to_both_repos(
            "feature-branch",
            private_repo=tmp_path,
            publish_script=publish_script,
        )
    assert ok is True
    assert "feature-branch" in msg
    # First call: private push.
    private_call = mrun.call_args_list[0]
    assert private_call.args[0][:3] == ["git", "push", "origin"]
    assert private_call.args[0][-1] == "feature-branch"
    # Second call: publish.py invocation.
    publish_call = mrun.call_args_list[1]
    pub_argv = publish_call.args[0]
    assert str(publish_script) in pub_argv
    assert "--branch" in pub_argv
    assert "feature-branch" in pub_argv


def test_push_branch_private_failure_short_circuits(tmp_path: Path) -> None:
    publish_script = tmp_path / "publish.py"
    publish_script.write_text("# fake")
    with patch("idea_board.ab_repo.subprocess.run") as mrun:
        mrun.side_effect = [_make_completed(1, stderr="rejected")]
        ok, msg = ab_repo.push_branch_to_both_repos(
            "br",
            private_repo=tmp_path,
            publish_script=publish_script,
        )
    assert ok is False
    assert "private push failed" in msg
    assert mrun.call_count == 1


def test_push_branch_public_failure_returns_false(tmp_path: Path) -> None:
    publish_script = tmp_path / "publish.py"
    publish_script.write_text("# fake")
    with patch("idea_board.ab_repo.subprocess.run") as mrun:
        mrun.side_effect = [
            _make_completed(0),                          # private OK
            _make_completed(1, stderr="secrets found"),  # public fails
        ]
        ok, msg = ab_repo.push_branch_to_both_repos(
            "br",
            private_repo=tmp_path,
            publish_script=publish_script,
        )
    assert ok is False
    assert "public push failed" in msg


def test_push_branch_missing_publish_script(tmp_path: Path) -> None:
    publish_script = tmp_path / "missing.py"  # not created
    with patch("idea_board.ab_repo.subprocess.run") as mrun:
        mrun.side_effect = [_make_completed(0)]  # private push OK
        ok, msg = ab_repo.push_branch_to_both_repos(
            "br",
            private_repo=tmp_path,
            publish_script=publish_script,
        )
    assert ok is False
    assert "publish.py not found" in msg
