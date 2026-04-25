"""Tests for the /quality dashboard A/B extensions and /quality/ab/<key>.

Coverage:

- Stories without an ``ab_test_pairs`` row render as a single row
  (existing behavior unchanged).
- A story with an A/B pair renders two grouped rows wrapped in
  ``<tbody class="ab-pair">`` plus a footer row carrying the winner
  label and a link to ``/quality/ab/<story_key>``.
- The detail route renders winner, reasoning, both run summaries, and
  the per-axis delta table.
- Missing pair → 404.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent import aiv_schema, ab_schema
from idea_board.web import app


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


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _insert_quality_row(story_key: str, title: str = "Some story") -> None:
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO story_quality (
            story_key, story_title, merged_at, validated_at,
            meets_requirements, code_quality, test_quality, security_safety,
            scope_discipline, edge_cases, product_impact, overall_score,
            red_flags_json, verification_method, verification_output,
            reasoning_json, error
        ) VALUES (?, ?, ?, ?, 9, 8, 7, 10, 9, 6, 8, 8.1, ?, 'tests-only', '', '{}', NULL)
        """,
        (story_key, title, now, now, json.dumps([])),
    )
    conn.commit()


def _insert_ab_pair(
    story_key: str,
    *,
    winner: str = "model_a",
    reasoning: str = "A had stronger tests.",
    merged_run_id: str | None = "ra",
) -> None:
    ab_schema.init_ab_db()
    conn = aiv_schema._get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO ab_test_runs
            (run_id, story_key, model, model_label, branch_name, commit_sha,
             started_at, ended_at, status, overall_score,
             meets_requirements, code_quality, test_quality, security_safety,
             scope_discipline, edge_cases, product_impact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'success', 8.2,
                9, 8, 7, 10, 9, 6, 8)
        """,
        ("ra", story_key, "qwen3-coder:30b-a3b-q4_K_M", "qwen3-coder",
         "br-a", "aaaa1234", now, now),
    )
    conn.execute(
        """
        INSERT INTO ab_test_runs
            (run_id, story_key, model, model_label, branch_name, commit_sha,
             started_at, ended_at, status, overall_score,
             meets_requirements, code_quality, test_quality, security_safety,
             scope_discipline, edge_cases, product_impact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'success', 7.4,
                8, 7, 6, 10, 8, 5, 7)
        """,
        ("rb", story_key, "qwen2.5-coder:32b-instruct-q5_K_M", "qwen2.5-coder",
         "br-b", "bbbb5678", now, now),
    )
    conn.execute(
        """
        INSERT INTO ab_test_pairs
            (story_key, model_a_run_id, model_b_run_id,
             comparison_winner, comparison_reasoning, delta_axes_json,
             merged_run_id, comparison_error, created_at)
        VALUES (?, 'ra', 'rb', ?, ?, ?, ?, '', ?)
        """,
        (story_key, winner, reasoning,
         json.dumps({"code_quality": 1, "test_quality": 1}),
         merged_run_id, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# /quality page rendering
# ---------------------------------------------------------------------------

def test_non_ab_story_renders_as_single_row(client) -> None:
    _insert_quality_row("TK-1")
    resp = client.get("/quality")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "TK-1" in body
    # No A/B markers in the rendered table (CSS class lives in <style>, but
    # no <tbody class='ab-pair'> wrapper or detail link is emitted).
    assert "<tbody class='ab-pair'>" not in body
    assert "/quality/ab/TK-1" not in body


def test_ab_story_renders_paired_rows_with_footer(client) -> None:
    _insert_quality_row("TK-2", title="Big change")
    _insert_ab_pair("TK-2", winner="model_a")
    resp = client.get("/quality")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "TK-2" in body
    assert "<tbody class='ab-pair'>" in body
    assert "qwen3-coder" in body
    assert "qwen2.5-coder" in body
    assert "/quality/ab/TK-2" in body
    assert "Winner" in body


def test_ab_story_marks_winner_class(client) -> None:
    _insert_quality_row("TK-3")
    _insert_ab_pair("TK-3", merged_run_id="ra")
    body = client.get("/quality").get_data(as_text=True)
    # The winner row carries the ab-winner class on a <tr>.
    assert "ab-row ab-winner" in body


def test_ab_story_no_merged_winner_renders_without_winner_class(client) -> None:
    """Both-failed scenario: merged_run_id is empty, no row is marked winner."""
    _insert_quality_row("TK-4")
    _insert_ab_pair("TK-4", winner="both_failed", merged_run_id=None)
    body = client.get("/quality").get_data(as_text=True)
    assert "<tbody class='ab-pair'>" in body
    # No <tr class='ab-row ab-winner'> when neither side merged.
    assert "ab-row ab-winner" not in body


# ---------------------------------------------------------------------------
# /quality/ab/<story_key> detail route
# ---------------------------------------------------------------------------

def test_ab_detail_renders_winner_and_reasoning(client) -> None:
    _insert_quality_row("TK-5")
    _insert_ab_pair("TK-5", winner="model_a", reasoning="A's tests are stronger.")
    resp = client.get("/quality/ab/TK-5")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "TK-5" in body
    assert "model_a" in body
    assert "A&#39;s tests are stronger" in body or "tests are stronger" in body
    assert "qwen3-coder" in body
    assert "qwen2.5-coder" in body
    # Per-axis delta table renders.
    assert "Per-axis delta" in body or "delta" in body.lower()


def test_ab_detail_404_when_no_pair(client) -> None:
    _insert_quality_row("TK-6")  # has a quality row but no AB pair
    resp = client.get("/quality/ab/TK-6")
    assert resp.status_code == 404


def test_ab_detail_renders_branch_and_commit(client) -> None:
    _insert_quality_row("TK-7")
    _insert_ab_pair("TK-7")
    body = client.get("/quality/ab/TK-7").get_data(as_text=True)
    assert "br-a" in body
    assert "br-b" in body
    assert "aaaa1234" in body
    assert "bbbb5678" in body
