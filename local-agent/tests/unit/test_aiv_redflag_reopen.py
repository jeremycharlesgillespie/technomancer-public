"""Tests for the AIV red-flag re-open gate (TK-694).

The red-flag gate is independent of the below-threshold gate: any
non-empty ``red_flags`` list on a scored story fires a re-open when
``aiv_reopen_on_any_red_flag`` is True — regardless of overall_score.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from filelock import FileLock

from agent import aiv_schema
from aiv import main as aiv_main
from aiv import state as aiv_state
from aiv.scorer import StoryQualityScores


# ---------------------------------------------------------------------------
# Fixtures — mirror test_aiv_threshold.py so both gates share isolation.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "aiv_state"
    state_dir.mkdir()
    monkeypatch.setattr(aiv_state, "STATE_DIR", state_dir)
    monkeypatch.setattr(aiv_state, "STATE_FILE", state_dir / ".aiv_state.json")
    monkeypatch.setattr(aiv_state, "LOCK_FILE", state_dir / ".aiv_state.lock")
    monkeypatch.setattr(aiv_state, "PID_FILE", state_dir / "aiv.pid")
    monkeypatch.setattr(
        aiv_state,
        "_lock",
        FileLock(str(state_dir / ".aiv_state.lock"), timeout=10),
    )


@pytest.fixture(autouse=True)
def _isolate_decisions_log(tmp_path, monkeypatch):
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(aiv_main, "DECISIONS_LOG_FILE", log_file)
    return log_file


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    """Start every test with both gates disabled so each opts in explicitly."""
    monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", None)
    monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", False)


def _high_score_with_red_flag() -> StoryQualityScores:
    """All axes at 9 (mean 9.0) with one red flag — triggers ONLY red-flag gate."""
    return StoryQualityScores(
        meets_requirements=9,
        code_quality=9,
        test_quality=9,
        security_safety=9,
        scope_discipline=9,
        edge_cases=9,
        product_impact=9,
        reasoning_map={"meets_requirements": "looks good"},
        red_flags=["no_tests_added"],
        error="",
    )


def _high_score_no_red_flags() -> StoryQualityScores:
    """All axes at 9, no red flags — should never fire either gate."""
    return StoryQualityScores(
        meets_requirements=9,
        code_quality=9,
        test_quality=9,
        security_safety=9,
        scope_discipline=9,
        edge_cases=9,
        product_impact=9,
        reasoning_map={},
        red_flags=[],
        error="",
    )


def _low_score_with_red_flags() -> StoryQualityScores:
    """Low overall + multiple red flags — lets us assert multi-flag listing."""
    return StoryQualityScores(
        meets_requirements=3,
        code_quality=4,
        test_quality=2,
        security_safety=5,
        scope_discipline=3,
        edge_cases=2,
        product_impact=4,
        reasoning_map={"meets_requirements": "missing criteria"},
        red_flags=["no_tests_added", "secret_leak"],
        error="",
    )


def _ok_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


# ---------------------------------------------------------------------------
# Acceptance #1: flag=True + story with any red flag → re-open fires even
# when overall=9.
# ---------------------------------------------------------------------------

class TestRedFlagFlagEnabled:
    """The teeth: a red flag is enough to trigger re-open."""

    def test_any_red_flag_fires_reopen_even_when_score_is_high(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            mock_api.side_effect = [
                _ok_response(204),  # PUT label
                _ok_response(201),  # POST comment
            ]

            fired = aiv_main.maybe_reopen_for_red_flags(
                _high_score_with_red_flag(), "TK-700"
            )

        assert fired is True

        # PUT to add pending-approval, then POST to comment endpoint.
        assert mock_api.call_count == 2

        put_args, put_kwargs = mock_api.call_args_list[0]
        assert put_args[0] == "put"
        assert put_args[1] == "/issue/TK-700"
        assert put_kwargs["json"] == {
            "update": {"labels": [{"add": "pending-approval"}]}
        }

        post_args, post_kwargs = mock_api.call_args_list[1]
        assert post_args[0] == "post"
        assert post_args[1] == "/issue/TK-700/comment"
        comment_text = post_kwargs["json"]["body"]["content"][0]["content"][0]["text"]
        assert "red flag" in comment_text.lower()
        assert "no_tests_added" in comment_text

        # Decision log written with the right shape.
        log_lines = _isolate_decisions_log.read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["story_key"] == "TK-700"
        assert entry["action"] == "reopen_red_flags"
        assert entry["red_flags"] == ["no_tests_added"]
        assert entry["label_applied"] is True
        assert entry["comment_posted"] is True

    def test_multiple_red_flags_all_listed_in_comment(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            mock_api.side_effect = [_ok_response(204), _ok_response(201)]

            fired = aiv_main.maybe_reopen_for_red_flags(
                _low_score_with_red_flags(), "TK-701"
            )

        assert fired is True
        post_kwargs = mock_api.call_args_list[1][1]
        comment_text = post_kwargs["json"]["body"]["content"][0]["content"][0]["text"]
        assert "no_tests_added" in comment_text
        assert "secret_leak" in comment_text

        entry = json.loads(
            _isolate_decisions_log.read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["red_flags"] == ["no_tests_added", "secret_leak"]


# ---------------------------------------------------------------------------
# Acceptance #2: flag=False → no mutations even with red flags present.
# ---------------------------------------------------------------------------

class TestRedFlagFlagDisabled:
    """Default config keeps AIV observation-only for red flags."""

    def test_flag_disabled_skips_reopen_even_with_red_flags(
        self, _isolate_decisions_log
    ):
        # _reset_config leaves aiv_reopen_on_any_red_flag=False.
        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_red_flags(
                _low_score_with_red_flags(), "TK-702"
            )

        assert fired is False
        mock_api.assert_not_called()
        assert not _isolate_decisions_log.exists()


# ---------------------------------------------------------------------------
# Flag enabled but no red flags → no mutations.
# ---------------------------------------------------------------------------

class TestNoRedFlags:
    def test_flag_enabled_but_no_red_flags_skips_reopen(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_red_flags(
                _high_score_no_red_flags(), "TK-703"
            )

        assert fired is False
        mock_api.assert_not_called()
        assert not _isolate_decisions_log.exists()


# ---------------------------------------------------------------------------
# Jira not configured → the decision is still logged; API calls short-circuit.
# ---------------------------------------------------------------------------

class TestJiraNotConfigured:
    def test_jira_unconfigured_logs_decision_with_false_flags(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)

        with patch("aiv.main.is_jira_configured", return_value=False), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_red_flags(
                _high_score_with_red_flag(), "TK-704"
            )

        assert fired is True
        mock_api.assert_not_called()

        entry = json.loads(
            _isolate_decisions_log.read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["label_applied"] is False
        assert entry["comment_posted"] is False
        assert entry["red_flags"] == ["no_tests_added"]


# ---------------------------------------------------------------------------
# process_row integration: persist + red-flag gate.
# ---------------------------------------------------------------------------

def _seed_pending(story_key: str, paths=None) -> None:
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    conn.execute(
        "INSERT INTO aiv_pending "
        "(story_key, merged_at, diff_paths_json, enqueued_at) "
        "VALUES (?, ?, ?, ?)",
        (
            story_key,
            "2026-04-18T13:35:00",
            json.dumps(list(paths or [])),
            "2026-04-18T13:36:00",
        ),
    )
    conn.commit()


class TestProcessRowRedFlagIntegration:
    """End-to-end: process_row persists, then triggers the red-flag gate."""

    def test_high_score_with_red_flag_through_process_row_reopens(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)
        _seed_pending("TK-710", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_high_score_with_red_flag()), \
             patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            mock_api.side_effect = [_ok_response(204), _ok_response(201)]
            result = aiv_main.process_row(row)

        assert result is not None
        # PUT label + POST comment from the red-flag gate.
        assert mock_api.call_count == 2

        # Persist still happened.
        conn = aiv_schema._get_conn()
        quality_row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-710",)
        ).fetchone()
        assert quality_row is not None

        log_lines = _isolate_decisions_log.read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["action"] == "reopen_red_flags"

    def test_red_flag_gate_failure_does_not_break_process_row(
        self, monkeypatch
    ):
        """A flaky Jira call in the red-flag gate must not undo persist."""
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_on_any_red_flag", True)
        _seed_pending("TK-711", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_high_score_with_red_flag()), \
             patch(
                 "aiv.main.maybe_reopen_for_red_flags",
                 side_effect=RuntimeError("jira blew up"),
             ):
            result = aiv_main.process_row(row)

        assert result is not None
        conn = aiv_schema._get_conn()
        quality_row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-711",)
        ).fetchone()
        assert quality_row is not None
