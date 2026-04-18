"""Tests for the AIV below-threshold re-open gate (TK-693)."""

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect aiv_schema at a temp SQLite DB per-test."""
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
    """Redirect AIV state files (aiv.log + .aiv_state.json) into tmp_path."""
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
    """Redirect the decisions.jsonl audit log into tmp_path."""
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(aiv_main, "DECISIONS_LOG_FILE", log_file)
    return log_file


@pytest.fixture(autouse=True)
def _reset_threshold(monkeypatch):
    """Default to threshold=None so individual tests opt in explicitly."""
    monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", None)


def _scores(overall_axes: int) -> StoryQualityScores:
    """Return a StoryQualityScores with every axis set to ``overall_axes``."""
    return StoryQualityScores(
        meets_requirements=overall_axes,
        code_quality=overall_axes,
        test_quality=overall_axes,
        security_safety=overall_axes,
        scope_discipline=overall_axes,
        edge_cases=overall_axes,
        product_impact=overall_axes,
        reasoning_map={"meets_requirements": "ok"},
        red_flags=[],
        error="",
    )


def _mixed_scores() -> StoryQualityScores:
    """Mixed-axis scores whose equal-weight mean is below 7.0.

    Five axes score below the threshold (4, 5, 3, 6, 4); two clear it
    (10, 8). Mean is 40/7 ≈ 5.71 — well below the configured 7.0 — so
    the gate must fire.
    """
    return StoryQualityScores(
        meets_requirements=4,
        code_quality=5,
        test_quality=3,
        security_safety=10,
        scope_discipline=6,
        edge_cases=4,
        product_impact=8,
        reasoning_map={"meets_requirements": "needs work"},
        red_flags=["no_tests_added"],
        error="",
    )


def _ok_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


# ---------------------------------------------------------------------------
# Acceptance: threshold=7.0 + story at 5.5 -> Jira PUT + comment + decisions
# ---------------------------------------------------------------------------

class TestBelowThresholdReopen:
    """The teeth: a low-quality merge gets pending-approval re-applied."""

    def test_below_threshold_fires_jira_put_and_comment_and_log(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)

        # Even when the upstream Jira config is empty in CI, the gate's
        # is_jira_configured() guard must be satisfied or the PUT/POST
        # are skipped before reaching the mock.
        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            mock_api.side_effect = [
                _ok_response(204),  # PUT label
                _ok_response(201),  # POST comment
            ]

            fired = aiv_main.maybe_reopen_for_low_score(_mixed_scores(), "TK-100")

        assert fired is True

        # PUT to add pending-approval, then POST to comment endpoint.
        assert mock_api.call_count == 2

        put_args, put_kwargs = mock_api.call_args_list[0]
        assert put_args[0] == "put"
        assert put_args[1] == "/issue/TK-100"
        assert put_kwargs["json"] == {
            "update": {"labels": [{"add": "pending-approval"}]}
        }

        post_args, post_kwargs = mock_api.call_args_list[1]
        assert post_args[0] == "post"
        assert post_args[1] == "/issue/TK-100/comment"
        comment_body = post_kwargs["json"]["body"]
        comment_text = comment_body["content"][0]["content"][0]["text"]
        # Comment names which axes scored below the threshold.
        assert "below the configured threshold" in comment_text
        assert "meets_requirements" in comment_text
        assert "test_quality" in comment_text
        # Above-threshold axes are NOT listed.
        assert "security_safety" not in comment_text

        # decisions.jsonl entry written with the right shape.
        log_lines = _isolate_decisions_log.read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["story_key"] == "TK-100"
        assert entry["action"] == "reopen_below_threshold"
        assert entry["threshold"] == 7.0
        assert entry["overall_score"] < 7.0
        assert entry["label_applied"] is True
        assert entry["comment_posted"] is True
        flagged_axes = {row["axis"] for row in entry["below_threshold_axes"]}
        assert "meets_requirements" in flagged_axes
        assert "security_safety" not in flagged_axes


# ---------------------------------------------------------------------------
# Acceptance: threshold=None -> no Jira mutations regardless of score
# ---------------------------------------------------------------------------

class TestThresholdDisabled:
    """No threshold configured -> AIV stays observation-only."""

    def test_no_threshold_no_jira_calls_for_low_score(
        self, monkeypatch, _isolate_decisions_log
    ):
        # threshold left at the autouse-fixture default of None.
        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(_mixed_scores(), "TK-200")

        assert fired is False
        mock_api.assert_not_called()
        assert not _isolate_decisions_log.exists()

    def test_no_threshold_no_jira_calls_for_failing_score(
        self, monkeypatch, _isolate_decisions_log
    ):
        # Even an all-zero (worst) score must not mutate Jira when the
        # threshold is disabled.
        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(_scores(0), "TK-201")

        assert fired is False
        mock_api.assert_not_called()
        assert not _isolate_decisions_log.exists()


# ---------------------------------------------------------------------------
# At-or-above threshold -> no mutation
# ---------------------------------------------------------------------------

class TestAtOrAboveThreshold:
    """Stories whose overall score meets the bar are left alone."""

    def test_overall_above_threshold_skips_reopen(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(_scores(8), "TK-300")

        assert fired is False
        mock_api.assert_not_called()
        assert not _isolate_decisions_log.exists()

    def test_overall_equal_threshold_skips_reopen(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(_scores(7), "TK-301")

        assert fired is False
        mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# All-sentinel scores -> nothing to gate on
# ---------------------------------------------------------------------------

class TestSentinelScores:
    def test_all_sentinel_scores_skip_reopen(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)

        with patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(
                StoryQualityScores.sentinel("parse_failure"), "TK-400"
            )

        assert fired is False
        mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# Jira disabled in env -> no calls, but decisions log still records
# ---------------------------------------------------------------------------

class TestJiraNotConfigured:
    def test_jira_unconfigured_logs_decision_with_false_flags(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)

        with patch("aiv.main.is_jira_configured", return_value=False), \
             patch("aiv.main._jira_api") as mock_api:
            fired = aiv_main.maybe_reopen_for_low_score(_mixed_scores(), "TK-500")

        # The gate fired (decision was made); the Jira mutations just
        # short-circuited because Jira isn't configured.
        assert fired is True
        mock_api.assert_not_called()

        log_lines = _isolate_decisions_log.read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["label_applied"] is False
        assert entry["comment_posted"] is False


# ---------------------------------------------------------------------------
# process_row integration: persist + reopen path
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
            "2026-04-18T05:29:00",
            json.dumps(list(paths or [])),
            "2026-04-18T05:30:00",
        ),
    )
    conn.commit()


class TestProcessRowReopenIntegration:
    """End-to-end: process_row persists, then triggers the threshold gate."""

    def test_low_score_through_process_row_calls_reopen(
        self, monkeypatch, _isolate_decisions_log
    ):
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)
        _seed_pending("TK-600", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_mixed_scores()), \
             patch("aiv.main.is_jira_configured", return_value=True), \
             patch("aiv.main._jira_api") as mock_api:
            mock_api.side_effect = [
                _ok_response(204),
                _ok_response(201),
            ]
            result = aiv_main.process_row(row)

        assert result is not None
        # PUT label + POST comment.
        assert mock_api.call_count == 2

        # Persist still happened.
        conn = aiv_schema._get_conn()
        quality_row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-600",)
        ).fetchone()
        assert quality_row is not None

        # Decision logged.
        log_lines = _isolate_decisions_log.read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1

    def test_reopen_failure_does_not_break_process_row(
        self, monkeypatch, _isolate_decisions_log
    ):
        """A flaky Jira call must not undo a successful persist."""
        monkeypatch.setattr(aiv_main.settings, "aiv_reopen_threshold", 7.0)
        _seed_pending("TK-601", paths=["agent/foo.py"])
        (row,) = aiv_main.fetch_pending_rows()

        with patch("aiv.main.scorer.score", return_value=_mixed_scores()), \
             patch(
                 "aiv.main.maybe_reopen_for_low_score",
                 side_effect=RuntimeError("jira blew up"),
             ):
            result = aiv_main.process_row(row)

        # process_row swallows the reopen failure and still returns the
        # successfully-persisted scores.
        assert result is not None
        conn = aiv_schema._get_conn()
        quality_row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-601",)
        ).fetchone()
        assert quality_row is not None
