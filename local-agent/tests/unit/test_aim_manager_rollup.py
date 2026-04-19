"""Tests for TK-619: daily rollup scheduled from the AIM manager."""

from __future__ import annotations

from datetime import datetime as real_datetime
from unittest.mock import patch

import pytest

from aim.manager import _maybe_run_daily_rollup
from aim.state import AIMState


@pytest.fixture
def state() -> AIMState:
    return AIMState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(state: AIMState, today: real_datetime, project: str = "TK") -> None:
    """Call _maybe_run_daily_rollup with a frozen clock and a given project key."""
    with patch("aim.manager.datetime") as mock_dt, \
         patch("agent.config.settings") as mock_settings:
        mock_dt.now.return_value = today
        mock_settings.jira_project_key = project
        _maybe_run_daily_rollup(state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaybeRunDailyRollup:

    def test_runs_on_first_cycle_of_new_day(self, state):
        """Clock advances past midnight → compute_and_write called once for yesterday."""
        state.last_rollup_date = "2026-04-18"

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 4, 19, 0, 5, 0))

        mock_rollup.assert_called_once_with("2026-04-18", "TK")
        assert state.last_rollup_date == "2026-04-19"

    def test_second_cycle_same_day_is_noop(self, state):
        """Second call on the same calendar day does NOT re-call compute_and_write."""
        state.last_rollup_date = "2026-04-19"

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 4, 19, 12, 0, 0))

        mock_rollup.assert_not_called()

    def test_rollup_failure_logs_warning_not_raise(self, state):
        """compute_and_write raising → warning logged, no exception, date updated."""
        state.last_rollup_date = "2026-04-18"

        with patch("aim.manager.compute_and_write", side_effect=RuntimeError("DB down")):
            # Must not raise
            _run(state, real_datetime(2026, 4, 19, 0, 5, 0))

        # Date is still advanced so we don't retry on every subsequent cycle
        assert state.last_rollup_date == "2026-04-19"

    def test_uses_jira_project_key_from_settings(self, state):
        """Project key is taken from settings.jira_project_key."""
        state.last_rollup_date = "2026-04-18"

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 4, 19, 8, 0, 0), project="FA")

        mock_rollup.assert_called_once_with("2026-04-18", "FA")

    def test_none_project_key_falls_back_to_tk(self, state):
        """None jira_project_key falls back to 'TK'."""
        state.last_rollup_date = "2026-04-18"

        with patch("aim.manager.compute_and_write") as mock_rollup, \
             patch("aim.manager.datetime") as mock_dt, \
             patch("agent.config.settings") as mock_settings:
            mock_dt.now.return_value = real_datetime(2026, 4, 19, 6, 0, 0)
            mock_settings.jira_project_key = None
            _maybe_run_daily_rollup(state)

        mock_rollup.assert_called_once_with("2026-04-18", "TK")

    def test_first_ever_run_empty_last_date(self, state):
        """Empty last_rollup_date triggers rollup on the very first call."""
        assert state.last_rollup_date == ""

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 4, 19, 6, 0, 0))

        mock_rollup.assert_called_once_with("2026-04-18", "TK")
        assert state.last_rollup_date == "2026-04-19"

    def test_yesterday_date_is_correct(self, state):
        """Yesterday is computed as (today - 1 day), not today."""
        state.last_rollup_date = "2026-01-01"

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 1, 2, 0, 1, 0))

        args = mock_rollup.call_args[0]
        assert args[0] == "2026-01-01"

    def test_state_last_rollup_date_persists_across_calls(self, state):
        """last_rollup_date is updated so the check is idempotent within a day."""
        state.last_rollup_date = "2026-04-18"

        with patch("aim.manager.compute_and_write") as mock_rollup:
            _run(state, real_datetime(2026, 4, 19, 0, 1, 0))
            _run(state, real_datetime(2026, 4, 19, 0, 2, 0))
            _run(state, real_datetime(2026, 4, 19, 23, 59, 0))

        mock_rollup.assert_called_once()
