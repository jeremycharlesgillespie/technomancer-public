"""Tests for the AIV Discord alert on low score or red flag (TK-695).

The alert fires independently of the Jira re-open gates: any scored
story whose overall falls below ``aiv_alert_threshold`` OR that carries
at least one red flag triggers a webhook post. Missing/unset webhook URL
is handled by :func:`agent.notifications.discord_send` returning an
error string rather than raising — the alert path must never propagate
an exception.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiv import main as aiv_main
from aiv.scorer import StoryQualityScores


@pytest.fixture(autouse=True)
def _reset_alert_threshold(monkeypatch):
    """Every test starts with the default 5.0 threshold."""
    monkeypatch.setattr(aiv_main.settings, "aiv_alert_threshold", 5.0)


def _scores(axes: int, red_flags=None) -> StoryQualityScores:
    """Return a StoryQualityScores with every axis at ``axes``."""
    return StoryQualityScores(
        meets_requirements=axes,
        code_quality=axes,
        test_quality=axes,
        security_safety=axes,
        scope_discipline=axes,
        edge_cases=axes,
        product_impact=axes,
        reasoning_map={"meets_requirements": "ok"},
        red_flags=list(red_flags or []),
        error="",
    )


# ---------------------------------------------------------------------------
# Acceptance: overall=3 fires the webhook with the story key in the body.
# ---------------------------------------------------------------------------

class TestLowScoreAlert:
    def test_overall_below_threshold_fires_webhook(self):
        with patch("aiv.main.discord_send") as mock_send:
            mock_send.return_value = "Sent to Discord: ..."
            fired = aiv_main.maybe_send_discord_alert(_scores(3), "TK-100")

        assert fired is True
        mock_send.assert_called_once()

        # Message body is the first positional arg.
        call_args, call_kwargs = mock_send.call_args
        body = call_args[0]
        assert "TK-100" in body
        # Title kwarg also echoes the story key.
        assert "TK-100" in call_kwargs.get("title", "")

    def test_overall_at_threshold_does_not_fire(self):
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(_scores(5), "TK-101")
        assert fired is False
        mock_send.assert_not_called()

    def test_overall_above_threshold_does_not_fire(self):
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(_scores(8), "TK-102")
        assert fired is False
        mock_send.assert_not_called()

    def test_story_title_appears_in_body_when_provided(self):
        with patch("aiv.main.discord_send") as mock_send:
            aiv_main.maybe_send_discord_alert(
                _scores(3), "TK-103", story_title="Fix the thing"
            )
        body = mock_send.call_args[0][0]
        assert "Fix the thing" in body


# ---------------------------------------------------------------------------
# Red flag alone is enough, regardless of score.
# ---------------------------------------------------------------------------

class TestRedFlagAlert:
    def test_red_flag_with_high_score_still_fires(self):
        scores = _scores(9, red_flags=["no_tests_added"])
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(scores, "TK-200")

        assert fired is True
        mock_send.assert_called_once()
        body = mock_send.call_args[0][0]
        assert "no_tests_added" in body
        assert "TK-200" in body

    def test_multiple_red_flags_all_listed(self):
        scores = _scores(9, red_flags=["no_tests_added", "secret_leak"])
        with patch("aiv.main.discord_send") as mock_send:
            aiv_main.maybe_send_discord_alert(scores, "TK-201")
        body = mock_send.call_args[0][0]
        assert "no_tests_added" in body
        assert "secret_leak" in body

    def test_no_red_flags_and_high_score_is_silent(self):
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(_scores(9), "TK-202")
        assert fired is False
        mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Webhook disabled / URL missing → no-op, no exception.
# ---------------------------------------------------------------------------

class TestWebhookDisabledOrFailing:
    def test_missing_webhook_url_is_noop_no_exception(self):
        """discord_send returns its ``no URL`` error string rather than
        raising — the alert wrapper must just call it and move on."""
        with patch(
            "aiv.main.discord_send",
            return_value="Error: No webhook URL configured",
        ) as mock_send:
            # Must not raise.
            fired = aiv_main.maybe_send_discord_alert(_scores(3), "TK-300")

        # We still count this as "fired" because we attempted the send.
        # The key guarantee is: no exception propagated.
        assert fired is True
        mock_send.assert_called_once()

    def test_discord_send_raising_does_not_propagate(self):
        with patch(
            "aiv.main.discord_send", side_effect=RuntimeError("network boom")
        ):
            fired = aiv_main.maybe_send_discord_alert(_scores(3), "TK-301")
        # No exception reached the caller; function reports it didn't fire.
        assert fired is False


# ---------------------------------------------------------------------------
# Sentinel scores (all axes = -1) have nothing to alert on.
# ---------------------------------------------------------------------------

class TestSentinelScores:
    def test_all_sentinel_scores_no_alert(self):
        scores = StoryQualityScores.sentinel("parse_failure")
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(scores, "TK-400")
        assert fired is False
        mock_send.assert_not_called()

    def test_sentinel_with_red_flag_still_fires(self):
        scores = StoryQualityScores.sentinel("parse_failure")
        # Inject a red flag even though every axis is sentinel.
        scores.red_flags = ["scorer_crashed"]
        with patch("aiv.main.discord_send") as mock_send:
            fired = aiv_main.maybe_send_discord_alert(scores, "TK-401")
        assert fired is True
        mock_send.assert_called_once()
        body = mock_send.call_args[0][0]
        assert "scorer_crashed" in body
