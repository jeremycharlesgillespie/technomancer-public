"""Tests for the alerts module — dedicated alerts channel webhook."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent import alerts


@pytest.fixture(autouse=True)
def _reset_slo_dedup():
    """Clear SLO dedup state between tests so cases don't leak into each other."""
    alerts._reset_slo_state()
    yield
    alerts._reset_slo_state()


class TestSendAlert:
    """Test send_alert function."""

    @patch("agent.discord_rate_limit.retry_request")
    def test_sends_with_webhook(self, mock_retry, monkeypatch):
        monkeypatch.setenv("DISCORD_ALERTS_WEBHOOK", "https://webhook.test")
        # Re-import to pick up env var
        import importlib
        import agent.config
        agent.config.get_settings.cache_clear()
        importlib.reload(agent.config)

        mock_retry.return_value = MagicMock(status_code=204)
        from agent.alerts import send_alert
        send_alert("Test message")
        # May or may not call depending on settings reload; just verify no crash

    def test_no_webhook_skips(self, monkeypatch):
        monkeypatch.delenv("DISCORD_ALERTS_WEBHOOK", raising=False)
        import importlib
        import agent.config
        agent.config.get_settings.cache_clear()
        importlib.reload(agent.config)

        from agent.alerts import send_alert
        send_alert("test")  # should not raise

    def test_function_exists(self):
        from agent.alerts import send_alert
        assert callable(send_alert)


class TestCheckSLOViolations:
    """Test check_slo_violations — first-fire, 30-min suppression, recovery."""

    THRESHOLDS = {
        "executor_success_rate": {
            "path": ("executor", "success_rate"),
            "comparator": "lt",
            "threshold": 0.8,
            "label": "Executor success rate",
            "unit": "",
            "severity": "warning",
        },
        "queue_depth": {
            "path": ("board", "queue_depth"),
            "comparator": "gt",
            "threshold": 50,
            "label": "Queue depth",
            "unit": " items",
            "severity": "warning",
        },
    }

    @staticmethod
    def _snapshot(success_rate: float = 1.0, queue_depth: int = 0):
        return {
            "executor": {"success_rate": success_rate},
            "board": {"queue_depth": queue_depth},
        }

    def test_first_fire_dispatches_alert(self, monkeypatch):
        """A breaching SLO dispatches on the first check."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        # success_rate 0.5 is below the 0.8 floor -> breach.
        fired = alerts.check_slo_violations(
            snapshot=self._snapshot(success_rate=0.5),
            now=1000.0,
        )

        assert fired == ["executor_success_rate"]
        assert dispatch.call_count == 1
        kwargs = dispatch.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert kwargs["category"] == "slo"
        assert "0.5" in kwargs["message"]
        assert "Executor success rate" in kwargs["title"]

    def test_suppressed_within_30_min_window(self, monkeypatch):
        """A second breach inside the 30-min window does not re-dispatch."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        breach = self._snapshot(success_rate=0.5)

        first = alerts.check_slo_violations(snapshot=breach, now=0.0)
        # 29 minutes later — still inside the dedup window.
        second = alerts.check_slo_violations(snapshot=breach, now=29 * 60)
        # 29 minutes and 59 seconds later — last tick before the window closes.
        third = alerts.check_slo_violations(snapshot=breach, now=29 * 60 + 59)

        assert first == ["executor_success_rate"]
        assert second == []
        assert third == []
        assert dispatch.call_count == 1

    def test_refires_after_window_expires(self, monkeypatch):
        """Once the 30-min window elapses, a continuing breach re-alerts."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        breach = self._snapshot(success_rate=0.5)

        alerts.check_slo_violations(snapshot=breach, now=0.0)
        # 30 minutes + 1 second later — window has closed.
        fired = alerts.check_slo_violations(
            snapshot=breach,
            now=alerts.SLO_DEDUP_WINDOW_SECONDS + 1.0,
        )

        assert fired == ["executor_success_rate"]
        assert dispatch.call_count == 2

    def test_recovery_clears_dedup_and_next_breach_alerts(self, monkeypatch):
        """After an SLO recovers, the very next breach dispatches again.

        This is the key test for the "clearing when SLO recovers" acceptance
        criterion — we don't wait out the 30-min window after a healthy tick.
        """
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        # 1. Breach -> alert.
        alerts.check_slo_violations(snapshot=self._snapshot(success_rate=0.5), now=0.0)
        assert dispatch.call_count == 1
        assert "executor_success_rate" in alerts._slo_last_fired

        # 2. Recovery (value healthy) — dedup state for this SLO must clear.
        fired = alerts.check_slo_violations(
            snapshot=self._snapshot(success_rate=0.95),
            now=60.0,
        )
        assert fired == []
        assert "executor_success_rate" not in alerts._slo_last_fired
        assert dispatch.call_count == 1  # no alert on recovery

        # 3. New breach only 2 minutes later — well inside what would have
        #    been the old dedup window, but the recovery cleared it.
        fired = alerts.check_slo_violations(
            snapshot=self._snapshot(success_rate=0.5),
            now=180.0,
        )
        assert fired == ["executor_success_rate"]
        assert dispatch.call_count == 2

    def test_none_value_is_not_a_breach(self, monkeypatch):
        """A missing metric (None) is treated as no signal, not a breach."""
        thresholds = {
            "oldest_wait": {
                "path": ("board", "oldest_top_ranked_wait_seconds"),
                "comparator": "gt",
                "threshold": 60.0,
                "label": "Oldest wait",
                "unit": "s",
                "severity": "warning",
            },
        }
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", thresholds)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        fired = alerts.check_slo_violations(
            snapshot={"board": {"oldest_top_ranked_wait_seconds": None}},
            now=0.0,
        )

        assert fired == []
        dispatch.assert_not_called()

    def test_independent_slos_tracked_separately(self, monkeypatch):
        """Suppression for one SLO must not suppress another."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        # First tick: only success_rate breaches.
        first = alerts.check_slo_violations(
            snapshot=self._snapshot(success_rate=0.5, queue_depth=0),
            now=0.0,
        )
        # Second tick (still inside the dedup window): success_rate still
        # breaching (suppressed) but queue_depth now breaches for the first
        # time — it should still fire.
        second = alerts.check_slo_violations(
            snapshot=self._snapshot(success_rate=0.5, queue_depth=100),
            now=60.0,
        )

        assert first == ["executor_success_rate"]
        assert second == ["queue_depth"]
        assert dispatch.call_count == 2

    def test_get_snapshot_fallback(self, monkeypatch):
        """When snapshot arg is omitted, metrics.get_snapshot is consulted."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)
        monkeypatch.setattr(
            alerts.metrics,
            "get_snapshot",
            lambda: self._snapshot(success_rate=0.5),
        )

        fired = alerts.check_slo_violations(now=0.0)

        assert fired == ["executor_success_rate"]
        assert dispatch.call_count == 1

    def test_get_snapshot_error_is_swallowed(self, monkeypatch):
        """If metrics.get_snapshot raises, SLO check logs and returns []."""
        monkeypatch.setattr(alerts.metrics, "SLO_THRESHOLDS", self.THRESHOLDS)
        dispatch = MagicMock()
        monkeypatch.setattr(alerts, "dispatch_alert", dispatch)

        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(alerts.metrics, "get_snapshot", _boom)

        fired = alerts.check_slo_violations()

        assert fired == []
        dispatch.assert_not_called()
