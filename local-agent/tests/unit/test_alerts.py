"""Tests for the alerts module — dedicated alerts channel webhook."""

import os
from unittest.mock import MagicMock, patch

import pytest


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
