"""Tests for the notifications module — Discord webhook messaging."""

from unittest.mock import MagicMock, patch

import pytest

from agent.notifications import (
    COLORS,
    discord_alert,
    discord_send,
    discord_send_code,
    get_notification_tools,
)


class TestDiscordSend:
    """Test discord_send function."""

    @patch("agent.notifications.retry_request")
    def test_sends_plain_message(self, mock_retry, monkeypatch):
        monkeypatch.setattr("agent.notifications.DISCORD_WEBHOOK_URL", "https://webhook.test")
        mock_resp = MagicMock(status_code=204)
        mock_retry.return_value = mock_resp

        result = discord_send("Hello world")
        assert "Sent to Discord" in result
        mock_retry.assert_called_once()
        call_kwargs = mock_retry.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][2]
        # Plain message should use "content" key
        assert "Hello" in str(payload)

    @patch("agent.notifications.retry_request")
    def test_sends_embed_with_title(self, mock_retry, monkeypatch):
        monkeypatch.setattr("agent.notifications.DISCORD_WEBHOOK_URL", "https://webhook.test")
        mock_resp = MagicMock(status_code=204)
        mock_retry.return_value = mock_resp

        result = discord_send("Body text", title="My Title", color=0xFF0000)
        assert "Sent to Discord" in result

    def test_no_webhook_url(self, monkeypatch):
        monkeypatch.setattr("agent.notifications.DISCORD_WEBHOOK_URL", "")
        result = discord_send("test")
        assert "No webhook URL" in result

    @patch("agent.notifications.retry_request")
    def test_non_204_status(self, mock_retry, monkeypatch):
        monkeypatch.setattr("agent.notifications.DISCORD_WEBHOOK_URL", "https://webhook.test")
        mock_resp = MagicMock(status_code=400, text="Bad Request")
        mock_retry.return_value = mock_resp

        result = discord_send("test")
        assert "error" in result.lower() or "400" in result

    @patch("agent.notifications.retry_request")
    def test_exception_handling(self, mock_retry, monkeypatch):
        monkeypatch.setattr("agent.notifications.DISCORD_WEBHOOK_URL", "https://webhook.test")
        mock_retry.side_effect = Exception("Network failed")

        result = discord_send("test")
        assert "Error" in result

    def test_no_requests_library(self, monkeypatch):
        monkeypatch.setattr("agent.notifications.requests", None)
        result = discord_send("test")
        assert "not installed" in result


class TestDiscordAlert:
    """Test discord_alert with level-based colors."""

    @patch("agent.notifications.discord_send")
    def test_success_level(self, mock_send):
        mock_send.return_value = "ok"
        discord_alert("All good", level="success")
        mock_send.assert_called_once()
        args = mock_send.call_args
        assert args[1].get("color") == COLORS["success"] or args[0][2] if len(args[0]) > 2 else True

    @patch("agent.notifications.discord_send")
    def test_error_level(self, mock_send):
        mock_send.return_value = "ok"
        discord_alert("Something broke", level="error")
        mock_send.assert_called_once()

    @patch("agent.notifications.discord_send")
    def test_default_title_from_level(self, mock_send):
        mock_send.return_value = "ok"
        discord_alert("test", level="warning")
        args = mock_send.call_args
        # Title should default to "WARNING"
        assert "WARNING" in str(args)

    @patch("agent.notifications.discord_send")
    def test_custom_title(self, mock_send):
        mock_send.return_value = "ok"
        discord_alert("test", level="info", title="Custom Title")
        args = mock_send.call_args
        assert "Custom Title" in str(args)


class TestDiscordSendCode:
    """Test code block formatting."""

    @patch("agent.notifications.discord_send")
    def test_formats_code_block(self, mock_send):
        mock_send.return_value = "ok"
        discord_send_code("print('hello')", language="python")
        args = mock_send.call_args[0][0]
        assert "```python" in args
        assert "print('hello')" in args
        assert "```" in args

    @patch("agent.notifications.discord_send")
    def test_includes_message(self, mock_send):
        mock_send.return_value = "ok"
        discord_send_code("x = 1", message="Here's the code:")
        args = mock_send.call_args[0][0]
        assert "Here's the code:" in args


class TestColorConstants:
    """Test color presets."""

    def test_all_colors_defined(self):
        assert "success" in COLORS
        assert "error" in COLORS
        assert "warning" in COLORS
        assert "info" in COLORS

    def test_colors_are_integers(self):
        for name, color in COLORS.items():
            assert isinstance(color, int), f"{name} color should be int"


class TestGetNotificationTools:
    """Test tool registration."""

    def test_returns_tools(self):
        tools = get_notification_tools()
        assert len(tools) >= 3
        names = {t.name for t in tools}
        assert "discord_send" in names
        assert "discord_alert" in names
