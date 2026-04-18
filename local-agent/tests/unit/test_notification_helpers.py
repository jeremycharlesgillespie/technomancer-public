"""Tests for idea_board.notification_helpers — /errors notification query."""

from __future__ import annotations

from unittest.mock import patch

from idea_board import notification_helpers
from idea_board.notification_helpers import (
    _extract_message_id,
    query_healthy_notifications,
)


class TestExtractMessageId:
    def test_full_discord_url(self):
        url = "https://discord.com/channels/1/2/3"
        assert _extract_message_id(url) == "3"

    def test_dm_style_url(self):
        url = "https://discord.com/channels/@me/42/999"
        assert _extract_message_id(url) == "999"

    def test_trailing_slash_is_stripped(self):
        url = "https://discord.com/channels/1/2/3/"
        assert _extract_message_id(url) == "3"

    def test_none_returns_none(self):
        assert _extract_message_id(None) is None

    def test_empty_string_returns_none(self):
        assert _extract_message_id("") is None


class TestQueryHealthyNotifications:
    def test_shapes_rows_with_message_id_and_discord_url(self):
        fake_rows = [
            {
                "timestamp": "2026-04-18T10:00:00",
                "message_url": "https://discord.com/channels/1/2/3",
            },
            {
                "timestamp": "2026-04-18T09:00:00",
                "message_url": "https://discord.com/channels/@me/42/999",
            },
        ]
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=fake_rows
        ):
            result = query_healthy_notifications(limit=5)

        assert result == [
            {
                "timestamp": "2026-04-18T10:00:00",
                "message_id": "3",
                "discord_url": "https://discord.com/channels/1/2/3",
            },
            {
                "timestamp": "2026-04-18T09:00:00",
                "message_id": "999",
                "discord_url": "https://discord.com/channels/@me/42/999",
            },
        ]

    def test_null_url_yields_null_message_id(self):
        fake_rows = [
            {"timestamp": "2026-04-18T10:00:00", "message_url": None},
        ]
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=fake_rows
        ):
            result = query_healthy_notifications()

        assert result == [
            {
                "timestamp": "2026-04-18T10:00:00",
                "message_id": None,
                "discord_url": None,
            }
        ]

    def test_empty_db_returns_empty_list(self):
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=[]
        ):
            assert query_healthy_notifications() == []

    def test_passes_limit_and_max_age_through(self):
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=[]
        ) as mock_query:
            query_healthy_notifications(limit=3, max_age_hours=48)

        mock_query.assert_called_once_with(limit=3, max_age_hours=48)

    def test_default_limit_is_5(self):
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=[]
        ) as mock_query:
            query_healthy_notifications()

        mock_query.assert_called_once_with(limit=5, max_age_hours=24)

    def test_max_age_none_is_propagated(self):
        with patch.object(
            notification_helpers, "_query_healthy_notifications", return_value=[]
        ) as mock_query:
            query_healthy_notifications(max_age_hours=None)

        mock_query.assert_called_once_with(limit=5, max_age_hours=None)
