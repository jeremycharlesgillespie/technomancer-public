"""Tests for the discord_errors module — error categorization, context recovery, and incident tracking."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agent.discord_errors import (
    ERROR_CATALOG,
    GatewayHealth,
    buffer_message,
    categorize_error,
    get_error_report,
    get_discord_error_tools,
    get_incident_summary,
    get_recent_context,
    handle_discord_error,
    init_db,
    log_incident,
    suggest_recovery_content,
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirect SQLite to a temp directory."""
    db_path = tmp_path / "discord_errors.db"
    monkeypatch.setattr("agent.discord_errors.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.discord_errors.DB_PATH", db_path)
    import agent.discord_errors as mod
    # Force new connection
    if hasattr(mod._local, "conn"):
        try:
            mod._local.conn.close()
        except Exception:
            pass
        del mod._local.conn
    # Clear the message buffer
    mod._message_buffer.clear()
    # Pre-init the DB
    init_db()


class TestCategorizeError:
    def test_rate_limit(self):
        cat = categorize_error(status_code=429)
        assert cat.name == "rate_limited"
        assert cat.severity == "low"

    def test_empty_message(self):
        cat = categorize_error(error_code=50006)
        assert cat.name == "empty_message"

    def test_auth_failed(self):
        cat = categorize_error(error_code=4004)
        assert cat.name == "auth_failed"
        assert cat.severity == "critical"

    def test_forbidden(self):
        cat = categorize_error(status_code=403)
        assert cat.name == "forbidden"

    def test_error_code_takes_priority(self):
        cat = categorize_error(status_code=400, error_code=50006)
        assert cat.name == "empty_message"  # error_code wins over status_code

    def test_text_inference_empty(self):
        cat = categorize_error(error_text="Cannot send an empty message")
        assert cat.name == "empty_message"

    def test_text_inference_auth(self):
        cat = categorize_error(error_text="authentication failed code 4004")
        assert cat.name == "auth_failed"

    def test_unknown_error(self):
        cat = categorize_error(status_code=999)
        assert cat.name == "unknown"


class TestMessageBuffer:
    def test_buffer_and_retrieve(self):
        buffer_message("TestUser", "What is spaghetti?", "msg123")
        recent = get_recent_context(5)
        assert len(recent) == 1
        assert recent[0]["user"] == "TestUser"
        assert "spaghetti" in recent[0]["content"]

    def test_buffer_limit(self):
        for i in range(25):
            buffer_message("user", f"message {i}")
        recent = get_recent_context(30)
        assert len(recent) == 20  # maxlen=20

    def test_suggest_recovery_with_context(self):
        buffer_message("TestUser", "Tell me about Python decorators")
        suggestion = suggest_recovery_content()
        assert "TestUser" in suggestion
        assert "decorator" in suggestion.lower()

    def test_suggest_recovery_empty_buffer(self):
        suggestion = suggest_recovery_content()
        assert "ready to help" in suggestion.lower()


class TestGatewayHealth:
    def test_initial_state(self):
        health = GatewayHealth()
        status = health.get_status()
        assert status["connected"] is False
        assert status["error_count"] == 0

    def test_connect_disconnect(self):
        health = GatewayHealth()
        health.record_connect()
        assert health.get_status()["connected"] is True
        health.record_disconnect()
        assert health.get_status()["connected"] is False
        assert health.get_status()["disconnect_count"] == 1

    def test_error_tracking(self):
        health = GatewayHealth()
        health.record_error("test error")
        health.record_error("another error")
        status = health.get_status()
        assert status["error_count"] == 2
        assert status["last_error"] == "another error"


class TestIncidentDb:
    def test_log_and_query(self):
        init_db()
        incident_id = log_incident("rate_limited", "low", status_code=429, error_text="Rate limited")
        assert incident_id > 0

        summary = get_incident_summary(hours=1)
        assert summary["total"] == 1
        assert summary["by_category"][0]["category"] == "rate_limited"

    def test_multiple_incidents(self):
        init_db()
        log_incident("rate_limited", "low", status_code=429)
        log_incident("rate_limited", "low", status_code=429)
        log_incident("empty_message", "low", error_code=50006)

        summary = get_incident_summary(hours=1)
        assert summary["total"] == 3


class TestHandleDiscordError:
    def test_handles_exception_with_status(self):
        # Simulate a discord.HTTPException-like object
        exc = Exception("Rate limited")
        exc.status = 429
        exc.code = None
        exc.text = "Rate limited"
        cat = handle_discord_error(exc, context="test")
        assert cat.name == "rate_limited"

    def test_handles_plain_exception(self):
        cat = handle_discord_error(Exception("something broke"), context="test")
        assert cat.name == "unknown"

    def test_logs_incident(self):
        handle_discord_error(Exception("test error"), context="testing")
        summary = get_incident_summary(hours=1)
        assert summary["total"] >= 1


class TestGetErrorReport:
    def test_empty_report(self):
        report = get_error_report(hours=1)
        assert "No Discord errors" in report

    def test_report_with_data(self):
        init_db()
        log_incident("rate_limited", "low", status_code=429, error_text="Too fast")
        log_incident("auth_failed", "critical", error_code=4004, error_text="Bad token")
        report = get_error_report(hours=1)
        assert "Discord Error Report" in report
        assert "rate_limited" in report
        assert "Gateway Health" in report


class TestErrorCatalog:
    def test_all_entries_have_fields(self):
        for code, cat in ERROR_CATALOG.items():
            assert cat.name
            assert cat.description
            assert cat.recovery
            assert cat.severity in ("low", "medium", "high", "critical")

    def test_known_codes_covered(self):
        assert 429 in ERROR_CATALOG
        assert 50006 in ERROR_CATALOG
        assert 4004 in ERROR_CATALOG
        assert 403 in ERROR_CATALOG


class TestGetTools:
    def test_returns_tool(self):
        tools = get_discord_error_tools()
        assert len(tools) == 1
        assert tools[0].name == "discord_error_report"

    def test_tool_runs(self):
        tools = get_discord_error_tools()
        result = tools[0].function()
        assert isinstance(result, str)
