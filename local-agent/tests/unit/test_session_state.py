"""Tests for agent/session_state.py — conversation persistence."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from agent.session_state import (
    format_session_context,
    get_previous_session,
    save_exchange,
    mark_clean_shutdown,
)


class TestSaveExchange:
    def test_saves_exchange(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        save_exchange("user1", "hello", "hi there")

        data = json.loads(state_file.read_text())
        assert len(data["exchanges"]) == 1
        assert data["exchanges"][0]["user"] == "user1"
        assert data["exchanges"][0]["message"] == "hello"

    def test_limits_to_10_exchanges(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        for i in range(15):
            save_exchange("user", f"msg {i}", f"resp {i}")

        data = json.loads(state_file.read_text())
        assert len(data["exchanges"]) == 10

    def test_truncates_long_messages(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        long_msg = "x" * 1000
        save_exchange("user", long_msg, "short")

        data = json.loads(state_file.read_text())
        assert len(data["exchanges"][0]["message"]) <= 500


class TestMarkCleanShutdown:
    def test_marks_shutdown(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        save_exchange("user", "hi", "hey")
        mark_clean_shutdown()

        data = json.loads(state_file.read_text())
        assert data["clean_shutdown"] is True


class TestGetPreviousSession:
    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        state_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        session = get_previous_session()
        assert session["exchanges"] == []
        assert session["was_crash"] is False

    def test_detects_crash(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        state_file.write_text(json.dumps({
            "exchanges": [{"user": "u", "user_message": "m", "bot_response": "r", "timestamp": datetime.now().isoformat()}],
            "clean_shutdown": False,
            "last_active": datetime.now().isoformat(),
        }))

        session = get_previous_session()
        assert session["was_crash"] is True

    def test_clean_shutdown_not_crash(self, tmp_path, monkeypatch):
        state_file = tmp_path / "session_state.json"
        monkeypatch.setattr("agent.session_state.STATE_FILE", state_file)

        state_file.write_text(json.dumps({
            "exchanges": [],
            "clean_shutdown": True,
            "last_active": datetime.now().isoformat(),
        }))

        session = get_previous_session()
        assert session["was_crash"] is False


class TestFormatSessionContext:
    def test_empty_session(self):
        result = format_session_context({"exchanges": [], "was_crash": False})
        assert result == ""

    def test_formats_exchanges(self):
        session = {
            "exchanges": [
                {"user": "bob", "message": "hello", "response": "hi"},
            ],
            "was_crash": False,
            "last_active": datetime.now().isoformat(),
            "time_since_last": 60,
        }
        result = format_session_context(session)
        assert "hello" in result
        assert "hi" in result

    def test_stale_context_returns_empty(self):
        session = {
            "exchanges": [
                {"user": "bob", "message": "hello", "response": "hi"},
            ],
            "was_crash": False,
            "last_active": (datetime.now() - timedelta(hours=12)).isoformat(),
            "time_since_last": 43200,
        }
        result = format_session_context(session)
        assert result == ""
