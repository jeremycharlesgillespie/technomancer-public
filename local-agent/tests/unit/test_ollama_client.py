"""Tests for agent/ollama_client.py — in-flight tracking and backpressure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent.ollama_client as oc
from agent.ollama_client import MAX_CONCURRENT, chat, get_inflight_count


@pytest.fixture(autouse=True)
def _reset_inflight():
    """Ensure the in-flight counter is zero before and after each test."""
    with oc._inflight_lock:
        oc._inflight_count = 0
    yield
    with oc._inflight_lock:
        oc._inflight_count = 0


def _ok_response(text: str = "hello") -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"response": text}
    return mock


# ---------------------------------------------------------------------------
# get_inflight_count
# ---------------------------------------------------------------------------


def test_inflight_count_starts_at_zero():
    assert get_inflight_count() == 0


def test_inflight_count_increments_during_call():
    """Counter is > 0 while the HTTP call is running."""
    observed: list[int] = []

    def fake_post(*_a, **_kw):
        observed.append(get_inflight_count())
        return _ok_response()

    with patch.object(requests, "post", side_effect=fake_post):
        chat("hi", "qwen3.5:latest")

    assert observed == [1]


def test_inflight_count_returns_to_zero_after_success():
    with patch.object(requests, "post", return_value=_ok_response()):
        chat("hi", "qwen3.5:latest")
    assert get_inflight_count() == 0


def test_inflight_count_returns_to_zero_after_network_error():
    with patch.object(
        requests, "post", side_effect=requests.ConnectionError("refused")
    ):
        result = chat("hi", "qwen3.5:latest")
    assert result is None
    assert get_inflight_count() == 0


def test_inflight_count_returns_to_zero_after_non200():
    mock = MagicMock()
    mock.status_code = 500
    mock.text = "internal error"
    with patch.object(requests, "post", return_value=mock):
        result = chat("hi", "qwen3.5:latest")
    assert result is None
    assert get_inflight_count() == 0


# ---------------------------------------------------------------------------
# Backpressure (hard cap)
# ---------------------------------------------------------------------------


def test_chat_sheds_load_when_cap_reached():
    """Returns None immediately when in-flight count == MAX_CONCURRENT."""
    with oc._inflight_lock:
        oc._inflight_count = MAX_CONCURRENT

    with patch.object(requests, "post") as mock_post:
        result = chat("hi", "qwen3.5:latest")

    assert result is None
    mock_post.assert_not_called()


def test_chat_allowed_one_below_cap():
    """Calls are allowed when inflight == MAX_CONCURRENT - 1."""
    with oc._inflight_lock:
        oc._inflight_count = MAX_CONCURRENT - 1

    with patch.object(requests, "post", return_value=_ok_response("ok")):
        result = chat("hi", "qwen3.5:latest")

    assert result == "ok"


# ---------------------------------------------------------------------------
# 503 → health monitor degraded
# ---------------------------------------------------------------------------


def test_503_notifies_health_monitor():
    """A 503 response marks the Ollama health monitor degraded."""
    mock = MagicMock()
    mock.status_code = 503
    mock.text = "Service Unavailable"

    with (
        patch.object(requests, "post", return_value=mock),
        patch("agent.ollama_client._notify_monitor_degraded") as notify,
    ):
        result = chat("hi", "qwen3.5:latest")

    assert result is None
    notify.assert_called_once()
    assert "503" in notify.call_args[0][0]


def test_503_resets_inflight_counter():
    """In-flight counter returns to 0 even on 503."""
    mock = MagicMock()
    mock.status_code = 503
    mock.text = "Service Unavailable"

    with (
        patch.object(requests, "post", return_value=mock),
        patch("agent.ollama_client._notify_monitor_degraded"),
    ):
        chat("hi", "qwen3.5:latest")

    assert get_inflight_count() == 0


# ---------------------------------------------------------------------------
# Normal happy path
# ---------------------------------------------------------------------------


def test_chat_returns_response_text():
    with patch.object(requests, "post", return_value=_ok_response("pong")):
        result = chat("ping", "qwen3.5:latest")
    assert result == "pong"


def test_chat_strips_whitespace():
    resp = _ok_response("  trimmed  \n")
    with patch.object(requests, "post", return_value=resp):
        result = chat("hi", "qwen3.5:latest")
    assert result == "trimmed"


def test_chat_returns_none_on_empty_response():
    resp = _ok_response("   ")
    with patch.object(requests, "post", return_value=resp):
        result = chat("hi", "qwen3.5:latest")
    assert result is None


def test_chat_returns_none_on_invalid_json():
    mock = MagicMock()
    mock.status_code = 200
    mock.json.side_effect = ValueError("bad json")
    with patch.object(requests, "post", return_value=mock):
        result = chat("hi", "qwen3.5:latest")
    assert result is None
