"""Tests for GPU gate timeout and model unloading functionality in ollama_client.py."""

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


@pytest.fixture(autouse=True)
def _reset_coder_active():
    """Ensure the coder active event is cleared before and after each test."""
    oc._coder_active.clear()
    yield
    oc._coder_active.clear()


def _ok_response(text: str = "hello") -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"response": text}
    return mock


def _ps_response(models: list[dict]) -> MagicMock:
    """Create a mock response for /api/ps endpoint."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"models": models}
    return mock


def _ps_response_empty() -> MagicMock:
    """Create a mock response for /api/ps endpoint with no models."""
    return _ps_response([])


def test_unload_competing_models_no_models():
    """Test _unload_competing_models with no models."""
    # Mock the ps endpoint to return empty list
    ps_mock = _ps_response_empty()
    
    with (
        patch.object(requests, "post", return_value=ps_mock) as mock_post,
        patch("agent.ollama_client.logger") as mock_logger
    ):
        # Call the unload function directly
        oc._unload_competing_models("coder_model:latest")
        
        # Should have called ps endpoint
        assert mock_post.call_count == 1
        
        # Should not have tried to unload any models (no calls to generate endpoint)


def test_unload_competing_models_single_model():
    """Test _unload_competing_models with single model (the coder)."""
    # Mock the ps endpoint to return only the coder model
    ps_mock = _ps_response([
        {"name": "coder_model:latest"}
    ])
    
    with (
        patch.object(requests, "post", return_value=ps_mock) as mock_post,
        patch("agent.ollama_client.logger") as mock_logger
    ):
        # Call the unload function directly
        oc._unload_competing_models("coder_model:latest")
        
        # Should have called ps endpoint
        assert mock_post.call_count == 1
        
        # Should not have tried to unload any models (no calls to generate endpoint)


def test_unload_competing_models_multiple_models():
    """Test _unload_competing_models with multiple models."""
    # Mock the ps endpoint to return multiple models
    ps_mock = _ps_response([
        {"name": "coder_model:latest"},
        {"name": "other_model:latest"},
        {"name": "another_model:latest"}
    ])
    
    # Mock the generate endpoint to return a successful response for unloading
    generate_mock = _ok_response("unloaded")
    
    with (
        patch.object(requests, "post", side_effect=[ps_mock, generate_mock, generate_mock]) as mock_post,
        patch("agent.ollama_client.logger") as mock_logger
    ):
        # Call the unload function directly
        oc._unload_competing_models("coder_model:latest")
        
        # Should have called ps endpoint and two unload calls
        assert mock_post.call_count == 3


def test_unload_competing_models_network_error_handling():
    """Test _unload_competing_models handles network errors gracefully."""
    # Mock the ps endpoint to return models
    ps_mock = _ps_response([
        {"name": "coder_model:latest"},
        {"name": "other_model:latest"}
    ])
    
    # Mock the generate endpoint to return a network error
    generate_mock = MagicMock()
    generate_mock.side_effect = requests.ConnectionError("timeout")
    
    with (
        patch.object(requests, "post", side_effect=[ps_mock, generate_mock]) as mock_post,
        patch("agent.ollama_client.logger") as mock_logger
    ):
        # Call the unload function directly
        oc._unload_competing_models("coder_model:latest")
        
        # Should have called ps endpoint and one unload call
        assert mock_post.call_count == 2


def test_normal_operation_not_affected():
    """Test that normal operation is not affected by the changes."""
    # Don't set the coder active flag - normal operation

    with patch.object(requests, "post", return_value=_ok_response("pong")):
        result = chat("ping", "qwen3.5:9b")
    assert result == "pong"
    assert get_inflight_count() == 0


def test_coder_active_waits_normal():
    """Test that when coder is active, normal wait behavior works."""
    # Set the coder active flag
    oc._coder_active.set()
    
    # Mock the generate endpoint to return a successful response
    generate_mock = _ok_response("test response")
    
    with (
        patch.object(requests, "post", return_value=generate_mock) as mock_post,
        patch("agent.ollama_client.logger") as mock_logger
    ):
        # Call chat with a normal timeout - should not timeout
        result = chat("test prompt", "coder_model:latest", timeout=1.0)
        
        # Should return the response
        assert result == "test response"
        
        # Should have called the generate endpoint
        assert mock_post.call_count >= 1
        
        # Should have logged the waiting action
        mock_logger.debug.assert_any_call("[ollama_client] coder active — waiting up to 30s for GPU slot")