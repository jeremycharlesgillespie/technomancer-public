"""Tests for OllamaCoder model eviction functionality.

Verifies that OllamaCoder evicts non-coder models before starting a story.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import OllamaCoder


@pytest.fixture
def mock_state():
    """Create a mock state object for OllamaCoder."""
    state = MagicMock()
    state.log = MagicMock()
    state.cancelled = False
    return state


def test_run_evicts_non_coder_models(mock_state):
    """Test that run() evicts non-coder models after acquiring priority."""
    # Mock the ps endpoint to return two loaded models
    ps_mock = MagicMock()
    ps_mock.status_code = 200
    ps_mock.json.return_value = {
        "models": [
            {"name": "qwen3.5:27b"},  # The coder model
            {"name": "llava:latest"},  # Non-coder model to evict
            {"name": "qwen3.5:9b"},    # Another non-coder model to evict
        ]
    }
    
    # Mock the generate endpoint to return successful unload responses
    generate_mock = MagicMock()
    generate_mock.status_code = 200
    generate_mock.json.return_value = {"response": "unloaded"}
    
    with (
        patch("agent.ollama_client.OLLAMA_HOST", "http://127.0.0.1:11434"),
        patch("agent.ollama_client.requests.post", side_effect=[ps_mock, generate_mock, generate_mock]) as mock_post,
        patch("idea_board.ollama_coder.acquire_coder_priority"),
        patch("idea_board.ollama_coder.release_coder_priority"),
        patch.object(OllamaCoder, "_start_mcp_bridge"),
        patch.object(OllamaCoder, "_stop_mcp_bridge"),
        patch.object(OllamaCoder, "_run_rounds"),
    ):
        # Create and run the coder
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp/test",
            idea_id="TK-922",
            state=mock_state,
            model="qwen3.5:27b",
        )
        coder.run()
        
        # Verify that _unload_competing_models was called with the correct model
        # It should be called once during run() after acquire_coder_priority()
        assert mock_post.call_count == 3
        
        # First call should be to /api/ps
        assert mock_post.call_args_list[0][0][0] == "http://127.0.0.1:11434/api/ps"
        
        # Second and third calls should be to /api/generate for unloading
        assert mock_post.call_args_list[1][0][0] == "http://127.0.0.1:11434/api/generate"
        assert mock_post.call_args_list[2][0][0] == "http://127.0.0.1:11434/api/generate"
        
        # Verify the generate calls have the correct payload (keep_alive=0)
        for call in mock_post.call_args_list[1:]:
            payload = call[1]["json"]
            assert payload["keep_alive"] == 0
            assert payload["prompt"] == ""
            # Should be unloading the non-coder models, not the coder model
            assert payload["model"] in ["llava:latest", "qwen3.5:9b"]


def test_run_no_eviction_needed(mock_state):
    """Test that run() doesn't try to unload when only coder model is loaded."""
    # Mock the ps endpoint to return only the coder model
    ps_mock = MagicMock()
    ps_mock.status_code = 200
    ps_mock.json.return_value = {
        "models": [
            {"name": "qwen3.5:27b"},  # Only the coder model
        ]
    }
    
    with (
        patch("agent.ollama_client.OLLAMA_HOST", "http://127.0.0.1:11434"),
        patch("agent.ollama_client.requests.post", return_value=ps_mock) as mock_post,
        patch("idea_board.ollama_coder.acquire_coder_priority"),
        patch("idea_board.ollama_coder.release_coder_priority"),
        patch.object(OllamaCoder, "_start_mcp_bridge"),
        patch.object(OllamaCoder, "_stop_mcp_bridge"),
        patch.object(OllamaCoder, "_run_rounds"),
    ):
        # Create and run the coder
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp/test",
            idea_id="TK-922",
            state=mock_state,
            model="qwen3.5:27b",
        )
        coder.run()
        
        # Should only call /api/ps, not /api/generate
        assert mock_post.call_count == 1
        assert mock_post.call_args[0][0] == "http://127.0.0.1:11434/api/ps"


def test_run_handles_ps_failure(mock_state):
    """Test that run() handles /api/ps failure gracefully."""
    # Mock the ps endpoint to return an error
    ps_mock = MagicMock()
    ps_mock.status_code = 500
    ps_mock.text = "Internal Server Error"
    
    with (
        patch("agent.ollama_client.OLLAMA_HOST", "http://127.0.0.1:11434"),
        patch("agent.ollama_client.requests.post", return_value=ps_mock),
        patch("idea_board.ollama_coder.acquire_coder_priority"),
        patch("idea_board.ollama_coder.release_coder_priority"),
        patch.object(OllamaCoder, "_start_mcp_bridge"),
        patch.object(OllamaCoder, "_stop_mcp_bridge"),
        patch.object(OllamaCoder, "_run_rounds"),
    ):
        # Create and run the coder
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp/test",
            idea_id="TK-922",
            state=mock_state,
            model="qwen3.5:27b",
        )
        coder.run()
        
        # Should still complete without raising an exception
        # The function handles failures gracefully


def test_run_handles_unload_failure(mock_state):
    """Test that run() handles /api/generate failure gracefully."""
    # Mock the ps endpoint to return two models
    ps_mock = MagicMock()
    ps_mock.status_code = 200
    ps_mock.json.return_value = {
        "models": [
            {"name": "qwen3.5:27b"},
            {"name": "llava:latest"},
        ]
    }
    
    # Mock the generate endpoint to return an error
    generate_mock = MagicMock()
    generate_mock.status_code = 500
    generate_mock.text = "Unload Failed"
    
    with (
        patch("agent.ollama_client.OLLAMA_HOST", "http://127.0.0.1:11434"),
        patch("agent.ollama_client.requests.post", side_effect=[ps_mock, generate_mock]),
        patch("idea_board.ollama_coder.acquire_coder_priority"),
        patch("idea_board.ollama_coder.release_coder_priority"),
        patch.object(OllamaCoder, "_start_mcp_bridge"),
        patch.object(OllamaCoder, "_stop_mcp_bridge"),
        patch.object(OllamaCoder, "_run_rounds"),
    ):
        # Create and run the coder
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp/test",
            idea_id="TK-922",
            state=mock_state,
            model="qwen3.5:27b",
        )
        coder.run()
        
        # Should still complete without raising an exception
        # The function handles failures gracefully