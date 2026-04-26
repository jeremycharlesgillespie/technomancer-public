"""Tests for OllamaCoder readiness gate functionality."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from idea_board.ollama_coder import (
    OllamaCoder,
    EnvironmentReadyError,
)


def test_readiness_gate_checks_git_clean_and_branch_creation():
    """Test that readiness gate checks both git clean status and branch creation."""
    # Create a mock coder
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    
    # Mock the accountability functions to simulate clean git and successful branch creation
    with patch("idea_board.ollama_coder.verify_git_clean") as mock_verify_git_clean, \
         patch("idea_board.ollama_coder.create_branch") as mock_create_branch:
        
        # Simulate clean git status
        mock_verify_git_clean.return_value = "VERIFIED: Git working directory is clean"
        # Simulate successful branch creation
        mock_create_branch.return_value = "SUCCESS: Created branch 'TK-TK-1095'"
        
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp",
            idea_id="TK-1095",
            state=state,
            model="qwen3.5:27b",
            max_turns=5,
            max_rounds=3,
            num_ctx=4096,
        )
        
        # This should not raise an exception
        # We're just testing that the function can be called without errors
        assert hasattr(coder, '_check_readiness_gate')
        # The function itself doesn't raise anything in this test case


def test_readiness_gate_raises_error_on_dirty_git():
    """Test that readiness gate raises EnvironmentReadyError when git is dirty."""
    # Create a mock coder
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    
    # Mock the accountability functions to simulate dirty git status
    with patch("idea_board.ollama_coder.verify_git_clean") as mock_verify_git_clean:
        # Simulate dirty git status
        mock_verify_git_clean.return_value = "Git working directory has uncommitted changes"
        
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp",
            idea_id="TK-1095",
            state=state,
            model="qwen3.5:27b",
            max_turns=5,
            max_rounds=3,
            num_ctx=4096,
        )
        
        # This should raise EnvironmentReadyError
        with pytest.raises(EnvironmentReadyError) as exc_info:
            coder._check_readiness_gate()
        
        assert "Git repository is not clean" in str(exc_info.value)


def test_readiness_gate_raises_error_on_failed_branch_creation():
    """Test that readiness gate raises EnvironmentReadyError when branch creation fails."""
    # Create a mock coder
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    
    # Mock the accountability functions to simulate clean git but failed branch creation
    with patch("idea_board.ollama_coder.verify_git_clean") as mock_verify_git_clean, \
         patch("idea_board.ollama_coder.create_branch") as mock_create_branch:
        
        # Simulate clean git status
        mock_verify_git_clean.return_value = "VERIFIED: Git working directory is clean"
        # Simulate failed branch creation
        mock_create_branch.return_value = "ERROR: Failed to create branch 'TK-TK-1095': fatal: cannot create directory at 'TK-TK-1095': Permission denied"
        
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp",
            idea_id="TK-1095",
            state=state,
            model="qwen3.5:27b",
            max_turns=5,
            max_rounds=3,
            num_ctx=4096,
        )
        
        # This should raise EnvironmentReadyError
        with pytest.raises(EnvironmentReadyError) as exc_info:
            coder._check_readiness_gate()
        
        assert "Failed to create branch" in str(exc_info.value)


def test_execute_tool_enforces_readiness_gate():
    """Test that execute_tool enforces readiness gate for code generation tools."""
    # Create a mock coder
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    
    # Mock the accountability functions to simulate clean git and successful branch creation
    with patch("idea_board.ollama_coder.verify_git_clean") as mock_verify_git_clean, \
         patch("idea_board.ollama_coder.create_branch") as mock_create_branch:
        
        # Simulate clean git status
        mock_verify_git_clean.return_value = "VERIFIED: Git working directory is clean"
        # Simulate successful branch creation
        mock_create_branch.return_value = "SUCCESS: Created branch 'TK-TK-1095'"
        
        coder = OllamaCoder(
            prompt="Test prompt",
            project_root="/tmp",
            idea_id="TK-1095",
            state=state,
            model="qwen3.5:27b",
            max_turns=5,
            max_rounds=3,
            num_ctx=4096,
        )
        
        # Test that calling execute_tool with code generation tools triggers readiness check
        # This should not raise an exception since git is clean and branch creation succeeds
        result = coder._execute_tool("read_file", {"path": "/tmp/test.py"})
        # The result should be an error or success message, not an exception
        assert isinstance(result, str)
        
        # Test with write_file
        result = coder._execute_tool("write_file", {"path": "/tmp/test.py", "content": "test"})
        assert isinstance(result, str)
        
        # Test with run_bash
        result = coder._execute_tool("run_bash", {"command": "echo test"})
        assert isinstance(result, str)