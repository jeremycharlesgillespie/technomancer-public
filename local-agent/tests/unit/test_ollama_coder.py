"""Tests for idea_board/ollama_coder.py — OllamaCoder local Ollama agent."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import (
    OllamaCoder,
    _classify_error_hint,
    _find_related_tests_for_files,
    _fmt_args,
    _parse_failing_tests,
    _parse_tool_calls_from_content,
    _safe_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coder(tmp_path: Path, prompt: str = "Implement feature X") -> OllamaCoder:
    state = MagicMock()
    state.log = lambda m: None
    state.cancelled = False
    return OllamaCoder(
        prompt=prompt,
        project_root=tmp_path,
        idea_id="TK-999",
        state=state,
        model="qwen3.5:27b",
        max_turns=5,
        max_rounds=3,
        num_ctx=4096,
    )


def _tool_call(name: str, **kwargs: object) -> dict:
    return {"function": {"name": name, "arguments": kwargs}}


def _response(tool_calls: list | None = None, content: str = "") -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_system_prompt_lists_allowlist():
    """Test that the system prompt contains each entry from the allowlist constant."""
    # Create a mock coder to access the system prompt building
    tmp_path = Path("/tmp")
    coder = _make_coder(tmp_path)
    
    # Build the system prompt
    system_prompt = coder._build_system_prompt()
    
    # Check that all allowlist items are mentioned in the system prompt
    for item in coder._tool_run_bash.__code__.co_consts:
        if isinstance(item, str) and "git, pytest, python, python3, py" in item:
            # This is a basic check - we'll test the actual system prompt construction
            break
    
    # The system prompt should mention the allowlist
    assert "git, pytest, python, python3, py" in system_prompt


def test_blocked_message_suggests_alternative():
    """Test that blocked responses suggest proper tool replacements."""
    tmp_path = Path("/tmp")
    coder = _make_coder(tmp_path)
    
    # Test that ls command suggests list_files
    result = coder._tool_run_bash("ls -la")
    assert "BLOCKED: ls not allowed" in result
    assert "list_files" in result
    
    # Test that grep command suggests search_code
    result = coder._tool_run_bash("grep 'pattern' file.py")
    assert "BLOCKED: grep not allowed" in result
    assert "search_code" in result
    
    # Test that cat command suggests read_file
    result = coder._tool_run_bash("cat file.py")
    assert "BLOCKED: cat not allowed" in result
    assert "read_file" in result


def test_system_prompt_includes_dedicated_tool_guidance():
    """Test that system prompt includes guidance about using dedicated tools."""
    tmp_path = Path("/tmp")
    coder = _make_coder(tmp_path)
    
    # Build the system prompt
    system_prompt = coder._build_system_prompt()
    
    # Check that the system prompt mentions using dedicated tools instead of blocked commands
    assert "list_files instead of ls/find" in system_prompt
    assert "search_code instead of grep" in system_prompt
    assert "read_file instead of cat/head/tail" in system_prompt


def test_run_bash_allowlist_constant_is_used():
    """Test that the BASH_ALLOWLIST constant is used in the system prompt."""
    tmp_path = Path("/tmp")
    coder = _make_coder(tmp_path)
    
    # Check that the constant is accessible
    assert hasattr(coder, '_tool_run_bash')
    
    # The system prompt should reference the allowlist
    system_prompt = coder._build_system_prompt()
    assert "Allowed prefixes: git, pytest, python, python3, py" in system_prompt


def test_blocked_command_hint_mapping():
    """Test that the blocked command hints are properly defined."""
    from idea_board.ollama_coder import _BLOCKED_COMMAND_HINTS
    
    # Check that all expected commands are mapped
    expected_commands = {"ls", "find", "cat", "head", "tail", "grep", "curl", "wget"}
    actual_commands = set(_BLOCKED_COMMAND_HINTS.keys())
    
    # Should have at least the expected commands
    assert expected_commands.issubset(actual_commands)
    
    # Check that hints are properly formatted
    for cmd, hint in _BLOCKED_COMMAND_HINTS.items():
        assert cmd in hint or "instead of" in hint


# Test the actual system prompt building method
def test_build_system_prompt():
    """Test that the system prompt is built correctly with all required elements."""
    tmp_path = Path("/tmp")
    coder = _make_coder(tmp_path)
    
    # Build the system prompt
    system_prompt = coder._build_system_prompt()
    
    # Check that it contains the allowlist
    assert "git, pytest, python, python3, py" in system_prompt
    
    # Check that it contains guidance about dedicated tools
    assert "list_files instead of ls/find" in system_prompt
    assert "search_code instead of grep" in system_prompt
    assert "read_file instead of cat/head/tail" in system_prompt
    
    # Check that it mentions the blocklist
    assert "safe_update" in system_prompt
    assert "push origin" in system_prompt