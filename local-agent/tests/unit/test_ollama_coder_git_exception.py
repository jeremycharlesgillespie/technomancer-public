"""Tests for Git exception handling in OllamaCoder."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import OllamaCoder
from agent.accountability import GitNotInstalledError, GitDirtyError


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


class TestGitExceptionHandling:
    """Test that Git exceptions stop code generation."""

    def test_write_file_not_called_after_git_exception(self, tmp_path: Path) -> None:
        """Test that write_file is not called if Git exception occurs."""
        coder = _make_coder(tmp_path)
        
        # Mock the _chat_with_tools to return a response that includes write_file call
        def mock_chat(sys_prompt, messages):
            return _response([_tool_call("write_file", path="test.py", content="print('hello')")])
        
        coder._chat_with_tools = mock_chat
        
        # Mock _run_pytest to return a failure so we go through the loop
        coder._run_pytest = lambda: {
            "passed": False,
            "failing": ["tests/unit/test_foo.py::test_bar"],
            "output": "FAILED tests/unit/test_foo.py::test_bar\nAssertionError: 0 != 1",
        }
        
        # Mock _get_changed_files to return empty list
        coder._get_changed_files = lambda: []
        
        # Mock _tag_round_commits to raise GitNotInstalledError
        def mock_tag_round_commits(round_num):
            if round_num == 0:
                # Raise GitNotInstalledError on first round commit tagging
                raise GitNotInstalledError("Git not installed")
            return None
            
        coder._tag_round_commits = mock_tag_round_commits
        
        # Mock _run_bash to avoid actual git commands
        coder._tool_run_bash = lambda cmd: "command executed" if "git" not in cmd else "git command blocked"
        
        # Mock the file system operations to track if they're called
        write_file_called = []
        edit_file_called = []
        
        original_write_file = coder._tool_write_file
        original_edit_file = coder._tool_edit_file
        
        def mock_write_file(path, content):
            write_file_called.append((path, content))
            return original_write_file(path, content)
            
        def mock_edit_file(path, old_string, new_string):
            edit_file_called.append((path, old_string, new_string))
            return original_edit_file(path, old_string, new_string)
            
        coder._tool_write_file = mock_write_file
        coder._tool_edit_file = mock_edit_file
        
        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            # This should raise GitNotInstalledError and not call write_file
            with pytest.raises(GitNotInstalledError):
                coder.run()
            
        # Verify that write_file and edit_file were never called
        assert len(write_file_called) == 0
        assert len(edit_file_called) == 0

    def test_write_file_not_called_after_git_dirty_exception(self, tmp_path: Path) -> None:
        """Test that write_file is not called if GitDirtyError occurs."""
        coder = _make_coder(tmp_path)
        
        # Mock the _chat_with_tools to return a response that includes write_file call
        def mock_chat(sys_prompt, messages):
            return _response([_tool_call("write_file", path="test.py", content="print('hello')")])
        
        coder._chat_with_tools = mock_chat
        
        # Mock _run_pytest to return a failure so we go through the loop
        coder._run_pytest = lambda: {
            "passed": False,
            "failing": ["tests/unit/test_foo.py::test_bar"],
            "output": "FAILED tests/unit/test_foo.py::test_bar\nAssertionError: 0 != 1",
        }
        
        # Mock _get_changed_files to return empty list
        coder._get_changed_files = lambda: []
        
        # Mock _tag_round_commits to raise GitDirtyError
        def mock_tag_round_commits(round_num):
            if round_num == 0:
                # Raise GitDirtyError on first round commit tagging
                raise GitDirtyError("Git working directory is dirty")
            return None
            
        coder._tag_round_commits = mock_tag_round_commits
        
        # Mock _run_bash to avoid actual git commands
        coder._tool_run_bash = lambda cmd: "command executed" if "git" not in cmd else "git command blocked"
        
        # Mock the file system operations to track if they're called
        write_file_called = []
        edit_file_called = []
        
        original_write_file = coder._tool_write_file
        original_edit_file = coder._tool_edit_file
        
        def mock_write_file(path, content):
            write_file_called.append((path, content))
            return original_write_file(path, content)
            
        def mock_edit_file(path, old_string, new_string):
            edit_file_called.append((path, old_string, new_string))
            return original_edit_file(path, old_string, new_string)
            
        coder._tool_write_file = mock_write_file
        coder._tool_edit_file = mock_edit_file
        
        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            # This should raise GitDirtyError and not call write_file
            with pytest.raises(GitDirtyError):
                coder.run()
            
        # Verify that write_file and edit_file were never called
        assert len(write_file_called) == 0
        assert len(edit_file_called) == 0

    def test_no_commits_made_on_git_exception(self, tmp_path: Path) -> None:
        """Test that no commits are made when Git exception occurs."""
        coder = _make_coder(tmp_path)
        
        # Mock the _chat_with_tools to return a response that includes write_file call
        def mock_chat(sys_prompt, messages):
            return _response([_tool_call("write_file", path="test.py", content="print('hello')")])
        
        coder._chat_with_tools = mock_chat
        
        # Mock _run_pytest to return a failure so we go through the loop
        coder._run_pytest = lambda: {
            "passed": False,
            "failing": ["tests/unit/test_foo.py::test_bar"],
            "output": "FAILED tests/unit/test_foo.py::test_bar\nAssertionError: 0 != 1",
        }
        
        # Mock _get_changed_files to return empty list
        coder._get_changed_files = lambda: []
        
        # Mock _tag_round_commits to raise GitNotInstalledError
        def mock_tag_round_commits(round_num):
            if round_num == 0:
                # Raise GitNotInstalledError on first round commit tagging
                raise GitNotInstalledError("Git not installed")
            return None
            
        coder._tag_round_commits = mock_tag_round_commits
        
        # Mock _run_bash to avoid actual git commands
        coder._tool_run_bash = lambda cmd: "command executed" if "git" not in cmd else "git command blocked"
        
        # Mock the file system operations to track if they're called
        write_file_called = []
        edit_file_called = []
        
        original_write_file = coder._tool_write_file
        original_edit_file = coder._tool_edit_file
        
        def mock_write_file(path, content):
            write_file_called.append((path, content))
            return original_write_file(path, content)
            
        def mock_edit_file(path, old_string, new_string):
            edit_file_called.append((path, old_string, new_string))
            return original_edit_file(path, old_string, new_string)
            
        coder._tool_write_file = mock_write_file
        coder._tool_edit_file = mock_edit_file
        
        with patch("agent.ollama_client.acquire_coder_priority"), \
             patch("agent.ollama_client.release_coder_priority"):
            # This should raise GitNotInstalledError and not call write_file
            with pytest.raises(GitNotInstalledError):
                coder.run()
            
        # Verify that write_file and edit_file were never called
        assert len(write_file_called) == 0
        assert len(edit_file_called) == 0