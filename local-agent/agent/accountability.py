"""
Accountability Check - Verify that claimed actions actually occurred.

Provides verification tools so the LLM can confirm its actions before
reporting success to the user.
"""

import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings
VAULT_PATH = settings.llm_memory_path

logger = logging.getLogger(__name__)


class GitNotInstalledError(Exception):
    """Raised when git is not installed or unavailable."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Git is not installed or not in PATH")
        logger.debug("GitNotInstalledError: %s", self)


class GitDirtyError(Exception):
    """Raised when the working directory has uncommitted changes."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Git working directory is dirty - uncommitted changes detected")
        logger.debug("GitDirtyError: %s", self)


def verify_file_exists(file_path: str) -> str:
    """
    Verify that a file exists at the given path.

    Args:
        file_path: Path to check (can be relative to vault or absolute)

    Returns:
        Verification result with details
    """
    # Try as absolute path first
    path = Path(file_path)
    if not path.is_absolute():
        # Try relative to vault
        path = VAULT_PATH / file_path

    if path.exists():
        # Get file stats
        stat = path.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        return f"VERIFIED: File exists at {file_path} ({size} bytes, modified {mtime.strftime('%Y-%m-%d %H:%M:%S')})"
    else:
        return f"NOT FOUND: File not found at {file_path}"


def verify_file_modified_recently(file_path: str, minutes: int = 5) -> str:
    """
    Verify that a file was modified within the specified number of minutes.

    Args:
        file_path: Path to check (can be relative to vault or absolute)
        minutes: Number of minutes to check back

    Returns:
        Verification result with details
    """
    # Try as absolute path first
    path = Path(file_path)
    if not path.is_absolute():
        # Try relative to vault
        path = VAULT_PATH / file_path

    if not path.exists():
        return f"NOT FOUND: File not found at {file_path}"

    # Get file stats
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime)
    now = datetime.now()
    diff = now - mtime

    if diff <= timedelta(minutes=minutes):
        return f"VERIFIED: File was modified {diff.seconds} seconds ago ({mtime.strftime('%Y-%m-%d %H:%M:%S')})"
    else:
        return f"NOT FOUND: File was modified {diff.seconds} seconds ago, but {minutes} minutes required"


def verify_content_contains(file_path: str, search_text: str) -> str:
    """
    Verify that a file contains the specified text.

    Args:
        file_path: Path to check (can be relative to vault or absolute)
        search_text: Text to search for

    Returns:
        Verification result with details
    """
    # Try as absolute path first
    path = Path(file_path)
    if not path.is_absolute():
        # Try relative to vault
        path = VAULT_PATH / file_path

    if not path.exists():
        return f"NOT FOUND: File not found at {file_path}"

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"ERROR: Failed to read file {file_path}: {str(e)}"

    # Truncate search text for display
    display_text = search_text if len(search_text) <= 50 else search_text[:50] + "..."

    if search_text in content:
        return f"VERIFIED: File contains '{display_text}'"
    else:
        return f"NOT FOUND: File does not contain '{display_text}'"


def verify_memory_saved(category: str) -> str:
    """
    Verify that a memory category was saved to the vault.

    Args:
        category: Memory category name

    Returns:
        Verification result with details
    """
    memory_file = VAULT_PATH / "Permanent" / f"{category}.md"

    if memory_file.exists():
        stat = memory_file.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        return f"VERIFIED: Memory saved in category '{category}' ({size} bytes, modified {mtime.strftime('%Y-%m-%d %H:%M:%S')})"
    else:
        return f"NOT FOUND: Memory not saved in category '{category}'"


def verify_git_clean() -> str:
    """
    Verify that the git working directory is clean (no uncommitted changes).

    Returns:
        Verification result with details
    """
    try:
        # First, explicitly check if git is available by running git rev-parse
        # This will catch cases where git is not in PATH or not executable
        git_check_result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # We want to handle non-zero exit codes explicitly
        )
        
        # If git command failed with non-zero exit code, check if it's because git is not installed
        if git_check_result.returncode != 0:
            # Check if the error indicates git is not found in PATH
            if "not found" in git_check_result.stderr.lower() or "command not found" in git_check_result.stderr.lower():
                raise GitNotInstalledError("Git is not installed or not in PATH")
            # If git command failed for other reasons, we'll let the existing logic handle it
            # This could be because we're not in a git repository, etc.
        
        # If we get here, git is available, so proceed with normal checks
        # First check if we're in a git repository by running git rev-parse
        # This will raise FileNotFoundError if git is not in PATH (though we already checked)
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # If we get here, git is installed and we're in a git repository
        # Now check if working directory is clean
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if status_result.stdout.strip():
            # Working directory has uncommitted changes
            return f"DIRTY: Git working directory has uncommitted changes:\n{status_result.stdout.strip()}"
        else:
            # Working directory is clean
            return "VERIFIED: Git working directory is clean"
            
    except subprocess.CalledProcessError as e:
        # Git command failed, likely because we're not in a git repository
        if "fatal: not a git repository" in e.stderr:
            return "NOT FOUND: Not in a git repository"
        else:
            # Some other git error
            return f"ERROR: Git command failed: {e.stderr.strip()}"
    except FileNotFoundError:
        # Git is not installed or not in PATH
        raise GitNotInstalledError("Git is not installed or not in PATH")


def get_accountability_tools():
    """
    Get all accountability verification tools.

    Returns:
        List of Tool objects for verification functions
    """
    from .tools import create_tool

    return [
        create_tool(
            name="verify_file_exists",
            description="Verify that a file exists at the given path",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to check (can be relative to vault or absolute)"
                    }
                },
                "required": ["file_path"]
            },
            function=verify_file_exists
        ),
        create_tool(
            name="verify_file_modified",
            description="Verify that a file was modified within the specified number of minutes",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to check (can be relative to vault or absolute)"
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "Number of minutes to check back",
                        "default": 5
                    }
                },
                "required": ["file_path"]
            },
            function=verify_file_modified_recently
        ),
        create_tool(
            name="verify_content",
            description="Verify that a file contains the specified text",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to check (can be relative to vault or absolute)"
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Text to search for"
                    }
                },
                "required": ["file_path", "search_text"]
            },
            function=verify_content_contains
        ),
        create_tool(
            name="verify_memory_saved",
            description="Verify that a memory category was saved to the vault",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Memory category name"
                    }
                },
                "required": ["category"]
            },
            function=verify_memory_saved
        ),
        create_tool(
            name="verify_git_clean",
            description="Verify that the git working directory is clean (no uncommitted changes)",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=verify_git_clean
        ),
    ]