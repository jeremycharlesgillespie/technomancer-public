"""
Accountability Check - Verify that claimed actions actually occurred.

Provides verification tools so the LLM can confirm its actions before
reporting success to the user.
"""

import logging
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
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
        return f"VERIFIED: File exists at {path}. Size: {size} bytes. Last modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return f"NOT FOUND: No file at {path}"


def verify_file_modified_recently(file_path: str, minutes: int = 5) -> str:
    """
    Verify that a file was modified within the last N minutes.

    Args:
        file_path: Path to check
        minutes: How recent the modification should be (default 5)

    Returns:
        Verification result
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = VAULT_PATH / file_path

    if not path.exists():
        return f"NOT FOUND: No file at {path}"

    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime)
    cutoff = datetime.now() - timedelta(minutes=minutes)

    if modified >= cutoff:
        return f"VERIFIED: File was modified {(datetime.now() - modified).seconds} seconds ago at {modified.strftime('%H:%M:%S')}"
    else:
        return f"STALE: File was last modified {modified.strftime('%Y-%m-%d %H:%M:%S')}, which is more than {minutes} minutes ago"


def verify_content_contains(file_path: str, search_text: str) -> str:
    """
    Verify that a file contains specific text.

    Args:
        file_path: Path to check
        search_text: Text to search for

    Returns:
        Verification result
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = VAULT_PATH / file_path

    if not path.exists():
        return f"NOT FOUND: No file at {path}"

    try:
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            return (
                f"VERIFIED: File contains the text '{search_text[:50]}...'"
                if len(search_text) > 50
                else f"VERIFIED: File contains '{search_text}'"
            )
        else:
            return (
                f"NOT FOUND: File exists but does not contain '{search_text[:50]}...'"
                if len(search_text) > 50
                else f"NOT FOUND: File exists but does not contain '{search_text}'"
            )
    except Exception as e:
        return f"ERROR: Could not read file: {e}"


def verify_memory_saved(category: str) -> str:
    """
    Verify that a memory was saved to a category.

    Args:
        category: Memory category (e.g., "user_info/{username}")

    Returns:
        Verification result
    """
    memory_path = VAULT_PATH / "Permanent" / f"{category}.md"

    if memory_path.exists():
        stat = memory_path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size

        # Check if modified recently (within last 2 minutes)
        if datetime.now() - modified < timedelta(minutes=2):
            return f"VERIFIED: Memory saved to {category}. Size: {size} bytes. Just modified at {modified.strftime('%H:%M:%S')}"
        else:
            return f"EXISTS: Memory file exists but was last modified at {modified.strftime('%H:%M:%S')} (not in last 2 minutes)"
    else:
        return f"NOT FOUND: No memory file for category '{category}'"


def verify_git_clean() -> str:
    """
    Verify that the git working directory is clean (no uncommitted changes).

    Returns:
        Verification result with details about git status
    """
    import subprocess
    import shlex

    try:
        # Check if git is available
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return f"NOT FOUND: Git repository not found or git is not installed"
        
        # Get git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        status_output = result.stdout
        
        if not status_output.strip():
            return "VERIFIED: Git working directory is clean - no uncommitted changes"
        else:
            # Parse the status output to get a summary
            lines = status_output.strip().split('\n')
            modified_files = [line for line in lines if line.startswith(" M") or line.startswith(" D")]
            untracked_files = [line for line in lines if line.startswith("?")]
            
            modified_summary = ", ".join(modified_files) if modified_files else "none"
            untracked_summary = ", ".join(untracked_files) if untracked_files else "none"
            
            return f"DIRTY: Git working directory has uncommitted changes: {modified_summary} {untracked_summary}"
    except subprocess.TimeoutExpired:
        return "ERROR: Git command timed out"
    except FileNotFoundError:
        return "NOT FOUND: Git is not installed or not in PATH"
    except Exception as e:
        return f"ERROR: Could not check git status: {e}"


def get_accountability_tools() -> list:
    """Get verification tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "verify_file_exists",
            (
                "Verify that a file exists. Use this BEFORE telling the user you saved/created a file. "
                "Returns VERIFIED if file exists, NOT FOUND otherwise."
            ),
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to verify (absolute or relative to vault)",
                    }
                },
                "required": ["file_path"],
            },
            verify_file_exists,
        ),
        create_tool(
            "verify_file_modified",
            (
                "Verify that a file was modified recently. Use this to confirm your write/edit actually worked. "
                "Returns VERIFIED if modified within N minutes, STALE otherwise."
            ),
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to check"},
                    "minutes": {"type": "integer", "description": "How recent (default 5 minutes)"},
                },
                "required": ["file_path"],
            },
            lambda file_path, minutes=5: verify_file_modified_recently(file_path, minutes),
        ),
        create_tool(
            "verify_content",
            (
                "Verify that a file contains specific text. Use this to confirm your write included the expected content."
            ),
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to check"},
                    "search_text": {
                        "type": "string",
                        "description": "Text that should be in the file",
                    },
                },
                "required": ["file_path", "search_text"],
            },
            verify_content_contains,
        ),
        create_tool(
            "verify_memory_saved",
            (
                "Verify that a memory was saved to a category. Use AFTER calling remember_permanently "
                "to confirm the save actually worked before telling the user."
            ),
            {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Memory category like 'user_info/username'",
                    }
                },
                "required": ["category"],
            },
            verify_memory_saved,
        ),
    ]
