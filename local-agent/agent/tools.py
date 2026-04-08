"""
Agent Tools - File I/O, memory, and system tools.

Tools follow the pattern from the Cortex system:
- Simple interfaces (1-3 required params)
- Single responsibility
- Natural to call
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .core import Tool, create_tool

# =============================================================================
# FILE TOOLS
# =============================================================================


def read_file(path: str, max_lines: int = 500) -> str:
    """Read a file's contents."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def append_file(path: str, content: str) -> str:
    """Append content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error appending to file: {e}"


def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List files in a directory."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        files = list(p.glob(pattern))
        result = []
        for f in sorted(files)[:100]:  # Limit to 100 entries
            stat = f.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            ftype = "DIR" if f.is_dir() else "FILE"
            result.append(f"{ftype:4} {size:>10} {mtime} {f.name}")

        if len(files) > 100:
            result.append(f"... and {len(files) - 100} more")

        return "\n".join(result) if result else "Empty directory"
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(directory: str, pattern: str, content_pattern: str = "") -> str:
    """Search for files by name pattern, optionally filtering by content."""
    try:
        p = Path(directory).expanduser().resolve()
        if not p.exists():
            return f"Error: Directory not found: {directory}"

        matches = []
        for f in p.rglob(pattern):
            if f.is_file():
                if content_pattern:
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")
                        if content_pattern.lower() in text.lower():
                            matches.append(str(f))
                    except Exception:
                        pass
                else:
                    matches.append(str(f))

            if len(matches) >= 50:
                break

        return "\n".join(matches) if matches else "No matches found"
    except Exception as e:
        return f"Error searching: {e}"


# =============================================================================
# SYSTEM TOOLS
# =============================================================================

# Patterns for destructive commands that need explicit permission
DESTRUCTIVE_PATTERNS = [
    r"\brm\s",
    r"\bdel\s",
    r"\brmdir\s",
    r"\brd\s",  # delete
    r"\bformat\b",
    r"\bfdisk\b",
    r"\bmkfs\b",  # disk operations
    r"\bmv\s.*\s/",
    r"\bmove\s",  # move (can overwrite)
    r"\bdd\s",
    r"\b>\s*/",
    r"\btruncate\b",  # overwrite/truncate
    r"\bchmod\s",
    r"\bchown\s",
    r"\battrib\b",  # permission changes
    r"\bkill\b",
    r"\btaskkill\b",
    r"\bpkill\b",  # process killing
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s",  # system control
    r"\bgit\s+push",
    r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean",  # destructive git
    r"\bnpm\s+publish",
    r"\bpip\s+uninstall",  # package management
    r"\bdrop\s+database",
    r"\bdrop\s+table",
    r"\btruncate\s+table",  # SQL
]

# Safe read-only command patterns (always allowed)
SAFE_PATTERNS = [
    r"^ls\b",
    r"^dir\b",
    r"^pwd\b",
    r"^cd\b",
    r"^echo\b",
    r"^cat\b",
    r"^type\b",
    r"^head\b",
    r"^tail\b",
    r"^less\b",
    r"^more\b",
    r"^find\b",
    r"^grep\b",
    r"^rg\b",
    r"^ag\b",
    r"^fd\b",
    r"^wc\b",
    r"^du\b",
    r"^df\b",
    r"^free\b",
    r"^ps\b",
    r"^top\b",
    r"^htop\b",
    r"^tasklist\b",
    r"^whoami\b",
    r"^hostname\b",
    r"^uname\b",
    r"^date\b",
    r"^time\b",
    r"^git\s+status",
    r"^git\s+log",
    r"^git\s+diff",
    r"^git\s+show",
    r"^git\s+branch",
    r"^python\s+--version",
    r"^node\s+--version",
    r"^npm\s+list",
    r"^curl\b",
    r"^wget\b",
    r"^ping\b",
    r"^nslookup\b",
    r"^dig\b",
    r"^env\b",
    r"^printenv\b",
    r"^set\b",
]


def _is_safe_command(command: str) -> bool:
    """Check if command is explicitly safe (read-only)."""
    cmd_lower = command.lower().strip()
    return any(re.match(p, cmd_lower) for p in SAFE_PATTERNS)


def _is_destructive_command(command: str) -> bool:
    """Check if command matches destructive patterns."""
    cmd_lower = command.lower()
    return any(re.search(p, cmd_lower) for p in DESTRUCTIVE_PATTERNS)


def run_command(command: str, timeout: int = 60, allow_destructive: bool = False) -> str:
    """Run a shell command and return output.

    Safe commands (ls, cat, grep, git status, etc.) run freely.
    Destructive commands (rm, del, kill, etc.) require allow_destructive=True.
    """
    # Check for destructive commands
    if _is_destructive_command(command) and not allow_destructive:
        return (
            f"BLOCKED: This command looks destructive: {command}\n"
            f"If you really want to run it, call with allow_destructive=True"
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output = f"[Exit code: {result.returncode}]\n{output}"
        return output[:5000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"


def get_system_info() -> str:
    """Get basic system information."""
    import platform
    import socket

    # Detect Windows version properly (Win11 uses NT 10.0 but build 22000+)
    os_name = platform.system()
    os_version = platform.version()
    if os_name == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
            if build >= 22000:
                os_name = "Windows 11"
            else:
                os_name = "Windows 10"
        except Exception:
            pass

    info = {
        "hostname": socket.gethostname(),
        "os": os_name,
        "os_build": os_version,
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }
    return json.dumps(info, indent=2)


def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return json.dumps(
        {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
        },
        indent=2,
    )


# =============================================================================
# TOOL REGISTRY
# =============================================================================


def get_file_tools() -> list[Tool]:
    """Get all file-related tools."""
    return [
        create_tool(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to read (default 500)",
                    },
                },
                "required": ["path"],
            },
            function=read_file,
        ),
        create_tool(
            name="write_file",
            description="Write content to a file (creates or overwrites)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            function=write_file,
        ),
        create_tool(
            name="append_file",
            description="Append content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to append"},
                },
                "required": ["path", "content"],
            },
            function=append_file,
        ),
        create_tool(
            name="list_directory",
            description="List files in a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current)"},
                    "pattern": {"type": "string", "description": "Glob pattern (default: *)"},
                },
                "required": [],
            },
            function=list_directory,
        ),
        create_tool(
            name="search_files",
            description="Search for files by name, optionally filtering by content",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search"},
                    "pattern": {"type": "string", "description": "Filename pattern (e.g., *.py)"},
                    "content_pattern": {
                        "type": "string",
                        "description": "Search within file contents",
                    },
                },
                "required": ["directory", "pattern"],
            },
            function=search_files,
        ),
    ]


def get_system_tools() -> list[Tool]:
    """Get system-related tools."""
    return [
        create_tool(
            name="run_command",
            description="Run a shell command. Safe commands (ls, cat, grep, git status, etc.) run freely. Destructive commands (rm, del, kill) need allow_destructive=True.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60)",
                    },
                    "allow_destructive": {
                        "type": "boolean",
                        "description": "Set to true to allow destructive commands (rm, del, kill, etc.)",
                    },
                },
                "required": ["command"],
            },
            function=run_command,
        ),
        create_tool(
            name="get_system_info",
            description="Get basic system information",
            parameters={"type": "object", "properties": {}, "required": []},
            function=get_system_info,
        ),
        create_tool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={"type": "object", "properties": {}, "required": []},
            function=get_current_time,
        ),
    ]


from .pdf_tools import get_pdf_tools  # noqa: E402


def get_all_tools() -> list[Tool]:
    """Get all available tools (file, system, PDF)."""
    return get_file_tools() + get_system_tools() + get_pdf_tools()
