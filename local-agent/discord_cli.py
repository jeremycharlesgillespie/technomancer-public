"""
Discord CLI Helper — Command-line interface for the Discord Bridge API.

Lets Claude Code (or any local script) interact with Discord from the
terminal without needing to know HTTP details.

Usage:
    python discord_cli.py send "Hello from Claude Code!"
    python discord_cli.py ask "Should I proceed with the refactor?"
    python discord_cli.py history
    python discord_cli.py history --limit 5
    python discord_cli.py status
    python discord_cli.py reply <message_id> "Here's the fix"

The bridge token is read automatically from .bridge_token in the same
directory. The bridge API must be running (starts with the Discord bot).

Exit codes:
    0 — Success
    1 — Error (message printed to stderr)
    2 — Timeout (for ask command)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

#: Bridge API base URL (localhost only)
BASE_URL: str = "http://127.0.0.1:8321"

#: Path to the bridge token file
TOKEN_FILE: Path = Path(__file__).parent / ".bridge_token"


def _load_token() -> str:
    """Load the bridge authentication token from disk.

    Returns:
        The token string

    Raises:
        SystemExit: If the token file doesn't exist
    """
    if not TOKEN_FILE.exists():
        print(f"Error: Bridge token not found at {TOKEN_FILE}", file=sys.stderr)
        print("Is the Discord bot running with the bridge API?", file=sys.stderr)
        sys.exit(1)
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def _headers() -> dict[str, str]:
    """Build request headers with authentication.

    Returns:
        Dict with Content-Type and X-Bridge-Token headers
    """
    return {
        "Content-Type": "application/json",
        "X-Bridge-Token": _load_token(),
    }


def _request(method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make an authenticated request to the bridge API.

    Args:
        method: HTTP method ("GET" or "POST")
        path: API path (e.g., "/api/send")
        data: JSON body for POST requests

    Returns:
        Parsed JSON response

    Raises:
        SystemExit: On connection error or non-200 response
    """
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, headers=_headers(), timeout=310)
        else:
            resp = requests.post(url, headers=_headers(), json=data, timeout=310)
    except requests.ConnectionError:
        print(
            "Error: Cannot connect to bridge API at "
            f"{BASE_URL}\nIs the Discord bot running?",
            file=sys.stderr,
        )
        sys.exit(1)

    result = resp.json()

    if resp.status_code == 408:
        # Timeout on /api/ask
        print("Timed out waiting for user reply.", file=sys.stderr)
        sys.exit(2)

    if resp.status_code != 200:
        print(f"Error ({resp.status_code}): {result.get('error', 'Unknown')}", file=sys.stderr)
        sys.exit(1)

    return result


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_send(message: str) -> None:
    """Send a message to the Discord channel.

    Args:
        message: The text to send
    """
    result = _request("POST", "/api/send", {"message": message})
    print(f"Sent (ID: {result['message_id']})")


def cmd_ask(question: str, timeout: int = 300) -> None:
    """Ask a question in Discord and wait for the user's reply.

    Blocks until the user replies or the timeout expires.

    Args:
        question: The question to ask
        timeout: Seconds to wait (default 300)
    """
    result = _request("POST", "/api/ask", {"question": question, "timeout": timeout})
    print(f"{result.get('reply_author', 'User')}: {result.get('reply', '')}")


def cmd_history(limit: int = 20) -> None:
    """Print recent message history from the Discord channel.

    Args:
        limit: Number of messages to show (default 20)
    """
    result = _request("GET", f"/api/history?limit={limit}")
    for msg in result.get("messages", []):
        prefix = "[BOT] " if msg["is_bot"] else ""
        ts = msg["timestamp"][:16]  # Trim to minutes
        print(f"[{ts}] {prefix}{msg['author']}: {msg['content'][:200]}")


def cmd_reply(message_id: str, message: str) -> None:
    """Reply to a specific Discord message.

    Args:
        message_id: The Discord message ID to reply to
        message: The reply text
    """
    result = _request("POST", "/api/reply", {"message_id": message_id, "message": message})
    print(f"Replied (ID: {result['reply_id']})")


def cmd_status() -> None:
    """Print bridge and bot status."""
    result = _request("GET", "/api/status")
    print(json.dumps(result, indent=2))


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

USAGE = """Discord CLI Helper — interact with Discord from the terminal.

Commands:
    send <message>              Send a message to Discord
    ask <question>              Ask a question and wait for reply
    history [--limit N]         Show recent messages (default 20)
    reply <message_id> <text>   Reply to a specific message
    status                      Show bridge/bot status

Examples:
    python discord_cli.py send "Build completed successfully!"
    python discord_cli.py ask "Should I deploy to production?"
    python discord_cli.py history --limit 5
"""

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        sys.exit(0)

    command = args[0].lower()

    if command == "send" and len(args) >= 2:
        cmd_send(" ".join(args[1:]))
    elif command == "ask" and len(args) >= 2:
        cmd_ask(" ".join(args[1:]))
    elif command == "history":
        limit = 20
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                limit = int(args[idx + 1])
        cmd_history(limit)
    elif command == "reply" and len(args) >= 3:
        cmd_reply(args[1], " ".join(args[2:]))
    elif command == "status":
        cmd_status()
    else:
        print(USAGE)
        sys.exit(1)
