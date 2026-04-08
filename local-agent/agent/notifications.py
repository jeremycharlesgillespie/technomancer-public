"""
Notifications - Send alerts to Discord, ntfy, etc.
"""

import os
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

from .discord_rate_limit import retry_request

# Webhook URL loaded from environment — never hardcode secrets
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def discord_send(
    message: str,
    title: str = None,
    color: int = 0x5865F2,  # Discord blurple
    webhook_url: str = None,
) -> str:
    """
    Send a message to Discord via webhook.

    Args:
        message: The message to send
        title: Optional embed title
        color: Embed color (hex as int)
        webhook_url: Override default webhook
    """
    if requests is None:
        return "Error: requests library not installed"

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return "Error: No webhook URL configured"

    try:
        # If title provided, use an embed for nicer formatting
        payload: dict[str, object]
        if title:
            payload = {
                "embeds": [
                    {
                        "title": title,
                        "description": message,
                        "color": color,
                        "timestamp": datetime.utcnow().isoformat(),
                        "footer": {"text": "Local Agent"},
                    }
                ]
            }
        else:
            # Simple message
            payload = {"content": message}

        response = retry_request(
            requests.post,
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 204:
            return f"Sent to Discord: {message[:50]}..."
        else:
            return f"Discord error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Error sending to Discord: {e}"


def discord_send_file(
    file_path: str,
    message: str = "",
    webhook_url: str = None,
) -> str:
    """Send a file to Discord."""
    if requests is None:
        return "Error: requests library not installed"

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return "Error: No webhook URL configured"

    try:
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "rb") as f:
            response = retry_request(
                requests.post,
                url,
                data={"content": message} if message else None,
                files={"file": (path.name, f)},
                timeout=30,
            )

        if response.status_code == 200:
            return f"Sent file to Discord: {path.name}"
        else:
            return f"Discord error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Error: {e}"


def discord_send_code(
    code: str,
    language: str = "",
    message: str = "",
    webhook_url: str = None,
) -> str:
    """Send a code block to Discord."""
    formatted = (
        f"{message}\n```{language}\n{code}\n```" if message else f"```{language}\n{code}\n```"
    )
    return discord_send(formatted, webhook_url=webhook_url)


# Color presets for embeds
COLORS = {
    "success": 0x57F287,  # Green
    "error": 0xED4245,  # Red
    "warning": 0xFEE75C,  # Yellow
    "info": 0x5865F2,  # Blurple
    "purple": 0x9B59B6,
}


def discord_alert(
    message: str,
    level: str = "info",  # success, error, warning, info
    title: str = None,
    webhook_url: str = None,
) -> str:
    """Send a colored alert to Discord."""
    color = COLORS.get(level, COLORS["info"])
    title = title or level.upper()
    return discord_send(message, title=title, color=color, webhook_url=webhook_url)


def get_notification_tools() -> list:
    """Get notification tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="discord_send",
            description="Send a message to Discord",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send"},
                    "title": {"type": "string", "description": "Optional embed title"},
                },
                "required": ["message"],
            },
            function=discord_send,
        ),
        create_tool(
            name="discord_alert",
            description="Send a colored alert to Discord (success/error/warning/info)",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Alert message"},
                    "level": {
                        "type": "string",
                        "description": "Alert level: success, error, warning, info",
                    },
                    "title": {"type": "string", "description": "Optional custom title"},
                },
                "required": ["message"],
            },
            function=discord_alert,
        ),
        create_tool(
            name="discord_send_code",
            description="Send a code block to Discord",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to send"},
                    "language": {
                        "type": "string",
                        "description": "Language for syntax highlighting",
                    },
                    "message": {"type": "string", "description": "Optional message before code"},
                },
                "required": ["code"],
            },
            function=discord_send_code,
        ),
        create_tool(
            name="discord_send_file",
            description="Send a file to Discord",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "message": {"type": "string", "description": "Optional message"},
                },
                "required": ["file_path"],
            },
            function=discord_send_file,
        ),
    ]
