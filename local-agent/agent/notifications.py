"""
Notifications - Send alerts to Discord, ntfy, etc.
"""

import os
from datetime import datetime, timezone
from typing import Any

try:
    import requests
except ImportError:
    requests = None

from .config import settings
from .discord_rate_limit import retry_request

# Webhook URL loaded from environment — never hardcode secrets
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def discord_send(
    message: str,
    title: str | None = None,
    color: int = 0x5865F2,  # Discord blurple
    webhook_url: str | None = None,
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
                        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    webhook_url: str | None = None,
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
    webhook_url: str | None = None,
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
    title: str | None = None,
    webhook_url: str | None = None,
) -> str:
    """Send a colored alert to Discord."""
    color = COLORS.get(level, COLORS["info"])
    title = title or level.upper()
    return discord_send(message, title=title, color=color, webhook_url=webhook_url)


# ---------------------------------------------------------------------------
# Executor run summary — posted to Discord on every completed executor run
# ---------------------------------------------------------------------------

# Status codes that count as success. Everything else renders as a failure
# and triggers the stderr tail in the embed body.
_EXECUTOR_SUCCESS_STATUSES = frozenset({"success"})

# How many trailing lines of stderr to include on failure runs.
_STDERR_TAIL_LINES = 20

# Idea board port — matches agent/bot_commands.py and idea_board/web.py.
_IDEA_BOARD_PORT = 8322


def _executor_summary_webhook() -> str:
    """Return the webhook URL for executor-run summaries.

    Falls back to ``discord_webhook_url`` when the dedicated
    ``executor_summary_webhook`` setting is empty, so ops get notifications
    out of the box without extra configuration.
    """
    dedicated = (settings.executor_summary_webhook or "").strip()
    if dedicated:
        return dedicated
    fallback = settings.discord_webhook_url or ""
    return fallback.strip()


def _format_duration_ms(duration_ms: int | float | str | None) -> str:
    """Render ``duration_ms`` for humans (e.g. ``12345`` -> ``12.3s``).

    Missing or non-numeric values render as ``"?"`` rather than raising so
    the summary still posts when instrumentation has a gap.
    """
    try:
        ms = int(duration_ms)
    except (TypeError, ValueError):
        return "?"
    if ms < 1000:
        return f"{ms}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    mins, secs_rem = divmod(secs, 60)
    return f"{int(mins)}m{secs_rem:04.1f}s"


def _format_cost_usd(cost_usd: int | float | str | None) -> str:
    """Render ``cost_usd`` for humans. Missing values render as ``$?``."""
    try:
        return f"${float(cost_usd):.4f}"
    except (TypeError, ValueError):
        return "$?"


def _stderr_tail(stderr: str | None, lines: int = _STDERR_TAIL_LINES) -> str:
    """Return the last ``lines`` lines of ``stderr``. Empty string when blank."""
    if not stderr:
        return ""
    tail = stderr.splitlines()[-lines:]
    return "\n".join(tail).strip()


def build_executor_summary_payload(run_record: dict[str, Any]) -> dict[str, Any]:
    """Build the Discord webhook payload for an executor run summary.

    The payload has a single embed:

    * title: ``[<jira_key>] <idea title>`` (falls back to the run_id)
    * color: green for success, red otherwise
    * fields: status, duration, cost
    * description: a link to ``/executor-runs#<run_id>``; for failures the
      last 20 lines of stderr are appended in a fenced code block.

    Args:
        run_record: Dict with executor run fields. Recognised keys:
            ``run_id``, ``jira_key``, ``title``, ``status``, ``duration_ms``,
            ``cost_usd``, ``stderr``. Missing fields render as ``"?"`` rather
            than raising.

    Returns:
        A dict in Discord webhook shape: ``{"embeds": [...]}``.
    """
    run_id = str(run_record.get("run_id") or "").strip()
    jira_key = (run_record.get("jira_key") or "").strip()
    title = (run_record.get("title") or jira_key or run_id or "executor run").strip()
    status = (run_record.get("status") or "unknown").strip()
    is_success = status in _EXECUTOR_SUCCESS_STATUSES
    color = COLORS["success"] if is_success else COLORS["error"]

    # Title prefix — "[TK-463] Foo" reads naturally even when title already
    # starts with the key (we strip the dupe), and drops the prefix when no
    # key is available.
    header_title = title
    if jira_key and not title.startswith(f"[{jira_key}]"):
        if title == jira_key:
            header_title = jira_key
        else:
            header_title = f"[{jira_key}] {title}"

    # Build the run link. server_host + hardcoded idea-board port matches the
    # pattern used across bot_commands.py and the executor module.
    host = settings.server_host or "localhost"
    if run_id:
        link = f"http://{host}:{_IDEA_BOARD_PORT}/executor-runs#{run_id}"
    else:
        link = f"http://{host}:{_IDEA_BOARD_PORT}/executor-runs"

    description_parts: list[str] = [f"[View run]({link})"]
    if not is_success:
        tail = _stderr_tail(run_record.get("stderr"))
        if tail:
            description_parts.append(f"```\n{tail}\n```")

    embed: dict[str, Any] = {
        "title": header_title,
        "description": "\n".join(description_parts),
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "Executor"},
        "fields": [
            {"name": "Status", "value": status, "inline": True},
            {
                "name": "Duration",
                "value": _format_duration_ms(run_record.get("duration_ms")),
                "inline": True,
            },
            {
                "name": "Cost",
                "value": _format_cost_usd(run_record.get("cost_usd")),
                "inline": True,
            },
        ],
    }
    if jira_key:
        embed["fields"].append(
            {"name": "Jira", "value": jira_key, "inline": True}
        )
    return {"embeds": [embed]}


def send_executor_summary(
    run_record: dict[str, Any],
    webhook_url: str | None = None,
) -> str:
    """Post a per-run summary to the executor-summary Discord webhook.

    Safe to call from the executor's completion hook — every failure mode
    (missing webhook, no ``requests``, HTTP error) returns a descriptive
    string instead of raising, so an instrumentation hiccup never aborts
    the finalizer.

    Args:
        run_record: Completed run metadata. See
            :func:`build_executor_summary_payload` for recognised keys.
        webhook_url: Override for the resolved settings webhook. Useful for
            tests and ops-triggered resends.

    Returns:
        Short status string describing the outcome.
    """
    if requests is None:
        return "Error: requests library not installed"

    url = (webhook_url or _executor_summary_webhook()).strip()
    if not url:
        return "Error: No executor summary webhook configured"

    payload = build_executor_summary_payload(run_record)

    try:
        response = retry_request(
            requests.post,
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return f"Error sending executor summary: {exc}"

    if response.status_code in (200, 204):
        run_id = run_record.get("run_id") or run_record.get("jira_key") or "run"
        return f"Sent executor summary for {run_id}"
    return f"Discord error {response.status_code}: {response.text}"


def get_notification_tools() -> list[Any]:
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
