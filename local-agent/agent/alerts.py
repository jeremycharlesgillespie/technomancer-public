"""
Alerts Channel — Send infrastructure and error alerts to a dedicated
Discord channel via webhook, separate from the main chat.

Configure via DISCORD_ALERTS_WEBHOOK in .env. All alert functions are
non-blocking and fail silently so they never disrupt the main bot.
"""

import logging

log = logging.getLogger(__name__)


def send_alert(message: str, title: str = "", level: str = "info") -> None:
    """Send an alert to the dedicated alerts Discord channel via webhook.

    Args:
        message: Alert text
        title: Optional embed title
        level: Severity level — "info", "warning", "error", "critical"
    """
    try:
        from .config import settings
        webhook_url = settings.discord_alerts_webhook
    except Exception:
        webhook_url = ""
    if not webhook_url:
        log.debug("[Alerts] No DISCORD_ALERTS_WEBHOOK configured, skipping")
        return

    try:
        import requests
        from .discord_rate_limit import retry_request

        colors = {
            "info": 0x5865F2,
            "warning": 0xFEE75C,
            "error": 0xED4245,
            "critical": 0xED4245,
            "success": 0x57F287,
        }
        color = colors.get(level, colors["info"])

        if title:
            from datetime import datetime
            payload = {
                "embeds": [{
                    "title": title,
                    "description": message[:4000],
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": f"Technomancer | {level.upper()}"},
                }]
            }
        else:
            payload = {"content": message[:2000]}

        retry_request(
            requests.post,
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except Exception:
        log.debug("[Alerts] Failed to send alert", exc_info=True)
