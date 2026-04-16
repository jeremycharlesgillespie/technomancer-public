"""
Alerts Channel — Send infrastructure and error alerts to a dedicated
Discord channel via webhook, separate from the main chat.

Webhook resolution order:
1. Explicit ``webhook_url`` argument (if provided)
2. Routing config (``routing_config.json``) matched by ``category`` / ``level``
3. ``DISCORD_ALERTS_WEBHOOK`` from .env (via settings)

All alert functions are non-blocking and fail silently so they never disrupt
the main bot.
"""

import logging
from typing import Optional

from .config import settings
from .error_routing import get_webhook_for_category

log = logging.getLogger(__name__)


def send_alert(
    message: str,
    title: str = "",
    level: str = "info",
    category: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> None:
    """Send an alert to the dedicated alerts Discord channel via webhook.

    Args:
        message: Alert text
        title: Optional embed title
        level: Severity level — "info", "warning", "error", "critical"
        category: Error category name for routing lookup (e.g. ``"rate_limited"``)
        webhook_url: Explicit webhook URL that bypasses routing entirely
    """
    resolved_url = webhook_url
    if resolved_url is None:
        try:
            resolved_url = get_webhook_for_category(category=category, severity=level)
        except Exception:
            log.debug("[Alerts] Routing lookup failed, falling back to settings", exc_info=True)
            resolved_url = settings.discord_alerts_webhook or ""

    if not resolved_url:
        log.debug("[Alerts] No webhook configured for category=%s level=%s, skipping",
                  category, level)
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
            resolved_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except Exception:
        log.debug("[Alerts] Failed to send alert", exc_info=True)
