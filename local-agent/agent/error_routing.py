"""
Error Routing — Map error categories and severity levels to Discord webhook URLs.

Loads ``routing_config.json`` from the project root to decide which webhook an
alert should be sent to. Allows different environments (dev/prod) or different
stakeholder groups to receive alerts on different channels without code changes.

Resolution order for :func:`get_webhook_for_category`:

1. ``error_types[category]`` — exact error category name match (e.g. ``"rate_limited"``)
2. ``categories[severity]`` — severity level match (e.g. ``"critical"``)
3. ``default_webhook`` — config-file-wide default
4. ``settings.discord_alerts_webhook`` — .env fallback
5. Empty string — no destination configured

The config file is intentionally cheap to reload; it's re-read on every call
so operators can edit ``routing_config.json`` without restarting the bot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import settings

log = logging.getLogger(__name__)

CONFIG_PATH: Path = Path(__file__).parent.parent / "routing_config.json"


def load_routing_config() -> dict[str, Any]:
    """Load and return the routing config, or ``{}`` if missing/malformed."""
    path = CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Invalid routing config at %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _lookup(mapping: Any, key: Optional[str]) -> Optional[str]:
    """Return ``mapping[key]`` if it's a non-empty string, else None."""
    if not key or not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def get_webhook_for_category(
    category: Optional[str] = None,
    severity: Optional[str] = None,
) -> str:
    """Resolve the webhook URL for an error category / severity.

    Args:
        category: Specific error category name (e.g. ``"rate_limited"``).
        severity: Severity level (e.g. ``"critical"``, ``"warning"``).

    Returns:
        The webhook URL to send to. Empty string if nothing is configured.
    """
    config = load_routing_config()

    routed = _lookup(config.get("error_types"), category)
    if routed:
        return routed

    routed = _lookup(config.get("categories"), severity)
    if routed:
        return routed

    default = config.get("default_webhook")
    if isinstance(default, str) and default.strip():
        return default

    fallback = getattr(settings, "discord_alerts_webhook", "") or ""
    return fallback
