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
import time as _time
from typing import Any, Optional

from . import metrics
from .config import settings
from .error_routing import get_webhook_for_category

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLO violation de-duplication state
# ---------------------------------------------------------------------------
# _slo_last_fired maps SLO id -> monotonic timestamp of the last dispatch.
# While an SLO is breaching we keep its entry; once it recovers we drop it so
# the very next breach alerts immediately instead of waiting out the window.
SLO_DEDUP_WINDOW_SECONDS: float = 30 * 60.0
_slo_last_fired: dict[str, float] = {}


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


# dispatch_alert is the public name used by SLO wiring and other call sites
# that want "fire an alert" semantics rather than "send_alert" phrasing.
dispatch_alert = send_alert


# ---------------------------------------------------------------------------
# SLO violation checks
# ---------------------------------------------------------------------------


def _walk_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Return ``payload[path[0]][path[1]]...`` or ``None`` if any key is missing."""
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _is_breach(value: Any, comparator: str, threshold: float) -> bool:
    """Return True when ``value`` violates ``threshold`` under ``comparator``.

    Non-numeric and ``None`` values are treated as "no signal" and never breach
    — callers rely on this to quietly ignore metrics that aren't populated yet
    (e.g. ``oldest_top_ranked_wait_seconds`` when the queue is empty).
    """
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if comparator == "lt":
        return value < threshold
    if comparator == "gt":
        return value > threshold
    return False


def _format_slo_message(spec: dict[str, Any], value: Any) -> str:
    comparator = spec.get("comparator", "gt")
    threshold = spec.get("threshold")
    unit = spec.get("unit", "")
    label = spec.get("label", "SLO")
    direction = "below" if comparator == "lt" else "above"
    return f"{label} is {value}{unit}, {direction} threshold {threshold}{unit}."


def _reset_slo_state() -> None:
    """Drop all dedup state. Tests use this between cases."""
    _slo_last_fired.clear()


def check_slo_violations(
    snapshot: Optional[dict[str, Any]] = None,
    now: Optional[float] = None,
) -> list[str]:
    """Walk ``metrics.SLO_THRESHOLDS`` and dispatch an alert for each breach.

    Duplicate alerts for the same SLO are suppressed within
    :data:`SLO_DEDUP_WINDOW_SECONDS` (30 minutes by default). When an SLO
    recovers — i.e. its metric is no longer breaching — its dedup entry is
    cleared so the next breach alerts immediately.

    Args:
        snapshot: Optional pre-fetched metrics snapshot. Defaults to calling
            :func:`metrics.get_snapshot`, which is cached for 30 seconds.
        now: Optional monotonic timestamp override, mostly for tests.

    Returns:
        The SLO ids that actually dispatched an alert on this call. Ids that
        were suppressed by the dedup window are not included.
    """
    if snapshot is None:
        try:
            snapshot = metrics.get_snapshot()
        except Exception:
            log.debug("[Alerts] metrics.get_snapshot failed; skipping SLO check", exc_info=True)
            return []

    current = now if now is not None else _time.monotonic()
    fired: list[str] = []

    for slo_id, spec in metrics.SLO_THRESHOLDS.items():
        value = _walk_path(snapshot, spec["path"])
        if _is_breach(value, spec["comparator"], spec["threshold"]):
            last = _slo_last_fired.get(slo_id)
            if last is not None and (current - last) < SLO_DEDUP_WINDOW_SECONDS:
                log.debug("[Alerts] SLO %s still breaching, suppressed", slo_id)
                continue
            try:
                dispatch_alert(
                    message=_format_slo_message(spec, value),
                    title=f"SLO breach: {spec.get('label', slo_id)}",
                    level=spec.get("severity", "warning"),
                    category="slo",
                )
            except Exception:
                log.debug("[Alerts] dispatch_alert failed for %s", slo_id, exc_info=True)
            _slo_last_fired[slo_id] = current
            fired.append(slo_id)
        else:
            # Recovered (or never breached) — clear so next breach alerts.
            _slo_last_fired.pop(slo_id, None)

    return fired
