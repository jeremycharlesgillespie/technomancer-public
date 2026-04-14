"""
API Usage Anomaly Detection — Monitor API key usage patterns and alert on anomalies.

Tracks per-endpoint request volume in rolling time windows, compares against
baselines computed from historical data in metrics_db, and fires Discord alerts
when usage exceeds thresholds or requests target unknown endpoints.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known / authorized endpoints
# ---------------------------------------------------------------------------
AUTHORIZED_ENDPOINTS: set[str] = {
    "ollama",
    "ollama_vision",
    "claude_api",
    "claude_cli",
}

# ---------------------------------------------------------------------------
# Defaults (overridable via config)
# ---------------------------------------------------------------------------
DEFAULT_SPIKE_MULTIPLIER = 2.0  # alert when usage >= 2x baseline
DEFAULT_WINDOW_SECONDS = 3600  # 1-hour rolling window
DEFAULT_BASELINE_HOURS = 24  # compute baseline from last 24h
DEFAULT_MIN_BASELINE_CALLS = 5  # need at least N calls to form a baseline
DEFAULT_COOLDOWN_SECONDS = 1800  # 30 min between repeat alerts per endpoint


@dataclass
class _WindowEntry:
    """A single API call timestamp for rate tracking."""

    timestamp: float  # time.monotonic()
    endpoint: str


@dataclass
class _AlertState:
    """Per-endpoint alert cooldown tracker."""

    last_spike_alert: float = 0.0
    last_unknown_alert: float = 0.0


class UsageAnomalyDetector:
    """Detects anomalous API usage patterns and sends alerts.

    Integrates with PerfMonitor — call ``check()`` after every recorded API call.
    """

    def __init__(
        self,
        spike_multiplier: float = DEFAULT_SPIKE_MULTIPLIER,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        baseline_hours: int = DEFAULT_BASELINE_HOURS,
        min_baseline_calls: int = DEFAULT_MIN_BASELINE_CALLS,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        authorized_endpoints: set[str] | None = None,
    ) -> None:
        self._spike_multiplier = spike_multiplier
        self._window_seconds = window_seconds
        self._baseline_hours = baseline_hours
        self._min_baseline_calls = min_baseline_calls
        self._cooldown_seconds = cooldown_seconds
        self._authorized_endpoints = (
            authorized_endpoints if authorized_endpoints is not None else AUTHORIZED_ENDPOINTS.copy()
        )

        self._window: list[_WindowEntry] = []
        self._alert_states: dict[str, _AlertState] = {}
        self._lock = threading.Lock()
        self._baselines: dict[str, float] = {}  # endpoint -> calls/hour baseline
        self._baseline_updated: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, endpoint: str) -> list[str]:
        """Check a new API call for anomalies. Returns list of alert messages (may be empty)."""
        now = time.monotonic()
        alerts: list[str] = []

        # Refresh baselines OUTSIDE the lock (it acquires the lock internally)
        self._maybe_refresh_baselines()

        with self._lock:
            # Record the call in the sliding window
            self._window.append(_WindowEntry(timestamp=now, endpoint=endpoint))
            self._trim_window(now)

            # Check for unknown endpoint
            unknown_alert = self._check_unknown_endpoint(endpoint, now)
            if unknown_alert:
                alerts.append(unknown_alert)

            # Check for usage spike (baselines already refreshed)
            spike_alert = self._check_spike(endpoint, now)
            if spike_alert:
                alerts.append(spike_alert)

        # Fire alerts outside lock
        for msg in alerts:
            self._send_alert(msg, endpoint)

        return alerts

    def get_window_counts(self) -> dict[str, int]:
        """Return current call counts per endpoint in the active window."""
        now = time.monotonic()
        with self._lock:
            self._trim_window(now)
            counts: dict[str, int] = {}
            for entry in self._window:
                counts[entry.endpoint] = counts.get(entry.endpoint, 0) + 1
            return counts

    def get_baselines(self) -> dict[str, float]:
        """Return current baseline rates (calls/hour per endpoint)."""
        self._maybe_refresh_baselines()
        with self._lock:
            return dict(self._baselines)

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of detector state for diagnostics."""
        now = time.monotonic()
        with self._lock:
            self._trim_window(now)
            window_counts = {}
            for entry in self._window:
                window_counts[entry.endpoint] = window_counts.get(entry.endpoint, 0) + 1

        baselines = self.get_baselines()
        return {
            "window_seconds": self._window_seconds,
            "spike_multiplier": self._spike_multiplier,
            "authorized_endpoints": sorted(self._authorized_endpoints),
            "baselines": baselines,
            "current_window_counts": window_counts,
            "cooldown_seconds": self._cooldown_seconds,
        }

    def add_authorized_endpoint(self, endpoint: str) -> None:
        """Add an endpoint to the authorized set."""
        with self._lock:
            self._authorized_endpoints.add(endpoint)

    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._window.clear()
            self._alert_states.clear()
            self._baselines.clear()
            self._baseline_updated = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_window(self, now: float) -> None:
        """Remove entries older than the window. Caller must hold _lock."""
        cutoff = now - self._window_seconds
        self._window = [e for e in self._window if e.timestamp >= cutoff]

    def _check_unknown_endpoint(self, endpoint: str, now: float) -> str | None:
        """Alert if the endpoint is not in the authorized set."""
        if endpoint in self._authorized_endpoints:
            return None

        state = self._alert_states.setdefault(endpoint, _AlertState())
        if now - state.last_unknown_alert < self._cooldown_seconds:
            return None

        state.last_unknown_alert = now
        return (
            f"UNKNOWN ENDPOINT detected: `{endpoint}` is not in the authorized "
            f"endpoint list ({', '.join(sorted(self._authorized_endpoints))}). "
            f"This may indicate unauthorized API access."
        )

    def _check_spike(self, endpoint: str, now: float) -> str | None:
        """Alert if the endpoint's call rate exceeds the baseline multiplier.

        Caller must hold _lock. Baselines must be refreshed beforehand.
        """
        baseline = self._baselines.get(endpoint)
        if baseline is None:
            return None  # no baseline yet — can't detect spike

        # Count calls for this endpoint in the current window
        window_count = sum(1 for e in self._window if e.endpoint == endpoint)
        # Scale baseline to the window duration
        window_hours = self._window_seconds / 3600
        threshold = baseline * window_hours * self._spike_multiplier

        if threshold < 1:
            return None  # baseline too low to be meaningful

        if window_count < threshold:
            return None

        state = self._alert_states.setdefault(endpoint, _AlertState())
        if now - state.last_spike_alert < self._cooldown_seconds:
            return None

        state.last_spike_alert = now
        return (
            f"USAGE SPIKE on `{endpoint}`: {window_count} calls in the last "
            f"{self._window_seconds // 60}min (baseline: ~{baseline:.1f} calls/hr, "
            f"threshold: {threshold:.0f} calls/{self._window_seconds // 60}min). "
            f"This is {window_count / max(baseline * window_hours, 0.01):.1f}x the baseline."
        )

    def _maybe_refresh_baselines(self) -> None:
        """Refresh baselines from metrics_db if stale (older than 1 hour)."""
        now = time.monotonic()
        if now - self._baseline_updated < 3600:
            return

        try:
            from . import metrics_db

            metrics_db.init_db()
            since = (datetime.now() - timedelta(hours=self._baseline_hours)).isoformat()
            rows = metrics_db._query_rows(
                """SELECT endpoint, COUNT(*) AS cnt
                   FROM llm_calls
                   WHERE timestamp >= ?
                   GROUP BY endpoint""",
                (since,),
            )

            with self._lock:
                self._baselines.clear()
                for row in rows:
                    cnt = row["cnt"]
                    if cnt >= self._min_baseline_calls:
                        # Convert to calls/hour
                        self._baselines[row["endpoint"]] = cnt / self._baseline_hours
                self._baseline_updated = now

            log.debug("Anomaly baselines refreshed: %s", self._baselines)
        except Exception:
            log.exception("Failed to refresh anomaly baselines from metrics_db")

    def _send_alert(self, message: str, endpoint: str) -> None:
        """Send an anomaly alert to the dedicated alerts channel."""
        try:
            from .alerts import send_alert

            send_alert(message, title="API Usage Anomaly", level="warning")
            log.warning("API anomaly alert: %s", message)
        except Exception:
            log.exception("Failed to send anomaly alert for %s", endpoint)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_detector: UsageAnomalyDetector | None = None


def get_detector() -> UsageAnomalyDetector:
    """Get or create the global anomaly detector, using config settings."""
    global _detector
    if _detector is None:
        try:
            from .config import settings

            _detector = UsageAnomalyDetector(
                spike_multiplier=settings.anomaly_spike_multiplier,
                window_seconds=settings.anomaly_window_seconds,
                baseline_hours=settings.anomaly_baseline_hours,
                cooldown_seconds=settings.anomaly_cooldown_seconds,
            )
        except Exception:
            _detector = UsageAnomalyDetector()
    return _detector


def check_usage(endpoint: str) -> list[str]:
    """Convenience: check an API call against the global detector."""
    return get_detector().check(endpoint)


def get_usage_anomaly_status() -> str:
    """Human-readable status for the Discord command."""
    det = get_detector()
    status = det.get_status()

    lines = ["**API Usage Anomaly Detection**", ""]
    lines.append(f"Window: {status['window_seconds'] // 60} min")
    lines.append(f"Spike threshold: {status['spike_multiplier']}x baseline")
    lines.append(f"Alert cooldown: {status['cooldown_seconds'] // 60} min")
    lines.append(f"Authorized endpoints: {', '.join(status['authorized_endpoints'])}")
    lines.append("")

    baselines = status["baselines"]
    if baselines:
        lines.append("**Baselines** (calls/hour):")
        for ep, rate in sorted(baselines.items()):
            lines.append(f"  {ep}: {rate:.1f}")
    else:
        lines.append("No baselines computed yet (need historical data).")
    lines.append("")

    counts = status["current_window_counts"]
    if counts:
        lines.append("**Current window:**")
        for ep, cnt in sorted(counts.items()):
            lines.append(f"  {ep}: {cnt} calls")
    else:
        lines.append("No calls in current window.")

    return "\n".join(lines)


def get_anomaly_tools() -> list:
    """Get anomaly detection tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="api_usage_status",
            description="Show API usage anomaly detection status, baselines, and current window counts",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: get_usage_anomaly_status(),
        ),
    ]
