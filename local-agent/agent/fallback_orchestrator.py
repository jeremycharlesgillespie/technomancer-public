"""
Fallback Orchestrator — Local-first fallback when Claude API is unavailable.

Monitors Claude API health (consecutive errors, latency) and automatically
routes requests to local Ollama models when the API is degraded.  Recovers
automatically after a cooldown period by retrying Claude.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    """Record of a fallback activation or recovery."""

    timestamp: float
    event_type: str  # "activated", "recovered", "probe_failed"
    reason: str
    endpoint: str = "claude_api"


class FallbackOrchestrator:
    """Monitors Claude API health and orchestrates local-first fallback.

    Usage::

        orch = get_orchestrator()

        if orch.should_use_fallback():
            # route to Ollama instead of Claude
            response = ollama_agent.run(prompt)
        else:
            response = claude_api_call(prompt)

        # After each Claude call, report success/failure:
        orch.record_claude_result(success=True, latency=1.2)
    """

    def __init__(
        self,
        max_consecutive_errors: int = 5,
        latency_threshold: float = 30.0,
        latency_window: int = 5,
        recovery_cooldown: int = 300,
    ) -> None:
        """
        Args:
            max_consecutive_errors: Errors in a row before fallback activates.
            latency_threshold: Avg latency (seconds) that triggers fallback.
            latency_window: Number of recent calls to average for latency check.
            recovery_cooldown: Seconds to wait before probing Claude again.
        """
        self._max_consecutive_errors = max_consecutive_errors
        self._latency_threshold = latency_threshold
        self._latency_window = latency_window
        self._recovery_cooldown = recovery_cooldown

        self._lock = threading.Lock()
        self._consecutive_errors: int = 0
        self._recent_latencies: list[float] = []
        self._fallback_active: bool = False
        self._fallback_activated_at: float = 0.0
        self._last_error: str = ""
        self._events: list[FallbackEvent] = []
        self._total_fallback_activations: int = 0
        self._total_fallback_requests: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_use_fallback(self) -> bool:
        """Return True if requests should be routed to Ollama instead of Claude.

        When fallback is active, periodically allows a probe request through
        to test whether Claude has recovered.
        """
        with self._lock:
            if not self._fallback_active:
                return False

            # Check if recovery cooldown has elapsed — allow a probe
            elapsed = time.monotonic() - self._fallback_activated_at
            if elapsed >= self._recovery_cooldown:
                log.info(
                    "Fallback cooldown elapsed (%.0fs), allowing probe request to Claude",
                    elapsed,
                )
                return False  # Let the request go to Claude as a probe

            self._total_fallback_requests += 1
            return True

    def record_claude_result(
        self,
        success: bool,
        latency: float = 0.0,
        error: str = "",
    ) -> None:
        """Report the outcome of a Claude API call.

        Call this after every Claude API attempt (including probes during fallback).
        """
        with self._lock:
            if success:
                self._consecutive_errors = 0
                self._recent_latencies.append(latency)
                if len(self._recent_latencies) > self._latency_window:
                    self._recent_latencies = self._recent_latencies[-self._latency_window :]

                if self._fallback_active:
                    self._recover()
            else:
                self._consecutive_errors += 1
                self._last_error = error[:200]
                self._recent_latencies.clear()

                if not self._fallback_active:
                    self._check_activate_on_errors()

            # Check latency threshold (only when not already in fallback)
            if success and not self._fallback_active:
                self._check_activate_on_latency()

    def get_status(self) -> dict[str, Any]:
        """Return current orchestrator state for diagnostics."""
        with self._lock:
            avg_latency = (
                sum(self._recent_latencies) / len(self._recent_latencies)
                if self._recent_latencies
                else 0.0
            )
            return {
                "fallback_active": self._fallback_active,
                "consecutive_errors": self._consecutive_errors,
                "max_consecutive_errors": self._max_consecutive_errors,
                "avg_recent_latency": round(avg_latency, 2),
                "latency_threshold": self._latency_threshold,
                "recovery_cooldown": self._recovery_cooldown,
                "last_error": self._last_error,
                "total_activations": self._total_fallback_activations,
                "total_fallback_requests": self._total_fallback_requests,
                "recent_events": [
                    {
                        "type": e.event_type,
                        "reason": e.reason,
                        "age_seconds": round(time.monotonic() - e.timestamp),
                    }
                    for e in self._events[-5:]
                ],
            }

    def force_fallback(self, reason: str = "manual") -> None:
        """Manually activate fallback mode."""
        with self._lock:
            if not self._fallback_active:
                self._activate(reason)

    def force_recover(self) -> None:
        """Manually deactivate fallback mode."""
        with self._lock:
            if self._fallback_active:
                self._recover()

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._consecutive_errors = 0
            self._recent_latencies.clear()
            self._fallback_active = False
            self._fallback_activated_at = 0.0
            self._last_error = ""
            self._events.clear()
            self._total_fallback_activations = 0
            self._total_fallback_requests = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_activate_on_errors(self) -> None:
        """Activate fallback if consecutive errors exceed threshold. Caller holds lock."""
        if self._consecutive_errors >= self._max_consecutive_errors:
            self._activate(
                f"{self._consecutive_errors} consecutive errors (last: {self._last_error})"
            )

    def _check_activate_on_latency(self) -> None:
        """Activate fallback if average latency exceeds threshold. Caller holds lock."""
        if len(self._recent_latencies) < self._latency_window:
            return
        avg = sum(self._recent_latencies) / len(self._recent_latencies)
        if avg >= self._latency_threshold:
            self._activate(f"avg latency {avg:.1f}s >= {self._latency_threshold}s threshold")

    def _activate(self, reason: str) -> None:
        """Switch to fallback mode. Caller holds lock."""
        self._fallback_active = True
        self._fallback_activated_at = time.monotonic()
        self._total_fallback_activations += 1

        event = FallbackEvent(
            timestamp=time.monotonic(),
            event_type="activated",
            reason=reason,
        )
        self._events.append(event)
        if len(self._events) > 50:
            self._events = self._events[-50:]

        log.warning("Fallback ACTIVATED: %s", reason)
        self._send_alert(f"Claude API fallback activated: {reason}")

    def _recover(self) -> None:
        """Return to normal Claude API routing. Caller holds lock."""
        self._fallback_active = False
        self._consecutive_errors = 0

        event = FallbackEvent(
            timestamp=time.monotonic(),
            event_type="recovered",
            reason="Claude API probe succeeded",
        )
        self._events.append(event)
        if len(self._events) > 50:
            self._events = self._events[-50:]

        log.info("Fallback RECOVERED: Claude API is available again")
        self._send_alert_recovery()

    def _send_alert(self, message: str) -> None:
        """Send fallback activation alert to the dedicated alerts channel."""
        try:
            from .alerts import send_alert

            send_alert(message, level="warning", title="API Fallback")
        except Exception:
            log.exception("Failed to send fallback alert")

    def _send_alert_recovery(self) -> None:
        """Send recovery notification to the dedicated alerts channel."""
        try:
            from .alerts import send_alert

            send_alert(
                "Claude API is available again. Resuming normal operation.",
                level="success",
                title="API Fallback Recovered",
            )
        except Exception:
            log.exception("Failed to send recovery alert")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_orchestrator: FallbackOrchestrator | None = None


def get_orchestrator() -> FallbackOrchestrator:
    """Get or create the global fallback orchestrator, using config settings."""
    global _orchestrator
    if _orchestrator is None:
        try:
            from .config import settings

            _orchestrator = FallbackOrchestrator(
                max_consecutive_errors=settings.fallback_max_errors,
                latency_threshold=settings.fallback_latency_threshold,
                recovery_cooldown=settings.fallback_recovery_cooldown,
            )
        except Exception:
            _orchestrator = FallbackOrchestrator()
    return _orchestrator


def should_use_fallback() -> bool:
    """Convenience: check if fallback is active."""
    return get_orchestrator().should_use_fallback()


def record_claude_result(success: bool, latency: float = 0.0, error: str = "") -> None:
    """Convenience: report a Claude API call result."""
    get_orchestrator().record_claude_result(success=success, latency=latency, error=error)


def get_fallback_status() -> str:
    """Human-readable fallback status for Discord."""
    orch = get_orchestrator()
    s = orch.get_status()

    lines = ["**Claude API Fallback Orchestrator**", ""]
    status_icon = "ACTIVE (using Ollama)" if s["fallback_active"] else "Normal (using Claude)"
    lines.append(f"Status: **{status_icon}**")
    lines.append(f"Consecutive errors: {s['consecutive_errors']}/{s['max_consecutive_errors']}")
    lines.append(f"Avg recent latency: {s['avg_recent_latency']}s (threshold: {s['latency_threshold']}s)")
    lines.append(f"Recovery cooldown: {s['recovery_cooldown']}s")
    lines.append(f"Total activations: {s['total_activations']}")
    lines.append(f"Total fallback requests: {s['total_fallback_requests']}")

    if s["last_error"]:
        lines.append(f"Last error: {s['last_error'][:100]}")

    events = s["recent_events"]
    if events:
        lines.append("")
        lines.append("**Recent events:**")
        for ev in events:
            lines.append(f"  [{ev['type']}] {ev['reason']} ({ev['age_seconds']}s ago)")

    return "\n".join(lines)


def get_fallback_tools() -> list:
    """Get fallback orchestrator tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="fallback_status",
            description="Show Claude API fallback orchestrator status and health",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: get_fallback_status(),
        ),
    ]
