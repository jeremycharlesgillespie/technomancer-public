"""
Performance Monitor — Track latency, token counts, and success/failure rates
for all LLM API calls (Ollama and Claude).

Provides a lightweight in-memory metrics store with per-endpoint aggregation.
Data is available via the `perf` Discord command and fed into the idea generator.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator


@dataclass
class CallRecord:
    """A single LLM API call record."""

    timestamp: str
    endpoint: str  # "ollama", "claude_api", "claude_cli"
    duration: float  # seconds
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str = ""


class PerfMonitor:
    """Tracks LLM call performance metrics across all endpoints.

    Thread-safe singleton that accumulates call records in memory.
    Old records are trimmed when the buffer exceeds max_records.
    """

    def __init__(self, max_records: int = 1000) -> None:
        """Initialize the performance monitor with a max record buffer size."""
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._max_records = max_records

    def record(
        self,
        endpoint: str,
        duration: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        error: str = "",
    ) -> None:
        """Record a single LLM call."""
        rec = CallRecord(
            timestamp=datetime.now().isoformat(),
            endpoint=endpoint,
            duration=duration,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            error=error if not success else "",
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]

        # Forward to Prometheus metrics (no-ops if prometheus_client not installed)
        from .prometheus_metrics import record_prometheus

        record_prometheus(
            endpoint=endpoint,
            duration=duration,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

        # Forward to SQLite for persistent trend analysis
        from . import metrics_db

        metrics_db.record(
            endpoint=endpoint,
            duration=duration,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            error=error if not success else "",
        )

        # Check for usage anomalies (rate spikes, unknown endpoints)
        from .api_usage_anomaly import check_usage

        check_usage(endpoint)

    @contextmanager
    def track(
        self,
        endpoint: str,
        model: str = "",
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager that times a call and records it automatically.

        Usage:
            with monitor.track("ollama", model="qwen3.5:27b") as ctx:
                response = ollama_client.chat(...)
                ctx["input_tokens"] = 100
                ctx["output_tokens"] = 50

        On exception, records as failure with the error message.
        """
        ctx: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
        start = time.perf_counter()
        try:
            yield ctx
            duration = time.perf_counter() - start
            self.record(
                endpoint=endpoint,
                duration=duration,
                success=True,
                input_tokens=ctx.get("input_tokens", 0),
                output_tokens=ctx.get("output_tokens", 0),
                model=model,
            )
        except Exception as e:
            duration = time.perf_counter() - start
            self.record(
                endpoint=endpoint,
                duration=duration,
                success=False,
                model=model,
                error=str(e)[:200],
            )
            raise

    def get_endpoint_stats(self, endpoint: str | None = None) -> dict[str, Any]:
        """Get aggregated stats, optionally filtered by endpoint."""
        with self._lock:
            records = list(self._records)

        if endpoint:
            records = [r for r in records if r.endpoint == endpoint]

        if not records:
            return {"calls": 0}

        durations = [r.duration for r in records]
        successes = sum(1 for r in records if r.success)
        failures = len(records) - successes
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        sorted_dur = sorted(durations)

        return {
            "calls": len(records),
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / len(records) * 100, 1),
            "avg_latency": round(sum(durations) / len(durations), 2),
            "min_latency": round(min(durations), 2),
            "max_latency": round(max(durations), 2),
            "p50_latency": round(sorted_dur[len(sorted_dur) // 2], 2),
            "p95_latency": round(
                sorted_dur[int(len(sorted_dur) * 0.95)]
                if len(sorted_dur) >= 20
                else max(durations),
                2,
            ),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

    def get_recent_errors(self, n: int = 5) -> list[dict[str, str]]:
        """Get the N most recent errors across all endpoints."""
        with self._lock:
            errors = [r for r in self._records if not r.success]
        return [
            {
                "timestamp": r.timestamp,
                "endpoint": r.endpoint,
                "model": r.model,
                "error": r.error,
                "duration": f"{r.duration:.2f}s",
            }
            for r in errors[-n:]
        ]

    def get_summary(self) -> str:
        """Human-readable performance summary for all endpoints."""
        with self._lock:
            all_records = list(self._records)

        if not all_records:
            return "No LLM call data recorded yet."

        endpoints = sorted(set(r.endpoint for r in all_records))
        lines = [
            f"**LLM Endpoint Metrics** ({len(all_records)} total calls)",
            "",
        ]

        for ep in endpoints:
            stats = self.get_endpoint_stats(ep)
            lines.append(f"**{ep}** — {stats['calls']} calls")
            lines.append(
                f"  Success rate: {stats['success_rate']}% "
                f"({stats['successes']} ok, {stats['failures']} failed)"
            )
            lines.append(
                f"  Latency: avg {stats['avg_latency']}s, "
                f"p50 {stats['p50_latency']}s, "
                f"p95 {stats['p95_latency']}s, "
                f"max {stats['max_latency']}s"
            )
            if stats["total_input_tokens"] or stats["total_output_tokens"]:
                lines.append(
                    f"  Tokens: {stats['total_input_tokens']:,} in, "
                    f"{stats['total_output_tokens']:,} out"
                )
            lines.append("")

        # Recent errors
        errors = self.get_recent_errors(3)
        if errors:
            lines.append("**Recent errors:**")
            for err in errors:
                lines.append(f"- [{err['endpoint']}] {err['error'][:80]}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_monitor = PerfMonitor()


def get_monitor() -> PerfMonitor:
    """Get the global PerfMonitor instance."""
    return _monitor


def record_llm_call(
    endpoint: str,
    duration: float,
    success: bool,
    **kwargs: Any,
) -> None:
    """Convenience: record a single LLM call to the global monitor."""
    _monitor.record(endpoint, duration, success, **kwargs)


def get_endpoint_summary() -> str:
    """Convenience: get the global endpoint performance summary."""
    return _monitor.get_summary()
