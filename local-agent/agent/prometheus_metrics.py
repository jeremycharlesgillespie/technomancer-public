"""
Prometheus metrics for LLM call performance monitoring.

Exposes latency histograms, token counters, and call counters for
Ollama and Claude endpoints via an HTTP endpoint (default port 9090).

Usage:
    from .prometheus_metrics import start_metrics_server

    start_metrics_server(port=9090)  # Call once at bot startup

Metrics are updated automatically by PerfMonitor when record() is called.
Scrape at http://localhost:9090/metrics for Prometheus-compatible output.
"""

import logging
import threading
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

log = logging.getLogger(__name__)

DEFAULT_PORT = 9090

# Latency buckets tuned for LLM calls: 0.5s to 120s
_LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

# ---------------------------------------------------------------------------
# Prometheus metric objects (created lazily to avoid import-time side effects
# when prometheus_client is not installed)
# ---------------------------------------------------------------------------

_llm_call_duration: Optional["Histogram"] = None
_llm_calls_total: Optional["Counter"] = None
_llm_call_errors_total: Optional["Counter"] = None
_llm_input_tokens_total: Optional["Counter"] = None
_llm_output_tokens_total: Optional["Counter"] = None

_initialized = False
_lock = threading.Lock()


def _ensure_metrics() -> bool:
    """Create Prometheus metric objects if not yet initialized.

    Returns True if metrics are available, False if prometheus_client
    is not installed.
    """
    global _llm_call_duration, _llm_calls_total, _llm_call_errors_total
    global _llm_input_tokens_total, _llm_output_tokens_total, _initialized

    if _initialized:
        return HAS_PROMETHEUS

    with _lock:
        if _initialized:
            return HAS_PROMETHEUS

        if not HAS_PROMETHEUS:
            _initialized = True
            return False

        _llm_call_duration = Histogram(
            "llm_call_duration_seconds",
            "Latency of LLM API calls in seconds",
            labelnames=["endpoint", "model"],
            buckets=_LATENCY_BUCKETS,
        )
        _llm_calls_total = Counter(
            "llm_calls_total",
            "Total number of LLM API calls",
            labelnames=["endpoint", "model", "status"],
        )
        _llm_call_errors_total = Counter(
            "llm_call_errors_total",
            "Total number of failed LLM API calls",
            labelnames=["endpoint", "model"],
        )
        _llm_input_tokens_total = Counter(
            "llm_input_tokens_total",
            "Total input tokens sent to LLM endpoints",
            labelnames=["endpoint", "model"],
        )
        _llm_output_tokens_total = Counter(
            "llm_output_tokens_total",
            "Total output tokens received from LLM endpoints",
            labelnames=["endpoint", "model"],
        )

        _initialized = True
        return True


def record_prometheus(
    endpoint: str,
    duration: float,
    success: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
) -> None:
    """Record a single LLM call to Prometheus metrics.

    Called automatically by PerfMonitor.record(). Safe to call even if
    prometheus_client is not installed (silently no-ops).
    """
    if not _ensure_metrics():
        return

    status = "success" if success else "error"
    _llm_call_duration.labels(endpoint=endpoint, model=model).observe(duration)
    _llm_calls_total.labels(endpoint=endpoint, model=model, status=status).inc()

    if not success:
        _llm_call_errors_total.labels(endpoint=endpoint, model=model).inc()

    if input_tokens > 0:
        _llm_input_tokens_total.labels(endpoint=endpoint, model=model).inc(input_tokens)
    if output_tokens > 0:
        _llm_output_tokens_total.labels(endpoint=endpoint, model=model).inc(output_tokens)


def start_metrics_server(port: int = DEFAULT_PORT) -> bool:
    """Start the Prometheus HTTP metrics server on the given port.

    Returns True if the server started successfully, False otherwise.
    Safe to call even if prometheus_client is not installed.
    """
    if not _ensure_metrics():
        log.warning("prometheus_client not installed - metrics endpoint disabled")
        return False

    try:
        start_http_server(port)
        log.info("Prometheus metrics server started on port %d", port)
        return True
    except OSError as e:
        log.warning("Could not start Prometheus metrics server on port %d: %s", port, e)
        return False
