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
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

log = logging.getLogger(__name__)

DEFAULT_PORT = 9090

# Latency buckets tuned for LLM calls: 0.5s to 120s
_LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

# Tighter buckets for knowledge_lookup: local cache hits are sub-millisecond,
# web tiers typically seconds.
_KNOWLEDGE_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

# ---------------------------------------------------------------------------
# Prometheus metric objects (created lazily to avoid import-time side effects
# when prometheus_client is not installed)
# ---------------------------------------------------------------------------

_llm_call_duration: Optional["Histogram"] = None
_llm_calls_total: Optional["Counter"] = None
_llm_call_errors_total: Optional["Counter"] = None
_llm_input_tokens_total: Optional["Counter"] = None
_llm_output_tokens_total: Optional["Counter"] = None
_knowledge_lookup_total: Optional["Counter"] = None
_knowledge_lookup_duration: Optional["Histogram"] = None
_ollama_model_loads_total: Optional["Counter"] = None
_ollama_model_unloads_total: Optional["Counter"] = None
_ollama_model_resident: Optional["Gauge"] = None

_initialized = False
_lock = threading.Lock()


def _ensure_metrics() -> bool:
    """Create Prometheus metric objects if not yet initialized.

    Returns True if metrics are available, False if prometheus_client
    is not installed.
    """
    global _llm_call_duration, _llm_calls_total, _llm_call_errors_total
    global _llm_input_tokens_total, _llm_output_tokens_total, _initialized
    global _knowledge_lookup_total, _knowledge_lookup_duration
    global _ollama_model_loads_total, _ollama_model_unloads_total, _ollama_model_resident

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
        _knowledge_lookup_total = Counter(
            "knowledge_lookup_total",
            "Total number of knowledge_lookup calls by result source",
            labelnames=["source"],
        )
        _knowledge_lookup_duration = Histogram(
            "knowledge_lookup_duration_seconds",
            "Latency of knowledge_lookup calls by winning tier",
            labelnames=["tier"],
            buckets=_KNOWLEDGE_LATENCY_BUCKETS,
        )
        _ollama_model_loads_total = Counter(
            "ollama_model_loads_total",
            "Total number of Ollama model loads",
            labelnames=["model"],
        )
        _ollama_model_unloads_total = Counter(
            "ollama_model_unloads_total",
            "Total number of Ollama model unloads",
            labelnames=["model"],
        )
        _ollama_model_resident = Gauge(
            "ollama_model_resident",
            "Ollama model residency status (1 = resident, 0 = not resident)",
            labelnames=["model"],
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


def record_knowledge_lookup(source: str, duration: float) -> None:
    """Record a single knowledge_lookup call by winning tier.

    ``source`` is one of: ``facts_db``, ``wikipedia``, ``web_search``,
    ``failure``. Safe to call when prometheus_client is not installed.
    """
    if not _ensure_metrics():
        return

    _knowledge_lookup_total.labels(source=source).inc()
    _knowledge_lookup_duration.labels(tier=source).observe(duration)


def record_ollama_model_load(model: str) -> None:
    """Record an Ollama model load event.

    This function should be called whenever a model is loaded into Ollama.
    Safe to call even if prometheus_client is not installed (silently no-ops).
    """
    if not _ensure_metrics():
        return

    _ollama_model_loads_total.labels(model=model).inc()


def record_ollama_model_unload(model: str) -> None:
    """Record an Ollama model unload event.

    This function should be called whenever a model is unloaded from Ollama.
    Safe to call even if prometheus_client is not installed (silently no-ops).
    """
    if not _ensure_metrics():
        return

    _ollama_model_unloads_total.labels(model=model).inc()


def set_ollama_model_resident(model: str, resident: bool) -> None:
    """Set the residency status of an Ollama model.

    This function should be called to update the gauge when a model's
    residency status changes.
    Safe to call even if prometheus_client is not installed (silently no-ops).
    """
    if not _ensure_metrics():
        return

    _ollama_model_resident.labels(model=model).set(1 if resident else 0)


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
