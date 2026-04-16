"""
Ollama Call Resilience — retries, exponential backoff, and health monitoring.

Wraps Ollama chat calls so the bot self-heals when the server restarts,
reloads a GPU model, or drops a connection mid-request. Three layers:

1. ``ollama_call_with_retries`` — exponential backoff + jitter on transient
   errors (connection refused, 5xx, read timeouts). Fails fast on permanent
   errors (4xx, unknown model).
2. ``OllamaHealthMonitor`` — background thread that polls ``/api/tags`` every
   ``ollama_health_check_interval`` seconds. Flips state to ``degraded`` or
   ``down`` before user calls start failing; flips back to ``healthy`` after
   a successful poll.
3. ``is_transient_error`` — single source of truth for what qualifies as a
   retryable failure so retry/health logic stays in sync.

The module does not auto-start the monitor on import — ``start_monitor()`` is
called explicitly from ``discord_memory_bot.py`` so tests can skip the thread.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests
from ollama._types import ResponseError

from .config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"
STATUS_UNKNOWN = "unknown"


# HTTP status codes that should be retried. 408 (timeout), 425 (too early),
# 429 (rate limit), 500/502/503/504 (server-side transient).
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


def is_transient_error(exc: BaseException) -> bool:
    """Return True when ``exc`` represents a retryable Ollama failure.

    Transient = worth retrying after a short backoff. Includes:
    - Connection errors (server restarting, GPU loading a model)
    - Read timeouts (slow response, not a hang because the client has a
      finite timeout from ``ollama_request_timeout``)
    - Ollama ``ResponseError`` with a 5xx or known-transient 4xx status

    Permanent errors (unknown model, invalid prompt, auth) are NOT retried
    because they will never succeed.
    """
    # requests / urllib3 network-layer errors
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True

    # httpx errors — ollama's client uses httpx; import lazily so we don't
    # force httpx as a hard dep at import time.
    try:
        import httpx

        if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout,
                            httpx.RemoteProtocolError, httpx.ConnectTimeout)):
            return True
    except ImportError:
        pass

    # Ollama-specific errors carry a status code
    if isinstance(exc, ResponseError):
        status = getattr(exc, "status_code", -1)
        if status in RETRYABLE_STATUS_CODES:
            return True
        return False

    # Generic socket/OS-level refusal
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionResetError, TimeoutError)):
        return True

    return False


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def _compute_backoff(attempt: int, base_delay: float, max_delay: float,
                     jitter_factor: float = 0.5) -> float:
    """Exponential backoff with uniform jitter.

    attempt is 0-indexed: 0 → base, 1 → 2*base, 2 → 4*base, ... capped at
    ``max_delay``. Jitter adds up to ``jitter_factor * delay`` so that many
    retrying clients do not synchronize into a thundering herd.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * jitter_factor)
    return delay + jitter


def ollama_call_with_retries(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    on_transient_error: Callable[[BaseException, int], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with retry/backoff on transient failures.

    Uses settings defaults (``ollama_max_retries``, ``ollama_retry_base_delay``,
    ``ollama_retry_max_delay``) when the corresponding argument is None, so
    callers can override per-call without knowing the config field names.

    The health monitor is updated so failures mark the bot as degraded before
    the retry budget is exhausted — users see the status flip in /health even
    when individual calls still succeed.
    """
    retries = settings.ollama_max_retries if max_retries is None else max_retries
    base = settings.ollama_retry_base_delay if base_delay is None else base_delay
    cap = settings.ollama_retry_max_delay if max_delay is None else max_delay

    last_exc: BaseException | None = None
    # attempts run 0..retries inclusive: retries=3 means 4 total tries.
    for attempt in range(retries + 1):
        try:
            result = fn(*args, **kwargs)
            # Success: if we had previously marked degraded, recover.
            if attempt > 0:
                log.info("Ollama call recovered after %d retry(s)", attempt)
                _monitor.mark_healthy()
            return result
        except Exception as exc:
            last_exc = exc
            if not is_transient_error(exc):
                # Permanent error — don't retry.
                log.debug("Ollama call failed with non-transient error: %s", exc)
                raise

            _monitor.mark_degraded(f"{type(exc).__name__}: {exc}")
            if on_transient_error is not None:
                try:
                    on_transient_error(exc, attempt)
                except Exception:
                    pass  # Hook must never break the retry loop.

            if attempt >= retries:
                log.warning("Ollama call failed after %d attempt(s): %s", attempt + 1, exc)
                _monitor.mark_down(f"{type(exc).__name__}: {exc}")
                raise

            delay = _compute_backoff(attempt, base, cap)
            log.info(
                "Ollama transient error (attempt %d/%d): %s — retrying in %.2fs",
                attempt + 1, retries + 1, exc, delay,
            )
            time.sleep(delay)

    # Unreachable in normal flow — the loop either returns or raises.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------


@dataclass
class _HealthState:
    """Thread-safe snapshot of Ollama's observed health."""

    status: str = STATUS_UNKNOWN
    last_checked: float = 0.0
    last_success: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0
    # Guards every field above so reads/writes are atomic across threads.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        """Return a dict copy (safe to hand to JSON endpoints)."""
        with self._lock:
            return {
                "status": self.status,
                "last_checked": self.last_checked,
                "last_success": self.last_success,
                "last_error": self.last_error,
                "consecutive_failures": self.consecutive_failures,
            }


class OllamaHealthMonitor:
    """Periodic health-check thread for Ollama's ``/api/tags`` endpoint.

    The monitor is a singleton started by ``start_monitor()``. Tests should
    not start it — they interact via ``mark_*`` / ``get_status`` directly.
    """

    def __init__(self) -> None:
        self._state = _HealthState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread if it isn't already running."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ollama-health-monitor",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Ollama health monitor started (interval=%ds, host=%s)",
            settings.ollama_health_check_interval,
            settings.ollama_host,
        )

    def stop(self) -> None:
        """Signal the background thread to stop (does not block)."""
        self._stop_event.set()

    # -- state API --------------------------------------------------------

    def get_status(self) -> str:
        """Return the current status string (healthy/degraded/down/unknown)."""
        with self._state._lock:
            return self._state.status

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot dict of the full health state."""
        return self._state.snapshot()

    def is_healthy(self) -> bool:
        """Convenience: True when the last observation was healthy."""
        return self.get_status() == STATUS_HEALTHY

    def mark_healthy(self) -> None:
        """Record a successful Ollama interaction."""
        with self._state._lock:
            prior = self._state.status
            self._state.status = STATUS_HEALTHY
            self._state.last_success = time.time()
            self._state.consecutive_failures = 0
            self._state.last_error = ""
        if prior != STATUS_HEALTHY:
            log.info("Ollama health: %s -> healthy", prior)

    def mark_degraded(self, reason: str) -> None:
        """Record a transient failure; marks degraded unless already down."""
        with self._state._lock:
            if self._state.status != STATUS_DOWN:
                self._state.status = STATUS_DEGRADED
            self._state.last_error = reason
            self._state.consecutive_failures += 1
            current = self._state.status
        log.debug("Ollama health: %s (%s)", current, reason)

    def mark_down(self, reason: str) -> None:
        """Record an exhausted retry budget or repeated failures."""
        with self._state._lock:
            prior = self._state.status
            self._state.status = STATUS_DOWN
            self._state.last_error = reason
            self._state.consecutive_failures += 1
        if prior != STATUS_DOWN:
            log.warning("Ollama health: %s -> down (%s)", prior, reason)

    # -- polling loop -----------------------------------------------------

    def _run_loop(self) -> None:
        """Poll Ollama until stopped. Uses stop_event.wait for sleep."""
        # Do an immediate check so the status flips off "unknown" quickly.
        self.check_once()
        while not self._stop_event.is_set():
            interval = max(5, int(settings.ollama_health_check_interval))
            if self._stop_event.wait(timeout=interval):
                return
            self.check_once()

    def check_once(self) -> str:
        """Perform one ``/api/tags`` probe and update state. Returns new status.

        Uses a short HTTP timeout so the probe can't itself hang. Exceptions
        are captured into state; they never escape.
        """
        host = settings.ollama_host.rstrip("/")
        url = f"{host}/api/tags"
        timeout = max(1.0, float(settings.ollama_health_check_timeout))
        now = time.time()
        try:
            resp = requests.get(url, timeout=timeout)
            with self._state._lock:
                self._state.last_checked = now
            if resp.status_code == 200:
                self.mark_healthy()
            elif resp.status_code in RETRYABLE_STATUS_CODES:
                self.mark_degraded(f"HTTP {resp.status_code}")
            else:
                # 4xx (non-retryable) is still a server reachable signal —
                # call it degraded rather than down so permanent config
                # problems still show up distinctly from a dead server.
                self.mark_degraded(f"HTTP {resp.status_code}")
        except Exception as exc:
            with self._state._lock:
                self._state.last_checked = now
            self.mark_down(f"{type(exc).__name__}: {exc}")
        return self.get_status()


# Module-level singleton. Code that needs health data talks to this instance.
_monitor = OllamaHealthMonitor()


def get_monitor() -> OllamaHealthMonitor:
    """Return the singleton monitor instance."""
    return _monitor


def start_monitor() -> None:
    """Start the singleton health monitor (idempotent)."""
    _monitor.start()


def stop_monitor() -> None:
    """Stop the singleton health monitor (for tests/shutdown)."""
    _monitor.stop()


def get_ollama_status() -> dict[str, Any]:
    """Return the current Ollama health snapshot (for /health endpoints)."""
    return _monitor.get_state()


# ---------------------------------------------------------------------------
# Pre-flight readiness probe
# ---------------------------------------------------------------------------


def _model_is_listed(name: str, tags: list[dict[str, Any]]) -> bool:
    """Check whether ``name`` matches any model listed in ``/api/tags``.

    Ollama lists models as ``name:tag`` (e.g. ``qwen3.5:27b``). A caller may
    pass either the full tagged name or just the base name, so accept either
    form: exact match, match after stripping our tag, or match against the
    listed base name.
    """
    listed: set[str] = {m.get("name", "") for m in tags if isinstance(m, dict)}
    if name in listed:
        return True
    base = name.split(":", 1)[0]
    return any(n == name or n.split(":", 1)[0] == base for n in listed)


def check_ollama_ready(model: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Pre-flight probe: confirm Ollama is up and ``model`` is ready to serve.

    Two sequential checks, each bounded by ``timeout`` seconds:

    1. ``GET /api/tags`` — verifies the server is reachable and the model is
       installed.
    2. 1-token ``generate`` via ``_ollama_client`` — verifies the model can
       actually respond (catches cold-start hangs where the GPU is still
       loading weights).

    Returns
    -------
    tuple[bool, str]
        ``(True, reason)`` when both checks pass. ``(False, reason)`` on
        connection refused, missing model, or >timeout response. Never raises.
    """
    host = settings.ollama_host.rstrip("/")

    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
    except (requests.ConnectionError, ConnectionRefusedError) as exc:
        return False, f"connection refused: {exc}"
    except (requests.Timeout, TimeoutError) as exc:
        return False, f"timeout contacting {host}/api/tags: {exc}"
    except Exception as exc:
        return False, f"error contacting {host}/api/tags: {type(exc).__name__}: {exc}"

    if resp.status_code != 200:
        return False, f"/api/tags returned HTTP {resp.status_code}"

    try:
        tags = resp.json().get("models", [])
    except Exception as exc:
        return False, f"invalid /api/tags response: {exc}"

    if not _model_is_listed(model, tags):
        available = sorted({m.get("name", "") for m in tags if isinstance(m, dict)})
        return False, f"model not loaded: {model} (available: {available})"

    # Local import avoids a circular dep at module-load time (core imports us).
    from .core import _ollama_client

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="ollama-readycheck"
    )
    try:
        start = time.monotonic()
        future = executor.submit(
            _ollama_client.generate,
            model=model,
            prompt="hi",
            options={"num_predict": 1},
        )
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False, f"generate warmup exceeded {timeout:.1f}s timeout"
        except (requests.ConnectionError, ConnectionRefusedError) as exc:
            return False, f"connection refused during generate: {exc}"
        except ResponseError as exc:
            status = getattr(exc, "status_code", -1)
            return False, f"generate returned HTTP {status}: {exc}"
        except Exception as exc:
            return False, f"generate failed: {type(exc).__name__}: {exc}"
    finally:
        # Don't block startup on a still-running probe thread.
        executor.shutdown(wait=False)

    elapsed = time.monotonic() - start
    return True, f"ready (warmup {elapsed:.2f}s)"
