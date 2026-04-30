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
    # Counts successful probes while status is ``down``. Drives the
    # hysteresis that prevents a single lucky probe from flipping a
    # flapping server back to healthy.
    recovery_successes: int = 0
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
                "recovery_successes": self.recovery_successes,
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
        """Record a successful Ollama interaction.

        Applies hysteresis when the current status is ``down``: requires
        ``settings.ollama_recovery_successes`` consecutive successes before
        flipping back to healthy. Transitions from ``unknown`` or
        ``degraded`` flip immediately because those states have already
        seen a working server, so one green check is enough.
        """
        threshold = max(1, int(settings.ollama_recovery_successes))
        with self._state._lock:
            prior = self._state.status
            if prior == STATUS_DOWN:
                self._state.recovery_successes += 1
                progress = self._state.recovery_successes
                if progress < threshold:
                    log.info(
                        "Ollama recovery probe %d/%d — staying down",
                        progress, threshold,
                    )
                    return
            self._state.status = STATUS_HEALTHY
            self._state.last_success = time.time()
            self._state.consecutive_failures = 0
            self._state.recovery_successes = 0
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
            # Any failure resets recovery progress — hysteresis requires
            # *consecutive* successes, not a running total.
            self._state.recovery_successes = 0
            current = self._state.status
        log.debug("Ollama health: %s (%s)", current, reason)

    def mark_down(self, reason: str) -> None:
        """Record an exhausted retry budget or repeated failures."""
        with self._state._lock:
            prior = self._state.status
            self._state.status = STATUS_DOWN
            self._state.last_error = reason
            self._state.consecutive_failures += 1
            self._state.recovery_successes = 0
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

    log.info("checking Ollama health for model=%s host=%s timeout=%.2fs", model, host, timeout)

    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
    except (requests.ConnectionError, ConnectionRefusedError) as exc:
        log.error("Ollama health check failed: status=unavailable error=%s", exc)
        return False, f"connection refused: {exc}"
    except (requests.Timeout, TimeoutError) as exc:
        log.error("Ollama health check failed: status=unavailable error=%s", exc)
        return False, f"timeout contacting {host}/api/tags: {exc}"
    except Exception as exc:
        log.error("Ollama health check failed: status=unavailable error=%s", exc)
        return False, f"error contacting {host}/api/tags: {type(exc).__name__}: {exc}"

    if resp.status_code != 200:
        log.error("Ollama health check failed: status=unavailable http_status=%d", resp.status_code)
        return False, f"/api/tags returned HTTP {resp.status_code}"

    try:
        tags = resp.json().get("models", [])
    except Exception as exc:
        log.error("Ollama health check failed: status=unavailable error=%s", exc)
        return False, f"invalid /api/tags response: {exc}"

    if not _model_is_listed(model, tags):
        available = sorted({m.get("name", "") for m in tags if isinstance(m, dict)})
        log.error("Ollama health check failed: status=unavailable model=%s available=%s", model, available)
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
            log.error("Ollama health check failed: status=unavailable timeout=%.2fs", timeout)
            return False, f"generate warmup exceeded {timeout:.1f}s timeout"
        except (requests.ConnectionError, ConnectionRefusedError) as exc:
            log.error("Ollama health check failed: status=unavailable error=%s", exc)
            return False, f"connection refused during generate: {exc}"
        except ResponseError as exc:
            status = getattr(exc, "status_code", -1)
            log.error("Ollama health check failed: status=unavailable http_status=%d error=%s", status, exc)
            return False, f"generate returned HTTP {status}: {exc}"
        except Exception as exc:
            log.error("Ollama health check failed: status=unavailable error=%s", exc)
            return False, f"generate failed: {type(exc).__name__}: {exc}"
    finally:
        # Don't block startup on a still-running probe thread.
        executor.shutdown(wait=False)

    elapsed = time.monotonic() - start
    log.info("Ollama health check succeeded: status=healthy model=%s version=%s warmup=%.2fs", model, host, elapsed)
    return True, f"ready (warmup {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# OllamaUnavailable — structured exception for gated call sites
# ---------------------------------------------------------------------------


class OllamaUnavailable(Exception):
    """Raised by :class:`OllamaHealthGate` when Ollama cannot serve a call.

    Carries a snapshot of the health state at the time of failure so
    upstream handlers can format user-facing messages without a second
    round-trip to the monitor. Handlers decide the degradation path:

    - chat → route the turn through Claude (``ask_claude``)
    - news digest → skip commentary, post headlines-only
    - image identification → skip local stage, go straight to Claude
    """

    def __init__(self, reason: str, state: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.state = state

    @property
    def status(self) -> str:
        """Shorthand for ``self.state['status']`` with an ``unknown`` fallback."""
        return self.state.get("status", STATUS_UNKNOWN)


# ---------------------------------------------------------------------------
# OllamaHealthGate — unified entry point for every Ollama call site
# ---------------------------------------------------------------------------


class OllamaHealthGate:
    """Gate every Ollama call through a single, health-aware entry point.

    State-driven behavior:

    - ``healthy``  → pass through with normal retry/backoff settings.
    - ``degraded`` → pass through with a shorter retry budget so we don't
      hammer a struggling server (settings: ``ollama_degraded_*``).
    - ``down``     → fast-fail with :class:`OllamaUnavailable` *without*
      invoking ``fn``. Recovery is driven by the health monitor's probes,
      which require ``ollama_recovery_successes`` consecutive greens to
      flip state back to healthy (hysteresis).

    Rate-limited Claude escalations are tracked here too — callers that
    catch ``OllamaUnavailable`` ask :meth:`should_escalate` before paying
    for a Claude request, and notify :meth:`record_escalation` afterward
    so the hourly count stays accurate and visible to operators.
    """

    def __init__(self, monitor: OllamaHealthMonitor | None = None) -> None:
        self._monitor = monitor if monitor is not None else _monitor
        self._escalation_lock = threading.Lock()
        self._escalation_times: list[float] = []

    # -- call path --------------------------------------------------------

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        on_transient_error: Callable[[BaseException, int], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute ``fn(*args, **kwargs)`` through the health gate.

        Raises
        ------
        OllamaUnavailable
            When the gate is ``down`` (fast-fail, ``fn`` is never invoked),
            or when retries exhaust and the monitor flips to ``down`` mid-call.
        Exception
            Any non-transient exception raised by ``fn`` (e.g. bad args,
            unknown model) is propagated unchanged.
        """
        status = self._monitor.get_status()
        if status == STATUS_DOWN:
            state = self._monitor.get_state()
            reason = state.get("last_error") or "Ollama unreachable"
            raise OllamaUnavailable(f"Ollama is down: {reason}", state)

        if status == STATUS_DEGRADED:
            if max_retries is None:
                max_retries = settings.ollama_degraded_max_retries
            if base_delay is None:
                base_delay = settings.ollama_degraded_base_delay
            if max_delay is None:
                max_delay = settings.ollama_degraded_max_delay

        try:
            return ollama_call_with_retries(
                fn,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                on_transient_error=on_transient_error,
                **kwargs,
            )
        except OllamaUnavailable:
            raise
        except Exception as exc:
            # Retries exhausted on a transient error → monitor flipped to
            # down. Translate so callers have one type to catch.
            if self._monitor.get_status() == STATUS_DOWN:
                state = self._monitor.get_state()
                raise OllamaUnavailable(
                    f"Ollama unavailable: {type(exc).__name__}: {exc}",
                    state,
                ) from exc
            raise

    # -- escalation tracking ---------------------------------------------

    def record_escalation(self, reason: str = "") -> int:
        """Log one Ollama→Claude auto-escalation; returns rolling-hour count.

        Emits a WARN log with the running count so operators can see the
        bot trading Ollama outages for Claude credits in real time.
        """
        now = time.time()
        cutoff = now - 3600.0
        with self._escalation_lock:
            self._escalation_times = [t for t in self._escalation_times if t >= cutoff]
            self._escalation_times.append(now)
            count = len(self._escalation_times)
        log.warning(
            "Ollama→Claude escalation #%d in past hour%s",
            count,
            f" (reason={reason})" if reason else "",
        )
        return count

    def escalation_rate(self, window_seconds: float = 3600.0) -> int:
        """Return the number of escalations in the trailing ``window_seconds``."""
        cutoff = time.time() - window_seconds
        with self._escalation_lock:
            self._escalation_times = [t for t in self._escalation_times if t >= cutoff]
            return len(self._escalation_times)

    def should_escalate(
        self,
        max_per_window: int | None = None,
        window_seconds: float = 3600.0,
    ) -> tuple[bool, str]:
        """Gate a Claude escalation by the configured rate-limit.

        Returns ``(allowed, reason)``. ``reason`` is suitable for logging
        or surfacing to Discord so the user sees why the bot did (or
        didn't) fall through to Claude.
        """
        limit = (
            settings.ollama_escalation_max_per_hour
            if max_per_window is None
            else max_per_window
        )
        count = self.escalation_rate(window_seconds)
        if count >= limit:
            return False, (
                f"escalation rate-limit hit ({count}/{limit} in "
                f"{int(window_seconds)}s) — serving fallback without Claude"
            )
        return True, f"escalation allowed ({count}/{limit} used in window)"

    # -- inspection -------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return the monitor snapshot plus gate-specific counters."""
        snap = self._monitor.get_state()
        snap["escalations_last_hour"] = self.escalation_rate(3600.0)
        return snap


# Module-level singleton — follow-up stories migrate individual call
# sites from ``ollama_call_with_retries`` to ``gated_call``.
_gate = OllamaHealthGate()


def get_gate() -> OllamaHealthGate:
    """Return the singleton :class:`OllamaHealthGate`."""
    return _gate


def gated_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Shortcut for ``get_gate().call(fn, *args, **kwargs)``.

    Intended as the canonical entry point for every Ollama call site.
    """
    return _gate.call(fn, *args, **kwargs)
