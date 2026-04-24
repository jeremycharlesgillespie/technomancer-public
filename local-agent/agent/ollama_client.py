"""Direct HTTP client for the local Ollama server.

Intentionally separate from agent/ollama_shim.py, which re-routes the
``import ollama`` namespace to claude -p. This module is a NEW path for
callers that explicitly want local-ollama inference (classification,
dedup, short-answer routing). Nothing here replaces the shim.

One function: :func:`chat`. On any failure (network, non-200, bad
JSON), returns ``None`` so the caller can fall back to claude -p.

Backpressure: we track in-flight requests with an atomic counter.
When the count reaches ``MAX_CONCURRENT``, new calls return ``None``
immediately instead of joining Ollama's internal queue (default 512).
This prevents runaway loops from piling up thousands of requests and
stalling the entire system. 503 responses are routed to the health
monitor so the ``/health`` endpoint reflects queue saturation in real
time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST: str = "http://127.0.0.1:11434"


#: Default num_ctx. Ollama loads the model's theoretical context_length
#: (262144 for qwen3.5) which over-allocates KV cache and spills ~5 GB
#: onto CPU on a 16 GB VRAM card — generation then bottlenecks on
#: CPU-GPU transfers and GPU utilization sits around 25% while inference
#: takes minutes. 8K is more than enough for every classification prompt
#: we send (the splitter prompt tops out around ~500 tokens).
DEFAULT_NUM_CTX: int = 8192

# ---------------------------------------------------------------------------
# In-flight request tracking
# ---------------------------------------------------------------------------

#: Hard cap on simultaneous Ollama requests from this process. Ollama's
#: internal queue is 512; we shed load at 30 so the system stays responsive
#: and the health monitor has time to react before things cascade.
MAX_CONCURRENT: int = 30

_inflight_lock = threading.Lock()
_inflight_count: int = 0

# ---------------------------------------------------------------------------
# GPU exclusivity gate — held by OllamaCoder during story implementation
# ---------------------------------------------------------------------------

#: Set while OllamaCoder owns the GPU. Other callers wait up to 60s then
#: proceed anyway so the bot stays responsive if the coder stalls.
_coder_active = threading.Event()


def acquire_coder_priority() -> None:
    """Signal that OllamaCoder is starting. Other chat() callers yield the GPU."""
    _coder_active.set()
    logger.info("[ollama_client] Coder priority acquired — other callers will wait")


def release_coder_priority() -> None:
    """Signal that OllamaCoder is done. Other chat() callers resume immediately."""
    _coder_active.clear()
    logger.info("[ollama_client] Coder priority released")


def get_inflight_count() -> int:
    """Return the number of Ollama requests currently in-flight."""
    with _inflight_lock:
        return _inflight_count


def _notify_monitor_degraded(reason: str) -> None:
    """Push a degraded signal to the health monitor without importing at module level."""
    try:
        from .ollama_health import get_monitor

        get_monitor().mark_degraded(reason)
    except Exception:
        pass


def chat(
    prompt: str,
    model: str,
    timeout: int = 60,
    options: dict[str, Any] | None = None,
    keep_alive: str | int = 0,
    format: str | None = None,
) -> str | None:
    """Send a single-turn prompt to local ollama and return the response text.

    Args:
        prompt: Full prompt text (user message content; no role structure).
        model: Ollama model tag (e.g. ``"qwen3.5:latest"``, ``"llama3.2"``).
        timeout: Seconds to wait before giving up. Default 60.
        options: Optional ollama generation options (temperature, seed,
            num_ctx, etc.). Passed through verbatim. If ``num_ctx`` is
            not set, we default to :data:`DEFAULT_NUM_CTX` (8192) so the
            model fits fully in VRAM — see the note above.
        keep_alive: How long to keep the model resident after this call.
            Default ``0`` — unload immediately. This prevents classification
            models (brain, splitter, karen, web_search) from sitting in
            memory and evicting the long-running OllamaCoder on
            memory-constrained machines (e.g. 36GB M-series unified).
            OllamaCoder pins itself with ``keep_alive=-1`` and should not
            be displaced by a one-shot classification call that finished
            seconds ago. Pass ``-1`` for indefinite, ``"24h"`` or similar
            to keep resident on machines with ample VRAM.

    Returns:
        Response text, or ``None`` on any failure. Never raises.
    """
    global _inflight_count

    # GPU gate: yield to OllamaCoder if it's actively coding. Wait up to 60s
    # then proceed anyway so the bot never hard-blocks on a stalled coder.
    if _coder_active.is_set():
        logger.debug("[ollama_client] coder active — waiting up to 60s for GPU slot")
        _coder_active.wait(timeout=60)

    # Backpressure: shed load before we can fill Ollama's internal queue.
    with _inflight_lock:
        if _inflight_count >= MAX_CONCURRENT:
            logger.warning(
                "[ollama_client] in-flight cap reached (%d/%d), shedding request for %s",
                _inflight_count,
                MAX_CONCURRENT,
                model,
            )
            return None
        _inflight_count += 1

    try:
        opts = dict(options or {})
        opts.setdefault("num_ctx", DEFAULT_NUM_CTX)
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": opts,
            # qwen3.5 and other Qwen variants emit reasoning in a separate
            # ``thinking`` field when think-mode is on, which leaves ``response``
            # empty. Classification callers want the answer, not the CoT, so
            # disable thinking by default. Callers that specifically want CoT
            # can pass ``options={"think": True}``.
            "think": opts.pop("think", False),
        }
        if format:
            body["format"] = format
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("[ollama_client] %s network error: %s", model, exc)
            return None
        if r.status_code == 503:
            # Ollama's queue is full — tell the health monitor so /health
            # reflects saturation immediately (not just on the next /api/tags poll).
            logger.warning(
                "[ollama_client] %s HTTP 503 — Ollama queue full (inflight=%d)",
                model,
                _inflight_count,
            )
            _notify_monitor_degraded("HTTP 503 - Ollama queue full")
            return None
        if r.status_code != 200:
            logger.warning(
                "[ollama_client] %s HTTP %d: %s",
                model,
                r.status_code,
                r.text[:200],
            )
            return None
        try:
            data = r.json()
        except ValueError:
            logger.warning("[ollama_client] %s returned non-JSON body", model)
            return None
        response = data.get("response")
        if not isinstance(response, str):
            return None
        return response.strip() or None
    finally:
        with _inflight_lock:
            _inflight_count -= 1
