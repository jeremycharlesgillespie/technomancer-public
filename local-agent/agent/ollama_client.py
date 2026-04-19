"""Direct HTTP client for the local Ollama server.

Intentionally separate from agent/ollama_shim.py, which re-routes the
``import ollama`` namespace to claude -p. This module is a NEW path for
callers that explicitly want local-ollama inference (classification,
dedup, short-answer routing). Nothing here replaces the shim.

One function: :func:`chat`. On any failure (network, non-200, bad
JSON), returns ``None`` so the caller can fall back to claude -p.
"""

from __future__ import annotations

import logging
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


def chat(
    prompt: str,
    model: str,
    timeout: int = 60,
    options: dict[str, Any] | None = None,
    keep_alive: str | int = "24h",
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
            Default ``"24h"`` so classification workers don't pay the
            ~30s reload cost. Use ``-1`` for indefinite, ``0`` to force
            immediate unload.

    Returns:
        Response text, or ``None`` on any failure. Never raises.
    """
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
    if r.status_code != 200:
        logger.warning(
            "[ollama_client] %s HTTP %d: %s",
            model, r.status_code, r.text[:200],
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
