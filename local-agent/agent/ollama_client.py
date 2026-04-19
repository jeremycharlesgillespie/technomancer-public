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


def chat(
    prompt: str,
    model: str,
    timeout: int = 60,
    options: dict[str, Any] | None = None,
    keep_alive: str | int = "24h",
) -> str | None:
    """Send a single-turn prompt to local ollama and return the response text.

    Args:
        prompt: Full prompt text (user message content; no role structure).
        model: Ollama model tag (e.g. ``"qwen3.5:latest"``, ``"llama3.2"``).
        timeout: Seconds to wait before giving up. Default 60.
        options: Optional ollama generation options (temperature, seed,
            num_ctx, etc.). Passed through verbatim.
        keep_alive: How long to keep the model resident after this call.
            Default ``"24h"`` so classification workers don't pay the
            ~30s reload cost. Use ``-1`` for indefinite, ``0`` to force
            immediate unload.

    Returns:
        Response text, or ``None`` on any failure. Never raises.
    """
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if options:
        body["options"] = options
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
