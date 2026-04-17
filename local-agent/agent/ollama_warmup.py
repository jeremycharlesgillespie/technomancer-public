"""
Ollama Model Warmup — preload models on bot startup.

Ollama loads model weights into VRAM lazily on the first request, so the first
user message after a restart can take 20-40s while a 17GB model streams in
from disk. This module fires a one-token generation per configured model from
``on_ready`` so the cold-start happens during startup (when nobody is waiting)
instead of during the user's first turn.

Failures are logged per model and never raised — a warmup is best-effort, and
refusing to start the bot because a vision model isn't pulled yet would be
worse than the latency it saves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable

from .core import _ollama_client

log = logging.getLogger(__name__)


def _warmup_one(model_name: str) -> float:
    """Send a single-token generation to ``model_name``. Returns elapsed seconds.

    Runs synchronously in a worker thread. Exceptions propagate to the caller
    so the async wrapper can log them with context — the wrapper is what makes
    warmup best-effort.
    """
    start = time.monotonic()
    _ollama_client.generate(
        model=model_name,
        prompt="hi",
        options={"num_predict": 1},
    )
    return time.monotonic() - start


async def warmup_models(model_names: Iterable[str]) -> None:
    """Fire a one-token generation at each model to force weight loading.

    Called fire-and-forget from ``discord_memory_bot.on_ready`` via
    ``asyncio.create_task`` so startup never blocks on model load. Each model
    is warmed sequentially — loading two 17GB models at once would evict one
    or thrash the GPU, so overlapping buys nothing.

    Empty / duplicate names are skipped silently. All exceptions are caught
    and logged; this function never raises.
    """
    # Preserve order but drop duplicates and empty strings. A common case is
    # ollama_model == ollama_vision_model (e.g. a multimodal qwen release);
    # warming it twice just doubles the latency with no benefit.
    seen: set[str] = set()
    unique: list[str] = []
    for name in model_names:
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)

    if not unique:
        log.info("Ollama warmup skipped: no models configured")
        return

    log.info("Ollama warmup starting for %d model(s): %s", len(unique), unique)

    for name in unique:
        try:
            elapsed = await asyncio.to_thread(_warmup_one, name)
            log.info("Ollama warmup: %s ready in %.2fs", name, elapsed)
        except Exception as exc:
            # Log and move on — a missing or broken model should not prevent
            # the other models (or the bot) from coming up.
            log.warning(
                "Ollama warmup failed for %s: %s: %s",
                name, type(exc).__name__, exc,
            )
