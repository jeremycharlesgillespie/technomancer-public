"""
Prompt Compression — shrink long Ollama message histories before the next call.

Ollama p95 latency scales with the number of input tokens: once the message
history crosses ~50K tokens, each turn pays the KV-cache cost for every byte
in context. Most of that weight is stale tool output that the LLM has already
acted on. This module collapses those older tool results into short stubs so
the next chat() call sees a tight prompt while the recent reasoning chain
stays intact.

Design:
- Always preserve the system prompt (index 0) and the last `keep_recent`
  non-system messages verbatim.
- Walk the middle of the list and replace bulky `tool` / `assistant` content
  with a short summary placeholder. User turns are left alone because they
  carry intent the model may need to re-check.
- Reports bytes saved so callers can log/observe the savings.

The helpers are pure functions so they can be tested without mocking Ollama.
"""

from __future__ import annotations

from typing import Any

# Compression is skipped unless the estimated context crosses this size.
# 200_000 chars ≈ 50K tokens — the point where Ollama latency starts to climb
# steeply on a 16GB GPU running qwen3.5:27b.
DEFAULT_COMPRESSION_THRESHOLD_CHARS = 200_000

# Minimum size for a single message before it's a compression candidate.
# Anything smaller isn't worth replacing with a stub.
MIN_COMPRESS_MSG_CHARS = 500

# Number of most recent non-system messages to keep fully intact.
DEFAULT_KEEP_RECENT = 4


def estimate_size_chars(messages: list[dict[str, Any]]) -> int:
    """Estimate the total character footprint of a message list.

    Sums `content` across all messages. Non-string content (dicts, lists for
    vision inputs) is ignored — those rarely dominate the footprint.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
    return total


def _summarize(content: str, max_chars: int = 120) -> str:
    """Return a short placeholder describing a collapsed message."""
    stripped = content.strip().replace("\n", " ")
    if len(stripped) <= max_chars:
        head = stripped
    else:
        head = stripped[:max_chars].rstrip() + "…"
    return f"[compressed {len(content):,} chars] {head}"


def compress_messages(
    messages: list[dict[str, Any]],
    max_chars: int = DEFAULT_COMPRESSION_THRESHOLD_CHARS,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[list[dict[str, Any]], int]:
    """Return a compressed copy of `messages` and the bytes saved.

    If the current footprint is already under `max_chars`, returns the
    original list unchanged with 0 bytes saved.

    Compression rules:
    - System messages are never touched.
    - The last `keep_recent` non-system messages are preserved verbatim.
    - Between those, tool and assistant messages over MIN_COMPRESS_MSG_CHARS
      are replaced with `_summarize()` stubs.
    - User messages are preserved so the model can re-read the original ask.
    """
    before = estimate_size_chars(messages)
    if before <= max_chars:
        return messages, 0

    if not messages:
        return messages, 0

    # Split system vs. body. System messages stay at the front in order.
    system_msgs = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]

    if keep_recent <= 0:
        keep_recent = 0
    cutoff = max(0, len(body) - keep_recent)
    compressible = body[:cutoff]
    recent = body[cutoff:]

    compressed_body: list[dict[str, Any]] = []
    for msg in compressible:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # Preserve user intent verbatim.
            compressed_body.append(msg)
            continue

        if not isinstance(content, str) or len(content) < MIN_COMPRESS_MSG_CHARS:
            compressed_body.append(msg)
            continue

        new_msg = dict(msg)
        new_msg["content"] = _summarize(content)
        compressed_body.append(new_msg)

    compressed = system_msgs + compressed_body + recent
    after = estimate_size_chars(compressed)
    saved = max(0, before - after)
    return compressed, saved


def is_cacheable_task(task: str, max_chars: int = 500) -> bool:
    """Decide whether a single user task is safe to serve from the response cache.

    Cache lookups use a hash of the normalized query. Long tasks are almost
    always user-specific (summaries, debugging, multi-paragraph requests) and
    would poison the cache with hits that aren't really equivalent. Short
    factual or conversational prompts (~"what time is it", "hi") are ideal
    cache candidates.
    """
    if not task or not isinstance(task, str):
        return False
    stripped = task.strip()
    if not stripped:
        return False
    return len(stripped) <= max_chars
