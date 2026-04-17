"""Drop-in replacement for the ``anthropic`` Python SDK that routes every
``messages.create`` call to Claude Code's ``claude -p`` subprocess.

Scope
-----
The codebase had three modules hitting the Anthropic API directly —
``agent/ask_claude.py``, ``agent/claude_bridge.py``, ``agent/claude_vault.py``
— plus ``agent/itinerary.py`` for a minor fallback. Burning API credits
while the user has a Claude Pro subscription is silly, and we also want
zero API-key surface in this process. So we shim the SDK.

What's supported
----------------
* ``anthropic.Anthropic(api_key=...)`` — constructor is a no-op; api_key
  is ignored. Client exposes ``.messages.create(...)``.
* ``.messages.create(model, messages, max_tokens, system=None,
  timeout=None, extra_headers=None, stream=False, ...)`` — returns a
  ``Message`` object with the same duck-typed shape callers expect:
  ``response.content[0].text``, ``response.usage.<token counters>``,
  ``response.stop_reason``.
* Error types ``APITimeoutError``, ``APIConnectionError``,
  ``APIStatusError``, ``APIError``, ``BadRequestError`` — re-exported so
  ``except anthropic.APITimeoutError`` paths keep working.

What's gone
-----------
* Streaming (``stream=True``) — ignored; full response returned at once.
* Tool use, files, batches, citations, prompt caching wire format —
  all discarded silently. ``claude -p`` handles its own caching.
* Image / vision inputs — dropped. Text only.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude binary discovery
# ---------------------------------------------------------------------------

_claude_binary_cache: str | None = None


def _find_claude_binary() -> str | None:
    global _claude_binary_cache
    if _claude_binary_cache:
        return _claude_binary_cache
    for name in ("claude", "claude.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(d) / name
            if cand.is_file():
                _claude_binary_cache = str(cand)
                return _claude_binary_cache
    ext_root = Path.home() / ".vscode" / "extensions"
    if ext_root.is_dir():
        for ext_dir in sorted(ext_root.glob("anthropic.claude-code-*"), reverse=True):
            for name in ("claude.exe", "claude"):
                cand = ext_dir / "resources" / "native-binary" / name
                if cand.is_file():
                    _claude_binary_cache = str(cand)
                    return _claude_binary_cache
    return None


# ---------------------------------------------------------------------------
# Exceptions (mimic anthropic.*)
# ---------------------------------------------------------------------------

class APIError(Exception):
    # Accept anthropic's kwargs (request=, body=, ...) so test code that
    # constructs the real exception shape keeps working with the shim.
    def __init__(
        self,
        message: str = "",
        *,
        request: Any = None,
        body: Any = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.request = request
        self.body = body


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(APIError):
    def __init__(
        self,
        message: str = "",
        *,
        status_code: int = 500,
        response: Any = None,
        request: Any = None,
        body: Any = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(message, request=request, body=body)
        self.status_code = status_code
        self.response = response


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


# ---------------------------------------------------------------------------
# Response shape (duck-typed to match anthropic.types.Message)
# ---------------------------------------------------------------------------

@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Message:
    content: list[_TextBlock] = field(default_factory=list)
    model: str = "claude-p"
    role: str = "assistant"
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    type: str = "message"
    id: str = ""
    usage: _Usage = field(default_factory=_Usage)


# ---------------------------------------------------------------------------
# Core subprocess wrapper
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = 120.0
DEFAULT_MAX_TURNS: int = 3


def _call_claude_p(prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    binary = _find_claude_binary()
    if not binary:
        raise APIConnectionError("Claude binary not found (anthropic_shim)")

    cmd = [
        binary, "-p", prompt,
        "--output-format", "text",
        "--max-turns", str(DEFAULT_MAX_TURNS),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise APITimeoutError(f"claude -p timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise APIConnectionError(f"claude binary not executable: {exc}") from exc

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-500:]
        raise APIStatusError(
            f"claude -p failed (rc={result.returncode}): {stderr_tail}",
            status_code=500,
        )
    return (result.stdout or "").strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _render_system(system: Any) -> str:
    """Flatten a system prompt (string OR list of content blocks) to text."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in system
        ).strip()
    return str(system)


def _render_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages or []:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        if isinstance(content, list):
            # Concatenate text blocks; skip non-text
            content = "".join(
                p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else ""
                for p in content
            )
        content = (content or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Messages resource
# ---------------------------------------------------------------------------

class _Messages:
    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def create(
        self,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        system: Any = None,
        temperature: float | None = None,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> _Message:
        system_text = _render_system(system)
        user_text = _render_messages(messages or [])
        if system_text:
            prompt = f"SYSTEM: {system_text}\n\n{user_text}"
        else:
            prompt = user_text

        call_timeout = float(timeout) if timeout else DEFAULT_TIMEOUT
        text = _call_claude_p(prompt=prompt, timeout=call_timeout)

        return _Message(
            content=[_TextBlock(text=text)],
            model=model or "claude-p",
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
            id="",
            usage=_Usage(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class Anthropic:
    """Drop-in replacement for ``anthropic.Anthropic``.

    api_key is accepted for compatibility but ignored — we never call
    the Anthropic API, only the ``claude -p`` subprocess.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **_kwargs: Any,
    ) -> None:
        # Anthropic client ignores all of these — claude -p owns timing.
        self.api_key = "claude-p-shim"
        self.base_url = base_url or "claude-p"
        self.timeout = timeout
        self.max_retries = max_retries
        self.messages = _Messages(self)


class AsyncAnthropic(Anthropic):
    """Async client not actually async — claude -p is subprocess-only.

    Exposed so ``anthropic.AsyncAnthropic`` imports don't crash. If any
    caller awaits a result we'll return a plain synchronous response.
    """


# ---------------------------------------------------------------------------
# Install as a stand-in for the real ``anthropic`` module
# ---------------------------------------------------------------------------

def install_as_anthropic() -> None:
    """Replace ``sys.modules['anthropic']`` with this shim so every
    ``import anthropic`` transparently picks us up.

    MUST run before other modules ``import anthropic``.
    """
    import sys
    import types as _t

    if sys.modules.get("anthropic") is sys.modules.get(__name__):
        return

    mod = _t.ModuleType("anthropic")
    mod.Anthropic = Anthropic
    mod.AsyncAnthropic = AsyncAnthropic
    mod.APIError = APIError
    mod.APIConnectionError = APIConnectionError
    mod.APITimeoutError = APITimeoutError
    mod.APIStatusError = APIStatusError
    mod.BadRequestError = BadRequestError
    mod.AuthenticationError = AuthenticationError
    mod.RateLimitError = RateLimitError
    sys.modules["anthropic"] = mod

    # anthropic.types submodule — provide the message types used by callers.
    types_mod = _t.ModuleType("anthropic.types")
    types_mod.Message = _Message
    types_mod.TextBlock = _TextBlock
    types_mod.Usage = _Usage
    sys.modules["anthropic.types"] = types_mod
    logger.info("anthropic_shim: installed as sys.modules['anthropic']")
