"""Drop-in replacement for the ``ollama`` Python client that routes every
call to Claude Code's ``claude -p`` subprocess.

Ollama was flaky on this host (OLLAMA_NUM_PARALLEL=1, VRAM contention,
queue saturation) and every path had been falling back to Claude anyway.
After that prolonged pain we gut it: all chat/generate calls route
through ``claude -p`` instead. The Anthropic API is ALSO gone from this
codebase — subprocess only — so we never leak credentials or burn API
credits.

What's supported
----------------
* ``Client.chat(model, messages, options=None, stream=False)``
* ``Client.generate(model, prompt, options=None, stream=False)``
* ``Client.list()`` / ``show()`` / ``ps()`` — stubs, return empty-ish
* ``ResponseError`` — re-exported for ``except ollama.ResponseError``

What's gone
-----------
* ``embeddings()`` raises ``ResponseError`` (no Claude embeddings endpoint).
* ``stream=True`` is ignored; full response returned at once.
* Image / vision inputs are **dropped silently**. Vision handling was a
  liability and rarely useful — text-only from here out.
* Ollama-specific options (temperature, num_ctx, stop) are discarded;
  Claude -p owns its own defaults. ``timeout`` is honoured.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude binary discovery (same logic as aim.brain._find_claude_binary)
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
# Exceptions (mimic ollama._types)
# ---------------------------------------------------------------------------

class ResponseError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = message


class RequestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: Iterable[dict[str, Any]]) -> str:
    """Flatten a chat-style messages list into a single prompt string.

    Keeps role prefixes so Claude can tell system/user/assistant turns
    apart. Text-only — any image parts in multi-part content are skipped.
    """
    lines: list[str] = []
    for m in messages or []:
        role = str(m.get("role", "user")).strip() or "user"
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else ""
                for p in content
            )
        content = (content or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core subprocess wrapper
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: int = 120
DEFAULT_MAX_TURNS: int = 3


def _call_claude_p(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run ``claude -p`` and return the text response.

    Raises ``ResponseError`` on timeout or non-zero exit.
    """
    binary = _find_claude_binary()
    if not binary:
        raise ResponseError("Claude binary not found on PATH or in VS Code extension", 500)

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
        raise ResponseError(f"claude -p timed out after {timeout}s", 504) from exc
    except FileNotFoundError as exc:
        raise ResponseError(f"claude binary not executable: {exc}", 500) from exc

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-500:]
        raise ResponseError(
            f"claude -p failed (rc={result.returncode}): {stderr_tail}",
            500,
        )
    return (result.stdout or "").strip()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Client:
    """Drop-in replacement for ``ollama.Client``."""

    def __init__(
        self,
        host: str | None = None,
        timeout: int | float | None = None,
        **_kwargs: Any,
    ) -> None:
        self.host = host or "claude-p"
        self.timeout = int(timeout) if timeout else DEFAULT_TIMEOUT

    def chat(
        self,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        format: Any | None = None,  # noqa: A002
        **_kwargs: Any,
    ) -> dict[str, Any]:
        prompt = _messages_to_prompt(messages or [])
        text = _call_claude_p(prompt=prompt, timeout=self.timeout)
        return {
            "model": model or "claude-p",
            "created_at": "",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
        }

    def generate(
        self,
        model: str | None = None,
        prompt: str | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        text = _call_claude_p(prompt=prompt or "", timeout=self.timeout)
        return {
            "model": model or "claude-p",
            "response": text,
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
        }

    def list(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": "claude-p:shim",
                    "model": "claude-p:shim",
                    "size": 0,
                    "modified_at": "",
                    "details": {
                        "format": "claude-p",
                        "family": "claude",
                        "parameter_size": "unknown",
                    },
                }
            ]
        }

    def show(self, model: str | None = None) -> dict[str, Any]:
        return {"modelfile": "", "parameters": "", "template": ""}

    def ps(self) -> dict[str, Any]:
        return {"models": []}

    def embeddings(self, model: str | None = None, prompt: str | None = None) -> dict[str, Any]:
        raise ResponseError(
            "embeddings() is not supported by the ollama_shim — "
            "no embeddings endpoint available in text-only mode",
            501,
        )

    def embed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.embeddings(*args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_client: Client | None = None


def _get_default() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def chat(
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _get_default().chat(model=model, messages=messages, **kwargs)


def generate(
    model: str | None = None,
    prompt: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _get_default().generate(model=model, prompt=prompt, **kwargs)


def list() -> dict[str, Any]:  # noqa: A001
    return _get_default().list()


def show(model: str | None = None) -> dict[str, Any]:
    return _get_default().show(model=model)


def ps() -> dict[str, Any]:
    return _get_default().ps()


def embeddings(model: str | None = None, prompt: str | None = None) -> dict[str, Any]:
    return _get_default().embeddings(model=model, prompt=prompt)


def embed(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _get_default().embed(*args, **kwargs)


# ---------------------------------------------------------------------------
# Submodule compat — ``from ollama._types import ResponseError``
# ---------------------------------------------------------------------------

class _TypesModule:
    ResponseError = ResponseError
    RequestError = RequestError


_types = _TypesModule()


# ---------------------------------------------------------------------------
# Install as a stand-in for the real ``ollama`` module
# ---------------------------------------------------------------------------

def install_as_ollama() -> None:
    """Replace ``sys.modules['ollama']`` with this shim so every
    ``import ollama`` transparently picks us up.

    Call this once from the process entry point (Discord bot, AIM
    manager, AIM worker). MUST run before other modules ``import ollama``.
    """
    import sys
    import types as _t

    if sys.modules.get("ollama") is sys.modules.get(__name__):
        return

    mod = _t.ModuleType("ollama")
    mod.Client = Client
    mod.chat = chat
    mod.generate = generate
    mod.list = list
    mod.show = show
    mod.ps = ps
    mod.embeddings = embeddings
    mod.embed = embed
    mod.ResponseError = ResponseError
    mod.RequestError = RequestError
    mod._types = _types  # noqa: SLF001

    types_mod = _t.ModuleType("ollama._types")
    types_mod.ResponseError = ResponseError
    types_mod.RequestError = RequestError

    sys.modules["ollama"] = mod
    sys.modules["ollama._types"] = types_mod
    logger.info("ollama_shim: installed as sys.modules['ollama']")
