"""Role-based LLM routing — ollama for classification, Claude for nuance.

Callers pass a **role** (``"aim_brain"``, ``"dedup_judge"``,
``"aimm_observer"``, ``"aimm_suggester"``) plus a prompt. The router
looks up the role's configured model from settings and dispatches to
either the local ollama HTTP API or a ``claude -p`` subprocess.

Model-name convention:
  - ``"ollama:<tag>"`` — route to local ollama (e.g. ``ollama:qwen3.5:27b``).
  - anything else — route to ``claude -p --model <name>`` (e.g.
    ``claude-haiku-4-5``, ``claude-sonnet-4-6``).

Fallback chain per call:
  1. Primary model as configured.
  2. If ollama primary fails (connection / 500 / timeout), fall through
     to the role's claude fallback model (``settings.llm_fallback_model``,
     default ``claude-haiku-4-5``). This keeps the system working when
     the ollama daemon is down.
  3. If both fail, return ``None``. Every caller already handles this
     cleanly because the existing claude-only path returned ``None`` on
     failure.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

from agent.config import get_settings
from agent.ollama_client import chat as ollama_chat

logger = logging.getLogger(__name__)

Role = Literal[
    "aim_brain",
    "dedup_judge",
    "aimm_observer",
    "aimm_suggester",
    "splitter_decomposer",
    "evergreen_generator",
    "aiv_classifier",
    "aiv_scorer",
    "aiv_ab_compare",
    "dev_learning",
]


_claude_binary_cache: str | None = None


def _find_claude_binary() -> str | None:
    """Locate the Claude Code binary (PATH + VS Code extension dir)."""
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
        for ext_dir in sorted(
            ext_root.glob("anthropic.claude-code-*"), reverse=True
        ):
            for name in ("claude.exe", "claude"):
                cand = ext_dir / "resources" / "native-binary" / name
                if cand.is_file():
                    _claude_binary_cache = str(cand)
                    return _claude_binary_cache
    return None


def _claude_chat(prompt: str, model: str, timeout: int) -> str | None:
    """Run ``claude -p --model <model>`` and return stdout, or None."""
    binary = _find_claude_binary()
    if not binary:
        logger.warning("[llm_router] claude binary not found")
        return None
    try:
        result = subprocess.run(
            [
                binary, "-p", prompt,
                "--output-format", "text",
                "--max-turns", "3",
                "--model", model,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        logger.warning("[llm_router] claude -p %s timed out after %ds", model, timeout)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm_router] claude -p %s error: %s", model, exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "[llm_router] claude -p %s returned %d: %s",
            model, result.returncode, (result.stderr or "")[:200],
        )
        return None
    out = (result.stdout or "").strip()
    return out or None


_ROLE_TO_SETTING: dict[Role, str] = {
    "aim_brain": "aim_brain_model",
    "dedup_judge": "dedup_judge_model",
    "aimm_observer": "aimm_observer_model",
    "aimm_suggester": "aimm_suggester_model",
    "splitter_decomposer": "splitter_decomposer_model",
    "evergreen_generator": "evergreen_generator_model",
    "aiv_classifier": "aiv_classifier_model",
    "aiv_scorer": "aiv_scorer_model",
    "aiv_ab_compare": "aiv_ab_compare_model",
    "dev_learning": "dev_learning_model",
}


_EXPERIMENT_ALIASES: dict[str, str] = {
    "ollama": "ollama:qwen3.5:latest",
    "claude": "claude-haiku-4-5",
}


def _resolve_primary(role: Role) -> str:
    """Resolve the primary model for a role.

    Order of precedence (highest wins):
      1. ``settings.llm_experiment_mode`` — if set, overrides ALL roles.
         Accepts aliases (``ollama``, ``claude``) or explicit model
         strings (``ollama:llama3.1:8b``, ``claude-sonnet-4-6``).
      2. ``settings.<role>_model`` — per-role default.
      3. ``claude-haiku-4-5`` — hard fallback if nothing else is set.

    The experiment mode is the single knob for "let me try <X> across
    every non-coding role." Unset it to return to per-role settings.
    """
    s = get_settings()
    experiment = (getattr(s, "llm_experiment_mode", "") or "").strip()
    if experiment:
        return _EXPERIMENT_ALIASES.get(experiment, experiment)
    attr = _ROLE_TO_SETTING[role]
    return getattr(s, attr, None) or "claude-haiku-4-5"


def current_routing() -> dict[str, str]:
    """Return the currently-active model per role.

    Useful for the ``python -m agent.llm_router`` CLI + hub /llm panel.
    Reflects the experiment-mode override when set.
    """
    return {role: _resolve_primary(role) for role in _ROLE_TO_SETTING}  # type: ignore[arg-type]


def _cli() -> int:
    """CLI entry point — print current routing for quick experiment checks.

    Usage:
        python -m agent.llm_router
    """
    routing = current_routing()
    s = get_settings()
    experiment = (getattr(s, "llm_experiment_mode", "") or "").strip() or "(unset — per-role settings apply)"
    print(f"LLM_EXPERIMENT_MODE: {experiment}")
    print(f"LLM_FALLBACK_MODEL:  {_resolve_fallback()}")
    print()
    print("Current routing:")
    for role, model in routing.items():
        print(f"  {role:20s} -> {model}")
    return 0


def _resolve_fallback(role: Role | None = None) -> str:
    """Return the claude fallback model, or empty string for no-fallback.

    Resolution order:
      1. ``settings.<role>_fallback_model`` — per-role override. Used to
         enable Claude fallback for a single role (e.g. ``aiv_scorer``)
         without flipping every classification role onto Claude quota.
      2. ``settings.llm_fallback_model`` — global default. Default is
         "" — an intentional choice to keep classification roles
         pure-Ollama and avoid silent Claude quota burn from transient
         Ollama errors. Callers already handle None gracefully.
    """
    s = get_settings()
    if role is not None:
        per_role_attr = f"{role}_fallback_model"
        per_role = getattr(s, per_role_attr, None)
        if per_role:
            return per_role
    return getattr(s, "llm_fallback_model", None) or ""


def complete(
    role: Role,
    prompt: str,
    timeout: int = 60,
    format: str | None = None,
) -> str | None:
    """Send ``prompt`` to the LLM configured for this ``role``.

    Returns stripped response text, or ``None`` on every-path failure.
    Never raises — callers already handle None (the legacy claude-only
    behavior returned None on any error).

    When the primary is Ollama and the call fails, we retry Ollama once
    before (optionally) falling back to Claude. This avoids silently
    burning Max-20x quota on transient Ollama hiccups — the backend we
    deliberately chose to offload to.

    Args:
        format: Optional Ollama ``format`` constraint (e.g. ``"json"``).
            Only applied when the primary is Ollama; ignored on Claude
            fallback because Claude doesn't support the same flag and
            forcing JSON via prompt is the existing convention there.
    """
    primary = _resolve_primary(role)
    fallback = _resolve_fallback(role)

    if primary.startswith("ollama:"):
        tag = primary.split(":", 1)[1]
        out = ollama_chat(prompt, tag, timeout=timeout, format=format)
        if out is not None:
            return out
        # Retry once before considering this a hard failure.
        logger.info("[llm_router] %s ollama:%s failed, retrying once", role, tag)
        out = ollama_chat(prompt, tag, timeout=timeout, format=format)
        if out is not None:
            return out
        if not fallback:
            logger.warning(
                "[llm_router] %s ollama:%s failed twice; no claude fallback configured, returning None",
                role, tag,
            )
            return None
        logger.info(
            "[llm_router] %s ollama:%s failed twice, falling back to %s",
            role, tag, fallback,
        )
        return _claude_chat(prompt, fallback, timeout=timeout)

    # Claude primary
    return _claude_chat(prompt, primary, timeout=timeout)


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_cli())
