"""Load the real ``anthropic`` SDK, bypassing the process-wide shim.

The ``agent.anthropic_shim`` monkey-patches ``sys.modules['anthropic']``
to redirect every ``import anthropic`` through a ``claude -p`` subprocess.
That's the right default for cost control — but it also strips
``cache_control`` markers, which defeats prompt caching.

This module finds the real ``anthropic`` package on disk and loads it
under a private name so callers can opt back into the SDK (and cache
economics) for specific code paths without disturbing everything else.

Usage::

    from agent import real_anthropic
    anthropic = real_anthropic.get()
    client = anthropic.Anthropic(api_key=...)
    client.messages.create(..., system=[{..., "cache_control": {...}}])
"""

from __future__ import annotations

import importlib.util
import logging
import site
import sys
import types as _t
from pathlib import Path

logger = logging.getLogger(__name__)

_cached_module: _t.ModuleType | None = None


def _find_real_init() -> Path | None:
    """Locate the real anthropic package's __init__.py on disk."""
    candidates: list[Path] = []
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # pragma: no cover — extremely unusual environments
        pass
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site))

    for base in candidates:
        init_path = base / "anthropic" / "__init__.py"
        if init_path.is_file():
            return init_path
    return None


def get() -> _t.ModuleType:
    """Return the real anthropic SDK module (cached after first call).

    The process has the shim installed as ``sys.modules['anthropic']``.
    The real SDK's internal absolute imports (``from anthropic.types.*``)
    would resolve to the shim's stub types module, which is missing most
    symbols. So during load we:

      1. snapshot and remove every ``anthropic[.*]`` sys.modules entry
      2. let importlib populate sys.modules with the real package tree
      3. snapshot that real tree
      4. restore the shim entries so other code still sees the shim

    Callers hold the returned reference — they must NOT do
    ``import anthropic`` themselves, since that still yields the shim.

    Raises ImportError if the real package cannot be found on disk.
    """
    global _cached_module
    if _cached_module is not None:
        return _cached_module

    if _find_real_init() is None:
        raise ImportError(
            "real_anthropic: could not locate anthropic/__init__.py on disk. "
            "Is the package installed? Try `pip install anthropic`."
        )

    # Snapshot every sys.modules entry under 'anthropic' (the shim + its
    # stub submodules). We'll restore these after the real load.
    shim_snapshot = {
        name: mod for name, mod in sys.modules.items()
        if name == "anthropic" or name.startswith("anthropic.")
    }
    for name in shim_snapshot:
        del sys.modules[name]

    try:
        # Fresh import now resolves from disk — populates sys.modules with
        # the real anthropic package tree (anthropic, anthropic.types,
        # anthropic.types.tool_use_block, anthropic._client, ...).
        import importlib
        real_mod = importlib.import_module("anthropic")
        _cached_module = real_mod
        logger.info("real_anthropic: loaded from %s", real_mod.__file__)
    finally:
        # Restore the shim for everyone else — only our private reference
        # keeps the real tree alive.
        for name, mod in shim_snapshot.items():
            sys.modules[name] = mod

    return _cached_module
