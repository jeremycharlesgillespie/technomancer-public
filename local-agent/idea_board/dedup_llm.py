"""LLM-judged near-exact dedup for idea-board stories.

Word-overlap dedup (the legacy mechanism in ``idea_board.models._is_duplicate``)
routinely misjudges nuance: TK-571 ("Unit tests for capability_request.py
50% -> 75%") was auto-vetoed against TK-321 (older abstract coverage story
already Done) because both share ``{unit, tests, capability_request}`` stems.
Word counting can't tell follow-up coverage lifts, refactors, and extensions
apart from rewordings of the same work.

This module exposes :func:`is_near_exact_duplicate` which shells out to
``claude -p --model claude-haiku-4-5`` with a short prompt comparing two
stories and returns ``(bool, reason)``. The judgment is deliberately
biased toward ``DIFFERENT`` — false-keeps cost one redundant commit;
false-vetoes silently kill real work.

Failure policy (all paths return ``(False, <error_code>)`` so the caller
falls open and the new story is *not* blocked by an LLM outage):

    * Missing claude binary       → ``(False, "no_binary")``
    * ``subprocess.TimeoutExpired`` → ``(False, "llm_timeout")``
    * Non-zero exit code          → ``(False, "llm_exit_<code>")``
    * No JSON object in stdout    → ``(False, "no_json")``
    * Malformed JSON              → ``(False, "parse_failure")``

The function never raises — every caught exception path returns a tuple
so callers can log the reason and proceed without try/except scaffolding.

Usage::

    from idea_board.dedup_llm import is_near_exact_duplicate
    is_dup, reason = is_near_exact_duplicate(
        a_title, a_desc, b_title, b_desc,
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "claude-haiku-4-5"
DEFAULT_TIMEOUT: int = 30
MAX_TITLE_CHARS: int = 200
MAX_DESC_CHARS: int = 2000

PROMPT_TEMPLATE: str = """You are comparing two Jira stories to decide whether story A and story B represent the SAME unit of work ("near-exact" match) or whether they are distinct even if thematically related.

Story A (new, being proposed):
Title: {a_title}
Description: {a_desc}

Story B (existing, already on the board or completed):
Title: {b_title}
Description: {b_desc}

RULES:
- "Near-exact" = same acceptance criteria, same files touched, same concrete outcome. Merging one onto main would satisfy the other.
- "Different" = any of: different files, different scope, different coverage target, different user, different acceptance criteria, different era/context.
- Shared topic or shared file is NOT enough - follow-up coverage lifts, refactors, and extensions are DIFFERENT.
- When in doubt, answer DIFFERENT. False-negatives (missing a true dup) are cheap; false-positives (killing real work) are expensive.

Respond with ONLY a JSON object:
{{"verdict": "SAME" or "DIFFERENT", "reason": "one sentence"}}
"""

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

_claude_binary_cache: Optional[str] = None


def _find_claude_binary() -> Optional[str]:
    """Locate the Claude Code binary.

    Mirrors the search in ``aiv/classifier_llm.py`` so every caller
    resolves the same binary. Result is cached process-wide.
    """
    global _claude_binary_cache
    if _claude_binary_cache:
        return _claude_binary_cache

    for name in ("claude", "claude.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if not d:
                continue
            candidate = Path(d) / name
            if candidate.is_file():
                _claude_binary_cache = str(candidate)
                return _claude_binary_cache

    ext_root = Path.home() / ".vscode" / "extensions"
    if ext_root.is_dir():
        for ext_dir in sorted(ext_root.glob("anthropic.claude-code-*"), reverse=True):
            for name in ("claude.exe", "claude"):
                candidate = ext_dir / "resources" / "native-binary" / name
                if candidate.is_file():
                    _claude_binary_cache = str(candidate)
                    return _claude_binary_cache

    return None


def _build_prompt(a_title: str, a_desc: str, b_title: str, b_desc: str) -> str:
    """Assemble the comparison prompt with per-field length caps.

    Caps protect against pathological inputs (e.g. a 50KB description
    pasted into an idea) blowing up the token budget on every dedup
    check. The caps are large enough that real stories pass through
    untouched.
    """
    return PROMPT_TEMPLATE.format(
        a_title=(a_title or "")[:MAX_TITLE_CHARS],
        a_desc=(a_desc or "")[:MAX_DESC_CHARS],
        b_title=(b_title or "")[:MAX_TITLE_CHARS],
        b_desc=(b_desc or "")[:MAX_DESC_CHARS],
    )


def _parse_verdict(raw: str) -> tuple[bool, str]:
    """Extract ``(is_duplicate, reason)`` from raw LLM stdout.

    Returns ``(False, "no_json")`` if no JSON object is present and
    ``(False, "parse_failure")`` if the JSON is malformed. Unknown
    verdict strings are treated as DIFFERENT.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False, "no_json"

    match = _JSON_BLOCK_RE.search(raw)
    if match is None:
        return False, "no_json"

    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return False, "parse_failure"

    if not isinstance(data, dict):
        return False, "parse_failure"

    verdict = str(data.get("verdict", "")).strip().upper()
    reason = str(data.get("reason", ""))[:300]
    return verdict == "SAME", reason


def is_near_exact_duplicate(
    a_title: str,
    a_desc: str,
    b_title: str,
    b_desc: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bool, str]:
    """Ask Haiku whether story A and story B are the same unit of work.

    Args:
        a_title: Title of the new (proposed) story.
        a_desc: Description of the new story.
        b_title: Title of the existing story.
        b_desc: Description of the existing story.
        timeout: Max seconds to wait on ``claude -p``.

    Returns:
        ``(True, reason)`` only when the LLM returns ``verdict="SAME"``.
        Every failure mode (missing binary, timeout, non-zero exit,
        malformed output, unknown verdict) returns
        ``(False, <error_code_or_reason>)`` so callers fall open and the
        new story is not blocked by an LLM outage.
    """
    from agent.llm_router import complete

    prompt = _build_prompt(a_title, a_desc, b_title, b_desc)
    raw = complete("dedup_judge", prompt, timeout=timeout)
    if raw is None:
        # Router already logged which layer failed. Fall open so a new
        # story isn't blocked by an LLM outage.
        return False, "llm_unavailable"
    return _parse_verdict(raw)
