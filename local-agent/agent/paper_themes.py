"""Paper Themes Loader — parse ``docs/paper_themes.md`` into theme dicts.

The AIMM curator + crafter read this file fresh every cycle, so operators can
edit priorities without a redeploy. The loader is deliberately tolerant:
malformed theme entries log a warning and are skipped rather than aborting
the whole cycle, so one bad edit can't wedge the scheduler.

Each returned theme is ``{"name": str, "target_per_week": int}``. Sections
without a ``target_per_week`` bullet (e.g. the "Editing notes" appendix at
the bottom of the file) are not themes and are silently ignored.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
THEMES_PATH: Path = PROJECT_ROOT / "docs" / "paper_themes.md"

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TARGET_RE = re.compile(
    r"^\s*-\s*target_per_week\s*:\s*(.*?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def load_paper_themes(path: Optional[Path] = None) -> list[dict]:
    """Load paper themes from ``docs/paper_themes.md``.

    Args:
        path: Override path to the themes file (for tests). Defaults to
            ``THEMES_PATH``.

    Returns:
        A list of ``{"name": str, "target_per_week": int}`` dicts in file
        order. Sections without a ``target_per_week`` bullet are skipped.

    Raises:
        ValueError: The themes file does not exist. Raised as ValueError
            (not FileNotFoundError) so callers can treat it as a config
            error rather than an unexpected OSError.
    """
    themes_path = Path(path) if path is not None else THEMES_PATH
    if not themes_path.exists():
        raise ValueError(f"paper_themes.md not found at {themes_path}")

    text = themes_path.read_text(encoding="utf-8")
    themes: list[dict] = []

    heading_matches = list(_HEADING_RE.finditer(text))
    for i, match in enumerate(heading_matches):
        name = match.group(1).strip()
        if not name:
            logger.warning("paper_themes.md: empty heading at position %d; skipping", match.start())
            continue
        section_start = match.end()
        section_end = (
            heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        )
        section = text[section_start:section_end]

        target_match = _TARGET_RE.search(section)
        if target_match is None:
            # No target_per_week bullet — not a theme section (e.g. "Editing notes").
            continue

        raw_value = target_match.group(1).strip()
        if not raw_value:
            logger.warning(
                "paper_themes.md: theme %r has empty target_per_week; skipping", name
            )
            continue
        try:
            target = int(raw_value)
        except ValueError:
            logger.warning(
                "paper_themes.md: theme %r has non-integer target_per_week %r; skipping",
                name,
                raw_value,
            )
            continue
        if target < 0:
            logger.warning(
                "paper_themes.md: theme %r has negative target_per_week %d; skipping",
                name,
                target,
            )
            continue

        themes.append({"name": name, "target_per_week": target})

    return themes
