"""
Incidents Log Helper — append rows to docs/incidents.md.

The incidents log is a hand-editable markdown table at docs/incidents.md.
This module provides append_incident() for programmatic appends (from
automation or manual scripts) while preserving the table format.

Idempotent: a (date, what) pair that already exists is silently skipped.
"""

from __future__ import annotations

from pathlib import Path

# docs/incidents.md lives two levels above local-agent/
INCIDENTS_PATH: Path = Path(__file__).parent.parent.parent / "docs" / "incidents.md"

# Table header sentinel — used to locate where rows start
_TABLE_HEADER = "| Date | What broke | How detected | Fix commit |"
_TABLE_SEP = "|------|------------|--------------|------------|"


def append_incident(
    date: str,
    what: str,
    detected: str,
    fix_commit: str,
    path: Path | None = None,
) -> bool:
    """Append a row to the incidents markdown table.

    Parameters
    ----------
    date:       ISO date string, e.g. ``"2026-04-19"``
    what:       Short description of what broke
    detected:   How the issue was detected
    fix_commit: Commit hash, Jira key, or ``"manual"``
    path:       Override the incidents file path (for tests)

    Returns
    -------
    True if a new row was appended, False if the (date, what) pair already
    exists (idempotent no-op).
    """
    target = path or INCIDENTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    text = target.read_text(encoding="utf-8") if target.exists() else ""

    if _is_duplicate(text, date, what):
        return False

    row = f"| {date} | {what} | {detected} | {fix_commit} |"

    if _TABLE_HEADER in text:
        # Append after the last table row
        lines = text.splitlines(keepends=True)
        # Find last line that starts with "|"
        last_pipe = max(
            (i for i, ln in enumerate(lines) if ln.startswith("|")),
            default=None,
        )
        if last_pipe is not None:
            lines.insert(last_pipe + 1, row + "\n")
            target.write_text("".join(lines), encoding="utf-8")
            return True

    # No table found — create a minimal file
    content = (
        "# Incidents Log\n\n"
        f"{_TABLE_HEADER}\n"
        f"{_TABLE_SEP}\n"
        f"{row}\n"
    )
    target.write_text(content, encoding="utf-8")
    return True


def _is_duplicate(text: str, date: str, what: str) -> bool:
    """Return True if a row with this (date, what) pair already exists."""
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [c.strip() for c in line.split("|")]
        # parts[0] == '' (before first |), parts[1] == date, parts[2] == what
        if len(parts) >= 3 and parts[1] == date and parts[2] == what:
            return True
    return False
