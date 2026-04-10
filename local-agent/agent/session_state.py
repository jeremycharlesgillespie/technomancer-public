"""
Session State — Persist recent conversation exchanges across bot restarts.

Saves the last N user/bot message pairs to a JSON file after each response.
On startup, loads the saved state so the bot can resume with awareness of
what was just being discussed — bridging the gap between restarts.

Unlike conversation_context.py (which stores batched summaries), this
preserves the literal recent exchanges for immediate continuity.

File: data/session_state.json
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = STATE_DIR / "session_state.json"

MAX_EXCHANGES = 10  # Keep the last 10 user/bot exchanges


def save_exchange(user: str, user_message: str, bot_response: str) -> None:
    """Save a user/bot exchange to the session state file.

    Called after each response in on_message. Appends to the exchange
    list and trims to MAX_EXCHANGES.
    """
    try:
        state = _load_raw()
        state["exchanges"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": user,
            "message": user_message[:500],
            "response": bot_response[:500],
        })
        # Keep only the last N exchanges
        state["exchanges"] = state["exchanges"][-MAX_EXCHANGES:]
        state["last_active"] = datetime.now().isoformat(timespec="seconds")
        state["clean_shutdown"] = False  # Will be set to True on graceful stop
        _save_raw(state)
    except Exception as e:
        log.debug(f"[SessionState] Failed to save exchange: {e}")


def mark_clean_shutdown() -> None:
    """Mark the session as cleanly shut down (no crash recovery needed)."""
    try:
        state = _load_raw()
        state["clean_shutdown"] = True
        state["shutdown_time"] = datetime.now().isoformat(timespec="seconds")
        _save_raw(state)
    except Exception:
        pass


def get_previous_session() -> dict[str, Any]:
    """Load the previous session state for continuity on startup.

    Returns a dict with:
        exchanges: List of recent user/bot message pairs
        was_crash: True if the previous session ended abnormally
        last_active: ISO timestamp of last activity
        time_since_last: Seconds since last activity
    """
    state = _load_raw()
    exchanges = state.get("exchanges", [])
    clean = state.get("clean_shutdown", True)
    last_active = state.get("last_active", "")

    # Calculate time since last activity
    time_since = 0.0
    if last_active:
        try:
            last_dt = datetime.fromisoformat(last_active)
            time_since = (datetime.now() - last_dt).total_seconds()
        except (ValueError, TypeError):
            pass

    return {
        "exchanges": exchanges,
        "was_crash": not clean and len(exchanges) > 0,
        "last_active": last_active,
        "time_since_last": time_since,
    }


def format_session_context(session: dict[str, Any], max_chars: int = 3000) -> str:
    """Format the previous session's exchanges as context for injection.

    Returns a string suitable for appending to the context_parts list
    in the message processing pipeline. Returns empty string if no
    relevant previous session or if it's been too long (>6 hours).
    """
    exchanges = session.get("exchanges", [])
    if not exchanges:
        return ""

    # Don't inject stale context (>6 hours old)
    if session.get("time_since_last", 0) > 6 * 3600:
        return ""

    lines = ["Previous conversation (from earlier session):"]
    total_chars = len(lines[0])

    for ex in exchanges:
        line = f"  {ex['user']}: {ex['message']}\n  Bot: {ex['response']}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _load_raw() -> dict[str, Any]:
    """Load raw state from disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"exchanges": [], "clean_shutdown": True, "last_active": ""}


def _save_raw(state: dict[str, Any]) -> None:
    """Save raw state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
