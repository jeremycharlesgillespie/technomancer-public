"""
Crash Triage — Auto-file Jira stories when new crashes land in crash_log.md.

A ``CrashWatcher`` thread polls ``Permanent/crash_log.md`` every 60 seconds,
parses each crash report, fingerprints it, and POSTs unique crashes to the
Jira-creation endpoint (``/api/jira/create`` on the local idea-board hub).

Dedup is backed by SQLite at ``crash_triage_seen.db`` (``crash_triage_seen``
table: ``hash TEXT PRIMARY KEY, first_seen TIMESTAMP, jira_key TEXT``) with a
7-day TTL.  ``_hash_crash`` hashes the exception type plus the deepest three
stack frames (basename + function), so a retry of the same crash is silently
dropped but a genuinely new callstack for the same exception does produce a
story.

The Jira POST is best-effort: it has a 5-second timeout and every failure
path (network error, non-2xx response, unparsable body) returns ``None`` so
a temporarily unreachable hub never raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


DEFAULT_JIRA_ENDPOINT = "http://localhost:8322/api/jira/create"
DEFAULT_POLL_INTERVAL = 60
DEDUPE_DAYS = 7
MAX_DESCRIPTION_BYTES = 4096
JIRA_TIMEOUT_SECONDS = 5
TOP_FRAMES_FOR_HASH = 3

_FRAME_RE = re.compile(r'File "([^"]+)", line \d+, in (\S+)')
_TRUNCATED_SUFFIX = "\n... (truncated)"


def _init_seen_db(db_path: Path) -> None:
    """Create the crash_triage_seen table if absent. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crash_triage_seen (
                hash TEXT PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                jira_key TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


class CrashWatcher:
    """Polls crash_log.md and files a Jira story per unique crash fingerprint."""

    def __init__(
        self,
        crash_log_path: str | Path,
        state_dir: str | Path,
        jira_endpoint: str = DEFAULT_JIRA_ENDPOINT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        db_path: str | Path | None = None,
    ) -> None:
        self.crash_log_path = Path(crash_log_path)
        state_root = Path(state_dir)
        state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = state_root / "crash_watcher_state.json"
        self.db_path = (
            Path(db_path) if db_path else state_root / "crash_triage_seen.db"
        )
        _init_seen_db(self.db_path)
        self.jira_endpoint = jira_endpoint
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CrashWatcher"
        )
        self._thread.start()
        logger.info(
            "CrashWatcher started (poll=%ss, log=%s)",
            self.poll_interval,
            self.crash_log_path,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:
                logger.exception("CrashWatcher poll failed")
            self._stop.wait(self.poll_interval)

    # ------------------------------------------------------------------ core
    def check_once(self) -> int:
        """Scan the crash log once. Return the number of Jira stories filed."""
        if not self.crash_log_path.exists():
            return 0

        try:
            content = self.crash_log_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read crash log at %s", self.crash_log_path)
            return 0

        encoded = content.encode("utf-8")
        file_size = len(encoded)

        state = self._load_state()
        last_position = int(state.get("position", 0))
        if file_size < last_position:
            last_position = 0

        new_content = encoded[last_position:].decode("utf-8", errors="replace")
        crashes = self._parse_crashes(new_content)

        filed = 0
        for crash in crashes:
            h = self._hash_crash(crash)
            if not h or self._seen_recently(h):
                continue
            jira_key = self._create_jira_for_crash(crash)
            if jira_key is None:
                continue
            self._mark_seen(h, jira_key)
            filed += 1

        self._save_state(
            {"position": file_size, "updated_at": datetime.now().isoformat()}
        )
        return filed

    # ------------------------------------------------------------------ parsing
    def _parse_crashes(self, text: str) -> list[dict[str, Any]]:
        """Split text by '# Bot Crash Report' and extract fields from each."""
        if not text or "# Bot Crash Report" not in text:
            return []

        parts = text.split("# Bot Crash Report")
        crashes: list[dict[str, Any]] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            crash: dict[str, Any] = {}
            for line in part.splitlines():
                if line.startswith("**Exception Type:**"):
                    crash["exception_type"] = line.replace(
                        "**Exception Type:**", ""
                    ).strip()
                elif line.startswith("**Exception Message:**"):
                    crash["exception_message"] = line.replace(
                        "**Exception Message:**", ""
                    ).strip()
                elif line.startswith("**Timestamp:**"):
                    crash["timestamp"] = line.replace("**Timestamp:**", "").strip()

            trace_lines = _extract_traceback_block(part)
            crash["traceback"] = "\n".join(trace_lines)

            frames: list[tuple[str, str]] = []
            for line in trace_lines:
                m = _FRAME_RE.search(line)
                if m:
                    frames.append((m.group(1), m.group(2)))
            crash["frames"] = frames

            crash["locals_block"] = _extract_locals_block(part)

            if not crash.get("exception_type"):
                continue
            crashes.append(crash)
        return crashes

    def _hash_crash(self, crash: dict[str, Any]) -> str:
        """Hash (exception_type, basename+function of top 3 frames)."""
        exc_type = crash.get("exception_type", "")
        if not exc_type:
            return ""
        frames: list[tuple[str, str]] = list(crash.get("frames", []))
        top = frames[-TOP_FRAMES_FOR_HASH:]
        parts: list[str] = [exc_type]
        for fname, func in top:
            parts.append(f"{Path(fname).name}:{func}")
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ dedup
    def _seen_recently(self, h: str, days: int = DEDUPE_DAYS) -> bool:
        """Return True if this hash was recorded within the last ``days``."""
        if not h:
            return False
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        try:
            row = conn.execute(
                "SELECT 1 FROM crash_triage_seen "
                "WHERE hash = ? AND first_seen > ?",
                (h, cutoff),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _mark_seen(self, h: str, jira_key: str | None) -> None:
        if not h:
            return
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO crash_triage_seen "
                "(hash, first_seen, jira_key) VALUES (?, ?, ?)",
                (h, datetime.now().isoformat(), jira_key),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ Jira
    def _build_story_payload(self, crash: dict[str, Any]) -> dict[str, Any]:
        exc_type = crash.get("exception_type", "UnknownError")
        frames: list[tuple[str, str]] = list(crash.get("frames", []))
        if frames:
            filename, func = frames[-1]
        else:
            filename, func = "", "unknown"
        module = Path(filename).stem if filename else "unknown"
        title = f"Crash: {exc_type} in {module}.{func}"

        exc_msg = crash.get("exception_message", "")
        tb = crash.get("traceback", "")
        locals_block = crash.get("locals_block", "")

        desc = (
            f"**Exception:** {exc_type}: {exc_msg}\n\n"
            f"**Traceback:**\n\n```\n{tb}\n```\n"
        )
        if locals_block:
            desc += f"\n**Local Variables:**\n\n```\n{locals_block}\n```\n"

        desc = _truncate_bytes(desc, MAX_DESCRIPTION_BYTES)

        return {
            "title": title,
            "description": desc,
            "category": "quality",
            "source": "crash_triage",
            "idea_type": "story",
        }

    def _create_jira_for_crash(self, crash: dict[str, Any]) -> str | None:
        """POST the crash to Jira. Returns the Jira key on success, ``None``
        on any failure (network error, non-2xx status, unparsable body).

        The 5-second timeout and blanket exception swallowing are intentional:
        the idea-board hub may be temporarily unreachable, and a crash
        watcher that raises during its poll would hide the original crash it
        was trying to file.
        """
        payload = self._build_story_payload(crash)
        try:
            resp = requests.post(
                self.jira_endpoint,
                json=payload,
                timeout=JIRA_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("crash_triage: Jira POST errored: %s", exc)
            return None

        if resp.status_code not in (200, 201):
            logger.warning(
                "crash_triage: Jira POST returned %s for %s",
                resp.status_code,
                payload["title"],
            )
            return None

        try:
            body = resp.json() or {}
        except Exception:
            return None
        key = body.get("key")
        return str(key) if key else None

    # ------------------------------------------------------------------ persistence
    def _load_state(self) -> dict[str, Any]:
        return _read_json(self.state_path)

    def _save_state(self, state: dict[str, Any]) -> None:
        _write_json(self.state_path, state)


def _extract_traceback_block(part: str) -> list[str]:
    """Pull the lines between ```python and ``` under '## Full Stack Trace'."""
    in_section = False
    in_code = False
    out: list[str] = []
    for line in part.splitlines():
        if "## Full Stack Trace" in line:
            in_section = True
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                continue
            break
        if in_code:
            out.append(line)
    return out


def _extract_locals_block(part: str) -> str:
    """Return the text of the '## Local Variables by Frame' section, if any.

    The section runs from its heading to the end of the crash report (i.e.
    the end of the buffer, since ``_parse_crashes`` already split on
    '# Bot Crash Report').
    """
    marker = "## Local Variables by Frame"
    idx = part.find(marker)
    if idx == -1:
        return ""
    return part[idx + len(marker):].strip()


def _truncate_bytes(text: str, limit: int) -> str:
    """Truncate ``text`` so its UTF-8 length is at most ``limit`` bytes.

    Appends ``_TRUNCATED_SUFFIX`` when truncation occurs. The suffix itself
    is counted against the byte budget so the total length stays within
    ``limit``.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    suffix_bytes = _TRUNCATED_SUFFIX.encode("utf-8")
    budget = max(0, limit - len(suffix_bytes))
    head = encoded[:budget].decode("utf-8", errors="ignore")
    return head + _TRUNCATED_SUFFIX


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning("Could not write %s", path)


# ---------------------------------------------------------------------------
# Module-level helper for discord_memory_bot startup
# ---------------------------------------------------------------------------

_watcher: CrashWatcher | None = None


def start_crash_watcher(
    vault_path: str | Path,
    state_dir: str | Path,
    jira_endpoint: str = DEFAULT_JIRA_ENDPOINT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> CrashWatcher:
    """Start the singleton CrashWatcher thread. Safe to call more than once."""
    global _watcher
    crash_log = Path(vault_path) / "LLM Memory" / "Permanent" / "crash_log.md"
    if _watcher is None:
        _watcher = CrashWatcher(
            crash_log_path=crash_log,
            state_dir=state_dir,
            jira_endpoint=jira_endpoint,
            poll_interval=poll_interval,
        )
    _watcher.start()
    return _watcher


def get_crash_watcher() -> CrashWatcher | None:
    return _watcher
