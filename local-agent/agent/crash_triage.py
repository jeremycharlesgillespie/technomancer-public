"""
Crash Triage — Auto-file Jira stories when new crashes land in crash_log.md.

A ``CrashWatcher`` thread polls ``Permanent/crash_log.md`` every 60 seconds,
parses each crash report, and POSTs unique crashes to the Jira-creation
endpoint (``/api/jira/create`` on the local idea-board hub).

Dedup is two-layered:

1. **Byte position** — ``crash_watcher_state.json`` stores the offset already
   scanned so a second poll skips content seen before. If the file shrinks
   (``write_crash_log`` overwrites instead of appending), the offset resets
   to 0 and the fingerprint cache catches anything already reported.
2. **Fingerprint cache** — ``crash_fingerprints.json`` keeps an MD5 over
   ``basename(filename) | function | exception_type`` for every crash filed
   in the last 30 days. This is the real dedup guarantee.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


DEFAULT_JIRA_ENDPOINT = "http://localhost:8322/api/jira/create"
DEFAULT_POLL_INTERVAL = 60
DEDUPE_DAYS = 30
TRACEBACK_TAIL_LINES = 30

_FRAME_RE = re.compile(r'File "([^"]+)", line \d+, in (\S+)')


class CrashWatcher:
    """Polls crash_log.md and files a Jira story per unique crash fingerprint."""

    def __init__(
        self,
        crash_log_path: str | Path,
        state_dir: str | Path,
        jira_endpoint: str = DEFAULT_JIRA_ENDPOINT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.crash_log_path = Path(crash_log_path)
        state_root = Path(state_dir)
        state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = state_root / "crash_watcher_state.json"
        self.fingerprints_path = state_root / "crash_fingerprints.json"
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

        new_bytes = encoded[last_position:]
        new_content = new_bytes.decode("utf-8", errors="replace")

        crashes = self._parse_crashes(new_content)
        fingerprints = self._prune_fingerprints(self._load_fingerprints())

        filed = 0
        now_iso = datetime.now().isoformat()
        for crash in crashes:
            fp = self._compute_fingerprint(crash)
            if not fp or fp in fingerprints:
                continue
            if self._post_to_jira(crash):
                fingerprints[fp] = now_iso
                filed += 1

        self._save_fingerprints(fingerprints)
        self._save_state({"position": file_size, "updated_at": now_iso})
        return filed

    # ------------------------------------------------------------------ parsing
    def _parse_crashes(self, text: str) -> list[dict[str, str]]:
        """Split text by '# Bot Crash Report' and extract fields from each."""
        if not text or "# Bot Crash Report" not in text:
            return []

        parts = text.split("# Bot Crash Report")
        crashes: list[dict[str, str]] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            crash: dict[str, str] = {}
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

            deepest = None
            for line in trace_lines:
                m = _FRAME_RE.search(line)
                if m:
                    deepest = (m.group(1), m.group(2))
            if deepest is not None:
                crash["top_filename"] = deepest[0]
                crash["top_function"] = deepest[1]

            if not crash.get("exception_type"):
                continue
            crashes.append(crash)
        return crashes

    def _compute_fingerprint(self, crash: dict[str, str]) -> str:
        exc_type = crash.get("exception_type", "")
        if not exc_type:
            return ""
        filename = crash.get("top_filename", "")
        func = crash.get("top_function", "")
        basename = Path(filename).name if filename else ""
        key = f"{basename}|{func}|{exc_type}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ Jira
    def _build_story_payload(self, crash: dict[str, str]) -> dict[str, Any]:
        exc_type = crash.get("exception_type", "UnknownError")
        filename = crash.get("top_filename", "")
        func = crash.get("top_function", "unknown")
        module = Path(filename).stem if filename else "unknown"
        title = f"Crash: {exc_type} in {module}.{func}"

        tb_lines = crash.get("traceback", "").splitlines()
        tail = tb_lines[-TRACEBACK_TAIL_LINES:]
        exc_msg = crash.get("exception_message", "")
        description = (
            f"**Exception:** {exc_type}: {exc_msg}\n\n"
            f"**Traceback (last {TRACEBACK_TAIL_LINES} lines):**\n\n"
            "```\n" + "\n".join(tail) + "\n```\n"
        )
        return {
            "title": title,
            "description": description,
            "category": "quality",
            "source": "crash",
            "idea_type": "story",
        }

    def _post_to_jira(self, crash: dict[str, str]) -> bool:
        payload = self._build_story_payload(crash)
        try:
            resp = requests.post(self.jira_endpoint, json=payload, timeout=10)
        except requests.RequestException as exc:
            logger.warning("Jira POST failed: %s", exc)
            return False
        if resp.status_code in (200, 201):
            logger.info("Filed crash story: %s", payload["title"])
            return True
        logger.warning(
            "Jira POST returned %s for %s", resp.status_code, payload["title"]
        )
        return False

    # ------------------------------------------------------------------ persistence
    def _load_state(self) -> dict[str, Any]:
        return _read_json(self.state_path)

    def _save_state(self, state: dict[str, Any]) -> None:
        _write_json(self.state_path, state)

    def _load_fingerprints(self) -> dict[str, str]:
        data = _read_json(self.fingerprints_path)
        return {k: str(v) for k, v in data.items()} if data else {}

    def _save_fingerprints(self, fp: dict[str, str]) -> None:
        _write_json(self.fingerprints_path, fp)

    def _prune_fingerprints(self, fp: dict[str, str]) -> dict[str, str]:
        cutoff = datetime.now() - timedelta(days=DEDUPE_DAYS)
        result: dict[str, str] = {}
        for key, ts in fp.items():
            try:
                if datetime.fromisoformat(ts) > cutoff:
                    result[key] = ts
            except (TypeError, ValueError):
                continue
        return result


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
