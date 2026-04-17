"""Tests for agent/crash_triage.py — CrashWatcher."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import requests as _requests_lib

from agent.crash_triage import (
    CrashWatcher,
    DEDUPE_DAYS,
    JIRA_TIMEOUT_SECONDS,
    MAX_DESCRIPTION_BYTES,
    SIGNATURE_DEDUP_HOURS,
    _signature_crash,
    _truncate_bytes,
)


CRASH_TEMPLATE = """# Bot Crash Report

**Timestamp:** {timestamp}
**Exception Type:** {exc_type}
**Exception Message:** {exc_msg}

## Full Stack Trace
```python
Traceback (most recent call last):
  File "{filename}", line 42, in {func}
    do_something()
{exc_type}: {exc_msg}
```

## Local Variables by Frame

### Frame 0: {func} ({filename}:42)
```python
x = 1
```
"""


def _make_crash(
    exc_type: str = "UnboundLocalError",
    exc_msg: str = "cannot access local variable 'foo'",
    filename: str = r"C:\path\to\agent\discord_memory_bot.py",
    func: str = "on_message",
    timestamp: str = "2026-04-16 10:00:00",
) -> str:
    return CRASH_TEMPLATE.format(
        timestamp=timestamp,
        exc_type=exc_type,
        exc_msg=exc_msg,
        filename=filename.replace("\\", "\\\\"),
        func=func,
    )


def _make_crash_multi_frame(exc_type: str = "ValueError") -> str:
    """Crash report with three stack frames for top-3-frame hash testing."""
    return f"""# Bot Crash Report

**Timestamp:** 2026-04-16 10:00:00
**Exception Type:** {exc_type}
**Exception Message:** oops

## Full Stack Trace
```python
Traceback (most recent call last):
  File "a.py", line 1, in outer
    middle()
  File "b.py", line 2, in middle
    inner()
  File "c.py", line 3, in inner
    boom()
{exc_type}: oops
```
"""


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    # Isolate executor_runs_db so the new crash_signatures table writes
    # don't touch the production data/executor_runs.db.
    from agent import executor_runs_db as erd
    monkeypatch.setattr(erd, "DB_DIR", tmp_path)
    monkeypatch.setattr(erd, "DB_PATH", tmp_path / "executor_runs.db")
    erd._local.__dict__.pop("conn", None)

    crash_log = tmp_path / "crash_log.md"
    state_dir = tmp_path / "state"
    yield CrashWatcher(
        crash_log_path=crash_log,
        state_dir=state_dir,
        jira_endpoint="http://localhost:8322/api/jira/create",
    )
    conn = getattr(erd._local, "conn", None)
    if conn:
        conn.close()
        erd._local.conn = None


@pytest.fixture
def mock_post():
    with patch("agent.crash_triage.requests.post") as mock:
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {"key": "TK-999"}
        mock.return_value = resp
        yield mock


# ---------------------------------------------------------------------------
# check_once — dedup + POST behavior
# ---------------------------------------------------------------------------


class TestCheckOnce:
    def test_single_new_crash_posts_once(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        filed = watcher.check_once()

        assert filed == 1
        assert mock_post.call_count == 1

    def test_duplicate_append_does_not_post_again(self, watcher, mock_post):
        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Append the same crash again — hash matches, SQLite dedup blocks it.
        watcher.crash_log_path.write_text(crash + crash, encoding="utf-8")
        filed = watcher.check_once()

        assert filed == 0
        assert mock_post.call_count == 1

    def test_new_unique_crash_posts_once(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Different exception type → new hash.
        second = _make_crash(exc_type="KeyError", exc_msg="'missing'")
        watcher.crash_log_path.write_text(
            _make_crash() + "\n" + second, encoding="utf-8"
        )
        filed = watcher.check_once()

        assert filed == 1
        assert mock_post.call_count == 2

    def test_missing_log_returns_zero(self, watcher, mock_post):
        assert not watcher.crash_log_path.exists()
        assert watcher.check_once() == 0
        assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Payload fields
# ---------------------------------------------------------------------------


class TestPayload:
    def test_title_includes_module_and_function(self, watcher, mock_post):
        watcher.crash_log_path.write_text(
            _make_crash(
                exc_type="ValueError",
                filename="/home/app/agent/core.py",
                func="run",
            ),
            encoding="utf-8",
        )
        watcher.check_once()

        payload = mock_post.call_args.kwargs["json"]
        assert payload["title"] == "Crash: ValueError in core.run"

    def test_payload_has_required_fields(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        payload = mock_post.call_args.kwargs["json"]
        assert payload["category"] == "quality"
        assert payload["source"] == "crash_triage"
        assert payload["idea_type"] == "story"
        assert "Traceback" in payload["description"]

    def test_description_includes_local_variables(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        payload = mock_post.call_args.kwargs["json"]
        assert "Local Variables" in payload["description"]
        assert "x = 1" in payload["description"]

    def test_description_truncates_to_4kb(self, watcher, mock_post):
        # Build a traceback whose combined description exceeds 4 KB.
        filler = "\n".join(f"  frame_line_{i}" for i in range(500))
        tb = (
            "```python\n"
            "Traceback (most recent call last):\n"
            + filler + "\n"
            + '  File "x.py", line 1, in foo\n'
            + "ValueError: oops\n"
            "```"
        )
        part = (
            "# Bot Crash Report\n\n"
            "**Timestamp:** 2026-04-16 10:00:00\n"
            "**Exception Type:** ValueError\n"
            "**Exception Message:** oops\n\n"
            "## Full Stack Trace\n" + tb + "\n"
        )
        watcher.crash_log_path.write_text(part, encoding="utf-8")
        watcher.check_once()

        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["description"].encode("utf-8")) <= MAX_DESCRIPTION_BYTES
        assert "truncated" in payload["description"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseCrashes:
    def test_empty_file(self, watcher):
        assert watcher._parse_crashes("") == []

    def test_no_crash_marker(self, watcher):
        assert watcher._parse_crashes("just text without a marker") == []

    def test_parses_exception_fields(self, watcher):
        crashes = watcher._parse_crashes(_make_crash(exc_type="TypeError"))
        assert len(crashes) == 1
        assert crashes[0]["exception_type"] == "TypeError"
        # Single-frame crash → frames list has one (filename, function) tuple
        # where function is "on_message".  The filename's exact form varies
        # with the template's backslash-escaping; we only assert the function.
        frames = crashes[0]["frames"]
        assert len(frames) == 1
        assert frames[0][1] == "on_message"

    def test_parses_multi_frame(self, watcher):
        crashes = watcher._parse_crashes(_make_crash_multi_frame())
        assert len(crashes) == 1
        frames = crashes[0]["frames"]
        assert frames == [("a.py", "outer"), ("b.py", "middle"), ("c.py", "inner")]


# ---------------------------------------------------------------------------
# Hashing — exception_type + top 3 frames
# ---------------------------------------------------------------------------


class TestHashCrash:
    def test_same_frames_same_hash(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash())[0]
        crash2 = watcher._parse_crashes(
            _make_crash(exc_msg="different message but same type")
        )[0]
        assert watcher._hash_crash(crash1) == watcher._hash_crash(crash2)

    def test_different_exception_type_different_hash(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash(exc_type="ValueError"))[0]
        crash2 = watcher._parse_crashes(_make_crash(exc_type="KeyError"))[0]
        assert watcher._hash_crash(crash1) != watcher._hash_crash(crash2)

    def test_different_function_different_hash(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash(func="on_message"))[0]
        crash2 = watcher._parse_crashes(_make_crash(func="on_ready"))[0]
        assert watcher._hash_crash(crash1) != watcher._hash_crash(crash2)

    def test_missing_exception_type_empty_hash(self, watcher):
        assert watcher._hash_crash({}) == ""

    def test_hash_uses_top_3_frames(self, watcher):
        """Frames beyond the deepest 3 must not affect the hash."""
        deep = """# Bot Crash Report

**Timestamp:** 2026-04-16 10:00:00
**Exception Type:** ValueError
**Exception Message:** oops

## Full Stack Trace
```python
Traceback (most recent call last):
  File "unrelated.py", line 1, in z1
    a()
  File "different.py", line 2, in z2
    b()
  File "a.py", line 1, in outer
    middle()
  File "b.py", line 2, in middle
    inner()
  File "c.py", line 3, in inner
    boom()
ValueError: oops
```
"""
        crash_deep = watcher._parse_crashes(deep)[0]
        crash_top3 = watcher._parse_crashes(_make_crash_multi_frame())[0]
        # Both have the same deepest 3 frames (outer/middle/inner).
        assert watcher._hash_crash(crash_deep) == watcher._hash_crash(crash_top3)


# ---------------------------------------------------------------------------
# SQLite dedup table
# ---------------------------------------------------------------------------


class TestSeenTable:
    def test_table_created_on_init(self, watcher):
        conn = sqlite3.connect(str(watcher.db_path))
        try:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(crash_triage_seen)"
            )}
        finally:
            conn.close()
        assert cols == {"hash", "first_seen", "jira_key"}

    def test_successful_post_persists_row(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        conn = sqlite3.connect(str(watcher.db_path))
        try:
            rows = conn.execute(
                "SELECT hash, jira_key FROM crash_triage_seen"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "TK-999"

    def test_seen_recently_within_window(self, watcher):
        watcher._mark_seen("abc", "TK-1")
        assert watcher._seen_recently("abc") is True

    def test_seen_recently_outside_window(self, watcher):
        # Insert with a first_seen past the 7-day window.
        conn = sqlite3.connect(str(watcher.db_path))
        try:
            old = (datetime.now() - timedelta(days=DEDUPE_DAYS + 1)).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO crash_triage_seen "
                "(hash, first_seen, jira_key) VALUES (?, ?, ?)",
                ("stale", old, "TK-OLD"),
            )
            conn.commit()
        finally:
            conn.close()

        assert watcher._seen_recently("stale") is False

    def test_seen_recently_empty_hash_returns_false(self, watcher):
        assert watcher._seen_recently("") is False

    def test_expired_hash_allows_repost(self, watcher, mock_post):
        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Backdate both dedup layers: the legacy crash_triage_seen (7-day
        # MD5 window) AND the newer crash_signatures (24-hour SHA1 window)
        # in the executor_runs DB. A stale row in either alone would be
        # suppressed by the other, so both must age out.
        old = (datetime.now() - timedelta(days=DEDUPE_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(watcher.db_path))
        try:
            conn.execute(
                "UPDATE crash_triage_seen SET first_seen = ?",
                (old,),
            )
            conn.commit()
        finally:
            conn.close()

        from agent import executor_runs_db as erd
        sig_conn = sqlite3.connect(str(erd.DB_PATH))
        try:
            sig_conn.execute(
                "UPDATE crash_signatures SET last_seen = ?",
                (old,),
            )
            sig_conn.commit()
        finally:
            sig_conn.close()
        erd._local.__dict__.pop("conn", None)

        # Same crash appended again — stale row, should re-post.
        watcher.crash_log_path.write_text(crash + crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# State file (byte-position tracking)
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_position_advances(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        state = json.loads(watcher.state_path.read_text(encoding="utf-8"))
        size = len(watcher.crash_log_path.read_text(encoding="utf-8").encode("utf-8"))
        assert state["position"] == size

    def test_position_resets_when_file_truncated(self, watcher, mock_post):
        big = _make_crash(exc_type="ValueError") + "\n" + _make_crash(
            exc_type="KeyError"
        )
        watcher.crash_log_path.write_text(big, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 2

        # File shrinks to a brand-new crash → byte offset resets, new hash posts.
        watcher.crash_log_path.write_text(
            _make_crash(exc_type="RuntimeError"), encoding="utf-8"
        )
        filed = watcher.check_once()

        assert filed == 1
        assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# _create_jira_for_crash — network failure modes must be silent
# ---------------------------------------------------------------------------


class TestCreateJiraForCrash:
    def test_jira_post_failure_does_not_cache(self, watcher):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            resp = MagicMock()
            resp.status_code = 500
            mock.return_value = resp
            filed = watcher.check_once()

        assert filed == 0
        conn = sqlite3.connect(str(watcher.db_path))
        try:
            rows = conn.execute("SELECT COUNT(*) FROM crash_triage_seen").fetchone()
        finally:
            conn.close()
        assert rows[0] == 0

    def test_jira_network_error_handled_silently(self, watcher):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            mock.side_effect = _requests_lib.ConnectionError("down")
            # The whole thing must not raise.
            filed = watcher.check_once()

        assert filed == 0

    def test_jira_timeout_handled_silently(self, watcher):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            mock.side_effect = _requests_lib.Timeout("slow")
            filed = watcher.check_once()

        assert filed == 0

    def test_jira_post_uses_5_second_timeout(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        assert mock_post.call_args.kwargs["timeout"] == JIRA_TIMEOUT_SECONDS
        assert JIRA_TIMEOUT_SECONDS == 5

    def test_unparsable_body_returns_none_but_does_not_raise(self, watcher):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            resp = MagicMock()
            resp.status_code = 201
            resp.json.side_effect = ValueError("not json")
            mock.return_value = resp

            filed = watcher.check_once()

        # No key in body → not cached, not counted as filed.
        assert filed == 0
        conn = sqlite3.connect(str(watcher.db_path))
        try:
            rows = conn.execute("SELECT COUNT(*) FROM crash_triage_seen").fetchone()
        finally:
            conn.close()
        assert rows[0] == 0


# ---------------------------------------------------------------------------
# _truncate_bytes helper
# ---------------------------------------------------------------------------


class TestSignatureDedup:
    """24-hour signature dedup against crash_signatures in executor_runs DB."""

    def test_signature_is_sha1_of_exc_type_and_top_frames(self, watcher):
        """Signature is sha1 over exception_type + top-3 file:function tuples
        (basename only, line numbers excluded)."""
        import hashlib

        crash = watcher._parse_crashes(_make_crash_multi_frame())[0]
        sig = _signature_crash(crash)

        expected = hashlib.sha1(
            "ValueError|a.py:outer|b.py:middle|c.py:inner".encode("utf-8")
        ).hexdigest()
        assert sig == expected

    def test_second_call_within_24h_suppresses_post(self, watcher, mock_post):
        """Same synthetic traceback twice within 24h → Jira POST exactly once."""
        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Reset position so parsing runs again, and wipe the legacy MD5 dedup
        # so the signature layer is the sole gatekeeper for the second call.
        watcher.state_path.write_text('{"position": 0}', encoding="utf-8")
        db_conn = sqlite3.connect(str(watcher.db_path))
        try:
            db_conn.execute("DELETE FROM crash_triage_seen")
            db_conn.commit()
        finally:
            db_conn.close()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()

        assert mock_post.call_count == 1

    def test_stale_signature_after_25h_reposts(self, watcher, mock_post):
        """After backdating last_seen to 25h ago, a second POST fires."""
        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Backdate the crash_signatures.last_seen past the 24h window.
        from agent import executor_runs_db as erd

        stale = (datetime.now() - timedelta(hours=25)).isoformat()
        conn = sqlite3.connect(str(erd.DB_PATH))
        try:
            conn.execute(
                "UPDATE crash_signatures SET last_seen = ?", (stale,)
            )
            conn.commit()
        finally:
            conn.close()
        erd._local.__dict__.pop("conn", None)

        # Clear the legacy 7-day dedup too so the signature layer is what
        # controls whether the second POST fires.
        old = (datetime.now() - timedelta(days=DEDUPE_DAYS + 1)).isoformat()
        db_conn = sqlite3.connect(str(watcher.db_path))
        try:
            db_conn.execute(
                "UPDATE crash_triage_seen SET first_seen = ?", (old,)
            )
            db_conn.commit()
        finally:
            db_conn.close()

        watcher.state_path.write_text('{"position": 0}', encoding="utf-8")
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()

        assert mock_post.call_count == 2

    def test_dedup_hit_bumps_count_and_last_seen(self, watcher, mock_post):
        """A dedup-suppressed call must still increment count + last_seen."""
        from agent import executor_runs_db as erd

        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()

        # Reset position + clear legacy MD5 dedup so the signature layer
        # is what suppresses the second POST.
        watcher.state_path.write_text('{"position": 0}', encoding="utf-8")
        md5_conn = sqlite3.connect(str(watcher.db_path))
        try:
            md5_conn.execute("DELETE FROM crash_triage_seen")
            md5_conn.commit()
        finally:
            md5_conn.close()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()

        conn = sqlite3.connect(str(erd.DB_PATH))
        try:
            row = conn.execute(
                "SELECT count, jira_key FROM crash_signatures"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 2
        # jira_key preserved on dedup hit (not overwritten).
        assert row[1] == "TK-999"

    def test_empty_signature_skips_check(self, watcher):
        """Crashes without exception_type produce empty signature → no dedup."""
        assert _signature_crash({}) == ""

    def test_dedup_hit_logs_sig_and_count(self, watcher, mock_post, caplog):
        """Dedup-suppressed call logs `crash_triage: dedup hit sig=... count=N jira=...`."""
        import logging as _logging

        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()

        watcher.state_path.write_text('{"position": 0}', encoding="utf-8")
        md5_conn = sqlite3.connect(str(watcher.db_path))
        try:
            md5_conn.execute("DELETE FROM crash_triage_seen")
            md5_conn.commit()
        finally:
            md5_conn.close()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        with caplog.at_level(_logging.INFO, logger="agent.crash_triage"):
            watcher.check_once()

        assert any(
            "crash_triage: dedup hit sig=" in rec.message
            and "count=2" in rec.message
            and "jira=TK-999" in rec.message
            for rec in caplog.records
        )

    def test_constant_is_24_hours(self):
        assert SIGNATURE_DEDUP_HOURS == 24


class TestTruncateBytes:
    def test_short_text_unchanged(self):
        assert _truncate_bytes("hi", 100) == "hi"

    def test_long_text_truncated_and_marked(self):
        long = "x" * 10_000
        out = _truncate_bytes(long, MAX_DESCRIPTION_BYTES)
        assert len(out.encode("utf-8")) <= MAX_DESCRIPTION_BYTES
        assert out.endswith("(truncated)")

    def test_truncation_budget_includes_suffix(self):
        """The encoded output — head + suffix — must fit within the limit."""
        out = _truncate_bytes("a" * 5000, 100)
        assert len(out.encode("utf-8")) <= 100
