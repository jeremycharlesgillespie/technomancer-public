"""Tests for agent/crash_triage.py — CrashWatcher."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.crash_triage import CrashWatcher, DEDUPE_DAYS


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


@pytest.fixture
def watcher(tmp_path):
    crash_log = tmp_path / "crash_log.md"
    state_dir = tmp_path / "state"
    return CrashWatcher(
        crash_log_path=crash_log,
        state_dir=state_dir,
        jira_endpoint="http://localhost:8322/api/jira/create",
    )


@pytest.fixture
def mock_post():
    with patch("agent.crash_triage.requests.post") as mock:
        resp = MagicMock()
        resp.status_code = 201
        mock.return_value = resp
        yield mock


class TestCheckOnceAppend:
    """Acceptance: one POST per unique fingerprint, zero on duplicate append."""

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

        # Append the same crash again — fingerprint matches, so no new POST.
        watcher.crash_log_path.write_text(crash + crash, encoding="utf-8")
        filed = watcher.check_once()

        assert filed == 0
        assert mock_post.call_count == 1

    def test_new_unique_crash_posts_once(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Append a different exception type — new fingerprint.
        second = _make_crash(exc_type="KeyError", exc_msg="'missing'")
        watcher.crash_log_path.write_text(
            _make_crash() + "\n" + second, encoding="utf-8"
        )
        filed = watcher.check_once()

        assert filed == 1
        assert mock_post.call_count == 2


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
        assert payload["source"] == "crash"
        assert payload["idea_type"] == "story"
        assert "Traceback" in payload["description"] or "traceback" in payload["description"].lower()

    def test_description_truncates_to_tail(self, watcher, mock_post):
        # Craft a traceback with >30 lines between the ```python fences.
        filler = "\n".join(f"  frame_line_{i}" for i in range(40))
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
        # frame_line_0 is far enough from the end to be dropped.
        assert "frame_line_0" not in payload["description"]
        assert "frame_line_39" in payload["description"]


class TestParseCrashes:
    def test_empty_file(self, watcher):
        assert watcher._parse_crashes("") == []

    def test_no_crash_marker(self, watcher):
        assert watcher._parse_crashes("just text without a marker") == []

    def test_parses_exception_fields(self, watcher):
        crashes = watcher._parse_crashes(_make_crash(exc_type="TypeError"))
        assert len(crashes) == 1
        assert crashes[0]["exception_type"] == "TypeError"
        assert crashes[0]["top_function"] == "on_message"


class TestFingerprint:
    def test_same_frames_same_fingerprint(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash())[0]
        crash2 = watcher._parse_crashes(
            _make_crash(exc_msg="different message but same type")
        )[0]
        assert watcher._compute_fingerprint(crash1) == watcher._compute_fingerprint(crash2)

    def test_different_exception_type_different_fingerprint(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash(exc_type="ValueError"))[0]
        crash2 = watcher._parse_crashes(_make_crash(exc_type="KeyError"))[0]
        assert watcher._compute_fingerprint(crash1) != watcher._compute_fingerprint(crash2)

    def test_different_function_different_fingerprint(self, watcher):
        crash1 = watcher._parse_crashes(_make_crash(func="on_message"))[0]
        crash2 = watcher._parse_crashes(_make_crash(func="on_ready"))[0]
        assert watcher._compute_fingerprint(crash1) != watcher._compute_fingerprint(crash2)

    def test_missing_exception_type_empty_fingerprint(self, watcher):
        assert watcher._compute_fingerprint({}) == ""


class TestStatePersistence:
    def test_position_advances(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        state = json.loads(watcher.state_path.read_text(encoding="utf-8"))
        size = len(watcher.crash_log_path.read_text(encoding="utf-8").encode("utf-8"))
        assert state["position"] == size

    def test_position_resets_when_file_truncated(self, watcher, mock_post):
        # Start with two different crashes so position advances past the first.
        big = _make_crash(exc_type="ValueError") + "\n" + _make_crash(exc_type="KeyError")
        watcher.crash_log_path.write_text(big, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 2

        # File gets overwritten with a smaller, BRAND-NEW crash.
        watcher.crash_log_path.write_text(
            _make_crash(exc_type="RuntimeError"), encoding="utf-8"
        )
        filed = watcher.check_once()

        # Offset reset to 0, but the RuntimeError fingerprint is new — one POST.
        assert filed == 1
        assert mock_post.call_count == 3

    def test_fingerprint_file_persists(self, watcher, mock_post):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")
        watcher.check_once()

        data = json.loads(watcher.fingerprints_path.read_text(encoding="utf-8"))
        assert len(data) == 1


class TestDedupExpiry:
    def test_prune_removes_old_fingerprints(self, watcher):
        old = (datetime.now() - timedelta(days=DEDUPE_DAYS + 1)).isoformat()
        fresh = datetime.now().isoformat()
        pruned = watcher._prune_fingerprints({"old_fp": old, "fresh_fp": fresh})
        assert "old_fp" not in pruned
        assert "fresh_fp" in pruned

    def test_expired_fingerprint_allows_repost(self, watcher, mock_post):
        crash = _make_crash()
        watcher.crash_log_path.write_text(crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 1

        # Backdate the saved fingerprint past the dedup window.
        data = json.loads(watcher.fingerprints_path.read_text(encoding="utf-8"))
        key = next(iter(data))
        data[key] = (datetime.now() - timedelta(days=DEDUPE_DAYS + 1)).isoformat()
        watcher.fingerprints_path.write_text(json.dumps(data), encoding="utf-8")

        # Same crash appended again — stale fingerprint was pruned, so it posts.
        watcher.crash_log_path.write_text(crash + crash, encoding="utf-8")
        watcher.check_once()
        assert mock_post.call_count == 2


class TestFailure:
    def test_missing_log_returns_zero(self, watcher, mock_post):
        assert not watcher.crash_log_path.exists()
        assert watcher.check_once() == 0
        assert mock_post.call_count == 0

    def test_jira_post_failure_does_not_cache_fingerprint(self, watcher):
        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            resp = MagicMock()
            resp.status_code = 500
            mock.return_value = resp
            filed = watcher.check_once()

        assert filed == 0
        # No fingerprint cached because the POST failed.
        data = json.loads(watcher.fingerprints_path.read_text(encoding="utf-8"))
        assert data == {}

    def test_jira_network_error_handled(self, watcher):
        import requests as req

        watcher.crash_log_path.write_text(_make_crash(), encoding="utf-8")

        with patch("agent.crash_triage.requests.post") as mock:
            mock.side_effect = req.ConnectionError("down")
            filed = watcher.check_once()

        assert filed == 0
