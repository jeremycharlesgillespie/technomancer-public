"""Tests for aim.event_log — append-only JSONL event stream."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from aim import event_log


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    """Redirect the event log to a temp directory for every test."""
    monkeypatch.setattr(event_log, "LOG_DIR", tmp_path)
    monkeypatch.setattr(event_log, "LOG_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(event_log, "BACKUP_FILE", tmp_path / "events.1.jsonl")


# ---------------------------------------------------------------------------
# append_event + read_events roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_append_then_read_single_event(self):
        event_log.append_event("worker_started", {"pid": 1234})

        events = event_log.read_events()

        assert len(events) == 1
        assert events[0]["type"] == "worker_started"
        assert events[0]["data"] == {"pid": 1234}
        assert "timestamp" in events[0]

    def test_timestamp_is_iso_format(self):
        event_log.append_event("test", {})

        ts = event_log.read_events()[0]["timestamp"]

        # Must parse as ISO 8601
        datetime.fromisoformat(ts)

    def test_data_preserves_nested_structures(self):
        nested = {"list": [1, 2, 3], "dict": {"a": "b"}, "bool": True, "null": None}
        event_log.append_event("complex", nested)

        events = event_log.read_events()

        assert events[0]["data"] == nested

    def test_append_is_jsonl_on_disk(self):
        event_log.append_event("a", {"n": 1})
        event_log.append_event("b", {"n": 2})

        raw = event_log.LOG_FILE.read_text(encoding="utf-8")
        lines = [line for line in raw.splitlines() if line.strip()]

        assert len(lines) == 2
        # Each line is valid JSON on its own
        for line in lines:
            json.loads(line)

    def test_read_events_on_missing_file_returns_empty(self):
        assert event_log.read_events() == []

    def test_non_serializable_values_use_default_str(self):
        # datetime isn't JSON-serializable by default
        event_log.append_event("with_dt", {"when": datetime(2026, 4, 15, 12, 0, 0)})

        events = event_log.read_events()

        assert "2026-04-15" in events[0]["data"]["when"]


# ---------------------------------------------------------------------------
# read_events ordering and limit
# ---------------------------------------------------------------------------

class TestReadOrdering:
    def test_events_returned_in_append_order(self):
        for i in range(5):
            event_log.append_event("tick", {"i": i})

        events = event_log.read_events()

        assert [e["data"]["i"] for e in events] == [0, 1, 2, 3, 4]

    def test_limit_returns_most_recent(self):
        for i in range(10):
            event_log.append_event("tick", {"i": i})

        events = event_log.read_events(limit=3)

        assert len(events) == 3
        assert [e["data"]["i"] for e in events] == [7, 8, 9]

    def test_limit_larger_than_log_returns_all(self):
        for i in range(3):
            event_log.append_event("tick", {"i": i})

        events = event_log.read_events(limit=100)

        assert len(events) == 3

    def test_limit_zero_returns_empty(self):
        event_log.append_event("tick", {})

        assert event_log.read_events(limit=0) == []

    def test_negative_limit_returns_empty(self):
        event_log.append_event("tick", {})

        assert event_log.read_events(limit=-5) == []

    def test_malformed_lines_are_skipped(self):
        event_log.append_event("ok", {"n": 1})
        # Inject garbage between valid events
        with event_log.LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("not json at all\n")
            f.write("\n")  # blank line
        event_log.append_event("ok2", {"n": 2})

        events = event_log.read_events()

        assert [e["type"] for e in events] == ["ok", "ok2"]


# ---------------------------------------------------------------------------
# Rotation past size threshold
# ---------------------------------------------------------------------------

class TestRotation:
    def test_rotation_fires_past_threshold(self, monkeypatch):
        monkeypatch.setattr(event_log, "MAX_LOG_BYTES", 200)

        # Pad data so each event is well over 20 bytes — cross the threshold fast.
        payload = {"payload": "x" * 100}
        for i in range(10):
            event_log.append_event("bulk", {**payload, "i": i})

        assert event_log.BACKUP_FILE.exists(), "backup should be created after rotation"

    def test_rotation_preserves_recent_events(self, monkeypatch):
        monkeypatch.setattr(event_log, "MAX_LOG_BYTES", 200)

        payload = {"payload": "x" * 100}
        for i in range(10):
            event_log.append_event("bulk", {**payload, "i": i})

        # The latest event must still be readable from the current file.
        events = event_log.read_events(limit=1)
        assert len(events) == 1
        assert events[0]["data"]["i"] == 9

    def test_only_one_backup_is_kept(self, monkeypatch):
        monkeypatch.setattr(event_log, "MAX_LOG_BYTES", 150)

        payload = {"payload": "x" * 100}
        for i in range(30):
            event_log.append_event("bulk", {**payload, "i": i})

        # After many rotations, still only current + 1 backup exist.
        files = sorted(p.name for p in event_log.LOG_DIR.iterdir())
        assert "events.jsonl" in files
        assert "events.1.jsonl" in files
        # No events.2.jsonl or similar accumulation.
        extras = [
            name for name in files
            if name.startswith("events.") and name not in {"events.jsonl", "events.1.jsonl"}
        ]
        assert extras == []

    def test_no_rotation_below_threshold(self, monkeypatch):
        monkeypatch.setattr(event_log, "MAX_LOG_BYTES", 10_000)

        for i in range(5):
            event_log.append_event("small", {"i": i})

        assert not event_log.BACKUP_FILE.exists()
        assert len(event_log.read_events()) == 5

    def test_rotation_starts_fresh_file(self, monkeypatch):
        monkeypatch.setattr(event_log, "MAX_LOG_BYTES", 200)

        payload = {"payload": "x" * 100}
        for i in range(5):
            event_log.append_event("bulk", {**payload, "i": i})

        # Backup holds old events; current file holds only post-rotation ones.
        assert event_log.BACKUP_FILE.exists()
        current_size = event_log.LOG_FILE.stat().st_size
        assert current_size < 200 + 200  # less than threshold + one oversized line
