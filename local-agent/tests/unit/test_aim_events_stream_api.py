"""Tests for the GET /api/aim/events/stream SSE endpoint in idea_board/web.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aim import event_log
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    """Point the event log at a temp file so tests don't touch real data."""
    log_file = tmp_path / "events.jsonl"
    monkeypatch.setattr(event_log, "LOG_DIR", tmp_path)
    monkeypatch.setattr(event_log, "LOG_FILE", log_file)
    monkeypatch.setattr(event_log, "BACKUP_FILE", tmp_path / "events.1.jsonl")
    return log_file


def _read_sse_chunks(response, max_bytes: int = 8192) -> str:
    """Drain bytes from the streaming response up to max_bytes or EOS."""
    out = bytearray()
    for chunk in response.response:
        out.extend(chunk)
        if len(out) >= max_bytes:
            break
    return out.decode("utf-8", errors="replace")


class TestStreamHeaders:
    def test_content_type_is_event_stream(self, client, monkeypatch):
        # Keep generate() short so the request returns quickly.
        monkeypatch.setattr("idea_board.web.time.sleep", lambda *_: None)
        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.content_type
        finally:
            resp.close()

    def test_cache_headers_disable_buffering(self, client, monkeypatch):
        monkeypatch.setattr("idea_board.web.time.sleep", lambda *_: None)
        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            assert resp.headers.get("Cache-Control") == "no-cache"
            assert resp.headers.get("X-Accel-Buffering") == "no"
        finally:
            resp.close()


class TestStreamBody:
    def test_missing_log_yields_waiting_comments(self, client, _isolated_log, monkeypatch):
        """If the log file never appears, stream emits heartbeat comments
        and then terminates gracefully once the wait budget elapses."""
        monkeypatch.setattr("idea_board.web.time.sleep", lambda *_: None)
        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            body = _read_sse_chunks(resp, max_bytes=1024)
            # No event frames were emitted.
            assert "event: event" not in body
            # At least one SSE comment line appeared.
            assert body.startswith(":") or ":" in body
        finally:
            resp.close()

    def test_new_lines_are_emitted_as_events(self, client, _isolated_log, monkeypatch):
        """After the stream opens, appending to the log produces SSE frames."""
        # Create the log before the request so generate() skips the wait.
        _isolated_log.write_text("", encoding="utf-8")

        # Queue up events that will be written on successive sleep() calls.
        pending_events = [
            {"timestamp": "2026-04-15T10:00:00", "type": "worker_started", "data": {"pid": 7}},
            {"timestamp": "2026-04-15T10:00:01", "type": "execution_started", "data": {"id": "TK-1"}},
        ]

        def fake_sleep(_):
            if pending_events:
                ev = pending_events.pop(0)
                with _isolated_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ev) + "\n")

        monkeypatch.setattr("idea_board.web.time.sleep", fake_sleep)

        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            body = _read_sse_chunks(resp, max_bytes=4096)
            assert "event: event" in body
            assert "worker_started" in body
            assert "execution_started" in body
        finally:
            resp.close()

    def test_malformed_lines_are_skipped(self, client, _isolated_log, monkeypatch):
        """Non-JSON lines appended to the log do not appear as SSE events."""
        _isolated_log.write_text("", encoding="utf-8")

        state = {"wrote": False}

        def fake_sleep(_):
            if not state["wrote"]:
                with _isolated_log.open("a", encoding="utf-8") as f:
                    f.write("this is not json\n")
                    f.write(json.dumps(
                        {"timestamp": "t", "type": "good", "data": {"ok": True}}
                    ) + "\n")
                state["wrote"] = True

        monkeypatch.setattr("idea_board.web.time.sleep", fake_sleep)

        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            body = _read_sse_chunks(resp, max_bytes=4096)
            assert "this is not json" not in body
            assert "\"type\": \"good\"" in body
        finally:
            resp.close()

    def test_only_appended_lines_stream_not_pre_existing(
        self, client, _isolated_log, monkeypatch
    ):
        """Events that existed in the log before the stream opened are NOT replayed —
        the generator seeks to end before reading."""
        pre = {"timestamp": "t", "type": "old_event", "data": {}}
        _isolated_log.write_text(json.dumps(pre) + "\n", encoding="utf-8")

        new = {"timestamp": "t2", "type": "new_event", "data": {}}

        def fake_sleep(_):
            with _isolated_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(new) + "\n")
            monkeypatch.setattr("idea_board.web.time.sleep", lambda *_: None)

        monkeypatch.setattr("idea_board.web.time.sleep", fake_sleep)

        resp = client.get("/api/aim/events/stream", buffered=False)
        try:
            body = _read_sse_chunks(resp, max_bytes=4096)
            assert "old_event" not in body
            assert "new_event" in body
        finally:
            resp.close()
