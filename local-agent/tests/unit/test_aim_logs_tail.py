"""Tests for the AIM + Worker log tail SSE endpoint (TK-554).

Covers:
- ``_aim_log_paths_for_project`` path resolution (primary vs per-project).
- ``_detect_project_for_idea`` scanning ``.aim_state.json`` files.
- ``_iter_matching_lines`` filtering by idea_id + timestamp-prefix check.
- ``GET /api/aim/logs/tail`` end-to-end: missing-param 400-style payload,
  no-log-files path, and historic-line interleave-by-timestamp path.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import idea_board.web as web_module
from idea_board.web import (
    _AIM_LOG_TS_RE,
    _aim_log_paths_for_project,
    _detect_project_for_idea,
    _iter_matching_lines,
    app,
)


@pytest.fixture
def isolated_aim_root(tmp_path, monkeypatch):
    """Point ``_AGENT_ROOT`` at a temp dir with an empty ``aim/`` tree.

    Production ``_AGENT_ROOT`` is ``local-agent/``; tests redirect it so
    we don't read real ``aim/aim.log`` files from the developer's machine.
    """
    aim_dir = tmp_path / "aim"
    aim_dir.mkdir()
    (aim_dir / "projects").mkdir()
    monkeypatch.setattr(web_module, "_AGENT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def fast_tail(monkeypatch):
    """Shrink the tail-loop safety nets so the generator terminates fast.

    ``max_idle_seconds = -1`` makes the very first idle check (which fires
    after the initial historic-line flush) send the ``done`` event so the
    Flask test client's buffered response returns immediately.
    """
    monkeypatch.setattr(web_module, "AIM_LOG_TAIL_MAX_IDLE_SECONDS", -1)
    monkeypatch.setattr(web_module, "AIM_LOG_TAIL_MAX_TOTAL_SECONDS", 3600)
    monkeypatch.setattr(web_module, "AIM_LOG_TAIL_POLL_INTERVAL", 0)


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    """Return ``[(event_name, json_payload)]`` from a raw SSE body."""
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        if event is None:
            continue
        payload = json.loads("".join(data_lines)) if data_lines else {}
        events.append((event, payload))
    return events


class TestAimLogPathsForProject:
    def test_primary_returns_top_level_logs(self, isolated_aim_root):
        paths = _aim_log_paths_for_project(None)
        assert [src for src, _p in paths] == ["aim", "worker"]
        assert paths[0][1] == isolated_aim_root / "aim" / "aim.log"
        assert paths[1][1] == isolated_aim_root / "aim" / "worker.log"

    def test_empty_string_project_treated_as_primary(self, isolated_aim_root):
        paths = _aim_log_paths_for_project("")
        assert paths[0][1] == isolated_aim_root / "aim" / "aim.log"

    def test_primary_literal_treated_as_primary(self, isolated_aim_root):
        paths = _aim_log_paths_for_project("primary")
        assert paths[0][1] == isolated_aim_root / "aim" / "aim.log"

    def test_named_project_uses_projects_subdir(self, isolated_aim_root):
        paths = _aim_log_paths_for_project("40acres")
        assert paths[0][1] == (
            isolated_aim_root / "aim" / "projects" / "40acres" / "aim.log"
        )
        assert paths[1][1] == (
            isolated_aim_root / "aim" / "projects" / "40acres" / "worker.log"
        )


class TestDetectProjectForIdea:
    def _write_state(self, path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")

    def test_finds_idea_in_primary_worker_current(self, isolated_aim_root):
        self._write_state(
            isolated_aim_root / "aim" / ".aim_state.json",
            {"worker": {"current_idea_id": "TK-554"}},
        )
        assert _detect_project_for_idea("TK-554") == "primary"

    def test_finds_idea_in_primary_recent_completions(self, isolated_aim_root):
        self._write_state(
            isolated_aim_root / "aim" / ".aim_state.json",
            {
                "board_snapshot": {
                    "recent_completions": [{"key": "TK-553"}, {"key": "TK-554"}]
                }
            },
        )
        assert _detect_project_for_idea("TK-554") == "primary"

    def test_finds_idea_in_per_project_state(self, isolated_aim_root):
        self._write_state(
            isolated_aim_root / "aim" / "projects" / "40acres" / ".aim_state.json",
            {"worker": {"current_idea_id": "FA-11"}},
        )
        assert _detect_project_for_idea("FA-11") == "40acres"

    def test_returns_none_when_not_found(self, isolated_aim_root):
        self._write_state(
            isolated_aim_root / "aim" / ".aim_state.json",
            {"worker": {"current_idea_id": "TK-1"}},
        )
        assert _detect_project_for_idea("TK-999") is None

    def test_returns_none_when_no_state_files(self, isolated_aim_root):
        assert _detect_project_for_idea("TK-554") is None

    def test_skips_unreadable_state_file(self, isolated_aim_root):
        bad = isolated_aim_root / "aim" / ".aim_state.json"
        bad.write_text("not json", encoding="utf-8")
        self._write_state(
            isolated_aim_root / "aim" / "projects" / "foo" / ".aim_state.json",
            {"worker": {"current_idea_id": "X-1"}},
        )
        assert _detect_project_for_idea("X-1") == "foo"


class TestAimLogTsRegex:
    def test_matches_log_line_prefix(self):
        assert _AIM_LOG_TS_RE.match("2026-04-17 16:32:08 [INFO] foo: bar")

    def test_rejects_missing_timestamp(self):
        assert not _AIM_LOG_TS_RE.match("  File \"foo.py\", line 1, in bar")


class TestIterMatchingLines:
    def test_returns_only_lines_containing_idea_id(self, tmp_path):
        log = tmp_path / "aim.log"
        log.write_text(
            "2026-04-17 10:00:00 [INFO] x: ASSIGN FA-11\n"
            "2026-04-17 10:00:01 [INFO] x: ASSIGN FA-12\n"
            "2026-04-17 10:00:02 [INFO] x: nothing here\n"
            "2026-04-17 10:00:03 [INFO] x: FA-11 done\n",
            encoding="utf-8",
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "aim", "FA-11")
        assert len(rows) == 2
        assert all("FA-11" in r[2] for r in rows)

    def test_prefixes_source_label(self, tmp_path):
        log = tmp_path / "worker.log"
        log.write_text(
            "2026-04-17 10:00:00 [INFO] x: FA-11 start\n", encoding="utf-8"
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "worker", "FA-11")
        assert rows[0][2].startswith("[worker] ")
        assert rows[0][1] == "worker"

    def test_skips_lines_without_timestamp_prefix(self, tmp_path):
        # Continuation lines from stack traces etc. — even if the idea id
        # happens to appear, they have no sortable timestamp and would
        # break the interleave merge.
        log = tmp_path / "aim.log"
        log.write_text(
            "2026-04-17 10:00:00 [ERROR] x: FA-11 crashed\n"
            "  Traceback mentioning FA-11 here\n",
            encoding="utf-8",
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "aim", "FA-11")
        assert len(rows) == 1
        assert "Traceback" not in rows[0][2]

    def test_timestamp_prefix_is_nineteen_chars(self, tmp_path):
        log = tmp_path / "aim.log"
        log.write_text(
            "2026-04-17 10:00:00 [INFO] x: FA-11 test\n", encoding="utf-8"
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "aim", "FA-11")
        assert rows[0][0] == "2026-04-17 10:00:00"
        assert len(rows[0][0]) == 19

    def test_since_filters_lines_before_cutoff(self, tmp_path):
        log = tmp_path / "aim.log"
        log.write_text(
            "2026-04-17 09:59:59 [INFO] x: FA-11 old\n"
            "2026-04-17 10:00:00 [INFO] x: FA-11 at cutoff\n"
            "2026-04-17 10:00:01 [INFO] x: FA-11 after\n",
            encoding="utf-8",
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "aim", "FA-11", since="2026-04-17 10:00:00")
        assert len(rows) == 2
        timestamps = [r[0] for r in rows]
        assert "2026-04-17 09:59:59" not in timestamps
        assert "2026-04-17 10:00:00" in timestamps
        assert "2026-04-17 10:00:01" in timestamps

    def test_since_none_returns_all_lines(self, tmp_path):
        log = tmp_path / "aim.log"
        log.write_text(
            "2026-04-17 09:00:00 [INFO] x: FA-11 early\n"
            "2026-04-17 10:00:00 [INFO] x: FA-11 late\n",
            encoding="utf-8",
        )
        with open(log, encoding="utf-8") as fh:
            rows = _iter_matching_lines(log, fh, "aim", "FA-11", since=None)
        assert len(rows) == 2


class TestAimLogsTailEndpoint:
    def test_missing_idea_param_yields_error_done(self, client, isolated_aim_root):
        resp = client.get("/api/aim/logs/tail")
        assert resp.status_code == 200
        # Flask sets charset on the content-type header.
        assert "text/event-stream" in resp.content_type
        events = _parse_sse_events(resp.get_data(as_text=True))
        assert any(
            ev == "done" and "missing" in payload.get("error", "")
            for ev, payload in events
        )

    def test_no_log_files_present_yields_done(
        self, client, isolated_aim_root, fast_tail
    ):
        # No aim.log / worker.log anywhere — generator should emit a state
        # event then a done event with reason=no_log_files.
        resp = client.get("/api/aim/logs/tail?idea=TK-554&project=primary")
        events = _parse_sse_events(resp.get_data(as_text=True))
        event_names = [ev for ev, _ in events]
        assert "state" in event_names
        assert "done" in event_names
        done_payloads = [p for ev, p in events if ev == "done"]
        assert done_payloads[-1].get("reason") == "no_log_files"

    def test_streams_historic_lines_interleaved_by_timestamp(
        self, client, isolated_aim_root, fast_tail
    ):
        aim_log = isolated_aim_root / "aim" / "aim.log"
        worker_log = isolated_aim_root / "aim" / "worker.log"
        # Interleaved timestamps: aim→worker→aim→worker.
        aim_log.write_text(
            "2026-04-17 10:00:00 [INFO] aim.brain: ASSIGN FA-11\n"
            "2026-04-17 10:00:02 [INFO] aim.manager: Verify FA-11 on main\n"
            "2026-04-17 10:00:04 [INFO] aim.brain: skip FA-999\n",
            encoding="utf-8",
        )
        worker_log.write_text(
            "2026-04-17 10:00:01 [INFO] worker: Starting FA-11\n"
            "2026-04-17 10:00:03 [INFO] worker: FA-11 done\n"
            "2026-04-17 10:00:05 [INFO] worker: unrelated FA-22\n",
            encoding="utf-8",
        )

        resp = client.get("/api/aim/logs/tail?idea=FA-11&project=primary")
        events = _parse_sse_events(resp.get_data(as_text=True))
        log_events = [p for ev, p in events if ev == "log"]
        assert log_events, "expected at least one 'log' event"

        # Flatten all lines across every log event the generator produced.
        all_lines: list[str] = []
        for payload in log_events:
            all_lines.extend(payload.get("lines", []))

        # Only FA-11 lines made it through.
        assert len(all_lines) == 4
        assert all("FA-11" in ln for ln in all_lines)

        # Sources are tagged and interleaved by wall-clock order.
        expected_sources = ["aim", "worker", "aim", "worker"]
        actual_sources = [
            "aim" if ln.startswith("[aim]") else "worker" for ln in all_lines
        ]
        assert actual_sources == expected_sources

        # Initial state event carried project + sources metadata.
        state_events = [p for ev, p in events if ev == "state"]
        assert state_events[0].get("project") == "primary"
        assert set(state_events[0].get("sources", [])) == {"aim", "worker"}

    def test_auto_detects_project_when_param_omitted(
        self, client, isolated_aim_root, fast_tail
    ):
        # Seed state file so the endpoint maps FA-11 → 40acres.
        state_path = (
            isolated_aim_root / "aim" / "projects" / "40acres" / ".aim_state.json"
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"worker": {"current_idea_id": "FA-11"}}),
            encoding="utf-8",
        )
        aim_log = (
            isolated_aim_root / "aim" / "projects" / "40acres" / "aim.log"
        )
        aim_log.write_text(
            "2026-04-17 10:00:00 [INFO] aim.brain: ASSIGN FA-11\n",
            encoding="utf-8",
        )

        resp = client.get("/api/aim/logs/tail?idea=FA-11")
        events = _parse_sse_events(resp.get_data(as_text=True))
        state_events = [p for ev, p in events if ev == "state"]
        assert state_events[0].get("project") == "40acres"

    def test_since_param_filters_historic_lines(
        self, client, isolated_aim_root, fast_tail
    ):
        aim_log = isolated_aim_root / "aim" / "aim.log"
        aim_log.write_text(
            "2026-04-17 09:00:00 [INFO] aim.brain: FA-11 old entry\n"
            "2026-04-17 10:00:00 [INFO] aim.brain: FA-11 at cutoff\n"
            "2026-04-17 10:00:01 [INFO] aim.brain: FA-11 after cutoff\n",
            encoding="utf-8",
        )

        resp = client.get(
            "/api/aim/logs/tail?idea=FA-11&project=primary&since=2026-04-17+10:00:00"
        )
        events = _parse_sse_events(resp.get_data(as_text=True))
        log_events = [p for ev, p in events if ev == "log"]
        all_lines: list[str] = []
        for payload in log_events:
            all_lines.extend(payload.get("lines", []))

        assert len(all_lines) == 2
        assert all("FA-11" in ln for ln in all_lines)
        assert not any("old entry" in ln for ln in all_lines)

    def test_since_param_echoed_in_initial_state_event(
        self, client, isolated_aim_root, fast_tail
    ):
        aim_log = isolated_aim_root / "aim" / "aim.log"
        aim_log.write_text(
            "2026-04-17 10:00:00 [INFO] aim.brain: FA-11 test\n",
            encoding="utf-8",
        )

        resp = client.get(
            "/api/aim/logs/tail?idea=FA-11&project=primary&since=2026-04-17+10:00:00"
        )
        events = _parse_sse_events(resp.get_data(as_text=True))
        state_events = [p for ev, p in events if ev == "state"]
        assert state_events[0].get("since") == "2026-04-17 10:00:00"


class TestLiveLogHtmlIncludesPanel:
    """The /live/<item_id> page must render the collapsible AIM panel."""

    def test_live_viewer_includes_aim_panel(self, client):
        resp = client.get("/live/FA-11")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        # Marker element + SSE endpoint must both be referenced.
        assert "aim-panel" in body
        assert "/api/aim/logs/tail?idea=FA-11" in body
