"""Tests for GET /api/executor/run/<id>/logs (TK-557).

Covers:
- Tail mode returns the last N lines with ``truncated`` flag when trimmed
- ``tail`` query param clamped to [1, 2000]
- ``offset`` query param returns only bytes after the offset
- Resolution path: integer DB id, sortable run_id string, raw idea_id
- 404 when neither the run row nor the conventional log file exists
"""

from __future__ import annotations

import pytest

from agent import executor_runs_db
from idea_board import web as web_module
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point executor_runs_db and EXECUTION_LOGS_DIR at temp paths."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()

    logs_dir = tmp_path / "execution_logs"
    logs_dir.mkdir()
    # web.py took its own binding via ``from .executor import EXECUTION_LOGS_DIR``,
    # so patch the web module's reference rather than executor's.
    monkeypatch.setattr(web_module, "EXECUTION_LOGS_DIR", logs_dir)

    yield logs_dir

    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _write_log(logs_dir, idea_id: str, lines: list[str]) -> None:
    # write_bytes avoids Windows' text-mode \n → \r\n translation so byte
    # offsets stay deterministic across platforms.
    body = ("\n".join(lines) + "\n").encode("utf-8")
    (logs_dir / f"{idea_id}.log").write_bytes(body)


# ---------------------------------------------------------------------------
# 404 paths
# ---------------------------------------------------------------------------


class TestMissingLog:
    def test_unknown_id_returns_404(self, client):
        resp = client.get("/api/executor/run/TK-DOES-NOT-EXIST/logs")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["error"] == "not_found"

    def test_known_run_row_without_log_file_returns_404(self, client, _isolate_db):
        """DB row exists but no on-disk log — still 404."""
        executor_runs_db.record_run(
            jira_key="TK-NOLOG",
            started_at="2026-04-17T10:00:00",
            status="success",
        )
        resp = client.get("/api/executor/run/TK-NOLOG/logs")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tail mode
# ---------------------------------------------------------------------------


class TestTailMode:
    def test_default_tail_returns_all_lines_when_file_is_small(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-SMALL", [f"line {i}" for i in range(10)])
        resp = client.get("/api/executor/run/TK-SMALL/logs")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["lines"] == [f"line {i}" for i in range(10)]
        assert body["truncated"] is False
        assert body["total_bytes"] > 0
        assert body["offset"] == 0
        assert body["idea_id"] == "TK-SMALL"

    def test_tail_50_returns_last_50_lines_and_sets_truncated(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-BIG", [f"line {i}" for i in range(500)])
        resp = client.get("/api/executor/run/TK-BIG/logs?tail=50")
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["lines"]) == 50
        assert body["lines"][0] == "line 450"
        assert body["lines"][-1] == "line 499"
        assert body["truncated"] is True

    def test_tail_clamped_to_max_2000(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-CLAMP", [f"l{i}" for i in range(2500)])
        resp = client.get("/api/executor/run/TK-CLAMP/logs?tail=99999")
        assert resp.status_code == 200
        body = resp.get_json()
        # tail clamped to 2000; the remaining 500 lines make it truncated
        assert len(body["lines"]) == 2000
        assert body["truncated"] is True

    def test_tail_clamped_to_min_1(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-ONE", [f"l{i}" for i in range(5)])
        resp = client.get("/api/executor/run/TK-ONE/logs?tail=0")
        body = resp.get_json()
        assert len(body["lines"]) == 1
        assert body["lines"][0] == "l4"

    def test_invalid_tail_falls_back_to_default(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-BAD", [f"l{i}" for i in range(3)])
        resp = client.get("/api/executor/run/TK-BAD/logs?tail=abc")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["lines"] == ["l0", "l1", "l2"]


# ---------------------------------------------------------------------------
# Offset (incremental polling) mode
# ---------------------------------------------------------------------------


class TestOffsetMode:
    def test_offset_returns_only_bytes_after_offset(self, client, _isolate_db):
        content = b"alpha\nbravo\ncharlie\n"
        (_isolate_db / "TK-OFF.log").write_bytes(content)
        # "alpha\n" is 6 bytes — skip past it
        resp = client.get("/api/executor/run/TK-OFF/logs?offset=6")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["lines"] == ["bravo", "charlie"]
        assert body["offset"] == 6
        assert body["total_bytes"] == len(content)
        assert body["truncated"] is False

    def test_offset_beyond_eof_returns_empty_lines(self, client, _isolate_db):
        (_isolate_db / "TK-EOF.log").write_bytes(b"hi\n")
        resp = client.get("/api/executor/run/TK-EOF/logs?offset=9999")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["lines"] == []

    def test_invalid_offset_falls_back_to_zero(self, client, _isolate_db):
        _write_log(_isolate_db, "TK-BADOFF", ["one", "two"])
        resp = client.get("/api/executor/run/TK-BADOFF/logs?offset=notanumber")
        body = resp.get_json()
        assert body["offset"] == 0
        assert body["lines"] == ["one", "two"]

    def test_offset_total_bytes_enables_roundtrip(self, client, _isolate_db):
        """Poll, then poll again at the returned total_bytes — second poll
        sees only whatever was appended in between."""
        log_path = _isolate_db / "TK-ROUND.log"
        log_path.write_bytes(b"first\n")

        first = client.get("/api/executor/run/TK-ROUND/logs").get_json()
        cursor = first["total_bytes"]

        with open(log_path, "ab") as fh:
            fh.write(b"second\nthird\n")

        second = client.get(f"/api/executor/run/TK-ROUND/logs?offset={cursor}").get_json()
        assert second["lines"] == ["second", "third"]


# ---------------------------------------------------------------------------
# ID resolution: int DB id, run_id string, raw idea_id
# ---------------------------------------------------------------------------


class TestIdResolution:
    def test_resolves_integer_db_id_to_jira_key(self, client, _isolate_db):
        run_pk = executor_runs_db.record_run(
            jira_key="TK-INT",
            started_at="2026-04-17T10:00:00",
            status="success",
        )
        _write_log(_isolate_db, "TK-INT", ["hello", "world"])
        resp = client.get(f"/api/executor/run/{run_pk}/logs")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["idea_id"] == "TK-INT"
        assert body["lines"] == ["hello", "world"]

    def test_resolves_sortable_run_id_to_jira_key(self, client, _isolate_db):
        executor_runs_db.record_run(
            jira_key="TK-SORT",
            run_id="20260417-100000-TK-SORT",
            started_at="2026-04-17T10:00:00",
            status="success",
        )
        _write_log(_isolate_db, "TK-SORT", ["sortable", "run"])
        resp = client.get("/api/executor/run/20260417-100000-TK-SORT/logs")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["idea_id"] == "TK-SORT"
        assert body["lines"] == ["sortable", "run"]

    def test_raw_idea_id_works_without_db_row(self, client, _isolate_db):
        """No DB row but the conventional log file exists — endpoint still serves it."""
        _write_log(_isolate_db, "TK-RAW", ["just", "raw"])
        resp = client.get("/api/executor/run/TK-RAW/logs")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["idea_id"] == "TK-RAW"
        assert body["lines"] == ["just", "raw"]
