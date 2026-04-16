"""Tests for GET /api/executor/run/<id>/tools — per-tool telemetry endpoint."""

from __future__ import annotations

import pytest

from agent import executor_runs_db
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point executor_runs_db at a temporary SQLite DB for each test."""
    db_path = tmp_path / "executor_runs.db"
    monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
    executor_runs_db._local.__dict__.pop("conn", None)
    executor_runs_db.init_db()
    yield
    conn = getattr(executor_runs_db._local, "conn", None)
    if conn:
        conn.close()
        executor_runs_db._local.conn = None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_unknown_run_returns_empty_list(client):
    resp = client.get("/api/executor/run/9999/tools")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {"run_id": 9999, "tool_calls": []}


def test_returns_rows_sorted_by_started_at(client):
    run_id = executor_runs_db.record_run(status="running")
    # Insert out-of-order: 3rd, 1st, 2nd
    executor_runs_db.record_tool_call(
        run_id=run_id, tool_name="third",
        started_at="2026-04-16T10:00:03", duration_ms=10, ok=True,
    )
    executor_runs_db.record_tool_call(
        run_id=run_id, tool_name="first",
        started_at="2026-04-16T10:00:01", duration_ms=20, ok=True,
    )
    executor_runs_db.record_tool_call(
        run_id=run_id, tool_name="second",
        started_at="2026-04-16T10:00:02", duration_ms=15, ok=True,
    )

    resp = client.get(f"/api/executor/run/{run_id}/tools")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_id"] == run_id
    names = [r["tool_name"] for r in data["tool_calls"]]
    assert names == ["first", "second", "third"]


def test_rows_include_expected_fields(client):
    run_id = executor_runs_db.record_run(status="running")
    executor_runs_db.record_tool_call(
        run_id=run_id,
        tool_name="Bash",
        started_at="2026-04-16T10:00:00",
        duration_ms=250,
        input_tokens=99,
        output_tokens=12,
        ok=False,
        error_message="boom",
    )
    data = client.get(f"/api/executor/run/{run_id}/tools").get_json()
    row = data["tool_calls"][0]
    for key in (
        "id", "run_id", "tool_name", "started_at", "duration_ms",
        "input_tokens", "output_tokens", "ok", "error_message",
    ):
        assert key in row
    assert row["tool_name"] == "Bash"
    assert row["duration_ms"] == 250
    assert row["input_tokens"] == 99
    assert row["output_tokens"] == 12
    assert row["ok"] == 0
    assert row["error_message"] == "boom"


def test_does_not_leak_rows_from_other_runs(client):
    run_a = executor_runs_db.record_run(jira_key="TK-A", status="running")
    run_b = executor_runs_db.record_run(jira_key="TK-B", status="running")
    executor_runs_db.record_tool_call(
        run_id=run_a, tool_name="OnlyInA",
        started_at="2026-04-16T10:00:00", duration_ms=1, ok=True,
    )
    executor_runs_db.record_tool_call(
        run_id=run_b, tool_name="OnlyInB",
        started_at="2026-04-16T10:00:00", duration_ms=1, ok=True,
    )

    data_a = client.get(f"/api/executor/run/{run_a}/tools").get_json()
    assert {r["tool_name"] for r in data_a["tool_calls"]} == {"OnlyInA"}


def test_content_type_is_json(client):
    run_id = executor_runs_db.record_run(status="running")
    resp = client.get(f"/api/executor/run/{run_id}/tools")
    assert "application/json" in resp.content_type


def test_sum_of_durations_within_5pct_of_run_duration(client):
    """Acceptance: total_ms across rows is within 5% of the run duration."""
    run_id = executor_runs_db.record_run(
        jira_key="TK-469",
        started_at="2026-04-16T10:00:00",
        duration_ms=10_200,
        status="success",
    )
    per_call = 10_000 // 4
    for i in range(4):
        executor_runs_db.record_tool_call(
            run_id=run_id,
            tool_name=f"T{i}",
            started_at=f"2026-04-16T10:00:0{i}",
            duration_ms=per_call,
            ok=True,
        )

    data = client.get(f"/api/executor/run/{run_id}/tools").get_json()
    summed = sum(r["duration_ms"] for r in data["tool_calls"])
    run_duration = executor_runs_db.get_recent()[0]["duration_ms"]
    assert abs(summed - run_duration) / run_duration <= 0.05
