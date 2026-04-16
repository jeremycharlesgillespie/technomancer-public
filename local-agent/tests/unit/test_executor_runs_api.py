"""Tests for GET /api/executor/runs and GET /executor-runs (TK-446).

Covers:
- JSON endpoint returns rows newest-first, capped at 100
- Error message is pulled from failed tool_calls for failed runs
- HTML page renders the expected shell (totals + table + fetch script)
- /aim dashboard links to the new page
"""

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


# ---------------------------------------------------------------------------
# /api/executor/runs
# ---------------------------------------------------------------------------


class TestExecutorRunsApi:
    def test_empty_returns_empty_array(self, client):
        resp = client.get("/api/executor/runs")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        assert resp.get_json() == []

    def test_returns_expected_fields(self, client):
        executor_runs_db.record_run(
            jira_key="TK-446",
            branch="branch-1",
            started_at="2026-04-16T10:00:00",
            duration_ms=1234,
            cost_usd=0.42,
            status="success",
        )
        rows = client.get("/api/executor/runs").get_json()
        assert len(rows) == 1
        row = rows[0]
        for key in (
            "id", "jira_key", "started_at", "duration_ms",
            "cost_usd", "status", "error_message",
        ):
            assert key in row
        assert row["jira_key"] == "TK-446"
        assert row["duration_ms"] == 1234
        assert row["cost_usd"] == 0.42
        assert row["status"] == "success"
        # Success runs carry no error_message lookup
        assert row["error_message"] is None

    def test_capped_at_100(self, client):
        """Insert 120 runs; API must return exactly 100."""
        for i in range(120):
            executor_runs_db.record_run(
                jira_key=f"TK-{i}",
                started_at=f"2026-04-16T10:{i:02d}:00",
                status="success",
            )
        rows = client.get("/api/executor/runs").get_json()
        assert len(rows) == 100

    def test_newest_first_ordering(self, client):
        """Newer runs (higher id) appear before older ones."""
        first_id = executor_runs_db.record_run(
            jira_key="TK-OLD", started_at="2026-04-16T09:00:00", status="success",
        )
        second_id = executor_runs_db.record_run(
            jira_key="TK-NEW", started_at="2026-04-16T10:00:00", status="success",
        )
        rows = client.get("/api/executor/runs").get_json()
        assert rows[0]["id"] == second_id
        assert rows[1]["id"] == first_id

    def test_error_message_from_failed_tool_call(self, client):
        """Failed runs pick up the first failed tool_call's error_message."""
        run_id = executor_runs_db.record_run(
            jira_key="TK-BOOM",
            started_at="2026-04-16T10:00:00",
            duration_ms=500,
            status="failed",
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:01", duration_ms=10, ok=True,
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:02", duration_ms=10,
            ok=False, error_message="missing deps",
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:03", duration_ms=10,
            ok=False, error_message="second failure",
        )

        rows = client.get("/api/executor/runs").get_json()
        assert rows[0]["error_message"] == "missing deps"

    def test_no_error_message_when_no_failed_tool_calls(self, client):
        """Failed run without tool_call telemetry returns null error_message."""
        executor_runs_db.record_run(
            jira_key="TK-SILENT",
            started_at="2026-04-16T10:00:00",
            status="failed",
        )
        rows = client.get("/api/executor/runs").get_json()
        assert rows[0]["error_message"] is None

    def test_success_runs_skip_error_lookup(self, client):
        """Successful runs leave error_message null even if tool failures exist."""
        run_id = executor_runs_db.record_run(
            jira_key="TK-OK",
            started_at="2026-04-16T10:00:00",
            status="success",
        )
        # Shouldn't matter — success branch skips the lookup entirely.
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:01", duration_ms=5,
            ok=False, error_message="should not surface",
        )
        rows = client.get("/api/executor/runs").get_json()
        assert rows[0]["error_message"] is None


# ---------------------------------------------------------------------------
# /executor-runs HTML page
# ---------------------------------------------------------------------------


class TestExecutorRunsPage:
    def test_returns_200_and_html(self, client):
        resp = client.get("/executor-runs")
        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_contains_table_shell_and_totals(self, client):
        body = client.get("/executor-runs").get_data(as_text=True)
        # Totals
        assert 'id="total-runs"' in body
        assert 'id="total-cost"' in body
        assert 'id="avg-duration"' in body
        assert 'id="success-rate"' in body
        # Table headers include each spec column
        assert 'data-col="id"' in body
        assert 'data-col="jira_key"' in body
        assert 'data-col="started_at"' in body
        assert 'data-col="duration_ms"' in body
        assert 'data-col="cost_usd"' in body
        assert 'data-col="status"' in body
        assert 'data-col="error_message"' in body

    def test_fetches_from_api_endpoint(self, client):
        body = client.get("/executor-runs").get_data(as_text=True)
        assert "/api/executor/runs" in body

    def test_links_back_to_hub_and_aim(self, client):
        body = client.get("/executor-runs").get_data(as_text=True)
        assert 'href="/"' in body
        assert 'href="/aim"' in body


# ---------------------------------------------------------------------------
# /aim dashboard links to /executor-runs
# ---------------------------------------------------------------------------


class TestAimLinksToExecutorRuns:
    def test_aim_page_links_to_executor_runs(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert 'href="/executor-runs"' in body
