"""Tests for GET /api/executor/run/<run_id>/status (TK-516).

Read-only status endpoint used by external monitors to poll a single
executor run without loading the full runs list or streaming logs.

Covers:
- 200 with documented fields for a known run_id
- 404 {"error": "not_found"} for an unknown run_id
- 500 {"error": "db_error"} when the DB call raises
- Null fields are preserved (no omission or coercion)
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


class TestExecutorRunStatusEndpoint:
    def test_executor_run_status_endpoint_returns_documented_fields(self, client):
        """Acceptance: 200 response carries exactly the documented fields."""
        executor_runs_db.record_run(
            run_id="20260417-103000-TK-516",
            jira_key="TK-516",
            branch="2026-04-17-103240-TK-516",
            started_at="2026-04-17T10:30:00",
            ended_at="2026-04-17T10:35:00",
            status="success",
            exit_code=0,
            pid=12345,
        )

        resp = client.get("/api/executor/run/20260417-103000-TK-516/status")

        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        body = resp.get_json()
        assert body == {
            "run_id": "20260417-103000-TK-516",
            "idea_id": "TK-516",
            "status": "success",
            "started_at": "2026-04-17T10:30:00",
            "ended_at": "2026-04-17T10:35:00",
            "exit_code": 0,
            "pid": 12345,
        }

    def test_executor_run_status_endpoint_404_for_unknown_id(self, client):
        """Acceptance: 404 path — unknown run_id returns not_found."""
        resp = client.get("/api/executor/run/nonexistent-run-id/status")

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "not_found"}

    def test_executor_run_status_endpoint_preserves_null_fields(self, client):
        """Running runs have no ended_at / exit_code yet — must be null, not
        omitted, so polling monitors can rely on the schema."""
        executor_runs_db.record_run(
            run_id="20260417-104000-TK-516",
            jira_key="TK-516",
            status="running",
            pid=9999,
        )

        resp = client.get("/api/executor/run/20260417-104000-TK-516/status")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["run_id"] == "20260417-104000-TK-516"
        assert body["idea_id"] == "TK-516"
        assert body["status"] == "running"
        assert body["pid"] == 9999
        assert body["ended_at"] is None
        assert body["exit_code"] is None
        # started_at is auto-populated by record_run when not supplied
        assert body["started_at"] is not None

    def test_executor_run_status_endpoint_500_on_db_error(self, client, monkeypatch):
        """Acceptance: 500 with {"error": "db_error"} when the DB call raises."""
        import idea_board.web as web_module

        def _boom(run_id):
            raise RuntimeError("simulated db failure")

        # Patch the attribute the endpoint actually resolves at call time.
        # The endpoint does `from agent import executor_runs_db` inside the
        # handler, so we patch the real module attribute and both views
        # see the same object.
        monkeypatch.setattr(
            executor_runs_db, "get_run_by_run_id", _boom,
        )

        resp = client.get("/api/executor/run/any-id/status")

        assert resp.status_code == 500
        assert resp.get_json() == {"error": "db_error"}

    def test_executor_run_status_endpoint_most_recent_duplicate_wins(self, client):
        """When two rows share a run_id (schema allows it), the newest row
        (largest id) wins — mirrors get_run_by_run_id's ORDER BY id DESC."""
        executor_runs_db.record_run(
            run_id="20260417-105000-TK-516",
            jira_key="TK-516",
            status="failed",
            exit_code=1,
        )
        executor_runs_db.record_run(
            run_id="20260417-105000-TK-516",
            jira_key="TK-516",
            status="success",
            exit_code=0,
        )

        resp = client.get("/api/executor/run/20260417-105000-TK-516/status")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "success"
        assert body["exit_code"] == 0
