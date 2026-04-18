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


# ---------------------------------------------------------------------------
# /api/executor/runs/recent — paginated envelope (TK-575)
# ---------------------------------------------------------------------------


def _seed_runs(n: int) -> list[int]:
    """Insert ``n`` runs with increasing started_at; return their row ids."""
    ids = []
    for i in range(n):
        run_id = executor_runs_db.record_run(
            jira_key=f"TK-{i:04d}",
            started_at=f"2026-04-16T10:{i // 60:02d}:{i % 60:02d}",
            duration_ms=1000 + i,
            cost_usd=0.01 * i,
            status="success",
        )
        ids.append(run_id)
    return ids


class TestExecutorRunsRecent:
    def test_empty_returns_envelope(self, client):
        resp = client.get("/api/executor/runs/recent")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        body = resp.get_json()
        assert body == {"runs": [], "total": 0, "limit": 50, "offset": 0}

    def test_envelope_keys_and_total(self, client):
        _seed_runs(3)
        body = client.get("/api/executor/runs/recent").get_json()
        assert set(body.keys()) == {"runs", "total", "limit", "offset"}
        assert body["total"] == 3
        assert body["limit"] == 50
        assert body["offset"] == 0
        assert len(body["runs"]) == 3

    def test_each_run_has_required_fields(self, client):
        _seed_runs(1)
        body = client.get("/api/executor/runs/recent").get_json()
        run = body["runs"][0]
        for key in (
            "id", "jira_key", "status", "started_at", "ended_at",
            "cost_usd", "duration_ms",
        ):
            assert key in run, f"missing field: {key}"

    def test_default_limit_50(self, client):
        _seed_runs(75)
        body = client.get("/api/executor/runs/recent").get_json()
        assert len(body["runs"]) == 50
        assert body["total"] == 75
        assert body["limit"] == 50

    def test_custom_limit(self, client):
        _seed_runs(10)
        body = client.get("/api/executor/runs/recent?limit=5").get_json()
        assert len(body["runs"]) == 5
        assert body["limit"] == 5
        assert body["total"] == 10

    def test_limit_cap_at_200(self, client):
        """limit=999 should be silently clamped to 200."""
        # Only seed a few rows — we care that the *reported* limit is 200.
        _seed_runs(3)
        body = client.get("/api/executor/runs/recent?limit=999").get_json()
        assert body["limit"] == 200
        # And actual row count is bounded by total
        assert len(body["runs"]) == 3

    def test_negative_offset_rejected(self, client):
        resp = client.get("/api/executor/runs/recent?offset=-1")
        assert resp.status_code == 400
        assert "offset" in resp.get_json()["error"]

    def test_zero_limit_rejected(self, client):
        resp = client.get("/api/executor/runs/recent?limit=0")
        assert resp.status_code == 400

    def test_negative_limit_rejected(self, client):
        resp = client.get("/api/executor/runs/recent?limit=-10")
        assert resp.status_code == 400

    def test_non_integer_limit_rejected(self, client):
        resp = client.get("/api/executor/runs/recent?limit=abc")
        assert resp.status_code == 400

    def test_non_integer_offset_rejected(self, client):
        resp = client.get("/api/executor/runs/recent?offset=xyz")
        assert resp.status_code == 400

    def test_pagination_offset_skips_rows(self, client):
        ids = _seed_runs(10)
        page1 = client.get("/api/executor/runs/recent?limit=4&offset=0").get_json()
        page2 = client.get("/api/executor/runs/recent?limit=4&offset=4").get_json()
        # Newest-first: highest id first. ids[-1] is the newest.
        assert page1["runs"][0]["id"] == ids[-1]
        assert len(page1["runs"]) == 4
        assert page2["runs"][0]["id"] == ids[-5]
        assert len(page2["runs"]) == 4
        # No overlap
        page1_ids = {r["id"] for r in page1["runs"]}
        page2_ids = {r["id"] for r in page2["runs"]}
        assert page1_ids.isdisjoint(page2_ids)

    def test_offset_past_end_returns_empty(self, client):
        _seed_runs(3)
        body = client.get("/api/executor/runs/recent?offset=50").get_json()
        assert body["runs"] == []
        assert body["total"] == 3
        assert body["offset"] == 50

    def test_newest_first_ordering(self, client):
        first_id = executor_runs_db.record_run(
            jira_key="TK-OLD", started_at="2026-04-16T09:00:00", status="success",
        )
        second_id = executor_runs_db.record_run(
            jira_key="TK-NEW", started_at="2026-04-16T10:00:00", status="success",
        )
        body = client.get("/api/executor/runs/recent").get_json()
        assert body["runs"][0]["id"] == second_id
        assert body["runs"][1]["id"] == first_id

    def test_error_message_from_failed_tool_call(self, client):
        run_id = executor_runs_db.record_run(
            jira_key="TK-BOOM",
            started_at="2026-04-16T10:00:00",
            status="failed",
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:02", duration_ms=10,
            ok=False, error_message="missing deps",
        )
        body = client.get("/api/executor/runs/recent").get_json()
        assert body["runs"][0]["error_message"] == "missing deps"

    def test_success_runs_skip_error_lookup(self, client):
        run_id = executor_runs_db.record_run(
            jira_key="TK-OK",
            started_at="2026-04-16T10:00:00",
            status="success",
        )
        executor_runs_db.record_tool_call(
            run_id=run_id, tool_name="Bash",
            started_at="2026-04-16T10:00:01", duration_ms=5,
            ok=False, error_message="should not surface",
        )
        body = client.get("/api/executor/runs/recent").get_json()
        assert body["runs"][0]["error_message"] is None


# ---------------------------------------------------------------------------
# executor_runs_db.get_recent_paginated / count_runs
# ---------------------------------------------------------------------------


class TestGetRecentPaginated:
    def test_empty_returns_empty_list(self):
        assert executor_runs_db.get_recent_paginated() == []

    def test_respects_limit_and_offset(self):
        _seed_runs(10)
        page1 = executor_runs_db.get_recent_paginated(limit=3, offset=0)
        page2 = executor_runs_db.get_recent_paginated(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})

    def test_ordering_newest_first(self):
        ids = _seed_runs(5)
        page = executor_runs_db.get_recent_paginated(limit=5)
        assert [r["id"] for r in page] == list(reversed(ids))

    def test_negative_values_coerced_to_zero(self):
        _seed_runs(3)
        # Offset -1 is clamped to 0, so we get the full set back.
        page = executor_runs_db.get_recent_paginated(limit=10, offset=-1)
        assert len(page) == 3
        # Limit -5 is clamped to 0, so we get nothing back.
        page = executor_runs_db.get_recent_paginated(limit=-5, offset=0)
        assert page == []


class TestCountRuns:
    def test_empty_returns_zero(self):
        assert executor_runs_db.count_runs() == 0

    def test_counts_all_rows(self):
        _seed_runs(7)
        assert executor_runs_db.count_runs() == 7
