"""Tests for the flat counter metrics surfaced by ``GET /api/metrics``.

Covers the six counter keys the endpoint aggregates from SQLite tables +
the bot service state file: executor_runs totals (overall + last-24h),
crash_triage stories in the last 24h, jira_sync dead-letter queue depth,
embedding store row count, and bot uptime.

The cached observability payload from ``agent.metrics.get_snapshot`` is
stubbed to an empty dict so these tests only exercise the flat counter
path — the existing ``test_metrics_api.py`` owns that contract.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board import web
from idea_board.web import app


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def _stub_snapshot():
    """Keep get_snapshot out of the picture so counter tests stay hermetic."""
    with patch("idea_board.web.metrics.get_snapshot", return_value={}):
        yield


@pytest.fixture
def seeded_paths(tmp_path, monkeypatch):
    """Create empty SQLite DBs for each source and point web.py at them.

    Each DB is initialized with the production schema (minimal subset the
    counters actually need). Tests then INSERT rows to drive each counter.
    """
    executor_db = tmp_path / "executor_runs.db"
    crash_db = tmp_path / "crash_triage_seen.db"
    dlq_db = tmp_path / "jira_sync_dlq.db"
    embeddings_db = tmp_path / "embeddings.db"
    service_state = tmp_path / "service_state.json"

    # executor_runs — only the columns the counters touch.
    with sqlite3.connect(str(executor_db)) as conn:
        conn.execute(
            """CREATE TABLE executor_runs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                status     TEXT,
                started_at TEXT
            )"""
        )
        conn.execute(
            "CREATE INDEX idx_executor_runs_started "
            "ON executor_runs (started_at)"
        )
        conn.commit()

    with sqlite3.connect(str(crash_db)) as conn:
        conn.execute(
            """CREATE TABLE crash_triage_seen (
                hash       TEXT PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                jira_key   TEXT
            )"""
        )
        conn.commit()

    with sqlite3.connect(str(dlq_db)) as conn:
        conn.execute(
            """CREATE TABLE jira_sync_dlq (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id         TEXT NOT NULL,
                payload_json    TEXT NOT NULL,
                error           TEXT NOT NULL,
                attempts        INTEGER NOT NULL,
                first_failed_at TEXT NOT NULL,
                last_failed_at  TEXT NOT NULL
            )"""
        )
        conn.commit()

    with sqlite3.connect(str(embeddings_db)) as conn:
        conn.execute(
            """CREATE TABLE embeddings (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                source    TEXT NOT NULL,
                key       TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL
            )"""
        )
        conn.commit()

    monkeypatch.setattr(web, "_EXECUTOR_RUNS_DB", executor_db)
    monkeypatch.setattr(web, "_CRASH_TRIAGE_DB", crash_db)
    monkeypatch.setattr(web, "_JIRA_SYNC_DLQ_DB", dlq_db)
    monkeypatch.setattr(web, "_EMBEDDINGS_DB", embeddings_db)
    monkeypatch.setattr(web, "_SERVICE_STATE_FILE", service_state)

    return {
        "executor_runs": executor_db,
        "crash_triage": crash_db,
        "jira_sync_dlq": dlq_db,
        "embeddings": embeddings_db,
        "service_state": service_state,
    }


def _iso(dt: datetime) -> str:
    """SQLite-friendly ISO timestamp with a space separator."""
    return dt.isoformat(sep=" ", timespec="seconds")


def _insert_executor_runs(db_path: Path, rows: list[tuple[str, datetime]]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            "INSERT INTO executor_runs (status, started_at) VALUES (?, ?)",
            [(status, _iso(started)) for status, started in rows],
        )
        conn.commit()


def _insert_crash_seen(db_path: Path, entries: list[tuple[str, datetime]]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            "INSERT INTO crash_triage_seen (hash, first_seen, jira_key) "
            "VALUES (?, ?, ?)",
            [(h, _iso(seen), None) for h, seen in entries],
        )
        conn.commit()


def _insert_dlq(db_path: Path, count: int) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        now = _iso(datetime.now())
        conn.executemany(
            "INSERT INTO jira_sync_dlq "
            "(idea_id, payload_json, error, attempts, first_failed_at, "
            "last_failed_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(f"idea-{i}", "{}", "boom", 3, now, now) for i in range(count)],
        )
        conn.commit()


def _insert_embeddings(db_path: Path, count: int) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            "INSERT INTO embeddings (source, key, text_hash, embedding) "
            "VALUES (?, ?, ?, ?)",
            [("s", f"k{i}", "h", b"\x00") for i in range(count)],
        )
        conn.commit()


# =============================================================================
# SHAPE — all six keys present with correct types
# =============================================================================


class TestShape:
    def test_all_six_counter_keys_present(self, client, seeded_paths, _stub_snapshot):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        for key in (
            "executor_runs_total",
            "executor_runs_last_24h",
            "crash_triage_stories_last_24h",
            "jira_sync_dlq_depth",
            "embedding_store_rows",
            "bot_uptime_seconds",
        ):
            assert key in data, f"missing counter key: {key}"

    def test_values_match_empty_sources(self, client, seeded_paths, _stub_snapshot):
        """Every counter starts at 0 / {} / None with empty tables and no state."""
        resp = client.get("/api/metrics")
        data = resp.get_json()
        assert data["executor_runs_total"] == {}
        assert data["executor_runs_last_24h"] == 0
        assert data["crash_triage_stories_last_24h"] == 0
        assert data["jira_sync_dlq_depth"] == 0
        assert data["embedding_store_rows"] == 0
        assert data["bot_uptime_seconds"] is None

    def test_types_are_json_friendly(self, client, seeded_paths, _stub_snapshot):
        resp = client.get("/api/metrics")
        data = resp.get_json()
        assert isinstance(data["executor_runs_total"], dict)
        for scalar_key in (
            "executor_runs_last_24h",
            "crash_triage_stories_last_24h",
            "jira_sync_dlq_depth",
            "embedding_store_rows",
        ):
            assert isinstance(data[scalar_key], int)


# =============================================================================
# executor_runs_total — grouped by status
# =============================================================================


class TestExecutorRunsTotal:
    def test_groups_by_status(self, client, seeded_paths, _stub_snapshot):
        now = datetime.now()
        _insert_executor_runs(
            seeded_paths["executor_runs"],
            [
                ("success", now),
                ("success", now),
                ("success", now),
                ("failed", now),
                ("failed", now),
                ("running", now),
            ],
        )
        resp = client.get("/api/metrics")
        data = resp.get_json()
        assert data["executor_runs_total"] == {
            "success": 3,
            "failed": 2,
            "running": 1,
        }

    def test_null_status_rows_bucket_as_unknown(
        self, client, seeded_paths, _stub_snapshot
    ):
        now = datetime.now()
        with sqlite3.connect(str(seeded_paths["executor_runs"])) as conn:
            conn.execute(
                "INSERT INTO executor_runs (status, started_at) VALUES (NULL, ?)",
                (_iso(now),),
            )
            conn.execute(
                "INSERT INTO executor_runs (status, started_at) VALUES (?, ?)",
                ("success", _iso(now)),
            )
            conn.commit()
        data = client.get("/api/metrics").get_json()
        assert data["executor_runs_total"]["unknown"] == 1
        assert data["executor_runs_total"]["success"] == 1


# =============================================================================
# executor_runs_last_24h — time-windowed count
# =============================================================================


class TestExecutorRunsLast24h:
    def test_counts_only_recent_rows(self, client, seeded_paths, _stub_snapshot):
        now = datetime.now()
        _insert_executor_runs(
            seeded_paths["executor_runs"],
            [
                ("success", now - timedelta(hours=1)),
                ("success", now - timedelta(hours=23)),
                ("failed", now - timedelta(hours=25)),       # outside window
                ("failed", now - timedelta(days=7)),          # outside window
            ],
        )
        data = client.get("/api/metrics").get_json()
        assert data["executor_runs_last_24h"] == 2
        # Overall totals still capture every row.
        assert sum(data["executor_runs_total"].values()) == 4

    def test_ignores_rows_with_null_started_at(
        self, client, seeded_paths, _stub_snapshot
    ):
        with sqlite3.connect(str(seeded_paths["executor_runs"])) as conn:
            conn.execute(
                "INSERT INTO executor_runs (status, started_at) VALUES (?, NULL)",
                ("queued",),
            )
            conn.commit()
        data = client.get("/api/metrics").get_json()
        assert data["executor_runs_last_24h"] == 0


# =============================================================================
# crash_triage_stories_last_24h
# =============================================================================


class TestCrashTriageStoriesLast24h:
    def test_counts_rows_inside_window(self, client, seeded_paths, _stub_snapshot):
        now = datetime.now()
        _insert_crash_seen(
            seeded_paths["crash_triage"],
            [
                ("h1", now - timedelta(hours=1)),
                ("h2", now - timedelta(hours=23)),
                ("h3", now - timedelta(hours=48)),  # outside window
            ],
        )
        data = client.get("/api/metrics").get_json()
        assert data["crash_triage_stories_last_24h"] == 2


# =============================================================================
# jira_sync_dlq_depth
# =============================================================================


class TestJiraSyncDlqDepth:
    def test_counts_all_rows(self, client, seeded_paths, _stub_snapshot):
        _insert_dlq(seeded_paths["jira_sync_dlq"], 5)
        data = client.get("/api/metrics").get_json()
        assert data["jira_sync_dlq_depth"] == 5


# =============================================================================
# embedding_store_rows
# =============================================================================


class TestEmbeddingStoreRows:
    def test_counts_all_rows(self, client, seeded_paths, _stub_snapshot):
        _insert_embeddings(seeded_paths["embeddings"], 7)
        data = client.get("/api/metrics").get_json()
        assert data["embedding_store_rows"] == 7


# =============================================================================
# bot_uptime_seconds — derived from service_state.json
# =============================================================================


class TestBotUptimeSeconds:
    def test_computes_from_started_at(self, client, seeded_paths, _stub_snapshot):
        started = datetime.now() - timedelta(seconds=90)
        seeded_paths["service_state"].write_text(
            json.dumps({"bot_started_at": started.isoformat()}),
            encoding="utf-8",
        )
        data = client.get("/api/metrics").get_json()
        uptime = data["bot_uptime_seconds"]
        assert uptime is not None
        # Allow a small window — the clock ticks between write and read.
        assert 88 <= uptime <= 120

    def test_null_when_state_missing(self, client, seeded_paths, _stub_snapshot):
        # Fixture leaves the state file untouched — should not exist.
        assert not seeded_paths["service_state"].exists()
        data = client.get("/api/metrics").get_json()
        assert data["bot_uptime_seconds"] is None

    def test_null_when_started_at_missing(
        self, client, seeded_paths, _stub_snapshot
    ):
        seeded_paths["service_state"].write_text(
            json.dumps({"total_restarts": 3}), encoding="utf-8"
        )
        data = client.get("/api/metrics").get_json()
        assert data["bot_uptime_seconds"] is None

    def test_null_when_started_at_unparseable(
        self, client, seeded_paths, _stub_snapshot
    ):
        seeded_paths["service_state"].write_text(
            json.dumps({"bot_started_at": "not-a-timestamp"}),
            encoding="utf-8",
        )
        data = client.get("/api/metrics").get_json()
        assert data["bot_uptime_seconds"] is None

    def test_null_when_service_state_corrupt_json(
        self, client, seeded_paths, _stub_snapshot
    ):
        seeded_paths["service_state"].write_text("not json{", encoding="utf-8")
        data = client.get("/api/metrics").get_json()
        assert data["bot_uptime_seconds"] is None


# =============================================================================
# Graceful handling — missing DBs / missing tables
# =============================================================================


class TestMissingSources:
    def test_missing_db_files_report_zero(self, tmp_path, monkeypatch, client,
                                          _stub_snapshot):
        """Pointing at non-existent files must not 500 — report zeros."""
        monkeypatch.setattr(web, "_EXECUTOR_RUNS_DB", tmp_path / "nope1.db")
        monkeypatch.setattr(web, "_CRASH_TRIAGE_DB", tmp_path / "nope2.db")
        monkeypatch.setattr(web, "_JIRA_SYNC_DLQ_DB", tmp_path / "nope3.db")
        monkeypatch.setattr(web, "_EMBEDDINGS_DB", tmp_path / "nope4.db")
        monkeypatch.setattr(web, "_SERVICE_STATE_FILE", tmp_path / "state.json")
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["executor_runs_total"] == {}
        assert data["executor_runs_last_24h"] == 0
        assert data["crash_triage_stories_last_24h"] == 0
        assert data["jira_sync_dlq_depth"] == 0
        assert data["embedding_store_rows"] == 0
        assert data["bot_uptime_seconds"] is None

    def test_missing_tables_report_zero(self, tmp_path, monkeypatch, client,
                                        _stub_snapshot):
        """DB exists but tables are missing (e.g. just-created file)."""
        for name in ("exec", "crash", "dlq", "emb"):
            path = tmp_path / f"{name}.db"
            sqlite3.connect(str(path)).close()

        monkeypatch.setattr(web, "_EXECUTOR_RUNS_DB", tmp_path / "exec.db")
        monkeypatch.setattr(web, "_CRASH_TRIAGE_DB", tmp_path / "crash.db")
        monkeypatch.setattr(web, "_JIRA_SYNC_DLQ_DB", tmp_path / "dlq.db")
        monkeypatch.setattr(web, "_EMBEDDINGS_DB", tmp_path / "emb.db")
        monkeypatch.setattr(web, "_SERVICE_STATE_FILE", tmp_path / "state.json")
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["executor_runs_total"] == {}
        assert data["executor_runs_last_24h"] == 0
        assert data["crash_triage_stories_last_24h"] == 0
        assert data["jira_sync_dlq_depth"] == 0
        assert data["embedding_store_rows"] == 0


# =============================================================================
# Performance — <100ms with 10k executor_runs rows
# =============================================================================


class TestPerformance:
    def test_under_100ms_with_10k_executor_rows(
        self, client, seeded_paths, _stub_snapshot
    ):
        """Acceptance criterion: the endpoint must complete in under 100ms
        with 10k rows in ``executor_runs``. The indexed started_at column
        plus GROUP BY on status should keep this well inside the budget."""
        now = datetime.now()
        cutoff = now - timedelta(hours=24)
        bulk = []
        for i in range(10_000):
            # Half inside the 24h window, half outside — exercises both
            # the full-table scan and the index-assisted time range query.
            started = now - timedelta(minutes=i) if i % 2 == 0 else (
                cutoff - timedelta(minutes=i)
            )
            status = ("success", "failed", "running")[i % 3]
            bulk.append((status, started))
        _insert_executor_runs(seeded_paths["executor_runs"], bulk)

        # Warm-up request so SQLite opens the file; then measure the hot path.
        client.get("/api/metrics")

        start = time.perf_counter()
        resp = client.get("/api/metrics")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200
        data = resp.get_json()
        assert sum(data["executor_runs_total"].values()) == 10_000
        assert elapsed_ms < 100, f"endpoint took {elapsed_ms:.1f}ms (budget 100ms)"


# =============================================================================
# Does not regress the existing snapshot payload
# =============================================================================


class TestSnapshotCoexistence:
    def test_existing_snapshot_keys_still_present(self, client, seeded_paths):
        """Merging the flat counters into the response must not clobber
        the observability snapshot that ``test_metrics_api.py`` asserts."""
        fake_snapshot = {
            "executor": {"success_rate": 0.8},
            "board": {"queue_depth": 5},
            "generated_at": "2026-04-17T00:00:00",
            "stale_seconds": 0.0,
        }
        with patch("idea_board.web.metrics.get_snapshot",
                   return_value=fake_snapshot):
            resp = client.get("/api/metrics")
        data = resp.get_json()
        assert data["executor"]["success_rate"] == 0.8
        assert data["board"]["queue_depth"] == 5
        # And the new counters sit alongside the snapshot.
        assert "executor_runs_total" in data
        assert "bot_uptime_seconds" in data
