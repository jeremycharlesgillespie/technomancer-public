"""Tests for GET /api/stats/export.csv endpoint."""

import csv
import io

import pytest

from agent import daily_stats
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect daily_stats to a temp SQLite DB for test isolation."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", db_path)
    daily_stats._local.__dict__.pop("conn", None)
    daily_stats.init_db()
    yield
    conn = getattr(daily_stats._local, "conn", None)
    if conn:
        conn.close()
        daily_stats._local.__dict__.pop("conn", None)


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _insert(date: str, project: str, **kwargs) -> None:
    conn = daily_stats._get_conn()
    base: dict = {
        "shipped": 0, "failed": 0, "split_children": 0,
        "cost_usd": 0.0, "p50_wall_s": 0.0, "p95_wall_s": 0.0,
        "loc_added": 0, "loc_removed": 0, "first_attempt_success": 0,
    }
    base.update(kwargs)
    row = {"date": date, "project": project, **base}
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT INTO daily_stats ({cols}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    conn.commit()


class TestStatsCsvEndpoint:
    def test_content_type_is_csv(self, client):
        resp = client.get("/api/stats/export.csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type

    def test_empty_db_returns_header_only(self, client):
        resp = client.get("/api/stats/export.csv")
        assert resp.status_code == 200
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert rows == []
        assert reader.fieldnames is not None
        assert "date" in reader.fieldnames
        assert "project" in reader.fieldnames

    def test_all_columns_present(self, client):
        _insert("2026-04-01", "TK")
        resp = client.get("/api/stats/export.csv")
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        list(reader)
        for col in daily_stats.CSV_COLUMNS:
            assert col in (reader.fieldnames or []), f"missing column: {col}"

    def test_rows_round_trip(self, client):
        _insert("2026-04-01", "TK", shipped=3, failed=1, cost_usd=1.5)
        _insert("2026-04-02", "TK", shipped=5, failed=0, cost_usd=2.0)

        resp = client.get("/api/stats/export.csv")
        assert resp.status_code == 200
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["date"] == "2026-04-01"
        assert rows[0]["shipped"] == "3"
        assert rows[0]["cost_usd"] == "1.5"
        assert rows[1]["date"] == "2026-04-02"
        assert rows[1]["shipped"] == "5"

    def test_since_filter(self, client):
        _insert("2026-03-15", "TK", shipped=1)
        _insert("2026-04-01", "TK", shipped=2)
        _insert("2026-04-10", "TK", shipped=3)

        resp = client.get("/api/stats/export.csv?since=2026-04-01")
        assert resp.status_code == 200
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert len(rows) == 2
        assert all(r["date"] >= "2026-04-01" for r in rows)

    def test_project_filter(self, client):
        _insert("2026-04-01", "TK", shipped=2)
        _insert("2026-04-01", "FA", shipped=1)
        _insert("2026-04-02", "TK", shipped=4)

        resp = client.get("/api/stats/export.csv?project=TK")
        assert resp.status_code == 200
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert len(rows) == 2
        assert all(r["project"] == "TK" for r in rows)

    def test_since_and_project_combined(self, client):
        _insert("2026-03-01", "TK", shipped=1)
        _insert("2026-04-01", "TK", shipped=2)
        _insert("2026-04-01", "FA", shipped=3)
        _insert("2026-04-05", "TK", shipped=4)

        resp = client.get("/api/stats/export.csv?since=2026-04-01&project=TK")
        assert resp.status_code == 200
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert len(rows) == 2
        assert all(r["project"] == "TK" and r["date"] >= "2026-04-01" for r in rows)

    def test_content_disposition_attachment(self, client):
        resp = client.get("/api/stats/export.csv")
        assert resp.status_code == 200
        disposition = resp.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert "daily_stats.csv" in disposition

    def test_ordered_by_date_then_project(self, client):
        _insert("2026-04-03", "TK")
        _insert("2026-04-01", "FA")
        _insert("2026-04-01", "TK")

        resp = client.get("/api/stats/export.csv")
        reader = csv.DictReader(io.StringIO(resp.data.decode()))
        rows = list(reader)
        assert len(rows) == 3
        dates = [r["date"] for r in rows]
        assert dates == sorted(dates), "rows should be sorted by date ASC"
        # both 2026-04-01 rows: FA < TK alphabetically
        assert rows[0]["project"] == "FA"
        assert rows[1]["project"] == "TK"
