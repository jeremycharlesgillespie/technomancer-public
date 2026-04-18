"""Tests for agent.daily_stats — SQLite-backed daily aggregate rows."""

import sqlite3

import pytest

from agent import daily_stats


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point daily_stats at a temp SQLite DB and reset the connection cache."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", db_path)
    daily_stats._local.__dict__.pop("conn", None)
    yield
    conn = getattr(daily_stats._local, "conn", None)
    if conn:
        conn.close()
        daily_stats._local.__dict__.pop("conn", None)


class TestInitDb:
    def test_creates_file(self):
        assert not daily_stats.DB_PATH.exists()
        daily_stats.init_db()
        assert daily_stats.DB_PATH.exists()

    def test_creates_table(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='daily_stats'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        daily_stats.init_db()
        daily_stats.init_db()
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='daily_stats'"
        ).fetchall()
        assert len(tables) == 1

    def test_schema_has_required_columns(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        cols = {
            r["name"]: r["type"].upper()
            for r in conn.execute("PRAGMA table_info(daily_stats)").fetchall()
        }
        expected_types = {
            "date": "TEXT",
            "project": "TEXT",
            "shipped": "INTEGER",
            "failed": "INTEGER",
            "split_children": "INTEGER",
            "cost_usd": "REAL",
            "p50_wall_s": "REAL",
            "p95_wall_s": "REAL",
            "loc_added": "INTEGER",
            "loc_removed": "INTEGER",
            "first_attempt_success": "INTEGER",
            "splitter_child_success": "INTEGER",
            "splitter_child_fail": "INTEGER",
        }
        for name, expected_type in expected_types.items():
            assert name in cols, f"missing column {name}"
            assert cols[name] == expected_type, (
                f"column {name} has type {cols[name]}, expected {expected_type}"
            )

    def test_composite_primary_key_on_date_and_project(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        pk_rows = sorted(
            (r for r in conn.execute(
                "PRAGMA table_info(daily_stats)"
            ).fetchall() if r["pk"] > 0),
            key=lambda r: r["pk"],
        )
        assert [r["name"] for r in pk_rows] == ["date", "project"]

    def test_primary_key_rejects_duplicate_pair(self):
        """Inserting the same (date, project) twice must raise IntegrityError."""
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
            ("2026-04-17", "TK"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
                ("2026-04-17", "TK"),
            )

    def test_primary_key_allows_same_date_different_project(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
            ("2026-04-17", "TK"),
        )
        conn.execute(
            "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
            ("2026-04-17", "FA"),
        )
        rows = conn.execute("SELECT COUNT(*) FROM daily_stats").fetchone()
        assert rows[0] == 2


class TestRoundTrip:
    def test_insert_and_query_round_trips_all_columns(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        payload = {
            "date": "2026-04-17",
            "project": "TK",
            "shipped": 7,
            "failed": 2,
            "split_children": 4,
            "cost_usd": 1.2345,
            "p50_wall_s": 12.5,
            "p95_wall_s": 58.25,
            "loc_added": 321,
            "loc_removed": 44,
            "first_attempt_success": 5,
            "splitter_child_success": 3,
            "splitter_child_fail": 1,
        }
        cols = ", ".join(payload.keys())
        placeholders = ", ".join("?" for _ in payload)
        conn.execute(
            f"INSERT INTO daily_stats ({cols}) VALUES ({placeholders})",
            tuple(payload.values()),
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM daily_stats WHERE date = ? AND project = ?",
            ("2026-04-17", "TK"),
        ).fetchone()

        assert row is not None
        for key, expected in payload.items():
            assert row[key] == expected, f"{key}: got {row[key]!r}, want {expected!r}"

    def test_integer_columns_return_ints(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project, shipped, loc_added) "
            "VALUES (?, ?, ?, ?)",
            ("2026-04-17", "TK", 3, 100),
        )
        conn.commit()
        row = conn.execute(
            "SELECT shipped, loc_added FROM daily_stats"
        ).fetchone()
        assert isinstance(row["shipped"], int)
        assert isinstance(row["loc_added"], int)

    def test_real_columns_return_floats(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project, cost_usd, p50_wall_s, p95_wall_s) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-04-17", "TK", 0.5, 1.5, 2.5),
        )
        conn.commit()
        row = conn.execute(
            "SELECT cost_usd, p50_wall_s, p95_wall_s FROM daily_stats"
        ).fetchone()
        assert isinstance(row["cost_usd"], float)
        assert isinstance(row["p50_wall_s"], float)
        assert isinstance(row["p95_wall_s"], float)

    def test_text_columns_return_strings(self):
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
            ("2026-04-17", "TK"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT date, project FROM daily_stats"
        ).fetchone()
        assert isinstance(row["date"], str)
        assert isinstance(row["project"], str)

    def test_defaults_populate_unspecified_columns(self):
        """A minimal insert leaves numeric columns at their zero defaults.

        Splitter columns are nullable (TK-618) so an unspecified insert
        leaves them NULL rather than 0 — the NULL is what records 'Jira
        was unreachable when this row was rolled up'.
        """
        daily_stats.init_db()
        conn = daily_stats._get_conn()
        conn.execute(
            "INSERT INTO daily_stats (date, project) VALUES (?, ?)",
            ("2026-04-17", "TK"),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM daily_stats").fetchone()
        assert row["shipped"] == 0
        assert row["failed"] == 0
        assert row["split_children"] == 0
        assert row["cost_usd"] == 0.0
        assert row["p50_wall_s"] == 0.0
        assert row["p95_wall_s"] == 0.0
        assert row["loc_added"] == 0
        assert row["loc_removed"] == 0
        assert row["first_attempt_success"] == 0
        assert row["splitter_child_success"] is None
        assert row["splitter_child_fail"] is None


class TestImporters:
    """Only the rollup module is allowed to import ``daily_stats`` directly.

    Keeping the importer set small means all writes go through the rollup
    pipeline — no ad-hoc inserts scattered across the codebase.
    """

    ALLOWED_IMPORTERS = frozenset({"daily_stats.py", "daily_rollup.py"})

    def test_only_allowed_agent_modules_import_daily_stats(self):
        from pathlib import Path

        import agent

        agent_dir = Path(agent.__path__[0])
        offenders = []
        for py_file in agent_dir.glob("*.py"):
            if py_file.name in self.ALLOWED_IMPORTERS:
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if "daily_stats" in text:
                offenders.append(py_file.name)
        assert offenders == [], (
            f"Unexpected modules importing daily_stats: {offenders}. "
            f"Allowed: {sorted(self.ALLOWED_IMPORTERS)}"
        )
