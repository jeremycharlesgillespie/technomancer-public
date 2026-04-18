"""Tests for agent.aiv_schema — SQLite schema for the AIV pipeline."""

from __future__ import annotations

import json
import sqlite3

import pytest

from agent import aiv_schema


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect aiv_schema at a temporary SQLite DB for each test."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    # Drop any cached per-thread connection so we open the temp DB fresh.
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


class TestInitDb:
    def test_creates_aiv_pending_table(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='aiv_pending'"
        ).fetchone()
        assert row is not None

    def test_creates_story_quality_table(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='story_quality'"
        ).fetchone()
        assert row is not None

    def test_aiv_pending_has_full_column_set(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(aiv_pending)").fetchall()
        }
        assert cols == {
            "story_key",
            "merged_at",
            "diff_paths_json",
            "enqueued_at",
        }

    def test_aiv_pending_primary_key_is_story_key(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        rows = conn.execute("PRAGMA table_info(aiv_pending)").fetchall()
        pk_cols = [r["name"] for r in rows if r["pk"]]
        assert pk_cols == ["story_key"]

    def test_story_quality_has_full_column_set(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(story_quality)").fetchall()
        }
        expected = {
            "story_key",
            "story_title",
            "merged_at",
            "validated_at",
            "meets_requirements",
            "code_quality",
            "test_quality",
            "security_safety",
            "scope_discipline",
            "edge_cases",
            "product_impact",
            "overall_score",
            "red_flags_json",
            "verification_method",
            "verification_output",
            "reasoning_json",
            "error",
        }
        assert cols == expected

    def test_story_quality_has_seven_score_columns(self):
        """The seven-axis scoring contract is load-bearing — guard it explicitly."""
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(story_quality)").fetchall()
        }
        for axis in aiv_schema.SCORE_COLUMNS:
            assert axis in cols
        assert len(aiv_schema.SCORE_COLUMNS) == 7

    def test_story_quality_primary_key_is_story_key(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        rows = conn.execute("PRAGMA table_info(story_quality)").fetchall()
        pk_cols = [r["name"] for r in rows if r["pk"]]
        assert pk_cols == ["story_key"]

    def test_score_column_types_are_integer(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        rows = conn.execute("PRAGMA table_info(story_quality)").fetchall()
        types = {r["name"]: r["type"] for r in rows}
        for axis in aiv_schema.SCORE_COLUMNS:
            assert types[axis] == "INTEGER"
        assert types["overall_score"] == "REAL"

    def test_idempotent_on_repeat_calls(self):
        """Calling init_db() repeatedly must not raise or alter the schema."""
        aiv_schema.init_db()
        aiv_schema.init_db()
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        # Both tables still present, still exactly one of each
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name IN ('aiv_pending','story_quality') "
                "ORDER BY name"
            ).fetchall()
        ]
        assert tables == ["aiv_pending", "story_quality"]

    def test_idempotent_preserves_existing_data(self):
        """Re-running init_db() on a populated table must not drop rows."""
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        conn.execute(
            "INSERT INTO aiv_pending "
            "(story_key, merged_at, diff_paths_json, enqueued_at) "
            "VALUES (?, ?, ?, ?)",
            ("TK-1", "2026-04-18T00:00:00", "[]", "2026-04-18T00:00:01"),
        )
        conn.commit()

        aiv_schema.init_db()

        rows = conn.execute("SELECT story_key FROM aiv_pending").fetchall()
        assert [r["story_key"] for r in rows] == ["TK-1"]

    def test_creates_data_dir_if_missing(self, tmp_path, monkeypatch):
        """init_db() creates its parent directory if nothing exists yet."""
        fresh = tmp_path / "newdir"
        monkeypatch.setattr(aiv_schema, "DB_DIR", fresh)
        monkeypatch.setattr(aiv_schema, "DB_PATH", fresh / "aiv.db")
        aiv_schema._local.__dict__.pop("conn", None)
        try:
            aiv_schema.init_db()
            assert fresh.exists()
            assert (fresh / "aiv.db").exists()
        finally:
            conn = getattr(aiv_schema._local, "conn", None)
            if conn is not None:
                conn.close()
                aiv_schema._local.conn = None


class TestRoundTrip:
    """Insert a synthetic row into each table and verify every column
    survives the round trip with the correct type."""

    def test_aiv_pending_round_trip(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        payload = {
            "story_key": "TK-42",
            "merged_at": "2026-04-18T05:30:00",
            "diff_paths_json": json.dumps(
                ["agent/foo.py", "tests/unit/test_foo.py"]
            ),
            "enqueued_at": "2026-04-18T05:30:01",
        }
        conn.execute(
            "INSERT INTO aiv_pending "
            "(story_key, merged_at, diff_paths_json, enqueued_at) "
            "VALUES (:story_key, :merged_at, :diff_paths_json, :enqueued_at)",
            payload,
        )
        conn.commit()

        row = conn.execute(
            "SELECT story_key, merged_at, diff_paths_json, enqueued_at "
            "FROM aiv_pending WHERE story_key = ?",
            ("TK-42",),
        ).fetchone()
        assert row is not None
        assert dict(row) == payload
        # diff_paths_json survives JSON round trip
        assert json.loads(row["diff_paths_json"]) == [
            "agent/foo.py",
            "tests/unit/test_foo.py",
        ]

    def test_story_quality_round_trip(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        payload = {
            "story_key": "TK-99",
            "story_title": "Add AIV schema",
            "merged_at": "2026-04-18T06:00:00",
            "validated_at": "2026-04-18T06:01:30",
            "meets_requirements": 9,
            "code_quality": 8,
            "test_quality": 7,
            "security_safety": 10,
            "scope_discipline": 9,
            "edge_cases": 6,
            "product_impact": 8,
            "overall_score": 8.1,
            "red_flags_json": json.dumps(["broad_except"]),
            "verification_method": "tests-only",
            "verification_output": "all tests passed",
            "reasoning_json": json.dumps(
                {"meets_requirements": "Creates both tables as described."}
            ),
            "error": None,
        }
        columns = ", ".join(payload.keys())
        placeholders = ", ".join(f":{k}" for k in payload)
        conn.execute(
            f"INSERT INTO story_quality ({columns}) VALUES ({placeholders})",
            payload,
        )
        conn.commit()

        row = conn.execute(
            f"SELECT {columns} FROM story_quality WHERE story_key = ?",
            ("TK-99",),
        ).fetchone()
        assert row is not None
        assert dict(row) == payload

        # Type assertions — sqlite stores ints/reals/text natively.
        for axis in aiv_schema.SCORE_COLUMNS:
            assert isinstance(row[axis], int), axis
        assert isinstance(row["overall_score"], float)
        assert isinstance(row["story_title"], str)
        assert row["error"] is None
        # JSON columns round-trip as strings we can re-decode.
        assert json.loads(row["red_flags_json"]) == ["broad_except"]
        assert json.loads(row["reasoning_json"]) == {
            "meets_requirements": "Creates both tables as described."
        }

    def test_aiv_pending_story_key_is_unique(self):
        """PRIMARY KEY on story_key means duplicate inserts raise IntegrityError."""
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        conn.execute(
            "INSERT INTO aiv_pending "
            "(story_key, merged_at, diff_paths_json, enqueued_at) "
            "VALUES (?, ?, ?, ?)",
            ("TK-1", "t", "[]", "t"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO aiv_pending "
                "(story_key, merged_at, diff_paths_json, enqueued_at) "
                "VALUES (?, ?, ?, ?)",
                ("TK-1", "t2", "[]", "t2"),
            )
            conn.commit()

    def test_story_quality_story_key_is_unique(self):
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        conn.execute(
            "INSERT INTO story_quality (story_key) VALUES (?)",
            ("TK-1",),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO story_quality (story_key) VALUES (?)",
                ("TK-1",),
            )
            conn.commit()


class TestNoOtherImports:
    """The story explicitly says 'no other module imports aiv_schema yet.'
    Guard that invariant so future refactors don't silently wire it in
    before the downstream stories land."""

    def test_agent_package_has_no_aiv_schema_importers(self):
        import pathlib

        agent_dir = pathlib.Path(aiv_schema.__file__).parent
        offenders: list[str] = []
        for py in agent_dir.rglob("*.py"):
            if py.name == "aiv_schema.py":
                continue
            text = py.read_text(encoding="utf-8", errors="replace")
            if "aiv_schema" in text:
                offenders.append(str(py.relative_to(agent_dir)))
        assert not offenders, (
            f"Unexpected aiv_schema importers: {offenders}. "
            "Story TK-677 says no other module imports aiv_schema yet."
        )
