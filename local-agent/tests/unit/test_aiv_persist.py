"""Tests for aiv.persist — atomic write of story_quality + dequeue aiv_pending."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from agent import aiv_schema
from agent.aiv_schema import SCORE_COLUMNS
from aiv import persist
from aiv.scorer import StoryQualityScores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect aiv_schema at a temporary SQLite DB for each test."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


def _seed_pending(conn, story_key: str, paths=None) -> None:
    conn.execute(
        "INSERT INTO aiv_pending "
        "(story_key, merged_at, diff_paths_json, enqueued_at) "
        "VALUES (?, ?, ?, ?)",
        (
            story_key,
            "2026-04-18T05:30:00",
            json.dumps(list(paths or [])),
            "2026-04-18T05:30:01",
        ),
    )
    conn.commit()


def _happy_scores(**overrides) -> StoryQualityScores:
    base = {
        "meets_requirements": 9,
        "code_quality": 8,
        "test_quality": 7,
        "security_safety": 10,
        "scope_discipline": 9,
        "edge_cases": 6,
        "product_impact": 8,
        "reasoning_map": {"meets_requirements": "Covers all AC."},
        "red_flags": [],
        "error": "",
    }
    base.update(overrides)
    return StoryQualityScores(**base)


# ---------------------------------------------------------------------------
# _parse_weights
# ---------------------------------------------------------------------------

class TestParseWeights:
    def test_empty_returns_equal_weights(self):
        assert persist._parse_weights("") == (1.0,) * 7
        assert persist._parse_weights(None) == (1.0,) * 7

    def test_wrong_length_returns_equal_weights(self):
        assert persist._parse_weights("1,2,3") == (1.0,) * 7
        assert persist._parse_weights("1,2,3,4,5,6,7,8") == (1.0,) * 7

    def test_non_numeric_returns_equal_weights(self):
        assert persist._parse_weights("1,2,abc,4,5,6,7") == (1.0,) * 7

    def test_all_zero_returns_equal_weights(self):
        assert persist._parse_weights("0,0,0,0,0,0,0") == (1.0,) * 7

    def test_valid_weights_parsed(self):
        assert persist._parse_weights("2,1,1,1,1,1,1") == (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def test_floats_parsed(self):
        result = persist._parse_weights("0.5,1.5,1,1,1,1,1")
        assert result == (0.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# compute_overall_score
# ---------------------------------------------------------------------------

class TestComputeOverallScore:
    def test_equal_weights_simple_mean(self):
        scores = {axis: 7 for axis in SCORE_COLUMNS}
        assert persist.compute_overall_score(scores, weights=(1.0,) * 7) == 7.0

    def test_weighted_formula(self):
        # Double-weight meets_requirements, else equal.
        # mean = (2*10 + 1*4 + 1*4 + 1*4 + 1*4 + 1*4 + 1*4) / (2+1+1+1+1+1+1)
        #      = (20 + 24) / 8 = 5.5
        scores = {axis: 4 for axis in SCORE_COLUMNS}
        scores["meets_requirements"] = 10
        weights = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert persist.compute_overall_score(scores, weights=weights) == pytest.approx(5.5)

    def test_sentinel_axes_excluded(self):
        """An axis of -1 is dropped from the mean, not treated as 0."""
        scores = {axis: 8 for axis in SCORE_COLUMNS}
        scores["edge_cases"] = -1
        # 6 axes * 8 / 6 weights = 8.0 (not 6.86 you'd get if -1 counted).
        assert persist.compute_overall_score(scores, weights=(1.0,) * 7) == 8.0

    def test_all_sentinel_returns_no_score(self):
        scores = {axis: -1 for axis in SCORE_COLUMNS}
        assert persist.compute_overall_score(scores, weights=(1.0,) * 7) == persist.NO_SCORE

    def test_missing_axes_treated_as_sentinel(self):
        """Missing axes contribute nothing (same as sentinel)."""
        scores = {"meets_requirements": 10, "code_quality": 8}
        assert persist.compute_overall_score(scores, weights=(1.0,) * 7) == 9.0

    def test_wrong_weight_length_falls_back_to_equal(self):
        scores = {axis: 7 for axis in SCORE_COLUMNS}
        # Three weights given instead of seven → equal-weight fallback.
        assert persist.compute_overall_score(scores, weights=(2.0, 3.0, 5.0)) == 7.0

    def test_uses_settings_when_no_weights_passed(self):
        scores = {axis: 6 for axis in SCORE_COLUMNS}
        with patch.object(persist.settings, "aiv_weights", "1,1,1,1,1,1,1"):
            assert persist.compute_overall_score(scores) == 6.0

    def test_non_numeric_axis_values_are_skipped(self):
        scores = {axis: 5 for axis in SCORE_COLUMNS}
        scores["code_quality"] = "abc"
        # 6 valid axes * 5 / 6 weights = 5.0
        assert persist.compute_overall_score(scores, weights=(1.0,) * 7) == 5.0


# ---------------------------------------------------------------------------
# record — happy path
# ---------------------------------------------------------------------------

class TestRecordHappyPath:
    def test_inserts_story_quality_row(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(_happy_scores(), {"story_key": "TK-1", "merged_at": "2026-04-18T05:30:00"})

        row = conn.execute(
            "SELECT * FROM story_quality WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert row is not None
        assert row["story_key"] == "TK-1"
        assert row["meets_requirements"] == 9
        assert row["code_quality"] == 8
        assert row["test_quality"] == 7
        assert row["security_safety"] == 10
        assert row["scope_discipline"] == 9
        assert row["edge_cases"] == 6
        assert row["product_impact"] == 8

    def test_dequeues_pending_row(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(_happy_scores(), {"story_key": "TK-1"})

        rows = conn.execute(
            "SELECT story_key FROM aiv_pending WHERE story_key = ?", ("TK-1",)
        ).fetchall()
        assert rows == []

    def test_dequeue_is_scoped_to_story_key(self):
        """Other pending rows must survive."""
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")
        _seed_pending(conn, "TK-2")

        persist.record(_happy_scores(), {"story_key": "TK-1"})

        remaining = {
            r["story_key"]
            for r in conn.execute("SELECT story_key FROM aiv_pending").fetchall()
        }
        assert remaining == {"TK-2"}

    def test_overall_score_is_weighted_mean(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        # Equal weights (empty aiv_weights setting → fallback).
        with patch.object(persist.settings, "aiv_weights", ""):
            persist.record(_happy_scores(), {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT overall_score FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        # (9+8+7+10+9+6+8) / 7 = 57/7 ≈ 8.142857
        assert row["overall_score"] == pytest.approx((9 + 8 + 7 + 10 + 9 + 6 + 8) / 7)

    def test_overall_score_respects_configured_weights(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        # Triple-weight meets_requirements.
        with patch.object(persist.settings, "aiv_weights", "3,1,1,1,1,1,1"):
            persist.record(_happy_scores(), {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT overall_score FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        expected = (3 * 9 + 8 + 7 + 10 + 9 + 6 + 8) / 9  # 75/9
        assert row["overall_score"] == pytest.approx(expected)

    def test_stores_reasoning_and_red_flags_as_json(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        scores = _happy_scores(
            reasoning_map={"meets_requirements": "Covers AC.", "code_quality": "Clean."},
            red_flags=["no_tests_added", "scope_creep"],
        )
        persist.record(scores, {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT reasoning_json, red_flags_json FROM story_quality "
            "WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert json.loads(row["reasoning_json"]) == {
            "meets_requirements": "Covers AC.",
            "code_quality": "Clean.",
        }
        assert json.loads(row["red_flags_json"]) == ["no_tests_added", "scope_creep"]

    def test_stamps_validated_at(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(_happy_scores(), {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT validated_at FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert row["validated_at"]
        # ISO-8601 UTC → ends with +00:00 or Z.
        assert "T" in row["validated_at"]

    def test_accepts_plain_dict_instead_of_dataclass(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        scores_dict = {
            "meets_requirements": 5,
            "code_quality": 5,
            "test_quality": 5,
            "security_safety": 5,
            "scope_discipline": 5,
            "edge_cases": 5,
            "product_impact": 5,
            "reasoning_map": {},
            "red_flags": [],
            "error": "",
        }
        persist.record(scores_dict, {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT meets_requirements FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert row["meets_requirements"] == 5

    def test_stores_story_meta_columns(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(
            _happy_scores(),
            {
                "story_key": "TK-1",
                "story_title": "Implement persist",
                "merged_at": "2026-04-18T05:30:00",
                "verification_method": "pytest",
                "verification_output": "all tests passed",
            },
        )

        row = conn.execute(
            "SELECT story_title, merged_at, verification_method, verification_output "
            "FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert row["story_title"] == "Implement persist"
        assert row["merged_at"] == "2026-04-18T05:30:00"
        assert row["verification_method"] == "pytest"
        assert row["verification_output"] == "all tests passed"


# ---------------------------------------------------------------------------
# record — atomicity / rollback
# ---------------------------------------------------------------------------

class _FailingConn:
    """Proxy around a real sqlite3.Connection that raises on a target SQL
    keyword but otherwise delegates transparently.

    sqlite3.Connection.execute is a read-only C-level attribute so
    ``patch.object(conn, "execute", ...)`` can't replace it directly.
    We install this proxy as ``aiv_schema._local.conn`` for the duration
    of the test so the code under test (``record()``) talks to the
    proxy instead of the raw connection.
    """

    def __init__(self, real: sqlite3.Connection, fail_on: str) -> None:
        self._real = real
        self._fail_on = fail_on.upper()

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith(self._fail_on):
            raise sqlite3.OperationalError(
                f"simulated {self._fail_on.lower()} failure"
            )
        return self._real.execute(sql, *args, **kwargs)

    # Delegate context-manager protocol to the real connection so
    # ``with conn:`` still commits-on-success / rollback-on-error.
    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, tb):
        return self._real.__exit__(exc_type, exc_val, tb)

    def __getattr__(self, name):
        # Anything we don't wrap (commit, rollback, close, row_factory...)
        # goes straight to the real connection.
        return getattr(self._real, name)


class TestRecordAtomicity:
    def test_rolls_back_insert_when_delete_fails(self, monkeypatch):
        """Failure on DELETE must roll back the preceding INSERT — the
        central atomicity guarantee."""
        real = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(real, "TK-1")

        proxy = _FailingConn(real, "DELETE")
        monkeypatch.setattr(aiv_schema._local, "conn", proxy)

        with pytest.raises(sqlite3.OperationalError):
            persist.record(_happy_scores(), {"story_key": "TK-1"})

        # Restore the real connection for assertions.
        monkeypatch.setattr(aiv_schema._local, "conn", real)

        # Neither mutation survives — INSERT was rolled back, DELETE never ran.
        pending = real.execute(
            "SELECT story_key FROM aiv_pending WHERE story_key = ?", ("TK-1",)
        ).fetchall()
        assert len(pending) == 1, "pending row must survive (INSERT rolled back)"

        quality = real.execute(
            "SELECT story_key FROM story_quality WHERE story_key = ?", ("TK-1",)
        ).fetchall()
        assert quality == [], "story_quality row must not have been inserted"

    def test_propagates_insert_errors(self, monkeypatch):
        """A failure on INSERT itself surfaces to the caller and leaves
        the pending row untouched."""
        real = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(real, "TK-1")

        proxy = _FailingConn(real, "INSERT")
        monkeypatch.setattr(aiv_schema._local, "conn", proxy)

        with pytest.raises(sqlite3.OperationalError):
            persist.record(_happy_scores(), {"story_key": "TK-1"})

        monkeypatch.setattr(aiv_schema._local, "conn", real)

        pending = real.execute(
            "SELECT story_key FROM aiv_pending WHERE story_key = ?", ("TK-1",)
        ).fetchall()
        assert len(pending) == 1


# ---------------------------------------------------------------------------
# record — UPSERT on re-validation
# ---------------------------------------------------------------------------

class TestRecordUpsert:
    def test_re_record_same_story_key_updates_scores(self):
        """A second record() for the same story_key must overwrite, not
        raise UNIQUE constraint errors."""
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(_happy_scores(meets_requirements=5), {"story_key": "TK-1"})
        # Second validation with improved scores. Pending row is gone by
        # now — record() must still UPSERT cleanly.
        persist.record(_happy_scores(meets_requirements=10), {"story_key": "TK-1"})

        rows = conn.execute(
            "SELECT meets_requirements FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["meets_requirements"] == 10

    def test_re_record_does_not_raise_unique_error(self):
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()

        # Seed a story_quality row directly — no aiv_pending row needed.
        conn.execute(
            "INSERT INTO story_quality (story_key) VALUES (?)", ("TK-7",)
        )
        conn.commit()

        # Should not raise sqlite3.IntegrityError.
        persist.record(_happy_scores(), {"story_key": "TK-7"})

        row = conn.execute(
            "SELECT meets_requirements FROM story_quality WHERE story_key = ?",
            ("TK-7",),
        ).fetchone()
        assert row["meets_requirements"] == 9


# ---------------------------------------------------------------------------
# record — input validation / sentinel handling
# ---------------------------------------------------------------------------

class TestRecordValidation:
    def test_missing_story_key_raises(self):
        with pytest.raises(ValueError, match="story_key"):
            persist.record(_happy_scores(), {})

    def test_empty_story_key_raises(self):
        with pytest.raises(ValueError, match="story_key"):
            persist.record(_happy_scores(), {"story_key": "   "})

    def test_sentinel_scores_stored_verbatim(self):
        """Sentinel axes must still land in the DB unchanged — they're
        the record of a failed scoring attempt."""
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        sentinel = StoryQualityScores.sentinel("parse_failure")
        persist.record(sentinel, {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT meets_requirements, error, overall_score "
            "FROM story_quality WHERE story_key = ?",
            ("TK-1",),
        ).fetchone()
        assert row["meets_requirements"] == -1
        assert row["error"] == "parse_failure"
        # All axes sentinel → overall_score is NO_SCORE (-1.0).
        assert row["overall_score"] == persist.NO_SCORE

    def test_error_none_when_empty_string(self):
        """Happy-path scores carry ``error=''`` — stored as NULL, not ''."""
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        _seed_pending(conn, "TK-1")

        persist.record(_happy_scores(error=""), {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT error FROM story_quality WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert row["error"] is None

    def test_rejects_unsupported_scores_type(self):
        with pytest.raises(TypeError):
            persist.record("not-a-scores-object", {"story_key": "TK-1"})

    def test_works_when_pending_row_absent(self):
        """DELETE of a missing row is a no-op — record() still writes
        story_quality cleanly."""
        conn = aiv_schema._get_conn()
        aiv_schema.init_db()
        # No _seed_pending call.

        persist.record(_happy_scores(), {"story_key": "TK-1"})

        row = conn.execute(
            "SELECT story_key FROM story_quality WHERE story_key = ?", ("TK-1",)
        ).fetchone()
        assert row is not None
