"""Tests for agent.story_timings — SQLite-backed phase timing rows."""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from agent import story_timings


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point story_timings at a temp SQLite DB and reset the connection cache."""
    db_path = tmp_path / "story_timings.db"
    monkeypatch.setattr(story_timings, "DB_DIR", tmp_path)
    monkeypatch.setattr(story_timings, "DB_PATH", db_path)
    # Drop any cached per-thread connection from a prior test.
    story_timings._local.__dict__.pop("conn", None)
    story_timings.init_db()
    yield
    conn = getattr(story_timings._local, "conn", None)
    if conn:
        conn.close()
        story_timings._local.__dict__.pop("conn", None)


class TestInitDb:
    def test_creates_table(self):
        conn = story_timings._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='story_phase_timings'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        story_timings.init_db()
        story_timings.init_db()

    def test_schema_has_required_columns(self):
        conn = story_timings._get_conn()
        cols = {
            r["name"]
            for r in conn.execute(
                "PRAGMA table_info(story_phase_timings)"
            ).fetchall()
        }
        expected = {
            "id", "run_id", "story_id", "project", "phase",
            "started_at", "ended_at", "duration_ms", "success", "metadata",
        }
        assert expected.issubset(cols)

    def test_required_indexes_exist(self):
        """story_id, phase, and started_at all need covering indexes."""
        conn = story_timings._get_conn()
        names = {
            r["name"]
            for r in conn.execute(
                "PRAGMA index_list('story_phase_timings')"
            ).fetchall()
        }
        assert "idx_story_phase_timings_story_id" in names
        assert "idx_story_phase_timings_phase" in names
        assert "idx_story_phase_timings_started_at" in names


class TestPhaseTimerHappyPath:
    def test_records_single_row_on_success(self):
        with story_timings.phase_timer(
            run_id="run-1",
            story_id="TK-572",
            project="TK",
            phase="plan",
        ):
            pass

        rows = story_timings.get_phases_for_story("TK-572")
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "run-1"
        assert row["story_id"] == "TK-572"
        assert row["project"] == "TK"
        assert row["phase"] == "plan"
        assert row["success"] == 1
        assert row["duration_ms"] >= 0
        assert row["started_at"] is not None
        assert row["ended_at"] is not None

    def test_duration_reflects_elapsed_time(self):
        with story_timings.phase_timer("r", "TK-1", "TK", "code"):
            time.sleep(0.02)

        rows = story_timings.get_phases_for_story("TK-1")
        assert rows[0]["duration_ms"] >= 15  # ~20ms with some slack

    def test_metadata_is_json_stringified(self):
        meta = {"worker": "aim-1", "attempt": 3}
        with story_timings.phase_timer(
            "r", "TK-2", "TK", "plan", metadata=meta,
        ):
            pass

        rows = story_timings.get_phases_for_story("TK-2")
        assert rows[0]["metadata"] is not None
        assert json.loads(rows[0]["metadata"]) == meta

    def test_metadata_none_stores_null(self):
        with story_timings.phase_timer("r", "TK-3", "TK", "plan"):
            pass
        rows = story_timings.get_phases_for_story("TK-3")
        assert rows[0]["metadata"] is None

    def test_non_serializable_metadata_falls_back_to_str(self):
        """Unknown types must not break instrumentation."""
        class Weird:
            def __str__(self):
                return "<weird>"

        # dict with a non-serializable *value* that default=str can handle
        with story_timings.phase_timer(
            "r", "TK-4", "TK", "plan", metadata={"obj": Weird()},
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-4")
        assert "<weird>" in rows[0]["metadata"]


class TestPhaseTimerExceptionPath:
    def test_exception_writes_success_zero_row(self):
        with pytest.raises(RuntimeError, match="boom"):
            with story_timings.phase_timer("r", "TK-5", "TK", "code"):
                raise RuntimeError("boom")

        rows = story_timings.get_phases_for_story("TK-5")
        assert len(rows) == 1
        assert rows[0]["success"] == 0
        assert rows[0]["duration_ms"] >= 0

    def test_exception_is_reraised_unchanged(self):
        """Original exception must propagate with its message intact."""
        original = ValueError("specific-message")
        with pytest.raises(ValueError) as exc_info:
            with story_timings.phase_timer("r", "TK-6", "TK", "test"):
                raise original
        assert exc_info.value is original

    def test_keyboard_interrupt_still_records(self):
        """BaseException subclasses (e.g. KeyboardInterrupt) should also
        record success=0 — callers instrumenting long loops care about
        seeing when a cancel happened."""
        with pytest.raises(KeyboardInterrupt):
            with story_timings.phase_timer("r", "TK-7", "TK", "code"):
                raise KeyboardInterrupt()

        rows = story_timings.get_phases_for_story("TK-7")
        assert len(rows) == 1
        assert rows[0]["success"] == 0

    def test_metadata_captured_on_failure(self):
        """Metadata passed to phase_timer must land on the row even when
        the wrapped block raises — the breakdown dashboard needs context
        (worker id, attempt number, etc.) *especially* for failed phases."""
        meta = {"worker": "aim-1", "attempt": 3, "branch": "feature-x"}
        with pytest.raises(RuntimeError, match="crashed"):
            with story_timings.phase_timer(
                "run-fail", "TK-META-FAIL", "TK", "code", metadata=meta,
            ):
                raise RuntimeError("crashed")

        rows = story_timings.get_phases_for_story("TK-META-FAIL")
        assert len(rows) == 1
        assert rows[0]["success"] == 0
        assert rows[0]["metadata"] is not None
        assert json.loads(rows[0]["metadata"]) == meta
        # Identifying columns should also survive the failure path.
        assert rows[0]["run_id"] == "run-fail"
        assert rows[0]["phase"] == "code"
        assert rows[0]["project"] == "TK"


class TestNestedPhases:
    def test_nested_records_two_independent_rows(self):
        with story_timings.phase_timer("r", "TK-8", "TK", "outer"):
            with story_timings.phase_timer("r", "TK-8", "TK", "inner"):
                pass

        rows = story_timings.get_phases_for_story("TK-8")
        phases = [r["phase"] for r in rows]
        assert "outer" in phases
        assert "inner" in phases
        assert len(rows) == 2

    def test_inner_exception_records_inner_failure_and_outer_failure(self):
        """An inner failure that bubbles through outer must mark both
        rows success=0 — otherwise the breakdown lies about which phase
        owned the failure."""
        with pytest.raises(RuntimeError):
            with story_timings.phase_timer("r", "TK-9", "TK", "outer"):
                with story_timings.phase_timer("r", "TK-9", "TK", "inner"):
                    raise RuntimeError("bubble")

        rows = {r["phase"]: r for r in
                story_timings.get_phases_for_story("TK-9")}
        assert rows["inner"]["success"] == 0
        assert rows["outer"]["success"] == 0


class TestConcurrentWrites:
    def test_parallel_threads_each_get_a_row(self):
        """SQLite + per-thread connections — N threads each writing one
        phase must produce N rows with no lock errors."""
        N = 10
        errors: list[BaseException] = []
        barrier = threading.Barrier(N)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                with story_timings.phase_timer(
                    "r", f"TK-C{idx}", "TK", "code",
                ):
                    time.sleep(0.01)
            except BaseException as exc:  # noqa: BLE001 — test diagnostics
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        conn = story_timings._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM story_phase_timings"
        ).fetchone()[0]
        assert count == N


class TestSerializeMetadata:
    """Direct tests for the private ``_serialize_metadata`` helper.

    The helper is covered indirectly through ``phase_timer``/``record_phase``
    elsewhere, but pinning its contract directly makes refactors safer and
    surfaces failures at the helper level instead of bubbling up through the
    full write path.
    """

    def test_none_returns_none(self):
        """``None`` must stay ``None`` so the column stores SQL NULL, not
        the string ``"null"``."""
        assert story_timings._serialize_metadata(None) is None

    def test_dict_round_trips_through_json(self):
        out = story_timings._serialize_metadata({"a": 1, "b": "two"})
        assert json.loads(out) == {"a": 1, "b": "two"}

    def test_list_round_trips_through_json(self):
        out = story_timings._serialize_metadata([1, 2, 3])
        assert json.loads(out) == [1, 2, 3]

    def test_primitive_values_serialize(self):
        assert story_timings._serialize_metadata(42) == "42"
        assert story_timings._serialize_metadata("hi") == '"hi"'
        assert story_timings._serialize_metadata(True) == "true"
        assert story_timings._serialize_metadata(False) == "false"

    def test_keys_are_sorted(self):
        """``sort_keys=True`` keeps output deterministic across platforms."""
        out = story_timings._serialize_metadata({"c": 3, "a": 1, "b": 2})
        assert out == '{"a": 1, "b": 2, "c": 3}'

    def test_default_str_handles_nested_non_serializable(self):
        """``default=str`` lets serialization succeed when a *value* inside
        the payload is not JSON-native (e.g. datetimes, custom objects)."""
        class Stamp:
            def __str__(self):
                return "STAMP"

        out = story_timings._serialize_metadata({"when": Stamp()})
        assert "STAMP" in out
        assert json.loads(out) == {"when": "STAMP"}

    def test_fully_non_serializable_top_level_falls_back_to_str(self):
        """When ``json.dumps`` raises even with ``default=str``, the helper
        must still return a string — instrumentation never breaks callers."""
        class Bomb:
            def __str__(self):
                return "BOMB"

        # Patch json.dumps to simulate a TypeError that default=str can't fix.
        with patch.object(
            story_timings.json, "dumps", side_effect=TypeError("unserializable")
        ):
            out = story_timings._serialize_metadata(Bomb())
        assert out == "BOMB"

    def test_value_error_also_falls_back_to_str(self):
        """``ValueError`` (e.g. inf/nan edge cases) also triggers fallback."""
        with patch.object(
            story_timings.json, "dumps", side_effect=ValueError("bad float")
        ):
            out = story_timings._serialize_metadata({"x": 1})
        # Fallback is ``str(metadata)`` — a dict repr, not a JSON string.
        assert "x" in out and "1" in out


class TestRecordPhase:
    def test_direct_insert(self):
        row_id = story_timings.record_phase(
            run_id="r",
            story_id="TK-10",
            project="TK",
            phase="deploy",
            started_at="2026-04-17T10:00:00",
            ended_at="2026-04-17T10:00:05",
            duration_ms=5000,
            success=True,
            metadata={"k": "v"},
        )
        assert row_id > 0
        rows = story_timings.get_phases_for_story("TK-10")
        assert len(rows) == 1
        assert rows[0]["duration_ms"] == 5000
        assert rows[0]["success"] == 1
        assert json.loads(rows[0]["metadata"]) == {"k": "v"}

    def test_success_false_stores_zero(self):
        story_timings.record_phase(
            run_id="r", story_id="TK-11", project="TK", phase="test",
            started_at="s", ended_at="e", duration_ms=1,
            success=False,
        )
        rows = story_timings.get_phases_for_story("TK-11")
        assert rows[0]["success"] == 0


class TestGetPhasesForStory:
    def test_orders_by_started_at_ascending(self):
        story_timings.record_phase(
            run_id="r", story_id="TK-12", project="TK", phase="b",
            started_at="2026-04-17T10:00:02", ended_at="x",
            duration_ms=1, success=True,
        )
        story_timings.record_phase(
            run_id="r", story_id="TK-12", project="TK", phase="a",
            started_at="2026-04-17T10:00:01", ended_at="x",
            duration_ms=1, success=True,
        )
        rows = story_timings.get_phases_for_story("TK-12")
        assert [r["phase"] for r in rows] == ["a", "b"]

    def test_unknown_story_returns_empty_list(self):
        assert story_timings.get_phases_for_story("TK-does-not-exist") == []


class TestMetadataEdgeCases:
    def test_empty_dict_metadata_serializes_to_empty_object(self):
        """An empty dict must round-trip, not become NULL."""
        with story_timings.phase_timer(
            "r", "TK-M1", "TK", "plan", metadata={}
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M1")
        assert rows[0]["metadata"] == "{}"
        assert json.loads(rows[0]["metadata"]) == {}

    def test_empty_list_metadata(self):
        with story_timings.phase_timer(
            "r", "TK-M2", "TK", "plan", metadata=[]
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M2")
        assert json.loads(rows[0]["metadata"]) == []

    def test_nested_metadata_round_trips(self):
        meta = {
            "worker": "aim-1",
            "attempts": [1, 2, 3],
            "config": {"timeout": 30, "retries": {"plan": 3, "code": 5}},
            "tags": ["fast", "autonomous"],
        }
        with story_timings.phase_timer(
            "r", "TK-M3", "TK", "plan", metadata=meta
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M3")
        assert json.loads(rows[0]["metadata"]) == meta

    def test_large_metadata_stored_verbatim(self):
        """~100KB of metadata must store and read back intact."""
        big = {"logs": ["line " + "x" * 90] * 1000}
        with story_timings.phase_timer(
            "r", "TK-M4", "TK", "plan", metadata=big
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M4")
        decoded = json.loads(rows[0]["metadata"])
        assert decoded == big
        assert len(rows[0]["metadata"]) > 50_000

    def test_primitive_metadata_values(self):
        """Numbers, strings, and bools are all valid JSON top-level values."""
        for i, value in enumerate([42, "hello", True, False, 3.14, None]):
            story_id = f"TK-M5-{i}"
            with story_timings.phase_timer(
                "r", story_id, "TK", "plan", metadata=value
            ):
                pass
            rows = story_timings.get_phases_for_story(story_id)
            if value is None:
                assert rows[0]["metadata"] is None
            else:
                assert json.loads(rows[0]["metadata"]) == value

    def test_metadata_keys_are_sorted(self):
        """sort_keys=True keeps output deterministic across platforms."""
        with story_timings.phase_timer(
            "r", "TK-M6", "TK", "plan", metadata={"b": 1, "a": 2, "c": 3}
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M6")
        raw = rows[0]["metadata"]
        assert raw.index('"a"') < raw.index('"b"') < raw.index('"c"')

    def test_fully_non_serializable_metadata_falls_back_to_str(self):
        """If default=str can't help either, the whole object str()s."""
        class Unpicklable:
            def __repr__(self):
                return "Unpicklable(instance)"

        with story_timings.phase_timer(
            "r", "TK-M7", "TK", "plan", metadata=Unpicklable()
        ):
            pass
        rows = story_timings.get_phases_for_story("TK-M7")
        assert rows[0]["metadata"] is not None
        assert "Unpicklable" in rows[0]["metadata"]


class TestNullableColumns:
    """Every string column is nullable — the table must accept None."""

    def test_all_ids_may_be_none(self):
        story_timings.record_phase(
            run_id=None, story_id=None, project=None, phase=None,
            started_at="s", ended_at="e", duration_ms=1, success=True,
        )
        conn = story_timings._get_conn()
        row = conn.execute(
            "SELECT run_id, story_id, project, phase "
            "FROM story_phase_timings ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["run_id"] is None
        assert row["story_id"] is None
        assert row["project"] is None
        assert row["phase"] is None

    def test_phase_timer_accepts_none_identifiers(self):
        with story_timings.phase_timer(None, None, None, None):
            pass
        conn = story_timings._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM story_phase_timings WHERE story_id IS NULL"
        ).fetchone()[0]
        assert count == 1


class TestQueryByPhase:
    """`WHERE phase = ?` — aggregate a single phase across all stories."""

    def _seed(self):
        for story in ["TK-Q1", "TK-Q2", "TK-Q3"]:
            for phase, dur in [("plan", 100), ("code", 500), ("test", 300)]:
                story_timings.record_phase(
                    run_id="r", story_id=story, project="TK", phase=phase,
                    started_at=f"2026-04-17T10:00:00",
                    ended_at=f"2026-04-17T10:01:00",
                    duration_ms=dur, success=True,
                )

    def test_filter_by_phase_returns_matching_rows(self):
        self._seed()
        conn = story_timings._get_conn()
        rows = conn.execute(
            "SELECT story_id, duration_ms FROM story_phase_timings "
            "WHERE phase = ? ORDER BY story_id",
            ("code",),
        ).fetchall()
        assert [r["story_id"] for r in rows] == ["TK-Q1", "TK-Q2", "TK-Q3"]
        assert {r["duration_ms"] for r in rows} == {500}

    def test_aggregate_average_per_phase(self):
        """Dashboard pattern: mean duration per phase across all stories."""
        self._seed()
        conn = story_timings._get_conn()
        avg_code = conn.execute(
            "SELECT AVG(duration_ms) FROM story_phase_timings "
            "WHERE phase = ?",
            ("code",),
        ).fetchone()[0]
        assert avg_code == 500

    def test_phase_index_is_used(self):
        """EXPLAIN QUERY PLAN confirms the index, not a full scan."""
        self._seed()
        conn = story_timings._get_conn()
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM story_phase_timings WHERE phase = ?",
            ("code",),
        ).fetchall()
        plan_text = " ".join(str(dict(r)) for r in plan)
        assert "idx_story_phase_timings_phase" in plan_text


class TestQueryByStartedAt:
    """`ORDER BY started_at DESC` — recent-first timeline."""

    def _seed(self):
        for i, started in enumerate([
            "2026-04-17T10:00:00",
            "2026-04-17T11:00:00",
            "2026-04-17T12:00:00",
        ]):
            story_timings.record_phase(
                run_id="r", story_id=f"TK-T{i}", project="TK", phase="plan",
                started_at=started, ended_at=started,
                duration_ms=10, success=True,
            )

    def test_recent_first_ordering(self):
        self._seed()
        conn = story_timings._get_conn()
        rows = conn.execute(
            "SELECT story_id FROM story_phase_timings "
            "ORDER BY started_at DESC"
        ).fetchall()
        assert [r["story_id"] for r in rows] == ["TK-T2", "TK-T1", "TK-T0"]

    def test_range_filter(self):
        self._seed()
        conn = story_timings._get_conn()
        rows = conn.execute(
            "SELECT story_id FROM story_phase_timings "
            "WHERE started_at >= ? AND started_at < ? "
            "ORDER BY started_at",
            ("2026-04-17T10:30:00", "2026-04-17T11:30:00"),
        ).fetchall()
        assert [r["story_id"] for r in rows] == ["TK-T1"]


class TestConcurrentStress:
    """Heavier concurrency than the smoke test in TestConcurrentWrites."""

    def test_many_threads_many_writes_each(self):
        """20 threads × 25 writes = 500 rows, no locks, no errors."""
        THREADS = 20
        PER_THREAD = 25
        errors: list[BaseException] = []
        barrier = threading.Barrier(THREADS)

        def worker(tid: int) -> None:
            try:
                barrier.wait(timeout=10)
                for i in range(PER_THREAD):
                    with story_timings.phase_timer(
                        f"run-{tid}", f"TK-S{tid}", "TK", f"phase-{i % 5}",
                        metadata={"thread": tid, "iter": i},
                    ):
                        pass
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == []
        conn = story_timings._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM story_phase_timings"
        ).fetchone()[0]
        assert count == THREADS * PER_THREAD

    def test_sequential_bulk_insert(self):
        """500 sequential rows must all commit (no transaction leaks)."""
        for i in range(500):
            with story_timings.phase_timer(
                "r", "TK-BULK", "TK", "plan"
            ):
                pass
        rows = story_timings.get_phases_for_story("TK-BULK")
        assert len(rows) == 500
        ids = [r["id"] for r in rows]
        assert ids == sorted(ids)  # AUTOINCREMENT monotonic


class TestPhaseTimerWriteFailure:
    """sqlite3.Error during recording must NOT mask the wrapped exception."""

    def test_sqlite_error_swallowed_on_success_path(self):
        """If write fails on a successful block, caller sees no exception."""
        with patch.object(
            story_timings, "record_phase",
            side_effect=sqlite3.OperationalError("disk full"),
        ):
            # Must not raise.
            with story_timings.phase_timer("r", "TK-F1", "TK", "plan"):
                pass

    def test_sqlite_error_does_not_mask_wrapped_exception(self):
        """Instrumentation failure must never hide a real error."""
        with patch.object(
            story_timings, "record_phase",
            side_effect=sqlite3.OperationalError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="real-error"):
                with story_timings.phase_timer("r", "TK-F2", "TK", "plan"):
                    raise RuntimeError("real-error")


class TestIntegrationStoryLifecycle:
    """End-to-end: record a realistic story execution and query its breakdown."""

    def test_full_lifecycle_breakdown(self):
        run_id = "20260417-173048-TK-LC1"
        story_id = "TK-LC1"

        phases = [
            ("queue-wait", 2000, {"queue_depth": 3}),
            ("worker-pickup", 50, None),
            ("plan", 8000, {"prompt_tokens": 1500}),
            ("code", 45000, {"files_changed": 4}),
            ("test", 12000, {"tests_run": 160}),
            ("deploy", 3000, {"branch": "main"}),
        ]
        for phase, sleep_ms, meta in phases:
            story_timings.record_phase(
                run_id=run_id, story_id=story_id, project="TK", phase=phase,
                started_at=f"2026-04-17T17:30:{phases.index((phase, sleep_ms, meta)):02d}",
                ended_at=f"2026-04-17T17:31:{phases.index((phase, sleep_ms, meta)):02d}",
                duration_ms=sleep_ms, success=True, metadata=meta,
            )

        rows = story_timings.get_phases_for_story(story_id)
        assert len(rows) == len(phases)
        total = sum(r["duration_ms"] for r in rows)
        assert total == sum(p[1] for p in phases)

        # Breakdown by phase — the dashboard's primary lens.
        conn = story_timings._get_conn()
        breakdown = {
            r["phase"]: r["duration_ms"]
            for r in conn.execute(
                "SELECT phase, duration_ms FROM story_phase_timings "
                "WHERE story_id = ?", (story_id,),
            ).fetchall()
        }
        assert breakdown["code"] == 45000
        assert breakdown["deploy"] == 3000

    def test_failed_phase_mixed_with_successful(self):
        """A failed phase shouldn't hide successful predecessors."""
        with story_timings.phase_timer("r", "TK-LC2", "TK", "plan"):
            pass
        with pytest.raises(RuntimeError):
            with story_timings.phase_timer("r", "TK-LC2", "TK", "code"):
                raise RuntimeError("compile failed")

        rows = story_timings.get_phases_for_story("TK-LC2")
        by_phase = {r["phase"]: r for r in rows}
        assert by_phase["plan"]["success"] == 1
        assert by_phase["code"]["success"] == 0

    def test_multiple_stories_do_not_cross_contaminate(self):
        """Writing to story A must not leak into story B's query result."""
        for story in ["TK-A", "TK-B", "TK-C"]:
            for phase in ["plan", "code"]:
                story_timings.record_phase(
                    run_id="r", story_id=story, project="TK", phase=phase,
                    started_at="s", ended_at="e",
                    duration_ms=1, success=True,
                )
        assert len(story_timings.get_phases_for_story("TK-A")) == 2
        assert len(story_timings.get_phases_for_story("TK-B")) == 2
        assert len(story_timings.get_phases_for_story("TK-C")) == 2


class TestMonotonicClockIsolation:
    """phase_timer's docstring promises ``duration_ms`` is derived from
    :func:`time.monotonic`, so a backwards wall-clock jump (NTP slew, DST
    transition, user setting the system clock) cannot produce a negative
    or nonsensically-inflated duration. These tests pin that contract."""

    def test_backwards_wall_clock_jump_keeps_duration_non_negative(self):
        """Pretend ``datetime.now`` jumps one hour backward mid-block.
        ``duration_ms`` must remain >= 0 because it's computed from
        :func:`time.monotonic`, not from the wall-clock strings."""
        real_datetime = datetime
        calls = {"count": 0}

        class FakeDatetime:
            @classmethod
            def now(cls):
                calls["count"] += 1
                base = real_datetime.now()
                # 2nd call (ended_at) pretends clock slid 1 hour backward.
                return base - timedelta(hours=1) if calls["count"] >= 2 else base

        with patch.object(story_timings, "datetime", FakeDatetime):
            with story_timings.phase_timer("r", "TK-MONO1", "TK", "plan"):
                pass

        rows = story_timings.get_phases_for_story("TK-MONO1")
        assert len(rows) == 1
        # Wall-clock strings may be out-of-order (that's the point of the
        # test — we don't rely on them), but duration must stay sane.
        assert rows[0]["duration_ms"] >= 0
        assert rows[0]["duration_ms"] < 5000  # sanity: not an hour in ms

    def test_forwards_wall_clock_jump_does_not_inflate_duration(self):
        """Wall-clock jumping forward by an hour must not make
        ``duration_ms`` claim an hour elapsed — monotonic is authoritative."""
        real_datetime = datetime
        calls = {"count": 0}

        class FakeDatetime:
            @classmethod
            def now(cls):
                calls["count"] += 1
                base = real_datetime.now()
                return base + timedelta(hours=1) if calls["count"] >= 2 else base

        with patch.object(story_timings, "datetime", FakeDatetime):
            with story_timings.phase_timer("r", "TK-MONO2", "TK", "plan"):
                pass

        rows = story_timings.get_phases_for_story("TK-MONO2")
        # Real elapsed time is ~milliseconds; duration must reflect that,
        # not the faked hour-long gap between wall-clock strings.
        assert rows[0]["duration_ms"] < 5000


class TestRecordPhaseBoundaries:
    def test_zero_duration_accepted(self):
        row_id = story_timings.record_phase(
            run_id="r", story_id="TK-B1", project="TK", phase="instant",
            started_at="s", ended_at="e", duration_ms=0, success=True,
        )
        assert row_id > 0
        rows = story_timings.get_phases_for_story("TK-B1")
        assert rows[0]["duration_ms"] == 0

    def test_duration_coerced_to_int(self):
        """Float durations are accepted — record_phase int()s them."""
        story_timings.record_phase(
            run_id="r", story_id="TK-B2", project="TK", phase="p",
            started_at="s", ended_at="e",
            duration_ms=1234.9, success=True,
        )
        rows = story_timings.get_phases_for_story("TK-B2")
        assert rows[0]["duration_ms"] == 1234

    def test_lastrowid_strictly_increases(self):
        ids = []
        for i in range(5):
            ids.append(story_timings.record_phase(
                run_id="r", story_id="TK-B3", project="TK", phase="p",
                started_at="s", ended_at="e",
                duration_ms=1, success=True,
            ))
        assert ids == sorted(ids)
        assert len(set(ids)) == 5
