"""Tests for agent.fn_profiler — @profile_fn decorator, registry, SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

import pytest

import agent.fn_profiler as fp
from agent.fn_profiler import (
    FnStats,
    MAX_DURATIONS,
    _percentile,
    all_stats,
    flush_stats,
    get_stats,
    init_db,
    load_stats,
    profile_fn,
    reset_registry,
    start_background_flush,
    stop_background_flush,
)


@pytest.fixture(autouse=True)
def _isolate_fn_profiler(tmp_path, monkeypatch):
    """Point fn_profiler at a tmp DB and reset all module state between tests."""
    monkeypatch.setattr(fp, "DB_DIR", tmp_path)
    monkeypatch.setattr(fp, "DB_PATH", tmp_path / "fn_stats.db")
    fp._reset_connection()
    reset_registry()
    stop_background_flush()
    yield
    stop_background_flush()
    reset_registry()
    fp._reset_connection()


# =========================================================================
# FnStats + _percentile
# =========================================================================


class TestFnStats:
    def test_defaults(self):
        s = FnStats("mod.fn")
        assert s.name == "mod.fn"
        assert s.call_count == 0
        assert s.total_seconds == 0.0
        assert list(s.last_n_durations) == []

    def test_record_updates_all_fields(self):
        s = FnStats("mod.fn")
        s.record(0.5)
        s.record(1.5)
        assert s.call_count == 2
        assert s.total_seconds == pytest.approx(2.0)
        assert list(s.last_n_durations) == [0.5, 1.5]

    def test_last_n_durations_bounded(self):
        s = FnStats("mod.fn")
        for i in range(MAX_DURATIONS + 50):
            s.record(float(i))
        assert s.call_count == MAX_DURATIONS + 50
        assert len(s.last_n_durations) == MAX_DURATIONS
        # The oldest 50 samples should have been evicted.
        assert list(s.last_n_durations)[0] == float(50)
        assert list(s.last_n_durations)[-1] == float(MAX_DURATIONS + 49)

    def test_to_dict_shape(self):
        s = FnStats("mod.fn")
        s.record(1.0)
        s.record(2.0)
        d = s.to_dict()
        assert d["name"] == "mod.fn"
        assert d["call_count"] == 2
        assert d["total_seconds"] == pytest.approx(3.0)
        assert "p50_seconds" in d
        assert "p95_seconds" in d
        assert d["last_n_durations"] == [1.0, 2.0]


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50.0) == 0.0
        assert _percentile([], 95.0) == 0.0

    def test_single_value(self):
        assert _percentile([1.5], 50.0) == 1.5
        assert _percentile([1.5], 95.0) == 1.5

    def test_matches_numpy_on_small_input(self):
        numpy = pytest.importorskip("numpy")
        values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        for pct in (25.0, 50.0, 75.0, 90.0, 95.0, 99.0):
            expected = float(numpy.percentile(values, pct))
            actual = _percentile(values, pct)
            assert actual == pytest.approx(expected, rel=1e-9)

    def test_matches_numpy_on_random_input(self):
        numpy = pytest.importorskip("numpy")
        rng = numpy.random.default_rng(42)
        values = list(rng.random(100).tolist())
        for pct in (50.0, 95.0):
            expected = float(numpy.percentile(values, pct))
            actual = _percentile(values, pct)
            assert actual == pytest.approx(expected, rel=1e-9)

    def test_fn_stats_p50_p95_match_numpy(self):
        numpy = pytest.importorskip("numpy")
        s = FnStats("mod.fn")
        values = [0.05 * i for i in range(1, 61)]  # 60 samples, 0.05..3.0
        for v in values:
            s.record(v)
        assert s.p50() == pytest.approx(float(numpy.percentile(values, 50)), rel=1e-9)
        assert s.p95() == pytest.approx(float(numpy.percentile(values, 95)), rel=1e-9)


# =========================================================================
# @profile_fn decorator
# =========================================================================


class TestDecorator:
    def test_records_duration_for_sync_function(self):
        @profile_fn
        def slow_fn():
            time.sleep(0.01)
            return "ok"

        result = slow_fn()
        assert result == "ok"

        stats = get_stats(slow_fn.__wrapped_fn_name__)
        assert stats is not None
        assert stats.call_count == 1
        assert stats.total_seconds >= 0.01
        assert len(stats.last_n_durations) == 1

    def test_registry_accumulates_across_calls(self):
        @profile_fn
        def fn():
            return 1

        for _ in range(5):
            fn()

        stats = get_stats(fn.__wrapped_fn_name__)
        assert stats is not None
        assert stats.call_count == 5
        assert len(stats.last_n_durations) == 5

    def test_records_duration_for_async_function(self):
        @profile_fn
        async def afn():
            await asyncio.sleep(0.01)
            return "ok"

        result = asyncio.run(afn())
        assert result == "ok"

        stats = get_stats(afn.__wrapped_fn_name__)
        assert stats is not None
        assert stats.call_count == 1
        assert stats.total_seconds >= 0.01

    def test_records_duration_even_when_function_raises(self):
        @profile_fn
        def boom():
            time.sleep(0.005)
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            boom()

        stats = get_stats(boom.__wrapped_fn_name__)
        assert stats is not None
        assert stats.call_count == 1
        assert stats.total_seconds >= 0.005

    def test_distinct_functions_get_distinct_entries(self):
        @profile_fn
        def a():
            return 1

        @profile_fn
        def b():
            return 2

        a(); a(); a()
        b()

        all_keys = all_stats()
        assert a.__wrapped_fn_name__ in all_keys
        assert b.__wrapped_fn_name__ in all_keys
        assert all_keys[a.__wrapped_fn_name__].call_count == 3
        assert all_keys[b.__wrapped_fn_name__].call_count == 1

    def test_preserves_wrapped_metadata(self):
        @profile_fn
        def documented():
            """The docstring."""
            return 42

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "The docstring."

    def test_thread_safe_under_concurrent_calls(self):
        @profile_fn
        def fn():
            return 1

        threads = [
            threading.Thread(target=lambda: [fn() for _ in range(100)])
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = get_stats(fn.__wrapped_fn_name__)
        assert stats is not None
        assert stats.call_count == 800


# =========================================================================
# flush_stats / load_stats — SQLite persistence
# =========================================================================


class TestFlushStats:
    def test_flush_writes_rows_to_sqlite(self, tmp_path):
        @profile_fn
        def fn():
            return 1

        for _ in range(3):
            fn()

        written = flush_stats()
        assert written == 1

        db_path = tmp_path / "fn_stats.db"
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT name, call_count, total_seconds, p50_seconds, p95_seconds, "
                "last_n_durations, updated_at FROM fn_stats"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        name, call_count, total_seconds, p50, p95, durations_json, updated_at = row
        assert name == fn.__wrapped_fn_name__
        assert call_count == 3
        assert total_seconds > 0.0
        durations = json.loads(durations_json)
        assert isinstance(durations, list)
        assert len(durations) == 3
        assert updated_at  # ISO timestamp string

    def test_flush_returns_zero_when_registry_empty(self):
        assert flush_stats() == 0

    def test_flush_upserts_without_duplicating(self, tmp_path):
        @profile_fn
        def fn():
            return 1

        fn()
        flush_stats()

        fn(); fn()
        written = flush_stats()
        assert written == 1  # one row written, not two

        db_path = tmp_path / "fn_stats.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name, call_count FROM fn_stats"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0][1] == 3  # cumulative call_count

    def test_flush_writes_multiple_functions(self, tmp_path):
        @profile_fn
        def a():
            return 1

        @profile_fn
        def b():
            return 2

        a()
        b(); b()
        written = flush_stats()
        assert written == 2

        conn = sqlite3.connect(str(tmp_path / "fn_stats.db"))
        try:
            names = {r[0] for r in conn.execute("SELECT name FROM fn_stats").fetchall()}
        finally:
            conn.close()
        assert a.__wrapped_fn_name__ in names
        assert b.__wrapped_fn_name__ in names


class TestLoadStats:
    def test_load_restores_state(self):
        @profile_fn
        def fn():
            return 1

        for _ in range(4):
            fn()
        flush_stats()

        # Simulate a restart: wipe the in-memory registry.
        reset_registry()
        assert get_stats(fn.__wrapped_fn_name__) is None

        loaded = load_stats()
        assert loaded == 1

        restored = get_stats(fn.__wrapped_fn_name__)
        assert restored is not None
        assert restored.call_count == 4
        assert len(restored.last_n_durations) == 4

    def test_load_preserves_p50_p95_semantics(self):
        numpy = pytest.importorskip("numpy")

        @profile_fn
        def fn():
            return 1

        # Seed the stats directly so we can verify percentiles survive a round-trip.
        stats = FnStats(
            name="mod.test_fn",
            call_count=50,
            total_seconds=25.0,
            last_n_durations=[0.05 * i for i in range(1, 51)],
        )
        with fp._lock:
            fp._registry["mod.test_fn"] = stats

        expected_p50 = float(numpy.percentile(list(stats.last_n_durations), 50))
        expected_p95 = float(numpy.percentile(list(stats.last_n_durations), 95))

        flush_stats()
        reset_registry()
        load_stats()

        restored = get_stats("mod.test_fn")
        assert restored is not None
        assert restored.p50() == pytest.approx(expected_p50, rel=1e-9)
        assert restored.p95() == pytest.approx(expected_p95, rel=1e-9)

    def test_load_on_empty_db_returns_zero(self):
        init_db()
        assert load_stats() == 0
        assert all_stats() == {}

    def test_load_handles_corrupt_json_gracefully(self, tmp_path):
        init_db()
        conn = sqlite3.connect(str(tmp_path / "fn_stats.db"))
        try:
            conn.execute(
                "INSERT INTO fn_stats (name, call_count, total_seconds, "
                "p50_seconds, p95_seconds, last_n_durations, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("mod.broken", 5, 1.0, 0.0, 0.0, "not-json", "2026-04-16T00:00:00"),
            )
            conn.commit()
        finally:
            conn.close()
        # Drop the cached connection so load_stats sees the row written above.
        fp._reset_connection()

        loaded = load_stats()
        assert loaded == 1
        stats = get_stats("mod.broken")
        assert stats is not None
        assert stats.call_count == 5
        assert list(stats.last_n_durations) == []


# =========================================================================
# Background flush timer
# =========================================================================


class TestBackgroundFlush:
    def test_start_is_idempotent(self):
        start_background_flush(interval=60.0)
        first_timer = fp._flush_timer
        start_background_flush(interval=60.0)
        assert fp._flush_timer is first_timer

    def test_stop_clears_state(self):
        start_background_flush(interval=60.0)
        assert fp._flush_running is True
        stop_background_flush()
        assert fp._flush_running is False
        assert fp._flush_timer is None

    def test_short_interval_actually_flushes(self, tmp_path):
        @profile_fn
        def fn():
            return 1

        fn(); fn()
        start_background_flush(interval=0.05)
        # Give the timer time to fire at least once.
        time.sleep(0.25)
        stop_background_flush()

        conn = sqlite3.connect(str(tmp_path / "fn_stats.db"))
        try:
            row = conn.execute(
                "SELECT call_count FROM fn_stats WHERE name = ?",
                (fn.__wrapped_fn_name__,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == 2
