"""Tests for the story_model_usage table + executor hook.

Verifies:
* ``executor_runs_db.init_db`` creates and re-creates the table idempotently.
* ``record_story_model_usage`` round-trips model + cost correctly.
* ``_accumulate_model_usage`` extracts (model, usage) from stream-json lines
  and buckets per-model call_count + cost_usd.
* End-to-end: simulating a claude -p run (feeding assistant events through
  the accumulator, then flushing) writes one story_model_usage row per
  model with correct call_count and cost_usd.
"""

import json

import pytest

from agent import executor_runs_db
from agent.claude_code_runner import _cost_from_usage_dict
from idea_board import executor as executor_mod


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


class TestSchema:
    def test_table_created(self):
        conn = executor_runs_db._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='story_model_usage'"
        ).fetchone()
        assert row is not None

    def test_schema_columns(self):
        conn = executor_runs_db._get_conn()
        cols = {
            r["name"]
            for r in conn.execute(
                "PRAGMA table_info(story_model_usage)"
            ).fetchall()
        }
        assert {
            "story_key", "model", "call_count", "cost_usd",
            "cache_read_tokens", "cache_write_tokens", "recorded_at",
        }.issubset(cols)

    def test_index_exists(self):
        conn = executor_runs_db._get_conn()
        names = {
            r["name"]
            for r in conn.execute(
                "PRAGMA index_list('story_model_usage')"
            ).fetchall()
        }
        assert "idx_story_model_usage_story" in names

    def test_init_db_idempotent(self):
        """Running init_db twice must not raise or duplicate the table."""
        executor_runs_db.init_db()
        executor_runs_db.init_db()
        conn = executor_runs_db._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sqlite_master "
            "WHERE type='table' AND name='story_model_usage'"
        ).fetchone()
        assert row["n"] == 1


class TestRecordStoryModelUsage:
    def test_insert_round_trip(self):
        row_id = executor_runs_db.record_story_model_usage(
            story_key="TK-613",
            model="claude-opus-4-6",
            call_count=3,
            cost_usd=0.42,
        )
        assert row_id > 0
        rows = executor_runs_db.get_story_model_usage("TK-613")
        assert len(rows) == 1
        row = rows[0]
        assert row["story_key"] == "TK-613"
        assert row["model"] == "claude-opus-4-6"
        assert row["call_count"] == 3
        assert row["cost_usd"] == pytest.approx(0.42)
        assert row["recorded_at"]

    def test_multiple_models_per_story(self):
        executor_runs_db.record_story_model_usage(
            "TK-613", "claude-opus-4-6", 2, 0.50
        )
        executor_runs_db.record_story_model_usage(
            "TK-613", "claude-haiku-4-5-20251001", 5, 0.03
        )
        rows = executor_runs_db.get_story_model_usage("TK-613")
        assert len(rows) == 2
        models = {r["model"] for r in rows}
        assert models == {"claude-opus-4-6", "claude-haiku-4-5-20251001"}

    def test_recorded_at_override(self):
        ts = "2026-04-17T12:00:00"
        executor_runs_db.record_story_model_usage(
            "TK-1", "claude-opus-4-6", 1, 0.10, recorded_at=ts
        )
        rows = executor_runs_db.get_story_model_usage("TK-1")
        assert rows[0]["recorded_at"] == ts

    def test_get_empty_when_no_rows(self):
        assert executor_runs_db.get_story_model_usage("TK-999") == []


class TestAccumulateModelUsage:
    def _assistant_line(self, model: str, usage: dict) -> str:
        return json.dumps({
            "type": "assistant",
            "message": {
                "model": model,
                "usage": usage,
                "content": [{"type": "text", "text": "hi"}],
            },
        })

    def test_single_assistant_event_accumulates(self):
        bucket: dict = {}
        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        line = self._assistant_line("claude-opus-4-6", usage)
        executor_mod._accumulate_model_usage(line, bucket)

        assert "claude-opus-4-6" in bucket
        assert bucket["claude-opus-4-6"]["call_count"] == 1
        expected = _cost_from_usage_dict(usage, model="claude-opus-4-6")
        assert bucket["claude-opus-4-6"]["cost_usd"] == pytest.approx(expected)

    def test_multiple_models_tracked_separately(self):
        bucket: dict = {}
        opus_usage = {"input_tokens": 1000, "output_tokens": 500}
        haiku_usage = {"input_tokens": 2000, "output_tokens": 200}
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-opus-4-6", opus_usage), bucket
        )
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-opus-4-6", opus_usage), bucket
        )
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-haiku-4-5-20251001", haiku_usage),
            bucket,
        )
        assert bucket["claude-opus-4-6"]["call_count"] == 2
        assert bucket["claude-haiku-4-5-20251001"]["call_count"] == 1

    def test_invalid_json_ignored(self):
        bucket: dict = {"existing": {"call_count": 1, "cost_usd": 0.5}}
        executor_mod._accumulate_model_usage("not json", bucket)
        executor_mod._accumulate_model_usage("", bucket)
        executor_mod._accumulate_model_usage(
            '{"type": "tool_use"}', bucket
        )
        # Bucket unchanged
        assert bucket == {"existing": {"call_count": 1, "cost_usd": 0.5}}

    def test_missing_model_or_usage_ignored(self):
        bucket: dict = {}
        # No usage field
        executor_mod._accumulate_model_usage(
            json.dumps({"type": "assistant", "message": {"model": "x"}}),
            bucket,
        )
        # No model field
        executor_mod._accumulate_model_usage(
            json.dumps({
                "type": "assistant",
                "message": {"usage": {"input_tokens": 1}},
            }),
            bucket,
        )
        assert bucket == {}

    def test_cache_tokens_accumulated(self):
        """cache_read_input_tokens and cache_creation_input_tokens are tracked."""
        bucket: dict = {}
        # First call: writes cache (creation tokens)
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-sonnet-4-6", {
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 3800,
            }),
            bucket,
        )
        # Second call: reads cache
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-sonnet-4-6", {
                "input_tokens": 0,
                "output_tokens": 300,
                "cache_read_input_tokens": 3800,
                "cache_creation_input_tokens": 0,
            }),
            bucket,
        )
        s = bucket["claude-sonnet-4-6"]
        assert s["cache_write_tokens"] == 3800
        assert s["cache_read_tokens"] == 3800
        assert s["call_count"] == 2

    def test_cache_tokens_default_zero_when_absent(self):
        """Usage blocks without cache fields default to 0."""
        bucket: dict = {}
        executor_mod._accumulate_model_usage(
            self._assistant_line("claude-sonnet-4-6", {
                "input_tokens": 100,
                "output_tokens": 50,
            }),
            bucket,
        )
        assert bucket["claude-sonnet-4-6"]["cache_read_tokens"] == 0
        assert bucket["claude-sonnet-4-6"]["cache_write_tokens"] == 0


class TestFlushStoryModelUsage:
    def test_flush_writes_one_row_per_model(self):
        bucket = {
            "claude-opus-4-6": {"call_count": 3, "cost_usd": 0.75},
            "claude-haiku-4-5-20251001": {"call_count": 7, "cost_usd": 0.05},
        }
        executor_mod._flush_story_model_usage("TK-613", bucket)
        rows = executor_runs_db.get_story_model_usage("TK-613")
        assert len(rows) == 2
        by_model = {r["model"]: r for r in rows}
        assert by_model["claude-opus-4-6"]["call_count"] == 3
        assert by_model["claude-opus-4-6"]["cost_usd"] == pytest.approx(0.75)
        assert by_model["claude-haiku-4-5-20251001"]["call_count"] == 7
        assert by_model["claude-haiku-4-5-20251001"]["cost_usd"] == pytest.approx(
            0.05
        )

    def test_flush_empty_bucket_is_noop(self):
        executor_mod._flush_story_model_usage("TK-613", {})
        assert executor_runs_db.get_story_model_usage("TK-613") == []

    def test_flush_swallows_db_errors(self, monkeypatch):
        """DB failures must not abort the run — log and move on."""
        def _boom(**kw):
            raise RuntimeError("db down")

        monkeypatch.setattr(
            executor_runs_db, "record_story_model_usage", _boom
        )
        bucket = {"claude-opus-4-6": {"call_count": 1, "cost_usd": 0.10}}
        # Must not raise
        executor_mod._flush_story_model_usage("TK-613", bucket)


class TestSimulatedClaudeRun:
    """End-to-end: feed a realistic claude -p stream-json sequence through
    the accumulator, flush at the end, and verify one row per model with
    correct aggregate model + cost."""

    def test_run_with_single_model(self):
        bucket: dict = {}
        events = [
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 250,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                    "content": [{"type": "text", "text": "reading..."}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "usage": {
                        "input_tokens": 800,
                        "output_tokens": 400,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                    "content": [{"type": "text", "text": "editing..."}],
                },
            },
            {"type": "result", "result": "done", "session_id": "abc"},
        ]
        for evt in events:
            executor_mod._accumulate_model_usage(json.dumps(evt), bucket)
        executor_mod._flush_story_model_usage("TK-613", bucket)

        rows = executor_runs_db.get_story_model_usage("TK-613")
        assert len(rows) == 1
        row = rows[0]
        assert row["model"] == "claude-sonnet-4-6"
        assert row["call_count"] == 2
        # Sum of two usage blocks' costs
        expected = _cost_from_usage_dict(
            events[0]["message"]["usage"], model="claude-sonnet-4-6",
        ) + _cost_from_usage_dict(
            events[1]["message"]["usage"], model="claude-sonnet-4-6",
        )
        assert row["cost_usd"] == pytest.approx(expected)

    def test_run_with_brain_and_worker_models(self):
        """The cost-slide use case: haiku brain + opus worker split."""
        bucket: dict = {}
        haiku_usage = {
            "input_tokens": 2000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        opus_usage = {
            "input_tokens": 5000,
            "output_tokens": 2000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        events = [
            {"type": "assistant", "message": {
                "model": "claude-haiku-4-5-20251001",
                "usage": haiku_usage, "content": [],
            }},
            {"type": "assistant", "message": {
                "model": "claude-haiku-4-5-20251001",
                "usage": haiku_usage, "content": [],
            }},
            {"type": "assistant", "message": {
                "model": "claude-opus-4-6",
                "usage": opus_usage, "content": [],
            }},
        ]
        for evt in events:
            executor_mod._accumulate_model_usage(json.dumps(evt), bucket)
        executor_mod._flush_story_model_usage("TK-613", bucket)

        rows = executor_runs_db.get_story_model_usage("TK-613")
        assert len(rows) == 2
        by_model = {r["model"]: r for r in rows}

        assert by_model["claude-haiku-4-5-20251001"]["call_count"] == 2
        haiku_expected = 2 * _cost_from_usage_dict(
            haiku_usage, model="claude-haiku-4-5-20251001",
        )
        assert by_model["claude-haiku-4-5-20251001"]["cost_usd"] == pytest.approx(
            haiku_expected
        )

        assert by_model["claude-opus-4-6"]["call_count"] == 1
        opus_expected = _cost_from_usage_dict(
            opus_usage, model="claude-opus-4-6",
        )
        assert by_model["claude-opus-4-6"]["cost_usd"] == pytest.approx(
            opus_expected
        )
        # Opus cost should dominate — the whole point of splitting them out
        assert (
            by_model["claude-opus-4-6"]["cost_usd"]
            > by_model["claude-haiku-4-5-20251001"]["cost_usd"]
        )

    def test_cache_hit_run_stored_and_retrieved(self):
        """cache_read_tokens and cache_write_tokens round-trip through DB."""
        bucket: dict = {}
        # Simulate a 16-turn run: call 1 writes cache, calls 2-15 read it.
        events = []
        events.append({
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-6",
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 400,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 3800,
                },
                "content": [],
            },
        })
        for _ in range(15):
            events.append({
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 300,
                        "cache_read_input_tokens": 3800,
                        "cache_creation_input_tokens": 0,
                    },
                    "content": [],
                },
            })
        for evt in events:
            executor_mod._accumulate_model_usage(json.dumps(evt), bucket)
        executor_mod._flush_story_model_usage("TK-633", bucket)

        rows = executor_runs_db.get_story_model_usage("TK-633")
        assert len(rows) == 1
        row = rows[0]
        assert row["call_count"] == 16
        assert row["cache_write_tokens"] == 3800       # only call 1
        assert row["cache_read_tokens"] == 15 * 3800   # calls 2-16
