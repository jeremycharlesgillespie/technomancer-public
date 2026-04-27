"""Integration test for null project scenario in executor.

This test verifies that when an execution is run with a null idea_id (no project key),
the system handles it correctly and the resulting project column in story_phase_timings
is NULL.

WHO: QA Engineer
WHAT: Create a test case that runs an idea execution where idea_id is empty/null,
triggering the null-project path.
WHEN: Test suite runs.
WHERE: tests/unit/test_executor_null_project.py
WHY: Validates the end-to-end behavior of the fallback logic for missing projects.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board.executor import ExecutionState, _state_timer
from agent import story_timings
from agent.story_timings import init_db, _local


@pytest.fixture(autouse=True)
def _isolate_story_timings_db(tmp_path, monkeypatch):
    """Point story_timings at a temp DB for all tests in this module."""
    monkeypatch.setattr("agent.story_timings.DB_PATH", tmp_path / "story_timings.db")
    monkeypatch.setattr("agent.story_timings.DB_DIR", tmp_path)
    # Clear cached connection so we get a fresh one with the new DB path
    _local.__dict__.pop("conn", None)
    init_db()
    yield
    # Clean up the DB after the test
    conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
    conn.execute("DELETE FROM story_phase_timings")
    conn.commit()
    conn.close()


class TestNullProjectIntegration:
    """Integration test for null project scenario in executor."""

    def test_execution_with_null_idea_id_results_in_null_project(self):
        """Test that execution with idea_id=None results in NULL project column.

        This integration test:
        1. Creates an ExecutionState with None idea_id
        2. Uses _state_timer which calls _project_key_for
        3. Verifies the timing row in the database has project=NULL
        """
        # Create an ExecutionState with None idea_id
        state = ExecutionState(idea_id=None, run_id="test-run-null")

        # Use _state_timer which internally calls _project_key_for
        # This should return None for None idea_id
        timer = _state_timer(state, "executor.claude_work", metadata={"test": "data"})
        with timer:
            pass

        # Verify the timing row was created with NULL project. Read the
        # path off the live module — the autouse fixture monkeypatches
        # ``agent.story_timings.DB_PATH``, so importing the symbol at the
        # top of this file would freeze the pre-patch value and miss the
        # row entirely (it'd query the production DB instead).
        conn = sqlite3.connect(str(story_timings.DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM story_phase_timings WHERE run_id = ?",
            ("test-run-null",),
        )
        row = cursor.fetchone()
        conn.close()

        # Assertions
        assert row is not None, "Timing row should be created"
        assert row["story_id"] is None, "Story ID should be None"
        assert row["project"] is None, "Project should be NULL"
        assert row["phase"] == "executor.claude_work", "Phase should match"
        assert row["success"] == 1, "Phase should be marked as successful"
        assert row["metadata"] == '{"test": "data"}', "Metadata should be JSON-stringified"

    def test_multiple_phases_with_null_idea_id(self, tmp_path, monkeypatch):
        """Test that multiple phases with null idea_id all have NULL project.

        This test verifies that when idea_id is None, all timing rows
        for that execution have NULL project values.
        """
        # Create a temporary DB for this test
        temp_db = tmp_path / "story_timings_multi_null.db"
        monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
        monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
        # Clear cached connection so we get a fresh one with the new DB path
        _local.__dict__.pop("conn", None)
        init_db()

        # Create an ExecutionState with None idea_id
        state = ExecutionState(idea_id=None, run_id="test-run-multi-null")

        # Record multiple phases
        phases = ["executor.plan", "executor.code", "executor.test"]
        for phase in phases:
            timer = _state_timer(state, phase)
            with timer:
                pass

        # Verify all timing rows have NULL project
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT project, phase FROM story_phase_timings WHERE run_id = ?",
            ("test-run-multi-null",),
        )
        rows = cursor.fetchall()
        conn.close()

        # Assertions
        assert len(rows) == len(phases), f"Should have {len(phases)} timing rows"
        for row in rows:
            assert row["project"] is None, (
                f"All timing rows should have NULL project for null idea_id"
            )
            assert row["phase"] in phases, f"Phase {row['phase']} should be in expected phases"

    def test_null_project_with_exception(self, tmp_path, monkeypatch):
        """Test that exceptions during execution with null idea_id still record NULL project.

        This test verifies that even when an exception occurs, the timing row
        is created with NULL project.
        """
        # Create a temporary DB for this test
        temp_db = tmp_path / "story_timings_null_exception.db"
        monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
        monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
        # Clear cached connection so we get a fresh one with the new DB path
        _local.__dict__.pop("conn", None)
        init_db()

        # Create an ExecutionState with None idea_id
        state = ExecutionState(idea_id=None, run_id="test-run-null-exception")

        # Record a phase that raises an exception
        with pytest.raises(ValueError, match="test exception"):
            timer = _state_timer(state, "executor.claude_work")
            with timer:
                raise ValueError("test exception")

        # Verify the timing row was created with NULL project and success=0
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT project, success FROM story_phase_timings WHERE run_id = ?",
            ("test-run-null-exception",),
        )
        row = cursor.fetchone()
        conn.close()

        # Assertions
        assert row is not None, "Timing row should be created even on exception"
        assert row["project"] is None, "Project should be NULL even on exception"
        assert row["success"] == 0, "Phase should be marked as failed"

    def test_null_project_with_metadata(self, tmp_path, monkeypatch):
        """Test that NULL project works correctly with various metadata types.

        This test verifies that the NULL project handling works correctly
        when different types of metadata are passed.
        """
        # Create a temporary DB for this test
        temp_db = tmp_path / "story_timings_null_metadata.db"
        monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
        monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
        # Clear cached connection so we get a fresh one with the new DB path
        _local.__dict__.pop("conn", None)
        init_db()

        # Create an ExecutionState with None idea_id
        state = ExecutionState(idea_id=None, run_id="test-run-null-metadata")

        # Record a phase with different metadata types
        test_cases = [
            {"key": "value"},
            {"list": [1, 2, 3]},
            {"nested": {"a": 1, "b": 2}},
            "simple string",
            123,
        ]

        for i, metadata in enumerate(test_cases):
            timer = _state_timer(state, f"executor.phase_{i}", metadata=metadata)
            with timer:
                pass

        # Verify all timing rows have NULL project
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT project, metadata FROM story_phase_timings WHERE run_id = ?",
            ("test-run-null-metadata",),
        )
        rows = cursor.fetchall()
        conn.close()

        # Assertions
        assert len(rows) == len(test_cases), f"Should have {len(test_cases)} timing rows"
        for row in rows:
            assert row["project"] is None, (
                f"All timing rows should have NULL project for null idea_id"
            )
            assert row["metadata"] is not None, "Metadata should be stored"

    def test_concurrent_null_project_executions(self, tmp_path, monkeypatch):
        """Test that multiple concurrent executions with null idea_id work correctly.

        This test verifies that the NULL project handling is thread-safe
        when multiple executions run concurrently.
        """
        # Create a temporary DB for this test
        temp_db = tmp_path / "story_timings_concurrent_null.db"
        monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
        monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
        # Clear cached connection so we get a fresh one with the new DB path
        _local.__dict__.pop("conn", None)
        init_db()

        # Create multiple ExecutionState instances with None idea_id
        states = [
            ExecutionState(idea_id=None, run_id=f"test-run-concurrent-{i}")
            for i in range(5)
        ]

        # Record phases concurrently
        def record_phase(state, phase_name):
            timer = _state_timer(state, phase_name)
            with timer:
                pass

        threads = []
        for i, state in enumerate(states):
            thread = threading.Thread(
                target=record_phase, args=(state, f"executor.phase_{i}")
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Verify all timing rows have NULL project
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT project, run_id FROM story_phase_timings ORDER BY run_id",
        )
        rows = cursor.fetchall()
        conn.close()

        # Assertions
        assert len(rows) == len(states), f"Should have {len(states)} timing rows"
        for row in rows:
            assert row["project"] is None, (
                f"All timing rows should have NULL project for null idea_id"
            )
            assert row["run_id"] in [f"test-run-concurrent-{i}" for i in range(5)], (
                "Run ID should match one of the expected states"
            )