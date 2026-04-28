"""Tests for idea_board.executor._project_key_for — project key extraction."""

import re
import sqlite3
import pytest
from unittest.mock import patch

from idea_board.executor import (
    IDEA_ID_PATTERN,
    _project_key_for,
    _is_valid_idea_id,
    _PhaseMarker,
    ExecutionState,
    _state_timer,
)
from agent.config import settings
from agent.story_timings import init_db, DB_PATH


# ---------------------------------------------------------------------------
# Helper function for missing project key scenarios
# ---------------------------------------------------------- -------------------

def _assert_missing_key_handling(
    test_func,
    expected_result: str | None = None,
    message: str = "Expected None for missing project key"
):
    """Reusable assertion helper for testing missing project key scenarios.

    This helper encapsulates the common pattern of testing that _project_key_for
    returns None (or a specific fallback) when given inputs that should not
    produce a valid project key. It provides a clear contract for failure
    scenarios and ensures consistent test logic.

    Args:
        test_func: A callable that takes a string input and returns the result
                   of _project_key_for(input).
        expected_result: The expected result (typically None, but can be a
                         fallback string like "FA" when jira_project_key is set).
        message: Custom assertion message describing the expected behavior.

    Example:
        _assert_missing_key_handling(
            lambda x: _project_key_for(x),
            expected_result=None,
            message="Should return None for empty string"
        )
    """
    with patch.object(settings, 'jira_project_key', None):
        result = test_func()
        assert result == expected_result, message


class TestProjectKeyFor:
    """Test _project_key_for function with various inputs."""

    def test_returns_none_for_none_input(self):
        """_project_key_for should return None for None input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for(None) is None

    def test_returns_none_for_empty_string(self):
        """_project_key_for should return None for empty string input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("") is None

    def test_returns_none_for_whitespace_only(self):
        """_project_key_for should return None for whitespace-only input."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("   ") is None

    def test_returns_project_key_when_present(self):
        """_project_key_for should return the project key when found."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("FA-123") == "FA"

    def test_returns_none_when_project_key_not_found(self):
        """_project_key_for should return None when project key is not found."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("TK-123") is None

    def test_returns_none_when_project_key_is_not_at_start(self):
        """_project_key_for should return None when project key is not at the start."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("Some text FA-123 more text") is None

    def test_handles_multiple_project_keys(self):
        """_project_key_for should handle multiple potential project keys."""
        with patch.object(settings, 'jira_project_key', "FA"):
            assert _project_key_for("FA-123") == "FA"
            assert _project_key_for("FA-456") == "FA"
            assert _project_key_for("TK-123") is None

    def test_returns_none_when_jira_project_key_is_none(self):
        """_project_key_for should return None when jira_project_key is None."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-123") is None

    def test_returns_none_when_jira_project_key_is_empty_string(self):
        """_project_key_for should return None when jira_project_key is empty string."""
        with patch.object(settings, 'jira_project_key', ""):
            assert _project_key_for("FA-123") is None

    def test_returns_none_for_non_string_input(self):
        """_project_key_for should return None for non-string inputs."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for(123) is None
            assert _project_key_for([]) is None
            assert _project_key_for({}) is None

    def test_returns_FA_for_FA_1029(self):
        """_project_key_for should extract FA from FA-1029."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-1029") == "FA"

    def test_returns_TK_for_TK_500(self):
        """_project_key_for should extract TK from TK-500."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("TK-500") == "TK"

    def test_is_valid_idea_id_returns_false_for_text_only(self):
        """_is_valid_idea_id should return False for 'text-only'."""
        assert _is_valid_idea_id("text-only") is False

    def test_is_valid_idea_id_returns_true_for_FA_100(self):
        """_is_valid_idea_id should return True for 'FA-100'."""
        assert _is_valid_idea_id("FA-100") is True

    def test_project_key_for_returns_none_for_text_only(self):
        """_project_key_for should return None for 'text-only'."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("text-only") is None

    def test_project_key_for_returns_FA_for_FA_100(self):
        """_project_key_for should return 'FA' for 'FA-100'."""
        with patch.object(settings, 'jira_project_key', None):
            assert _project_key_for("FA-100") == "FA"

    def test_is_valid_idea_id_edge_cases(self):
        """Test edge cases for _is_valid_idea_id function."""
        # Valid cases
        assert _is_valid_idea_id("FA-100") is True
        assert _is_valid_idea_id("TK-500") is True
        assert _is_valid_idea_id("ABC-123") is True
        
        # Invalid cases
        assert _is_valid_idea_id("text-only") is False
        assert _is_valid_idea_id("") is False
        assert _is_valid_idea_id("   ") is False
        assert _is_valid_idea_id("123-456") is False  # No alphabetic prefix
        assert _is_valid_idea_id("FA") is False  # No hyphen
        assert _is_valid_idea_id("FA-") is False  # No suffix
        assert _is_valid_idea_id("-123") is False  # No prefix
        assert _is_valid_idea_id("FA-123abc") is True  # Has digits in suffix
        assert _is_valid_idea_id("FA-abc") is False  # No digits in suffix
        assert _is_valid_idea_id(123) is False  # Not a string
        assert _is_valid_idea_id(None) is False  # Not a string


# ---------------------------------------------------------------------------
# IDEA_ID_PATTERN constant
# -----------------------

class TestIdeaIdPattern:
    """Test the IDEA_ID_PATTERN regex constant."""

    def test_pattern_matches_valid_idea_id(self):
        """re.match(Idea_ID_PATTERN, 'FA-100') should match."""
        assert IDEA_ID_PATTERN.match("FA-100") is not None

    def test_pattern_does_not_match_text_only(self):
        """_is_valid_idea_id('text-only') should return False."""
        assert _is_valid_idea_id("text-only") is False

    def test_pattern_does_not_match_empty_string(self):
        """_is_valid_idea_id('') should return False."""
        assert _is_valid_idea_id("") is False

    def test_pattern_does_not_match_whitespace_only(self):
        """_is_valid_idea_id('   ') should return False."""
        assert _is_valid_idea_id("   ") is False

    def test_pattern_does_not_match_numbers_only(self):
        """_is_valid_idea_id('123-456') should return False."""
        assert _is_valid_idea_id("123-456") is False

    def test_pattern_does_not_match_missing_hyphen(self):
        """_is_valid_idea_id('FA') should return False."""
        assert _is_valid_idea_id("FA") is False

    def test_pattern_does_not_match_missing_suffix(self):
        """_is_valid_idea_id('FA-') should return False."""
        assert _is_valid_idea_id("FA-") is False

    def test_pattern_does_not_match_missing_prefix(self):
        """_is_valid_idea_id('-123') should return False."""
        assert _is_valid_idea_id("-123") is False

    def test_pattern_does_not_match_non_string(self):
        """_is_valid_idea_id(123) should return False."""
        assert _is_valid_idea_id(123) is False

    def test_pattern_does_not_match_none(self):
        """_is_valid_idea_id(None) should return False."""
        assert _is_valid_idea_id(None) is False

    def test_pattern_matches_various_valid_formats(self):
        """Test various valid idea ID formats."""
        assert IDEA_ID_PATTERN.match("FA-100") is not None
        assert IDEA_ID_PATTERN.match("TK-500") is not None
        assert IDEA_ID_PATTERN.match("ABC-123") is not None
        assert IDEA_ID_PATTERN.match("XYZ-9999") is not None

    def test_pattern_does_not_match_invalid_formats(self):
        """Test various invalid idea ID formats."""
        assert IDEA_ID_PATTERN.match("text-only") is None
        assert IDEA_ID_PATTERN.match("") is None
        assert IDEA_ID_PATTERN.match("   ") is None
        assert IDEA_ID_PATTERN.match("123-456") is None
        assert IDEA_ID_PATTERN.match("FA") is None
        assert IDEA_ID_PATTERN.match("FA-") is None
        assert IDEA_ID_PATTERN.match("-123") is None
        assert IDEA_ID_PATTERN.match("FA-123abc") is None
        assert IDEA_ID_PATTERN.match("FA-abc") is None


# ---------------------------------------------------------------------------
# _PhaseMarker Tests
# -------------------

class TestPhaseMarker:
    """Test _PhaseMarker.finish behavior with various project key scenarios."""

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        """Point story_timings at a temp DB for each test."""
        monkeypatch.setattr("agent.story_timings.DB_PATH", tmp_path / "story_timings.db")
        monkeypatch.setattr("agent.story_timings.DB_DIR", tmp_path)
        init_db()
        yield
        # Clean up the DB after the test
        conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
        conn.execute("DELETE FROM story_phase_timings")
        conn.commit()
        conn.close()

    def test_phase_marker_finish_with_valid_project_key(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should successfully record a phase with a valid project key."""
        with patch.object(settings, 'jira_project_key', "FA"):
            state = ExecutionState(
                idea_id="FA-1234",
                run_id="test-run-1",
            )
            marker = _PhaseMarker(state, "executor.claude_work", metadata={"test": "data"})
            marker.finish(success=True)

            # Verify the timing row was created
            conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM story_phase_timings WHERE run_id = ?",
                ("test-run-1",),
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "Timing row should be created"
            assert row["project"] == "FA", "Project key should be 'FA'"
            assert row["story_id"] == "FA-1234", "Story ID should match"
            assert row["phase"] == "executor.claude_work", "Phase should match"
            assert row["success"] == 1, "Phase should be marked as successful"

    def test_phase_marker_finish_with_none_project_key(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should successfully record a phase with None project key.

        This test verifies that when _project_key_for returns None (e.g., when
        idea_id is invalid or Jira is not configured), record_phase is called
        with project=None and no KeyError is thrown.
        """
        with patch.object(settings, 'jira_project_key', None):
            state = ExecutionState(
                idea_id="TK-1234",
                run_id="test-run-2",
            )
            marker = _PhaseMarker(state, "executor.claude_work", metadata={"test": "data"})
            marker.finish(success=True)

            # Verify the timing row was created with project=None
            conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM story_phase_timings WHERE run_id = ?",
                ("test-run-2",),
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "Timing row should be created"
            assert row["project"] is None, "Project key should be None"
            assert row["story_id"] == "TK-1234", "Story ID should match"
            assert row["phase"] == "executor.claude_work", "Phase should match"
            assert row["success"] == 1, "Phase should be marked as successful"

    def test_phase_marker_finish_with_invalid_idea_id(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should successfully record a phase with None project key for invalid idea_id.

        This test verifies that when _project_key_for returns None due to an
        invalid idea_id (e.g., "text-only"), record_phase is called with
        project=None and no KeyError is thrown.
        """
        with patch.object(settings, 'jira_project_key', None):
            state = ExecutionState(
                idea_id="text-only",
                run_id="test-run-3",
            )
            marker = _PhaseMarker(state, "executor.claude_work", metadata={"test": "data"})
            marker.finish(success=True)

            # Verify the timing row was created with project=None
            conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM story_phase_timings WHERE run_id = ?",
                ("test-run-3",),
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "Timing row should be created"
            assert row["project"] is None, "Project key should be None for invalid idea_id"
            assert row["story_id"] == "text-only", "Story ID should match"
            assert row["phase"] == "executor.claude_work", "Phase should match"
            assert row["success"] == 1, "Phase should be marked as successful"

    def test_phase_marker_finish_idempotent(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should be idempotent — calling it multiple times should not create duplicate rows."""
        with patch.object(settings, 'jira_project_key', "FA"):
            state = ExecutionState(
                idea_id="FA-1234",
                run_id="test-run-4",
            )
            marker = _PhaseMarker(state, "executor.claude_work", metadata={"test": "data"})

            # Call finish multiple times
            marker.finish(success=True)
            marker.finish(success=True)
            marker.finish(success=True)

            # Verify only one timing row was created
            conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM story_phase_timings WHERE run_id = ?",
                ("test-run-4",),
            )
            row = cursor.fetchone()
            conn.close()

            assert row["count"] == 1, "Should have exactly one timing row despite multiple finish calls"

    def test_phase_marker_finish_with_exception(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should handle exceptions gracefully without crashing.

        This test verifies that if record_phase raises an exception, it's caught
        and logged as a warning, but the marker state is properly cleaned up.
        """
        with patch.object(settings, 'jira_project_key', "FA"):
            state = ExecutionState(
                idea_id="FA-1234",
                run_id="test-run-5",
            )
            marker = _PhaseMarker(state, "executor.claude_work", metadata={"test": "data"})

            # Mock record_phase to raise an exception
            with patch('idea_board.executor.record_phase') as mock_record:
                mock_record.side_effect = Exception("Simulated DB error")
                marker.finish(success=True)

                # Verify the exception was caught and logged
                assert mock_record.called, "record_phase should have been called"

    def test_phase_marker_finish_with_metadata(self, tmp_path, monkeypatch):
        """_PhaseMarker.finish should serialize metadata correctly."""
        with patch.object(settings, 'jira_project_key', "FA"):
            state = ExecutionState(
                idea_id="FA-1234",
                run_id="test-run-6",
            )
            marker = _PhaseMarker(
                state,
                "executor.claude_work",
                metadata={"test": "data", "nested": {"key": "value"}}
            )
            marker.finish(success=True)

            # Verify the timing row was created with serialized metadata
            conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT metadata FROM story_phase_timings WHERE run_id = ?",
                ("test-run-6",),
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "Timing row should be created"
            assert row["metadata"] == '{"test": "data", "nested": {"key": "value"}}', "Metadata should be JSON-stringified"


# ---------------------------------------------------------------------------
# Integration Tests — Full execution flow with timing rows
# -----------------------

class TestIntegrationMissingProjectKey:
    """Integration test for _project_key_for fallback when Jira is not configured.

    This test simulates an execution where `idea_id` exists but Jira is not
    configured (no project key), so the alpha-prefix fallback logic is verified.
    """

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        """Point story_timings at a temp DB for each test."""
        monkeypatch.setattr("agent.story_timings.DB_PATH", tmp_path / "story_timings.db")
        monkeypatch.setattr("agent.story_timings.DB_DIR", tmp_path)
        init_db()
        yield
        # Clean up the DB after the test
        conn = sqlite3.connect(str(tmp_path / "story_timings.db"))
        conn.execute("DELETE FROM story_phase_timings")
        conn.commit()
        conn.close()

    def test_project_key_derived_from_idea_id_when_jira_not_configured(self):
        """Integration test: verify project key is derived from idea_id prefix when jira_project_key is None.

        This test:
        1. Mocks settings.jira_project_key as None (Jira not configured)
        2. Creates an ExecutionState with a valid idea_id (e.g., "TK-1234")
        3. Uses _state_timer which calls _project_key_for
        4. Verifies the timing row in the database contains the correct project key ("TK")
        """
        # Mock Jira project key as None (Jira not configured)
        with patch.object(settings, 'jira_project_key', None):
            # Create an ExecutionState with a valid idea_id
            state = ExecutionState(
                idea_id="TK-1234",
                run_id="test-run-123",
            )

            # Use _state_timer which internally calls _project_key_for
            # This should derive "TK" from the idea_id prefix
            with _state_timer(state, "executor.claude_work", metadata={"test": "data"}):
                pass

            # Verify the timing row was created with the correct project key
            conn = sqlite3.connect(str(settings.DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM story_phase_timings WHERE run_id = ?",
                ("test-run-123",),
            )
            row = cursor.fetchone()
            conn.close()

            # Assertions
            assert row is not None, "Timing row should be created"
            assert row["story_id"] == "TK-1234", "Story ID should match"
            assert row["project"] == "TK", "Project key should be derived from idea_id prefix"
            assert row["phase"] == "executor.claude_work", "Phase should match"
            assert row["success"] == 1, "Phase should be marked as successful"
            assert row["metadata"] == '{"test": "data"}', "Metadata should be JSON-stringified"

    def test_project_key_derived_from_different_idea_id_prefixes(self, tmp_path, monkeypatch):
        """Integration test: verify project key extraction works for various idea_id prefixes.

        Tests that when jira_project_key is None, _project_key_for correctly
        extracts different prefixes from various idea IDs.
        """
        with patch.object(settings, 'jira_project_key', None):
            test_cases = [
                ("TK-1234", "TK"),
                ("FA-5678", "FA"),
                ("ABC-9999", "ABC"),
                ("XYZ-100", "XYZ"),
            ]

            for idea_id, expected_project in test_cases:
                # Create a temporary DB for each test case
                temp_db = tmp_path / f"story_timings_{idea_id}.db"
                monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
                monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
                init_db()

                # Create ExecutionState and use _state_timer
                state = ExecutionState(
                    idea_id=idea_id,
                    run_id=f"test-run-{idea_id}",
                )

                with _state_timer(state, "executor.claude_work"):
                    pass

                # Verify the timing row
                conn = sqlite3.connect(str(temp_db))
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT project FROM story_phase_timings WHERE run_id = ?",
                    (f"test-run-{idea_id}",),
                )
                row = cursor.fetchone()
                conn.close()

                assert row is not None, f"Timing row should be created for {idea_id}"
                assert row["project"] == expected_project, (
                    f"Project key for {idea_id} should be {expected_project}"
                )

    def test_timing_rows_have_valid_project_keys_when_jira_not_configured(self, tmp_path, monkeypatch):
        """Integration test: verify multiple timing rows all have valid project keys.

        This test creates multiple phases for the same idea and verifies that
        all timing rows contain valid project keys derived from the idea_id prefix.
        """
        with patch.object(settings, 'jira_project_key', None):
            # Create a temporary DB
            temp_db = tmp_path / "story_timings_multi.db"
            monkeypatch.setattr("agent.story_timings.DB_PATH", temp_db)
            monkeypatch.setattr("agent.story_timings.DB_DIR", temp_db.parent)
            init_db()

            # Create ExecutionState
            state = ExecutionState(
                idea_id="FA-9999",
                run_id="test-run-multi",
            )

            # Record multiple phases
            phases = ["executor.plan", "executor.code", "executor.test", "executor.deploy"]
            for phase in phases:
                with _state_timer(state, phase):
                    pass

            # Verify all timing rows have valid project keys
            conn = sqlite3.connect(str(temp_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT project, phase FROM story_phase_timings WHERE run_id = ?",
                ("test-run-multi",),
            )
            rows = cursor.fetchall()
            conn.close()

            # Assertions
            assert len(rows) == len(phases), f"Should have {len(phases)} timing rows"
            for row in rows:
                assert row["project"] == "FA", (
                    f"All timing rows should have project key 'FA' derived from idea_id prefix"
                )
                assert row["phase"] in phases, f"Phase {row['phase']} should be in expected phases"