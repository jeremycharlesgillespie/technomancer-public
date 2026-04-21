"""
Integration tests for startup_checks readiness gate with mocked daily_stats.

Verifies that the readiness gate correctly handles:
- Missing daily_stats database (raises RuntimeError)
- Valid daily_stats database (passes validation)
- Mocked validate_daily_stats_db function for isolated testing

These tests ensure no hidden side-effects or import cycles break the test suite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import daily_stats
from agent.startup_checks import Check, CheckResult, ReadinessReport, run_readiness_checks, STATUS_FAIL, STATUS_OK, STATUS_TIMEOUT


# =============================================================================
# Fixtures — isolate daily_stats DB for each test
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_daily_stats_db(tmp_path, monkeypatch):
    """Point daily_stats at a temporary SQLite DB for each test."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr(daily_stats, "DB_DIR", tmp_path)
    monkeypatch.setattr(daily_stats, "DB_PATH", db_path)
    # Clear thread-local connection cache
    daily_stats._local.__dict__.pop("conn", None)
    yield
    # Cleanup: close connection if open
    conn = getattr(daily_stats._local, "conn", None)
    if conn:
        conn.close()
        daily_stats._local.__dict__.pop("conn", None)


# =============================================================================
# Test class: TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests for the readiness gate with mocked daily_stats."""

    def test_full_gate_with_missing_stats(self, tmp_path, monkeypatch):
        """
        Verify readiness gate fails when daily_stats database is missing.

        The validate_daily_stats_db function should raise RuntimeError when
        the database doesn't exist, and the readiness gate should capture
        this as a failed check.
        """
        # Ensure the database file doesn't exist
        db_path = tmp_path / "daily_stats.db"
        assert not db_path.exists()

        # Mock validate_daily_stats_db to raise RuntimeError (simulating missing DB)
        def mock_validate_missing():
            raise RuntimeError("Daily stats database missing: daily_stats.db")

        monkeypatch.setattr(daily_stats, "validate_daily_stats_db", mock_validate_missing)

        # Create a check that validates daily_stats
        validate_check = Check(
            name="daily_stats",
            fn=daily_stats.validate_daily_stats_db,
            required=True,
            timeout=5.0,
        )

        # Run the readiness gate with our custom check
        report = run_readiness_checks(checks=[validate_check])

        # Verify the report indicates failure
        assert report.ok is False
        assert not report.skipped

        # Find the failed check
        failed_check = next(
            (r for r in report.results if r.name == "daily_stats"),
            None,
        )
        assert failed_check is not None
        assert failed_check.status == STATUS_FAIL
        assert failed_check.error is not None
        assert "Daily stats database missing" in failed_check.error

    def test_full_gate_with_valid_stats(self, tmp_path, monkeypatch):
        """
        Verify readiness gate passes when daily_stats database exists and is valid.

        The validate_daily_stats_db function should return True when the
        database exists and is accessible, and the readiness gate should
        capture this as a successful check.
        """
        # Create a temporary database with the daily_stats table
        db_path = tmp_path / "daily_stats.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE daily_stats (
                date                   TEXT    NOT NULL,
                project                TEXT    NOT NULL,
                shipped                INTEGER NOT NULL DEFAULT 0,
                failed                 INTEGER NOT NULL DEFAULT 0,
                split_children         INTEGER NOT NULL DEFAULT 0,
                cost_usd               REAL    NOT NULL DEFAULT 0.0,
                p50_wall_s             REAL    NOT NULL DEFAULT 0.0,
                p95_wall_s             REAL    NOT NULL DEFAULT 0.0,
                loc_added              INTEGER NOT NULL DEFAULT 0,
                loc_removed            INTEGER NOT NULL DEFAULT 0,
                first_attempt_success  INTEGER NOT NULL DEFAULT 0,
                splitter_child_success INTEGER,
                splitter_child_fail    INTEGER,
                phase_timings_json     TEXT,
                PRIMARY KEY (date, project)
            )
        """)
        conn.execute("INSERT INTO daily_stats VALUES ('2026-01-01', 'TK', 10, 2, 5, 0.5, 1.0, 2.0, 100, 50, 1, 0, 0, '{}')")
        conn.commit()
        conn.close()

        # Mock validate_daily_stats_db to return True (simulating valid DB)
        def mock_validate_valid():
            return True

        monkeypatch.setattr(daily_stats, "validate_daily_stats_db", mock_validate_valid)

        # Create a check that validates daily_stats
        validate_check = Check(
            name="daily_stats",
            fn=daily_stats.validate_daily_stats_db,
            required=True,
            timeout=5.0,
        )

        # Run the readiness gate with our custom check
        report = run_readiness_checks(checks=[validate_check])

        # Verify the report indicates success
        assert report.ok is True
        assert not report.skipped

        # Find the successful check
        success_check = next(
            (r for r in report.results if r.name == "daily_stats"),
            None,
        )
        assert success_check is not None
        assert success_check.status == STATUS_OK
        assert success_check.error is None
        assert success_check.attempts == 1

    def test_full_gate_with_timeout(self, tmp_path, monkeypatch):
        """
        Verify readiness gate handles timeout correctly for daily_stats validation.

        When validate_daily_stats_db takes too long, the check should timeout
        and be marked as such in the report.
        """
        # Create a temporary database
        db_path = tmp_path / "daily_stats.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE daily_stats (date TEXT, project TEXT)")
        conn.commit()
        conn.close()

        # Mock validate_daily_stats_db to simulate a slow operation
        def mock_validate_slow():
            time.sleep(10)  # Simulate a slow operation
            return True

        import time
        monkeypatch.setattr(daily_stats, "validate_daily_stats_db", mock_validate_slow)

        # Create a check with a short timeout
        validate_check = Check(
            name="daily_stats",
            fn=daily_stats.validate_daily_stats_db,
            required=True,
            timeout=0.1,  # Very short timeout to trigger timeout
        )

        # Run the readiness gate with our custom check
        report = run_readiness_checks(checks=[validate_check])

        # Verify the report indicates timeout
        assert report.ok is False
        assert not report.skipped

        # Find the timed-out check
        timeout_check = next(
            (r for r in report.results if r.name == "daily_stats"),
            None,
        )
        assert timeout_check is not None
        assert timeout_check.status == STATUS_TIMEOUT
        assert timeout_check.error is not None
        assert "timeout" in timeout_check.error.lower()

    def test_full_gate_with_corrupted_db(self, tmp_path, monkeypatch):
        """
        Verify readiness gate handles corrupted daily_stats database correctly.

        When the database exists but is corrupted (e.g., missing table),
        the check should fail with an appropriate error message.
        """
        # Create a temporary database without the daily_stats table
        db_path = tmp_path / "daily_stats.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other_table (id INTEGER)")
        conn.commit()
        conn.close()

        # Mock validate_daily_stats_db to raise OperationalError (simulating corrupted DB)
        def mock_validate_corrupted():
            raise sqlite3.OperationalError("no such table: daily_stats")

        monkeypatch.setattr(daily_stats, "validate_daily_stats_db", mock_validate_corrupted)

        # Create a check that validates daily_stats
        validate_check = Check(
            name="daily_stats",
            fn=daily_stats.validate_daily_stats_db,
            required=True,
            timeout=5.0,
        )

        # Run the readiness gate with our custom check
        report = run_readiness_checks(checks=[validate_check])

        # Verify the report indicates failure
        assert report.ok is False
        assert not report.skipped

        # Find the failed check
        failed_check = next(
            (r for r in report.results if r.name == "daily_stats"),
            None,
        )
        assert failed_check is not None
        assert failed_check.status == STATUS_FAIL
        assert failed_check.error is not None
        assert "no such table" in failed_check.error.lower()

    def test_full_gate_with_multiple_checks(self, tmp_path, monkeypatch):
        """
        Verify readiness gate handles multiple checks including daily_stats.

        This test ensures that when multiple checks are run, the daily_stats
        check integrates correctly with other checks.
        """
        # Create a temporary database
        db_path = tmp_path / "daily_stats.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE daily_stats (date TEXT, project TEXT)")
        conn.commit()
        conn.close()

        # Mock validate_daily_stats_db to return True
        def mock_validate_valid():
            return True

        monkeypatch.setattr(daily_stats, "validate_daily_stats_db", mock_validate_valid)

        # Create multiple checks including daily_stats
        checks = [
            Check(name="ollama", fn=lambda: None, required=True, timeout=5.0),
            Check(name="vault", fn=lambda: None, required=True, timeout=5.0),
            Check(name="executor_db", fn=lambda: None, required=True, timeout=5.0),
            Check(
                name="daily_stats",
                fn=daily_stats.validate_daily_stats_db,
                required=True,
                timeout=5.0,
            ),
        ]

        # Run the readiness gate with multiple checks
        report = run_readiness_checks(checks=checks)

        # Verify all checks passed
        assert report.ok is True
        assert not report.skipped

        # Verify we have 4 results
        assert len(report.results) == 4

        # Find the daily_stats check
        daily_stats_check = next(
            (r for r in report.results if r.name == "daily_stats"),
            None,
        )
        assert daily_stats_check is not None
        assert daily_stats_check.status == STATUS_OK
        assert daily_stats_check.error is None
