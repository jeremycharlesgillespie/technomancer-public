"""Tests for scripts.generate_stats_ppt — stats data aggregation logic."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import agent

from scripts import generate_stats_ppt


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point daily_stats at a temp SQLite DB and reset the connection cache."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr("agent.daily_stats.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.daily_stats.DB_PATH", db_path)
    agent.daily_stats._local.__dict__.pop("conn", None)
    yield
    conn = getattr(agent.daily_stats._local, "conn", None)
    if conn:
        conn.close()
        agent.daily_stats._local.__dict__.pop("conn", None)


@pytest.fixture
def sample_daily_stats_data(tmp_path):
    """Create sample daily_stats rows for testing."""
    db_path = tmp_path / "daily_stats.db"
    import agent.daily_stats as daily_stats_module

    daily_stats_module.init_db()
    conn = daily_stats_module._get_conn()

    # Insert sample data for multiple dates
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")

    conn.execute(
        """
        INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (yesterday, "TK", 5, 1, 12.50, 4),
    )
    conn.execute(
        """
        INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (today, "TK", 3, 0, 8.25, 3),
    )
    conn.execute(
        """
        INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (today, "OTHER", 2, 1, 5.00, 1),
    )
    conn.commit()

    return db_path


class TestAggregateStats:
    """Tests for aggregate_stats function."""

    def test_aggregates_multiple_rows(self, sample_daily_stats_data):
        """Should sum values across all matching rows."""
        result = generate_stats_ppt.aggregate_stats("2024-01-01", project="TK")

        assert result["shipped"] == 8  # 5 + 3
        assert result["failed"] == 1  # 1 + 0
        assert result["cost_usd"] == 20.75  # 12.50 + 8.25
        assert result["first_attempt_success"] == 7  # 4 + 3

    def test_filters_by_project(self, sample_daily_stats_data):
        """Should only include rows matching the project filter."""
        result = generate_stats_ppt.aggregate_stats("2024-01-01", project="OTHER")

        assert result["shipped"] == 2  # Only OTHER project
        assert result["failed"] == 1
        assert result["cost_usd"] == 5.00
        assert result["first_attempt_success"] == 1

    def test_filters_by_date(self, sample_daily_stats_data):
        """Should only include rows on or after the since date."""
        result = generate_stats_ppt.aggregate_stats("2024-01-01", project="TK")

        assert result["shipped"] == 8  # Both TK rows
        assert result["failed"] == 1
        assert result["cost_usd"] == 20.75
        assert result["first_attempt_success"] == 7

    def test_empty_date_range(self, sample_daily_stats_data):
        """Should return zeros when no data exists for the date range."""
        result = generate_stats_ppt.aggregate_stats("2099-12-31", project="TK")

        assert result["shipped"] == 0
        assert result["failed"] == 0
        assert result["cost_usd"] == 0.0
        assert result["first_attempt_success"] == 0

    def test_no_data_at_all(self, tmp_path):
        """Should return zeros when DB is empty."""
        result = generate_stats_ppt.aggregate_stats("2024-01-01")

        assert result["shipped"] == 0
        assert result["failed"] == 0
        assert result["cost_usd"] == 0.0
        assert result["first_attempt_success"] == 0

    def test_handles_missing_columns(self, tmp_path):
        """Should gracefully handle rows with missing columns."""
        import agent.daily_stats as daily_stats_module

        daily_stats_module.init_db()
        conn = daily_stats_module._get_conn()

        # Insert a row with missing columns
        conn.execute(
            """
            INSERT INTO daily_stats (date, project)
            VALUES (?, ?)
        """,
            ("2024-01-01", "TK"),
        )
        conn.commit()

        result = generate_stats_ppt.aggregate_stats("2024-01-01")

        assert result["shipped"] == 0  # Missing column defaults to 0
        assert result["failed"] == 0
        assert result["cost_usd"] == 0.0
        assert result["first_attempt_success"] == 0

    def test_handles_sql_error(self, tmp_path, monkeypatch):
        """Should return zeros on SQL errors without raising."""
        import agent.daily_stats as daily_stats_module

        daily_stats_module.init_db()
        conn = daily_stats_module._get_conn()

        # Insert a row
        conn.execute(
            """
            INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            ("2024-01-01", "TK", 5, 1, 10.00, 4),
        )
        conn.commit()

        # Mock get_rows to raise an exception
        with patch.object(daily_stats_module, "get_rows", side_effect=Exception("SQL error")):
            result = generate_stats_ppt.aggregate_stats("2024-01-01")

        # Should return zeros instead of raising
        assert result["shipped"] == 0
        assert result["failed"] == 0
        assert result["cost_usd"] == 0.0
        assert result["first_attempt_success"] == 0

    def test_handles_none_project(self, sample_daily_stats_data):
        """Should aggregate all projects when project is None."""
        result = generate_stats_ppt.aggregate_stats("2024-01-01", project=None)

        assert result["shipped"] == 10  # TK (8) + OTHER (2)
        assert result["failed"] == 2  # TK (1) + OTHER (1)
        assert result["cost_usd"] == 25.75  # TK (20.75) + OTHER (5.00)
        assert result["first_attempt_success"] == 8  # TK (7) + OTHER (1)

    def test_handles_zero_values(self, tmp_path):
        """Should correctly sum zero values."""
        import agent.daily_stats as daily_stats_module

        daily_stats_module.init_db()
        conn = daily_stats_module._get_conn()

        # Insert rows with zero values
        conn.execute(
            """
            INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            ("2024-01-01", "TK", 0, 0, 0.0, 0),
        )
        conn.execute(
            """
            INSERT INTO daily_stats (date, project, shipped, failed, cost_usd, first_attempt_success)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            ("2024-01-02", "TK", 0, 0, 0.0, 0),
        )
        conn.commit()

        result = generate_stats_ppt.aggregate_stats("2024-01-01")

        assert result["shipped"] == 0
        assert result["failed"] == 0
        assert result["cost_usd"] == 0.0
        assert result["first_attempt_success"] == 0