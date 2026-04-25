"""Tests for executor_runs_db.py integration with leak_counter."""

import pytest
from unittest.mock import patch

from agent.executor_runs_db import init_db


class TestExecutorRunsDBIntegration:
    """Test that executor_runs_db works with the new leak_counter module."""

    def test_init_db_does_not_fail(self):
        """Test that init_db still works after our changes."""
        # This should not raise an exception
        init_db()
        
    def test_leak_counter_import_works(self):
        """Test that leak_counter can be imported from executor_runs_db."""
        # This tests that our import changes didn't break anything
        from agent.executor_runs_db import leak_counter, reset
        
        # Basic functionality test
        assert hasattr(leak_counter, 'value')
        assert callable(reset)