"""Tests for agent.leak_counter — leak counter reset mechanism."""

import threading

import pytest

from agent.leak_counter import leak_counter, reset


class TestReset:
    """Test leak_counter.reset() function."""

    def test_reset_clears_counter(self):
        """reset() should clear all thread-local state."""
        # Set some attributes
        leak_counter.count = 5
        leak_counter.active = True

        # Reset should clear them
        reset()

        # All attributes should be gone
        assert not hasattr(leak_counter, "count")
        assert not hasattr(leak_counter, "active")

    def test_reset_multiple_times(self):
        """reset() should be idempotent — can be called multiple times."""
        leak_counter.value = 42

        reset()
        assert not hasattr(leak_counter, "value")

        reset()  # Should not raise
        assert not hasattr(leak_counter, "value")

    def test_reset_with_no_attributes(self):
        """reset() should not raise when called with no attributes."""
        # Should not raise even if no attributes are set
        reset()

    def test_reset_is_callable(self):
        """reset() should be importable and callable."""
        assert callable(reset)

    def test_leak_counter_is_thread_local(self):
        """leak_counter should be a threading.local() object."""
        assert isinstance(leak_counter, threading.local)

    def test_reset_does_not_affect_other_threads(self):
        """reset() should only affect the current thread's state."""
        # Set attribute in current thread
        leak_counter.test_value = "current_thread"

        # Reset current thread
        reset()

        # Should be cleared in current thread
        assert not hasattr(leak_counter, "test_value")

        # In a new thread, the attribute should not exist (thread-local isolation)
        def check_other_thread():
            assert not hasattr(leak_counter, "test_value")

        other_thread = threading.Thread(target=check_other_thread)
        other_thread.start()
        other_thread.join()