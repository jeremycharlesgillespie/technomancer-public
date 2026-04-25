"""Tests for agent.leak_counter module."""

import threading
from unittest.mock import patch

import pytest

from agent.leak_counter import reset, get_count, increment


class TestLeakCounter:
    def test_reset_clears_counter(self):
        """Test that reset clears the counter to zero."""
        # Set a counter value
        increment()
        increment()
        assert get_count() == 2
        
        # Reset should clear it
        reset()
        assert get_count() == 0

    def test_reset_is_thread_safe(self):
        """Test that reset works correctly in a threaded environment."""
        # Set counter in main thread
        increment()
        increment()
        assert get_count() == 2
        
        # Reset in main thread
        reset()
        assert get_count() == 0
        
        # Create a new thread and check it's independent
        def check_thread_counter():
            assert get_count() == 0  # Should be zero in new thread
            increment()
            assert get_count() == 1
            
        thread = threading.Thread(target=check_thread_counter)
        thread.start()
        thread.join()
        
        # Main thread should still be at 0
        assert get_count() == 0

    def test_increment_increments_counter(self):
        """Test that increment correctly increments the counter."""
        reset()  # Start with clean state
        assert get_count() == 0
        
        increment()
        assert get_count() == 1
        
        increment()
        assert get_count() == 2

    def test_get_count_returns_zero_when_no_counter_set(self):
        """Test that get_count returns 0 when no counter has been set."""
        # This tests the case where reset() was called or no increment was done
        reset()
        assert get_count() == 0

    def test_importable_and_callable(self):
        """Test that the module can be imported and functions are callable."""
        # This tests that the module can be imported and functions are accessible
        from agent import leak_counter_reset, get_count, increment
        
        # Test that functions are callable
        assert callable(leak_counter_reset)
        assert callable(get_count)
        assert callable(increment)
        
        # Test basic functionality
        leak_counter_reset()
        assert get_count() == 0
        increment()
        assert get_count() == 1