"""Tests for agent/leak_counter.py — leak counter functionality."""

import pytest
from unittest.mock import patch

from agent.leak_counter import leak_counter, reset


class TestLeakCounter:
    """Test leak counter functionality."""

    def test_counter_initializes_to_zero(self):
        """Test that leak_counter.value initializes to 0."""
        assert leak_counter.value == 0

    def test_counter_can_be_incremented(self):
        """Test that leak_counter.value can be incremented."""
        leak_counter.value += 1
        assert leak_counter.value == 1

    def test_reset_function_clears_counter(self):
        """Test that reset() function clears the counter to zero."""
        leak_counter.value += 5
        assert leak_counter.value == 5
        
        reset()
        assert leak_counter.value == 0

    def test_reset_function_is_callable(self):
        """Test that reset() function is callable."""
        assert callable(reset)

    def test_counter_is_thread_local(self):
        """Test that counter values are thread-local."""
        # Set counter in main thread
        leak_counter.value = 42
        
        # Create a mock thread to verify it's separate
        import threading
        import time
        
        def check_counter():
            # This should be 0 in the new thread
            assert leak_counter.value == 0
            
        thread = threading.Thread(target=check_counter)
        thread.start()
        thread.join()
        
        # Main thread should still have 42
        assert leak_counter.value == 42


def test_importability():
    """Test that leak_counter and reset can be imported from agent module."""
    from agent import leak_counter as agent_leak_counter
    from agent import reset as agent_reset
    
    assert agent_leak_counter.value == 0
    assert callable(agent_reset)