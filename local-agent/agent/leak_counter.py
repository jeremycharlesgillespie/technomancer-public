"""
Leak Counter — Thread-local counter for detecting connection leaks.

This module provides a thread-local counter that can be used to track
resource usage or connection leaks across different threads. The counter
is initialized to 0 and can be reset to 0 using the reset() function.

Usage:
    from agent.leak_counter import leak_counter, reset
    
    # Increment the counter
    leak_counter.value += 1
    
    # Reset the counter to 0
    reset()
"""

import threading
from typing import Any

# Leak counter for detecting connection leaks
leak_counter = threading.local()

# Initialize the counter to 0
leak_counter.value = 0


def reset() -> None:
    """Reset the leak counter to zero.
    
    This function is designed to be called by test fixtures to ensure
    a clean state between tests.
    """
    leak_counter.value = 0