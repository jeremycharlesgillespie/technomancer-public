"""
Leak Counter — Thread-local counter for detecting connection leaks.

This module provides a thread-local leak counter that can be used to
track resource usage and detect potential leaks. The counter is
resettable for testing purposes.
"""

import threading
from typing import Any

# Leak counter for detecting connection leaks
leak_counter = threading.local()


def reset() -> None:
    """Reset the leak counter to zero.
    
    This function clears the counter for the current thread, allowing
    test fixtures to reliably reset the counter to zero.
    """
    leak_counter.__dict__.clear()


def increment() -> None:
    """Increment the leak counter for the current thread."""
    if not hasattr(leak_counter, 'count'):
        leak_counter.count = 0
    leak_counter.count += 1


def get_count() -> int:
    """Get the current leak counter value for the current thread."""
    return getattr(leak_counter, 'count', 0)