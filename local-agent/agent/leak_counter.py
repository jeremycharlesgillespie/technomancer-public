"""
Leak Counter — Reset mechanism for connection leak detection.

Provides a clean reset mechanism for test fixtures to reset the leak counter
to zero. The leak counter is a threading.local() object used to track
connection leaks across threads.

File: agent/leak_counter.py
"""

from __future__ import annotations

import threading

# Leak counter for detecting connection leaks
leak_counter = threading.local()


def reset() -> None:
    """Reset the leak counter to zero.

    Clears all thread-local state for the leak counter. This is useful for
    test fixtures that need to ensure a clean state before each test run.

    Example:
        >>> from agent.leak_counter import leak_counter, reset
        >>> # ... some operations ...
        >>> reset()  # Clear the counter
    """
    # Clear all attributes on the thread-local object
    leak_counter.__dict__.clear()