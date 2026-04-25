"""Simple test for leak_counter functionality to validate reset works."""

import sys
import os

# Add the project root to path to avoid import issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_leak_counter_reset():
    """Test that leak_counter.reset() function works correctly."""
    # Import directly from the module to avoid circular import issues
    from agent.leak_counter import leak_counter, reset
    
    # Test initial state
    assert leak_counter.value == 0
    
    # Test incrementing
    leak_counter.value += 5
    assert leak_counter.value == 5
    
    # Test reset
    reset()
    assert leak_counter.value == 0
    
    # Test that reset is callable
    assert callable(reset)
    
    print("All leak_counter tests passed!")

if __name__ == "__main__":
    test_leak_counter_reset()