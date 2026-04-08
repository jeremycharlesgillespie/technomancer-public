"""
Tests for agent/enhancements.py - Enhancement queue functionality.
"""

from agent.enhancements import (
    _get_next_number,
    add_enhancement,
    detect_enhancement_idea,
    get_pending_enhancements,
    save_context_for_claude,
)


class TestAddEnhancement:
    """Tests for add_enhancement function."""

    def test_add_enhancement_creates_file(self, patched_enhancements):
        """First enhancement creates enhancements.md."""
        add_enhancement("Add dark mode support")

        enhancements_file = patched_enhancements / "Permanent" / "enhancements.md"
        assert enhancements_file.exists()
        assert "Add dark mode support" in enhancements_file.read_text()

    def test_add_enhancement_auto_numbers(self, patched_enhancements):
        """Enhancements get sequential numbers."""
        result1 = add_enhancement("First feature")
        result2 = add_enhancement("Second feature")

        assert "#1" in result1
        assert "#2" in result2

        content = (patched_enhancements / "Permanent" / "enhancements.md").read_text()
        assert "#1:" in content
        assert "#2:" in content

    def test_add_enhancement_adds_to_pending(self, patched_enhancements):
        """Enhancement is added under Pending section."""
        add_enhancement("New feature")

        content = (patched_enhancements / "Permanent" / "enhancements.md").read_text()
        # Check it's in Pending section
        pending_section = content.split("## Pending")[1].split("## In Progress")[0]
        assert "New feature" in pending_section


class TestGetPendingEnhancements:
    """Tests for get_pending_enhancements function."""

    def test_get_pending_enhancements_empty(self, patched_enhancements):
        """Returns message when no enhancements file."""
        result = get_pending_enhancements()

        assert "No enhancements file" in result or "No pending" in result

    def test_get_pending_enhancements_reads_file(self, patched_enhancements):
        """get_pending_enhancements reads from file."""
        # Add some enhancements
        add_enhancement("Feature A")
        add_enhancement("Feature B")

        result = get_pending_enhancements()

        assert "Feature A" in result
        assert "Feature B" in result
        assert "2" in result  # Should show count

    def test_get_pending_enhancements_shows_count(self, patched_enhancements):
        """Returns count of pending items."""
        add_enhancement("One")
        add_enhancement("Two")
        add_enhancement("Three")

        result = get_pending_enhancements()

        assert "(3)" in result or "3" in result


class TestGetNextNumber:
    """Tests for _get_next_number helper."""

    def test_first_number_is_one(self, patched_enhancements):
        """First enhancement number is 1."""
        assert _get_next_number() == 1

    def test_increments_correctly(self, patched_enhancements):
        """Numbers increment correctly."""
        add_enhancement("First")
        assert _get_next_number() == 2

        add_enhancement("Second")
        assert _get_next_number() == 3


class TestDetectEnhancementIdea:
    """Tests for detect_enhancement_idea pattern matching."""

    def test_detect_cool_if_pattern(self):
        """Detects 'it would be cool if...' pattern."""
        result = detect_enhancement_idea("It would be cool if we had dark mode")

        assert result is not None
        assert "dark mode" in result.lower()

    def test_detect_should_add_pattern(self):
        """Detects 'we should add...' pattern."""
        result = detect_enhancement_idea("We should add better error handling")

        assert result is not None
        assert "error handling" in result.lower()

    def test_detect_can_you_add_pattern(self):
        """Detects 'can you add...' pattern."""
        result = detect_enhancement_idea("Can you add a logout button?")

        assert result is not None
        assert "logout" in result.lower()

    def test_detect_feature_request_pattern(self):
        """Detects 'feature request:' pattern."""
        result = detect_enhancement_idea("Feature request: add PDF export")

        assert result is not None
        assert "pdf" in result.lower()

    def test_detect_wish_pattern(self):
        """Detects 'I wish you could...' pattern."""
        result = detect_enhancement_idea("I wish you could remember my preferences")

        assert result is not None
        assert "remember" in result.lower()

    def test_detect_negative_no_pattern(self):
        """Returns None for non-enhancement text."""
        result = detect_enhancement_idea("Hello, how are you today?")
        assert result is None

        result = detect_enhancement_idea("Can you help me debug this code?")
        assert result is None

        result = detect_enhancement_idea("What is the weather like?")
        assert result is None


class TestSaveContextForClaude:
    """Tests for save_context_for_claude function."""

    def test_save_context_creates_handoff(self, patched_enhancements):
        """save_context_for_claude creates handoff file."""
        save_context_for_claude("Some error trace here", "#1")

        handoff_file = patched_enhancements / "Permanent" / "claude_handoff.md"
        assert handoff_file.exists()
        assert "Some error trace here" in handoff_file.read_text()

    def test_save_context_includes_reference(self, patched_enhancements):
        """Context includes enhancement reference."""
        save_context_for_claude("Debug info", "#3")

        content = (patched_enhancements / "Permanent" / "claude_handoff.md").read_text()
        assert "#3" in content

    def test_save_context_appends(self, patched_enhancements):
        """Multiple saves append to file."""
        save_context_for_claude("First context", "#1")
        save_context_for_claude("Second context", "#2")

        content = (patched_enhancements / "Permanent" / "claude_handoff.md").read_text()
        assert "First context" in content
        assert "Second context" in content

    def test_save_context_no_reference(self, patched_enhancements):
        """Works without enhancement reference."""
        result = save_context_for_claude("General context")

        assert "general" in result.lower()
        handoff_file = patched_enhancements / "Permanent" / "claude_handoff.md"
        assert "General context" in handoff_file.read_text()
