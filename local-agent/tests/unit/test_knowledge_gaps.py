"""
Tests for agent/knowledge_gaps.py - Knowledge gap tracking functionality.
"""

from agent.knowledge_gaps import (
    detect_knowledge_gap,
    get_gap_summary,
    get_open_gaps,
    log_knowledge_gap,
    resolve_gap,
)


class TestDetectKnowledgeGap:
    """Tests for detect_knowledge_gap detection logic."""

    def test_detects_uncertainty_phrase(self):
        """Detects 'I'm not sure' as uncertainty."""
        gap = detect_knowledge_gap(
            "What is the capital of Narnia?",
            "I'm not sure about that, but I think it might be Cair Paravel.",
        )
        assert gap is not None
        assert gap["gap_type"] == "uncertainty"

    def test_detects_failure_phrase(self):
        """Detects 'I cannot answer' as failure."""
        gap = detect_knowledge_gap(
            "What is the stock price of XYZ?",
            "I cannot answer that question with my current knowledge.",
        )
        assert gap is not None
        assert gap["gap_type"] == "failure"

    def test_failure_takes_priority(self):
        """Failure phrases are detected before uncertainty."""
        gap = detect_knowledge_gap(
            "Complex question?",
            "I'm not sure and I cannot answer this properly.",
        )
        assert gap is not None
        assert gap["gap_type"] == "failure"

    def test_no_gap_for_confident_response(self):
        """Returns None for confident responses."""
        gap = detect_knowledge_gap(
            "What is 2 + 2?",
            "That's 4! Simple arithmetic.",
        )
        assert gap is None

    def test_no_gap_for_empty_inputs(self):
        """Returns None for empty query or response."""
        assert detect_knowledge_gap("", "some response") is None
        assert detect_knowledge_gap("some query", "") is None

    def test_tracks_escalation(self):
        """Records whether response was escalated to Claude."""
        gap = detect_knowledge_gap(
            "Hard question?",
            "I'm not sure about this one.",
            was_escalated=True,
        )
        assert gap is not None
        assert gap["was_escalated"] is True

    def test_no_escalation_by_default(self):
        """Escalation defaults to False."""
        gap = detect_knowledge_gap(
            "Question?",
            "I don't know the answer to that.",
        )
        assert gap is not None
        assert gap["was_escalated"] is False

    def test_truncates_long_query(self):
        """Long queries are truncated to 500 chars."""
        long_query = "x" * 1000
        gap = detect_knowledge_gap(long_query, "I'm not sure.")
        assert gap is not None
        assert len(gap["query"]) == 500

    def test_truncates_long_response(self):
        """Long responses are snipped to 300 chars."""
        long_response = "I'm not sure " + "x" * 500
        gap = detect_knowledge_gap("Question?", long_response)
        assert gap is not None
        assert len(gap["response_snippet"]) <= 304  # 300 + "..."

    def test_case_insensitive_detection(self):
        """Detection is case-insensitive."""
        gap = detect_knowledge_gap(
            "Question?",
            "I'M NOT SURE about that.",
        )
        assert gap is not None

    def test_multiple_uncertainty_phrases(self):
        """Still returns a single gap even with multiple phrases."""
        gap = detect_knowledge_gap(
            "Question?",
            "I'm not sure and I can't verify this, you might want to check.",
        )
        assert gap is not None
        assert gap["gap_type"] == "uncertainty"

    def test_contains_timestamp(self):
        """Gap entries include a timestamp."""
        gap = detect_knowledge_gap("Q?", "I don't know.")
        assert gap is not None
        assert "timestamp" in gap
        assert len(gap["timestamp"]) > 0


class TestLogKnowledgeGap:
    """Tests for log_knowledge_gap file writing."""

    def test_creates_file_on_first_gap(self, patched_knowledge_gaps):
        """First gap creates the knowledge_gaps.md file."""
        gap = detect_knowledge_gap("Test query?", "I'm not sure.")
        log_knowledge_gap(gap)

        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        assert gaps_file.exists()
        content = gaps_file.read_text()
        assert "Test query?" in content
        assert "UNCERTAINTY" in content

    def test_appends_to_existing_file(self, patched_knowledge_gaps):
        """Additional gaps are appended."""
        gap1 = detect_knowledge_gap("First question?", "I'm not sure.")
        gap2 = detect_knowledge_gap("Second question?", "I don't know.")
        log_knowledge_gap(gap1)
        log_knowledge_gap(gap2)

        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        content = gaps_file.read_text()
        assert "First question?" in content
        assert "Second question?" in content

    def test_file_has_sections(self, patched_knowledge_gaps):
        """Created file has Open Gaps and Resolved sections."""
        gap = detect_knowledge_gap("Q?", "I'm not sure.")
        log_knowledge_gap(gap)

        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        content = gaps_file.read_text()
        assert "## Open Gaps" in content
        assert "## Resolved" in content

    def test_returns_confirmation(self, patched_knowledge_gaps):
        """Returns a confirmation string."""
        gap = detect_knowledge_gap("My query?", "I don't know.")
        result = log_knowledge_gap(gap)
        assert "Logged" in result
        assert "My query?" in result

    def test_logs_escalated_gap(self, patched_knowledge_gaps):
        """Escalated gaps are marked in the file."""
        gap = detect_knowledge_gap("Hard q?", "I'm not sure.", was_escalated=True)
        log_knowledge_gap(gap)

        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        content = gaps_file.read_text()
        assert "[escalated]" in content


class TestGetOpenGaps:
    """Tests for get_open_gaps reading."""

    def test_no_file_returns_message(self, patched_knowledge_gaps):
        """Returns message when no gaps file exists."""
        result = get_open_gaps()
        assert "No knowledge gaps" in result

    def test_reads_open_gaps(self, patched_knowledge_gaps):
        """Reads and returns open gaps."""
        gap = detect_knowledge_gap("Test query?", "I'm not sure.")
        log_knowledge_gap(gap)

        result = get_open_gaps()
        assert "Test query?" in result
        assert "(1)" in result

    def test_counts_multiple_gaps(self, patched_knowledge_gaps):
        """Correctly counts multiple open gaps."""
        for i in range(3):
            gap = detect_knowledge_gap(f"Question {i}?", "I don't know.")
            log_knowledge_gap(gap)

        result = get_open_gaps()
        assert "(3)" in result


class TestResolveGap:
    """Tests for resolve_gap moving entries."""

    def test_resolves_by_fragment(self, patched_knowledge_gaps):
        """Can resolve a gap by matching query fragment."""
        gap = detect_knowledge_gap("What is the speed of light?", "I'm not sure.")
        log_knowledge_gap(gap)

        result = resolve_gap("speed of light")
        assert "Resolved" in result

        # Check it moved to Resolved section
        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        content = gaps_file.read_text()
        resolved_section = content.split("## Resolved")[1]
        assert "speed of light" in resolved_section

    def test_resolve_with_note(self, patched_knowledge_gaps):
        """Resolution includes a note."""
        gap = detect_knowledge_gap("Question about X?", "I don't know.")
        log_knowledge_gap(gap)

        resolve_gap("Question about X", resolution="Added to memories")

        gaps_file = patched_knowledge_gaps / "Permanent" / "knowledge_gaps.md"
        content = gaps_file.read_text()
        assert "Added to memories" in content

    def test_resolve_nonexistent_returns_error(self, patched_knowledge_gaps):
        """Resolving a non-existent gap returns error message."""
        gap = detect_knowledge_gap("Real question?", "I'm not sure.")
        log_knowledge_gap(gap)

        result = resolve_gap("nonexistent query xyz")
        assert "No open gap" in result

    def test_resolve_no_file_returns_error(self, patched_knowledge_gaps):
        """Resolving when no file exists returns error."""
        result = resolve_gap("anything")
        assert "No knowledge gaps file" in result


class TestGetGapSummary:
    """Tests for get_gap_summary statistics."""

    def test_no_file_returns_message(self, patched_knowledge_gaps):
        """Returns message when no file exists."""
        result = get_gap_summary()
        assert "No knowledge gaps" in result

    def test_summary_counts_by_type(self, patched_knowledge_gaps):
        """Summary breaks down by uncertainty vs failure."""
        gap1 = detect_knowledge_gap("Q1?", "I'm not sure about this.")
        gap2 = detect_knowledge_gap("Q2?", "I cannot answer that question.")
        gap3 = detect_knowledge_gap("Q3?", "I don't know the answer.")
        log_knowledge_gap(gap1)
        log_knowledge_gap(gap2)
        log_knowledge_gap(gap3)

        result = get_gap_summary()
        assert "2 uncertainty" in result
        assert "1 failure" in result

    def test_summary_counts_escalated(self, patched_knowledge_gaps):
        """Summary counts escalated gaps."""
        gap = detect_knowledge_gap("Q?", "I'm not sure.", was_escalated=True)
        log_knowledge_gap(gap)

        result = get_gap_summary()
        assert "Escalated to Claude: 1" in result

    def test_summary_counts_resolved(self, patched_knowledge_gaps):
        """Summary counts resolved gaps."""
        gap = detect_knowledge_gap("Q?", "I don't know.")
        log_knowledge_gap(gap)
        resolve_gap("Q?")

        result = get_gap_summary()
        assert "Resolved: 1" in result
