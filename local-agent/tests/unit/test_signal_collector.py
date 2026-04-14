"""Tests for idea_generator signal collector — collect_signals and its helpers."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.idea_generator import (
    collect_signals,
    _collect_conversation_topics,
    _collect_coverage_gaps,
    _collect_pending_ideas,
    _collect_recent_changes,
    _collect_recent_errors,
    _collect_slow_operations,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def crash_log(tmp_path, monkeypatch):
    """Create a temp crash log and patch the module-level path."""
    log_file = tmp_path / "crash_log.md"
    monkeypatch.setattr("agent.idea_generator.CRASH_LOG", log_file)
    return log_file


@pytest.fixture
def coverage_file(tmp_path, monkeypatch):
    """Create a temp coverage.json and patch the module-level path."""
    cov_file = tmp_path / "coverage.json"
    monkeypatch.setattr("agent.idea_generator.COVERAGE_FILE", cov_file)
    return cov_file


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    """Patch REPO_ROOT to a temp directory."""
    monkeypatch.setattr("agent.idea_generator.REPO_ROOT", tmp_path)
    return tmp_path


# =============================================================================
# _collect_recent_errors
# =============================================================================


class TestCollectRecentErrors:
    def test_no_crash_log(self, crash_log):
        """Returns message when crash log doesn't exist."""
        # crash_log fixture patches the path but doesn't create the file
        result = _collect_recent_errors()
        assert "No crash log" in result

    def test_empty_crash_log(self, crash_log):
        """Returns no-errors message for empty log."""
        crash_log.write_text("", encoding="utf-8")
        result = _collect_recent_errors()
        assert "No errors" in result

    def test_old_entries_excluded(self, crash_log):
        """Entries older than 1 hour are not returned."""
        old_ts = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        crash_log.write_text(
            f"## {old_ts}\nSome old error\nstack trace here\n",
            encoding="utf-8",
        )
        result = _collect_recent_errors()
        assert "No errors in the last hour" in result

    def test_recent_entries_included(self, crash_log):
        """Entries within the last hour are returned."""
        recent_ts = (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        crash_log.write_text(
            f"## {recent_ts}\nUnboundLocalError: something broke\nTraceback line\n",
            encoding="utf-8",
        )
        result = _collect_recent_errors()
        assert "UnboundLocalError" in result

    def test_custom_since_parameter(self, crash_log):
        """Respects a custom 'since' cutoff."""
        ts = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        crash_log.write_text(
            f"## {ts}\nTimeoutError: connection timed out\n",
            encoding="utf-8",
        )
        # Default (1 hour) should miss it
        result_default = _collect_recent_errors()
        assert "No errors" in result_default

        # Custom cutoff 4 hours ago should find it
        result_custom = _collect_recent_errors(since=datetime.now() - timedelta(hours=4))
        assert "TimeoutError" in result_custom

    def test_multiple_entries_filtered(self, crash_log):
        """Mix of old and recent entries — only recent ones returned."""
        old_ts = (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        new_ts = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        crash_log.write_text(
            f"## {old_ts}\nOld error\n\n## {new_ts}\nNew error: KeyError\n",
            encoding="utf-8",
        )
        result = _collect_recent_errors()
        assert "Old error" not in result
        assert "KeyError" in result

    def test_entry_truncated_to_500_chars(self, crash_log):
        """Long entries are truncated to 500 characters."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        long_trace = "x" * 1000
        crash_log.write_text(f"## {ts}\n{long_trace}\n", encoding="utf-8")
        result = _collect_recent_errors()
        # The entry itself should be capped at 500 chars
        assert len(result) < 600


# =============================================================================
# _collect_slow_operations
# =============================================================================


class TestCollectSlowOperations:
    def test_no_data(self):
        """Returns message when monitor has no records."""
        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.return_value = {"calls": 0}
        with patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor):
            result = _collect_slow_operations()
        assert "No performance data" in result

    def test_no_slow_endpoints(self):
        """Returns 'no slow' when all p95 < 5s."""
        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.side_effect = [
            {"calls": 100},  # First call: all stats
            {"calls": 50, "p95_latency": 2.1, "avg_latency": 1.0, "failures": 0},
        ]
        mock_monitor._lock = MagicMock()
        mock_monitor._lock.__enter__ = MagicMock(return_value=None)
        mock_monitor._lock.__exit__ = MagicMock(return_value=False)
        rec = MagicMock()
        rec.endpoint = "ollama"
        mock_monitor._records = [rec]

        with patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor):
            result = _collect_slow_operations()
        assert "No slow operations" in result

    def test_slow_endpoint_reported(self):
        """Endpoints with p95 > 5s are listed."""
        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.side_effect = [
            {"calls": 100},  # First call: all stats
            {"calls": 80, "p95_latency": 8.5, "avg_latency": 4.2, "failures": 3},
        ]
        mock_monitor._lock = MagicMock()
        mock_monitor._lock.__enter__ = MagicMock(return_value=None)
        mock_monitor._lock.__exit__ = MagicMock(return_value=False)
        rec = MagicMock()
        rec.endpoint = "claude_api"
        mock_monitor._records = [rec]

        with patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor):
            result = _collect_slow_operations()
        assert "claude_api" in result
        assert "p95=8.5s" in result
        assert "3 failures" in result


# =============================================================================
# _collect_conversation_topics
# =============================================================================


class TestCollectConversationTopics:
    def test_memory_not_available(self):
        """Returns fallback when memory system isn't initialised."""
        with patch("agent.idea_generator.get_memory_system", side_effect=ValueError("not init")):
            # Import is inside the function, mock at module level
            result = _collect_conversation_topics()
        assert "not available" in result.lower() or "No conversations" in result

    def test_no_recent_conversations(self):
        """Returns 'no conversations' when deque is empty."""
        mock_mem = MagicMock()
        mock_mem.recent_conversations = []
        with patch("agent.idea_generator.get_memory_system", return_value=mock_mem):
            result = _collect_conversation_topics()
        assert "No conversations" in result

    def test_recent_conversations_listed(self):
        """Recent entries appear with user and truncated message."""
        now = datetime.now()
        entry1 = MagicMock()
        entry1.timestamp = now - timedelta(minutes=10)
        entry1.user = "Jeremy"
        entry1.message = "How do I fix the rate limiter?"

        entry2 = MagicMock()
        entry2.timestamp = now - timedelta(minutes=5)
        entry2.user = "Jeremy"
        entry2.message = "Show me the perf stats"

        mock_mem = MagicMock()
        mock_mem.recent_conversations = [entry1, entry2]
        with patch("agent.idea_generator.get_memory_system", return_value=mock_mem):
            result = _collect_conversation_topics()

        assert "rate limiter" in result
        assert "perf stats" in result
        assert "Jeremy" in result

    def test_old_conversations_excluded(self):
        """Conversations older than the window are filtered out."""
        old = MagicMock()
        old.timestamp = datetime.now() - timedelta(hours=3)
        old.user = "someone"
        old.message = "old question"

        mock_mem = MagicMock()
        mock_mem.recent_conversations = [old]
        with patch("agent.idea_generator.get_memory_system", return_value=mock_mem):
            result = _collect_conversation_topics()
        assert "No conversations" in result

    def test_caps_at_20_entries(self):
        """At most 20 entries are included."""
        now = datetime.now()
        entries = []
        for i in range(30):
            e = MagicMock()
            e.timestamp = now - timedelta(minutes=i)
            e.user = "user"
            e.message = f"question {i}"
            entries.append(e)

        mock_mem = MagicMock()
        mock_mem.recent_conversations = entries
        with patch("agent.idea_generator.get_memory_system", return_value=mock_mem):
            result = _collect_conversation_topics()
        # Should have exactly 20 bullet points
        assert result.count("- [") == 20


# =============================================================================
# _collect_coverage_gaps
# =============================================================================


class TestCollectCoverageGaps:
    def test_no_coverage_file(self, coverage_file):
        """Returns message when coverage.json doesn't exist."""
        result = _collect_coverage_gaps()
        assert "No coverage data" in result

    def test_all_above_threshold(self, coverage_file):
        """Returns positive message when everything is well-covered."""
        data = {
            "files": {
                "agent/core.py": {"summary": {"percent_covered": 90, "num_statements": 200}},
                "agent/tools.py": {"summary": {"percent_covered": 85, "num_statements": 150}},
            },
            "totals": {},
        }
        coverage_file.write_text(json.dumps(data), encoding="utf-8")
        result = _collect_coverage_gaps()
        assert "All modules above 50%" in result

    def test_low_coverage_reported(self, coverage_file):
        """Modules below 50% coverage are listed."""
        data = {
            "files": {
                "agent/well_covered.py": {"summary": {"percent_covered": 80, "num_statements": 100}},
                "agent/poorly_covered.py": {"summary": {"percent_covered": 25, "num_statements": 50}},
                "agent/also_bad.py": {"summary": {"percent_covered": 10, "num_statements": 30}},
            },
            "totals": {},
        }
        coverage_file.write_text(json.dumps(data), encoding="utf-8")
        result = _collect_coverage_gaps()
        assert "poorly_covered.py" in result
        assert "also_bad.py" in result
        assert "well_covered.py" not in result

    def test_tiny_files_ignored(self, coverage_file):
        """Files with <= 10 statements are skipped even if low coverage."""
        data = {
            "files": {
                "agent/__init__.py": {"summary": {"percent_covered": 0, "num_statements": 3}},
            },
            "totals": {},
        }
        coverage_file.write_text(json.dumps(data), encoding="utf-8")
        result = _collect_coverage_gaps()
        assert "All modules above 50%" in result

    def test_invalid_json(self, coverage_file):
        """Handles corrupted coverage file gracefully."""
        coverage_file.write_text("not json!", encoding="utf-8")
        result = _collect_coverage_gaps()
        assert "Could not parse" in result

    def test_backslash_normalised(self, coverage_file):
        """Windows backslashes in paths are normalised to forward slashes."""
        data = {
            "files": {
                "agent\\bad_module.py": {"summary": {"percent_covered": 20, "num_statements": 50}},
            },
            "totals": {},
        }
        coverage_file.write_text(json.dumps(data), encoding="utf-8")
        result = _collect_coverage_gaps()
        assert "agent/bad_module.py" in result


# =============================================================================
# _collect_recent_changes
# =============================================================================


class TestCollectRecentChanges:
    def test_no_commits(self, repo_root):
        """Returns message when git log has no output."""
        with patch("agent.idea_generator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            result = _collect_recent_changes()
        assert "No commits" in result

    def test_commits_and_files_listed(self, repo_root):
        """Parses git log output into commits and files."""
        git_output = "abc1234 Fix rate limiter\nagent/discord_rate_limit.py\n\ndef5678 Add test\ntests/unit/test_rate_limit.py\n"
        with patch("agent.idea_generator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=git_output, returncode=0)
            result = _collect_recent_changes()
        assert "Fix rate limiter" in result
        assert "discord_rate_limit.py" in result
        assert "test_rate_limit.py" in result

    def test_deduplicates_files(self, repo_root):
        """Same file in multiple commits appears only once."""
        git_output = "abc1234 First change\nagent/core.py\n\ndef5678 Second change\nagent/core.py\n"
        with patch("agent.idea_generator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=git_output, returncode=0)
            result = _collect_recent_changes()
        assert result.count("agent/core.py") == 1

    def test_subprocess_error(self, repo_root):
        """Handles git command failure gracefully."""
        with patch("agent.idea_generator.subprocess.run", side_effect=OSError("git not found")):
            result = _collect_recent_changes()
        assert "Could not read" in result


# =============================================================================
# _collect_pending_ideas
# =============================================================================


class TestCollectPendingIdeas:
    def test_no_ideas(self):
        """Returns message when idea board is empty."""
        with patch("agent.idea_generator.load_ideas", return_value=[]):
            result = _collect_pending_ideas()
        assert "empty" in result.lower()

    def test_no_pending(self):
        """Returns message when no ideas are approved/executing."""
        idea = MagicMock()
        idea.state = "proposed"
        idea.id = "idea-001"
        idea.title = "Some proposed idea"
        with patch("agent.idea_generator.load_ideas", return_value=[idea]):
            result = _collect_pending_ideas()
        assert "No approved/executing" in result

    def test_pending_listed(self):
        """Approved and executing ideas are listed."""
        approved = MagicMock()
        approved.state = "approved"
        approved.id = "idea-042"
        approved.title = "Add caching layer"

        executing = MagicMock()
        executing.state = "executing"
        executing.id = "idea-043"
        executing.title = "Fix memory leak"

        done = MagicMock()
        done.state = "done"
        done.id = "idea-044"
        done.title = "Already finished"

        with patch("agent.idea_generator.load_ideas", return_value=[approved, executing, done]):
            result = _collect_pending_ideas()
        assert "idea-042" in result
        assert "idea-043" in result
        assert "idea-044" not in result

    def test_load_failure(self):
        """Handles import/load errors gracefully."""
        with patch("agent.idea_generator.load_ideas", side_effect=ImportError("no module")):
            result = _collect_pending_ideas()
        assert "Could not load" in result


# =============================================================================
# collect_signals (integration of all collectors)
# =============================================================================


class TestCollectSignals:
    def test_returns_all_sections(self, crash_log, coverage_file, repo_root):
        """The combined output has all 6 signal sections."""
        crash_log.write_text("", encoding="utf-8")
        coverage_file.write_text(json.dumps({"files": {}, "totals": {}}), encoding="utf-8")

        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.return_value = {"calls": 0}

        with (
            patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor),
            patch("agent.idea_generator.get_memory_system", side_effect=ValueError),
            patch("agent.idea_generator.subprocess.run", return_value=MagicMock(stdout="")),
            patch("agent.idea_generator.load_ideas", return_value=[]),
        ):
            result = collect_signals()

        assert "### RECENT ERRORS" in result
        assert "### SLOW OPERATIONS" in result
        assert "### CONVERSATION TOPICS" in result
        assert "### TEST COVERAGE GAPS" in result
        assert "### RECENTLY CHANGED FILES" in result
        assert "### PENDING IDEAS" in result

    def test_returns_string(self, crash_log, coverage_file, repo_root):
        """collect_signals always returns a string."""
        crash_log.write_text("", encoding="utf-8")
        coverage_file.write_text(json.dumps({"files": {}, "totals": {}}), encoding="utf-8")

        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.return_value = {"calls": 0}

        with (
            patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor),
            patch("agent.idea_generator.get_memory_system", side_effect=ValueError),
            patch("agent.idea_generator.subprocess.run", return_value=MagicMock(stdout="")),
            patch("agent.idea_generator.load_ideas", return_value=[]),
        ):
            result = collect_signals()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_since_propagated(self, crash_log, coverage_file, repo_root):
        """Custom 'since' parameter is passed to time-windowed collectors."""
        recent_ts = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        crash_log.write_text(
            f"## {recent_ts}\nOld-ish error\n",
            encoding="utf-8",
        )
        coverage_file.write_text(json.dumps({"files": {}, "totals": {}}), encoding="utf-8")

        mock_monitor = MagicMock()
        mock_monitor.get_endpoint_stats.return_value = {"calls": 0}

        with (
            patch("agent.idea_generator.get_perf_monitor", return_value=mock_monitor),
            patch("agent.idea_generator.get_memory_system", side_effect=ValueError),
            patch("agent.idea_generator.subprocess.run", return_value=MagicMock(stdout="")),
            patch("agent.idea_generator.load_ideas", return_value=[]),
        ):
            # Default 1-hour window should miss 3-hour-old entry
            result_default = collect_signals()
            assert "Old-ish error" not in result_default

            # 4-hour window should include it
            result_wide = collect_signals(since=datetime.now() - timedelta(hours=4))
            assert "Old-ish error" in result_wide
