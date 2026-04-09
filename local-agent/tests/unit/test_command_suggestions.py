"""Tests for the command_suggestions module — typo detection and context-aware suggestions."""

import pytest

from agent.command_suggestions import (
    COMMANDS,
    CommandInfo,
    _COMMAND_LOOKUP,
    _edit_distance,
    find_closest_command,
    format_context_suggestions,
    format_suggestion,
    format_typo_suggestion,
    suggest_commands_for_context,
)


class TestCommandRegistry:
    """Test the command registry is well-formed."""

    def test_registry_not_empty(self):
        assert len(COMMANDS) > 10

    def test_all_commands_have_name(self):
        for cmd in COMMANDS:
            assert cmd.name, f"Command missing name: {cmd}"

    def test_all_commands_have_description(self):
        for cmd in COMMANDS:
            assert cmd.description, f"{cmd.name} missing description"

    def test_all_commands_have_category(self):
        for cmd in COMMANDS:
            assert cmd.category, f"{cmd.name} missing category"

    def test_all_commands_have_keywords(self):
        for cmd in COMMANDS:
            assert cmd.keywords, f"{cmd.name} missing keywords"

    def test_lookup_contains_all_names(self):
        for cmd in COMMANDS:
            assert cmd.name.lower() in _COMMAND_LOOKUP
            for alias in cmd.aliases:
                assert alias.lower() in _COMMAND_LOOKUP

    def test_known_commands_present(self):
        names = {c.name for c in COMMANDS}
        assert "betterDev" in names
        assert "techNews" in names
        assert "perf" in names
        assert "metrics" in names
        assert "think" in names
        assert "commands" in names


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_char_diff(self):
        assert _edit_distance("perf", "perv") == 1

    def test_insertion(self):
        assert _edit_distance("metric", "metrics") == 1

    def test_deletion(self):
        assert _edit_distance("metrics", "metric") == 1

    def test_empty(self):
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3

    def test_completely_different(self):
        assert _edit_distance("abc", "xyz") == 3


class TestFindClosestCommand:
    def test_exact_match_returns_none(self):
        # Exact matches are handled by the dispatcher, not this function
        # But if called, it would match with distance 0
        result = find_closest_command("perf")
        assert result is not None
        assert result.name == "perf"

    def test_typo_one_char(self):
        result = find_closest_command("perfr")
        assert result is not None
        assert result.name == "perf"

    def test_typo_betterdev(self):
        result = find_closest_command("beterdev")
        assert result is not None
        assert result.name == "betterDev"

    def test_typo_metrics(self):
        result = find_closest_command("metrcs")
        assert result is not None
        assert result.name == "metrics"

    def test_typo_technews(self):
        result = find_closest_command("technws")
        assert result is not None
        assert result.name == "techNews"

    def test_no_match_for_gibberish(self):
        result = find_closest_command("xyzabcdefghij")
        assert result is None

    def test_no_match_for_short_input(self):
        result = find_closest_command("ab")
        assert result is None

    def test_no_match_for_empty(self):
        result = find_closest_command("")
        assert result is None

    def test_respects_max_distance(self):
        result = find_closest_command("perfffff", max_distance=1)
        assert result is None


class TestSuggestCommandsForContext:
    def test_python_context(self):
        context = "We were discussing python decorators and async patterns"
        suggestions = suggest_commands_for_context(context)
        assert len(suggestions) >= 1
        names = {s.name for s in suggestions}
        assert "betterDev" in names

    def test_performance_context(self):
        context = "The API calls seem slow, latency is high"
        suggestions = suggest_commands_for_context(context)
        names = {s.name for s in suggestions}
        assert "perf" in names or "metrics" in names

    def test_news_context(self):
        context = "What are the latest tech industry trends"
        suggestions = suggest_commands_for_context(context)
        names = {s.name for s in suggestions}
        assert "techNews" in names

    def test_youtube_context(self):
        context = "I want to download some youtube videos from a channel"
        suggestions = suggest_commands_for_context(context)
        names = {s.name for s in suggestions}
        # Should suggest at least one youtube command
        youtube_cmds = {"listVideos", "searchVideos", "downloadVideo", "downloadChannel"}
        assert names & youtube_cmds

    def test_empty_context(self):
        assert suggest_commands_for_context("") == []

    def test_limit_respected(self):
        context = "python oracle system design learning news tech article"
        suggestions = suggest_commands_for_context(context, limit=2)
        assert len(suggestions) <= 2

    def test_no_duplicates(self):
        context = "python learning article study education tutorial"
        suggestions = suggest_commands_for_context(context)
        names = [s.name for s in suggestions]
        assert len(names) == len(set(names))

    def test_complaint_context(self):
        context = "This is really frustrating and broken and annoying"
        suggestions = suggest_commands_for_context(context)
        names = {s.name for s in suggestions}
        assert "karen" in names


class TestFormatting:
    def test_format_suggestion(self):
        cmd = CommandInfo("perf", (), "Show profiling data", "performance", ("perf",), "perf")
        result = format_suggestion(cmd)
        assert "`perf`" in result
        assert "profiling" in result

    def test_format_typo_suggestion(self):
        cmd = CommandInfo("metrics", (), "Show trends", "performance", ("metrics",), "metrics")
        result = format_typo_suggestion("metrcs", cmd)
        assert "Did you mean" in result
        assert "`metrics`" in result

    def test_format_context_suggestions(self):
        commands = [
            CommandInfo("perf", (), "Show profiling", "performance", (), "perf"),
            CommandInfo("metrics", (), "Show trends", "performance", (), "metrics"),
        ]
        result = format_context_suggestions(commands)
        assert "Suggested commands" in result
        assert "`perf`" in result
        assert "`metrics`" in result

    def test_format_context_suggestions_empty(self):
        assert format_context_suggestions([]) == ""
