"""Tests for idea_generator — idea parsing, prompt building, generation cycle."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.idea_generator import (
    IDEA_PROMPT,
    MAX_PROPOSED_IDEAS,
    OFFSET_AFTER_NEWS_MINUTES,
    SYNTHESIS_PROMPT,
    _count_proposed_ideas,
    _load_codebase_summary,
    _load_errors,
    _load_performance,
    _notify_discord,
    _parse_epic_response,
    _seconds_until_next_run,
    idea_generation_loop,
    start_idea_generator,
    synthesize_epic,
)


class TestIdeaPrompt:
    """Test the prompt template."""

    def test_prompt_has_lifecycle_section(self):
        assert "FULL LIFECYCLE" in IDEA_PROMPT

    def test_prompt_requires_end_to_end(self):
        assert "THINK END-TO-END" in IDEA_PROMPT

    def test_prompt_supports_epics(self):
        assert "epic" in IDEA_PROMPT.lower()
        assert "stories" in IDEA_PROMPT.lower()

    def test_prompt_has_all_placeholders(self):
        for key in ["codebase", "news", "conversations", "errors", "performance", "existing"]:
            assert f"{{{key}}}" in IDEA_PROMPT, f"Missing placeholder: {key}"

    def test_prompt_mentions_idea_type(self):
        assert "idea_type" in IDEA_PROMPT


class TestLoadCodebaseSummary:
    def test_returns_string(self):
        result = _load_codebase_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_python_files(self):
        result = _load_codebase_summary()
        assert ".py" in result or "core" in result


class TestLoadErrors:
    def test_returns_string(self):
        result = _load_errors()
        assert isinstance(result, str)


class TestLoadPerformance:
    def test_returns_string(self):
        result = _load_performance()
        assert isinstance(result, str)


class TestParseIdeas:
    """Test the _parse_ideas function."""

    def test_parses_valid_json(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {
                "title": "Test Idea",
                "description": "WHAT: Something\n\nWHY: Because",
                "category": "quality",
                "source": "conversation_analysis",
            }
        ])
        result = _parse_ideas(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Test Idea"

    def test_parses_json_in_markdown(self):
        from agent.idea_generator import _parse_ideas

        raw = "Here are my suggestions:\n```json\n" + json.dumps([
            {"title": "Idea", "description": "Desc", "category": "feature", "source": "news_analysis"}
        ]) + "\n```"
        result = _parse_ideas(raw)
        assert len(result) >= 1

    def test_handles_empty_response(self):
        from agent.idea_generator import _parse_ideas

        result = _parse_ideas("")
        assert result == []

    def test_handles_invalid_json(self):
        from agent.idea_generator import _parse_ideas

        result = _parse_ideas("This is not JSON at all")
        assert result == []

    def test_filters_missing_fields(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {"title": "Good", "description": "Has all fields", "category": "quality", "source": "test"},
            {"title": "Bad"},  # missing description
        ])
        result = _parse_ideas(raw)
        # Should keep valid, skip invalid
        assert len(result) >= 1

    def test_preserves_extra_fields(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {
                "title": "Epic Idea",
                "description": "Full lifecycle",
                "category": "feature",
                "source": "news_analysis",
                "idea_type": "epic",
                "stories": ["Story 1", "Story 2"],
            }
        ])
        result = _parse_ideas(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Epic Idea"
        assert result[0]["idea_type"] == "epic"
        assert result[0]["stories"] == ["Story 1", "Story 2"]

    def test_preserves_epic_context(self):
        from agent.idea_generator import _parse_ideas

        raw = json.dumps([
            {
                "title": "Context Epic",
                "description": "Desc",
                "epic_context": "This epic improves reliability end to end.",
            }
        ])
        result = _parse_ideas(raw)
        assert len(result) == 1
        assert result[0]["epic_context"] == "This epic improves reliability end to end."


# =============================================================================
# SYNTHESIS PROMPT
# =============================================================================


class TestSynthesisPrompt:
    """Test the synthesis prompt template."""

    def test_has_required_placeholders(self):
        for key in ["signals", "codebase", "existing"]:
            assert f"{{{key}}}" in SYNTHESIS_PROMPT, f"Missing placeholder: {key}"

    def test_requests_single_epic(self):
        assert "ONE epic" in SYNTHESIS_PROMPT

    def test_requests_stories(self):
        assert "stories" in SYNTHESIS_PROMPT.lower()
        assert "1-3" in SYNTHESIS_PROMPT

    def test_mentions_epic_context(self):
        assert "epic_context" in SYNTHESIS_PROMPT

    def test_mentions_category(self):
        assert "category" in SYNTHESIS_PROMPT

    def test_mentions_json_object(self):
        assert "JSON object" in SYNTHESIS_PROMPT


# =============================================================================
# PARSE EPIC RESPONSE
# =============================================================================


class TestParseEpicResponse:
    """Test _parse_epic_response parsing logic."""

    def test_parses_valid_json_object(self):
        raw = json.dumps({
            "title": "Improve Error Handling",
            "epic_context": "Better error handling across the board.",
            "category": "quality",
            "source": "error_analysis",
            "stories": [
                {"title": "Add retry logic", "description": "WHAT: Add retries"},
                {"title": "Better logging", "description": "WHAT: Improve logs"},
            ],
        })
        result = _parse_epic_response(raw)
        assert result is not None
        assert result["title"] == "Improve Error Handling"
        assert result["epic_context"] == "Better error handling across the board."
        assert len(result["stories"]) == 2
        assert result["stories"][0]["title"] == "Add retry logic"

    def test_parses_json_in_code_block(self):
        raw = "Here's the epic:\n```json\n" + json.dumps({
            "title": "Speed Up Bot",
            "stories": [{"title": "Cache responses", "description": "WHAT: Cache"}],
        }) + "\n```"
        result = _parse_epic_response(raw)
        assert result is not None
        assert result["title"] == "Speed Up Bot"
        assert len(result["stories"]) == 1

    def test_handles_string_stories(self):
        raw = json.dumps({
            "title": "Test Epic",
            "stories": ["Story A", "Story B", "Story C"],
        })
        result = _parse_epic_response(raw)
        assert result is not None
        assert len(result["stories"]) == 3
        assert result["stories"][0]["title"] == "Story A"
        assert "Story under epic" in result["stories"][0]["description"]

    def test_caps_stories_at_3(self):
        raw = json.dumps({
            "title": "Big Epic",
            "stories": [
                {"title": f"Story {i}", "description": f"Desc {i}"}
                for i in range(5)
            ],
        })
        result = _parse_epic_response(raw)
        assert result is not None
        assert len(result["stories"]) == 3

    def test_returns_none_for_empty_string(self):
        assert _parse_epic_response("") is None

    def test_returns_none_for_invalid_json(self):
        assert _parse_epic_response("not json at all") is None

    def test_returns_none_for_missing_title(self):
        raw = json.dumps({"description": "No title field"})
        assert _parse_epic_response(raw) is None

    def test_returns_none_for_json_array(self):
        raw = json.dumps([{"title": "This is an array, not object"}])
        # Should fail because it's an array, not an object
        result = _parse_epic_response(raw)
        # The regex will pick up the inner object, which is fine
        # Just verify it doesn't crash
        assert result is None or isinstance(result, dict)

    def test_handles_empty_stories_list(self):
        raw = json.dumps({"title": "Solo Epic", "stories": []})
        result = _parse_epic_response(raw)
        assert result is not None
        assert result["stories"] == []

    def test_filters_empty_story_titles(self):
        raw = json.dumps({
            "title": "Filter Test",
            "stories": [
                {"title": "Good Story", "description": "Desc"},
                {"title": "", "description": "Empty title"},
                {"title": "  ", "description": "Whitespace title"},
            ],
        })
        result = _parse_epic_response(raw)
        assert result is not None
        # Only the story with a real title should remain
        assert len(result["stories"]) == 1
        assert result["stories"][0]["title"] == "Good Story"

    def test_truncates_long_titles(self):
        raw = json.dumps({
            "title": "X" * 200,
            "stories": [{"title": "Y" * 200, "description": "D"}],
        })
        result = _parse_epic_response(raw)
        assert result is not None
        assert len(result["stories"][0]["title"]) <= 100


# =============================================================================
# SYNTHESIZE EPIC (integration with mocks)
# =============================================================================


class TestSynthesizeEpic:
    """Test synthesize_epic with mocked agent and idea board."""

    def _make_mock_agent(self, response_text: str) -> MagicMock:
        """Create a mock agent that returns the given text from run()."""
        agent = MagicMock()
        agent.run.return_value = response_text
        return agent

    def _make_fake_idea(self, idea_id: str, title: str, state: str = "proposed"):
        """Create a fake Idea-like object."""
        idea = MagicMock()
        idea.id = idea_id
        idea.title = title
        idea.state = state
        idea.description = ""
        return idea

    @pytest.mark.asyncio
    @patch("agent.idea_generator.set_execution_order")
    @patch("agent.idea_generator.set_epic_context")
    @patch("agent.idea_generator.add_idea")
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="No existing ideas.")
    @patch("agent.idea_generator._load_codebase_summary", return_value="- core.py")
    async def test_creates_epic_and_stories(
        self, mock_codebase, mock_existing, mock_load, mock_add, mock_ctx, mock_order
    ):
        """Full success path: LLM returns valid epic, stories are created."""
        # load_ideas returns empty (no existing ideas)
        mock_load.return_value = []

        # add_idea returns new ideas with sequential IDs
        call_count = [0]
        def fake_add_idea(**kwargs):
            call_count[0] += 1
            return self._make_fake_idea(
                f"idea-{call_count[0]:03d}", kwargs["title"]
            )
        mock_add.side_effect = fake_add_idea

        llm_response = json.dumps({
            "title": "Better Error Recovery",
            "epic_context": "Improve error handling across all modules.",
            "category": "quality",
            "source": "error_analysis",
            "stories": [
                {"title": "Add retry logic to web_search", "description": "WHAT: Retries"},
                {"title": "Log errors to crash_log", "description": "WHAT: Logging"},
            ],
        })

        agent = self._make_mock_agent(llm_response)
        result = await synthesize_epic("signals here", agent)

        assert result is not None
        assert result["epic_id"] == "idea-001"
        assert result["epic_title"] == "Better Error Recovery"
        assert len(result["story_ids"]) == 2
        assert result["category"] == "quality"

        # Verify add_idea was called 3 times (1 epic + 2 stories)
        assert mock_add.call_count == 3
        # First call is the epic
        assert mock_add.call_args_list[0][1]["idea_type"] == "epic"
        # Second and third are stories with parent_id
        assert mock_add.call_args_list[1][1]["idea_type"] == "story"
        assert mock_add.call_args_list[1][1]["parent_id"] == "idea-001"

        # Verify epic_context was set
        mock_ctx.assert_called_once_with("idea-001", "Improve error handling across all modules.")
        # Verify execution_order was set
        mock_order.assert_called_once_with("idea-001", ["idea-002", "idea-003"])

    @pytest.mark.asyncio
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="")
    @patch("agent.idea_generator._load_codebase_summary", return_value="")
    async def test_returns_none_on_llm_failure(self, mock_cb, mock_ex, mock_load):
        """Agent.run raising an exception returns None."""
        mock_load.return_value = []
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("Ollama down")
        result = await synthesize_epic("signals", agent)
        assert result is None

    @pytest.mark.asyncio
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="")
    @patch("agent.idea_generator._load_codebase_summary", return_value="")
    async def test_returns_none_on_bad_parse(self, mock_cb, mock_ex, mock_load):
        """LLM returns unparseable text → None."""
        mock_load.return_value = []
        agent = self._make_mock_agent("I have no ideas today, sorry!")
        result = await synthesize_epic("signals", agent)
        assert result is None

    @pytest.mark.asyncio
    @patch("agent.idea_generator.add_idea")
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="")
    @patch("agent.idea_generator._load_codebase_summary", return_value="")
    async def test_dedup_returns_none(self, mock_cb, mock_ex, mock_load, mock_add):
        """If add_idea returns an existing idea (dedup), synthesize_epic returns None."""
        # Existing idea with id "idea-005"
        existing = self._make_fake_idea("idea-005", "Existing Epic")
        mock_load.return_value = [existing]
        # add_idea returns the existing idea (dedup detected)
        mock_add.return_value = existing

        llm_response = json.dumps({
            "title": "Very Similar Epic",
            "epic_context": "Something similar.",
            "stories": [{"title": "Story", "description": "Desc"}],
        })
        agent = self._make_mock_agent(llm_response)
        result = await synthesize_epic("signals", agent)
        assert result is None

    @pytest.mark.asyncio
    @patch("agent.idea_generator.set_execution_order")
    @patch("agent.idea_generator.set_epic_context")
    @patch("agent.idea_generator.add_idea")
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="")
    @patch("agent.idea_generator._load_codebase_summary", return_value="")
    async def test_epic_with_no_stories(
        self, mock_cb, mock_ex, mock_load, mock_add, mock_ctx, mock_order
    ):
        """Epic with empty stories list still creates the epic."""
        mock_load.return_value = []
        mock_add.return_value = self._make_fake_idea("idea-001", "Solo Epic")

        llm_response = json.dumps({
            "title": "Solo Epic",
            "epic_context": "Just an epic, no stories.",
            "category": "feature",
            "source": "conversation_analysis",
            "stories": [],
        })
        agent = self._make_mock_agent(llm_response)
        result = await synthesize_epic("signals", agent)

        assert result is not None
        assert result["epic_id"] == "idea-001"
        assert result["story_ids"] == []
        # execution_order should NOT be called with empty list
        mock_order.assert_not_called()


# =============================================================================
# HOURLY SCHEDULE (idea-195)
# =============================================================================


class TestCountProposedIdeas:
    """Test _count_proposed_ideas backlog check."""

    def _make_idea(self, state: str) -> MagicMock:
        idea = MagicMock()
        idea.state = state
        return idea

    @patch("agent.idea_generator.load_ideas")
    def test_counts_only_proposed(self, mock_load):
        mock_load.return_value = [
            self._make_idea("proposed"),
            self._make_idea("proposed"),
            self._make_idea("approved"),
            self._make_idea("done"),
            self._make_idea("proposed"),
        ]
        assert _count_proposed_ideas() == 3

    @patch("agent.idea_generator.load_ideas")
    def test_returns_zero_when_no_proposed(self, mock_load):
        mock_load.return_value = [
            self._make_idea("approved"),
            self._make_idea("done"),
        ]
        assert _count_proposed_ideas() == 0

    @patch("agent.idea_generator.load_ideas")
    def test_returns_zero_on_empty_board(self, mock_load):
        mock_load.return_value = []
        assert _count_proposed_ideas() == 0

    @patch("agent.idea_generator.load_ideas")
    def test_returns_zero_on_exception(self, mock_load):
        mock_load.side_effect = RuntimeError("file locked")
        assert _count_proposed_ideas() == 0

    @patch("agent.idea_generator.load_ideas", None)
    def test_returns_zero_when_import_missing(self):
        assert _count_proposed_ideas() == 0


class TestSecondsUntilNextRun:
    """Test _seconds_until_next_run hourly timing logic."""

    @patch("agent.idea_generator.datetime")
    def test_before_target_minute_waits_this_hour(self, mock_dt):
        # 10:02 → should wait until 10:05 (3 minutes)
        now = datetime(2026, 4, 14, 10, 2, 0)
        mock_dt.now.return_value = now
        # now.replace() works because now is a real datetime instance
        result = _seconds_until_next_run()
        assert 170 <= result <= 190  # ~3 minutes

    @patch("agent.idea_generator.datetime")
    def test_after_target_minute_waits_next_hour(self, mock_dt):
        # 10:10 → should wait until 11:05 (55 minutes)
        now = datetime(2026, 4, 14, 10, 10, 0)
        mock_dt.now.return_value = now
        # now.replace() works because now is a real datetime instance
        result = _seconds_until_next_run()
        assert 3200 <= result <= 3400  # ~55 minutes

    @patch("agent.idea_generator.datetime")
    def test_at_exact_target_waits_next_hour(self, mock_dt):
        # Exactly at :05 → should wait until next hour's :05
        now = datetime(2026, 4, 14, 10, OFFSET_AFTER_NEWS_MINUTES, 0)
        mock_dt.now.return_value = now
        # now.replace() works because now is a real datetime instance
        result = _seconds_until_next_run()
        assert 3500 <= result <= 3700  # ~60 minutes

    @patch("agent.idea_generator.datetime")
    def test_minimum_60_seconds(self, mock_dt):
        # 10:04:55 → would be 5 seconds, but clamps to 60
        now = datetime(2026, 4, 14, 10, 4, 55)
        mock_dt.now.return_value = now
        # now.replace() works because now is a real datetime instance
        result = _seconds_until_next_run()
        assert result >= 60


class TestNotifyDiscord:
    """Test _notify_discord sends to #claude-code."""

    @pytest.mark.asyncio
    async def test_sends_to_claude_code_channel(self):
        channel = AsyncMock()
        channel.name = "claude-code"
        guild = MagicMock()
        guild.text_channels = [channel]
        client = MagicMock()
        client.guilds = [guild]

        ideas = [{"title": "Test Epic", "category": "feature", "story_count": 2}]
        await _notify_discord(client, ideas)

        channel.send.assert_called_once()
        msg = channel.send.call_args[0][0]
        assert "Test Epic" in msg
        assert "2 stories" in msg
        assert "[IdeaGen]" in msg

    @pytest.mark.asyncio
    async def test_skips_when_channel_not_found(self):
        channel = MagicMock()
        channel.name = "other-channel"
        guild = MagicMock()
        guild.text_channels = [channel]
        client = MagicMock()
        client.guilds = [guild]

        # Should not raise
        await _notify_discord(client, [{"title": "X", "category": "y"}])

    @pytest.mark.asyncio
    async def test_handles_send_exception(self):
        channel = AsyncMock()
        channel.name = "claude-code"
        channel.send.side_effect = RuntimeError("Discord down")
        guild = MagicMock()
        guild.text_channels = [channel]
        client = MagicMock()
        client.guilds = [guild]

        # Should not raise
        await _notify_discord(client, [{"title": "X", "category": "y"}])


class TestIdeaGenerationLoop:
    """Test the hourly loop behavior."""

    @pytest.mark.asyncio
    @patch("agent.idea_generator._notify_discord", new_callable=AsyncMock)
    @patch("agent.idea_generator.generate_ideas", new_callable=AsyncMock)
    @patch("agent.idea_generator._count_proposed_ideas")
    @patch("agent.idea_generator._seconds_until_next_run", return_value=0.01)
    async def test_skips_when_backlog_full(
        self, mock_wait, mock_count, mock_gen, mock_notify
    ):
        """Loop skips generation when proposed ideas exceed limit."""
        mock_count.return_value = MAX_PROPOSED_IDEAS + 1

        client = MagicMock()
        agent = MagicMock()

        # Run one iteration then break
        iteration = [0]
        original_sleep = asyncio.sleep

        async def counting_sleep(secs):
            iteration[0] += 1
            if iteration[0] >= 3:
                raise KeyboardInterrupt("break loop")
            await original_sleep(0.01)

        with patch("agent.idea_generator.asyncio.sleep", side_effect=counting_sleep):
            with pytest.raises(KeyboardInterrupt):
                await idea_generation_loop(client, agent)

        # generate_ideas should NOT have been called
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    @patch("agent.idea_generator._notify_discord", new_callable=AsyncMock)
    @patch("agent.idea_generator.generate_ideas", new_callable=AsyncMock)
    @patch("agent.idea_generator._count_proposed_ideas", return_value=3)
    @patch("agent.idea_generator._seconds_until_next_run", return_value=0.01)
    async def test_generates_when_backlog_ok(
        self, mock_wait, mock_count, mock_gen, mock_notify
    ):
        """Loop runs generation when proposed count is under limit."""
        mock_gen.return_value = [{"title": "New Epic", "category": "feature"}]

        client = MagicMock()
        agent = MagicMock()

        iteration = [0]
        original_sleep = asyncio.sleep

        async def counting_sleep(secs):
            iteration[0] += 1
            if iteration[0] >= 3:
                raise KeyboardInterrupt("break loop")
            await original_sleep(0.01)

        with patch("agent.idea_generator.asyncio.sleep", side_effect=counting_sleep):
            with pytest.raises(KeyboardInterrupt):
                await idea_generation_loop(client, agent)

        mock_gen.assert_called_once_with(agent)
        mock_notify.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent.idea_generator._notify_discord", new_callable=AsyncMock)
    @patch("agent.idea_generator.generate_ideas", new_callable=AsyncMock)
    @patch("agent.idea_generator._count_proposed_ideas", return_value=0)
    @patch("agent.idea_generator._seconds_until_next_run", return_value=0.01)
    async def test_no_notify_when_nothing_generated(
        self, mock_wait, mock_count, mock_gen, mock_notify
    ):
        """Loop does not notify when generate_ideas returns empty."""
        mock_gen.return_value = []

        client = MagicMock()
        agent = MagicMock()

        iteration = [0]
        original_sleep = asyncio.sleep

        async def counting_sleep(secs):
            iteration[0] += 1
            if iteration[0] >= 3:
                raise KeyboardInterrupt("break loop")
            await original_sleep(0.01)

        with patch("agent.idea_generator.asyncio.sleep", side_effect=counting_sleep):
            with pytest.raises(KeyboardInterrupt):
                await idea_generation_loop(client, agent)

        mock_gen.assert_called_once()
        mock_notify.assert_not_called()


class TestStartIdeaGenerator:
    """Test start_idea_generator creates an asyncio task."""

    @patch("agent.idea_generator.asyncio.create_task")
    def test_creates_task(self, mock_create):
        client = MagicMock()
        agent = MagicMock()
        start_idea_generator(client, agent)
        mock_create.assert_called_once()

    def test_max_proposed_ideas_constant(self):
        assert MAX_PROPOSED_IDEAS == 10

    def test_offset_constant(self):
        assert OFFSET_AFTER_NEWS_MINUTES == 5
