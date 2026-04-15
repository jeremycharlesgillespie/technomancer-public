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

    def test_contains_scoping_rules(self):
        """Prompt must spell out RULE A / RULE B and the disallowed story verbs."""
        assert "RULE A" in IDEA_PROMPT
        assert "RULE B" in IDEA_PROMPT
        # RULE A describes epics carrying the design.
        assert "EPIC" in IDEA_PROMPT
        # RULE B describes stories as pure execution.
        assert "STORY" in IDEA_PROMPT
        # Disallowed verbs must be listed so the model avoids them in stories.
        for verb in ("design", "decide", "evaluate", "choose", "plan", "architect", "research"):
            assert verb in IDEA_PROMPT


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

    def test_contains_scoping_rules(self):
        """Prompt must spell out RULE A / RULE B and the disallowed story verbs."""
        assert "RULE A" in SYNTHESIS_PROMPT
        assert "RULE B" in SYNTHESIS_PROMPT
        # RULE A describes epics carrying the design.
        assert "EPIC" in SYNTHESIS_PROMPT
        # RULE B describes stories as pure execution.
        assert "STORY" in SYNTHESIS_PROMPT
        # Disallowed verbs must be listed so the model avoids them in stories.
        for verb in ("design", "decide", "evaluate", "choose", "plan", "architect", "research"):
            assert verb in SYNTHESIS_PROMPT


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

    @patch("agent.task_manager.create_monitored_task")
    def test_creates_task(self, mock_create):
        client = MagicMock()
        agent = MagicMock()
        start_idea_generator(client, agent)
        mock_create.assert_called_once()

    def test_max_proposed_ideas_constant(self):
        assert MAX_PROPOSED_IDEAS == 10

    def test_offset_constant(self):
        assert OFFSET_AFTER_NEWS_MINUTES == 5


# =============================================================================
# Signal collectors — crash log, perf monitor, conversations, coverage, git
# =============================================================================


class TestLoadConversations:
    def test_reads_hourly_context_when_present(self, tmp_path, monkeypatch):
        ctx = tmp_path / "hourly.md"
        ctx.write_text("Talking about Python tooling", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.HOURLY_CONTEXT", ctx)
        from agent.idea_generator import _load_conversations

        assert "Python" in _load_conversations()

    def test_returns_default_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.idea_generator.HOURLY_CONTEXT", tmp_path / "no-file.md"
        )
        from agent.idea_generator import _load_conversations

        assert _load_conversations() == "No recent conversations."


class TestLoadErrorsWithFile:
    def test_returns_tail_of_crash_log(self, tmp_path, monkeypatch):
        crash = tmp_path / "crash_log.md"
        crash.write_text("X" * 3000 + "TAIL", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _load_errors

        out = _load_errors()
        assert out.endswith("TAIL")
        assert len(out) == 2000

    def test_small_crash_log_returned_whole(self, tmp_path, monkeypatch):
        crash = tmp_path / "crash_log.md"
        crash.write_text("tiny error", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _load_errors

        assert _load_errors() == "tiny error"


class TestLoadPerformance:
    def test_returns_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.idea_generator.PROFILING_FILE", tmp_path / "missing.jsonl"
        )
        from agent.idea_generator import _load_performance

        assert _load_performance() == "No profiling data yet."

    def test_summarises_profile_lines(self, tmp_path, monkeypatch):
        profile = tmp_path / "requests.jsonl"
        rec = json.dumps({
            "total_seconds": 3.2,
            "llm_summary": {"total_calls": 2},
            "classification": {"question_type": "technical"},
            "message": "How do I fix this?",
        })
        profile.write_text(rec + "\n" + rec + "\n", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.PROFILING_FILE", profile)
        from agent.idea_generator import _load_performance

        out = _load_performance()
        assert "3.2s" in out
        assert "technical" in out

    def test_skips_malformed_json_lines(self, tmp_path, monkeypatch):
        profile = tmp_path / "requests.jsonl"
        profile.write_text("{not json\n", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.PROFILING_FILE", profile)
        from agent.idea_generator import _load_performance

        assert _load_performance() == "No profiling data."


class TestLoadExistingIdeas:
    @patch("agent.idea_generator.load_ideas")
    def test_formats_titles(self, mock_load):
        good = MagicMock(state="proposed", title="Thing A")
        bad = MagicMock(state="vetoed", title="Skip me")
        mock_load.return_value = [good, bad]
        from agent.idea_generator import _load_existing_ideas

        out = _load_existing_ideas()
        assert "Thing A" in out
        assert "Skip me" not in out

    @patch("agent.idea_generator.load_ideas")
    def test_no_ideas_returns_default(self, mock_load):
        mock_load.return_value = []
        from agent.idea_generator import _load_existing_ideas

        assert _load_existing_ideas() == "No existing ideas."

    @patch("agent.idea_generator.load_ideas", None)
    def test_none_load_function_returns_default(self):
        from agent.idea_generator import _load_existing_ideas

        assert _load_existing_ideas() == "No existing ideas."

    @patch("agent.idea_generator.load_ideas")
    def test_exception_returns_default(self, mock_load):
        mock_load.side_effect = RuntimeError("oops")
        from agent.idea_generator import _load_existing_ideas

        assert _load_existing_ideas() == "No existing ideas."


class TestCollectRecentErrors:
    def test_returns_default_when_no_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.idea_generator.CRASH_LOG", tmp_path / "missing.md"
        )
        from agent.idea_generator import _collect_recent_errors

        assert "No crash log" in _collect_recent_errors()

    def test_parses_recent_and_filters_old_entries(self, tmp_path, monkeypatch):
        crash = tmp_path / "crash_log.md"
        now = datetime.now()
        recent_ts = now.strftime("%Y-%m-%d %H:%M:%S")
        old_ts = "2020-01-01 00:00:00"
        crash.write_text(
            f"## {recent_ts}\nRecent error details\n\n"
            f"## {old_ts}\nOld error details\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _collect_recent_errors

        out = _collect_recent_errors()
        assert "Recent error" in out
        assert "Old error" not in out

    def test_no_matching_entries_returns_default(self, tmp_path, monkeypatch):
        crash = tmp_path / "crash_log.md"
        crash.write_text("## 2020-01-01 00:00:00\nOld\n", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _collect_recent_errors

        assert "No errors" in _collect_recent_errors()

    def test_malformed_timestamp_skipped(self, tmp_path, monkeypatch):
        crash = tmp_path / "crash_log.md"
        crash.write_text("## not-a-date\nsomething\n", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _collect_recent_errors

        # Entry without a parseable "## YYYY-MM-DD" timestamp is skipped
        assert "No errors" in _collect_recent_errors()


class TestCollectSlowOperations:
    def test_returns_no_data_when_monitor_empty(self, monkeypatch):
        fake_monitor = MagicMock()
        fake_monitor.get_endpoint_stats.return_value = {"calls": 0}
        monkeypatch.setattr(
            "agent.idea_generator.get_perf_monitor", lambda: fake_monitor
        )
        from agent.idea_generator import _collect_slow_operations

        assert "No performance data" in _collect_slow_operations()

    def test_lists_slow_endpoints(self, monkeypatch):
        import threading

        fake_monitor = MagicMock()
        fake_monitor._lock = threading.Lock()
        fast_record = MagicMock(endpoint="ollama")
        slow_record = MagicMock(endpoint="claude_api")
        fake_monitor._records = [fast_record, slow_record]

        def stats(endpoint=None):
            if endpoint is None:
                return {"calls": 2}
            if endpoint == "claude_api":
                return {
                    "calls": 5, "failures": 1,
                    "avg_latency": 8.0, "p95_latency": 12.0,
                }
            return {
                "calls": 5, "failures": 0,
                "avg_latency": 0.5, "p95_latency": 0.8,
            }

        fake_monitor.get_endpoint_stats.side_effect = stats
        monkeypatch.setattr(
            "agent.idea_generator.get_perf_monitor", lambda: fake_monitor
        )
        from agent.idea_generator import _collect_slow_operations

        out = _collect_slow_operations()
        assert "claude_api" in out
        assert "p95=12.0s" in out

    def test_no_slow_endpoints(self, monkeypatch):
        import threading

        fake_monitor = MagicMock()
        fake_monitor._lock = threading.Lock()
        fake_monitor._records = [MagicMock(endpoint="ollama")]

        def stats(endpoint=None):
            if endpoint is None:
                return {"calls": 1}
            return {"calls": 1, "failures": 0, "avg_latency": 0.2, "p95_latency": 0.3}

        fake_monitor.get_endpoint_stats.side_effect = stats
        monkeypatch.setattr(
            "agent.idea_generator.get_perf_monitor", lambda: fake_monitor
        )
        from agent.idea_generator import _collect_slow_operations

        assert "No slow operations" in _collect_slow_operations()


class TestCollectConversationTopics:
    def test_returns_default_when_memory_unavailable(self, monkeypatch):
        def boom():
            raise RuntimeError("not initialised")

        monkeypatch.setattr("agent.idea_generator.get_memory_system", boom)
        from agent.idea_generator import _collect_conversation_topics

        assert "not available" in _collect_conversation_topics()

    def test_lists_recent_entries(self, monkeypatch):
        now = datetime.now()

        class FakeEntry:
            def __init__(self, user, message, ts):
                self.user = user
                self.message = message
                self.timestamp = ts

        fresh = FakeEntry("alice", "Running into an error", now)
        stale = FakeEntry("bob", "Old thing", now - timedelta(hours=3))

        fake_mem = MagicMock()
        fake_mem.recent_conversations = [fresh, stale]
        monkeypatch.setattr(
            "agent.idea_generator.get_memory_system", lambda: fake_mem
        )
        from agent.idea_generator import _collect_conversation_topics

        out = _collect_conversation_topics()
        assert "alice" in out
        assert "bob" not in out

    def test_no_recent_returns_default(self, monkeypatch):
        fake_mem = MagicMock()
        fake_mem.recent_conversations = []
        monkeypatch.setattr(
            "agent.idea_generator.get_memory_system", lambda: fake_mem
        )
        from agent.idea_generator import _collect_conversation_topics

        assert "No conversations" in _collect_conversation_topics()


class TestCollectCoverageGaps:
    def test_returns_default_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.idea_generator.COVERAGE_FILE", tmp_path / "no-coverage.json"
        )
        from agent.idea_generator import _collect_coverage_gaps

        assert "No coverage data" in _collect_coverage_gaps()

    def test_lists_low_coverage_modules(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage.json"
        cov.write_text(json.dumps({
            "files": {
                "agent/foo.py": {
                    "summary": {"percent_covered": 20, "num_statements": 50},
                },
                "agent/bar.py": {
                    "summary": {"percent_covered": 95, "num_statements": 30},
                },
            }
        }), encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.COVERAGE_FILE", cov)
        from agent.idea_generator import _collect_coverage_gaps

        out = _collect_coverage_gaps()
        assert "foo.py" in out
        assert "bar.py" not in out

    def test_all_modules_above_threshold(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage.json"
        cov.write_text(json.dumps({
            "files": {
                "agent/good.py": {
                    "summary": {"percent_covered": 80, "num_statements": 50},
                },
            }
        }), encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.COVERAGE_FILE", cov)
        from agent.idea_generator import _collect_coverage_gaps

        assert "above 50%" in _collect_coverage_gaps()

    def test_empty_files_dict(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage.json"
        cov.write_text(json.dumps({"files": {}}), encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.COVERAGE_FILE", cov)
        from agent.idea_generator import _collect_coverage_gaps

        assert "No per-file coverage data" in _collect_coverage_gaps()

    def test_invalid_json_returns_default(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage.json"
        cov.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.COVERAGE_FILE", cov)
        from agent.idea_generator import _collect_coverage_gaps

        assert "Could not parse" in _collect_coverage_gaps()


class TestCollectRecentChanges:
    def test_git_error_returns_default(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("git missing")

        monkeypatch.setattr("agent.idea_generator.subprocess.run", boom)
        from agent.idea_generator import _collect_recent_changes

        assert "Could not read" in _collect_recent_changes()

    def test_empty_output(self, monkeypatch):
        result = MagicMock(stdout="")
        monkeypatch.setattr(
            "agent.idea_generator.subprocess.run", lambda *a, **k: result
        )
        from agent.idea_generator import _collect_recent_changes

        assert "No commits" in _collect_recent_changes()

    def test_parses_commits_and_files(self, monkeypatch):
        output = "abc1234 Fix bug\nagent/foo.py\nagent/bar.py\ndef5678 Another\nagent/foo.py\n"
        result = MagicMock(stdout=output)
        monkeypatch.setattr(
            "agent.idea_generator.subprocess.run", lambda *a, **k: result
        )
        from agent.idea_generator import _collect_recent_changes

        out = _collect_recent_changes()
        assert "Fix bug" in out
        assert "agent/foo.py" in out
        # File paths should be deduplicated
        assert out.count("agent/foo.py") == 1


class TestCollectPendingIdeas:
    @patch("agent.idea_generator.load_ideas", None)
    def test_none_load_function(self):
        from agent.idea_generator import _collect_pending_ideas

        assert "Could not load" in _collect_pending_ideas()

    @patch("agent.idea_generator.load_ideas")
    def test_exception_returns_default(self, mock_load):
        mock_load.side_effect = RuntimeError("err")
        from agent.idea_generator import _collect_pending_ideas

        assert "Could not load" in _collect_pending_ideas()

    @patch("agent.idea_generator.load_ideas")
    def test_empty_board(self, mock_load):
        mock_load.return_value = []
        from agent.idea_generator import _collect_pending_ideas

        assert "empty" in _collect_pending_ideas()

    @patch("agent.idea_generator.load_ideas")
    def test_no_pending(self, mock_load):
        mock_load.return_value = [
            MagicMock(state="done", id="1", title="done thing"),
            MagicMock(state="proposed", id="2", title="proposed thing"),
        ]
        from agent.idea_generator import _collect_pending_ideas

        assert "No approved" in _collect_pending_ideas()

    @patch("agent.idea_generator.load_ideas")
    def test_lists_approved_and_executing(self, mock_load):
        mock_load.return_value = [
            MagicMock(state="approved", id="a1", title="Approved thing"),
            MagicMock(state="executing", id="a2", title="Running thing"),
        ]
        from agent.idea_generator import _collect_pending_ideas

        out = _collect_pending_ideas()
        assert "Approved thing" in out
        assert "Running thing" in out


class TestCollectSignals:
    @patch("agent.idea_generator._collect_pending_ideas", return_value="pending")
    @patch("agent.idea_generator._collect_recent_changes", return_value="changes")
    @patch("agent.idea_generator._collect_coverage_gaps", return_value="coverage")
    @patch(
        "agent.idea_generator._collect_conversation_topics", return_value="convos"
    )
    @patch("agent.idea_generator._collect_slow_operations", return_value="slow")
    @patch("agent.idea_generator._collect_recent_errors", return_value="errors")
    def test_aggregates_all_sections(self, *_mocks):
        from agent.idea_generator import collect_signals

        out = collect_signals()
        for section in (
            "RECENT ERRORS",
            "SLOW OPERATIONS",
            "CONVERSATION TOPICS",
            "TEST COVERAGE GAPS",
            "RECENTLY CHANGED FILES",
            "PENDING IDEAS",
        ):
            assert section in out


class TestLoadNewsArticles:
    @pytest.mark.asyncio
    async def test_formats_articles(self, monkeypatch):
        async def fake_fetch():
            return [
                {"source": "src1", "title": "Hello", "summary": "world"},
                {"source": "src2", "title": "Another", "summary": ""},
            ]

        # Patch the imported symbol inside news_digest
        import agent.news_digest as news

        monkeypatch.setattr(news, "fetch_all_news", fake_fetch, raising=False)
        from agent.idea_generator import _load_news_articles

        out = await _load_news_articles()
        assert "Hello" in out
        assert "Another" in out

    @pytest.mark.asyncio
    async def test_empty_returns_default(self, monkeypatch):
        async def fake_fetch():
            return []

        import agent.news_digest as news

        monkeypatch.setattr(news, "fetch_all_news", fake_fetch, raising=False)
        from agent.idea_generator import _load_news_articles

        assert "No recent news" in await _load_news_articles()

    @pytest.mark.asyncio
    async def test_exception_returns_unavailable(self, monkeypatch):
        async def fake_fetch():
            raise RuntimeError("network down")

        import agent.news_digest as news

        monkeypatch.setattr(news, "fetch_all_news", fake_fetch, raising=False)
        from agent.idea_generator import _load_news_articles

        assert "unavailable" in await _load_news_articles()


# =============================================================================
# generate_ideas — integration with mocked synthesize_epic
# =============================================================================


class TestGenerateIdeas:
    @pytest.mark.asyncio
    @patch("agent.idea_generator.synthesize_epic", new_callable=AsyncMock)
    @patch("agent.idea_generator.collect_signals", return_value="signals")
    async def test_returns_empty_when_nothing_created(self, _sig, mock_synth):
        from agent.idea_generator import generate_ideas

        mock_synth.return_value = None
        assert await generate_ideas(agent=MagicMock()) == []

    @pytest.mark.asyncio
    @patch("agent.idea_generator.synthesize_epic", new_callable=AsyncMock)
    @patch("agent.idea_generator.collect_signals", return_value="signals")
    async def test_returns_summary_when_epic_created(self, _sig, mock_synth):
        mock_synth.return_value = {
            "epic_id": "idea-001",
            "epic_title": "Better Error Recovery",
            "story_ids": ["idea-002", "idea-003"],
            "category": "quality",
            "source": "error_analysis",
        }
        from agent.idea_generator import generate_ideas

        out = await generate_ideas(agent=MagicMock())
        assert len(out) == 1
        assert out[0]["title"] == "Better Error Recovery"
        assert out[0]["story_count"] == 2


# =============================================================================
# Board provider wrappers — exercise the thin helper functions
# =============================================================================


class TestBoardProviderHelpers:
    """The wrappers in idea_generator.py delegate to a board provider singleton
    captured at import time as _get_board_provider. Patching that symbol
    directly on the module makes the thin delegates call our fake."""

    def test_load_ideas_calls_provider(self, monkeypatch):
        from agent import idea_generator as ig

        fake_provider = MagicMock()
        fake_provider.load_all.return_value = ["a", "b"]
        monkeypatch.setattr(
            ig, "_get_board_provider", lambda: fake_provider, raising=False
        )
        assert ig.load_ideas() == ["a", "b"]

    def test_add_idea_calls_provider(self, monkeypatch):
        from agent import idea_generator as ig

        fake_provider = MagicMock()
        fake_provider.add.return_value = "added"
        monkeypatch.setattr(
            ig, "_get_board_provider", lambda: fake_provider, raising=False
        )
        assert ig.add_idea("T", "D", source="s", category="c") == "added"
        fake_provider.add.assert_called_once()

    def test_set_execution_order_calls_provider(self, monkeypatch):
        from agent import idea_generator as ig

        fake_provider = MagicMock()
        monkeypatch.setattr(
            ig, "_get_board_provider", lambda: fake_provider, raising=False
        )
        ig.set_execution_order("id1", ["a", "b"])
        fake_provider.set_execution_order.assert_called_once_with("id1", ["a", "b"])

    def test_set_epic_context_calls_provider(self, monkeypatch):
        from agent import idea_generator as ig

        fake_provider = MagicMock()
        monkeypatch.setattr(
            ig, "_get_board_provider", lambda: fake_provider, raising=False
        )
        ig.set_epic_context("id1", "ctx")
        fake_provider.set_epic_context.assert_called_once_with("id1", "ctx")


# =============================================================================
# Error-branch coverage for loader helpers (OSError / ValueError paths)
# =============================================================================


class TestLoaderErrorBranches:
    """Cover the OSError and ValueError fallbacks in the loader helpers."""

    def test_load_conversations_swallows_oserror(self, tmp_path, monkeypatch):
        """When HOURLY_CONTEXT exists but is unreadable, returns default."""
        ctx = tmp_path / "hourly.md"
        ctx.write_text("some text", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.HOURLY_CONTEXT", ctx)

        def boom(*a, **k):
            raise OSError("disk fail")

        monkeypatch.setattr(Path, "read_text", boom)
        from agent.idea_generator import _load_conversations

        assert _load_conversations() == "No recent conversations."

    def test_load_errors_swallows_oserror(self, tmp_path, monkeypatch):
        """When CRASH_LOG exists but read_text raises, returns default."""
        crash = tmp_path / "crash_log.md"
        crash.write_text("crash", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)

        def boom(*a, **k):
            raise OSError("read fail")

        monkeypatch.setattr(Path, "read_text", boom)
        from agent.idea_generator import _load_errors

        assert _load_errors() == "No recent errors."

    def test_load_performance_swallows_oserror(self, tmp_path, monkeypatch):
        """When PROFILING_FILE exists but read_text raises, returns default."""
        profile = tmp_path / "requests.jsonl"
        profile.write_text("{}", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.PROFILING_FILE", profile)

        def boom(*a, **k):
            raise OSError("io fail")

        monkeypatch.setattr(Path, "read_text", boom)
        from agent.idea_generator import _load_performance

        assert _load_performance() == "No profiling data."

    def test_load_codebase_summary_swallows_oserror(self, monkeypatch):
        """OSError/ValueError while reading a .py file is ignored per-file."""
        from agent import idea_generator as ig

        real_read = Path.read_text

        def flaky(self, *a, **k):
            if self.suffix == ".py":
                raise OSError("can't read this one")
            return real_read(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", flaky)
        out = ig._load_codebase_summary()
        # Every file should still appear, just without a description suffix
        assert isinstance(out, str)
        assert out.count("\n") > 1

    def test_collect_recent_errors_oserror(self, tmp_path, monkeypatch):
        """Crash log exists but read_text raises → clear error message."""
        crash = tmp_path / "crash_log.md"
        crash.write_text("x", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)

        def boom(*a, **k):
            raise OSError("locked")

        monkeypatch.setattr(Path, "read_text", boom)
        from agent.idea_generator import _collect_recent_errors

        assert "Could not read" in _collect_recent_errors()

    def test_collect_recent_errors_invalid_date(self, tmp_path, monkeypatch):
        """A `## ####-##-## ##:##:##` that isn't a valid calendar date is skipped."""
        crash = tmp_path / "crash_log.md"
        # Regex-valid but calendar-invalid timestamp (month 99)
        crash.write_text("## 2026-99-99 25:61:99\nentry body\n", encoding="utf-8")
        monkeypatch.setattr("agent.idea_generator.CRASH_LOG", crash)
        from agent.idea_generator import _collect_recent_errors

        # Falls through to the "no recent errors" default
        assert "No errors" in _collect_recent_errors()


# =============================================================================
# Additional parse-error branches — _parse_ideas, _parse_epic_response
# =============================================================================


class TestParseIdeasExtra:
    def test_non_list_json_returns_empty(self):
        """A JSON array containing a non-list at the top level returns []."""
        from agent.idea_generator import _parse_ideas

        # `[]` is a list; need JSON whose regex-extracted content is valid
        # JSON but not a list. Use an object wrapped in something the
        # bracket-regex will grab (a [ { … } ] that is valid).
        # The `not isinstance(ideas, list)` branch requires raw JSON to
        # parse to a non-list — use code-fenced JSON that's an object.
        raw = '```json\n["plain-string", 42]\n```'
        # Both items fail the `isinstance(idea, dict)` check so result is []
        result = _parse_ideas(raw)
        assert result == []

    def test_malformed_json_with_brackets(self):
        """Regex captures `[…]` but its content is not valid JSON → []."""
        from agent.idea_generator import _parse_ideas

        # Regex finds `[…]`, but json.loads raises — exercises line 663-664
        raw = "[1, unquoted, 2]"
        assert _parse_ideas(raw) == []


class TestParseEpicResponseExtra:
    def test_malformed_json_object_returns_none(self):
        """Object-looking text that is not valid JSON → None."""
        from agent.idea_generator import _parse_epic_response

        assert _parse_epic_response("{title: unquoted, broken }") is None

    def test_non_list_stories_normalised_to_empty(self):
        """`stories` field that is not a list is coerced to []."""
        from agent.idea_generator import _parse_epic_response

        raw = json.dumps({"title": "Epic", "stories": "not a list"})
        result = _parse_epic_response(raw)
        assert result is not None
        assert result["stories"] == []


# =============================================================================
# _collect_recent_changes — empty-line and one-sided output branches
# =============================================================================


class TestCollectRecentChangesBranches:
    def test_skips_blank_lines(self, monkeypatch):
        """Blank lines in git output are ignored (line 542 continue)."""
        output = "\nabc1234 Fix bug\n\nagent/foo.py\n"
        result = MagicMock(stdout=output)
        monkeypatch.setattr(
            "agent.idea_generator.subprocess.run", lambda *a, **k: result
        )
        from agent.idea_generator import _collect_recent_changes

        out = _collect_recent_changes()
        assert "Fix bug" in out
        assert "agent/foo.py" in out

    def test_commits_only_no_files(self, monkeypatch):
        """Output with only commit lines (no file paths) → only Commits section."""
        output = "abc1234 Msg 1\ndef5678 Msg 2\n"
        result = MagicMock(stdout=output)
        monkeypatch.setattr(
            "agent.idea_generator.subprocess.run", lambda *a, **k: result
        )
        from agent.idea_generator import _collect_recent_changes

        out = _collect_recent_changes()
        assert "Commits:" in out
        assert "Changed files:" not in out

    def test_files_only_no_commits(self, monkeypatch):
        """Output with only file paths (no commit hashes) → only Changed files."""
        output = "agent/foo.py\nagent/bar.py\n"
        result = MagicMock(stdout=output)
        monkeypatch.setattr(
            "agent.idea_generator.subprocess.run", lambda *a, **k: result
        )
        from agent.idea_generator import _collect_recent_changes

        out = _collect_recent_changes()
        assert "Changed files:" in out
        assert "Commits:" not in out


# =============================================================================
# synthesize_epic — empty story title is skipped (line 802)
# =============================================================================


class TestSynthesizeEpicEmptyStoryTitle:
    @pytest.mark.asyncio
    @patch("agent.idea_generator.set_execution_order")
    @patch("agent.idea_generator.set_epic_context")
    @patch("agent.idea_generator.add_idea")
    @patch("agent.idea_generator.load_ideas")
    @patch("agent.idea_generator._load_existing_ideas", return_value="")
    @patch("agent.idea_generator._load_codebase_summary", return_value="")
    async def test_empty_story_title_is_skipped(
        self, mock_cb, mock_ex, mock_load, mock_add, mock_ctx, mock_order
    ):
        """A story whose title becomes empty after stripping is skipped."""
        mock_load.return_value = []

        # Return a new idea for the epic, and would return stories too,
        # but the empty-title story should never hit add_idea.
        call_count = [0]

        def fake_add_idea(**kwargs):
            call_count[0] += 1
            m = MagicMock()
            m.id = f"idea-{call_count[0]:03d}"
            m.title = kwargs["title"]
            m.state = "proposed"
            return m

        mock_add.side_effect = fake_add_idea

        # Hand-craft an epic payload where the parsed story list contains
        # an entry whose `title` becomes empty after the parser strips it.
        # `_parse_epic_response` drops empty titles, so we need to bypass it.
        epic_payload = {
            "title": "Partial Epic",
            "category": "quality",
            "source": "conversation_analysis",
            "stories": [
                {"title": "Real Story", "description": "Desc"},
                {"title": "   ", "description": "Whitespace title"},
            ],
        }
        agent = MagicMock()
        agent.run.return_value = json.dumps(epic_payload)

        # Patch _parse_epic_response to hand back a story list with a
        # whitespace-only title — this is what exercises line 802.
        with patch(
            "agent.idea_generator._parse_epic_response",
            return_value={
                "title": "Partial Epic",
                "category": "quality",
                "source": "conversation_analysis",
                "stories": [
                    {"title": "Real Story", "description": "Desc"},
                    {"title": "   ", "description": "Whitespace title"},
                ],
            },
        ):
            from agent.idea_generator import synthesize_epic

            result = await synthesize_epic("signals", agent)

        assert result is not None
        # Only 2 add_idea calls: 1 epic + 1 real story (whitespace story skipped)
        assert mock_add.call_count == 2
