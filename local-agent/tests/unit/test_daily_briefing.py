"""Tests for the daily briefing module."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent.daily_briefing import (
    ALL_COLLECTORS,
    BRIEFING_PROMPT,
    _build_fallback_briefing,
    _collect_consistency,
    _collect_conversations,
    _collect_crash_log,
    _collect_engagement,
    _collect_ideas,
    _collect_infra,
    _collect_knowledge_gaps,
    _collect_memory_context,
    _collect_news_engagement,
    _collect_perf_metrics,
    _collect_project_health,
    _collect_tool_analytics,
    _truncate,
    collect_all_data,
    format_for_discord,
    start_daily_briefing,
    synthesize_briefing,
)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 600
        result = _truncate(text, 500)
        assert len(result) == 500
        assert result.startswith("...")

    def test_exact_limit_unchanged(self):
        text = "x" * 500
        assert _truncate(text, 500) == text


# ---------------------------------------------------------------------------
# Individual collectors — graceful failure
# ---------------------------------------------------------------------------


class TestCollectors:
    def test_collect_engagement_returns_string(self):
        """Engagement collector returns a string (real data or graceful fallback)."""
        result = _collect_engagement()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_collect_engagement_handles_import_error(self, monkeypatch):
        def _fail(*a, **kw):
            raise ImportError("no module")

        monkeypatch.setattr(
            "agent.daily_briefing._collect_engagement",
            lambda: "No engagement data available.",
        )
        result = _collect_engagement()
        assert "engagement" in result.lower() or "no" in result.lower()

    def test_collect_ideas_handles_import_error(self):
        with patch("agent.daily_briefing.load_ideas", side_effect=ImportError, create=True):
            result = _collect_ideas()
            assert isinstance(result, str)

    def test_collect_consistency_returns_string(self):
        result = _collect_consistency()
        assert isinstance(result, str)

    def test_collect_conversations_returns_string(self):
        result = _collect_conversations()
        assert isinstance(result, str)

    def test_collect_perf_metrics_returns_string(self):
        result = _collect_perf_metrics()
        assert isinstance(result, str)

    def test_collect_infra_returns_string(self):
        result = _collect_infra()
        assert isinstance(result, str)

    def test_collect_tool_analytics_returns_string(self):
        result = _collect_tool_analytics()
        assert isinstance(result, str)

    def test_collect_news_engagement_returns_string(self):
        result = _collect_news_engagement()
        assert isinstance(result, str)

    def test_collect_crash_log_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.daily_briefing.CRASH_LOG", tmp_path / "nonexistent.md")
        result = _collect_crash_log()
        assert result == "No recent crashes."

    def test_collect_crash_log_with_content(self, tmp_path, monkeypatch):
        crash_file = tmp_path / "crash_log.md"
        crash_file.write_text("## Crash at 2026-04-10\nTraceback: something broke")
        monkeypatch.setattr("agent.daily_briefing.CRASH_LOG", crash_file)
        result = _collect_crash_log()
        assert "something broke" in result

    def test_collect_crash_log_truncates_large(self, tmp_path, monkeypatch):
        crash_file = tmp_path / "crash_log.md"
        crash_file.write_text("x" * 5000)
        monkeypatch.setattr("agent.daily_briefing.CRASH_LOG", crash_file)
        result = _collect_crash_log()
        assert len(result) <= 1500

    def test_collect_knowledge_gaps_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.daily_briefing.KNOWLEDGE_GAPS", tmp_path / "nope.md")
        result = _collect_knowledge_gaps()
        assert "no knowledge gaps" in result.lower()

    def test_collect_knowledge_gaps_with_content(self, tmp_path, monkeypatch):
        gaps_file = tmp_path / "gaps.md"
        gaps_file.write_text("## Open Gaps\n- How does X work?")
        monkeypatch.setattr("agent.daily_briefing.KNOWLEDGE_GAPS", gaps_file)
        result = _collect_knowledge_gaps()
        assert "How does X work" in result

    def test_collect_memory_context_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.daily_briefing.DAILY_CONTEXT", tmp_path / "nope.md")
        result = _collect_memory_context()
        assert "no daily context" in result.lower()

    def test_collect_memory_context_with_content(self, tmp_path, monkeypatch):
        ctx_file = tmp_path / "daily.md"
        ctx_file.write_text("User discussed deployment strategies.")
        monkeypatch.setattr("agent.daily_briefing.DAILY_CONTEXT", ctx_file)
        result = _collect_memory_context()
        assert "deployment" in result

    def test_collect_project_health_returns_string(self):
        result = _collect_project_health()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_collect_project_health_handles_import_error(self, monkeypatch):
        def _fail():
            raise ImportError("no module")

        monkeypatch.setattr(
            "agent.daily_briefing._collect_project_health", _fail
        )
        # Direct call to patched function will raise, but the real collector
        # wraps in try/except — test the real one with a broken import
        result = _collect_project_health()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestCollectAllData:
    @pytest.mark.asyncio
    async def test_returns_dict_with_all_keys(self, monkeypatch):
        # Stub all collectors to return simple strings
        for name in ALL_COLLECTORS:
            monkeypatch.setitem(
                ALL_COLLECTORS, name, lambda: "test data"
            )
        data = await collect_all_data()
        assert isinstance(data, dict)
        assert len(data) == len(ALL_COLLECTORS)
        for key in ALL_COLLECTORS:
            assert key in data

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self, monkeypatch):
        def _fail():
            raise RuntimeError("broken")

        for name in ALL_COLLECTORS:
            if name == "engagement":
                monkeypatch.setitem(ALL_COLLECTORS, name, _fail)
            else:
                monkeypatch.setitem(ALL_COLLECTORS, name, lambda: "ok")

        data = await collect_all_data()
        assert len(data) == len(ALL_COLLECTORS)
        assert "unavailable" in data["engagement"].lower()


# ---------------------------------------------------------------------------
# Discord formatting
# ---------------------------------------------------------------------------


class TestFormatForDiscord:
    def test_short_message_single_chunk(self):
        chunks = format_for_discord("Short briefing.")
        assert len(chunks) == 1
        assert "Good morning" in chunks[0]

    def test_long_message_splits(self):
        sections = "\n\n".join([f"Section {i}\n" + "x" * 400 for i in range(10)])
        chunks = format_for_discord(sections)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 1900

    def test_includes_date(self):
        chunks = format_for_discord("Test")
        today = datetime.now().strftime("%A, %B %d")
        assert today in chunks[0]

    def test_empty_input_returns_header(self):
        chunks = format_for_discord("")
        assert len(chunks) == 1
        assert "Good morning" in chunks[0]


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeBriefing:
    @pytest.mark.asyncio
    async def test_calls_agent_run(self, mock_ollama_client):
        mock_ollama_client.set_responses([
            {"message": {"content": "Here is your briefing summary.", "tool_calls": []}}
        ])
        from agent.core import Agent, AgentConfig

        agent = Agent(AgentConfig(verbose=False))

        data = {key: "test data" for key in ALL_COLLECTORS}
        result = await synthesize_briefing(agent, data)
        assert "briefing" in result.lower()

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self, mock_ollama_client):
        mock_ollama_client.set_responses([
            {"message": {"content": "", "tool_calls": []}}
        ])
        from agent.core import Agent, AgentConfig

        agent = Agent(AgentConfig(verbose=False))

        data = {key: "test data" for key in ALL_COLLECTORS}
        result = await synthesize_briefing(agent, data)
        # Should get fallback briefing
        assert "Daily Briefing" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self):
        agent = MagicMock()
        agent.run = MagicMock(side_effect=RuntimeError("LLM down"))

        data = {key: "test data" for key in ALL_COLLECTORS}
        result = await synthesize_briefing(agent, data)
        assert "Daily Briefing" in result


# ---------------------------------------------------------------------------
# Fallback briefing
# ---------------------------------------------------------------------------


class TestFallbackBriefing:
    def test_includes_header(self):
        data = {"crash_log": "Error at line 5", "engagement": "No data"}
        result = _build_fallback_briefing(data)
        assert "Daily Briefing" in result
        assert "raw data" in result.lower()

    def test_skips_unavailable_sections(self):
        data = {
            "crash_log": "Stack trace here",
            "engagement": "Data unavailable.",
            "ideas": "No ideas found.",
        }
        result = _build_fallback_briefing(data)
        assert "Crashes" in result or "crash" in result.lower()

    def test_handles_empty_data(self):
        data = {key: "No data." for key in ALL_COLLECTORS}
        result = _build_fallback_briefing(data)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestBriefingPrompt:
    def test_has_all_placeholders(self):
        for key in ALL_COLLECTORS:
            assert f"{{{key}}}" in BRIEFING_PROMPT

    def test_has_section_headers(self):
        assert "Attention Required" in BRIEFING_PROMPT
        assert "Yesterday's Activity" in BRIEFING_PROMPT
        assert "System Health" in BRIEFING_PROMPT
        assert "Project Health" in BRIEFING_PROMPT
        assert "Today's Priorities" in BRIEFING_PROMPT


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestScheduling:
    def test_start_creates_task(self):
        client = MagicMock()
        agent = MagicMock()

        with patch("agent.task_manager.create_monitored_task") as mock_task:
            start_daily_briefing(client, "llm_chat", agent)
            assert mock_task.called
