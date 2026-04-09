"""Tests for the news_engagement module — article engagement tracking."""

import sqlite3
from unittest.mock import patch

import pytest

from agent.news_engagement import (
    get_engagement_report,
    get_news_engagement_tools,
    get_source_stats,
    get_top_articles,
    init_db,
    is_news_message,
    record_article_sent,
    record_reaction,
    record_reply,
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirect SQLite to a temp directory."""
    monkeypatch.setattr("agent.news_engagement.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.news_engagement.DB_PATH", tmp_path / "news_engagement.db")
    import agent.news_engagement as mod
    if hasattr(mod._local, "conn"):
        del mod._local.conn
    init_db()


class TestInitDb:
    def test_creates_tables(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "news_engagement.db"))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "news_articles" in tables
        assert "engagement_events" in tables

    def test_idempotent(self):
        init_db()
        init_db()


class TestRecordArticleSent:
    def test_records_article(self):
        aid = record_article_sent("123456", "abc123", "Test Article", "TechCrunch", "https://tc.com")
        assert aid > 0

    def test_returns_existing_on_duplicate(self):
        aid1 = record_article_sent("123456", "abc123", "Test", "Source")
        aid2 = record_article_sent("123456", "abc123", "Test", "Source")
        assert aid1 == aid2

    def test_different_messages_different_ids(self):
        aid1 = record_article_sent("111", "hash1", "Article 1", "Source A")
        aid2 = record_article_sent("222", "hash2", "Article 2", "Source B")
        assert aid1 != aid2


class TestIsNewsMessage:
    def test_tracked_message(self):
        record_article_sent("999", "hash", "Title", "Source")
        assert is_news_message("999") is True

    def test_untracked_message(self):
        assert is_news_message("000") is False


class TestRecordReaction:
    def test_records_reaction(self):
        record_article_sent("100", "h1", "Title", "Source")
        record_reaction("100", "👍", "Jeremy")
        # Verify via stats
        stats = get_source_stats(days=1)
        assert stats[0]["reactions"] == 1

    def test_ignores_unknown_message(self):
        # Should not raise
        record_reaction("nonexistent", "👍", "Jeremy")

    def test_multiple_reactions(self):
        record_article_sent("200", "h2", "Title", "Source")
        record_reaction("200", "👍", "Jeremy")
        record_reaction("200", "🔥", "Jeremy")
        record_reaction("200", "👀", "Alice")
        stats = get_source_stats(days=1)
        assert stats[0]["reactions"] == 3


class TestRecordReply:
    def test_records_reply(self):
        record_article_sent("300", "h3", "Title", "Source")
        record_reply("300", "Jeremy", "Interesting article!")
        stats = get_source_stats(days=1)
        assert stats[0]["replies"] == 1

    def test_ignores_unknown_message(self):
        record_reply("nonexistent", "Jeremy", "text")

    def test_truncates_long_snippets(self):
        record_article_sent("400", "h4", "Title", "Source")
        record_reply("400", "Jeremy", "x" * 500)
        # Should not raise, snippet truncated to 200


class TestGetSourceStats:
    def test_empty(self):
        stats = get_source_stats(days=30)
        assert stats == []

    def test_multiple_sources(self):
        record_article_sent("500", "h5", "Title A", "TechCrunch")
        record_article_sent("501", "h6", "Title B", "Hacker News")
        record_reaction("500", "👍", "Jeremy")
        record_reaction("500", "🔥", "Jeremy")
        record_reply("501", "Jeremy", "Cool")

        stats = get_source_stats(days=30)
        assert len(stats) == 2
        tc = next(s for s in stats if s["source"] == "TechCrunch")
        hn = next(s for s in stats if s["source"] == "Hacker News")
        assert tc["reactions"] == 2
        assert hn["replies"] == 1


class TestGetTopArticles:
    def test_empty(self):
        assert get_top_articles(days=30) == []

    def test_ranks_by_engagement(self):
        record_article_sent("600", "h7", "Popular Article", "Source")
        record_article_sent("601", "h8", "Boring Article", "Source")
        record_reaction("600", "👍", "Jeremy")
        record_reaction("600", "🔥", "Jeremy")
        record_reply("600", "Jeremy", "Great!")

        top = get_top_articles(days=30, limit=5)
        assert len(top) == 1  # Only "Popular" has engagement
        assert top[0]["title"] == "Popular Article"
        assert top[0]["engagements"] == 3


class TestGetEngagementReport:
    def test_empty_report(self):
        report = get_engagement_report(days=30)
        assert "No news articles" in report

    def test_report_with_data(self):
        record_article_sent("700", "h9", "Test Article", "TechCrunch")
        record_reaction("700", "👍", "Jeremy")
        report = get_engagement_report(days=30)
        assert "News Engagement Report" in report
        assert "TechCrunch" in report
        assert "1" in report  # articles sent


class TestGetTools:
    def test_returns_one_tool(self):
        tools = get_news_engagement_tools()
        assert len(tools) == 1
        assert tools[0].name == "news_engagement_report"

    def test_tool_runs(self):
        tools = get_news_engagement_tools()
        result = tools[0].function()
        assert isinstance(result, str)
