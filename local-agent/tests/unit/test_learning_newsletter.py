"""Tests for the learning_newsletter module — weekly digest generation."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.learning_newsletter import (
    _get_articles_for_week,
    _read_article_excerpt,
    _split_message,
    generate_newsletter,
    handle_newsletter_command,
)


@pytest.fixture
def temp_learning_dir(tmp_path, monkeypatch):
    """Set up a temp learning articles directory with sample articles."""
    learning_dir = tmp_path / "Learning"
    learning_dir.mkdir()
    monkeypatch.setattr("agent.learning_newsletter.LEARNING_ARTICLES_DIR", learning_dir)

    today = datetime.now()
    # Create articles for this week
    for i in range(3):
        day = today - timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")
        filename = f"{date_str}_python_topic-{i}.md"
        content = f"""---
topic: Python Topic {i}
category: python
date: {date_str} 10:00
---

# Python Topic {i}

*Category: python*

---

This is a detailed article about Python topic {i}. It covers many aspects
of the subject including practical examples and best practices for real-world
software engineering applications.
"""
        (learning_dir / filename).write_text(content, encoding="utf-8")

    return learning_dir


class TestGetArticlesForWeek:
    def test_finds_this_weeks_articles(self, temp_learning_dir):
        with patch("agent.learning_newsletter.list_learning_articles") as mock_list:
            today = datetime.now()
            mock_list.return_value = [
                {"date": today.strftime("%Y-%m-%d"), "category": "python",
                 "topic": "Topic 0", "filename": f"{today.strftime('%Y-%m-%d')}_python_topic-0.md",
                 "number": "1"},
                {"date": (today - timedelta(days=1)).strftime("%Y-%m-%d"), "category": "python",
                 "topic": "Topic 1", "filename": f"{(today - timedelta(days=1)).strftime('%Y-%m-%d')}_python_topic-1.md",
                 "number": "2"},
            ]
            articles = _get_articles_for_week()
        assert len(articles) >= 1

    def test_empty_week(self):
        with patch("agent.learning_newsletter.list_learning_articles", return_value=[]):
            articles = _get_articles_for_week()
        assert articles == []

    def test_filters_by_week_boundary(self):
        today = datetime.now()
        old_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        with patch("agent.learning_newsletter.list_learning_articles") as mock_list:
            mock_list.return_value = [
                {"date": old_date, "category": "python",
                 "topic": "Old Article", "filename": "old.md", "number": "1"},
            ]
            articles = _get_articles_for_week()
        assert articles == []


class TestReadArticleExcerpt:
    def test_reads_excerpt(self, temp_learning_dir):
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}_python_topic-0.md"
        excerpt = _read_article_excerpt(filename)
        assert "detailed article" in excerpt.lower()

    def test_missing_file(self, temp_learning_dir):
        assert _read_article_excerpt("nonexistent.md") == ""

    def test_truncates_long_excerpt(self, temp_learning_dir):
        # Create a file with a very long paragraph
        long_file = temp_learning_dir / "long.md"
        long_file.write_text("---\ntopic: Long\n---\n\n# Long\n\n" + "word " * 200)
        excerpt = _read_article_excerpt("long.md", max_chars=100)
        assert len(excerpt) <= 110  # 100 + "..."


class TestSplitMessage:
    def test_short_message(self):
        assert _split_message("hello", 2000) == ["hello"]

    def test_splits_at_line_boundary(self):
        msg = "\n".join([f"Line {i}" for i in range(100)])
        chunks = _split_message(msg, 200)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_preserves_all_content(self):
        msg = "\n".join([f"Line {i}" for i in range(50)])
        chunks = _split_message(msg, 100)
        reassembled = "\n".join(chunks)
        assert reassembled == msg


class TestGenerateNewsletter:
    def test_empty_week(self):
        with patch("agent.learning_newsletter._get_articles_for_week", return_value=[]):
            result = generate_newsletter()
        assert "No learning articles" in result

    def test_with_articles(self):
        today = datetime.now()
        articles = [
            {"date": today.strftime("%Y-%m-%d"), "category": "python",
             "topic": "Decorators Deep Dive", "filename": "article.md", "number": "1"},
        ]
        with patch("agent.learning_newsletter._get_articles_for_week", return_value=articles), \
             patch("agent.learning_newsletter._read_article_excerpt", return_value="An excerpt."), \
             patch("agent.learning_newsletter._get_github_url", return_value=""):
            result = generate_newsletter()

        assert "Weekly Developer Learning Digest" in result
        assert "Decorators Deep Dive" in result
        assert "1 article(s)" in result
        assert "Python" in result

    def test_multiple_categories(self):
        today = datetime.now()
        articles = [
            {"date": today.strftime("%Y-%m-%d"), "category": "python",
             "topic": "Python Topic", "filename": "a.md", "number": "1"},
            {"date": today.strftime("%Y-%m-%d"), "category": "oracle",
             "topic": "Oracle Topic", "filename": "b.md", "number": "2"},
        ]
        with patch("agent.learning_newsletter._get_articles_for_week", return_value=articles), \
             patch("agent.learning_newsletter._read_article_excerpt", return_value=""), \
             patch("agent.learning_newsletter._get_github_url", return_value=""):
            result = generate_newsletter()

        assert "2 article(s)" in result
        assert "Python" in result
        assert "Oracle" in result


class TestHandleNewsletterCommand:
    def test_returns_string(self):
        with patch("agent.learning_newsletter.generate_newsletter", return_value="test"):
            result = handle_newsletter_command()
        assert result == "test"
