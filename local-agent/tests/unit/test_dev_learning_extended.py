"""Extended tests for dev_learning — article management, topic selection, history."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.dev_learning import (
    LEARNING_TOPICS,
    get_learning_article,
    get_unsent_topic,
    handle_learning_history_command,
    handle_show_learning_command,
    list_learning_articles,
    save_learning_article,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Python Decorators") == "python-decorators"

    def test_special_chars(self):
        result = slugify("What's the 'best' approach?")
        assert "'" not in result
        assert "?" not in result

    def test_truncates_long(self):
        result = slugify("a" * 100)
        assert len(result) <= 50

    def test_empty_string(self):
        assert slugify("") == ""

    def test_numbers(self):
        assert slugify("Python 3.12") == "python-3-12"


class TestSaveLearningArticle:
    def test_saves_markdown_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        monkeypatch.setattr("agent.dev_learning.settings", MagicMock(github_pages_enabled=False))

        path, url = save_learning_article("Test Topic", "python", "Article content here")
        assert path.exists()
        assert "Test Topic" in path.read_text(encoding="utf-8")
        assert url is None  # github pages disabled

    def test_filename_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        monkeypatch.setattr("agent.dev_learning.settings", MagicMock(github_pages_enabled=False))

        path, _ = save_learning_article("My Topic", "oracle", "Content")
        assert "oracle" in path.name
        assert "my-topic" in path.name


class TestListLearningArticles:
    def test_empty_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        assert list_learning_articles() == []

    def test_lists_articles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        (tmp_path / "2026-04-09_python_decorators.md").write_text(
            "---\ntopic: Python Decorators\ncategory: python\n---\nContent"
        )
        (tmp_path / "2026-04-08_oracle_hints.md").write_text(
            "---\ntopic: Oracle Hints\ncategory: oracle\n---\nContent"
        )

        articles = list_learning_articles(limit=10)
        assert len(articles) == 2
        assert articles[0]["topic"] == "Python Decorators"

    def test_respects_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        for i in range(5):
            (tmp_path / f"2026-04-0{i}_python_topic-{i}.md").write_text(
                f"---\ntopic: Topic {i}\ncategory: python\n---\nContent"
            )
        assert len(list_learning_articles(limit=3)) == 3


class TestGetLearningArticle:
    def test_returns_article(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        (tmp_path / "2026-04-09_python_decorators.md").write_text(
            "---\ntopic: Decorators\ncategory: python\ndate: 2026-04-09\n---\n\n# Decorators\n\nContent here"
        )
        result = get_learning_article(1)
        assert result is not None
        topic, content = result
        assert topic == "Decorators"
        assert "Content here" in content

    def test_invalid_number(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        assert get_learning_article(999) is None

    def test_zero_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        assert get_learning_article(0) is None


class TestGetUnsentTopic:
    def test_returns_topic(self, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.load_sent_topics", lambda: [])
        result = get_unsent_topic()
        assert result is not None
        category, topic = result
        assert category in LEARNING_TOPICS

    def test_filters_by_category(self, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.load_sent_topics", lambda: [])
        result = get_unsent_topic("python")
        assert result is not None
        assert result[0] == "python"

    def test_invalid_category_returns_from_all(self, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.load_sent_topics", lambda: [])
        result = get_unsent_topic("nonexistent")
        assert result is not None


class TestHandleLearningHistoryCommand:
    def test_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        result = handle_learning_history_command()
        assert "No learning articles" in result

    def test_with_articles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        (tmp_path / "2026-04-09_python_test.md").write_text(
            "---\ntopic: Test\ncategory: python\n---\nContent"
        )
        result = handle_learning_history_command()
        assert "Test" in result


class TestHandleShowLearningCommand:
    def test_invalid_number(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        result = handle_show_learning_command("abc")
        assert "Invalid" in result

    def test_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.dev_learning.LEARNING_ARTICLES_DIR", tmp_path)
        result = handle_show_learning_command("999")
        assert "not found" in result.lower() or "No article" in result
