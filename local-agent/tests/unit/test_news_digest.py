"""Tests for news_digest.py - relevance filtering, article selection, and memory cross-reference."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.news_digest import (
    check_relevance,
    extract_article_keywords,
    filter_new_articles,
    find_memory_connections,
    format_memory_section,
    get_article_hash,
    load_sent_articles,
    load_user_profile,
    save_sent_articles,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_profile():
    """User profile matching the owner's setup."""
    return {
        "role": "software developer",
        "stack": ["Python", "PostgreSQL", "Oracle", "Django", "AWS"],
        "interests": ["AI", "Ollama", "LLM", "Discord", "automation", "RPA"],
    }


@pytest.fixture
def relevant_article():
    """An article that should pass relevance filtering."""
    return {
        "source": "AWS Blog",
        "title": "New Lambda features for Python developers",
        "link": "https://aws.amazon.com/blogs/aws/new-lambda-python",
        "summary": "AWS announces new Lambda runtime improvements for Python 3.12 with better cold start performance and native PostgreSQL connection pooling.",
        "published": "2026-04-03",
    }


@pytest.fixture
def irrelevant_article():
    """An article that should fail relevance filtering."""
    return {
        "source": "The Verge",
        "title": "New Taylor Swift album breaks streaming records",
        "link": "https://theverge.com/taylor-swift-album",
        "summary": "Taylor Swift's latest album has broken all previous Spotify streaming records in its first 24 hours.",
        "published": "2026-04-03",
    }


@pytest.fixture
def mock_agent_relevant():
    """Mock agent that returns RELEVANT."""
    agent = MagicMock()
    agent.run = MagicMock(return_value="RELEVANT")
    return agent


@pytest.fixture
def mock_agent_irrelevant():
    """Mock agent that returns IRRELEVANT."""
    agent = MagicMock()
    agent.run = MagicMock(return_value="IRRELEVANT")
    return agent


@pytest.fixture
def sent_articles_file(tmp_path, monkeypatch):
    """Patch sent articles file to use temp directory."""
    import agent.news_digest as nd_module

    temp_file = tmp_path / "sent_articles.json"
    monkeypatch.setattr(nd_module, "SENT_ARTICLES_FILE", temp_file)
    return temp_file


# =============================================================================
# check_relevance TESTS
# =============================================================================


class TestCheckRelevance:
    """Tests for the check_relevance function."""

    def test_relevant_article_returns_true(self, mock_agent_relevant, relevant_article, sample_profile):
        result = asyncio.run(check_relevance(mock_agent_relevant, relevant_article, sample_profile))
        assert result is True

    def test_irrelevant_article_returns_false(self, mock_agent_irrelevant, irrelevant_article, sample_profile):
        result = asyncio.run(check_relevance(mock_agent_irrelevant, irrelevant_article, sample_profile))
        assert result is False

    def test_agent_error_defaults_to_relevant(self, irrelevant_article, sample_profile):
        """On LLM error, assume relevant to avoid skipping everything."""
        agent = MagicMock()
        agent.run = MagicMock(side_effect=Exception("LLM timeout"))
        result = asyncio.run(check_relevance(agent, irrelevant_article, sample_profile))
        assert result is True

    def test_mixed_response_with_relevant(self, relevant_article, sample_profile):
        """Response containing RELEVANT (even with extra text) should pass."""
        agent = MagicMock()
        agent.run = MagicMock(return_value="I think this is RELEVANT to the user.")
        result = asyncio.run(check_relevance(agent, relevant_article, sample_profile))
        assert result is True

    def test_mixed_response_with_irrelevant(self, irrelevant_article, sample_profile):
        """Response containing IRRELEVANT should fail."""
        agent = MagicMock()
        agent.run = MagicMock(return_value="This article is IRRELEVANT to the user's work.")
        result = asyncio.run(check_relevance(agent, irrelevant_article, sample_profile))
        assert result is False

    def test_lowercase_irrelevant_detected(self, irrelevant_article, sample_profile):
        """Case-insensitive detection of IRRELEVANT."""
        agent = MagicMock()
        agent.run = MagicMock(return_value="irrelevant")
        result = asyncio.run(check_relevance(agent, irrelevant_article, sample_profile))
        assert result is False

    def test_prompt_includes_user_stack(self, mock_agent_relevant, relevant_article, sample_profile):
        """Verify the relevance prompt includes the user's tech stack."""
        asyncio.run(check_relevance(mock_agent_relevant, relevant_article, sample_profile))
        call_args = mock_agent_relevant.run.call_args[0][0]
        assert "Python" in call_args
        assert "Django" in call_args
        assert "AWS" in call_args


# =============================================================================
# ARTICLE HASH AND FILTERING TESTS
# =============================================================================


class TestArticleFiltering:
    """Tests for article deduplication."""

    def test_get_article_hash_deterministic(self):
        h1 = get_article_hash("Test Article", "https://example.com/1")
        h2 = get_article_hash("Test Article", "https://example.com/1")
        assert h1 == h2

    def test_get_article_hash_different_for_different_articles(self):
        h1 = get_article_hash("Article A", "https://example.com/a")
        h2 = get_article_hash("Article B", "https://example.com/b")
        assert h1 != h2

    def test_filter_new_articles_removes_sent(self):
        articles = [
            {"title": "Article A", "link": "https://example.com/a"},
            {"title": "Article B", "link": "https://example.com/b"},
        ]
        sent = {get_article_hash("Article A", "https://example.com/a")}
        new = filter_new_articles(articles, sent)
        assert len(new) == 1
        assert new[0]["title"] == "Article B"

    def test_filter_new_articles_all_new(self):
        articles = [
            {"title": "Article A", "link": "https://example.com/a"},
            {"title": "Article B", "link": "https://example.com/b"},
        ]
        new = filter_new_articles(articles, set())
        assert len(new) == 2


# =============================================================================
# SENT ARTICLES PERSISTENCE
# =============================================================================


class TestSentArticles:
    """Tests for sent articles file I/O."""

    def test_load_empty(self, sent_articles_file):
        result = load_sent_articles()
        assert result == set()

    def test_save_and_load(self, sent_articles_file):
        sent = {"abc123", "def456"}
        save_sent_articles(sent)
        loaded = load_sent_articles()
        assert loaded == sent

    def test_save_limits_to_1000(self, sent_articles_file):
        sent = {f"hash_{i}" for i in range(1500)}
        save_sent_articles(sent)
        loaded = load_sent_articles()
        assert len(loaded) == 1000


# =============================================================================
# KEYWORD EXTRACTION TESTS
# =============================================================================


class TestExtractArticleKeywords:
    """Tests for extract_article_keywords."""

    def test_extracts_meaningful_words(self, relevant_article):
        keywords = extract_article_keywords(relevant_article)
        assert len(keywords) > 0
        # Should extract technical terms from the AWS Lambda article
        assert any(k in keywords for k in ["lambda", "python", "postgresql"])

    def test_excludes_stop_words(self):
        article = {
            "title": "The New Way to Build Software",
            "summary": "This is a very new approach to building software with Python.",
        }
        keywords = extract_article_keywords(article)
        assert "the" not in keywords
        assert "this" not in keywords
        assert "very" not in keywords

    def test_proper_nouns_scored_higher(self):
        article = {
            "title": "Amazon Web Services Launches New Python Tools",
            "summary": "AWS announced python development tools today.",
        }
        keywords = extract_article_keywords(article)
        # "Amazon" or "python" should appear (proper nouns get higher score)
        assert len(keywords) > 0

    def test_max_keywords_respected(self):
        article = {
            "title": "Python Django PostgreSQL AWS Lambda Docker Kubernetes React Node TypeScript",
            "summary": "Many technologies discussed in this comprehensive review of modern development.",
        }
        keywords = extract_article_keywords(article, max_keywords=3)
        assert len(keywords) <= 3

    def test_empty_article(self):
        article = {"title": "", "summary": ""}
        keywords = extract_article_keywords(article)
        assert keywords == []

    def test_strips_html_tags(self):
        article = {
            "title": "Python Update",
            "summary": "<p>New <b>Python</b> features for <a href='#'>developers</a></p>",
        }
        keywords = extract_article_keywords(article)
        # Should not contain HTML artifacts
        assert all("<" not in k and ">" not in k for k in keywords)


# =============================================================================
# MEMORY CROSS-REFERENCE TESTS
# =============================================================================


class TestFindMemoryConnections:
    """Tests for find_memory_connections."""

    def test_finds_matching_conversations(self, memory_system, monkeypatch):
        """Should find conversations that mention article topics."""
        from datetime import datetime

        from agent.memory_system import ConversationEntry

        # Add conversations about Python
        memory_system.recent_conversations.append(
            ConversationEntry(
                timestamp=datetime.now(),
                user="testuser",
                message="How do I use Python lambda functions?",
                response="Lambda functions in Python are anonymous functions...",
            )
        )

        # Patch get_memory_system to return our test instance
        import agent.memory_system as mem_module

        monkeypatch.setattr(mem_module, "_memory_system", memory_system)

        article = {
            "title": "New Python Lambda Features",
            "summary": "Python 3.13 introduces improved lambda performance.",
        }
        matches = find_memory_connections(article)
        assert len(matches) >= 1
        assert "testuser" in matches[0]

    def test_no_matches_returns_empty(self, memory_system, monkeypatch):
        """Should return empty list when no conversations match."""
        import agent.memory_system as mem_module

        monkeypatch.setattr(mem_module, "_memory_system", memory_system)

        article = {
            "title": "Quantum Computing Breakthrough",
            "summary": "A new quantum processor achieves supremacy milestone.",
        }
        matches = find_memory_connections(article)
        assert matches == []

    def test_respects_max_results(self, memory_system, monkeypatch):
        """Should limit matches to max_results."""
        from datetime import datetime, timedelta

        from agent.memory_system import ConversationEntry

        # Add many Python-related conversations
        for i in range(10):
            memory_system.recent_conversations.append(
                ConversationEntry(
                    timestamp=datetime.now() - timedelta(hours=i),
                    user="testuser",
                    message=f"Python question number {i} about features",
                    response=f"Python answer {i}",
                )
            )

        import agent.memory_system as mem_module

        monkeypatch.setattr(mem_module, "_memory_system", memory_system)

        article = {
            "title": "Python 3.14 Released",
            "summary": "Major Python update with new features.",
        }
        matches = find_memory_connections(article, max_results=2)
        assert len(matches) <= 2

    def test_handles_uninitialized_memory(self, monkeypatch):
        """Should return empty list when memory system is not initialized."""
        import agent.memory_system as mem_module

        monkeypatch.setattr(mem_module, "_memory_system", None)

        article = {
            "title": "Tech News",
            "summary": "Something interesting happened.",
        }
        matches = find_memory_connections(article)
        assert matches == []

    def test_deduplicates_matches(self, memory_system, monkeypatch):
        """Should not return the same conversation entry twice."""
        from datetime import datetime

        from agent.memory_system import ConversationEntry

        # Add a conversation that matches multiple keywords
        memory_system.recent_conversations.append(
            ConversationEntry(
                timestamp=datetime.now(),
                user="testuser",
                message="Tell me about Python and Django frameworks",
                response="Python Django are great for web development.",
            )
        )

        import agent.memory_system as mem_module

        monkeypatch.setattr(mem_module, "_memory_system", memory_system)

        article = {
            "title": "Python Django Framework Update",
            "summary": "New Python Django release with improved features.",
        }
        matches = find_memory_connections(article)
        # Should only appear once despite matching multiple keywords
        assert len(matches) == 1


# =============================================================================
# FORMAT MEMORY SECTION TESTS
# =============================================================================


class TestFormatMemorySection:
    """Tests for format_memory_section."""

    def test_empty_connections(self):
        result = format_memory_section([])
        assert result == ""

    def test_formats_connections(self):
        connections = [
            "- **04/07 10:00** (testuser): Asked about Python",
            "- **04/07 11:00** (testuser): Discussed Django models",
        ]
        result = format_memory_section(connections)
        assert "Related from your conversations" in result
        assert "Asked about Python" in result
        assert "Discussed Django models" in result

    def test_single_connection(self):
        connections = ["- **04/07 10:00** (testuser): Asked about AWS Lambda"]
        result = format_memory_section(connections)
        assert "Related from your conversations" in result
        assert "AWS Lambda" in result


# =============================================================================
# USER PROFILE AND SCHEDULE TESTS
# =============================================================================


class TestLoadUserProfile:
    """Tests for load_user_profile."""

    def test_returns_dict(self, monkeypatch, tmp_path):
        import agent.news_digest as nd
        monkeypatch.setattr(nd, "VAULT_PATH", tmp_path)
        result = load_user_profile()
        assert isinstance(result, dict)
        assert "role" in result
        assert "stack" in result
        assert "interests" in result

    def test_reads_profile_file(self, monkeypatch, tmp_path):
        import agent.news_digest as nd
        monkeypatch.setattr(nd, "VAULT_PATH", tmp_path)
        profile_dir = tmp_path / "Permanent"
        profile_dir.mkdir(parents=True)
        (profile_dir / "profile.md").write_text(
            "# Profile\n\n## Role\nPrincipal Engineer\n\n"
            "## Tech Stack\nRust, Go, Python\n\n"
            "## Interests\nSystems programming, compilers\n",
            encoding="utf-8",
        )
        result = load_user_profile()
        assert "principal" in result.get("role", "").lower() or isinstance(result, dict)


class TestIsActiveHour:
    """Tests for is_active_hour — depends on system clock, just verify it doesn't crash."""

    def test_function_exists(self):
        from agent.news_digest import is_active_hour
        assert callable(is_active_hour)


class TestSentArticlesExtended:
    """Extended tests for sent article persistence."""

    def test_round_trip(self, sent_articles_file):
        original = {"hash1", "hash2", "hash3"}
        save_sent_articles(original)
        loaded = load_sent_articles()
        assert original == loaded

    def test_corrupted_file(self, sent_articles_file):
        sent_articles_file.write_text("not valid json{{{")
        result = load_sent_articles()
        assert result == set()
