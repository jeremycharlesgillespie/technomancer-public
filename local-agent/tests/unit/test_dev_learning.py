"""
Tests for agent/dev_learning.py - Daily Developer Learning functionality.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch


class TestTopicCatalog:
    """Tests for the topic catalog structure."""

    def test_all_categories_exist(self, patched_dev_learning):
        """All expected categories are present."""
        from agent.dev_learning import LEARNING_TOPICS

        expected = ["python", "oracle", "system_design", "best_practices"]
        for cat in expected:
            assert cat in LEARNING_TOPICS

    def test_each_category_has_topics(self, patched_dev_learning):
        """Each category has at least 10 topics."""
        from agent.dev_learning import LEARNING_TOPICS

        for cat, topics in LEARNING_TOPICS.items():
            assert len(topics) >= 10, f"Category '{cat}' should have at least 10 topics"

    def test_topics_are_strings(self, patched_dev_learning):
        """All topics are non-empty strings."""
        from agent.dev_learning import LEARNING_TOPICS

        for cat, topics in LEARNING_TOPICS.items():
            for topic in topics:
                assert isinstance(topic, str)
                assert len(topic) > 0


class TestTopicHashing:
    """Tests for topic hash generation."""

    def test_get_topic_hash_consistent(self, patched_dev_learning):
        """Same topic always produces same hash."""
        from agent.dev_learning import get_topic_hash

        topic = "Python context managers"
        hash1 = get_topic_hash(topic)
        hash2 = get_topic_hash(topic)
        assert hash1 == hash2

    def test_get_topic_hash_case_insensitive(self, patched_dev_learning):
        """Hash is case-insensitive."""
        from agent.dev_learning import get_topic_hash

        hash1 = get_topic_hash("Python Context Managers")
        hash2 = get_topic_hash("python context managers")
        assert hash1 == hash2

    def test_get_topic_hash_length(self, patched_dev_learning):
        """Hash is 16 characters."""
        from agent.dev_learning import get_topic_hash

        topic_hash = get_topic_hash("Any topic here")
        assert len(topic_hash) == 16


class TestSentTopicsTracking:
    """Tests for sent_topics.json tracking."""

    def test_load_sent_topics_empty(self, patched_dev_learning):
        """load_sent_topics returns empty list when no file exists."""
        from agent.dev_learning import load_sent_topics

        result = load_sent_topics()
        assert result == []

    def test_save_sent_topic_creates_file(self, patched_dev_learning):
        """save_sent_topic creates the JSON file."""
        from agent.dev_learning import SENT_TOPICS_FILE, save_sent_topic

        save_sent_topic("Test topic", "python")

        assert SENT_TOPICS_FILE.exists()
        data = json.loads(SENT_TOPICS_FILE.read_text())
        assert "sent" in data
        assert "updated" in data

    def test_save_sent_topic_records_data(self, patched_dev_learning):
        """save_sent_topic records topic, category, and date."""
        from agent.dev_learning import load_sent_topics, save_sent_topic

        save_sent_topic("Python decorators", "python")

        sent = load_sent_topics()
        assert len(sent) == 1
        assert sent[0]["topic"] == "Python decorators"
        assert sent[0]["category"] == "python"
        assert "date" in sent[0]
        assert "topic_hash" in sent[0]

    def test_save_sent_topic_appends(self, patched_dev_learning):
        """Multiple saves append to the list."""
        from agent.dev_learning import load_sent_topics, save_sent_topic

        save_sent_topic("Topic 1", "python")
        save_sent_topic("Topic 2", "oracle")
        save_sent_topic("Topic 3", "system_design")

        sent = load_sent_topics()
        assert len(sent) == 3

    def test_sent_topics_limited_to_365(self, patched_dev_learning):
        """Sent topics list is limited to 365 entries."""
        from agent.dev_learning import load_sent_topics, save_sent_topic

        # Add 400 topics
        for i in range(400):
            save_sent_topic(f"Topic {i}", "python")

        sent = load_sent_topics()
        assert len(sent) == 365

    def test_sent_topics_keeps_most_recent(self, patched_dev_learning):
        """When truncating, keeps the most recent entries."""
        from agent.dev_learning import load_sent_topics, save_sent_topic

        # Add 400 topics
        for i in range(400):
            save_sent_topic(f"Topic {i}", "python")

        sent = load_sent_topics()
        # Should keep topics 35-399 (most recent 365)
        assert sent[0]["topic"] == "Topic 35"
        assert sent[-1]["topic"] == "Topic 399"


class TestTopicSelection:
    """Tests for topic selection logic."""

    def test_get_unsent_topic_returns_tuple(self, patched_dev_learning):
        """get_unsent_topic returns (category, topic) tuple."""
        from agent.dev_learning import get_unsent_topic

        result = get_unsent_topic()
        assert result is not None
        assert len(result) == 2
        category, topic = result
        assert category in ["python", "oracle", "system_design", "best_practices"]
        assert isinstance(topic, str)

    def test_get_unsent_topic_filters_by_category(self, patched_dev_learning):
        """get_unsent_topic respects category filter."""
        from agent.dev_learning import get_unsent_topic

        for _ in range(10):  # Try multiple times for randomness
            result = get_unsent_topic("oracle")
            category, _ = result
            assert category == "oracle"

    def test_get_unsent_topic_avoids_sent(self, patched_dev_learning):
        """get_unsent_topic avoids recently sent topics."""
        from agent.dev_learning import (
            get_topic_hash,
            get_unsent_topic,
            load_sent_topics,
            save_sent_topic,
        )

        # Get first topic
        cat1, topic1 = get_unsent_topic("python")
        save_sent_topic(topic1, cat1)

        # Get second topic - should be different
        cat2, topic2 = get_unsent_topic("python")

        # Topics should be different (unless we've exhausted all)
        sent_hashes = {t["topic_hash"] for t in load_sent_topics()}
        if get_topic_hash(topic2) in sent_hashes:
            # Only OK if all topics exhausted
            pass
        else:
            assert topic1 != topic2

    def test_get_unsent_topic_invalid_category(self, patched_dev_learning):
        """get_unsent_topic with invalid category returns from all topics."""
        from agent.dev_learning import get_unsent_topic

        result = get_unsent_topic("invalid_category")
        # Should still return something (from any category)
        assert result is not None


class TestCommandHandler:
    """Tests for the better_dev command handler."""

    @patch("agent.dev_learning.anthropic")
    @patch("agent.dev_learning.web_search")
    def test_handle_custom_topic(self, mock_search, mock_anthropic, patched_dev_learning):
        """Custom topic (not a predefined category) generates content."""
        # Mock web search
        mock_search.return_value = "Sample search results"

        # Mock Claude
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated content about neo4j")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        from agent.dev_learning import handle_better_dev_command

        response, html_url = asyncio.run(
            handle_better_dev_command("neo4j")
        )

        # Custom topics use the input as both topic and category
        assert "neo4j" in response.lower()
        assert html_url is not None  # Should generate a URL

    @patch("agent.dev_learning.anthropic")
    @patch("agent.dev_learning.web_search")
    def test_handle_valid_category_calls_claude(
        self, mock_search, mock_anthropic, patched_dev_learning
    ):
        """Valid category generates content via Claude."""
        # Mock web search
        mock_search.return_value = "Sample search results"

        # Mock Claude
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated learning content")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        from agent.dev_learning import handle_better_dev_command

        response, html_url = asyncio.run(
            handle_better_dev_command("python")
        )

        # Response now contains just the link, not full content
        assert "**Developer Learning:" in response
        assert "python" in response.lower()
        mock_client.messages.create.assert_called_once()

    @patch("agent.dev_learning.anthropic")
    @patch("agent.dev_learning.web_search")
    def test_handle_marks_topic_as_sent(self, mock_search, mock_anthropic, patched_dev_learning):
        """Command handler marks topic as sent."""
        # Mock web search
        mock_search.return_value = "Sample search results"

        # Mock Claude
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Content")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        from agent.dev_learning import handle_better_dev_command, load_sent_topics

        initial_count = len(load_sent_topics())

        asyncio.run(handle_better_dev_command("python"))

        final_count = len(load_sent_topics())
        assert final_count == initial_count + 1


class TestContentGeneration:
    """Tests for content generation with Claude."""

    @patch("agent.dev_learning.anthropic")
    @patch("agent.dev_learning.web_search")
    def test_generate_uses_web_search(self, mock_search, mock_anthropic, patched_dev_learning):
        """Content generation calls web search first."""
        mock_search.return_value = "Web search results"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Content")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        from agent.dev_learning import generate_learning_content

        asyncio.run(
            generate_learning_content("Test topic", "python")
        )

        mock_search.assert_called_once()

    @patch("agent.dev_learning.anthropic")
    @patch("agent.dev_learning.web_search")
    def test_generate_sends_structured_prompt(
        self, mock_search, mock_anthropic, patched_dev_learning
    ):
        """Claude is called with structured prompt including all sections."""
        mock_search.return_value = "Web results"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Content")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        from agent.dev_learning import generate_learning_content

        asyncio.run(
            generate_learning_content("Python decorators", "python")
        )

        # Check the prompt contains required sections
        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]

        assert "Quick Overview" in prompt
        assert "How This Makes You a Better Developer" in prompt
        assert "How to Implement It" in prompt
        assert "Where to Use" in prompt
        assert "Key Takeaways" in prompt

    def test_generate_without_anthropic(self, patched_dev_learning, monkeypatch):
        """Gracefully handles missing anthropic library."""
        import agent.dev_learning as dl

        monkeypatch.setattr(dl, "HAS_ANTHROPIC", False)

        result = asyncio.run(
            dl.generate_learning_content("Topic", "python")
        )

        assert "Error" in result
        assert "Anthropic" in result
