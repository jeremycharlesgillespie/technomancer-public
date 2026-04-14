"""Tests for the web_search module — DuckDuckGo search, URL fetching, and smart search."""

from unittest.mock import MagicMock, patch

import pytest

from agent.web_search import (
    CREDIBILITY_TIERS,
    DEFAULT_CREDIBILITY,
    DOMAIN_CREDIBILITY,
    _rewrite_query_with_llm,
    _tier_label,
    get_domain_credibility,
    get_web_tools,
    web_fetch,
    web_search,
    web_search_news,
    web_search_smart,
)


class TestWebSearch:
    """Test DuckDuckGo web search."""

    @patch("agent.web_search.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Python Docs", "body": "Official Python documentation", "href": "https://docs.python.org"},
            {"title": "Real Python", "body": "Tutorials for all levels", "href": "https://realpython.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search("python tutorial")
        assert "Python Docs" in result
        assert "Real Python" in result
        assert "https://docs.python.org" in result

    @patch("agent.web_search.DDGS")
    def test_no_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search("xyznonexistent12345")
        assert "No results found" in result

    @patch("agent.web_search.DDGS")
    def test_handles_exception(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("Network error")
        result = web_search("test")
        assert "Search error" in result

    @patch("agent.web_search.DDGS")
    def test_respects_max_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": f"Result {i}", "body": f"Body {i}", "href": f"https://example.com/{i}"}
            for i in range(3)
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search("test", max_results=3)
        assert "Result 0" in result
        mock_ddgs.text.assert_called_once_with("test", max_results=3)


class TestWebSearchNews:
    """Test DuckDuckGo news search."""

    @patch("agent.web_search.DDGS")
    def test_returns_news_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.news.return_value = [
            {"title": "Breaking News", "body": "Something happened", "source": "CNN",
             "date": "2026-04-09", "url": "https://cnn.com/article"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_news("latest news")
        assert "Breaking News" in result

    @patch("agent.web_search.DDGS")
    def test_handles_empty_news(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.news.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_news("nonexistent topic")
        assert "No results" in result or "No news" in result


class TestWebFetch:
    """Test URL content fetching."""

    @patch("agent.web_search.requests")
    def test_fetches_and_extracts(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><article><p>Hello World</p></article></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        result = web_fetch("https://example.com")
        assert "Hello World" in result

    @patch("agent.web_search.requests")
    def test_strips_scripts_and_nav(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            "<html><body>"
            "<nav>Navigation</nav>"
            "<script>alert('bad')</script>"
            "<article><p>Content here</p></article>"
            "<footer>Footer stuff</footer>"
            "</body></html>"
        )
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        result = web_fetch("https://example.com")
        assert "Content here" in result
        assert "alert" not in result
        assert "Navigation" not in result

    @patch("agent.web_search.requests")
    def test_handles_timeout(self, mock_requests):
        import requests as _req
        mock_requests.get.side_effect = _req.exceptions.Timeout("timed out")
        mock_requests.exceptions = _req.exceptions

        result = web_fetch("https://slow-site.com")
        assert "timed out" in result.lower() or "error" in result.lower()

    @patch("agent.web_search.requests")
    def test_truncates_long_content(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><article><p>" + "word " * 5000 + "</p></article></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        result = web_fetch("https://example.com")
        assert len(result) <= 9000  # 8000 content + URL header


class TestGetWebTools:
    """Test tool registration."""

    def test_returns_four_tools(self):
        tools = get_web_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "web_search_smart" in names
        assert "web_search_news" in names
        assert "web_fetch" in names

    def test_smart_search_tool_has_timeout(self):
        tools = get_web_tools()
        smart = [t for t in tools if t.name == "web_search_smart"][0]
        assert smart.timeout == 60


class TestDomainCredibility:
    """Test domain credibility scoring."""

    def test_official_docs_score_high(self):
        score, tier = get_domain_credibility("https://docs.python.org/3/library/json.html")
        assert score >= 90
        assert tier == "[OFFICIAL]"

    def test_stackoverflow_is_trusted(self):
        score, tier = get_domain_credibility("https://stackoverflow.com/questions/123")
        assert 70 <= score < 90
        assert tier == "[TRUSTED]"

    def test_reddit_is_community(self):
        score, tier = get_domain_credibility("https://www.reddit.com/r/python/comments/abc")
        assert 30 <= score < 50
        assert tier == "[COMMUNITY]"

    def test_medium_is_informative(self):
        score, tier = get_domain_credibility("https://medium.com/@author/article")
        assert 50 <= score < 70
        assert tier == "[INFO]"

    def test_unknown_domain_gets_default(self):
        score, tier = get_domain_credibility("https://randomsite12345.xyz/page")
        assert score == DEFAULT_CREDIBILITY

    def test_www_prefix_stripped(self):
        score1, _ = get_domain_credibility("https://www.stackoverflow.com/q/1")
        score2, _ = get_domain_credibility("https://stackoverflow.com/q/1")
        assert score1 == score2

    def test_parent_domain_fallback(self):
        """Subdomains like blog.python.org should match python.org patterns."""
        score, tier = get_domain_credibility("https://blog.python.org/2026/04/release.html")
        # blog. prefix pattern gives 60
        assert score > 0

    def test_docs_subdomain_pattern(self):
        """Unknown docs.* subdomains get the docs heuristic boost."""
        score, tier = get_domain_credibility("https://docs.someframework.io/guide")
        assert score == 90
        assert tier == "[OFFICIAL]"

    def test_forum_subdomain_pattern(self):
        """forum.* subdomains get low score."""
        score, _ = get_domain_credibility("https://forum.unknownsite.com/thread/1")
        assert score == 40

    def test_invalid_url_returns_default(self):
        score, _ = get_domain_credibility("not a url at all")
        assert score == DEFAULT_CREDIBILITY

    def test_empty_url_returns_default(self):
        score, _ = get_domain_credibility("")
        assert score == DEFAULT_CREDIBILITY

    def test_github_is_high_tier2(self):
        score, tier = get_domain_credibility("https://github.com/python/cpython")
        assert score >= 80

    def test_aws_docs_official(self):
        score, tier = get_domain_credibility("https://docs.aws.amazon.com/lambda/latest/dg/")
        assert score >= 90
        assert tier == "[OFFICIAL]"

    def test_all_tiers_have_labels(self):
        """Every tier threshold maps to a label."""
        for threshold, name, label in CREDIBILITY_TIERS:
            assert label.startswith("[")
            assert label.endswith("]")

    def test_tier_label_boundaries(self):
        assert _tier_label(100) == "[OFFICIAL]"
        assert _tier_label(90) == "[OFFICIAL]"
        assert _tier_label(89) == "[TRUSTED]"
        assert _tier_label(70) == "[TRUSTED]"
        assert _tier_label(69) == "[INFO]"
        assert _tier_label(50) == "[INFO]"
        assert _tier_label(49) == "[COMMUNITY]"
        assert _tier_label(30) == "[COMMUNITY]"
        assert _tier_label(29) == "[UNVERIFIED]"
        assert _tier_label(0) == "[UNVERIFIED]"


class TestQueryRewriting:
    """Test LLM query rewriting."""

    @patch("agent.web_search._ollama_client")
    @patch("agent.web_search.settings")
    def test_rewrites_query(self, mock_settings, mock_ollama):
        mock_settings.ollama_model = "qwen3.5:9b"
        mock_ollama.chat.return_value = {
            "message": {"content": "python asyncio tutorial official"}
        }

        result = _rewrite_query_with_llm("how do I use asyncio in python?")
        assert result == "python asyncio tutorial official"
        mock_ollama.chat.assert_called_once()

    @patch("agent.web_search._ollama_client")
    @patch("agent.web_search.settings")
    def test_falls_back_on_error(self, mock_settings, mock_ollama):
        mock_settings.ollama_model = "qwen3.5:9b"
        mock_ollama.chat.side_effect = Exception("Ollama down")

        result = _rewrite_query_with_llm("test query")
        assert result == "test query"

    @patch("agent.web_search._ollama_client")
    @patch("agent.web_search.settings")
    def test_falls_back_on_empty_response(self, mock_settings, mock_ollama):
        mock_settings.ollama_model = "qwen3.5:9b"
        mock_ollama.chat.return_value = {"message": {"content": ""}}

        result = _rewrite_query_with_llm("test query")
        assert result == "test query"

    @patch("agent.web_search._ollama_client")
    @patch("agent.web_search.settings")
    def test_falls_back_on_too_long_response(self, mock_settings, mock_ollama):
        mock_settings.ollama_model = "qwen3.5:9b"
        mock_ollama.chat.return_value = {"message": {"content": "x " * 200}}

        result = _rewrite_query_with_llm("test query")
        assert result == "test query"

    @patch("agent.web_search._ollama_client")
    @patch("agent.web_search.settings")
    def test_strips_thinking_tags(self, mock_settings, mock_ollama):
        mock_settings.ollama_model = "qwen3.5:9b"
        mock_ollama.chat.return_value = {
            "message": {"content": "<think>let me think</think>optimized query"}
        }

        result = _rewrite_query_with_llm("original query")
        assert result == "optimized query"


class TestWebSearchSmart:
    """Test context-aware smart search."""

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_returns_credibility_scored_results(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "python async await"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Python Docs", "body": "Official asyncio docs", "href": "https://docs.python.org/3/library/asyncio.html"},
            {"title": "Reddit Thread", "body": "User discussion", "href": "https://reddit.com/r/python/asyncio"},
            {"title": "Real Python", "body": "Asyncio tutorial", "href": "https://realpython.com/async-io-python/"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("how do I use asyncio?")

        # Official docs should appear first (highest credibility)
        python_docs_pos = result.index("Python Docs")
        reddit_pos = result.index("Reddit Thread")
        assert python_docs_pos < reddit_pos

        # Check credibility badges are present
        assert "[OFFICIAL]" in result
        assert "[COMMUNITY]" in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_shows_optimized_query(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "optimized query"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result", "body": "Body", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("original question")
        assert "optimized query" in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_no_results(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "test query"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("xyznonexistent")
        assert "No results found" in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_handles_search_exception(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "test"
        mock_ddgs_cls.side_effect = Exception("Network error")

        result = web_search_smart("test")
        assert "Search error" in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_includes_credibility_legend(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "test"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result", "body": "Body", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("test")
        assert "Credibility:" in result
        assert "[OFFICIAL]" in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_respects_max_results(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "test"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": f"Result {i}", "body": f"Body {i}", "href": f"https://example{i}.com"}
            for i in range(12)
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("test", max_results=3)
        # Count numbered results (lines starting with "N. ")
        numbered = [line for line in result.split("\n") if line and line[0].isdigit() and ". " in line[:4]]
        assert len(numbered) == 3

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_same_query_not_shown_as_optimized(self, mock_ddgs_cls, mock_rewrite):
        """When rewrite returns the same query, don't show 'optimized query' line."""
        mock_rewrite.return_value = "same query"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result", "body": "Body", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("same query")
        assert "optimized query" not in result

    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_scores_shown_in_output(self, mock_ddgs_cls, mock_rewrite):
        mock_rewrite.return_value = "django docs"

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Django Docs", "body": "Official", "href": "https://docs.djangoproject.com/en/5.0/"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = web_search_smart("django")
        assert "credibility: 95/100" in result
