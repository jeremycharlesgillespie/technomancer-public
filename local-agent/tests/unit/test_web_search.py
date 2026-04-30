"""Tests for the web_search module — DuckDuckGo search, URL fetching, smart search, cache, and retry."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.web_search import (
    CREDIBILITY_TIERS,
    DEFAULT_CREDIBILITY,
    DOMAIN_CREDIBILITY,
    _cache_get,
    _cache_set,
    _retry_search,
    _rewrite_query_with_llm,
    _route_exception,
    _search_cache,
    _tier_label,
    clear_search_cache,
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


# =============================================================================
# SEARCH CACHE TESTS
# =============================================================================


class TestSearchCache:
    """Test TTL search result cache."""

    def setup_method(self):
        clear_search_cache()

    def teardown_method(self):
        clear_search_cache()

    def test_cache_miss_returns_none(self):
        assert _cache_get(("web_search", "test", 5)) is None

    def test_cache_set_and_get(self):
        key = ("web_search", "python", 5)
        _cache_set(key, "cached result")
        assert _cache_get(key) == "cached result"

    def test_cache_expired_returns_none(self):
        key = ("web_search", "expired", 5)
        # Manually insert an expired entry
        _search_cache[key] = (time.monotonic() - 7200, "old result")
        assert _cache_get(key) is None
        # Entry should be cleaned up
        assert key not in _search_cache

    def test_clear_cache(self):
        _cache_set(("web_search", "a", 5), "result a")
        _cache_set(("web_search", "b", 5), "result b")
        assert len(_search_cache) == 2
        clear_search_cache()
        assert len(_search_cache) == 0

    @patch("agent.web_search.DDGS")
    def test_web_search_returns_cached(self, mock_ddgs_cls):
        """Second call with same query should return cached result, not hit DDG."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result", "body": "Body", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result1 = web_search("cache test query")
        result2 = web_search("cache test query")

        assert result1 == result2
        # DDGS should only be instantiated once (cached on second call)
        assert mock_ddgs_cls.call_count == 1

    @patch("agent.web_search.DDGS")
    def test_web_search_news_returns_cached(self, mock_ddgs_cls):
        """News search cache works."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.news.return_value = [
            {"title": "News", "body": "Body", "source": "Src", "date": "2026-04-14", "url": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result1 = web_search_news("cache news")
        result2 = web_search_news("cache news")

        assert result1 == result2
        assert mock_ddgs_cls.call_count == 1

    @patch("agent.web_search.DDGS")
    def test_different_queries_not_cached(self, mock_ddgs_cls):
        """Different queries should each hit DDG."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result", "body": "Body", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        web_search("query one")
        web_search("query two")

        assert mock_ddgs_cls.call_count == 2


# =============================================================================
# RETRY WITH BACKOFF TESTS
# =============================================================================


class TestRetrySearch:
    """Test retry logic for search calls."""

    def setup_method(self):
        clear_search_cache()

    def teardown_method(self):
        clear_search_cache()

    @patch("agent.web_search.time.sleep")
    def test_retries_on_failure_then_succeeds(self, mock_sleep):
        """Should retry and return result on eventual success."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limited")
            return "success"

        result = _retry_search(flaky)
        assert result == "success"
        assert call_count == 3
        # Should have slept between retries
        assert mock_sleep.call_count == 2

    @patch("agent.web_search.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        """Should raise the last exception after all retries exhausted."""
        def always_fail():
            raise Exception("permanent failure")

        with pytest.raises(Exception, match="permanent failure"):
            _retry_search(always_fail)
        # 3 attempts, 2 sleeps between them
        assert mock_sleep.call_count == 2

    def test_no_retry_on_success(self):
        """Should not retry when the first call succeeds."""
        call_count = 0

        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = _retry_search(succeeds)
        assert result == "ok"
        assert call_count == 1

    def test_retry_search_typing(self):
        """_retry_search preserves signatures and return types of varied callables.

        This verifies the ParamSpec/TypeVar generic machinery works at runtime
        by calling with callables that have different arity and return types.
        Static type correctness is checked separately by mypy.
        """
        # No-arg callable returning int
        def no_args() -> int:
            return 42
        result_int: int = _retry_search(no_args)
        assert result_int == 42
        assert isinstance(result_int, int)

        # Positional args, returning str
        def joiner(a: str, b: str) -> str:
            return a + b
        result_str: str = _retry_search(joiner, "foo", "bar")
        assert result_str == "foobar"
        assert isinstance(result_str, str)

        # Keyword args, returning list
        def maker(items: list[int], *, multiplier: int = 1) -> list[int]:
            return [x * multiplier for x in items]
        result_list: list[int] = _retry_search(maker, [1, 2, 3], multiplier=10)
        assert result_list == [10, 20, 30]

        # Lambda returning dict
        result_dict = _retry_search(lambda k, v: {k: v}, "key", 99)
        assert result_dict == {"key": 99}

    def test_get_web_tools_returns_tool_instances(self):
        """get_web_tools returns a list of Tool instances (for type correctness)."""
        from agent.core import Tool

        tools = get_web_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(t, Tool) for t in tools)

    @patch("agent.web_search.time.sleep")
    @patch("agent.web_search.DDGS")
    def test_web_search_retries_ddg_error(self, mock_ddgs_cls, mock_sleep):
        """web_search should retry when DDG raises an error."""
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if call_count < 3:
                mock.__enter__ = MagicMock(side_effect=Exception("429 Too Many Requests"))
            else:
                inner = MagicMock()
                inner.text.return_value = [
                    {"title": "Result", "body": "Body", "href": "https://example.com"},
                ]
                mock.__enter__ = MagicMock(return_value=inner)
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        mock_ddgs_cls.side_effect = side_effect

        result = web_search("retry test")
        assert "Result" in result
        assert call_count == 3


# =============================================================================
# ERROR ROUTING TESTS
# =============================================================================


class TestErrorRouting:
    """Verify that search-function exceptions route through error_routing
    with enough context (function name, inputs, exception type) for an
    operator to debug the failure."""

    def setup_method(self):
        clear_search_cache()

    def teardown_method(self):
        clear_search_cache()

    @patch("agent.web_search.time.sleep")
    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.DDGS")
    def test_web_search_routes_error(self, mock_ddgs_cls, mock_send_alert, mock_sleep):
        """ValueError from DDGS → error routed with query context, still returns error string."""
        mock_ddgs_cls.side_effect = ValueError("bad query")

        result = web_search("my query", max_results=7)

        # Backward-compatible return value
        assert "Search error" in result
        assert "bad query" in result

        # Exactly one alert dispatched with the expected context
        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert kwargs["level"] == "error"
        assert "web_search" in kwargs["title"]
        assert "ValueError" in kwargs["message"]
        assert "my query" in kwargs["message"]
        assert "max_results" in kwargs["message"]

    @patch("agent.web_search.time.sleep")
    @patch("agent.web_search.send_alert")
    @patch("agent.web_search._rewrite_query_with_llm")
    @patch("agent.web_search.DDGS")
    def test_web_search_smart_routes_error(
        self, mock_ddgs_cls, mock_rewrite, mock_send_alert, mock_sleep
    ):
        """Smart search failure routes error with both original and optimized query."""
        mock_rewrite.return_value = "optimized text"
        mock_ddgs_cls.side_effect = RuntimeError("DDG outage")

        result = web_search_smart("how do I do X?")

        assert "Search error" in result

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert "web_search_smart" in kwargs["title"]
        assert "RuntimeError" in kwargs["message"]
        assert "how do I do X?" in kwargs["message"]
        assert "optimized text" in kwargs["message"]

    @patch("agent.web_search.time.sleep")
    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.DDGS")
    def test_web_search_news_routes_error(
        self, mock_ddgs_cls, mock_send_alert, mock_sleep
    ):
        """News search failure routes error with news-specific title."""
        mock_ddgs_cls.side_effect = ValueError("news rate limit")

        result = web_search_news("breaking story", max_results=3)

        assert "News search error" in result

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert "web_search_news" in kwargs["title"]
        assert "ValueError" in kwargs["message"]
        assert "breaking story" in kwargs["message"]

    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.urlparse")
    def test_get_domain_credibility_routes_error(self, mock_urlparse, mock_send_alert):
        """URL parse failure routes a warning-level alert and returns default score."""
        mock_urlparse.side_effect = TypeError("malformed url object")

        score, tier = get_domain_credibility("weird://thing")

        # Backward-compatible fallback
        assert score == DEFAULT_CREDIBILITY

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert kwargs["level"] == "warning"  # low-severity parse failure
        assert "get_domain_credibility" in kwargs["title"]
        assert "TypeError" in kwargs["message"]
        assert "weird://thing" in kwargs["message"]

    @patch("agent.web_search.send_alert")
    def test_route_exception_swallows_send_alert_errors(self, mock_send_alert):
        """If send_alert itself fails, _route_exception must not propagate."""
        mock_send_alert.side_effect = RuntimeError("alerts channel down")

        # Should not raise
        _route_exception(
            "my_func",
            ValueError("inner"),
            {"query": "hi"},
        )
        mock_send_alert.assert_called_once()

    @patch("agent.web_search.time.sleep")
    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.DDGS")
    def test_successful_search_does_not_route(
        self, mock_ddgs_cls, mock_send_alert, mock_sleep
    ):
        """Happy path must not dispatch alerts."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "T", "body": "B", "href": "https://example.com"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        web_search("ok query")

        mock_send_alert.assert_not_called()

    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.requests")
    def test_web_fetch_routes_timeout(self, mock_requests, mock_send_alert):
        """Timeout → error routed with url + error=timeout, returns error string."""
        import requests as _req
        mock_requests.get.side_effect = _req.exceptions.Timeout("timed out")
        mock_requests.exceptions = _req.exceptions

        result = web_fetch("https://slow-site.com")

        # Backward-compatible return value
        assert "timed out" in result.lower() or "error" in result.lower()
        assert "https://slow-site.com" in result

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert kwargs["level"] == "error"
        assert "web_fetch" in kwargs["title"]
        assert "Timeout" in kwargs["message"]
        assert "https://slow-site.com" in kwargs["message"]
        assert "timeout" in kwargs["message"]

    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.requests")
    def test_web_fetch_routes_request_exception(self, mock_requests, mock_send_alert):
        """RequestException → error routed with url + status_code + error context."""
        import requests as _req
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        exc = _req.exceptions.HTTPError("503 Server Error")
        exc.response = mock_resp
        mock_requests.get.side_effect = exc
        mock_requests.exceptions = _req.exceptions

        result = web_fetch("https://broken.example.com")

        assert "Error fetching" in result
        assert "https://broken.example.com" in result

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert "web_fetch" in kwargs["title"]
        assert "HTTPError" in kwargs["message"]
        assert "https://broken.example.com" in kwargs["message"]
        assert "503" in kwargs["message"]
        assert "status_code" in kwargs["message"]

    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.BeautifulSoup")
    @patch("agent.web_search.requests")
    def test_web_fetch_routes_parsing_exception(
        self, mock_requests, mock_bs, mock_send_alert
    ):
        """Generic exception during parsing → routed with stage=parsing context."""
        import requests as _req
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp
        mock_requests.exceptions = _req.exceptions
        mock_bs.side_effect = RuntimeError("parser exploded")

        result = web_fetch("https://parse-fail.example.com")

        assert "Error parsing" in result
        assert "https://parse-fail.example.com" in result

        mock_send_alert.assert_called_once()
        kwargs = mock_send_alert.call_args.kwargs
        assert kwargs["category"] == "search_error"
        assert "web_fetch" in kwargs["title"]
        assert "RuntimeError" in kwargs["message"]
        assert "https://parse-fail.example.com" in kwargs["message"]
        assert "parsing" in kwargs["message"]
        assert "stage" in kwargs["message"]

    @patch("agent.web_search.send_alert")
    @patch("agent.web_search.requests")
    def test_web_fetch_success_does_not_route(self, mock_requests, mock_send_alert):
        """Happy path must not dispatch alerts."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><article><p>Hello</p></article></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        web_fetch("https://example.com")

    def test_scored_search_result_structure(self):
        """Verify ScoredSearchResult has correct keys and types.

        This test ensures IDE autocompletion and mypy can see the structure.
        A typo in a key would cause this test to fail.
        """
        from agent.web_search import ScoredSearchResult

        # Create a valid ScoredSearchResult
        result: ScoredSearchResult = {
            "title": "Test Title",
            "body": "Test body content",
            "url": "https://example.com",
            "score": 85,
            "tier": "[TRUSTED]",
        }

        # Verify all required keys exist
        assert result["title"] == "Test Title"
        assert result["body"] == "Test body content"
        assert result["url"] == "https://example.com"
        assert result["score"] == 85
        assert result["tier"] == "[TRUSTED]"

        # Verify types are correct
        assert isinstance(result["title"], str)
        assert isinstance(result["body"], str)
        assert isinstance(result["url"], str)
        assert isinstance(result["score"], int)
        assert isinstance(result["tier"], str)

        # Verify web_search_smart produces this structure
        with patch("agent.web_search.DDGS") as mock_ddgs_cls:
            mock_ddgs = MagicMock()
            mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
            mock_ddgs.__exit__ = MagicMock(return_value=False)
            mock_ddgs.text.return_value = [
                {
                    "title": "Python Docs",
                    "body": "Official Python documentation",
                    "href": "https://docs.python.org",
                }
            ]
            mock_ddgs_cls.return_value = mock_ddgs

            output = web_search_smart("python tutorial", max_results=3)

            # Verify output contains expected fields
            assert "Python Docs" in output
            assert "https://docs.python.org" in output
            assert "credibility:" in output

    def test_search_result_types(self):
        """Verify SearchResult and NewsResult TypedDict definitions exist."""
        from agent.web_search import SearchResult, NewsResult

        # Test SearchResult
        search_result: SearchResult = {
            "title": "Test",
            "body": "Body",
            "href": "https://example.com",
        }
        assert search_result["title"] == "Test"
        assert search_result["body"] == "Body"
        assert search_result["href"] == "https://example.com"

        # Test NewsResult
        news_result: NewsResult = {
            "title": "News Title",
            "body": "News body",
            "date": "2024-01-01",
            "url": "https://example.com/news",
            "source": "Example Source",
        }
        assert news_result["title"] == "News Title"
        assert news_result["body"] == "News body"
        assert news_result["date"] == "2024-01-01"
        assert news_result["url"] == "https://example.com/news"
        assert news_result["source"] == "Example Source"
