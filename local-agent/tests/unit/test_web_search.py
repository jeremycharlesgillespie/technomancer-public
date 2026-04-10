"""Tests for the web_search module — DuckDuckGo search and URL fetching."""

from unittest.mock import MagicMock, patch

import pytest

from agent.web_search import get_web_tools, web_fetch, web_search, web_search_news


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

    def test_returns_three_tools(self):
        tools = get_web_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "web_search_news" in names
        assert "web_fetch" in names
