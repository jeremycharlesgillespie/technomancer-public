"""Tests for the utility_tools module — calculator, Wikipedia, Python exec."""

from unittest.mock import MagicMock, patch

import pytest

from agent.utility_tools import get_utility_tools


class TestCalculate:
    """Test the calculate function."""

    def _calc(self, expr):
        tools = get_utility_tools()
        calc_tool = next(t for t in tools if t.name == "calculate")
        return calc_tool.function(expression=expr)

    def test_basic_addition(self):
        assert "7" in self._calc("3 + 4")

    def test_multiplication(self):
        assert "12" in self._calc("3 * 4")

    def test_division(self):
        result = self._calc("10 / 3")
        assert "3.33" in result or "3.3" in result

    def test_power(self):
        assert "8" in self._calc("2 ** 3")

    def test_complex_expression(self):
        result = self._calc("(100 + 50) * 0.15")
        assert "22.5" in result

    def test_invalid_expression(self):
        result = self._calc("import os")
        assert "error" in result.lower() or "Error" in result

    def test_empty_expression(self):
        result = self._calc("")
        assert isinstance(result, str)


class TestWikipediaSummary:
    """Test the wikipedia_summary function."""

    @patch("requests.get")
    def test_successful_lookup(self, mock_get):
        from agent.utility_tools import wikipedia_summary
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {"pages": {"123": {"title": "Python", "extract": "Python is a programming language."}}}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = wikipedia_summary("Python")
        assert "Python" in result or "programming" in result

    @patch("requests.get")
    def test_no_article_found(self, mock_get):
        from agent.utility_tools import wikipedia_summary
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"query": {"pages": {"-1": {"missing": ""}}}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = wikipedia_summary("xyznonexistent12345")
        assert "no" in result.lower() or "not found" in result.lower() or "No Wikipedia" in result

    @patch("requests.get", side_effect=Exception("Network error"))
    def test_network_error(self, mock_get):
        from agent.utility_tools import wikipedia_summary
        result = wikipedia_summary("test")
        assert "error" in result.lower() or "Error" in result


class TestGetUtilityTools:
    """Test tool registration."""

    def test_returns_tools(self):
        tools = get_utility_tools()
        assert len(tools) >= 1
        names = {t.name for t in tools}
        assert "calculate" in names
