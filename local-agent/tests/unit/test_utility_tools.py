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


class TestRunPython:
    """Test safe Python execution."""

    def test_basic_print(self):
        from agent.utility_tools import run_python

        result = run_python("print(2 + 2)")
        assert "4" in result

    def test_math_operations(self):
        from agent.utility_tools import run_python

        result = run_python("print(math.sqrt(16))")
        assert "4" in result

    def test_rejects_os_import(self):
        from agent.utility_tools import run_python

        result = run_python("import os\nos.listdir('.')")
        assert "Error" in result or "disallowed" in result

    def test_rejects_subprocess(self):
        from agent.utility_tools import run_python

        result = run_python("import subprocess\nsubprocess.run(['ls'])")
        assert "Error" in result or "disallowed" in result

    def test_rejects_open(self):
        from agent.utility_tools import run_python

        result = run_python("f = open('/etc/passwd')\nprint(f.read())")
        assert "Error" in result or "disallowed" in result

    def test_no_output(self):
        from agent.utility_tools import run_python

        result = run_python("x = 42")
        assert "no output" in result.lower() or "executed" in result.lower()

    def test_json_usage(self):
        from agent.utility_tools import run_python

        result = run_python("print(json.dumps({'a': 1}))")
        assert '"a"' in result

    def test_exception_returns_error(self):
        from agent.utility_tools import run_python

        result = run_python("print(1/0)")
        assert "Error" in result or "ZeroDivision" in result


class TestCalculateInjection:
    """Test that calculate rejects injection attempts."""

    def test_rejects_import(self):
        from agent.utility_tools import calculate

        result = calculate("__import__('os').system('ls')")
        assert "disallowed" in result.lower() or "Error" in result

    def test_rejects_exec(self):
        from agent.utility_tools import calculate

        result = calculate("exec('print(1)')")
        assert "disallowed" in result.lower() or "Error" in result

    def test_rejects_dunder(self):
        from agent.utility_tools import calculate

        result = calculate("__builtins__")
        assert "disallowed" in result.lower() or "Error" in result

    def test_sqrt(self):
        from agent.utility_tools import calculate

        result = calculate("sqrt(144)")
        assert "12" in result

    def test_trig(self):
        from agent.utility_tools import calculate

        result = calculate("sin(0)")
        assert "0" in result


class TestWikipediaEdgeCases:
    """Additional Wikipedia tests."""

    @patch("requests.get")
    def test_timeout(self, mock_get):
        import requests as req
        from agent.utility_tools import wikipedia_summary

        mock_get.side_effect = req.Timeout("timed out")
        result = wikipedia_summary("test")
        assert "timed out" in result.lower()

    @patch("requests.get")
    def test_404_with_search_fallback(self, mock_get):
        from agent.utility_tools import wikipedia_summary

        # First call returns 404, second (search) returns a result, third gets the article
        resp_404 = MagicMock()
        resp_404.status_code = 404

        resp_search = MagicMock()
        resp_search.json.return_value = {"query": {"search": [{"title": "Python (programming language)"}]}}

        resp_article = MagicMock()
        resp_article.status_code = 200
        resp_article.json.return_value = {
            "title": "Python",
            "extract": "Python is a programming language.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
        }

        mock_get.side_effect = [resp_404, resp_search, resp_article]
        result = wikipedia_summary("python programming")
        assert "Python" in result


class TestGetUtilityTools:
    """Test tool registration."""

    def test_returns_tools(self):
        tools = get_utility_tools()
        assert len(tools) >= 1
        names = {t.name for t in tools}
        assert "calculate" in names

    def test_all_tools_present(self):
        tools = get_utility_tools()
        names = {t.name for t in tools}
        assert "calculate" in names
        assert "wikipedia" in names
        assert "run_python" in names
