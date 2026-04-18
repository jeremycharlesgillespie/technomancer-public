"""Tests for aiv.verifiers.web_render — HTTP capture helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from aiv.verifiers.web_render import MAX_HTML_CHARS, capture


def _mock_response(status: int = 200, text: str = "<html>ok</html>") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


class TestCaptureSuccess:
    def test_returns_status_html_and_elapsed(self):
        resp = _mock_response(200, "<html><body>hello</body></html>")
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp) as mock_get:
            result = capture("/quality")

        mock_get.assert_called_once()
        assert result["status"] == 200
        assert result["html"] == "<html><body>hello</body></html>"
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], int)
        assert result["elapsed_ms"] >= 0
        assert "error" not in result

    def test_default_base_url_joined_with_route(self):
        resp = _mock_response()
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp) as mock_get:
            capture("/quality")

        url = mock_get.call_args[0][0]
        assert url == "http://localhost:8322/quality"

    def test_custom_base_url(self):
        resp = _mock_response()
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp) as mock_get:
            capture("/metrics", base_url="http://example.com:9000")

        url = mock_get.call_args[0][0]
        assert url == "http://example.com:9000/metrics"

    def test_base_url_with_trailing_slash(self):
        resp = _mock_response()
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp) as mock_get:
            capture("/quality", base_url="http://localhost:8322/")

        url = mock_get.call_args[0][0]
        assert url == "http://localhost:8322/quality"

    def test_route_without_leading_slash(self):
        resp = _mock_response()
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp) as mock_get:
            capture("quality")

        url = mock_get.call_args[0][0]
        assert url == "http://localhost:8322/quality"

    def test_non_200_status_still_returns_capture(self):
        resp = _mock_response(404, "<html>not found</html>")
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp):
            result = capture("/nope")

        assert result["status"] == 404
        assert result["html"] == "<html>not found</html>"
        assert "error" not in result


class TestCaptureTruncation:
    def test_html_over_8k_is_truncated(self):
        big_html = "x" * (MAX_HTML_CHARS + 5_000)
        resp = _mock_response(200, big_html)
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp):
            result = capture("/quality")

        assert len(result["html"]) == MAX_HTML_CHARS
        assert result["html"] == "x" * MAX_HTML_CHARS

    def test_html_exactly_8k_not_truncated(self):
        exact_html = "y" * MAX_HTML_CHARS
        resp = _mock_response(200, exact_html)
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp):
            result = capture("/quality")

        assert len(result["html"]) == MAX_HTML_CHARS
        assert result["html"] == exact_html

    def test_small_html_unchanged(self):
        small = "<p>small</p>"
        resp = _mock_response(200, small)
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp):
            result = capture("/quality")

        assert result["html"] == small

    def test_empty_body_becomes_empty_string(self):
        resp = _mock_response(204, "")
        with patch("aiv.verifiers.web_render.requests.get", return_value=resp):
            result = capture("/quality")

        assert result["status"] == 204
        assert result["html"] == ""


class TestCaptureError:
    def test_connection_error_returns_error_field(self):
        err = requests.ConnectionError("connection refused")
        with patch("aiv.verifiers.web_render.requests.get", side_effect=err):
            result = capture("/quality")

        assert "error" in result
        assert "connection refused" in result["error"]
        assert "status" not in result
        assert "html" not in result

    def test_timeout_returns_error_field(self):
        with patch(
            "aiv.verifiers.web_render.requests.get",
            side_effect=requests.Timeout("read timed out"),
        ):
            result = capture("/quality")

        assert "error" in result
        assert "timed out" in result["error"]

    def test_generic_request_exception_returns_error(self):
        with patch(
            "aiv.verifiers.web_render.requests.get",
            side_effect=requests.RequestException("boom"),
        ):
            result = capture("/quality")

        assert result == {"error": "boom"}
