"""Tests for aiv.verifiers.api_call — JSON endpoint capture helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from aiv.verifiers.api_call import SUPPORTED_METHODS, capture


def _mock_response(
    status: int = 200,
    json_body: object | None = None,
    content: bytes = b"",
) -> MagicMock:
    """Build a mock Response.

    When ``json_body`` is provided, ``.json()`` returns it. When it is
    ``None``, ``.json()`` raises ``ValueError`` to simulate a non-JSON
    body, and ``.content`` returns the supplied raw bytes.
    """
    resp = MagicMock()
    resp.status_code = status
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("not json")
    resp.content = content
    return resp


class TestCaptureGet:
    def test_returns_status_response_and_elapsed(self):
        resp = _mock_response(200, {"key": "TK-42", "title": "ok"})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            result = capture("GET", "/api/state")

        mock_get.assert_called_once()
        assert result["status"] == 200
        assert result["response"] == {"key": "TK-42", "title": "ok"}
        assert isinstance(result["elapsed_ms"], int)
        assert result["elapsed_ms"] >= 0
        assert "error" not in result
        assert "response_raw" not in result

    def test_default_base_url_joined_with_path(self):
        resp = _mock_response(200, {})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            capture("GET", "/api/state")

        url = mock_get.call_args[0][0]
        assert url == "http://localhost:8322/api/state"

    def test_custom_base_url(self):
        resp = _mock_response(200, {})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            capture("GET", "/api/state", base_url="http://example.com:9000")

        url = mock_get.call_args[0][0]
        assert url == "http://example.com:9000/api/state"

    def test_base_url_with_trailing_slash_and_path_without_leading_slash(self):
        resp = _mock_response(200, {})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            capture("GET", "api/state", base_url="http://localhost:8322/")

        url = mock_get.call_args[0][0]
        assert url == "http://localhost:8322/api/state"

    def test_lowercase_method_accepted(self):
        resp = _mock_response(200, {"ok": True})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            result = capture("get", "/api/state")

        mock_get.assert_called_once()
        assert result["response"] == {"ok": True}

    def test_get_body_is_ignored(self):
        resp = _mock_response(200, {})
        with patch(
            "aiv.verifiers.api_call.requests.get", return_value=resp
        ) as mock_get:
            capture("GET", "/api/state", body={"should": "be ignored"})

        _, kwargs = mock_get.call_args
        assert "json" not in kwargs
        assert "data" not in kwargs


class TestCapturePost:
    def test_post_sends_json_body(self):
        resp = _mock_response(201, {"key": "TK-999"})
        body = {"title": "New story", "description": "..."}
        with patch(
            "aiv.verifiers.api_call.requests.post", return_value=resp
        ) as mock_post:
            result = capture("POST", "/api/jira/create", body=body)

        _, kwargs = mock_post.call_args
        assert kwargs["json"] == body
        assert result["status"] == 201
        assert result["response"] == {"key": "TK-999"}

    def test_post_without_body_sends_none(self):
        resp = _mock_response(200, {})
        with patch(
            "aiv.verifiers.api_call.requests.post", return_value=resp
        ) as mock_post:
            capture("POST", "/api/trigger")

        _, kwargs = mock_post.call_args
        assert kwargs["json"] is None


class TestCaptureStatusCodes:
    def test_404_still_captures_body(self):
        resp = _mock_response(404, {"error": "not found"})
        with patch("aiv.verifiers.api_call.requests.get", return_value=resp):
            result = capture("GET", "/api/missing")

        assert result["status"] == 404
        assert result["response"] == {"error": "not found"}
        assert "error" not in result

    def test_500_still_captures_body(self):
        resp = _mock_response(500, {"error": "boom"})
        with patch("aiv.verifiers.api_call.requests.get", return_value=resp):
            result = capture("GET", "/api/broken")

        assert result["status"] == 500
        assert result["response"] == {"error": "boom"}


class TestCaptureMalformedJson:
    def test_non_json_body_falls_back_to_response_raw(self):
        raw = b"<html>boom</html>"
        resp = _mock_response(200, json_body=None, content=raw)
        with patch("aiv.verifiers.api_call.requests.get", return_value=resp):
            result = capture("GET", "/api/state")

        assert result["status"] == 200
        assert result["response_raw"] == raw
        assert "response" not in result

    def test_non_json_error_page_still_captures_status(self):
        raw = b"<html>internal error</html>"
        resp = _mock_response(500, json_body=None, content=raw)
        with patch("aiv.verifiers.api_call.requests.get", return_value=resp):
            result = capture("GET", "/api/broken")

        assert result["status"] == 500
        assert result["response_raw"] == raw


class TestCaptureUnsupportedMethod:
    def test_put_returns_error(self):
        result = capture("PUT", "/api/state")
        assert "error" in result
        assert "PUT" in result["error"]
        assert "status" not in result
        assert "response" not in result

    def test_delete_returns_error(self):
        result = capture("DELETE", "/api/state")
        assert "error" in result
        assert "DELETE" in result["error"]

    def test_supported_methods_is_exactly_get_and_post(self):
        assert SUPPORTED_METHODS == {"GET", "POST"}


class TestCaptureNetworkErrors:
    def test_connection_error_returns_error_field(self):
        with patch(
            "aiv.verifiers.api_call.requests.get",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            result = capture("GET", "/api/state")

        assert "error" in result
        assert "connection refused" in result["error"]
        assert "status" not in result
        assert "response" not in result

    def test_timeout_returns_error_field(self):
        with patch(
            "aiv.verifiers.api_call.requests.get",
            side_effect=requests.Timeout("read timed out"),
        ):
            result = capture("GET", "/api/state")

        assert "error" in result
        assert "timed out" in result["error"]

    def test_post_network_error_returns_error_field(self):
        with patch(
            "aiv.verifiers.api_call.requests.post",
            side_effect=requests.RequestException("boom"),
        ):
            result = capture("POST", "/api/create", body={"x": 1})

        assert result == {"error": "boom"}
