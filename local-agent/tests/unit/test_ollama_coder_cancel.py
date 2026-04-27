"""Tests for the cancellable streaming chat in OllamaCoder.

The previous implementation used ``requests.post(timeout=900)`` which
meant a wedged generation could ignore ``state.cancelled`` for up to
15 minutes — long enough that the worker's 30-second post-cancel join
window expired and the worker entered "refusing to stack" mode until
something restarted the process.

The new implementation streams the response and polls cancel between
chunks. These tests pin:

1. Streaming response is reassembled into the same dict shape the
   caller used to receive from a non-streaming POST.
2. Cancel mid-stream returns None and closes the underlying response.
3. Network errors still return None (existing contract).
4. HTTP 500 retry-with-backoff path still works.
5. Cancel during the HTTP 500 retry sleep returns None promptly.
6. Tool-call chunks are accumulated correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_board.ollama_coder import OllamaCoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coder(state=None):
    """Construct a minimal OllamaCoder for testing _chat_with_tools.

    All the heavy collaborators (state, project_root, prompt) are
    irrelevant to the chat layer — we only need ``self.host``,
    ``self.model``, ``self.num_ctx``, and the cancel hook.
    """
    coder = OllamaCoder.__new__(OllamaCoder)
    coder.host = "http://test:11434"
    coder.model = "test-model"
    coder.num_ctx = 16000
    coder.state = state if state is not None else MagicMock(cancelled=False)
    return coder


def _stream_response(lines, status_code=200):
    """Build a fake requests.Response that supports streaming + iter_lines.

    The returned object also acts as its own context manager
    (``__enter__`` / ``__exit__``) so the production code's ``with
    requests.post(...) as r:`` pattern works.
    """
    r = MagicMock()
    r.status_code = status_code
    r.text = ""
    r.iter_lines = MagicMock(return_value=iter(lines))
    r.close = MagicMock()
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    return r


# ===========================================================================
# Happy path — streaming reassembly
# ===========================================================================


class TestStreamingReassembly:
    """Reassembled response should look exactly like the old non-streaming dict."""

    def test_simple_streaming_reassembles_content(self):
        coder = _make_coder()
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Hello "}}),
            json.dumps({"message": {"content": "world"}}),
            json.dumps({
                "message": {},
                "done": True,
                "eval_count": 5,
                "prompt_eval_count": 10,
            }),
        ]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            result = coder._chat_with_tools("sys", [{"role": "user", "content": "hi"}])

        assert result is not None
        msg = result["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert result["done"] is True
        assert result["eval_count"] == 5
        assert result["prompt_eval_count"] == 10

    def test_streaming_collects_tool_calls(self):
        coder = _make_coder()
        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": {"path": "agent/foo.py"},
            },
        }
        lines = [
            json.dumps({"message": {"role": "assistant", "content": ""}}),
            json.dumps({"message": {"tool_calls": [tool_call]}}),
            json.dumps({"message": {}, "done": True}),
        ]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            result = coder._chat_with_tools("sys", [])

        assert result is not None
        msg = result["message"]
        assert msg["tool_calls"] == [tool_call]

    def test_blank_keepalive_lines_are_ignored(self):
        coder = _make_coder()
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "a"}}),
            "",  # heartbeat
            "",
            json.dumps({"message": {"content": "b"}}),
            json.dumps({"message": {}, "done": True}),
        ]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            result = coder._chat_with_tools("sys", [])

        assert result["message"]["content"] == "ab"


# ===========================================================================
# Cancellation
# ===========================================================================


class TestCancellation:
    """The whole point of the refactor — cancel must respond promptly."""

    def test_cancel_before_first_chunk_returns_none(self):
        state = MagicMock(cancelled=True)
        coder = _make_coder(state=state)

        # Even though the response would have content, cancel is checked
        # before the first chunk — so we never yield to the body.
        lines = [
            json.dumps({"message": {"content": "should not appear"}}),
            json.dumps({"message": {}, "done": True}),
        ]
        resp = _stream_response(lines)
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=resp,
        ):
            result = coder._chat_with_tools("sys", [])

        assert result is None
        # The response must be closed when we bail.
        resp.close.assert_called()

    def test_cancel_mid_stream_returns_none_and_closes_response(self):
        """Cancel flag flips after the second chunk — we must abort."""
        state = MagicMock()
        cancel_state = {"cancelled": False}

        def cancel_after_two(*args, **kwargs):
            return cancel_state["cancelled"]

        type(state).cancelled = property(lambda self: cancel_state["cancelled"])
        coder = _make_coder(state=state)

        # The third iter_lines call flips the flag.
        lines_iter = iter([
            json.dumps({"message": {"content": "first"}}),
            json.dumps({"message": {"content": "second"}}),
            json.dumps({"message": {"content": "third"}}),
        ])

        def lines_with_cancel():
            for i, line in enumerate(lines_iter):
                if i == 2:
                    cancel_state["cancelled"] = True
                yield line

        resp = _stream_response([])
        resp.iter_lines = MagicMock(return_value=lines_with_cancel())

        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=resp,
        ):
            result = coder._chat_with_tools("sys", [])

        assert result is None
        resp.close.assert_called()


# ===========================================================================
# Error paths preserved from the old contract
# ===========================================================================


class TestErrorPaths:
    def test_network_error_returns_none(self):
        import requests as _req
        coder = _make_coder()

        with patch(
            "idea_board.ollama_coder.requests.post",
            side_effect=_req.ConnectionError("DNS fail"),
        ):
            result = coder._chat_with_tools("sys", [])

        assert result is None

    def test_http_500_retries_and_eventually_succeeds(self):
        coder = _make_coder()

        # First call: 500. Second call: 200 with valid body.
        good_lines = [
            json.dumps({"message": {"content": "recovered"}}),
            json.dumps({"message": {}, "done": True}),
        ]
        responses = [
            _stream_response([], status_code=500),
            _stream_response(good_lines, status_code=200),
        ]
        responses[0].text = "transient parse bug"

        post_mock = MagicMock(side_effect=responses)
        with patch("idea_board.ollama_coder.requests.post", post_mock), \
             patch("idea_board.ollama_coder.time.sleep"):  # don't actually sleep
            result = coder._chat_with_tools("sys", [])

        assert result is not None
        assert result["message"]["content"] == "recovered"
        assert post_mock.call_count == 2

    def test_http_500_exhausts_retries_returns_none(self):
        coder = _make_coder()
        responses = [
            _stream_response([], status_code=500),
            _stream_response([], status_code=500),
            _stream_response([], status_code=500),
            _stream_response([], status_code=500),
        ]
        for r in responses:
            r.text = "boom"

        post_mock = MagicMock(side_effect=responses)
        with patch("idea_board.ollama_coder.requests.post", post_mock), \
             patch("idea_board.ollama_coder.time.sleep"):
            result = coder._chat_with_tools("sys", [])

        assert result is None
        assert post_mock.call_count == 4  # 1 initial + 3 retries

    def test_http_404_returns_none_no_retry(self):
        coder = _make_coder()
        resp = _stream_response([], status_code=404)
        resp.text = "not found"

        post_mock = MagicMock(return_value=resp)
        with patch("idea_board.ollama_coder.requests.post", post_mock):
            result = coder._chat_with_tools("sys", [])

        assert result is None
        assert post_mock.call_count == 1  # no retries for non-500

    def test_non_json_stream_line_returns_none(self):
        coder = _make_coder()
        lines = [
            "this is not json",
            json.dumps({"message": {}, "done": True}),
        ]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            result = coder._chat_with_tools("sys", [])

        assert result is None

    def test_cancel_during_http_500_retry_sleep(self):
        """If cancel fires during the retry backoff, we must bail."""
        cancel_state = {"cancelled": False}

        state = MagicMock()
        type(state).cancelled = property(lambda self: cancel_state["cancelled"])
        coder = _make_coder(state=state)

        responses = [_stream_response([], status_code=500)]
        responses[0].text = "transient"
        post_mock = MagicMock(side_effect=responses)

        # When sleep is called, flip the cancel flag.
        def fake_sleep(s):
            cancel_state["cancelled"] = True

        with patch("idea_board.ollama_coder.requests.post", post_mock), \
             patch("idea_board.ollama_coder.time.sleep", side_effect=fake_sleep):
            result = coder._chat_with_tools("sys", [])

        assert result is None
        # First request fired, then we hit the cancel during backoff
        # and didn't fire a second.
        assert post_mock.call_count == 1


# ===========================================================================
# Sanity: the request body must still ask for streaming
# ===========================================================================


class TestRequestShape:
    def test_request_body_has_stream_true(self):
        coder = _make_coder()
        lines = [json.dumps({"message": {}, "done": True})]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ) as mock_post:
            coder._chat_with_tools("sys", [{"role": "user", "content": "hi"}])

        assert mock_post.call_count == 1
        body = mock_post.call_args.kwargs["json"]
        assert body["stream"] is True
        # Existing important defaults stay set.
        assert body["keep_alive"] == -1
        assert body["model"] == "test-model"
        assert body["options"]["num_ctx"] == 16000

    def test_request_uses_streaming_kwarg(self):
        coder = _make_coder()
        lines = [json.dumps({"message": {}, "done": True})]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ) as mock_post:
            coder._chat_with_tools("sys", [])

        # ``stream=True`` so requests doesn't buffer the entire response.
        assert mock_post.call_args.kwargs.get("stream") is True


# ===========================================================================
# Diagnostic logging — context for the qwen3 timeout investigation
# ===========================================================================


class TestDiagnosticLogging:
    """Verify the timeout-investigation logs fire with the expected fields.

    The qwen3-coder model on a 30b parameter checkpoint can spend
    20-40s on prompt-eval before the first token arrives. Our
    ``CHAT_CHUNK_TIMEOUT`` is 30s — close enough to the prompt-eval
    ceiling that healthy generations sometimes time out. These tests
    pin the diagnostic fields we rely on to decide whether the fix is
    a longer first-chunk timeout, a retry, or a smaller prompt.
    """

    def test_request_log_includes_prompt_size_and_model(self, caplog):
        coder = _make_coder()
        lines = [json.dumps({"message": {}, "done": True})]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            with caplog.at_level("INFO", logger="idea_board.ollama_coder"):
                coder._chat_with_tools(
                    "system prompt text",
                    [{"role": "user", "content": "user message body"}],
                )

        request_logs = [r for r in caplog.records if "chat request:" in r.message]
        assert len(request_logs) == 1, "Expected one chat-request log per call"
        msg = request_logs[0].message
        assert "model=test-model" in msg
        assert "host=http://test:11434" in msg
        # System ("system prompt text" = 18 chars) + user ("user message body" = 17)
        # = 35 chars total. Just check it's >= 35 to keep the test resilient.
        assert "prompt_chars=" in msg
        # Pull out the integer and verify it captured both messages.
        for token in msg.split():
            if token.startswith("prompt_chars="):
                value = int(token.split("=", 1)[1])
                assert value >= 35, f"prompt_chars too small: {value}"

    def test_success_log_includes_first_chunk_latency(self, caplog):
        coder = _make_coder()
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "hi"}}),
            json.dumps({
                "message": {},
                "done": True,
                "eval_count": 7,
                "prompt_eval_count": 42,
            }),
        ]
        with patch(
            "idea_board.ollama_coder.requests.post",
            return_value=_stream_response(lines),
        ):
            with caplog.at_level("INFO", logger="idea_board.ollama_coder"):
                coder._chat_with_tools("sys", [{"role": "user", "content": "x"}])

        # Both the first-chunk log and the success log should fire.
        first_chunk_logs = [r for r in caplog.records if "first chunk arrived" in r.message]
        done_logs = [r for r in caplog.records if "chat done:" in r.message]
        assert len(first_chunk_logs) == 1
        assert len(done_logs) == 1
        assert "prompt_eval_count=42" in done_logs[0].message
        assert "eval_count=7" in done_logs[0].message
        assert "first_chunk_latency=" in done_logs[0].message

    def test_network_error_log_includes_elapsed_and_prompt_size(self, caplog):
        import requests as real_requests
        coder = _make_coder()
        with patch(
            "idea_board.ollama_coder.requests.post",
            side_effect=real_requests.ConnectionError("connection refused"),
        ):
            with caplog.at_level("WARNING", logger="idea_board.ollama_coder"):
                result = coder._chat_with_tools(
                    "system text",
                    [{"role": "user", "content": "user text"}],
                )

        assert result is None
        net_logs = [r for r in caplog.records if "Network error after" in r.message]
        assert len(net_logs) == 1
        msg = net_logs[0].message
        assert "model=test-model" in msg
        assert "prompt_chars=" in msg
        assert "attempt=" in msg

    def test_stream_read_error_log_distinguishes_first_vs_subsequent_chunk(self, caplog):
        """Mid-stream timeout BEFORE first chunk should be flagged ``stalled_on=first_chunk``.

        This is the diagnostic that tells us whether the qwen3 timeout
        is happening during prompt-eval (first chunk) or during
        generation (subsequent chunks). Two completely different fixes.
        """
        import requests as real_requests
        coder = _make_coder()

        def _raising_iter_lines(decode_unicode=True):
            raise real_requests.exceptions.ReadTimeout("read timeout=30")

        r = MagicMock()
        r.status_code = 200
        r.text = ""
        r.iter_lines = _raising_iter_lines
        r.close = MagicMock()
        r.__enter__ = MagicMock(return_value=r)
        r.__exit__ = MagicMock(return_value=False)

        with patch("idea_board.ollama_coder.requests.post", return_value=r):
            with caplog.at_level("WARNING", logger="idea_board.ollama_coder"):
                coder._chat_with_tools("sys", [])

        stall_logs = [r for r in caplog.records if "stream read error after" in r.message]
        assert len(stall_logs) == 1
        msg = stall_logs[0].message
        assert "stalled_on=first_chunk" in msg, f"unexpected log: {msg}"
        assert "first_chunk_latency=never" in msg
        assert "chunks_received=0" in msg
