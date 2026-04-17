"""Tests for agent.jira_retry — timeout-bounded, retrying Jira request wrapper.

Covers the two acceptance scenarios from TK-558:
  1. 2x 503 followed by 200 → single success with 2 retries logged, no row
     written to ``jira_sync_failures``.
  2. 3x timeout → final WARNING logged, one row inserted into
     ``jira_sync_failures``, no exception propagated.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent import jira_retry
from agent.jira_retry import (
    BACKOFF_SECONDS,
    DEFAULT_TIMEOUT,
    MAX_ATTEMPTS,
    RETRYABLE_STATUS,
    _backoff,
    _init_db,
    get_failures,
    jira_request,
    record_failure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_failures_db(tmp_path, monkeypatch):
    """Redirect the ``jira_sync_failures`` DB at a temp path for each test."""
    db_path = tmp_path / "jira_sync_failures.db"
    monkeypatch.setattr(jira_retry, "DB_DIR", tmp_path)
    monkeypatch.setattr(jira_retry, "DB_PATH", db_path)
    conn = getattr(jira_retry._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    jira_retry._local.__dict__.pop("conn", None)
    yield
    conn = getattr(jira_retry._local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        jira_retry._local.__dict__.pop("conn", None)


def _resp(status: int, text: str = "", headers: dict | None = None):
    """Build a MagicMock imitating ``requests.Response``."""
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = text
    r.headers = headers or {}
    return r


def _fake_sleep():
    """Return a sleep callable that records delays without blocking."""
    calls: list[float] = []

    def _fn(delay: float) -> None:
        calls.append(delay)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_timeout_is_connect_read_tuple(self):
        assert DEFAULT_TIMEOUT == (5.0, 15.0)

    def test_max_attempts_is_three(self):
        assert MAX_ATTEMPTS == 3

    def test_backoff_is_one_three_nine(self):
        assert BACKOFF_SECONDS == (1.0, 3.0, 9.0)

    def test_retryable_status_is_5xx_set(self):
        assert RETRYABLE_STATUS == frozenset({500, 502, 503, 504})


# ---------------------------------------------------------------------------
# _backoff
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_first_attempt_returns_first_entry(self):
        assert _backoff(1) == 1.0

    def test_second_attempt_returns_second_entry(self):
        assert _backoff(2) == 3.0

    def test_third_attempt_returns_third_entry(self):
        assert _backoff(3) == 9.0

    def test_attempts_beyond_list_clamp_to_last(self):
        assert _backoff(99) == BACKOFF_SECONDS[-1]

    def test_attempt_zero_clamps_to_first(self):
        assert _backoff(0) == BACKOFF_SECONDS[0]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


class TestFailuresDb:
    def test_init_db_creates_table_idempotently(self):
        _init_db()
        _init_db()  # second call must not raise
        conn = jira_retry._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='jira_sync_failures'"
        ).fetchone()
        assert row is not None

    def test_record_failure_inserts_row(self):
        rowid = record_failure("idea-001", "TK-42", "boom")
        assert rowid > 0

        rows = get_failures()
        assert len(rows) == 1
        assert rows[0]["idea_id"] == "idea-001"
        assert rows[0]["jira_key"] == "TK-42"
        assert rows[0]["error"] == "boom"
        assert rows[0]["attempted_at"]  # non-empty

    def test_record_failure_truncates_long_error(self):
        record_failure("idea-X", None, "x" * 5000)
        rows = get_failures()
        assert len(rows[0]["error"]) == 2000

    def test_record_failure_allows_null_ids(self):
        rowid = record_failure(None, None, "no ids")
        assert rowid > 0
        rows = get_failures()
        assert rows[0]["idea_id"] is None
        assert rows[0]["jira_key"] is None


# ---------------------------------------------------------------------------
# jira_request — success & retry paths
# ---------------------------------------------------------------------------


class TestJiraRequestSuccess:
    @patch("agent.jira_retry.requests")
    def test_success_on_first_try_returns_response(self, mock_requests):
        mock_requests.get.return_value = _resp(200)
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        sleep = _fake_sleep()
        resp = jira_request(
            "get", "https://example/issue/TK-1",
            idea_id="idea-1", sleep=sleep,
        )

        assert resp is not None
        assert resp.status_code == 200
        assert sleep.calls == []  # no retries
        assert mock_requests.get.call_count == 1
        # Failures table should be empty
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_default_timeout_is_five_fifteen(self, mock_requests):
        mock_requests.post.return_value = _resp(201)
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        jira_request("post", "https://example/issue", json={"k": "v"})
        kwargs = mock_requests.post.call_args[1]
        assert kwargs["timeout"] == (5.0, 15.0)

    @patch("agent.jira_retry.requests")
    def test_caller_can_override_timeout(self, mock_requests):
        mock_requests.get.return_value = _resp(200)
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        jira_request("get", "https://example/issue/TK-1", timeout=(1.0, 2.0))
        kwargs = mock_requests.get.call_args[1]
        assert kwargs["timeout"] == (1.0, 2.0)


class TestJiraRequestRetries:
    @patch("agent.jira_retry.requests")
    def test_two_503s_then_200_single_successful_response(self, mock_requests, caplog):
        """Acceptance: 2x 503 then 200 → success with 2 retries logged."""
        mock_requests.get.side_effect = [
            _resp(503, "svc down"),
            _resp(503, "still down"),
            _resp(200, "ok"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        sleep = _fake_sleep()
        with caplog.at_level("WARNING", logger="agent.jira_retry"):
            resp = jira_request(
                "get", "https://example/issue/TK-1",
                idea_id="idea-1", jira_key="TK-1", sleep=sleep,
            )

        assert resp is not None
        assert resp.status_code == 200
        assert mock_requests.get.call_count == 3
        # Two retry warnings, one per 503 before the eventual success
        retry_warnings = [
            r for r in caplog.records
            if "retry in" in r.getMessage()
        ]
        assert len(retry_warnings) == 2
        # No "failed after ... attempts" final warning
        assert not any(
            "failed after" in r.getMessage() for r in caplog.records
        )
        # Backoff sequence honoured between 1st→2nd and 2nd→3rd attempts
        assert sleep.calls == [BACKOFF_SECONDS[0], BACKOFF_SECONDS[1]]
        # Failures table should be empty
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_three_timeouts_logs_warning_records_row_no_exception(
        self, mock_requests, caplog,
    ):
        """Acceptance: 3x timeout → WARNING + failure row + no exception."""
        mock_requests.get.side_effect = [
            requests.ReadTimeout("read timed out"),
            requests.ReadTimeout("read timed out"),
            requests.ReadTimeout("read timed out"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        sleep = _fake_sleep()
        with caplog.at_level("WARNING", logger="agent.jira_retry"):
            resp = jira_request(
                "get", "https://example/issue/TK-1",
                idea_id="idea-42", jira_key="TK-1", sleep=sleep,
            )

        assert resp is None
        assert mock_requests.get.call_count == MAX_ATTEMPTS
        # Final WARNING includes idea_id and jira_key
        final = [
            r for r in caplog.records
            if "failed after" in r.getMessage()
        ]
        assert len(final) == 1
        msg = final[0].getMessage()
        assert "idea-42" in msg
        assert "TK-1" in msg
        # Exactly one row inserted
        rows = get_failures()
        assert len(rows) == 1
        assert rows[0]["idea_id"] == "idea-42"
        assert rows[0]["jira_key"] == "TK-1"
        assert "ReadTimeout" in rows[0]["error"]
        # Backoff only between attempts (no sleep after the final one)
        assert sleep.calls == [BACKOFF_SECONDS[0], BACKOFF_SECONDS[1]]

    @patch("agent.jira_retry.requests")
    def test_three_connection_errors_records_failure(
        self, mock_requests, caplog,
    ):
        mock_requests.get.side_effect = [
            requests.ConnectionError("reset"),
            requests.ConnectionError("reset"),
            requests.ConnectionError("reset"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        resp = jira_request(
            "get", "https://example/issue/TK-1",
            idea_id="idea-7", sleep=_fake_sleep(),
        )

        assert resp is None
        rows = get_failures()
        assert len(rows) == 1
        assert "ConnectionError" in rows[0]["error"]

    @patch("agent.jira_retry.requests")
    def test_connection_error_then_success_no_failure_row(self, mock_requests):
        mock_requests.get.side_effect = [
            requests.ConnectionError("reset"),
            _resp(200, "ok"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        sleep = _fake_sleep()
        resp = jira_request(
            "get", "https://example/issue/TK-1",
            idea_id="idea-9", sleep=sleep,
        )

        assert resp is not None
        assert resp.status_code == 200
        assert sleep.calls == [BACKOFF_SECONDS[0]]
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_three_500s_records_failure(self, mock_requests):
        mock_requests.get.side_effect = [
            _resp(500, "oops"),
            _resp(502, "gateway"),
            _resp(504, "timeout"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        resp = jira_request(
            "get", "https://example/issue/TK-1",
            idea_id="idea-88", jira_key="TK-1", sleep=_fake_sleep(),
        )

        assert resp is None
        rows = get_failures()
        assert len(rows) == 1
        # Last error should reflect the final 504
        assert "504" in rows[0]["error"]


class TestJiraRequestNonRetryable:
    @patch("agent.jira_retry.requests")
    def test_401_returned_without_retry(self, mock_requests):
        mock_requests.get.return_value = _resp(401, "unauthorized")
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        sleep = _fake_sleep()
        resp = jira_request(
            "get", "https://example/issue/TK-1", sleep=sleep,
        )

        assert resp is not None
        assert resp.status_code == 401
        assert mock_requests.get.call_count == 1
        assert sleep.calls == []
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_429_returned_without_retry(self, mock_requests):
        """429 is a rate limit, not in RETRYABLE_STATUS — caller handles it."""
        mock_requests.get.return_value = _resp(429, "rate limited")
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        resp = jira_request("get", "https://example/issue/TK-1")
        assert resp is not None
        assert resp.status_code == 429
        assert mock_requests.get.call_count == 1
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_unknown_exception_propagates(self, mock_requests):
        """Errors outside ConnectionError/ReadTimeout still raise upstream."""
        mock_requests.get.side_effect = ValueError("bad url")
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        with pytest.raises(ValueError):
            jira_request("get", "https://example/issue/TK-1")
        assert get_failures() == []

    @patch("agent.jira_retry.requests")
    def test_custom_max_attempts(self, mock_requests):
        """Caller can shorten the retry loop."""
        mock_requests.get.side_effect = [
            _resp(503),
            _resp(200, "ok"),
        ]
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.ReadTimeout = requests.ReadTimeout

        resp = jira_request(
            "get", "https://example/issue/TK-1",
            max_attempts=2, sleep=_fake_sleep(),
        )
        assert resp is not None
        assert resp.status_code == 200
        assert mock_requests.get.call_count == 2
