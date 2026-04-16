"""Shared fixtures for integration tests exercising AIM/Worker flows.

These fixtures are opt-in (not autouse): tests declare ``patched_git`` or
``patched_jira_reader`` in their signature to isolate themselves from the
real git working tree and real Jira HTTP calls.

See also ``feedback_aim_tests_wipe_branches`` in auto-memory — running
AIM/Worker code through pytest without mocking these surfaces has, in
the past, wiped the feature branch pytest was running from (via
``git checkout --force main``) and issued live Jira writes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def patched_git():
    """Mock the git-mutating helpers inside AIM manager and worker.

    Yields a dict with two ``MagicMock`` handles so tests can assert
    call counts or arguments:

        def test_something(patched_git):
            run_failure_path()
            assert patched_git["cleanup"].called

    Patched targets:
      - ``aim.manager._cleanup_git_and_executions`` — runs real
        ``git checkout --force main`` on worker failure.
      - ``aim.worker._ensure_git_clean`` — same reset, triggered
        before each execution attempt.
    """
    with patch("aim.manager._cleanup_git_and_executions") as cleanup, \
         patch("aim.worker._ensure_git_clean") as ensure_clean:
        yield {"cleanup": cleanup, "ensure_clean": ensure_clean}


@pytest.fixture
def patched_jira_reader():
    """Mock the HTTP entry point used by ``aim.jira_reader``.

    ``aim.jira_reader`` calls ``_api`` (re-exported from
    ``idea_board.jira_sync``) to reach the real Jira REST API. Patching
    ``aim.jira_reader._api`` short-circuits every read path
    (``count_issues_by_status``, ``list_todo_issues``,
    ``get_recent_completions``) without needing each test to stub them
    individually.

    The default return value is an empty ``{"issues": [], "isLast": True}``
    response wrapped in a 200 status — which means reader functions see
    "no issues" rather than a failure. Tests that want specific data
    should reassign ``mock.return_value`` or ``mock.side_effect``.
    """
    default_response = MagicMock()
    default_response.status_code = 200
    default_response.json.return_value = {"issues": [], "isLast": True}

    with patch("aim.jira_reader._api", return_value=default_response) as api_mock:
        yield api_mock
