"""Smoke test for the ``patched_git`` and ``patched_jira_reader`` fixtures.

Verifies the fixtures are importable and that — while the fixture is
active — the patched targets are ``MagicMock`` instances rather than
the real implementations. If these assertions fail, AIM/Worker
integration tests have lost their guard rails and could wipe the
feature branch pytest is running from or hit real Jira.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_patched_git_replaces_git_mutating_calls(patched_git):
    import aim.manager
    import aim.worker

    assert isinstance(aim.manager._cleanup_git_and_executions, MagicMock)
    assert isinstance(aim.worker._ensure_git_clean, MagicMock)

    aim.manager._cleanup_git_and_executions()
    aim.worker._ensure_git_clean()

    assert patched_git["cleanup"].call_count == 1
    assert patched_git["ensure_clean"].call_count == 1


def test_patched_jira_reader_blocks_real_http(patched_jira_reader):
    import aim.jira_reader

    assert isinstance(aim.jira_reader._api, MagicMock)
    assert aim.jira_reader._api is patched_jira_reader
