"""Tests for project_tracker.py — SQLite-backed project tracking.

Tests CRUD operations (add, remove, list, detail, blocker, update)
and Discord command handlers.
"""

import asyncio
import sqlite3
import threading
from unittest.mock import patch

import pytest

from agent.project_tracker import (
    DB_PATH,
    _get_conn,
    _local,
    add_blocker,
    add_project,
    get_all_projects_raw,
    get_project,
    get_project_health_summary,
    get_project_tracker_tools,
    init_db,
    list_projects,
    remove_project,
    update_project,
)
from agent.bot_commands import (
    handle_blocker,
    handle_project_detail,
    handle_projects,
    handle_track,
    handle_untrack,
)
from tests.conftest import MockDiscordMessage


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_msg(content="test", user="TestUser"):
    return MockDiscordMessage(content=content, author_name=user, channel_name="llm_chat")


# ---------------------------------------------------------------------------
# Fixture: isolated in-memory DB per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Redirect project_tracker to a temporary SQLite DB for each test."""
    db_path = tmp_path / "projects.db"
    monkeypatch.setattr("agent.project_tracker.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.project_tracker.DB_PATH", db_path)
    # Clear the thread-local connection so a fresh one is created
    if hasattr(_local, "conn"):
        try:
            _local.conn.close()
        except Exception:
            pass
        del _local.conn
    init_db()
    yield
    # Clean up thread-local connection
    if hasattr(_local, "conn"):
        try:
            _local.conn.close()
        except Exception:
            pass
        del _local.conn


# ================================================================
# CRUD unit tests
# ================================================================

class TestAddProject:
    def test_add_new_project(self):
        result = add_project("myapp", "https://github.com/user/myapp")
        assert "myapp" in result
        assert "tracked" in result.lower()

    def test_add_project_without_url(self):
        result = add_project("simple")
        assert "simple" in result
        assert "tracked" in result.lower()

    def test_add_duplicate_project(self):
        add_project("dup")
        result = add_project("dup")
        assert "already" in result.lower()

    def test_add_project_with_notes(self):
        add_project("noted", notes="Work in progress")
        detail = get_project("noted")
        assert "Work in progress" in detail

    def test_add_project_strips_whitespace(self):
        add_project("  padded  ", "  https://example.com  ")
        detail = get_project("padded")
        assert "padded" in detail
        assert "https://example.com" in detail


class TestRemoveProject:
    def test_remove_existing(self):
        add_project("removeme")
        result = remove_project("removeme")
        assert "removed" in result.lower()

    def test_remove_nonexistent(self):
        result = remove_project("ghost")
        assert "no project" in result.lower()

    def test_remove_actually_deletes(self):
        add_project("gone")
        remove_project("gone")
        result = get_project("gone")
        assert "no project" in result.lower()


class TestListProjects:
    def test_empty_list(self):
        result = list_projects()
        assert "no projects" in result.lower()

    def test_list_single(self):
        add_project("alpha", "https://github.com/a/alpha")
        result = list_projects()
        assert "alpha" in result
        assert "https://github.com/a/alpha" in result

    def test_list_multiple_sorted(self):
        add_project("zeta")
        add_project("alpha")
        result = list_projects()
        alpha_pos = result.index("alpha")
        zeta_pos = result.index("zeta")
        assert alpha_pos < zeta_pos

    def test_list_shows_blockers(self):
        add_project("blocked")
        add_blocker("blocked", "need API key")
        result = list_projects()
        assert "need API key" in result


class TestGetProject:
    def test_get_existing(self):
        add_project("detail", "https://github.com/u/detail", notes="test notes")
        result = get_project("detail")
        assert "detail" in result
        assert "https://github.com/u/detail" in result
        assert "test notes" in result
        assert "active" in result.lower()

    def test_get_nonexistent(self):
        result = get_project("nope")
        assert "no project" in result.lower()

    def test_get_shows_created_at(self):
        add_project("timestamped")
        result = get_project("timestamped")
        assert "Created:" in result


class TestAddBlocker:
    def test_add_blocker(self):
        add_project("proj")
        result = add_blocker("proj", "waiting on deploy")
        assert "blocker added" in result.lower()

    def test_add_multiple_blockers(self):
        add_project("proj")
        add_blocker("proj", "first issue")
        add_blocker("proj", "second issue")
        detail = get_project("proj")
        assert "first issue" in detail
        assert "second issue" in detail

    def test_blocker_on_nonexistent(self):
        result = add_blocker("ghost", "blocked")
        assert "no project" in result.lower()


class TestUpdateProject:
    def test_update_status(self):
        add_project("proj")
        result = update_project("proj", status="paused")
        assert "updated" in result.lower()
        detail = get_project("proj")
        assert "paused" in detail

    def test_update_notes(self):
        add_project("proj")
        update_project("proj", notes="new notes")
        detail = get_project("proj")
        assert "new notes" in detail

    def test_update_nonexistent(self):
        result = update_project("ghost", status="done")
        assert "no project" in result.lower()

    def test_update_nothing(self):
        result = update_project("proj")
        assert "nothing" in result.lower()

    def test_update_ignores_unknown_fields(self):
        add_project("proj")
        result = update_project("proj", bad_field="hacked")  # type: ignore[arg-type]
        assert "nothing" in result.lower()


class TestGetAllProjectsRaw:
    def test_empty(self):
        assert get_all_projects_raw() == []

    def test_returns_dicts(self):
        add_project("a")
        add_project("b")
        rows = get_all_projects_raw()
        assert len(rows) == 2
        assert isinstance(rows[0], dict)
        assert "name" in rows[0]


# ================================================================
# Health summary tests
# ================================================================


class TestGetProjectHealthSummary:
    def test_no_projects(self):
        result = get_project_health_summary()
        assert "no projects" in result.lower()

    def test_all_healthy(self):
        add_project("healthy", "https://github.com/u/healthy")
        result = get_project_health_summary()
        assert "healthy" in result.lower()

    def test_project_with_blockers(self):
        add_project("blocked")
        add_blocker("blocked", "waiting on API key")
        result = get_project_health_summary()
        assert "blocked" in result
        assert "waiting on API key" in result

    def test_skips_done_projects(self):
        add_project("finished")
        update_project("finished", status="done")
        result = get_project_health_summary()
        assert "finished" not in result

    def test_paused_project_flagged(self):
        add_project("onhold")
        update_project("onhold", status="paused")
        result = get_project_health_summary()
        assert "onhold" in result
        assert "paused" in result.lower()

    def test_never_synced_project(self):
        add_project("nosync", "https://github.com/u/nosync")
        result = get_project_health_summary()
        assert "nosync" in result
        assert "never synced" in result.lower()

    def test_stale_sync_flagged(self):
        add_project("stale", "https://github.com/u/stale")
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        update_project("stale", last_synced=old_date)
        result = get_project_health_summary()
        assert "stale" in result
        assert "no sync in" in result.lower()

    def test_recent_sync_not_flagged(self):
        add_project("fresh", "https://github.com/u/fresh")
        from datetime import datetime

        update_project("fresh", last_synced=datetime.now().isoformat())
        result = get_project_health_summary()
        # fresh should not appear since it has no issues
        assert "fresh" not in result

    def test_multiple_issues(self):
        add_project("troubled", "https://github.com/u/troubled")
        add_blocker("troubled", "deploy broken")
        from datetime import datetime, timedelta

        old_date = (datetime.now() - timedelta(days=14)).isoformat()
        update_project("troubled", last_synced=old_date)
        result = get_project_health_summary()
        assert "troubled" in result
        assert "deploy broken" in result
        assert "no sync in" in result.lower()

    def test_header_includes_count(self):
        add_project("a")
        add_blocker("a", "issue")
        add_project("b")
        add_blocker("b", "issue")
        result = get_project_health_summary()
        assert "2 tracked" in result


# ================================================================
# Tool interface tests
# ================================================================

class TestToolInterface:
    def test_get_tools_returns_list(self):
        tools = get_project_tracker_tools()
        assert isinstance(tools, list)
        assert len(tools) == 6

    def test_tool_names(self):
        tools = get_project_tracker_tools()
        names = {t.name for t in tools}
        assert "track_project" in names
        assert "untrack_project" in names
        assert "list_tracked_projects" in names
        assert "get_project_detail" in names
        assert "add_project_blocker" in names
        assert "update_project" in names

    def test_track_tool_callable(self):
        tools = get_project_tracker_tools()
        track_tool = next(t for t in tools if t.name == "track_project")
        result = track_tool.function(name="tooltest", repo_url="https://example.com")
        assert "tooltest" in result

    def test_list_tool_callable(self):
        add_project("listed")
        tools = get_project_tracker_tools()
        list_tool = next(t for t in tools if t.name == "list_tracked_projects")
        result = list_tool.function()
        assert "listed" in result


# ================================================================
# Discord command handler tests
# ================================================================

BOT_OWNER = "testowner"


@pytest.fixture(autouse=True)
def _mock_bot_owner(monkeypatch):
    """Set bot owner for command handler tests."""
    monkeypatch.setattr("agent.bot_commands.settings.bot_owner", BOT_OWNER)


class TestHandleTrack:
    def test_track_success(self):
        msg = _make_msg("track myapp https://github.com/u/myapp", user=BOT_OWNER)
        _run(handle_track(msg, msg.content, BOT_OWNER))
        assert "myapp" in msg.replied_to
        assert "tracked" in msg.replied_to.lower()

    def test_track_no_url(self):
        msg = _make_msg("track solo", user=BOT_OWNER)
        _run(handle_track(msg, msg.content, BOT_OWNER))
        assert "solo" in msg.replied_to
        assert "tracked" in msg.replied_to.lower()

    def test_track_missing_name(self):
        msg = _make_msg("track", user=BOT_OWNER)
        _run(handle_track(msg, msg.content, BOT_OWNER))
        assert "usage" in msg.replied_to.lower()

    def test_track_non_owner(self):
        msg = _make_msg("track hax", user="rando")
        _run(handle_track(msg, msg.content, "rando"))
        assert "owner" in msg.replied_to.lower()


class TestHandleUntrack:
    def test_untrack_success(self):
        add_project("removable")
        msg = _make_msg("untrack removable", user=BOT_OWNER)
        _run(handle_untrack(msg, msg.content, BOT_OWNER))
        assert "removed" in msg.replied_to.lower()

    def test_untrack_missing_name(self):
        msg = _make_msg("untrack", user=BOT_OWNER)
        _run(handle_untrack(msg, msg.content, BOT_OWNER))
        assert "usage" in msg.replied_to.lower()

    def test_untrack_non_owner(self):
        msg = _make_msg("untrack x", user="rando")
        _run(handle_untrack(msg, msg.content, "rando"))
        assert "owner" in msg.replied_to.lower()


class TestHandleProjects:
    def test_empty_list(self):
        msg = _make_msg("projects")
        _run(handle_projects(msg))
        assert "no projects" in msg.replied_to.lower()

    def test_with_projects(self):
        add_project("proj1")
        add_project("proj2")
        msg = _make_msg("projects")
        _run(handle_projects(msg))
        assert "proj1" in msg.replied_to
        assert "proj2" in msg.replied_to


class TestHandleProjectDetail:
    def test_detail_success(self):
        add_project("detail", "https://github.com/u/detail")
        msg = _make_msg("project detail")
        _run(handle_project_detail(msg, msg.content))
        assert "detail" in msg.replied_to
        assert "https://github.com/u/detail" in msg.replied_to

    def test_detail_missing_name(self):
        msg = _make_msg("project")
        _run(handle_project_detail(msg, msg.content))
        assert "usage" in msg.replied_to.lower()

    def test_detail_nonexistent(self):
        msg = _make_msg("project ghost")
        _run(handle_project_detail(msg, msg.content))
        assert "no project" in msg.replied_to.lower()


class TestHandleBlocker:
    def test_blocker_success(self):
        add_project("proj")
        msg = _make_msg("blocker proj waiting on API key", user=BOT_OWNER)
        _run(handle_blocker(msg, msg.content, BOT_OWNER))
        assert "blocker added" in msg.replied_to.lower()

    def test_blocker_missing_text(self):
        msg = _make_msg("blocker proj", user=BOT_OWNER)
        _run(handle_blocker(msg, msg.content, BOT_OWNER))
        assert "usage" in msg.replied_to.lower()

    def test_blocker_non_owner(self):
        msg = _make_msg("blocker proj issue", user="rando")
        _run(handle_blocker(msg, msg.content, "rando"))
        assert "owner" in msg.replied_to.lower()
