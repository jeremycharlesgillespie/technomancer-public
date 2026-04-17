"""Tests for project_tracker.py — SQLite-backed project tracking.

Tests CRUD operations (add, remove, list, detail, blocker, update),
Discord command handlers, and GitHub API sync.

Network policy: a module-level autouse fixture replaces
``agent.project_tracker.aiohttp.ClientSession`` with a guard that raises on
construction. Tests that exercise ``github_fetch`` override this via the
``mock_aiohttp_session`` fixture (for direct HTTP mocking) or
``mock_github_fetch`` (to mock at the function boundary). Either way, no
test in this file opens a real socket.
"""

import asyncio
import json
import sqlite3
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.project_tracker import (
    DB_PATH,
    _get_conn,
    _get_github_data,
    _local,
    _parse_last_page,
    add_blocker,
    add_project,
    get_all_projects_raw,
    get_project,
    get_project_health_summary,
    get_project_tracker_tools,
    github_fetch,
    init_db,
    list_projects,
    parse_github_repo,
    remove_project,
    start_github_sync,
    sync_all_projects,
    sync_project_github,
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
# Helpers for building mock aiohttp responses / sessions
# ---------------------------------------------------------------------------

def _make_mock_response(status, json_data, headers=None):
    """Create a mock aiohttp response with async context manager support."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.headers = headers or {}
    # Wrap as async context manager so `async with session.get() as r:` works
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_session(*responses):
    """Build a mock aiohttp.ClientSession with sequenced get() responses."""
    session = AsyncMock()
    if len(responses) == 1:
        session.get = MagicMock(return_value=responses[0])
    else:
        session.get = MagicMock(side_effect=list(responses))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _block_real_http(monkeypatch):
    """Default guard: any code path that opens ``aiohttp.ClientSession``
    without an explicit in-test mock raises immediately.

    Tests that need to exercise ``github_fetch`` (or anything calling it)
    use ``mock_aiohttp_session`` / ``mock_github_fetch``.
    """

    def _blocked(*_a, **_kw):
        raise AssertionError(
            "Real aiohttp.ClientSession() opened in a test. "
            "Use the mock_aiohttp_session or mock_github_fetch fixture."
        )

    monkeypatch.setattr("agent.project_tracker.aiohttp.ClientSession", _blocked)


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


@pytest.fixture
def fake_github_token(monkeypatch):
    """Set a fake GitHub token on the shared settings instance."""
    monkeypatch.setattr(
        "agent.project_tracker.settings.github_token", "fake-token", raising=False
    )


@pytest.fixture
def mock_github_fetch():
    """Patch ``agent.project_tracker.github_fetch`` with an AsyncMock.

    Usage:
        def test_foo(mock_github_fetch):
            mock_github_fetch.return_value = {"stars": 10, ...}
            # ... code that calls sync_project_github / sync_all_projects ...
    """
    with patch(
        "agent.project_tracker.github_fetch", new_callable=AsyncMock
    ) as mock:
        yield mock


@pytest.fixture
def mock_aiohttp_session():
    """Factory fixture that replaces ``aiohttp.ClientSession`` with a mock
    session returning a sequence of responses.

    Usage:
        def test_foo(mock_aiohttp_session):
            mock_aiohttp_session(
                _make_mock_response(200, {"stargazers_count": 1}),
                _make_mock_response(200, []),
                _make_mock_response(200, []),
            )
            result = _run(github_fetch("https://github.com/u/r"))
    """

    with patch("agent.project_tracker.aiohttp.ClientSession") as mock_cs:

        def install(*responses):
            session = _mock_session(*responses)
            mock_cs.return_value = session
            return session

        yield install


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


# ================================================================
# GitHub URL parsing tests
# ================================================================


class TestParseGithubRepo:
    def test_https_url(self):
        assert parse_github_repo("https://github.com/user/repo") == ("user", "repo")

    def test_http_url(self):
        assert parse_github_repo("http://github.com/owner/project") == ("owner", "project")

    def test_url_with_git_suffix(self):
        assert parse_github_repo("https://github.com/user/repo.git") == ("user", "repo")

    def test_url_with_trailing_path(self):
        assert parse_github_repo("https://github.com/user/repo/tree/main") == ("user", "repo")

    def test_non_github_url(self):
        assert parse_github_repo("https://gitlab.com/user/repo") is None

    def test_empty_string(self):
        assert parse_github_repo("") is None

    def test_bare_github_url(self):
        assert parse_github_repo("github.com/user/repo") == ("user", "repo")

    def test_url_with_query_params(self):
        assert parse_github_repo("https://github.com/user/repo?tab=issues") == ("user", "repo")


# ================================================================
# _parse_last_page tests
# ================================================================


class TestParseLastPage:
    def test_link_with_last(self):
        resp = MagicMock()
        resp.headers = {
            "Link": '<https://api.github.com/repos/u/r/pulls?page=5>; rel="last"'
        }
        assert _parse_last_page(resp) == 5

    def test_link_with_next_and_last(self):
        resp = MagicMock()
        resp.headers = {
            "Link": (
                '<https://api.github.com/repos/u/r/pulls?page=2>; rel="next", '
                '<https://api.github.com/repos/u/r/pulls?page=42>; rel="last"'
            )
        }
        assert _parse_last_page(resp) == 42

    def test_no_link_header(self):
        resp = MagicMock()
        resp.headers = {}
        assert _parse_last_page(resp) == 0

    def test_link_without_last(self):
        resp = MagicMock()
        resp.headers = {
            "Link": '<https://api.github.com/repos/u/r/pulls?page=2>; rel="next"'
        }
        assert _parse_last_page(resp) == 0


# ================================================================
# _get_github_data tests
# ================================================================


class TestGetGithubData:
    def test_valid_json(self):
        data = {"open_prs": 3, "open_issues": 5}
        project = {"github_data": json.dumps(data)}
        assert _get_github_data(project) == data

    def test_empty_string(self):
        assert _get_github_data({"github_data": ""}) is None

    def test_missing_key(self):
        assert _get_github_data({}) is None

    def test_invalid_json(self):
        assert _get_github_data({"github_data": "not-json"}) is None


# ================================================================
# github_fetch tests — uses mock_aiohttp_session fixture so no real HTTP
# ================================================================


class TestGithubFetch:
    def test_non_github_url(self):
        # parse_github_repo returns None before aiohttp is ever touched.
        result = _run(github_fetch("https://gitlab.com/u/r"))
        assert result is None

    def test_successful_fetch(self, mock_aiohttp_session):
        mock_aiohttp_session(
            _make_mock_response(200, {
                "description": "A cool project",
                "stargazers_count": 42,
                "open_issues_count": 7,
            }),
            _make_mock_response(200, [{"id": 1}], {
                "Link": '<https://api.github.com/repos/u/r/pulls?page=3>; rel="last"'
            }),
            _make_mock_response(200, [{
                "sha": "abc1234567890",
                "commit": {"committer": {"date": "2026-04-10T12:00:00Z"}},
            }]),
        )

        result = _run(github_fetch("https://github.com/user/repo", token="test-token"))

        assert result is not None
        assert result["description"] == "A cool project"
        assert result["stars"] == 42
        assert result["open_issues"] == 7
        assert result["open_prs"] == 3
        assert result["last_commit_sha"] == "abc1234"
        assert result["last_commit_date"] == "2026-04-10T12:00:00Z"

    def test_repo_api_failure(self, mock_aiohttp_session):
        mock_aiohttp_session(_make_mock_response(404, {}))
        result = _run(github_fetch("https://github.com/user/repo"))
        assert result is None

    def test_no_prs(self, mock_aiohttp_session):
        mock_aiohttp_session(
            _make_mock_response(200, {
                "description": "Empty",
                "stargazers_count": 0,
                "open_issues_count": 0,
            }),
            _make_mock_response(200, []),
            _make_mock_response(200, []),
        )

        result = _run(github_fetch("https://github.com/user/repo"))

        assert result is not None
        assert result["open_prs"] == 0
        assert result["last_commit_sha"] == ""


# ================================================================
# sync_project_github tests — uses mock_github_fetch
# ================================================================


class TestSyncProjectGithub:
    def test_sync_stores_data(self, mock_github_fetch, fake_github_token):
        add_project("myproj", "https://github.com/user/myproj")
        gh_data = {
            "description": "Test",
            "stars": 10,
            "open_issues": 3,
            "open_prs": 2,
            "last_commit_sha": "abc1234",
            "last_commit_date": "2026-04-10T12:00:00Z",
        }
        mock_github_fetch.return_value = gh_data

        result = _run(sync_project_github("myproj", "https://github.com/user/myproj"))

        assert result is True
        # Verify data was stored
        raw = get_all_projects_raw()
        proj = raw[0]
        assert proj["last_synced"] != ""
        stored = json.loads(proj["github_data"])
        assert stored["open_prs"] == 2
        assert stored["stars"] == 10

    def test_sync_returns_false_on_fetch_failure(
        self, mock_github_fetch, fake_github_token
    ):
        add_project("failproj", "https://github.com/user/failproj")
        mock_github_fetch.return_value = None

        result = _run(
            sync_project_github("failproj", "https://github.com/user/failproj")
        )

        assert result is False


# ================================================================
# sync_all_projects tests — uses mock_github_fetch
# ================================================================


class TestSyncAllProjects:
    def test_syncs_github_projects_only(self, mock_github_fetch, fake_github_token):
        add_project("with_gh", "https://github.com/user/with_gh")
        add_project("no_url")
        add_project("gitlab", "https://gitlab.com/user/gitlab")

        mock_github_fetch.return_value = {
            "description": "Test",
            "stars": 1,
            "open_issues": 0,
            "open_prs": 0,
            "last_commit_sha": "abc",
            "last_commit_date": "",
        }

        results = _run(sync_all_projects())

        # Only the GitHub project should be in results
        assert "with_gh" in results
        assert results["with_gh"] is True
        assert "no_url" not in results
        assert "gitlab" not in results

    def test_empty_projects(self):
        results = _run(sync_all_projects())
        assert results == {}


# ================================================================
# list_projects with GitHub data tests
# ================================================================


class TestListProjectsWithGithub:
    def test_list_shows_pr_count(self):
        add_project("ghproj", "https://github.com/user/ghproj")
        gh_data = json.dumps({"open_prs": 5, "open_issues": 12, "stars": 100})
        update_project("ghproj", github_data=gh_data)
        result = list_projects()
        assert "5 PRs" in result
        assert "12 issues" in result
        assert "100" in result  # stars

    def test_list_no_github_data(self):
        add_project("plain")
        result = list_projects()
        assert "plain" in result
        assert "PRs" not in result


# ================================================================
# get_project with GitHub data tests
# ================================================================


class TestGetProjectWithGithub:
    def test_detail_shows_github_section(self):
        add_project("ghdetail", "https://github.com/user/ghdetail")
        gh_data = json.dumps({
            "description": "A detailed project",
            "open_prs": 3,
            "open_issues": 8,
            "stars": 50,
            "last_commit_sha": "abc1234",
            "last_commit_date": "2026-04-10T12:00:00Z",
        })
        update_project("ghdetail", github_data=gh_data)
        result = get_project("ghdetail")
        assert "**GitHub**" in result
        assert "A detailed project" in result
        assert "Open PRs: 3" in result
        assert "Open Issues: 8" in result
        assert "Stars: 50" in result
        assert "abc1234" in result

    def test_detail_without_github_data(self):
        add_project("nogh")
        result = get_project("nogh")
        assert "**GitHub**" not in result


# ================================================================
# Health summary with GitHub data tests
# ================================================================


class TestHealthSummaryWithGithub:
    def test_open_prs_flagged(self):
        add_project("prproj", "https://github.com/u/prproj")
        gh_data = json.dumps({"open_prs": 3, "open_issues": 2, "stars": 0})
        update_project("prproj", github_data=gh_data, last_synced=datetime.now().isoformat())
        result = get_project_health_summary()
        assert "prproj" in result
        assert "Open PRs: 3" in result

    def test_high_issues_flagged(self):
        add_project("issueproj", "https://github.com/u/issueproj")
        gh_data = json.dumps({"open_prs": 0, "open_issues": 10, "stars": 0})
        update_project("issueproj", github_data=gh_data, last_synced=datetime.now().isoformat())
        result = get_project_health_summary()
        assert "issueproj" in result
        assert "Open issues: 10" in result

    def test_low_issues_not_flagged(self):
        add_project("fineproj", "https://github.com/u/fineproj")
        gh_data = json.dumps({"open_prs": 0, "open_issues": 3, "stars": 0})
        update_project("fineproj", github_data=gh_data, last_synced=datetime.now().isoformat())
        result = get_project_health_summary()
        # 3 issues is below threshold of 5, and 0 PRs — should be healthy
        assert "fineproj" not in result


# ================================================================
# start_github_sync tests
# ================================================================


class TestStartGithubSync:
    def test_no_token_does_not_start(self, monkeypatch):
        monkeypatch.setattr(
            "agent.project_tracker.settings.github_token", None, raising=False
        )
        with patch("agent.task_manager.create_monitored_task") as mock_task:
            start_github_sync()
            mock_task.assert_not_called()

    def test_with_token_starts_task(self, fake_github_token):
        # Replace the loop factory with a plain callable so we don't leak an
        # unawaited coroutine (patch() on an async function defaults to an
        # AsyncMock whose return is itself an un-awaited coroutine).
        with patch(
            "agent.project_tracker._github_sync_loop", new=lambda: None
        ), patch("agent.task_manager.create_monitored_task") as mock_task:
            start_github_sync()
            mock_task.assert_called_once()


# ================================================================
# DB migration tests
# ================================================================


class TestDbMigration:
    def test_github_data_column_exists(self):
        """Verify the github_data column is present after init_db."""
        conn = _get_conn()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
        assert "github_data" in cols

    def test_add_project_with_github_data(self):
        """Verify github_data defaults to empty string on new projects."""
        add_project("migtest", "https://github.com/u/migtest")
        raw = get_all_projects_raw()
        proj = next(p for p in raw if p["name"] == "migtest")
        assert proj["github_data"] == ""
