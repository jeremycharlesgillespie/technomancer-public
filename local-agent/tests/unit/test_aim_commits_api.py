"""Tests for the GET /api/aim/commits endpoint in idea_board/web.py."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _git_log_stdout(commits: list[tuple[str, str, str, str]]) -> str:
    """Build a NUL-delimited git log output from (sha, subj, author, ts) tuples."""
    return "\n".join("\x00".join(parts) for parts in commits)


def _mock_success(commits: list[tuple[str, str, str, str]]):
    """Return a CompletedProcess-like mock for a successful git log call."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = _git_log_stdout(commits)
    result.stderr = ""
    return result


def _mock_failure(stderr: str = "fatal: not a git repository"):
    """Return a CompletedProcess-like mock for a failing git log call."""
    result = MagicMock()
    result.returncode = 128
    result.stdout = ""
    result.stderr = stderr
    return result


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_returns_200(self, client):
        with patch("idea_board.web.subprocess.run", return_value=_mock_success([])):
            resp = client.get("/api/aim/commits")
            assert resp.status_code == 200

    def test_content_type_is_json(self, client):
        with patch("idea_board.web.subprocess.run", return_value=_mock_success([])):
            resp = client.get("/api/aim/commits")
            assert "application/json" in resp.content_type

    def test_default_repo_is_both(self, client):
        """Acceptance: default response has private and public keys."""
        with patch("idea_board.web.subprocess.run", return_value=_mock_success([])):
            data = client.get("/api/aim/commits").get_json()
            assert set(data.keys()) == {"private", "public"}
            assert data["private"] == []
            assert data["public"] == []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parses_nul_delimited_commits(self, client):
        """Each commit dict must have sha, subject, author, timestamp."""
        commits = [
            ("abc1234", "Add feature X", "Jeremy", "2026-04-16T00:12:26-04:00"),
            ("def5678", "Fix bug Y", "Claude", "2026-04-15T23:45:00-04:00"),
        ]
        with patch(
            "idea_board.web.subprocess.run",
            return_value=_mock_success(commits),
        ):
            data = client.get("/api/aim/commits?repo=private").get_json()

        assert len(data) == 2
        assert data[0] == {
            "sha": "abc1234",
            "subject": "Add feature X",
            "author": "Jeremy",
            "timestamp": "2026-04-16T00:12:26-04:00",
        }
        assert data[1]["sha"] == "def5678"

    def test_survives_malformed_line(self, client):
        """Lines that don't have exactly 4 NUL-delimited parts are skipped."""
        stdout = (
            "abc1234\x00Subject\x00Author\x002026-04-16T00:00:00-04:00\n"
            "junk-line-without-delimiters\n"
            "def5678\x00Other\x00Author\x002026-04-15T00:00:00-04:00\n"
        )
        result = MagicMock()
        result.returncode = 0
        result.stdout = stdout
        result.stderr = ""
        with patch("idea_board.web.subprocess.run", return_value=result):
            data = client.get("/api/aim/commits?repo=private").get_json()

        assert len(data) == 2
        assert data[0]["sha"] == "abc1234"
        assert data[1]["sha"] == "def5678"

    def test_empty_repo_returns_empty_list(self, client):
        with patch("idea_board.web.subprocess.run", return_value=_mock_success([])):
            data = client.get("/api/aim/commits?repo=private").get_json()
            assert data == []


# ---------------------------------------------------------------------------
# Limit handling
# ---------------------------------------------------------------------------


class TestLimit:
    def test_default_limit_is_10(self, client):
        """Without ?limit, git log is invoked with -10."""
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?repo=private")

        args = mock.call_args.args[0]
        assert "-10" in args

    def test_accepts_custom_limit(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?limit=5&repo=private")

        args = mock.call_args.args[0]
        assert "-5" in args

    def test_clamps_low_limit_to_1(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?limit=0&repo=private")

        args = mock.call_args.args[0]
        assert "-1" in args

    def test_clamps_high_limit_to_50(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?limit=9999&repo=private")

        args = mock.call_args.args[0]
        assert "-50" in args

    def test_clamps_negative_limit_to_1(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?limit=-5&repo=private")

        args = mock.call_args.args[0]
        assert "-1" in args

    def test_invalid_limit_falls_back_to_default(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?limit=not-a-number&repo=private")

        args = mock.call_args.args[0]
        assert "-10" in args


# ---------------------------------------------------------------------------
# Repo selection
# ---------------------------------------------------------------------------


class TestRepoSelection:
    def test_repo_private_returns_list(self, client):
        with patch(
            "idea_board.web.subprocess.run",
            return_value=_mock_success([
                ("abc1234", "Subject", "Author", "2026-04-16T00:00:00-04:00"),
            ]),
        ):
            data = client.get("/api/aim/commits?repo=private").get_json()

        assert isinstance(data, list)
        assert data[0]["sha"] == "abc1234"

    def test_repo_public_returns_list(self, client):
        with patch(
            "idea_board.web.subprocess.run",
            return_value=_mock_success([
                ("xyz9999", "Public subject", "Author", "2026-04-16T00:00:00-04:00"),
            ]),
        ):
            data = client.get("/api/aim/commits?repo=public").get_json()

        assert isinstance(data, list)
        assert data[0]["sha"] == "xyz9999"

    def test_repo_both_returns_dict(self, client):
        """repo=both returns {"private": [...], "public": [...]}."""
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            calls.append(kwargs.get("cwd", ""))
            # Differentiate private vs public by cwd, so each side gets
            # distinctive commits — makes the keying assertion meaningful.
            if "public" in (kwargs.get("cwd") or ""):
                return _mock_success([
                    ("pub0001", "Public commit", "Author",
                     "2026-04-16T00:00:00-04:00"),
                ])
            return _mock_success([
                ("prv0001", "Private commit", "Author",
                 "2026-04-16T00:00:00-04:00"),
            ])

        with patch("idea_board.web.subprocess.run", side_effect=fake_run):
            data = client.get("/api/aim/commits?repo=both").get_json()

        assert set(data.keys()) == {"private", "public"}
        assert data["private"][0]["sha"] == "prv0001"
        assert data["public"][0]["sha"] == "pub0001"
        assert len(calls) == 2

    def test_invalid_repo_falls_back_to_both(self, client):
        with patch("idea_board.web.subprocess.run", return_value=_mock_success([])):
            data = client.get("/api/aim/commits?repo=bogus").get_json()
        assert set(data.keys()) == {"private", "public"}

    def test_acceptance_limit_5_both(self, client):
        """Acceptance: ?limit=5 returns private + public with 5 commits each."""
        five_commits = [
            (f"sha{i:04d}", f"Subject {i}", "Author",
             f"2026-04-{16 - i:02d}T00:00:00-04:00")
            for i in range(5)
        ]

        def fake_run(cmd, **kwargs):
            return _mock_success(five_commits)

        with patch("idea_board.web.subprocess.run", side_effect=fake_run):
            data = client.get("/api/aim/commits?limit=5").get_json()

        assert len(data["private"]) == 5
        assert len(data["public"]) == 5
        for commit in data["private"] + data["public"]:
            assert set(commit.keys()) == {"sha", "subject", "author", "timestamp"}


# ---------------------------------------------------------------------------
# Git command shape
# ---------------------------------------------------------------------------


class TestGitCommand:
    def test_uses_git_log_with_pretty_format(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?repo=private&limit=3")

        args = mock.call_args.args[0]
        assert args[0] == "git"
        assert args[1] == "log"
        assert args[2] == "-3"
        # NUL-delimited pretty format for robust parsing.
        assert args[3] == "--pretty=format:%h%x00%s%x00%aN%x00%aI"

    def test_uses_timeout_and_capture(self, client):
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?repo=private")

        kwargs = mock.call_args.kwargs
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") == 10

    def test_private_and_public_cwds_differ(self, client):
        """Private and public repos must resolve to different directories."""
        cwds: list[str] = []

        def fake_run(cmd, **kwargs):
            cwds.append(kwargs["cwd"])
            return _mock_success([])

        with patch("idea_board.web.subprocess.run", side_effect=fake_run):
            client.get("/api/aim/commits?repo=both")

        assert len(cwds) == 2
        assert cwds[0] != cwds[1]

    def test_public_cwd_is_sibling_technomancer_public(self, client):
        """Public repo sits next to the private repo as technomancer-public."""
        mock = MagicMock(return_value=_mock_success([]))
        with patch("idea_board.web.subprocess.run", mock):
            client.get("/api/aim/commits?repo=public")

        cwd = mock.call_args.kwargs["cwd"]
        # Forward-slash normalize for cross-platform string matching.
        assert cwd.replace("\\", "/").rstrip("/").endswith("technomancer-public")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_500_on_git_nonzero_exit(self, client):
        with patch(
            "idea_board.web.subprocess.run",
            return_value=_mock_failure("fatal: not a git repository"),
        ):
            resp = client.get("/api/aim/commits?repo=private")

        assert resp.status_code == 500
        body = resp.get_json()
        assert "error" in body
        assert "not a git repository" in body["error"]

    def test_500_on_subprocess_exception(self, client):
        with patch(
            "idea_board.web.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git log", timeout=10),
        ):
            resp = client.get("/api/aim/commits?repo=private")

        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_500_on_os_error(self, client):
        """If the repo path doesn't exist (FileNotFoundError), return 500."""
        with patch(
            "idea_board.web.subprocess.run",
            side_effect=FileNotFoundError("no such directory"),
        ):
            resp = client.get("/api/aim/commits?repo=public")

        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_500_on_both_if_either_fails(self, client):
        """repo=both fails fast if either repo errors — returns 500."""
        calls = {"count": 0}

        def fake_run(cmd, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return _mock_success([])
            return _mock_failure("oops")

        with patch("idea_board.web.subprocess.run", side_effect=fake_run):
            resp = client.get("/api/aim/commits?repo=both")

        assert resp.status_code == 500
