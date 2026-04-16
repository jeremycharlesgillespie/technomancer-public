"""Tests for BoardProvider.get_comments across LocalProvider and JiraProvider.

Covers the provider-agnostic Comment shape, marker parsing, and the fact
that both backends expose the same read interface so consumers (like the
executor's retry-memory injector) can be backend-agnostic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from board.provider import Comment, parse_marker


# ---------------------------------------------------------------------------
# parse_marker
# ---------------------------------------------------------------------------


class TestParseMarker:
    def test_returns_known_marker_at_start(self):
        assert parse_marker("[Execution Log - Failed]\nstack trace...") == "[Execution Log - Failed]"

    def test_returns_progress_marker(self):
        assert parse_marker("[AIM Progress]\nline1\nline2") == "[AIM Progress]"

    def test_returns_author_prefix_as_marker(self):
        # Any leading bracketed prefix qualifies; callers filter by literal.
        assert parse_marker("[claude] please retry") == "[claude]"

    def test_returns_none_for_free_form(self):
        assert parse_marker("just a free-form comment") is None

    def test_returns_none_for_empty(self):
        assert parse_marker("") is None

    def test_returns_none_for_whitespace_only(self):
        assert parse_marker("   \n\n   ") is None

    def test_skips_leading_blank_lines(self):
        assert parse_marker("\n\n[Execution Log]\nbody") == "[Execution Log]"

    def test_tolerates_leading_spaces(self):
        assert parse_marker("   [Epic Context]\nnarrative") == "[Epic Context]"

    def test_brackets_must_be_at_line_start(self):
        # Bracket mid-line does not count as a marker.
        assert parse_marker("hello [Execution Log - Failed]") is None


# ---------------------------------------------------------------------------
# LocalProvider — roundtrip through idea_board.models
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_local(tmp_path, monkeypatch):
    """Point idea_board.models at a temp ideas.json + vault dir."""
    import idea_board.models as models_module

    monkeypatch.setattr(models_module, "IDEAS_FILE", tmp_path / "ideas.json")
    monkeypatch.setattr(models_module, "VAULT_IDEAS_DIR", tmp_path / "vault_ideas")
    (tmp_path / "vault_ideas").mkdir()
    yield tmp_path


class TestLocalProviderGetComments:
    def test_roundtrip_comment_with_failure_marker(self, patched_local):
        from board.local_provider import LocalProvider
        from idea_board.models import add_idea

        provider = LocalProvider()
        idea = add_idea("Retry target", "do a thing", category="quality")

        provider.add_comment(
            idea.id,
            "claude",
            "[Execution Log - Failed]\ntrace: missing import",
        )

        comments = provider.get_comments(idea.id)
        assert len(comments) == 1
        c = comments[0]
        assert isinstance(c, Comment)
        assert c.author == "claude"
        assert c.text.startswith("[Execution Log - Failed]")
        assert "missing import" in c.text
        assert c.marker == "[Execution Log - Failed]"
        assert c.created  # populated by Comment.__post_init__

    def test_returns_multiple_comments_in_order(self, patched_local):
        from board.local_provider import LocalProvider
        from idea_board.models import add_idea

        provider = LocalProvider()
        idea = add_idea("Multi-comment story", "desc here", category="quality")

        provider.add_comment(idea.id, "owner", "free-form feedback")
        provider.add_comment(idea.id, "claude", "[AIM Progress]\nworking on step 3")
        provider.add_comment(idea.id, "claude", "[Execution Log - Failed]\ntest failure")

        comments = provider.get_comments(idea.id)
        assert [c.marker for c in comments] == [None, "[AIM Progress]", "[Execution Log - Failed]"]
        assert [c.author for c in comments] == ["owner", "claude", "claude"]

    def test_returns_empty_for_missing_item(self, patched_local):
        from board.local_provider import LocalProvider

        provider = LocalProvider()
        assert provider.get_comments("idea-does-not-exist") == []

    def test_returns_empty_when_item_has_no_comments(self, patched_local):
        from board.local_provider import LocalProvider
        from idea_board.models import add_idea

        provider = LocalProvider()
        idea = add_idea("No comments", "description text here", category="quality")
        assert provider.get_comments(idea.id) == []


# ---------------------------------------------------------------------------
# JiraProvider — mocked _api
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_jira_configured(monkeypatch):
    monkeypatch.setattr("idea_board.jira_sync.is_jira_configured", lambda: True)
    monkeypatch.setattr("board.jira_provider.is_jira_configured", lambda: True)


def _adf_body(text: str) -> dict:
    """ADF document with a single paragraph containing ``text``."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ],
    }


def _jira_comment(
    *,
    author: str = "Claude",
    body_text: str = "hello world",
    created: str = "2026-04-15T10:00:00.000+0000",
) -> dict:
    return {
        "id": "10000",
        "author": {"displayName": author},
        "body": _adf_body(body_text),
        "created": created,
    }


@pytest.fixture
def provider():
    from board.jira_provider import JiraProvider

    return JiraProvider()


class TestJiraProviderGetComments:
    def test_parses_failure_marker(self, provider):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "comments": [
                _jira_comment(
                    author="Worker Bot",
                    body_text="[Execution Log - Failed]\ntraceback line 1",
                    created="2026-04-15T12:00:00.000+0000",
                ),
            ],
        }
        with patch("board.jira_provider._api", return_value=resp) as api:
            comments = provider.get_comments("TK-42")

        api.assert_called_once()
        assert api.call_args.args[0] == "get"
        assert "/issue/TK-42/comment" in api.call_args.args[1]

        assert len(comments) == 1
        c = comments[0]
        assert c.author == "Worker Bot"
        assert c.marker == "[Execution Log - Failed]"
        assert "traceback line 1" in c.text
        assert c.created == "2026-04-15T12:00:00.000+0000"

    def test_parses_progress_marker(self, provider):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "comments": [
                _jira_comment(body_text="[AIM Progress]\nstep 1\nstep 2"),
            ],
        }
        with patch("board.jira_provider._api", return_value=resp):
            comments = provider.get_comments("TK-1")

        assert [c.marker for c in comments] == ["[AIM Progress]"]

    def test_free_form_comment_has_none_marker(self, provider):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "comments": [_jira_comment(body_text="casual note from a reviewer")],
        }
        with patch("board.jira_provider._api", return_value=resp):
            comments = provider.get_comments("TK-1")

        assert comments[0].marker is None
        assert comments[0].text.startswith("casual note")

    def test_preserves_order_from_api(self, provider):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "comments": [
                _jira_comment(body_text="first"),
                _jira_comment(body_text="[AIM Progress]\nsecond"),
                _jira_comment(body_text="[Execution Log - Failed]\nthird"),
            ],
        }
        with patch("board.jira_provider._api", return_value=resp):
            comments = provider.get_comments("TK-1")

        assert [c.marker for c in comments] == [None, "[AIM Progress]", "[Execution Log - Failed]"]

    def test_returns_empty_on_non_200(self, provider):
        resp = MagicMock()
        resp.status_code = 404
        with patch("board.jira_provider._api", return_value=resp):
            assert provider.get_comments("TK-missing") == []

    def test_returns_empty_on_api_exception(self, provider):
        with patch("board.jira_provider._api", side_effect=RuntimeError("boom")):
            assert provider.get_comments("TK-1") == []

    def test_returns_empty_when_jira_not_configured(self, provider, monkeypatch):
        monkeypatch.setattr("board.jira_provider.is_jira_configured", lambda: False)
        # _api should not be called at all — keep it raising to prove it.
        with patch("board.jira_provider._api", side_effect=AssertionError("should not call")):
            assert provider.get_comments("TK-1") == []

    def test_handles_missing_author_field(self, provider):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "comments": [
                {
                    "id": "1",
                    "body": _adf_body("[Execution Log - Failed]\nx"),
                    "created": "2026-04-15T00:00:00.000+0000",
                    # no author
                },
            ],
        }
        with patch("board.jira_provider._api", return_value=resp):
            comments = provider.get_comments("TK-1")

        assert comments[0].author == ""
        assert comments[0].marker == "[Execution Log - Failed]"


# ---------------------------------------------------------------------------
# Protocol compliance — both providers satisfy BoardProvider
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_local_provider_exposes_get_comments(self):
        from board import BoardProvider
        from board.local_provider import LocalProvider

        assert isinstance(LocalProvider(), BoardProvider)
        assert hasattr(LocalProvider(), "get_comments")

    def test_jira_provider_exposes_get_comments(self):
        from board import BoardProvider
        from board.jira_provider import JiraProvider

        assert isinstance(JiraProvider(), BoardProvider)
        assert hasattr(JiraProvider(), "get_comments")
