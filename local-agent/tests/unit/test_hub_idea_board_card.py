"""Tests for the hub home page's Idea Board card — Jira-aware linking (TK-545)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _empty_ideas():
    """Render the hub without relying on real board data."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


class TestIdeaBoardCardWithoutJira:
    """When Jira isn't configured, the card stays local and unchanged."""

    def test_href_falls_back_to_ideas(self, client):
        with patch("idea_board.web.settings.jira_url", None), \
             patch("idea_board.web.settings.jira_project_key", None):
            body = client.get("/").get_data(as_text=True)
        assert '<a href="/ideas"' in body

    def test_no_target_blank_when_local(self, client):
        with patch("idea_board.web.settings.jira_url", None), \
             patch("idea_board.web.settings.jira_project_key", None):
            body = client.get("/").get_data(as_text=True)
        assert '<a href="/ideas" class="card green">' in body

    def test_no_using_jira_subtext_when_local(self, client):
        with patch("idea_board.web.settings.jira_url", None), \
             patch("idea_board.web.settings.jira_project_key", None):
            body = client.get("/").get_data(as_text=True)
        assert "Using Jira" not in body

    def test_jira_url_without_project_key_falls_back(self, client):
        """A half-configured Jira (url but no project key) must not produce a broken link."""
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", None):
            body = client.get("/").get_data(as_text=True)
        assert '<a href="/ideas"' in body
        assert "atlassian.net/jira/software" not in body


class TestIdeaBoardCardWithJira:
    """When Jira is configured, the card links out and shows the subtext."""

    def test_href_points_to_jira_boards(self, client):
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        assert 'href="https://acme.atlassian.net/jira/software/projects/TK/boards"' in body

    def test_opens_in_new_tab(self, client):
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        # The anchor must have target="_blank" so Jira opens in a new tab.
        assert (
            'href="https://acme.atlassian.net/jira/software/projects/TK/boards"'
            ' target="_blank"'
        ) in body

    def test_trailing_slash_on_jira_url_is_stripped(self, client):
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net/"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        assert 'href="https://acme.atlassian.net/jira/software/projects/TK/boards"' in body
        assert "atlassian.net//jira/software" not in body

    def test_uses_configured_project_key(self, client):
        """Project key must be interpolated, not hardcoded to TK."""
        with patch("idea_board.web.settings.jira_url", "https://other.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "FA"):
            body = client.get("/").get_data(as_text=True)
        assert 'href="https://other.atlassian.net/jira/software/projects/FA/boards"' in body
        assert "Using Jira — idea board backed by the FA project." in body

    def test_using_jira_subtext_rendered(self, client):
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        assert "Using Jira — idea board backed by the TK project." in body

    def test_completion_badge_preserved(self, client):
        """Adding the Jira link must not remove the done/total counter badge."""
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        assert 'class="badge"' in body
        # With no ideas loaded by the fixture, the badge reads 0/0.
        assert "0/0 done" in body

    def test_card_still_green_variant(self, client):
        """The green styling should be preserved."""
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/").get_data(as_text=True)
        assert 'class="card green"' in body
