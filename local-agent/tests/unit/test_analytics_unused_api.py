"""Tests for /api/analytics/unused and the clickable Unused Features card."""

from unittest.mock import patch

import pytest

import agent.engagement_analytics as ea
from idea_board.web import app


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect engagement DB to a temp dir for each test."""
    monkeypatch.setattr(ea, "DB_DIR", tmp_path)
    monkeypatch.setattr(ea, "DB_PATH", tmp_path / "engagement.db")
    if hasattr(ea._local, "eng_conn"):
        try:
            ea._local.eng_conn.close()
        except Exception:
            pass
        del ea._local.eng_conn
    ea.init_db()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestApiAnalyticsUnused:
    def test_returns_200(self, client):
        resp = client.get("/api/analytics/unused")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        assert "days" in data
        assert "definition" in data
        assert "commands" in data
        assert isinstance(data["commands"], list)

    def test_default_days_is_14(self, client):
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        assert data["days"] == 14
        assert "14" in data["definition"]

    def test_days_query_param_respected(self, client):
        resp = client.get("/api/analytics/unused?days=7")
        data = resp.get_json()
        assert data["days"] == 7
        assert "7" in data["definition"]

    def test_command_fields(self, client):
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        # With a fresh DB, every registered command is "unused"
        assert len(data["commands"]) > 0
        for cmd in data["commands"]:
            assert "name" in cmd
            assert "description" in cmd
            assert "last_seen" in cmd
            assert "invocations" in cmd

    def test_used_command_excluded(self, client):
        from agent.command_suggestions import COMMANDS
        name = COMMANDS[0].name
        ea.track_command(name, user="alice")
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        names_lower = {c["name"].lower() for c in data["commands"]}
        assert name.lower() not in names_lower


class TestAnalyticsPageRendersClickableCard:
    def test_unused_card_has_click_handler(self, client):
        resp = client.get("/analytics")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        # The stat card should be clickable and wire up the modal open fn
        assert 'id="unused-card"' in body
        assert "openUnusedModal" in body
        assert 'id="unused-modal"' in body

    def test_page_includes_definition_tooltip(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        # Definition of "unused" is surfaced as a tooltip on the card
        assert "Click to see which commands" in body or "14 days" in body

    def test_modal_fetches_unused_api(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "/api/analytics/unused" in body

    def test_modal_has_accessibility_attributes(self, client):
        """Modal is announced to assistive tech with role+label."""
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert 'role="dialog"' in body
        assert 'aria-labelledby="unused-modal-title"' in body
        assert 'id="unused-modal-title"' in body

    def test_modal_closes_on_escape_key(self, client):
        """Escape key closes the modal — keyboard accessibility."""
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "if (e.key === 'Escape') closeUnusedModal();" in body

    def test_modal_js_guards_against_http_errors(self, client):
        """A 5xx from the API should surface as a real error, not a silent
        JSON-parse failure. The fetch handler must check resp.ok before
        parsing the body."""
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "if (!resp.ok)" in body
        assert "throw new Error('HTTP ' + resp.status)" in body

    def test_modal_handles_empty_unused_list(self, client):
        """Modal JS shows a friendly message when every command was used."""
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "Every command has been used recently." in body


class TestApiAnalyticsUnusedEdgeCases:
    def test_returns_empty_list_when_all_commands_used(self, client):
        from agent.command_suggestions import COMMANDS
        for cmd in COMMANDS:
            ea.track_command(cmd.name, user="alice")
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        assert data["commands"] == []

    def test_commands_sorted_by_name(self, client):
        resp = client.get("/api/analytics/unused")
        data = resp.get_json()
        names = [c["name"].lower() for c in data["commands"]]
        assert names == sorted(names)


class TestAnalyticsStatCardTooltips:
    """TK-550: every stat card on /analytics has a title tooltip with a
    one-sentence definition of the metric plus its time window."""

    def test_every_stat_card_has_title_attribute(self, client):
        import re
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        # Find every stat-card opening tag and confirm each one has title=.
        # Pattern matches both <div class="stat-card" ...> and the clickable variant.
        card_tags = re.findall(r'<div class="stat-card[^"]*"[^>]*>', body)
        assert len(card_tags) == 4, f"expected 4 stat cards, found {len(card_tags)}"
        for tag in card_tags:
            assert 'title="' in tag, f"stat card missing title tooltip: {tag}"

    def test_commands_card_tooltip_mentions_time_window(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert 'title="Total Discord bot command invocations' in body
        assert "in the last 14 days" in body

    def test_messages_card_tooltip_defines_metric(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert 'title="Total Discord messages sent in watched channels' in body

    def test_unique_commands_card_tooltip_defines_metric(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert 'title="Number of distinct commands' in body

    def test_unused_card_tooltip_defines_metric_and_action(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        # The tooltip should describe what the metric means *and* hint at
        # the click action (which still opens the modal).
        assert "received zero invocations" in body
        assert "Click to see the list" in body


class TestDiscordAnalyticsLabel:
    """TK-594: the analytics surface is explicitly labeled 'Discord Analytics'
    everywhere it appears (hub card, page heading, nav links) so operators
    know all metrics describe Discord bot usage."""

    def test_analytics_page_h1_says_discord_analytics(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "<h1>Discord Analytics</h1>" in body

    def test_analytics_page_title_says_discord_analytics(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert "<title>Discord Analytics" in body

    def test_analytics_page_active_nav_link_says_discord_analytics(self, client):
        resp = client.get("/analytics")
        body = resp.get_data(as_text=True)
        assert '<a href="/analytics" class="active">Discord Analytics</a>' in body

    def test_ideas_page_nav_link_says_discord_analytics(self, client):
        resp = client.get("/ideas")
        body = resp.get_data(as_text=True)
        assert '<a href="/analytics">Discord Analytics</a>' in body

    def test_hub_card_heading_says_discord_analytics(self, client):
        resp = client.get("/")
        body = resp.get_data(as_text=True)
        # The hub card linking to /analytics carries the Discord-Analytics title.
        assert "Discord Analytics" in body
        assert 'href="/analytics"' in body

    def test_no_bare_analytics_nav_link_remains(self, client):
        """Guard against regressions: no nav/card should render the ambiguous
        plain-'Analytics' label pointing at /analytics."""
        for path in ("/", "/analytics", "/ideas"):
            resp = client.get(path)
            body = resp.get_data(as_text=True)
            assert '<a href="/analytics">Analytics</a>' not in body
            assert '<a href="/analytics" class="active">Analytics</a>' not in body
