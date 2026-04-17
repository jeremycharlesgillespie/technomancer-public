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
