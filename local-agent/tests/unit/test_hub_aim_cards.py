"""Tests for the AIM cards on the Technomancer hub home page (TK-551)."""

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
    """Keep the hub render path independent of real board data."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


class TestHubAimCards:
    def test_hub_has_aim_status_card(self, client):
        body = client.get("/").get_data(as_text=True)
        assert 'href="/aim"' in body
        assert "AIM Status" in body

    def test_hub_has_aim_dashboard_card(self, client):
        body = client.get("/").get_data(as_text=True)
        assert 'href="/aim/dashboard"' in body
        assert "AIM Dashboard" in body

    def test_aim_cards_use_card_class(self, client):
        """Both AIM cards must reuse the shared .card CSS class."""
        body = client.get("/").get_data(as_text=True)
        # A quick-and-dirty check: the anchor tags include class="card"
        assert 'href="/aim"' in body
        # Find the /aim anchor snippet and make sure class="card" is present.
        idx = body.index('href="/aim"')
        # Walk backwards to the opening < of the anchor.
        start = body.rfind("<a ", 0, idx)
        snippet = body[start : idx + 200]
        assert "class=\"card" in snippet

        idx2 = body.index('href="/aim/dashboard"')
        start2 = body.rfind("<a ", 0, idx2)
        snippet2 = body[start2 : idx2 + 200]
        assert "class=\"card" in snippet2

    def test_aim_cards_have_status_badges(self, client):
        """Both AIM cards should include placeholder badges populated by JS."""
        body = client.get("/").get_data(as_text=True)
        assert 'id="aim-status-badge"' in body
        assert 'id="aim-dashboard-badge"' in body

    def test_hub_polls_aim_status_endpoint(self, client):
        """Hub JS must fetch /api/aim/status to populate badges."""
        body = client.get("/").get_data(as_text=True)
        assert "/api/aim/status" in body
        assert "updateAimStatus" in body
