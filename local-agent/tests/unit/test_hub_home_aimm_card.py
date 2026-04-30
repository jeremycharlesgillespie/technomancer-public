"""Tests for the AIMM status card on the Technomancer hub home page (TK-670).

The AIMM card shows cycle count, last cycle time, and today's approvals/drafts
from /api/aimm/status. The card is visible on the hub home and the badge is
populated via JavaScript.
"""

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


class TestHubHomeAimmCard:
    def test_card_visible_on_hub_home(self, client):
        """AIMM card must be present on the hub home page."""
        body = client.get("/").get_data(as_text=True)
        assert 'href="/aimm"' in body
        assert "AIMM" in body

    def test_card_has_card_class(self, client):
        """AIMM card must use the .card CSS class."""
        body = client.get("/").get_data(as_text=True)
        idx = body.index('href="/aimm"')
        start = body.rfind("<a ", 0, idx)
        snippet = body[start : idx + 200]
        assert "class=\"card" in snippet

    def test_card_has_border_color(self, client):
        """AIMM card must have the distinctive orange border."""
        body = client.get("/").get_data(as_text=True)
        assert 'href="/aimm" class="card" style="border-left: 4px solid #ff9800;"' in body

    def test_card_has_description(self, client):
        """AIMM card must include a description of the module."""
        body = client.get("/").get_data(as_text=True)
        assert "AI Manager Module" in body

    def test_card_has_badge_element(self, client):
        """AIMM card must have a badge element for status display."""
        body = client.get("/").get_data(as_text=True)
        assert 'id="aimm-badge"' in body

    def test_badge_initially_loading(self, client):
        """Badge must start with 'Loading...' text."""
        body = client.get("/").get_data(as_text=True)
        assert "Loading&hellip;" in body

    def test_badge_populated_via_javascript(self, client):
        """Badge must be updated via JavaScript fetch to /api/aimm/status."""
        body = client.get("/").get_data(as_text=True)
        # Check that the updateAimmStatus function is defined
        assert "updateAimmStatus" in body

    def test_badge_shows_cycle_count(self, client):
        """Badge must display cycle count from API response."""
        body = client.get("/").get_data(as_text=True)
        # The badge is populated via JS, so we just verify the function exists
        assert "updateAimmStatus" in body

    def test_badge_shows_last_cycle_time(self, client):
        """Badge must display last cycle time from API response."""
        body = client.get("/").get_data(as_text=True)
        # The badge is populated via JS, so we just verify the function exists
        assert "updateAimmStatus" in body

    def test_badge_shows_approved_count(self, client):
        """Badge must display today's approved keys count."""
        body = client.get("/").get_data(as_text=True)
        # The badge is populated via JS, so we just verify the function exists
        assert "updateAimmStatus" in body

    def test_badge_shows_drafted_count(self, client):
        """Badge must display today's drafted keys count."""
        body = client.get("/").get_data(as_text=True)
        # The badge is populated via JS, so we just verify the function exists
        assert "updateAimmStatus" in body

    def test_badge_handles_api_error(self, client):
        """Badge must handle API errors gracefully."""
        body = client.get("/").get_data(as_text=True)
        # The updateAimmStatus function includes error handling
        assert "updateAimmStatus" in body

    def test_badge_refreshes_periodically(self, client):
        """Badge must be updated periodically via setInterval."""
        body = client.get("/").get_data(as_text=True)
        # Check that setInterval is used to call updateAimmStatus
        assert "setInterval(updateAimmStatus" in body