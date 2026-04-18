"""Tests for the conditional /errors card on the Technomancer hub home (TK-650).

The /errors card is hidden from the hub homepage when the 7-day crash
count is zero, and visible when it's greater than zero. The /errors
endpoint itself must stay reachable regardless.
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


def _stats(counts_7d: int) -> dict:
    """Build a crash-log-stats dict shaped like _crash_log_stats() output."""
    return {
        "total": counts_7d,
        "counts_24h": 0,
        "counts_7d": counts_7d,
        "counts_30d": counts_7d,
        "last_mtime": None,
        "last_check": "never",
        "file_exists": counts_7d > 0,
        "days_since_last_crash": 0 if counts_7d > 0 else None,
    }


class TestErrorsCardVisibility:
    def test_card_hidden_when_7d_crashes_zero(self, client):
        with patch("idea_board.web._crash_log_stats", return_value=_stats(0)):
            body = client.get("/").get_data(as_text=True)
        assert 'href="/errors" class="card"' not in body
        assert "Errors &amp; Crashes" not in body

    def test_card_visible_when_7d_crashes_positive(self, client):
        with patch("idea_board.web._crash_log_stats", return_value=_stats(3)):
            body = client.get("/").get_data(as_text=True)
        assert 'href="/errors" class="card"' in body
        assert "Errors &amp; Crashes" in body
        assert "3 in 7d" in body

    def test_card_visible_when_7d_crashes_is_one(self, client):
        with patch("idea_board.web._crash_log_stats", return_value=_stats(1)):
            body = client.get("/").get_data(as_text=True)
        assert 'href="/errors" class="card"' in body
        assert "1 in 7d" in body

    def test_stats_exception_hides_card(self, client):
        """If _crash_log_stats raises, the card is hidden (fail-safe)."""
        with patch(
            "idea_board.web._crash_log_stats", side_effect=RuntimeError("boom")
        ):
            body = client.get("/").get_data(as_text=True)
        assert 'href="/errors" class="card"' not in body

    def test_errors_endpoint_accessible_when_card_hidden(self, client):
        """The /errors URL stays reachable even when the hub card is hidden."""
        with patch("idea_board.web._crash_log_stats", return_value=_stats(0)):
            # Hub home omits the grid card...
            hub_body = client.get("/").get_data(as_text=True)
            assert 'href="/errors" class="card"' not in hub_body
            # ...but the /errors page itself still renders.
            resp = client.get("/errors")
        assert resp.status_code == 200

    def test_service_health_header_errors_link_always_visible(self, client):
        """The small 'View errors →' link in the Service Health header stays,
        so /errors remains discoverable even when the grid card is gone."""
        with patch("idea_board.web._crash_log_stats", return_value=_stats(0)):
            body = client.get("/").get_data(as_text=True)
        assert 'href="/errors"' in body  # header link, not the card
        assert "View errors" in body
