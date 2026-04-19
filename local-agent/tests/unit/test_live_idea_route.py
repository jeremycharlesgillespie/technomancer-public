"""Tests verifying the /live/<idea_id> route is registered and callable (TK-780).

Acceptance criteria:
- Route /live/<idea_id> is defined in idea_board/web.py
- Route returns HTTP 200 or 404 (not 500) for valid/invalid idea IDs
- Unit test verifies route handler is callable
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from idea_board.web import app, live_log_viewer


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _empty_ideas():
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


class TestLiveIdeaRouteRegistered:
    def test_route_handler_is_importable(self):
        assert callable(live_log_viewer)

    def test_route_is_registered_in_app(self):
        rules = {r.rule for r in app.url_map.iter_rules()}
        assert "/live/<item_id>" in rules

    def test_known_jira_key_returns_200(self, client):
        resp = client.get("/live/TK-780")
        assert resp.status_code == 200

    def test_unknown_idea_id_not_500(self, client):
        resp = client.get("/live/idea-does-not-exist-99999")
        assert resp.status_code in (200, 404)

    def test_response_is_html(self, client):
        resp = client.get("/live/TK-780")
        assert "text/html" in resp.content_type

    def test_response_includes_idea_id(self, client):
        resp = client.get("/live/TK-780")
        body = resp.get_data(as_text=True)
        assert "TK-780" in body
