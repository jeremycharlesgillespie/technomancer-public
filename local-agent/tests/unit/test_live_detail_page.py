"""Tests for the /live/<idea_id> detail page route (TK-776).

Acceptance criteria:
- GET /live/<idea_id> returns 200 with execution details (idea name, status, log output)
- Route handles invalid idea_id gracefully (not 500)
- Unit tests verify route renders without errors
"""

from __future__ import annotations

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestLiveDetailPageReturns200:
    """Route returns 200 for any idea_id (valid or unknown)."""

    def test_jira_key_returns_200(self, client):
        resp = client.get("/live/TK-776")
        assert resp.status_code == 200

    def test_local_idea_id_returns_200(self, client):
        resp = client.get("/live/idea-123")
        assert resp.status_code == 200

    def test_unknown_id_returns_200_not_500(self, client):
        resp = client.get("/live/nonexistent-idea-9999")
        assert resp.status_code in (200, 404)

    def test_content_type_is_html(self, client):
        resp = client.get("/live/TK-776")
        assert "text/html" in resp.content_type


class TestLiveDetailPageContents:
    """Page body includes the idea id, a status indicator, and log container."""

    def test_page_includes_idea_name(self, client):
        resp = client.get("/live/TK-776")
        assert "TK-776" in resp.data.decode()

    def test_page_includes_status_element(self, client):
        resp = client.get("/live/TK-776")
        body = resp.data.decode()
        assert "status" in body.lower()

    def test_page_includes_log_container(self, client):
        resp = client.get("/live/TK-776")
        body = resp.data.decode()
        assert "log" in body.lower()

    def test_page_connects_to_log_stream(self, client):
        """The page wires up the SSE log stream endpoint for the given idea_id."""
        resp = client.get("/live/TK-42")
        body = resp.data.decode()
        assert "TK-42" in body
        assert "/api/ideas/TK-42/log/stream" in body

    def test_different_ids_produce_different_pages(self, client):
        body_a = client.get("/live/TK-1").data.decode()
        body_b = client.get("/live/TK-2").data.decode()
        assert "TK-1" in body_a
        assert "TK-2" in body_b
        assert "TK-2" not in body_a
        assert "TK-1" not in body_b
