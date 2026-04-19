"""Unit tests for the View Live Executions card on the hub home page (TK-777).

Acceptance criteria:
- Home page renders the new card in the grid
- Card has href="/live" link
- Home page renders without errors or timeouts
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _git_mock():
    m = MagicMock()
    m.stdout = "abc1234\n"
    m.returncode = 0
    return m


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    monkeypatch.setattr("idea_board.web.subprocess.run", lambda *a, **kw: _git_mock())
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("no network")),
    )


class TestLiveExecutionsCard:
    def test_home_page_returns_200(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        assert resp.status_code == 200

    def test_card_present_in_grid(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        body = resp.get_data(as_text=True)
        assert "View Live Executions" in body

    def test_card_href_links_to_live(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        body = resp.get_data(as_text=True)
        assert 'href="/live"' in body

    def test_home_renders_without_error(self, client):
        with patch("idea_board.web.load_ideas", return_value=[]):
            resp = client.get("/")
        assert "text/html" in resp.content_type
        body = resp.get_data(as_text=True)
        assert len(body) > 100
