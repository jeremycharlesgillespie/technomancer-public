"""Tests for the AI Dev Team Dashboard page at /aim/dashboard."""

from __future__ import annotations

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def body(client):
    """HTML body of /aim/dashboard as text."""
    return client.get("/aim/dashboard").get_data(as_text=True)


class TestSmoke:
    def test_returns_200(self, client):
        resp = client.get("/aim/dashboard")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client):
        resp = client.get("/aim/dashboard")
        assert "text/html" in resp.content_type

    def test_has_doctype(self, body):
        assert body.lstrip().lower().startswith("<!doctype html>")


class TestEndpointsReferenced:
    def test_references_metrics_endpoint(self, body):
        assert "/api/aim/metrics" in body

    def test_references_backlog_endpoint(self, body):
        assert "/api/aim/backlog" in body

    def test_references_commits_endpoint(self, body):
        assert "/api/aim/commits" in body


class TestCharts:
    def test_includes_chart_js_cdn(self, body):
        assert "cdn.jsdelivr.net/npm/chart.js" in body

    def test_includes_canvas_elements(self, body):
        assert "<canvas" in body

    def test_includes_completions_chart_canvas(self, body):
        assert 'id="completions-chart"' in body

    def test_includes_success_failure_chart_canvas(self, body):
        assert 'id="success-failure-chart"' in body


class TestDarkMode:
    def test_has_style_block(self, body):
        assert "<style>" in body
        assert "</style>" in body

    def test_dark_background(self, body):
        # Dark mode palette — background color defined in :root
        assert "--bg: #1a1a1a" in body


class TestWindowSelector:
    def test_selector_container_rendered(self, body):
        assert 'id="window-selector"' in body

    def test_has_all_four_window_buttons(self, body):
        assert 'data-hours="1"' in body
        assert 'data-hours="6"' in body
        assert 'data-hours="24"' in body
        assert 'data-hours="168"' in body

    def test_button_labels_present(self, body):
        assert ">1h<" in body
        assert ">6h<" in body
        assert ">24h<" in body
        assert ">7d<" in body

    def test_24h_is_default_active(self, body):
        # The 24h button ships with the .active class so metrics fire at 24h
        # on first load (matches Decision 4 default window).
        assert 'data-hours="24" class="active"' in body


class TestLayoutBands:
    def test_current_work_band_rendered(self, body):
        assert 'id="current-work"' in body
        assert 'id="current-work-body"' in body

    def test_backlog_counts_band_rendered(self, body):
        assert 'id="backlog-counts"' in body
        assert 'id="count-todo"' in body
        assert 'id="count-in-progress"' in body
        assert 'id="count-done-today"' in body
        assert 'id="count-failed-today"' in body
        assert 'id="count-veto"' in body

    def test_charts_band_rendered(self, body):
        assert 'id="charts"' in body

    def test_commits_band_rendered(self, body):
        assert 'id="commits"' in body
        assert 'id="private-commits"' in body
        assert 'id="public-commits"' in body


class TestPollingIntervals:
    def test_backlog_polls_every_10s(self, body):
        assert "BACKLOG_POLL_MS = 10000" in body

    def test_commits_polls_every_30s(self, body):
        assert "COMMITS_POLL_MS = 30000" in body

    def test_metrics_polls_every_60s(self, body):
        assert "METRICS_POLL_MS = 60000" in body


class TestCurrentWorkLinksToLive:
    def test_links_in_progress_key_to_live_log(self, body):
        # Clicking the current-work key drills into the live log viewer.
        assert "'/live/' + encodeURIComponent(inProgress.key)" in body
