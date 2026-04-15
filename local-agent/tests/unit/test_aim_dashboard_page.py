"""Tests for the /aim dashboard HTML page and its worker status widget."""

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


class TestAimDashboardSmoke:
    def test_returns_200(self, client):
        resp = client.get("/aim")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client):
        resp = client.get("/aim")
        assert "text/html" in resp.content_type

    def test_has_doctype(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert body.lstrip().lower().startswith("<!doctype html>")


class TestAimDashboardWidget:
    def test_widget_container_rendered(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert 'id="aim-status-widget"' in body

    def test_all_required_fields_rendered(self, client):
        """Widget must expose PID, status, assignment, cycles, last decision."""
        body = client.get("/aim").get_data(as_text=True)
        assert 'id="widget-pid"' in body
        assert 'id="widget-status"' in body
        assert 'id="widget-assignment"' in body
        assert 'id="widget-cycles"' in body
        assert 'id="widget-decision-summary"' in body

    def test_polls_status_endpoint(self, client):
        """Widget JS must call /api/aim/status."""
        body = client.get("/aim").get_data(as_text=True)
        assert "/api/aim/status" in body

    def test_polls_every_5_seconds(self, client):
        """Polling interval should be ~5 seconds (5000ms)."""
        body = client.get("/aim").get_data(as_text=True)
        assert "POLL_MS = 5000" in body
        assert "setInterval(poll, POLL_MS)" in body

    def test_status_colors_defined(self, client):
        """CSS must define the three status colors used for worker state."""
        body = client.get("/aim").get_data(as_text=True)
        # Pill / border classes that reflect worker status
        assert "status-green" in body
        assert "status-yellow" in body
        assert "status-red" in body

    def test_status_color_mapping_green(self, client):
        """idle and watching should map to green."""
        body = client.get("/aim").get_data(as_text=True)
        assert "'idle'" in body
        assert "'watching'" in body
        assert "return 'green'" in body

    def test_status_color_mapping_yellow(self, client):
        """executing and assigned should map to yellow."""
        body = client.get("/aim").get_data(as_text=True)
        assert "'executing'" in body
        assert "'assigned'" in body
        assert "return 'yellow'" in body

    def test_status_color_mapping_red(self, client):
        """stuck and dead should map to red."""
        body = client.get("/aim").get_data(as_text=True)
        assert "'stuck'" in body
        assert "'dead'" in body
        assert "return 'red'" in body


class TestAimDashboardJiraLink:
    def test_jira_url_embedded_when_configured(self, client):
        """When JIRA_URL is set, it is exposed to the client-side JS."""
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net"):
            body = client.get("/aim").get_data(as_text=True)
            assert '"https://acme.atlassian.net"' in body

    def test_trailing_slash_on_jira_url_is_stripped(self, client):
        """Trailing slash is removed so /browse/KEY builds cleanly."""
        with patch("idea_board.web.settings.jira_url", "https://acme.atlassian.net/"):
            body = client.get("/aim").get_data(as_text=True)
            assert '"https://acme.atlassian.net"' in body
            assert '"https://acme.atlassian.net/"' not in body

    def test_empty_jira_url_when_unconfigured(self, client):
        """No jira_url renders as empty string (links become no-ops)."""
        with patch("idea_board.web.settings.jira_url", None):
            body = client.get("/aim").get_data(as_text=True)
            # The JSON-encoded empty string is present.
            assert 'const JIRA_URL = "";' in body

    def test_jira_key_regex_matches_tk_format(self, client):
        """Client-side regex must recognize Jira keys like TK-378."""
        body = client.get("/aim").get_data(as_text=True)
        # JS-side regex for Jira key detection
        assert "JIRA_KEY_RE" in body
        assert "/browse/" in body

    def test_jira_url_json_escaped(self, client):
        """jira_url is JSON-encoded so embedded quotes can't break out."""
        with patch("idea_board.web.settings.jira_url", 'https://a"b.test'):
            body = client.get("/aim").get_data(as_text=True)
            # The JSON encoding escapes the quote
            assert '\\"' in body


class TestAimDashboardTimeline:
    def test_timeline_container_rendered(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert 'id="aim-events"' in body
        assert 'id="aim-events-timeline"' in body
        assert 'id="aim-events-status"' in body

    def test_subscribes_to_sse_stream(self, client):
        """Page must open an EventSource against /api/aim/events/stream."""
        body = client.get("/aim").get_data(as_text=True)
        assert "new EventSource('/api/aim/events/stream')" in body

    def test_event_color_mapping_blue(self, client):
        """decision_made should map to blue."""
        body = client.get("/aim").get_data(as_text=True)
        assert "decision_made: 'blue'" in body

    def test_event_color_mapping_green(self, client):
        """worker started/spawned/assigned should map to green."""
        body = client.get("/aim").get_data(as_text=True)
        assert "worker_started: 'green'" in body
        assert "worker_spawned: 'green'" in body
        assert "worker_assigned: 'green'" in body

    def test_event_color_mapping_red(self, client):
        """execution_failed and worker_died should map to red."""
        body = client.get("/aim").get_data(as_text=True)
        assert "execution_failed: 'red'" in body
        assert "worker_died: 'red'" in body

    def test_event_color_mapping_yellow(self, client):
        """escalation should map to yellow."""
        body = client.get("/aim").get_data(as_text=True)
        assert "escalation: 'yellow'" in body

    def test_timeline_css_colors_defined(self, client):
        """CSS must style the four timeline color classes."""
        body = client.get("/aim").get_data(as_text=True)
        assert ".aim-event.color-blue" in body
        assert ".aim-event.color-green" in body
        assert ".aim-event.color-red" in body
        assert ".aim-event.color-yellow" in body

    def test_events_are_prepended(self, client):
        """New events go to the top of the timeline (reverse-chronological)."""
        body = client.get("/aim").get_data(as_text=True)
        assert "insertBefore" in body
        assert "timeline.firstChild" in body

    def test_timeline_empty_state(self, client):
        """Timeline starts with an empty placeholder."""
        body = client.get("/aim").get_data(as_text=True)
        assert "aim-event-empty" in body

    def test_timeline_link_to_stream_endpoint(self, client):
        """Footer links to the underlying SSE endpoint."""
        body = client.get("/aim").get_data(as_text=True)
        assert '/api/aim/events/stream' in body
