"""Integration tests for GET /errors and GET /api/errors.

Exercises the full Flask route end-to-end: writes a real ``crash_log.md``
under a temp vault, hits ``/errors`` and ``/api/errors`` via the Flask test
client, and asserts the rendered HTML / JSON contain the expected
notification rows, navigation links, timestamps, and empty-state copy.

These complement the unit tests in ``tests/unit/test_errors_api.py`` by
running through the full request lifecycle (URL routing + template
rendering + crash-log parsing) instead of calling the helpers directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board.web import app


SINGLE_CRASH = """# Bot Crash Report

**Timestamp:** 2026-04-13 14:30:45
**Exception Type:** UnboundLocalError
**Exception Message:** cannot access local variable 'foo'

## Full Stack Trace
```python
Traceback (most recent call last):
  File "agent/discord_memory_bot.py", line 1198, in on_message
    foo()
UnboundLocalError: cannot access local variable 'foo'
```

## Local Variables by Frame

### Frame 0: on_message (agent/discord_memory_bot.py:1198)
```python
message = <Message id=42>
```
"""


def _write_crash_log(vault_root: Path, body: str) -> Path:
    """Drop ``body`` into ``<vault_root>/LLM Memory/Permanent/crash_log.md``."""
    crash_file = vault_root / "LLM Memory" / "Permanent" / "crash_log.md"
    crash_file.parent.mkdir(parents=True, exist_ok=True)
    crash_file.write_text(body, encoding="utf-8")
    return crash_file


def _multi_crash_body(timestamps: list[datetime]) -> str:
    """Build a crash_log.md body with one entry per timestamp."""
    parts = []
    for i, ts in enumerate(timestamps):
        parts.append(
            "# Bot Crash Report\n\n"
            f"**Timestamp:** {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Exception Type:** SampleError{i}\n"
            f"**Exception Message:** sample message {i}\n\n"
            "## Full Stack Trace\n"
            "```python\n"
            f"Traceback: error {i}\n"
            "```\n"
        )
    return "\n".join(parts)


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestErrorsEndpointWithNotifications:
    """Story TK-653 — populated /errors page renders clickable notification rows."""

    def test_returns_200_with_crash_entries(self, client, tmp_path):
        """Endpoint returns 200 OK when crash_log.md has entries."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_renders_clickable_error_card(self, client, tmp_path):
        """Each crash entry renders as a card with a click-to-expand handler."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="error-card"' in page_html
        assert 'class="error-header" onclick="toggleError(0)"' in page_html
        assert 'id="detail-0"' in page_html
        assert 'id="toggle-0"' in page_html

    def test_includes_navigation_links(self, client, tmp_path):
        """Hub and API navigation hrefs are clickable links in the response."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'href="/"' in page_html
        assert 'href="/api/errors"' in page_html

    def test_link_format_includes_timestamp(self, client, tmp_path):
        """The rendered card contains the entry's timestamp string verbatim."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "2026-04-13 14:30:45" in page_html
        assert "UnboundLocalError" in page_html

    def test_multiple_notifications_each_get_unique_toggle(self, client, tmp_path):
        """Distinct card ids per entry so the toggle script can target each one."""
        now = datetime.now()
        body = _multi_crash_body([
            now - timedelta(hours=1),
            now - timedelta(hours=2),
            now - timedelta(hours=3),
        ])
        _write_crash_log(tmp_path, body)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # Three error-card divs, each with its own onclick index + detail id.
        assert page_html.count('class="error-card"') == 3
        assert 'onclick="toggleError(0)"' in page_html
        assert 'onclick="toggleError(1)"' in page_html
        assert 'onclick="toggleError(2)"' in page_html
        assert 'id="detail-0"' in page_html
        assert 'id="detail-1"' in page_html
        assert 'id="detail-2"' in page_html

    def test_summary_bar_renders_when_notifications_present(self, client, tmp_path):
        """Page header still includes the per-window summary bar."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "summary-bar" in page_html
        assert "crashes in 24h" in page_html
        assert "in 7d" in page_html
        assert "in 30d" in page_html


class TestErrorsEndpointWithoutNotifications:
    """Story TK-653 — empty /errors page renders the reassuring empty state."""

    def test_returns_200_with_no_crash_log(self, client, tmp_path):
        """Endpoint returns 200 OK even when crash_log.md does not exist."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_renders_empty_state_message(self, client, tmp_path):
        """Body contains the empty-state copy + 'no crashes' headline."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="empty-state"' in page_html
        assert "never written a crash report" in page_html
        # Empty state should NOT render any error-card elements.
        assert 'class="error-card"' not in page_html

    def test_empty_state_keeps_navigation_links(self, client, tmp_path):
        """Hub and API hrefs are still present so users can navigate away."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'href="/"' in page_html
        assert 'href="/api/errors"' in page_html

    def test_empty_state_renders_discord_channel_link(self, client, tmp_path):
        """When ``discord_alerts_channel`` is set, the empty state names it.

        The Discord channel reference is the user-visible pointer to where new
        crash notifications will land — this is what the story description
        calls the 'Discord message URL' in the empty state.
        """
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            mock_settings.discord_alerts_channel = "bot_alerts"
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "#bot_alerts" in page_html
        assert "Crash alerts post to" in page_html

    def test_empty_state_renders_last_checked_timestamp(self, client, tmp_path):
        """Empty state always shows a 'Last checked' line (TK-648)."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="last-checked">Last checked:' in page_html

    def test_empty_state_renders_per_window_counts(self, client, tmp_path):
        """Empty state still surfaces the 24h / 7d / 30d count chips."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="empty-counts"' in page_html
        assert page_html.count('class="count-chip"') == 3
        assert "in 24h" in page_html
        assert "in 7d" in page_html
        assert "in 30d" in page_html


class TestApiErrorsEndpointAlignment:
    """JSON API mirrors the same data the HTML page renders."""

    def test_api_returns_200_when_empty(self, client, tmp_path):
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 0
        assert data["errors"] == []

    def test_api_returns_entries_with_timestamp(self, client, tmp_path):
        """API response carries the same timestamp the HTML page renders."""
        _write_crash_log(tmp_path, SINGLE_CRASH)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        entry = data["errors"][0]
        assert entry["timestamp"] == "2026-04-13 14:30:45"
        assert entry["exception_type"] == "UnboundLocalError"
