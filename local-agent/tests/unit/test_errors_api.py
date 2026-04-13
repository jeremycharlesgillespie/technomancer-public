"""
Tests for the /api/errors and /errors endpoints in idea_board/web.py.

Validates crash log parsing, JSON response structure, and HTML rendering.
"""

import html
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import _parse_crash_log, app

SAMPLE_CRASH_LOG = """# Bot Crash Report

**Timestamp:** 2026-04-13 14:30:45
**Exception Type:** UnboundLocalError
**Exception Message:** cannot access local variable 'get_recent_summaries'

## Full Stack Trace
```python
Traceback (most recent call last):
  File "agent/discord_memory_bot.py", line 1198, in on_message
    if "no conversation summaries" not in past_summaries:
UnboundLocalError: cannot access local variable 'get_recent_summaries'
```

## Local Variables by Frame

### Frame 0: on_message (agent/discord_memory_bot.py:1198)
```python
message = <Message id=123>
context_tier = 'standard'
```
"""

MULTI_CRASH_LOG = """# Bot Crash Report

**Timestamp:** 2026-04-13 10:00:00
**Exception Type:** KeyError
**Exception Message:** 'ollama'

## Full Stack Trace
```python
Traceback (most recent call last):
  File "agent/core.py", line 50, in run
    client = config['ollama']
KeyError: 'ollama'
```

## Local Variables by Frame

### Frame 0: run (agent/core.py:50)
```python
config = {}
```
# Bot Crash Report

**Timestamp:** 2026-04-13 14:30:45
**Exception Type:** ValueError
**Exception Message:** invalid literal for int()

## Full Stack Trace
```python
Traceback (most recent call last):
  File "agent/tools.py", line 22, in parse_input
    val = int(user_input)
ValueError: invalid literal for int()
```

## Local Variables by Frame

### Frame 0: parse_input (agent/tools.py:22)
```python
user_input = 'abc'
```
"""


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestParseCrashLog:
    """Tests for the _parse_crash_log helper function."""

    def test_returns_empty_when_no_file(self, tmp_path):
        """Returns empty list when crash_log.md doesn't exist."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert result == []

    def test_returns_empty_for_empty_file(self, tmp_path):
        """Returns empty list when crash_log.md is empty."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text("", encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert result == []

    def test_parses_single_crash(self, tmp_path):
        """Parses a single crash entry correctly."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert len(result) == 1
            entry = result[0]
            assert entry["timestamp"] == "2026-04-13 14:30:45"
            assert entry["exception_type"] == "UnboundLocalError"
            assert "get_recent_summaries" in entry["exception_message"]
            assert "UnboundLocalError" in entry["stack_trace"]
            assert entry["summary"] != ""

    def test_parses_multiple_crashes(self, tmp_path):
        """Parses multiple crash entries and returns newest first."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(MULTI_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert len(result) == 2
            # Newest first (reversed order)
            assert result[0]["exception_type"] == "ValueError"
            assert result[1]["exception_type"] == "KeyError"

    def test_extracts_stack_trace(self, tmp_path):
        """Stack trace is extracted from the code block."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            trace = result[0]["stack_trace"]
            assert "Traceback (most recent call last):" in trace
            assert "on_message" in trace

    def test_extracts_local_variables(self, tmp_path):
        """Local variables section is captured."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert "Local Variables by Frame" in result[0]["local_variables"]

    def test_max_10_entries(self, tmp_path):
        """Returns at most 10 entries."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        # Create 12 crash entries
        parts = []
        for i in range(12):
            parts.append(f"""# Bot Crash Report

**Timestamp:** 2026-04-13 {i:02d}:00:00
**Exception Type:** Error{i}
**Exception Message:** msg {i}

## Full Stack Trace
```python
Traceback: error {i}
```
""")
        (crash_dir / "crash_log.md").write_text("\n".join(parts), encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            result = _parse_crash_log()
            assert len(result) == 10


class TestErrorsEndpoint:
    """Tests for GET /api/errors."""

    def test_returns_200(self, client, tmp_path):
        """Errors endpoint returns 200."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
            assert resp.status_code == 200

    def test_response_structure(self, client, tmp_path):
        """Response has errors list and count."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
            data = resp.get_json()
            assert "errors" in data
            assert "count" in data
            assert isinstance(data["errors"], list)
            assert data["count"] == 0

    def test_returns_parsed_entries(self, client, tmp_path):
        """Returns parsed crash entries when file exists."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
            data = resp.get_json()
            assert data["count"] == 1
            entry = data["errors"][0]
            assert entry["exception_type"] == "UnboundLocalError"
            assert entry["timestamp"] == "2026-04-13 14:30:45"

    def test_entry_has_required_fields(self, client, tmp_path):
        """Each error entry has all required fields."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
            data = resp.get_json()
            entry = data["errors"][0]
            for field in ["timestamp", "exception_type", "exception_message",
                          "stack_trace", "summary", "local_variables"]:
                assert field in entry, f"Missing field: {field}"


class TestErrorsPage:
    """Tests for GET /errors HTML page."""

    def test_returns_200(self, client, tmp_path):
        """Errors page returns 200."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            assert resp.status_code == 200

    def test_empty_state_when_no_crashes(self, client, tmp_path):
        """Shows empty state message when no crashes exist."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            page_html = resp.data.decode()
            assert "No crash reports found" in page_html
            assert "empty-state" in page_html

    def test_shows_crash_entry(self, client, tmp_path):
        """Renders crash entry as collapsible card."""
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            page_html = resp.data.decode()
            assert "error-card" in page_html
            assert "UnboundLocalError" in page_html
            assert "2026-04-13 14:30:45" in page_html
            assert "error-detail" in page_html
            assert "toggleError" in page_html

    def test_has_back_link_to_hub(self, client, tmp_path):
        """Page has navigation link back to hub."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            page_html = resp.data.decode()
            assert 'href="/"' in page_html
            assert "Hub" in page_html

    def test_has_api_link(self, client, tmp_path):
        """Page links to the JSON API."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            page_html = resp.data.decode()
            assert "/api/errors" in page_html

    def test_mobile_viewport_meta(self, client, tmp_path):
        """Page includes mobile viewport meta tag."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
            page_html = resp.data.decode()
            assert 'name="viewport"' in page_html
            assert "width=device-width" in page_html


class TestHubErrorsLink:
    """Tests that the hub page links to the errors page."""

    def test_hub_has_errors_card(self, client):
        """Hub page contains an Errors & Crashes card."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")):
            mock_run.return_value = MagicMock(stdout="abc1234")
            resp = client.get("/")
            page_html = resp.data.decode()
            assert 'href="/errors"' in page_html
            assert "Errors" in page_html

    def test_health_panel_links_to_errors(self, client):
        """Health panel header has a link to the errors page."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")):
            mock_run.return_value = MagicMock(stdout="abc1234")
            resp = client.get("/")
            page_html = resp.data.decode()
            assert "View errors" in page_html
