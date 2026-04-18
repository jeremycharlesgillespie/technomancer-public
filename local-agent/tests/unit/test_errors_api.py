"""
Tests for the /api/errors and /errors endpoints in idea_board/web.py.

Validates crash log parsing, JSON response structure, and HTML rendering.
"""

import html
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.web import (
    _crash_log_stats,
    _format_relative_time,
    _parse_crash_log,
    app,
)

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
            # Redesigned empty state: page reads as "working", not "broken".
            assert "empty-state" in page_html
            assert "never written a crash report" in page_html

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
        """Hub page contains an Errors & Crashes card when 7d crashes > 0.

        The card is hidden when the 7-day crash count is zero (TK-650); this
        test forces a non-zero count so the card renders and the link is
        asserted.
        """
        stats = {
            "total": 2,
            "counts_24h": 0,
            "counts_7d": 2,
            "counts_30d": 2,
            "last_mtime": None,
            "last_check": "5m ago",
            "file_exists": True,
            "days_since_last_crash": 0,
        }
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")), \
             patch("idea_board.web._crash_log_stats", return_value=stats):
            mock_run.return_value = MagicMock(stdout="abc1234")
            resp = client.get("/")
            page_html = resp.data.decode()
            assert 'href="/errors" class="card"' in page_html
            assert "Errors &amp; Crashes" in page_html

    def test_health_panel_links_to_errors(self, client):
        """Health panel header has a link to the errors page."""
        with patch("idea_board.web.subprocess.run") as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("conn refused")):
            mock_run.return_value = MagicMock(stdout="abc1234")
            resp = client.get("/")
            page_html = resp.data.decode()
            assert "View errors" in page_html


def _write_crash_entries(crash_file: Path, timestamps: list[datetime]) -> None:
    """Write a crash_log.md with one entry per timestamp."""
    parts = []
    for i, ts in enumerate(timestamps):
        parts.append(
            "# Bot Crash Report\n\n"
            f"**Timestamp:** {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Exception Type:** Error{i}\n"
            f"**Exception Message:** msg {i}\n\n"
            "## Full Stack Trace\n"
            "```python\nTraceback: error\n```\n"
        )
    crash_file.parent.mkdir(parents=True, exist_ok=True)
    crash_file.write_text("\n".join(parts), encoding="utf-8")


class TestFormatRelativeTime:
    """Tests for the _format_relative_time helper."""

    def test_seconds(self):
        assert _format_relative_time(timedelta(seconds=30)) == "just now"

    def test_minutes(self):
        assert _format_relative_time(timedelta(minutes=2)) == "2m ago"

    def test_hours(self):
        assert _format_relative_time(timedelta(hours=3)) == "3h ago"

    def test_days(self):
        assert _format_relative_time(timedelta(days=5)) == "5d ago"

    def test_negative_delta_is_just_now(self):
        """Clock skew should not produce weird output."""
        assert _format_relative_time(timedelta(seconds=-10)) == "just now"


class TestCrashLogStats:
    """Tests for the _crash_log_stats helper function."""

    def test_no_file_returns_zeros(self, tmp_path):
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            stats = _crash_log_stats()
        assert stats["total"] == 0
        assert stats["counts_24h"] == 0
        assert stats["counts_7d"] == 0
        assert stats["counts_30d"] == 0
        assert stats["file_exists"] is False
        assert stats["last_check"] == "never"

    def test_empty_file_returns_zeros_but_exists(self, tmp_path):
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        crash_file.parent.mkdir(parents=True)
        crash_file.write_text("", encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            stats = _crash_log_stats()
        assert stats["total"] == 0
        assert stats["file_exists"] is True
        assert stats["last_check"] != "never"

    def test_per_window_counts(self, tmp_path):
        """Each window slices strictly on the 24h/7d/30d cutoffs."""
        now = datetime.now()
        timestamps = [
            now - timedelta(hours=1),    # in 24h, 7d, 30d
            now - timedelta(hours=12),   # in 24h, 7d, 30d
            now - timedelta(days=2),     # in 7d, 30d only
            now - timedelta(days=10),    # in 30d only
            now - timedelta(days=40),    # out of all windows
        ]
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, timestamps)
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            stats = _crash_log_stats()
        assert stats["total"] == 5
        assert stats["counts_24h"] == 2
        assert stats["counts_7d"] == 3
        assert stats["counts_30d"] == 4

    def test_last_check_reflects_mtime(self, tmp_path):
        """last_check is derived from the file's mtime, not entry timestamps."""
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [datetime.now() - timedelta(days=5)])
        # Pin mtime to ~2 minutes ago so we can assert the relative string.
        target = (datetime.now() - timedelta(minutes=2)).timestamp()
        os.utime(crash_file, (target, target))
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            stats = _crash_log_stats()
        assert stats["last_check"] == "2m ago"
        assert stats["last_mtime"] is not None

    def test_unparseable_timestamps_are_skipped(self, tmp_path):
        """Entries with malformed timestamps do not crash or inflate counts."""
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        crash_file.parent.mkdir(parents=True)
        crash_file.write_text(
            "# Bot Crash Report\n\n"
            "**Timestamp:** not a date\n"
            "**Exception Type:** X\n",
            encoding="utf-8",
        )
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            stats = _crash_log_stats()
        assert stats["total"] == 0
        assert stats["counts_30d"] == 0


class TestErrorsEmptyStateRedesign:
    """TK-543: /errors empty state shows a summary header, not a broken page."""

    def test_empty_state_shows_zero_counts_header(self, client, tmp_path):
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # Summary bar is present with all three windows.
        assert "summary-bar" in page_html
        assert "crashes in 24h" in page_html
        assert "in 7d" in page_html
        assert "in 30d" in page_html

    def test_empty_state_headline_reassures(self, client, tmp_path):
        """The zero-crash state should read as 'working', not 'broken'."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "never written a crash report" in page_html

    def test_empty_state_with_old_log_shows_last_check(self, client, tmp_path):
        """When crash_log.md exists but has no recent crashes, show mtime."""
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [datetime.now() - timedelta(days=40)])
        target = (datetime.now() - timedelta(minutes=2)).timestamp()
        os.utime(crash_file, (target, target))
        # _parse_crash_log still returns the old entry (10 newest), so we force
        # an "empty" render by filtering on entries=[] via a tiny monkey-patch.
        with patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web._parse_crash_log", return_value=[]):
            mock_settings.vault_path = tmp_path
            mock_settings.discord_alerts_channel = "bot_alerts"
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "last checked 2m ago" in page_html
        assert "#bot_alerts" in page_html
        # 30d window should show 1 since the only entry is 40d old → counts 0.
        assert ">0</span> in 30d" in page_html

    def test_counts_render_with_real_data(self, client, tmp_path):
        """Summary bar reflects real windowed counts when crashes exist."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [
            now - timedelta(hours=1),   # 24h
            now - timedelta(days=3),    # 7d
            now - timedelta(days=15),   # 30d
        ])
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert ">1</span> crashes in 24h" in page_html
        assert ">2</span> in 7d" in page_html
        assert ">3</span> in 30d" in page_html

    def test_api_errors_includes_stats(self, client, tmp_path):
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [now - timedelta(hours=2)])
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/api/errors")
        data = resp.get_json()
        assert "stats" in data
        assert data["stats"]["counts_24h"] == 1
        assert data["stats"]["counts_7d"] == 1
        assert data["stats"]["counts_30d"] == 1
        assert data["stats"]["total"] == 1


class TestEmptyStateTimestampAndCounts:
    """TK-648: empty state itself shows last-checked timestamp + per-window counts."""

    def test_empty_state_renders_counts_block(self, client, tmp_path):
        """With no crash_log.md, the empty state contains the counts block."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # The counts block lives inside the empty-state div.
        empty_start = page_html.find('<div class="empty-state">')
        empty_end = page_html.find("</div>", page_html.find('class="last-checked"'))
        assert empty_start != -1
        assert empty_end != -1
        empty_html = page_html[empty_start:empty_end]
        assert 'class="empty-counts"' in empty_html
        # All three windows rendered, all showing zero.
        assert empty_html.count('class="count-chip"') == 3
        assert '<span class="num">0</span> in 24h' in empty_html
        assert '<span class="num">0</span> in 7d' in empty_html
        assert '<span class="num">0</span> in 30d' in empty_html

    def test_empty_state_renders_last_checked_when_no_file(self, client, tmp_path):
        """When crash_log.md does not exist, show 'Last checked: never'."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="last-checked">Last checked: never' in page_html

    def test_empty_state_renders_last_checked_with_mtime(self, client, tmp_path):
        """When crash_log.md exists, the empty state shows a relative timestamp."""
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [datetime.now() - timedelta(days=40)])
        target = (datetime.now() - timedelta(minutes=2)).timestamp()
        os.utime(crash_file, (target, target))
        # Force empty cards list even though the file has a (too-old) entry.
        with patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web._parse_crash_log", return_value=[]):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # The last-checked line appears inside the empty-state div.
        empty_start = page_html.find('<div class="empty-state">')
        assert empty_start != -1
        empty_html = page_html[empty_start:]
        assert 'class="last-checked">Last checked: 2m ago' in empty_html

    def test_empty_state_counts_reflect_time_windows(self, client, tmp_path):
        """Count chips reflect real window data even when the entry list is empty."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        # 40d-old entry — drops out of all windows, so 24h=0, 7d=0, 30d=0.
        _write_crash_entries(crash_file, [now - timedelta(days=40)])
        with patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web._parse_crash_log", return_value=[]):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        empty_start = page_html.find('<div class="empty-state">')
        assert empty_start != -1
        empty_html = page_html[empty_start:]
        assert '<span class="num">0</span> in 24h' in empty_html
        assert '<span class="num">0</span> in 7d' in empty_html
        assert '<span class="num">0</span> in 30d' in empty_html

    def test_empty_state_counts_reflect_older_entries(self, client, tmp_path):
        """An entry inside the 30d window shows 30d=1 inside the empty-state block."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [now - timedelta(days=15)])
        # Force empty cards even though a crash exists in the log.
        with patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web._parse_crash_log", return_value=[]):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        empty_start = page_html.find('<div class="empty-state">')
        assert empty_start != -1
        empty_html = page_html[empty_start:]
        assert '<span class="num">0</span> in 24h' in empty_html
        assert '<span class="num">0</span> in 7d' in empty_html
        assert '<span class="num">1</span> in 30d' in empty_html


class TestPillStyleCounters:
    """TK-544: top-of-page counters render as pill-style elements regardless of list content."""

    def test_three_pills_render_with_no_crashes(self, client, tmp_path):
        """Empty /errors still shows the three counter pills."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert page_html.count('class="stat pill zero"') == 3

    def test_pills_render_before_error_cards(self, client, tmp_path):
        """Summary bar with pills appears above the error cards."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [now - timedelta(hours=1)])
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # Look at the DOM nodes, not the embedded CSS rules.
        bar_pos = page_html.find('<div class="summary-bar')
        card_pos = page_html.find('<div class="error-card"')
        assert bar_pos != -1
        assert card_pos != -1
        assert bar_pos < card_pos

    def test_pill_severity_classes_when_crashes_exist(self, client, tmp_path):
        """Each active window is tagged with a distinct severity class."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [
            now - timedelta(hours=1),
            now - timedelta(days=3),
            now - timedelta(days=15),
        ])
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "stat pill severity-high" in page_html
        assert "stat pill severity-medium" in page_html
        assert "stat pill severity-low" in page_html

    def test_pill_zero_class_when_window_quiet(self, client, tmp_path):
        """Only the populated windows get severity classes; quiet ones stay zero."""
        now = datetime.now()
        crash_file = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
        _write_crash_entries(crash_file, [now - timedelta(days=15)])
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert page_html.count('class="stat pill zero"') == 2
        assert "stat pill severity-low" in page_html
        assert "stat pill severity-high" not in page_html
        assert "stat pill severity-medium" not in page_html

    def test_pill_css_uses_border_radius(self, client, tmp_path):
        """Pill styling requires a rounded border-radius on the stat elements."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert ".summary-bar .stat.pill" in page_html
        assert "border-radius: 999px" in page_html

    def test_pills_dropped_middot_separators(self, client, tmp_path):
        """The old `&middot;` separators are gone — pills separate themselves."""
        with patch("idea_board.web.settings") as mock_settings:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert '<span class="sep">' not in page_html


class TestRenderHealthyNotificationsBlock:
    """The /errors empty state links to recent healthy-bot notifications (TK-652)."""

    def test_empty_state_shows_no_notifications_message(self, client, tmp_path):
        """When query returns an empty list, render the 'No notifications found' copy."""
        with patch("idea_board.web.settings") as mock_settings, \
             patch(
                 "idea_board.web.query_healthy_notifications", return_value=[]
             ) as q:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        q.assert_called()
        assert 'class="healthy-notifications"' in page_html
        assert "No notifications found" in page_html
        # Sanity: no stray list item rendered when the list is empty.
        assert 'class="notifications-list"' not in page_html

    def test_empty_state_renders_clickable_notification_links(self, client, tmp_path):
        """Populated notifications render as clickable <a> tags with timestamps."""
        notifications = [
            {
                "timestamp": "2026-04-18T10:00:00",
                "message_url": "https://discord.com/channels/111/222/333",
            },
            {
                "timestamp": "2026-04-18T09:00:00",
                "message_url": "https://discord.com/channels/111/222/444",
            },
        ]
        with patch("idea_board.web.settings") as mock_settings, \
             patch(
                 "idea_board.web.query_healthy_notifications",
                 return_value=notifications,
             ):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "Recent healthy-bot notifications:" in page_html
        assert 'class="notifications-list"' in page_html
        # Each entry renders as an <a> link containing its timestamp.
        assert 'href="https://discord.com/channels/111/222/333"' in page_html
        assert 'href="https://discord.com/channels/111/222/444"' in page_html
        assert "2026-04-18T10:00:00" in page_html
        assert "2026-04-18T09:00:00" in page_html
        # Links must open in a new tab safely.
        assert 'target="_blank"' in page_html
        assert 'rel="noopener"' in page_html
        # "No notifications" copy must NOT appear when notifications exist.
        assert "No notifications found" not in page_html

    def test_notification_without_url_renders_as_plain_timestamp(self, client, tmp_path):
        """Notifications that lack a message URL render as plain <li> text, not links."""
        notifications = [{"timestamp": "2026-04-18T08:30:00", "message_url": None}]
        with patch("idea_board.web.settings") as mock_settings, \
             patch(
                 "idea_board.web.query_healthy_notifications",
                 return_value=notifications,
             ):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert "2026-04-18T08:30:00" in page_html
        assert 'class="notifications-list"' in page_html
        # No anchor tag inside the notifications list when url is None.
        assert "<li>2026-04-18T08:30:00</li>" in page_html

    def test_block_absent_when_crash_entries_exist(self, client, tmp_path):
        """When there ARE crashes, the healthy notifications block should not render.

        The block is an *empty-state* affordance — it would clutter the page
        when actual error cards are on screen.
        """
        crash_dir = tmp_path / "LLM Memory" / "Permanent"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_log.md").write_text(SAMPLE_CRASH_LOG, encoding="utf-8")
        with patch("idea_board.web.settings") as mock_settings, \
             patch(
                 "idea_board.web.query_healthy_notifications",
                 return_value=[{"timestamp": "2026-04-18T07:00:00", "message_url": "u"}],
             ) as q:
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        assert 'class="error-card"' in page_html
        assert 'class="healthy-notifications"' not in page_html
        assert "Recent healthy-bot notifications:" not in page_html
        # Query should not have been called when there are crash cards.
        q.assert_not_called()

    def test_malicious_url_is_html_escaped(self, client, tmp_path):
        """Even if a garbage message_url slips in, the href is HTML-escaped."""
        notifications = [
            {
                "timestamp": "2026-04-18T06:00:00",
                "message_url": 'https://x/"><script>alert(1)</script>',
            }
        ]
        with patch("idea_board.web.settings") as mock_settings, \
             patch(
                 "idea_board.web.query_healthy_notifications",
                 return_value=notifications,
             ):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # Raw <script> tag must never make it to the rendered output.
        assert "<script>alert(1)</script>" not in page_html


class TestErrorsEmptyStateHealthSignal:
    """TK-647: /errors empty state must never leak a 'healthy-signal' div.

    The empty-state block already expresses health through its headline,
    zero-count chips, and healthy-notifications list. A separate
    'healthy-signal' element would be a conflicting signal that confuses
    operators — especially when ``service_state.json`` is missing and the
    bot's started-at timestamp cannot be read. This suite pins that the
    rendered HTML never contains a ``healthy-signal`` class.
    """

    def test_empty_state_hides_bot_started_signal_when_state_missing(
        self, client, tmp_path
    ):
        """No ``healthy-signal`` div when service_state.json is absent.

        Points ``_SERVICE_STATE_FILE`` at a path that does not exist so
        ``_bot_uptime_seconds`` returns ``None``, then asserts the rendered
        /errors empty state contains no ``healthy-signal`` class.
        """
        missing_state = tmp_path / "service_state.json"
        assert not missing_state.exists()
        with patch("idea_board.web.settings") as mock_settings, \
             patch("idea_board.web._SERVICE_STATE_FILE", missing_state):
            mock_settings.vault_path = tmp_path
            resp = client.get("/errors")
        page_html = resp.data.decode()
        # Empty state should be rendered (no crashes present).
        assert 'class="empty-state"' in page_html
        # The defensive guarantee: no 'healthy-signal' element anywhere.
        assert "healthy-signal" not in page_html
