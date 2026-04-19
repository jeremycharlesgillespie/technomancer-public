"""Integration test for /live rendering every execution link (TK-797).

Unit tests in ``test_live_landing.py`` cover the individual layers
(``_collect_live_executions`` aggregation, single-row rendering, HTML
escaping). This test closes the loop end-to-end: seed 3-5 executions
into fake AIM state files, GET /live, then verify each execution id
shows up both as text AND inside a clickable href target.

Skipped if the Flask app or its transitive imports are unavailable so
the integration test degrades gracefully in minimal environments (per
TK-797 acceptance criteria).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from idea_board.web import app as _app

    _IMPORT_ERR: str | None = None
except Exception as exc:  # pragma: no cover - only trips on broken deps
    _app = None  # type: ignore[assignment]
    _IMPORT_ERR = f"idea_board.web unavailable: {exc}"


pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None,
    reason=_IMPORT_ERR or "idea_board.web import failed",
)


@pytest.fixture
def client():
    _app.config["TESTING"] = True
    with _app.test_client() as c:
        yield c


@pytest.fixture
def fake_agent_root(tmp_path, monkeypatch):
    """Redirect ``_AGENT_ROOT`` so we read temp AIM state files only."""
    (tmp_path / "aim").mkdir()
    (tmp_path / "aim" / "projects").mkdir()
    monkeypatch.setattr("idea_board.web._AGENT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _empty_ideas():
    """Isolate /live from real board data read during any hub fallbacks."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_live_with_multiple_executions_renders_links(client, fake_agent_root):
    """GET /live with 5 seeded executions must render every id + href."""
    _write_state(
        fake_agent_root / "aim" / ".aim_state.json",
        {
            "worker": {
                "status": "executing",
                "current_idea_id": "TK-901",
                "started_at": "2026-04-19T09:00:00",
                "last_observation": "running unit tests",
            },
            "board_snapshot": {
                "recent_completions": [
                    {
                        "key": "TK-902",
                        "summary": "Fix retry loop",
                        "resolved": "2026-04-19T08:00:00+0000",
                    },
                    {
                        "key": "TK-903",
                        "summary": "Upgrade dep",
                        "resolved": "2026-04-19T07:00:00+0000",
                    },
                ]
            },
        },
    )
    _write_state(
        fake_agent_root / "aim" / "projects" / "40acres" / ".aim_state.json",
        {
            "worker": {
                "status": "executing",
                "current_idea_id": "FA-501",
                "started_at": "2026-04-19T09:30:00",
                "last_observation": "compiling assets",
            },
            "board_snapshot": {
                "recent_completions": [
                    {
                        "key": "FA-502",
                        "summary": "Ship sitemap",
                        "resolved": "2026-04-19T06:00:00+0000",
                    },
                ]
            },
        },
    )

    start = time.monotonic()
    with patch("idea_board.web.settings.jira_project_key", "TK"), \
         patch("idea_board.web._live_route_accessible", return_value=True):
        resp = client.get("/live")
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert elapsed < 5.0, f"/live took {elapsed:.2f}s (exceeds 5s budget)"

    body = resp.get_data(as_text=True)
    execution_ids = ["TK-901", "TK-902", "TK-903", "FA-501", "FA-502"]

    hrefs = set(re.findall(r'href="([^"]+)"', body))

    for key in execution_ids:
        assert key in body, f"{key} missing from rendered /live HTML"
        matching = [h for h in hrefs if key in h]
        assert matching, (
            f"No href target referencing {key}; hrefs seen: {sorted(hrefs)}"
        )


def test_live_renders_executions(client, fake_agent_root):
    """GET /live must render every seeded execution id as text (TK-795).

    Stops at the text-rendering layer — href formatting is covered by
    ``test_live_with_multiple_executions_renders_links``. This test
    proves data flows from the mocked state files into the response
    body, independent of link markup.
    """
    _write_state(
        fake_agent_root / "aim" / ".aim_state.json",
        {
            "worker": {
                "status": "executing",
                "current_idea_id": "exec-001",
                "started_at": "2026-04-19T09:00:00",
                "last_observation": "step 1",
            },
            "board_snapshot": {
                "recent_completions": [
                    {
                        "key": "exec-002",
                        "summary": "Completed task 2",
                        "resolved": "2026-04-19T08:00:00+0000",
                    },
                    {
                        "key": "exec-003",
                        "summary": "Completed task 3",
                        "resolved": "2026-04-19T07:00:00+0000",
                    },
                ]
            },
        },
    )

    start = time.monotonic()
    with patch("idea_board.web.settings.jira_project_key", "TK"):
        resp = client.get("/live")
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert elapsed < 2.0, f"/live took {elapsed:.2f}s (exceeds 2s budget)"

    body = resp.get_data(as_text=True)
    for key in ("exec-001", "exec-002", "exec-003"):
        assert key in body, f"{key} missing from rendered /live response body"


# ---------------------------------------------------------------------------
# TK-788 — href rendered only when route is accessible
# ---------------------------------------------------------------------------


class TestLiveHrefConditionalOnRouteAccess:
    """_render_live_landing must only emit <a href> when _live_route_accessible
    returns True for the given key (TK-788).
    """

    def _seed_executing(self, fake_agent_root: Path, key: str) -> None:
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {
                    "status": "executing",
                    "current_idea_id": key,
                    "started_at": "2026-04-19T10:00:00",
                    "last_observation": "step A",
                },
                "board_snapshot": {"recent_completions": []},
            },
        )

    def _seed_recent(self, fake_agent_root: Path, key: str) -> None:
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": key, "summary": "Done story", "resolved": "2026-04-19T09:00:00+0000"},
                    ]
                },
            },
        )

    def test_href_rendered_when_route_accessible_for_executing(
        self, client, fake_agent_root
    ):
        """Executing row gets an <a href> when _live_route_accessible returns True."""
        self._seed_executing(fake_agent_root, "TK-788")
        with patch("idea_board.web._live_route_accessible", return_value=True), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/live").get_data(as_text=True)
        assert "TK-788" in body
        assert 'href="/live/TK-788"' in body

    def test_href_not_rendered_when_route_inaccessible_for_executing(
        self, client, fake_agent_root
    ):
        """Executing row renders key as plain text when _live_route_accessible returns False."""
        self._seed_executing(fake_agent_root, "TK-788")
        with patch("idea_board.web._live_route_accessible", return_value=False), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/live").get_data(as_text=True)
        assert "TK-788" in body
        assert 'href="/live/TK-788"' not in body

    def test_href_rendered_when_route_accessible_for_recent(
        self, client, fake_agent_root
    ):
        """Recent-completion row gets an <a href> when _live_route_accessible returns True."""
        self._seed_recent(fake_agent_root, "TK-788")
        with patch("idea_board.web._live_route_accessible", return_value=True), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/live").get_data(as_text=True)
        assert "TK-788" in body
        assert 'href="/live/TK-788"' in body

    def test_href_not_rendered_when_route_inaccessible_for_recent(
        self, client, fake_agent_root
    ):
        """Recent-completion row renders key as plain text when _live_route_accessible returns False."""
        self._seed_recent(fake_agent_root, "TK-788")
        with patch("idea_board.web._live_route_accessible", return_value=False), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/live").get_data(as_text=True)
        assert "TK-788" in body
        assert 'href="/live/TK-788"' not in body

    def test_mixed_accessibility_renders_selectively(self, client, fake_agent_root):
        """When two keys exist and only one is accessible, only that key gets an href."""
        _write_state(
            fake_agent_root / "aim" / ".aim_state.json",
            {
                "worker": {"status": "idle", "current_idea_id": None},
                "board_snapshot": {
                    "recent_completions": [
                        {"key": "TK-100", "summary": "has log", "resolved": "2026-04-19T09:00:00+0000"},
                        {"key": "TK-200", "summary": "no log", "resolved": "2026-04-19T08:00:00+0000"},
                    ]
                },
            },
        )

        def _accessible(key: str) -> bool:
            return key == "TK-100"

        with patch("idea_board.web._live_route_accessible", side_effect=_accessible), \
             patch("idea_board.web.settings.jira_project_key", "TK"):
            body = client.get("/live").get_data(as_text=True)

        assert 'href="/live/TK-100"' in body, "TK-100 should have href (accessible)"
        assert 'href="/live/TK-200"' not in body, "TK-200 should not have href (inaccessible)"
        assert "TK-100" in body
        assert "TK-200" in body
