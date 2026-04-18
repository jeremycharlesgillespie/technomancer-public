"""Tests for /quality/<story_key> — per-story validation detail page (TK-692).

Covers the detail route that lets an operator investigate a single low-scored
story: full seven-axis scores, per-axis reasoning, verifier output blob,
red flags, and links to the Jira issue + merge commit. Unknown keys 404.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent import aiv_schema
from idea_board import web as web_mod
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    """Point aiv_schema at a per-test temporary SQLite DB."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@pytest.fixture(autouse=True)
def _stub_external_links(monkeypatch):
    """Disable git subprocess lookup and Jira URL config by default.

    Individual tests can re-enable via their own monkeypatch calls.
    """
    monkeypatch.setattr(web_mod, "_aiv_merge_commit_link", lambda _k: None)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _insert_row(
    *,
    story_key: str,
    story_title: str = "Some story",
    validated_at: str | None = None,
    merged_at: str | None = None,
    scores: tuple[int, int, int, int, int, int, int] = (9, 8, 7, 10, 9, 6, 8),
    overall: float = 8.1,
    red_flags: list[str] | None = None,
    verification_method: str = "tests-only",
    verification_output: str = "",
    reasoning_map: dict[str, str] | None = None,
    error: str | None = None,
) -> None:
    """Insert a synthetic story_quality row for the detail-page tests."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    if merged_at is None:
        merged_at = validated_at
    payload = {
        "story_key": story_key,
        "story_title": story_title,
        "merged_at": merged_at,
        "validated_at": validated_at,
        "meets_requirements": scores[0],
        "code_quality": scores[1],
        "test_quality": scores[2],
        "security_safety": scores[3],
        "scope_discipline": scores[4],
        "edge_cases": scores[5],
        "product_impact": scores[6],
        "overall_score": overall,
        "red_flags_json": json.dumps(red_flags or []),
        "verification_method": verification_method,
        "verification_output": verification_output,
        "reasoning_json": json.dumps(reasoning_map or {}),
        "error": error,
    }
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(f":{k}" for k in payload)
    conn.execute(
        f"INSERT INTO story_quality ({columns}) VALUES ({placeholders})",
        payload,
    )
    conn.commit()


class TestQualityDetailPage:
    """GET /quality/<story_key> — rendering for an existing row."""

    def test_renders_all_seven_scores_reasoning_verification_and_flags(self, client):
        """Acceptance: detail page shows every score + reasoning + verifier output + flags.

        Inserts a row with distinct integer scores on each axis, per-axis
        reasoning strings, a verifier output blob, and two red flags, then
        asserts every piece of that content appears in the rendered HTML.
        """
        reasoning = {
            "meets_requirements": "reason-for-requirements",
            "code_quality": "reason-for-code-quality",
            "test_quality": "reason-for-test-quality",
            "security_safety": "reason-for-security",
            "scope_discipline": "reason-for-scope",
            "edge_cases": "reason-for-edges",
            "product_impact": "reason-for-impact",
        }
        _insert_row(
            story_key="TK-123",
            story_title="Detail test story",
            scores=(9, 8, 7, 10, 9, 6, 8),
            overall=8.1,
            red_flags=["broad_except", "hardcoded_secret"],
            verification_method="tests-only",
            verification_output="PASS: 42 tests green\n+42 -3 lines",
            reasoning_map=reasoning,
        )

        resp = client.get("/quality/TK-123")

        assert resp.status_code == 200
        assert "text/html" in resp.content_type
        body = resp.get_data(as_text=True)

        # Story key + title in heading.
        assert "TK-123" in body
        assert "Detail test story" in body

        # Every axis score value must render in the body.
        for score in ("9", "8", "7", "10", "6"):
            assert score in body

        # Every per-axis reasoning string must appear.
        for reason in reasoning.values():
            assert reason in body

        # Verification output blob renders verbatim.
        assert "PASS: 42 tests green" in body
        assert "+42 -3 lines" in body

        # Both red flags render.
        assert "broad_except" in body
        assert "hardcoded_secret" in body

        # Verification method surfaces.
        assert "tests-only" in body

    def test_renders_axis_column_names(self, client):
        """Every story_quality axis column name renders on the detail page."""
        _insert_row(story_key="TK-201")

        resp = client.get("/quality/TK-201")
        body = resp.get_data(as_text=True)

        for axis in (
            "meets_requirements",
            "code_quality",
            "test_quality",
            "security_safety",
            "scope_discipline",
            "edge_cases",
            "product_impact",
        ):
            assert axis in body

    def test_empty_red_flags_renders_no_flags_message(self, client):
        """A clean row (no flags) shows the 'No red flags' copy."""
        _insert_row(story_key="TK-202", red_flags=[])

        resp = client.get("/quality/TK-202")
        body = resp.get_data(as_text=True)

        assert "No red flags" in body

    def test_empty_verification_output_renders_muted_placeholder(self, client):
        """Missing verifier output collapses to a muted 'No verifier output' note."""
        _insert_row(story_key="TK-203", verification_output="")

        resp = client.get("/quality/TK-203")
        body = resp.get_data(as_text=True)

        assert "No verifier output recorded" in body

    def test_renders_jira_link_when_configured(self, client, monkeypatch):
        """Jira URL in settings yields a browse link on the page."""
        monkeypatch.setattr(
            web_mod.settings, "jira_url", "https://example.atlassian.net"
        )
        _insert_row(story_key="TK-300")

        resp = client.get("/quality/TK-300")
        body = resp.get_data(as_text=True)

        assert "https://example.atlassian.net/browse/TK-300" in body

    def test_no_jira_link_when_not_configured(self, client, monkeypatch):
        """Without jira_url, page renders the 'Jira not configured' fallback."""
        monkeypatch.setattr(web_mod.settings, "jira_url", None)
        _insert_row(story_key="TK-301")

        resp = client.get("/quality/TK-301")
        body = resp.get_data(as_text=True)

        assert "Jira not configured" in body
        assert "atlassian.net/browse/TK-301" not in body

    def test_renders_merge_commit_link_when_git_returns_sha(self, client, monkeypatch):
        """When git log finds a merge SHA, a commit URL is rendered."""
        fake_sha = "abcdef1234567890abcdef1234567890abcdef12"
        monkeypatch.setattr(
            web_mod,
            "_aiv_merge_commit_link",
            lambda key: f"https://github.com/jeremycharlesgillespie/"
                        f"technomancer-public/commit/{fake_sha}"
                        if key == "TK-400" else None,
        )
        _insert_row(story_key="TK-400")

        resp = client.get("/quality/TK-400")
        body = resp.get_data(as_text=True)

        assert fake_sha in body
        assert "/commit/" in body

    def test_error_field_renders_error_panel(self, client):
        """A row with an error field shows an error banner."""
        _insert_row(story_key="TK-500", error="parse_failure")

        resp = client.get("/quality/TK-500")
        body = resp.get_data(as_text=True)

        assert "parse_failure" in body


class TestQualityDetail404:
    """Unknown keys must 404, not 500."""

    def test_unknown_key_returns_404(self, client):
        """Acceptance: GET /quality/<missing> returns 404."""
        resp = client.get("/quality/TK-999999")

        assert resp.status_code == 404

    def test_unknown_key_404_body_is_helpful_html(self, client):
        """404 body explains that no validation exists for the key."""
        resp = client.get("/quality/TK-999998")

        assert resp.status_code == 404
        body = resp.get_data(as_text=True)
        assert "TK-999998" in body
        assert "No validation" in body or "not found" in body.lower()

    def test_missing_db_file_returns_404(self, client, tmp_path, monkeypatch):
        """If the AIV DB file doesn't exist yet, any key 404s, never 500s."""
        ghost = tmp_path / "does-not-exist.db"
        monkeypatch.setattr(aiv_schema, "DB_PATH", ghost)

        resp = client.get("/quality/TK-1")

        assert resp.status_code == 404

    def test_blank_key_component_does_not_500(self, client):
        """Trailing-slash / empty key must not crash the detail route."""
        resp = client.get("/quality/")

        # 200 (list page), redirect, or 404 — anything but 500 is acceptable.
        assert resp.status_code != 500


class TestAivFetchDetail:
    """Unit-test the helper that loads a single story_quality row."""

    def test_returns_none_for_missing_key(self):
        """No row for key → None, not a partially-populated dict."""
        assert web_mod._aiv_fetch_detail("TK-NOSUCH") is None

    def test_returns_dict_with_expected_keys_for_existing_row(self):
        """Fetch returns the full enriched entry for an existing row."""
        _insert_row(
            story_key="TK-700",
            reasoning_map={"code_quality": "ok"},
            verification_output="some output",
            red_flags=["flag_a"],
        )

        entry = web_mod._aiv_fetch_detail("TK-700")

        assert entry is not None
        assert entry["story_key"] == "TK-700"
        assert entry["red_flags"] == ["flag_a"]
        assert entry["reasoning"] == {"code_quality": "ok"}
        assert entry["verification_output"] == "some output"
        for axis in (
            "meets_requirements",
            "code_quality",
            "test_quality",
            "security_safety",
            "scope_discipline",
            "edge_cases",
            "product_impact",
        ):
            assert axis in entry

    def test_malformed_reasoning_json_collapses_to_empty_dict(self):
        """Corrupt reasoning_json is tolerated — reasoning becomes {}."""
        aiv_schema.init_db()
        conn = aiv_schema._get_conn()
        conn.execute(
            "INSERT INTO story_quality (story_key, reasoning_json, red_flags_json) "
            "VALUES (?, ?, ?)",
            ("TK-701", "not-json-at-all", "[]"),
        )
        conn.commit()

        entry = web_mod._aiv_fetch_detail("TK-701")

        assert entry is not None
        assert entry["reasoning"] == {}
