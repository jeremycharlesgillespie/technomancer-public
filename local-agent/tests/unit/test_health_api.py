"""Tests for the /api/health Flask endpoint in idea_board/web.py.

Contract (TK-522):

* Response is ``{status, checks, timestamp}`` where ``status`` is
  ``'healthy'``, ``'degraded'``, or ``'unhealthy'``.
* ``checks`` has exactly five keys — bot, ollama, jira, executor, disk —
  each with at least ``{ok, latency_ms, detail}``.
* HTTP 200 for healthy and degraded; HTTP 503 for unhealthy.
* The hub HTML renders a card per check via ``health-<name>`` divs and
  polls ``/api/health`` every 10 seconds.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from idea_board import health
from idea_board.web import app


def _stub_check(ok: bool, detail: str = "stub", **extra):
    def inner():
        result = {"ok": ok, "detail": detail, "latency_ms": 1}
        result.update(extra)
        return result

    return inner


@pytest.fixture(autouse=True)
def _reset_cache():
    health.clear_cache()
    yield
    health.clear_cache()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def all_healthy():
    funcs = {name: _stub_check(True) for name in ("bot", "ollama", "jira", "executor", "disk")}
    with patch.object(health, "_CHECK_FUNCS", funcs):
        yield


class TestHealthEndpointContract:
    def test_returns_200_when_healthy(self, client, all_healthy):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_response_schema(self, client, all_healthy):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert set(data.keys()) >= {"status", "checks", "timestamp"}
        assert data["status"] == "healthy"

    def test_all_five_checks_present(self, client, all_healthy):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert set(data["checks"].keys()) == {"bot", "ollama", "jira", "executor", "disk"}

    def test_each_check_has_required_fields(self, client, all_healthy):
        resp = client.get("/api/health")
        data = resp.get_json()
        for name, check in data["checks"].items():
            assert "ok" in check, f"{name}.ok missing"
            assert "latency_ms" in check, f"{name}.latency_ms missing"
            assert "detail" in check, f"{name}.detail missing"

    def test_ollama_down_returns_200_degraded(self, client):
        """Acceptance: 'returns 200 with status=degraded when Ollama down'."""
        funcs = {
            "bot": _stub_check(True),
            "ollama": _stub_check(False, detail="unreachable"),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert data["checks"]["ollama"]["ok"] is False

    def test_bot_down_returns_503_unhealthy(self, client):
        """Acceptance: 'returns 503 when bot process dead'."""
        funcs = {
            "bot": _stub_check(False, detail="no pid file"),
            "ollama": _stub_check(True),
            "jira": _stub_check(True),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            resp = client.get("/api/health")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "unhealthy"

    def test_jira_dlq_nonempty_is_degraded_not_unhealthy(self, client):
        """DLQ backlog is optional — should not flip us to 503."""
        funcs = {
            "bot": _stub_check(True),
            "ollama": _stub_check(True),
            "jira": _stub_check(False, detail="3 unresolved DLQ entries"),
            "executor": _stub_check(True),
            "disk": _stub_check(True),
        }
        with patch.object(health, "_CHECK_FUNCS", funcs):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "degraded"


class TestHubPageHealthPanel:
    def test_hub_contains_five_health_cards(self, client):
        """Hub HTML must include a card div for each of the five checks."""
        resp = client.get("/")
        body = resp.data.decode()
        for name in ("bot", "ollama", "jira", "executor", "disk"):
            assert f"health-{name}" in body

    def test_hub_polls_health_endpoint(self, client):
        resp = client.get("/")
        body = resp.data.decode()
        assert "updateHealth" in body
        assert "setInterval(updateHealth, 10000)" in body
        assert "/api/health" in body
