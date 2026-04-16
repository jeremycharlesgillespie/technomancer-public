"""Tests for agent.error_routing — routing_config.json-driven webhook resolution.

Design note: these tests avoid ``monkeypatch.setenv`` + ``importlib.reload``
patterns because those mutate the module-level ``agent.config.settings``
object globally, leaking into later tests. Instead we patch
``agent.error_routing.settings`` and ``agent.alerts.settings`` directly with
fresh mocks, which cleanly scopes per-test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# load_routing_config
# ---------------------------------------------------------------------------


class TestLoadRoutingConfig:
    def test_missing_file_returns_empty(self, tmp_path):
        from agent import error_routing

        missing = tmp_path / "does_not_exist.json"
        with patch.object(error_routing, "CONFIG_PATH", missing):
            assert error_routing.load_routing_config() == {}

    def test_valid_file_loads(self, tmp_path):
        from agent import error_routing

        cfg = tmp_path / "routing.json"
        cfg.write_text(json.dumps({"default_webhook": "https://hook.example/default"}))
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            data = error_routing.load_routing_config()
        assert data == {"default_webhook": "https://hook.example/default"}

    def test_malformed_json_returns_empty(self, tmp_path):
        from agent import error_routing

        cfg = tmp_path / "bad.json"
        cfg.write_text("{ this is not json")
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            assert error_routing.load_routing_config() == {}

    def test_non_dict_root_returns_empty(self, tmp_path):
        from agent import error_routing

        cfg = tmp_path / "list.json"
        cfg.write_text(json.dumps(["not", "a", "dict"]))
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            assert error_routing.load_routing_config() == {}

    def test_nested_structure_preserved(self, tmp_path):
        from agent import error_routing

        payload = {
            "default_webhook": "https://hook/default",
            "categories": {"critical": "https://hook/crit"},
            "error_types": {"rate_limited": "https://hook/rl"},
        }
        cfg = tmp_path / "full.json"
        cfg.write_text(json.dumps(payload))
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            data = error_routing.load_routing_config()
        assert data == payload


# ---------------------------------------------------------------------------
# get_webhook_for_category
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, payload: dict) -> Path:
    cfg = tmp_path / "routing.json"
    cfg.write_text(json.dumps(payload))
    return cfg


class TestGetWebhookForCategory:
    """get_webhook_for_category resolution order:
    error_types[category] > categories[severity] > default_webhook > settings.
    """

    def test_error_type_takes_highest_priority(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "categories": {"critical": "https://hook/crit"},
            "error_types": {"rate_limited": "https://hook/rl"},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category="rate_limited", severity="critical"
            )
        assert result == "https://hook/rl"

    def test_severity_used_when_no_error_type(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "categories": {"critical": "https://hook/crit"},
            "error_types": {"rate_limited": "https://hook/rl"},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category="unmapped_type", severity="critical"
            )
        assert result == "https://hook/crit"

    def test_default_used_when_no_matches(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "categories": {"critical": "https://hook/crit"},
            "error_types": {"rate_limited": "https://hook/rl"},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category="nope", severity="info"
            )
        assert result == "https://hook/default"

    def test_settings_fallback_when_empty_config(self, tmp_path):
        from agent import error_routing

        missing = tmp_path / "absent.json"
        fake_settings = SimpleNamespace(discord_alerts_webhook="https://hook/env")
        with patch.object(error_routing, "CONFIG_PATH", missing), \
             patch.object(error_routing, "settings", fake_settings):
            result = error_routing.get_webhook_for_category(
                category="anything", severity="info"
            )
        assert result == "https://hook/env"

    def test_no_category_no_severity_returns_default(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {"default_webhook": "https://hook/default"})
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category()
        assert result == "https://hook/default"

    def test_none_category_falls_through_to_severity(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "categories": {"warning": "https://hook/warn"},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category=None, severity="warning"
            )
        assert result == "https://hook/warn"

    def test_empty_string_category_treated_as_none(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "error_types": {"": "https://hook/should-not-match"},
            "categories": {"info": "https://hook/info"},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category="", severity="info"
            )
        assert result == "https://hook/info"

    def test_null_entry_falls_through(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "error_types": {"rate_limited": None},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category="rate_limited", severity=None
            )
        assert result == "https://hook/default"

    def test_whitespace_entry_falls_through(self, tmp_path):
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "categories": {"critical": "   "},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category(
                category=None, severity="critical"
            )
        assert result == "https://hook/default"

    def test_returns_empty_string_when_nothing_configured(self, tmp_path):
        from agent import error_routing

        missing = tmp_path / "nowhere.json"
        fake_settings = SimpleNamespace(discord_alerts_webhook="")
        with patch.object(error_routing, "CONFIG_PATH", missing), \
             patch.object(error_routing, "settings", fake_settings):
            assert error_routing.get_webhook_for_category("rate_limited", "critical") == ""

    def test_settings_missing_attr_safe(self, tmp_path):
        """If settings somehow lacks discord_alerts_webhook, we get ''."""
        from agent import error_routing

        missing = tmp_path / "missing.json"
        fake_settings = SimpleNamespace()
        with patch.object(error_routing, "CONFIG_PATH", missing), \
             patch.object(error_routing, "settings", fake_settings):
            assert error_routing.get_webhook_for_category() == ""

    def test_non_string_value_rejected(self, tmp_path):
        """Non-string entries (e.g. numbers) fall through."""
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {
            "default_webhook": "https://hook/default",
            "error_types": {"rate_limited": 12345},
        })
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category("rate_limited")
        assert result == "https://hook/default"

    def test_missing_mappings_handled(self, tmp_path):
        """Config missing ``categories`` or ``error_types`` keys is fine."""
        from agent import error_routing

        cfg = _write_cfg(tmp_path, {"default_webhook": "https://hook/default"})
        with patch.object(error_routing, "CONFIG_PATH", cfg):
            result = error_routing.get_webhook_for_category("anything", "critical")
        assert result == "https://hook/default"


# ---------------------------------------------------------------------------
# Integration: alerts.send_alert with routing
# ---------------------------------------------------------------------------


class TestSendAlertRouting:
    """Verify send_alert consults routing_config when given a category."""

    def test_routes_by_error_type(self, tmp_path):
        from agent import alerts, error_routing

        cfg = _write_cfg(tmp_path, {
            "error_types": {"rate_limited": "https://hook/rl"},
        })
        captured: dict = {}

        def fake_retry(func, url, **kwargs):
            captured["url"] = url
            return MagicMock(status_code=204)

        with patch.object(error_routing, "CONFIG_PATH", cfg), \
             patch("agent.discord_rate_limit.retry_request", side_effect=fake_retry):
            alerts.send_alert("hello", category="rate_limited", level="low")
        assert captured["url"] == "https://hook/rl"

    def test_routes_by_severity(self, tmp_path):
        from agent import alerts, error_routing

        cfg = _write_cfg(tmp_path, {
            "categories": {"critical": "https://hook/crit"},
        })
        captured: dict = {}

        def fake_retry(func, url, **kwargs):
            captured["url"] = url
            return MagicMock(status_code=204)

        with patch.object(error_routing, "CONFIG_PATH", cfg), \
             patch("agent.discord_rate_limit.retry_request", side_effect=fake_retry):
            alerts.send_alert("boom", category="unmapped", level="critical")
        assert captured["url"] == "https://hook/crit"

    def test_explicit_webhook_overrides_routing(self, tmp_path):
        from agent import alerts, error_routing

        cfg = _write_cfg(tmp_path, {
            "error_types": {"rate_limited": "https://hook/routed"},
        })
        captured: dict = {}

        def fake_retry(func, url, **kwargs):
            captured["url"] = url
            return MagicMock(status_code=204)

        with patch.object(error_routing, "CONFIG_PATH", cfg), \
             patch("agent.discord_rate_limit.retry_request", side_effect=fake_retry):
            alerts.send_alert(
                "hi",
                category="rate_limited",
                webhook_url="https://hook/explicit",
            )
        assert captured["url"] == "https://hook/explicit"

    def test_no_webhook_anywhere_skips(self, tmp_path):
        from agent import alerts, error_routing

        missing = tmp_path / "nowhere.json"
        fake_settings = SimpleNamespace(discord_alerts_webhook="")

        sentinel = MagicMock()
        with patch.object(error_routing, "CONFIG_PATH", missing), \
             patch.object(error_routing, "settings", fake_settings), \
             patch("agent.discord_rate_limit.retry_request", sentinel):
            alerts.send_alert("no destination", category="rate_limited")

        sentinel.assert_not_called()

    def test_falls_back_to_settings_when_no_category_match(self, tmp_path):
        """With no routing match and no explicit URL, use settings.discord_alerts_webhook.

        This is the regression case: ``webhook_url=None`` must trigger the
        config lookup, which itself falls back to ``settings`` at the end.
        Both the config path and the settings module must be patched so
        neither can leak real env state into the test.
        """
        from agent import alerts, error_routing

        missing = tmp_path / "absent.json"
        fake_settings = SimpleNamespace(
            discord_alerts_webhook="https://discord.com/webhooks/fallback"
        )
        captured: dict = {}

        def fake_retry(func, url, **kwargs):
            captured["url"] = url
            return MagicMock(status_code=204)

        with patch.object(error_routing, "CONFIG_PATH", missing), \
             patch.object(error_routing, "settings", fake_settings), \
             patch("agent.discord_rate_limit.retry_request", side_effect=fake_retry):
            alerts.send_alert("needs fallback", category="unmapped", level="unmapped")

        assert captured["url"] == "https://discord.com/webhooks/fallback"


# ---------------------------------------------------------------------------
# Config file shape
# ---------------------------------------------------------------------------


class TestShippedConfig:
    """Validate the shipped routing_config.json is well-formed and has the
    expected top-level keys so operators can edit it safely.
    """

    def test_shipped_config_is_valid_json(self):
        from agent import error_routing

        # The real CONFIG_PATH points to local-agent/routing_config.json.
        assert error_routing.CONFIG_PATH.exists(), (
            f"routing_config.json should ship at {error_routing.CONFIG_PATH}"
        )
        data = json.loads(error_routing.CONFIG_PATH.read_text())
        assert isinstance(data, dict)
        # Expected structural keys (even if values are null by default).
        assert "default_webhook" in data
        assert "categories" in data
        assert "error_types" in data
