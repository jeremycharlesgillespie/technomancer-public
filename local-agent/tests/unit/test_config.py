"""Tests for agent/config.py — settings, computed properties, caching."""

from pathlib import Path

from agent.config import Settings, get_settings


class TestSettingsDefaults:
    """Test that default values are set correctly."""

    def test_discord_allowed_channel_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.discord_allowed_channel == "llm_chat"

    def test_discord_alerts_channel_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.discord_alerts_channel == "bot_alerts"

    def test_ollama_host_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.ollama_host == "http://127.0.0.1:11434"

    def test_briefing_hour_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.briefing_hour == 7

    def test_briefing_enabled_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.briefing_enabled is True

    def test_api_cost_alert_threshold_default(self):
        s = Settings(discord_bot_token="test-token")
        assert s.api_cost_alert_threshold == 1.0


class TestComputedProperties:
    """Test computed path properties."""

    def test_llm_memory_path(self):
        s = Settings(discord_bot_token="t", vault_path=Path("/test/vault"))
        assert s.llm_memory_path == Path("/test/vault/LLM Memory")

    def test_permanent_path(self):
        s = Settings(discord_bot_token="t", vault_path=Path("/test/vault"))
        assert s.permanent_path == Path("/test/vault/LLM Memory/Permanent")

    def test_context_path(self):
        s = Settings(discord_bot_token="t", vault_path=Path("/test/vault"))
        assert s.context_path == Path("/test/vault/LLM Memory/Context")


class TestGetSettings:
    """Test the cached settings accessor."""

    def test_returns_object_with_expected_attrs(self):
        s = get_settings()
        assert hasattr(s, "discord_bot_token")
        assert hasattr(s, "ollama_host")
        assert hasattr(s, "vault_path")

    def test_cached_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
