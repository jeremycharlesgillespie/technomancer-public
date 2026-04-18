"""Tests for AIMM daemon settings on agent/config.py."""

import pytest

from agent.config import Settings


class TestAimmDefaults:
    """Defaults must match the plan so AIMM has sane out-of-box behavior."""

    def test_aimm_cycle_interval_default(self):
        s = Settings(discord_bot_token="t")
        assert s.aimm_cycle_interval == 600

    def test_aimm_max_approvals_per_cycle_default(self):
        s = Settings(discord_bot_token="t")
        assert s.aimm_max_approvals_per_cycle == 5

    def test_aimm_max_approvals_per_day_default(self):
        s = Settings(discord_bot_token="t")
        assert s.aimm_max_approvals_per_day == 50

    def test_aimm_approve_feature_security_default(self):
        s = Settings(discord_bot_token="t")
        assert s.aimm_approve_feature_security is True

    def test_aimm_state_dir_default(self):
        s = Settings(discord_bot_token="t")
        assert s.aimm_state_dir == "aimm"


class TestAimmEnvOverrides:
    """Env vars prefixed AIMM_* must override defaults via Pydantic Settings."""

    def test_cycle_interval_from_env(self, monkeypatch):
        monkeypatch.setenv("AIMM_CYCLE_INTERVAL", "120")
        s = Settings(discord_bot_token="t")
        assert s.aimm_cycle_interval == 120

    def test_max_approvals_per_cycle_from_env(self, monkeypatch):
        monkeypatch.setenv("AIMM_MAX_APPROVALS_PER_CYCLE", "9")
        s = Settings(discord_bot_token="t")
        assert s.aimm_max_approvals_per_cycle == 9

    def test_max_approvals_per_day_from_env(self, monkeypatch):
        monkeypatch.setenv("AIMM_MAX_APPROVALS_PER_DAY", "200")
        s = Settings(discord_bot_token="t")
        assert s.aimm_max_approvals_per_day == 200

    def test_approve_feature_security_from_env(self, monkeypatch):
        monkeypatch.setenv("AIMM_APPROVE_FEATURE_SECURITY", "false")
        s = Settings(discord_bot_token="t")
        assert s.aimm_approve_feature_security is False

    def test_state_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("AIMM_STATE_DIR", "/tmp/aimm-state")
        s = Settings(discord_bot_token="t")
        assert s.aimm_state_dir == "/tmp/aimm-state"


class TestAimmTypeCoercion:
    """Integer fields must coerce string env values; invalid values must raise."""

    def test_cycle_interval_coerces_string_to_int(self, monkeypatch):
        monkeypatch.setenv("AIMM_CYCLE_INTERVAL", "42")
        s = Settings(discord_bot_token="t")
        assert isinstance(s.aimm_cycle_interval, int)
        assert s.aimm_cycle_interval == 42

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv("AIMM_CYCLE_INTERVAL", "not-an-int")
        with pytest.raises(Exception):
            Settings(discord_bot_token="t")
