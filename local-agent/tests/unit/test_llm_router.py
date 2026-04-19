"""Unit tests for agent/llm_router.py.

Covers the three routing paths:
  1. Ollama primary, ollama returns → no claude call.
  2. Ollama primary, ollama fails → falls back to claude -p.
  3. Claude primary → no ollama call.

And the settings mapping: each of the 4 roles reads its own
``<role>_model`` setting.
"""

from __future__ import annotations

from unittest.mock import patch

from agent import llm_router


# ---------------------------------------------------------------------------
# Ollama-primary path
# ---------------------------------------------------------------------------


def _fake_settings(**overrides):
    """Build a stand-in settings object with the AIM/AIMM routing keys."""
    from types import SimpleNamespace

    defaults = {
        "aim_brain_model": "ollama:qwen3.5:27b",
        "dedup_judge_model": "ollama:qwen3.5:27b",
        "aimm_observer_model": "ollama:qwen3.5:27b",
        "aimm_suggester_model": "ollama:qwen3.5:27b",
        "llm_fallback_model": "claude-haiku-4-5",
        "llm_experiment_mode": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestOllamaPrimary:
    def test_ollama_success_returns_its_output(self, monkeypatch):
        monkeypatch.setattr(llm_router, "get_settings", _fake_settings)
        with patch.object(llm_router, "ollama_chat", return_value="ollama reply"):
            with patch.object(llm_router, "_claude_chat") as mock_claude:
                out = llm_router.complete("aim_brain", "hi")
        assert out == "ollama reply"
        mock_claude.assert_not_called()

    def test_ollama_failure_falls_back_to_claude(self, monkeypatch):
        monkeypatch.setattr(llm_router, "get_settings", _fake_settings)
        with patch.object(llm_router, "ollama_chat", return_value=None):
            with patch.object(
                llm_router, "_claude_chat", return_value="claude reply"
            ) as mock_claude:
                out = llm_router.complete("dedup_judge", "hi")
        assert out == "claude reply"
        # Fallback invoked with the configured fallback model.
        args, kwargs = mock_claude.call_args
        assert args[1] == "claude-haiku-4-5"

    def test_both_paths_fail_returns_none(self, monkeypatch):
        monkeypatch.setattr(llm_router, "get_settings", _fake_settings)
        with patch.object(llm_router, "ollama_chat", return_value=None):
            with patch.object(llm_router, "_claude_chat", return_value=None):
                out = llm_router.complete("aimm_observer", "hi")
        assert out is None


# ---------------------------------------------------------------------------
# Claude-primary path
# ---------------------------------------------------------------------------


class TestClaudePrimary:
    def test_claude_primary_does_not_call_ollama(self, monkeypatch):
        settings = _fake_settings(aim_brain_model="claude-sonnet-4-6")
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        with patch.object(llm_router, "ollama_chat") as mock_ollama:
            with patch.object(
                llm_router, "_claude_chat", return_value="sonnet reply"
            ) as mock_claude:
                out = llm_router.complete("aim_brain", "hi")
        assert out == "sonnet reply"
        mock_ollama.assert_not_called()
        args, _ = mock_claude.call_args
        assert args[1] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Settings mapping — each role reads its own key
# ---------------------------------------------------------------------------


class TestRoleMapping:
    def test_each_role_uses_its_own_setting(self, monkeypatch):
        settings = _fake_settings(
            aim_brain_model="ollama:tag-a",
            dedup_judge_model="ollama:tag-b",
            aimm_observer_model="ollama:tag-c",
            aimm_suggester_model="ollama:tag-d",
        )
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        captured: list[str] = []

        def fake_ollama(prompt, tag, timeout=60, options=None):
            captured.append(tag)
            return f"reply from {tag}"

        with patch.object(llm_router, "ollama_chat", side_effect=fake_ollama):
            llm_router.complete("aim_brain", "x")
            llm_router.complete("dedup_judge", "x")
            llm_router.complete("aimm_observer", "x")
            llm_router.complete("aimm_suggester", "x")
        assert captured == ["tag-a", "tag-b", "tag-c", "tag-d"]

    def test_experiment_mode_ollama_overrides_all_roles(self, monkeypatch):
        """LLM_EXPERIMENT_MODE=ollama forces every role to ollama default."""
        settings = _fake_settings(
            aim_brain_model="claude-sonnet-4-6",
            dedup_judge_model="claude-sonnet-4-6",
            aimm_observer_model="claude-sonnet-4-6",
            aimm_suggester_model="claude-sonnet-4-6",
            llm_experiment_mode="ollama",
        )
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        captured: list[str] = []

        def fake_ollama(prompt, tag, timeout=60, options=None):
            captured.append(tag)
            return "ok"

        with patch.object(llm_router, "ollama_chat", side_effect=fake_ollama):
            llm_router.complete("aim_brain", "x")
            llm_router.complete("dedup_judge", "x")
            llm_router.complete("aimm_observer", "x")
            llm_router.complete("aimm_suggester", "x")
        # Every role should have hit ollama:qwen3.5:latest despite per-role settings.
        assert captured == ["qwen3.5:latest"] * 4

    def test_experiment_mode_claude_overrides_all_roles(self, monkeypatch):
        settings = _fake_settings(
            aim_brain_model="ollama:qwen3.5:latest",
            llm_experiment_mode="claude",
        )
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        with patch.object(llm_router, "ollama_chat") as mock_ollama:
            with patch.object(
                llm_router, "_claude_chat", return_value="r"
            ) as mock_claude:
                llm_router.complete("aim_brain", "x")
        mock_ollama.assert_not_called()
        args, _ = mock_claude.call_args
        assert args[1] == "claude-haiku-4-5"

    def test_experiment_mode_explicit_model_overrides(self, monkeypatch):
        """Any explicit model string works as the experiment mode value."""
        settings = _fake_settings(
            llm_experiment_mode="claude-sonnet-4-6",
        )
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        with patch.object(
            llm_router, "_claude_chat", return_value="r"
        ) as mock_claude:
            llm_router.complete("dedup_judge", "x")
        args, _ = mock_claude.call_args
        assert args[1] == "claude-sonnet-4-6"

    def test_current_routing_reflects_experiment_mode(self, monkeypatch):
        settings = _fake_settings(llm_experiment_mode="ollama")
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        routing = llm_router.current_routing()
        assert set(routing.values()) == {"ollama:qwen3.5:latest"}
        assert set(routing) == {
            "aim_brain", "dedup_judge", "aimm_observer", "aimm_suggester",
            "splitter_decomposer", "evergreen_generator",
        }

    def test_unset_setting_defaults_to_haiku_claude_path(self, monkeypatch):
        # Simulate a setting explicitly set to empty / None — _resolve_primary
        # falls back to claude-haiku-4-5, which means claude path fires.
        settings = _fake_settings(aim_brain_model="")
        monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
        with patch.object(llm_router, "ollama_chat") as mock_ollama:
            with patch.object(
                llm_router, "_claude_chat", return_value="default"
            ) as mock_claude:
                out = llm_router.complete("aim_brain", "x")
        assert out == "default"
        mock_ollama.assert_not_called()
        args, _ = mock_claude.call_args
        assert args[1] == "claude-haiku-4-5"
