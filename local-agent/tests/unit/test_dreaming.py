"""Tests for agent/dreaming.py - Memory consolidation / dreaming system."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.dreaming import (
    apply_consolidation,
    build_consolidation_prompt,
    parse_consolidation_response,
    run_dream_cycle,
    is_dream_hour,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dream_env(tmp_path, monkeypatch):
    """Patch all dreaming paths to use temp directory."""
    import agent.dreaming as dream_module
    import agent.auto_memory as am_module

    identity_dir = tmp_path / "Permanent" / "identity"
    identity_dir.mkdir(parents=True)
    conversations_dir = tmp_path / "Conversations"
    conversations_dir.mkdir(parents=True)

    monkeypatch.setattr(dream_module, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(dream_module, "IDENTITY_DIR", identity_dir)
    monkeypatch.setattr(dream_module, "DREAM_LOG", tmp_path / "dream_log.md")
    monkeypatch.setattr(dream_module, "CONVERSATIONS_DIR", conversations_dir)
    monkeypatch.setattr(dream_module, "DREAM_LOCK", tmp_path / ".dream_lock")

    # Also patch auto_memory paths (used by apply_consolidation)
    monkeypatch.setattr(am_module, "IDENTITY_DIR", identity_dir)
    monkeypatch.setattr(am_module, "VAULT_PATH", tmp_path)

    return {"root": tmp_path, "identity": identity_dir, "conversations": conversations_dir}


@pytest.fixture
def sample_consolidation_response():
    """Mock LLM response with consolidation results."""
    return json.dumps({
        "new_facts": [
            {"category": "preferences", "fact": "Prefers dark themes in all IDEs", "confidence": "high"}
        ],
        "duplicates": [
            {"category": "skills", "remove": "Good at Python programming", "keep": "Expert Python developer with 10+ years experience"}
        ],
        "contradictions": [
            {"category": "goals", "old_fact": "Wants to learn Java", "correction": "No longer interested in Java, focusing on Rust"}
        ],
        "stale": [
            {"category": "experiences", "fact": "Currently interviewing at Google"}
        ],
        "summary": "Added dark theme preference, cleaned up skill duplicates, updated goals"
    })


@pytest.fixture
def mock_dream_agent(sample_consolidation_response):
    """Mock agent that returns consolidation results."""
    agent = MagicMock()
    agent.run = MagicMock(return_value=sample_consolidation_response)
    return agent


# =============================================================================
# PARSE RESPONSE TESTS
# =============================================================================


class TestParseConsolidationResponse:
    """Tests for parsing dreaming LLM responses."""

    def test_parse_valid_json(self, sample_consolidation_response):
        result = parse_consolidation_response(sample_consolidation_response)
        assert len(result["new_facts"]) == 1
        assert len(result["duplicates"]) == 1
        assert len(result["contradictions"]) == 1
        assert len(result["stale"]) == 1
        assert "dark theme" in result["summary"]

    def test_parse_json_in_code_block(self):
        response = '```json\n{"new_facts": [], "duplicates": [], "contradictions": [], "stale": [], "summary": "Nothing to consolidate"}\n```'
        result = parse_consolidation_response(response)
        assert result["summary"] == "Nothing to consolidate"

    def test_parse_invalid_json(self):
        result = parse_consolidation_response("This is not valid JSON at all")
        assert result["new_facts"] == []
        assert "parse" in result["summary"].lower() or "Failed" in result["summary"]

    def test_parse_partial_response(self):
        response = json.dumps({"new_facts": [{"category": "skills", "fact": "Knows Go", "confidence": "medium"}]})
        result = parse_consolidation_response(response)
        assert len(result["new_facts"]) == 1
        assert result["duplicates"] == []  # Missing keys get defaults


# =============================================================================
# BUILD PROMPT TESTS
# =============================================================================


class TestBuildConsolidationPrompt:
    """Tests for consolidation prompt construction."""

    def test_prompt_includes_logs(self):
        logs = [("2026-04-01", "User asked about Python\nBot answered about Python")]
        prompt = build_consolidation_prompt(logs, {})
        assert "2026-04-01" in prompt
        assert "Python" in prompt

    def test_prompt_includes_identity(self):
        identity = {"preferences": "# Preferences\n\n- Likes dark mode\n"}
        prompt = build_consolidation_prompt([], identity)
        assert "Likes dark mode" in prompt

    def test_prompt_handles_empty_data(self):
        prompt = build_consolidation_prompt([], {})
        assert "Empty" in prompt or "empty" in prompt or "No recent" in prompt


# =============================================================================
# APPLY CONSOLIDATION TESTS
# =============================================================================


class TestApplyConsolidation:
    """Tests for applying consolidation results."""

    def test_add_new_facts(self, temp_dream_env):
        result = {
            "new_facts": [{"category": "preferences", "fact": "Likes vim keybindings", "confidence": "high"}],
            "duplicates": [],
            "contradictions": [],
            "stale": [],
        }
        counts = apply_consolidation(result)
        assert counts["added"] == 1

        content = (temp_dream_env["identity"] / "preferences.md").read_text(encoding="utf-8")
        assert "vim keybindings" in content

    def test_remove_duplicates(self, temp_dream_env):
        # Pre-populate with duplicate
        (temp_dream_env["identity"] / "skills.md").write_text(
            "# Skills\n\n- Good at Python programming\n- Expert Python developer\n",
            encoding="utf-8",
        )

        result = {
            "new_facts": [],
            "duplicates": [{"category": "skills", "remove": "Good at Python programming", "keep": "Expert Python developer"}],
            "contradictions": [],
            "stale": [],
        }
        counts = apply_consolidation(result)
        assert counts["removed_duplicates"] == 1

        content = (temp_dream_env["identity"] / "skills.md").read_text(encoding="utf-8")
        assert "Good at Python programming" not in content
        assert "Expert Python developer" in content

    def test_fix_contradictions(self, temp_dream_env):
        (temp_dream_env["identity"] / "goals.md").write_text(
            "# Goals\n\n- Wants to learn Java\n", encoding="utf-8"
        )

        result = {
            "new_facts": [],
            "duplicates": [],
            "contradictions": [{"category": "goals", "old_fact": "Wants to learn Java", "correction": "Focusing on Rust instead"}],
            "stale": [],
        }
        counts = apply_consolidation(result)
        assert counts["fixed_contradictions"] == 1

        content = (temp_dream_env["identity"] / "goals.md").read_text(encoding="utf-8")
        assert "Java" not in content
        assert "Rust" in content

    def test_remove_stale(self, temp_dream_env):
        (temp_dream_env["identity"] / "experiences.md").write_text(
            "# Experiences\n\n- Currently interviewing at Google\n", encoding="utf-8"
        )

        result = {
            "new_facts": [],
            "duplicates": [],
            "contradictions": [],
            "stale": [{"category": "experiences", "fact": "Currently interviewing at Google"}],
        }
        counts = apply_consolidation(result)
        assert counts["removed_stale"] == 1

        content = (temp_dream_env["identity"] / "experiences.md").read_text(encoding="utf-8")
        assert "Google" not in content

    def test_empty_consolidation(self, temp_dream_env):
        result = {"new_facts": [], "duplicates": [], "contradictions": [], "stale": []}
        counts = apply_consolidation(result)
        assert all(v == 0 for v in counts.values())


# =============================================================================
# FULL DREAM CYCLE
# =============================================================================


class TestDreamCycle:
    """Tests for the full dream cycle."""

    def test_dream_cycle_completes(self, temp_dream_env, mock_dream_agent):
        # Create a conversation log so there's data to process
        log_file = temp_dream_env["conversations"] / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        log_file.write_text("### 10:00 - user\n**Q:** Hello\n**A:** Hi there\n", encoding="utf-8")

        result = asyncio.run(run_dream_cycle(mock_dream_agent))
        assert result["status"] == "completed"
        assert "counts" in result
        assert result["duration_seconds"] >= 0

    def test_dream_cycle_no_data(self, temp_dream_env):
        agent = MagicMock()
        agent.run = MagicMock(return_value='{"new_facts":[],"duplicates":[],"contradictions":[],"stale":[],"summary":"Nothing"}')
        result = asyncio.run(run_dream_cycle(agent))
        # Should still complete even with no conversation logs
        assert result["status"] in ("completed", "skipped")

    def test_dream_cycle_llm_error(self, temp_dream_env):
        # Add a conversation log so it doesn't skip for "no data"
        log_file = temp_dream_env["conversations"] / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        log_file.write_text("### 10:00 - user\n**Q:** test\n**A:** test\n", encoding="utf-8")

        agent = MagicMock()
        agent.run = MagicMock(side_effect=Exception("LLM crashed"))
        result = asyncio.run(run_dream_cycle(agent))
        assert result["status"] == "error"

    def test_dream_log_created(self, temp_dream_env, mock_dream_agent):
        log_file = temp_dream_env["conversations"] / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        log_file.write_text("### 10:00 - user\n**Q:** test\n**A:** reply\n", encoding="utf-8")

        asyncio.run(run_dream_cycle(mock_dream_agent))

        import agent.dreaming as dm
        assert dm.DREAM_LOG.exists()
        content = dm.DREAM_LOG.read_text(encoding="utf-8")
        assert "Dream Consolidation" in content


# =============================================================================
# UTILITY
# =============================================================================


class TestDreamUtility:
    """Tests for dreaming utility functions."""

    def test_is_dream_hour(self, monkeypatch):
        import agent.dreaming as dm
        # Force dream hours to include current hour for testing
        monkeypatch.setattr(dm, "DREAM_START_HOUR", 0)
        monkeypatch.setattr(dm, "DREAM_END_HOUR", 24)
        assert is_dream_hour() is True

    def test_is_not_dream_hour(self, monkeypatch):
        import agent.dreaming as dm
        # Set to impossible hours
        monkeypatch.setattr(dm, "DREAM_START_HOUR", 25)
        monkeypatch.setattr(dm, "DREAM_END_HOUR", 26)
        assert is_dream_hour() is False
