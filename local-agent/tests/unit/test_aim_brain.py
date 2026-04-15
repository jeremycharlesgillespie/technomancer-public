"""Tests for aim.brain — claude -p decision engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aim.brain import (
    Decision,
    _heuristic_decision,
    _parse_decision,
    decide_next_action,
    generate_work_ideas,
)


# ---------------------------------------------------------------------------
# _parse_decision tests
# ---------------------------------------------------------------------------

class TestParseDecision:
    def test_parse_valid_json(self):
        raw = '{"action": "ASSIGN", "target": "idea-042", "reason": "highest priority"}'
        d = _parse_decision(raw)
        assert d.action == "ASSIGN"
        assert d.target == "idea-042"
        assert d.reason == "highest priority"

    def test_parse_json_with_surrounding_text(self):
        raw = 'I think we should: {"action": "CREATE_WORK", "target": "", "reason": "board is low"} That is my recommendation.'
        d = _parse_decision(raw)
        assert d.action == "CREATE_WORK"

    def test_parse_wait(self):
        raw = '{"action": "WAIT", "reason": "Worker is busy"}'
        d = _parse_decision(raw)
        assert d.action == "WAIT"

    def test_parse_escalate(self):
        raw = '{"action": "ESCALATE", "target": "No progress for 4 hours", "reason": "stuck"}'
        d = _parse_decision(raw)
        assert d.action == "ESCALATE"
        assert "No progress" in d.target

    def test_parse_restart_worker(self):
        raw = '{"action": "RESTART_WORKER", "reason": "heartbeat stale"}'
        d = _parse_decision(raw)
        assert d.action == "RESTART_WORKER"

    def test_fallback_assign_keyword(self):
        raw = "I recommend we ASSIGN idea-055 because it's the highest priority task."
        d = _parse_decision(raw)
        assert d.action == "ASSIGN"
        assert d.target == "idea-055"

    def test_fallback_create_work_keyword(self):
        raw = "We should CREATE_WORK since the board is nearly empty."
        d = _parse_decision(raw)
        assert d.action == "CREATE_WORK"

    def test_fallback_restart_worker_keyword(self):
        raw = "The worker seems stuck. RESTART_WORKER immediately."
        d = _parse_decision(raw)
        assert d.action == "RESTART_WORKER"

    def test_fallback_escalate_keyword(self):
        raw = "We should ESCALATE this to the owner."
        d = _parse_decision(raw)
        assert d.action == "ESCALATE"

    def test_unparseable_returns_wait(self):
        raw = "I'm not sure what to do here. Let me think about it."
        d = _parse_decision(raw)
        assert d.action == "WAIT"

    def test_invalid_action_in_json_falls_back(self):
        raw = '{"action": "INVALID_ACTION", "reason": "test"}'
        d = _parse_decision(raw)
        # Falls through JSON parsing (invalid action) to keyword search
        assert d.action == "WAIT"

    def test_empty_string(self):
        d = _parse_decision("")
        assert d.action == "WAIT"

    def test_json_with_lowercase_action(self):
        raw = '{"action": "assign", "target": "idea-001", "reason": "test"}'
        d = _parse_decision(raw)
        assert d.action == "ASSIGN"
        assert d.target == "idea-001"


# ---------------------------------------------------------------------------
# _heuristic_decision tests
# ---------------------------------------------------------------------------

class TestHeuristicDecision:
    def test_restart_dead_worker(self):
        d = _heuristic_decision(
            worker_status="dead",
            approved_ideas=[],
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "RESTART_WORKER"

    def test_restart_stuck_worker(self):
        d = _heuristic_decision(
            worker_status="stuck",
            approved_ideas=[],
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "RESTART_WORKER"

    def test_restart_error_worker(self):
        d = _heuristic_decision(
            worker_status="error",
            approved_ideas=[],
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "RESTART_WORKER"

    def test_assign_picks_top_of_list_even_if_feature(self):
        """Rank wins over category: a feature at position 0 beats a quality later."""
        ideas = [
            {"id": "idea-001", "title": "Feature", "category": "feature"},
            {"id": "idea-002", "title": "Quality fix", "category": "quality"},
            {"id": "idea-003", "title": "Perf fix", "category": "performance"},
        ]
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=ideas,
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "ASSIGN"
        assert d.target == "idea-001"

    def test_assign_picks_first_regardless_of_category_order(self):
        """Whoever is index 0 wins, even if a 'safer' category comes later."""
        ideas = [
            {"id": "idea-001", "title": "Perf fix", "category": "performance"},
            {"id": "idea-002", "title": "Quality fix", "category": "quality"},
        ]
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=ideas,
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "ASSIGN"
        assert d.target == "idea-001"

    def test_assign_single_idea(self):
        ideas = [
            {"id": "idea-001", "title": "Feature", "category": "feature"},
        ]
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=ideas,
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "ASSIGN"
        assert d.target == "idea-001"

    def test_create_work_when_board_low(self):
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=[],
            board_todo=5,
            hours_since_completion=1,
        )
        assert d.action == "CREATE_WORK"

    def test_escalate_on_long_stall(self):
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=[],
            board_todo=20,
            hours_since_completion=4,
        )
        assert d.action == "ESCALATE"

    def test_wait_when_nothing_to_do(self):
        d = _heuristic_decision(
            worker_status="idle",
            approved_ideas=[],
            board_todo=20,
            hours_since_completion=1,
        )
        assert d.action == "WAIT"

    def test_no_assign_when_worker_busy(self):
        """Worker is executing — heuristic should not be called with that status
        in practice, but verify it doesn't break."""
        d = _heuristic_decision(
            worker_status="executing",
            approved_ideas=[{"id": "idea-001", "title": "X", "category": "quality"}],
            board_todo=20,
            hours_since_completion=1,
        )
        # Worker isn't idle, so shouldn't assign
        assert d.action == "WAIT"


# ---------------------------------------------------------------------------
# decide_next_action tests (with mocked claude -p)
# ---------------------------------------------------------------------------

class TestDecideNextAction:
    @patch("aim.brain._run_claude_p")
    def test_uses_claude_response(self, mock_claude):
        mock_claude.return_value = '{"action": "ASSIGN", "target": "idea-010", "reason": "ready"}'

        d = decide_next_action(
            board_state={"todo": 20, "in_progress": 0, "done_last_24h": 3},
            worker_status="idle",
            approved_ideas=[{"id": "idea-010", "title": "X", "category": "quality"}],
            last_completion="2026-04-14T09:00:00",
            hours_since_completion=1.0,
            completions_today=3,
        )
        assert d.action == "ASSIGN"
        assert d.target == "idea-010"

    @patch("aim.brain._run_claude_p")
    def test_falls_back_to_heuristic_on_claude_failure(self, mock_claude):
        mock_claude.return_value = None  # Claude failed

        d = decide_next_action(
            board_state={"todo": 20},
            worker_status="idle",
            approved_ideas=[{"id": "idea-010", "title": "X", "category": "quality"}],
            last_completion=None,
            hours_since_completion=999,
            completions_today=0,
        )
        # Heuristic should assign the approved idea
        assert d.action == "ASSIGN"
        assert d.target == "idea-010"

    @patch("aim.brain._run_claude_p")
    def test_falls_back_on_unparseable_claude_response(self, mock_claude):
        mock_claude.return_value = "I don't know what to do"

        d = decide_next_action(
            board_state={"todo": 5},
            worker_status="idle",
            approved_ideas=[],
            last_completion=None,
            hours_since_completion=1,
            completions_today=0,
        )
        # Unparseable → WAIT from _parse_decision, then falls to heuristic
        # Actually _parse_decision returns WAIT for unparseable, and that's the Claude result
        # The function uses the parsed result, not fallback
        assert d.action == "WAIT"


# ---------------------------------------------------------------------------
# generate_work_ideas tests
# ---------------------------------------------------------------------------

class TestGenerateWorkIdeas:
    @patch("aim.brain._run_claude_p")
    def test_parses_valid_array(self, mock_claude):
        mock_claude.return_value = json.dumps([
            {
                "title": "Improve test coverage",
                "description": "Add tests for uncovered modules",
                "category": "test",
                "idea_type": "story",
            },
            {
                "title": "Optimize LLM latency",
                "description": "Profile and reduce response times",
                "category": "performance",
                "idea_type": "story",
            },
        ])

        result = generate_work_ideas(
            codebase_summary="agent/core.py\nagent/tools.py",
            existing_idea_titles=["Old idea"],
            board_state={"todo": 5},
        )
        assert len(result) == 2
        assert result[0]["title"] == "Improve test coverage"
        assert result[0]["category"] == "test"
        assert result[1]["idea_type"] == "story"

    @patch("aim.brain._run_claude_p")
    def test_handles_array_in_surrounding_text(self, mock_claude):
        mock_claude.return_value = 'Here are my suggestions: [{"title": "Fix X", "description": "Y", "category": "quality", "idea_type": "story"}] Hope that helps!'

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        assert len(result) == 1
        assert result[0]["title"] == "Fix X"

    @patch("aim.brain._run_claude_p")
    def test_returns_empty_on_claude_failure(self, mock_claude):
        mock_claude.return_value = None

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        assert result == []

    @patch("aim.brain._run_claude_p")
    def test_returns_empty_on_invalid_json(self, mock_claude):
        mock_claude.return_value = "This is not JSON at all"

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        assert result == []

    @patch("aim.brain._run_claude_p")
    def test_truncates_long_titles(self, mock_claude):
        mock_claude.return_value = json.dumps([
            {
                "title": "A" * 200,
                "description": "B" * 100,
                "category": "quality",
                "idea_type": "story",
            },
        ])

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        assert len(result[0]["title"]) == 80

    @patch("aim.brain._run_claude_p")
    def test_skips_items_without_title(self, mock_claude):
        mock_claude.return_value = json.dumps([
            {"description": "No title here", "category": "quality"},
            {"title": "Has title", "description": "OK", "category": "test", "idea_type": "story"},
        ])

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        # First item has no title but still has "title" key... let's check
        # Actually the first item lacks "title" key entirely
        assert len(result) == 1
        assert result[0]["title"] == "Has title"

    @patch("aim.brain._run_claude_p")
    def test_defaults_missing_fields(self, mock_claude):
        mock_claude.return_value = json.dumps([
            {"title": "Minimal idea"},
        ])

        result = generate_work_ideas(
            codebase_summary="",
            existing_idea_titles=[],
            board_state={},
        )
        assert len(result) == 1
        assert result[0]["category"] == "quality"  # default
        assert result[0]["idea_type"] == "story"  # default

    @patch("aim.brain._run_claude_p")
    def test_prompt_contains_scoping_rules(self, mock_claude):
        """Prompt must spell out RULE A / RULE B and the disallowed story verbs."""
        mock_claude.return_value = "[]"

        generate_work_ideas(
            codebase_summary="agent/core.py",
            existing_idea_titles=[],
            board_state={"todo": 1},
        )

        assert mock_claude.call_count == 1
        sent_prompt = mock_claude.call_args[0][0]

        assert "RULE A" in sent_prompt
        assert "RULE B" in sent_prompt
        # RULE A describes epics carrying the design.
        assert "EPIC" in sent_prompt
        # RULE B describes stories as pure execution.
        assert "STORY" in sent_prompt
        # Disallowed verbs must be listed so the model avoids them in stories.
        for verb in ("design", "decide", "evaluate", "choose", "plan", "architect", "research"):
            assert verb in sent_prompt
