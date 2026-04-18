"""Tests for aimm.suggester — approval recommendations audit log.

Verifies:
    * Acceptance: 3 pending stories → 3 findings entries + 3 decisions rows.
    * Idempotency: re-running on the same story is a no-op.
    * No Jira mutations anywhere in the module (imports + runtime).
    * Failure paths (missing rubric, LLM error, parse failure, missing
      story key) return a Suggestion with the matching reason code and
      append nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimm import decisions as aimm_decisions
from aimm import suggester
from aimm.suggester import (
    FINDINGS_HEADING,
    SUGGEST_APPROVE,
    SUGGEST_SKIP,
    Suggestion,
    _already_suggested,
    _build_prompt,
    _parse_response,
    suggest_approval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_RUBRIC = (
    "# Paper-Worthiness Rubric\n\n"
    "Criteria: concrete, reproducible, narrative-forward, observable metric."
)


@pytest.fixture
def tmp_findings(tmp_path):
    return tmp_path / "raw_findings.md"


@pytest.fixture
def tmp_decisions_log(tmp_path, monkeypatch):
    """Redirect the decisions.jsonl log to a per-test temp file."""
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(aimm_decisions, "LOG_DIR", tmp_path)
    monkeypatch.setattr(aimm_decisions, "LOG_FILE", log_file)
    return log_file


@pytest.fixture
def three_pending_stories():
    return [
        {
            "key": "TK-501",
            "title": "Stall-detector race in worker",
            "description": "Exposes a latent race between stall timer and subprocess completion.",
            "category": "quality",
            "labels": ["pending-approval", "cat:quality"],
        },
        {
            "key": "TK-502",
            "title": "Rename a private helper in tools.py",
            "description": "Pure rename, no behavior change.",
            "category": "quality",
            "labels": ["pending-approval"],
        },
        {
            "key": "TK-503",
            "title": "Measure commits-per-day across both projects",
            "description": "Adds a SQL query + CLI command that reports commit volume.",
            "category": "feature",
            "labels": ["pending-approval", "cat:feature"],
        },
    ]


def _claude_ok(result_text: str) -> dict:
    return {
        "success": True,
        "result": result_text,
        "cost_usd": 0.0,
        "session_id": "sess-1",
        "duration": 0.1,
        "error": None,
    }


def _claude_err(error: str = "boom") -> dict:
    return {
        "success": False,
        "result": "",
        "cost_usd": 0.0,
        "session_id": "",
        "duration": 0.1,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Suggestion dataclass
# ---------------------------------------------------------------------------


class TestSuggestion:
    def test_defaults(self):
        s = Suggestion()
        assert s.story_key == ""
        assert s.recommend is False
        assert s.reasoning == ""
        assert s.reason == ""

    def test_to_dict_round_trip(self):
        s = Suggestion(
            story_key="TK-1",
            recommend=True,
            reasoning="meets all four criteria",
            reason="ok",
        )
        assert s.to_dict() == {
            "story_key": "TK-1",
            "recommend": True,
            "reasoning": "meets all four criteria",
            "reason": "ok",
        }


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_recommend(self):
        raw = '{"recommend": true, "reasoning": "meets all four criteria"}'
        s = _parse_response(raw, "TK-1")
        assert s.recommend is True
        assert s.reasoning == "meets all four criteria"
        assert s.reason == "ok"
        assert s.story_key == "TK-1"

    def test_valid_skip(self):
        raw = '{"recommend": false, "reasoning": "no observable metric"}'
        s = _parse_response(raw, "TK-2")
        assert s.recommend is False
        assert s.reasoning == "no observable metric"
        assert s.reason == "ok"

    def test_json_embedded_in_prose(self):
        raw = 'Verdict:\n{"recommend": true, "reasoning": "r"}\nend.'
        s = _parse_response(raw, "TK-3")
        assert s.recommend is True
        assert s.reasoning == "r"

    def test_malformed_json_returns_parse_failure(self):
        s = _parse_response("{not valid", "TK-4")
        assert s.reason == "parse_failure"
        assert s.recommend is False
        assert s.story_key == "TK-4"

    def test_no_json_block_returns_parse_failure(self):
        assert _parse_response("no json here", "TK-5").reason == "parse_failure"

    def test_empty_string_returns_parse_failure(self):
        assert _parse_response("", "TK-6").reason == "parse_failure"

    def test_non_string_returns_parse_failure(self):
        assert _parse_response(None, "TK-7").reason == "parse_failure"  # type: ignore[arg-type]

    def test_json_array_returns_parse_failure(self):
        assert _parse_response("[1,2]", "TK-8").reason == "parse_failure"

    def test_null_reasoning_coerced_to_empty(self):
        s = _parse_response('{"recommend": true, "reasoning": null}', "TK-9")
        assert s.reasoning == ""


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_rubric_and_story(self, three_pending_stories):
        prompt = _build_prompt(SAMPLE_RUBRIC, three_pending_stories[0])
        assert SAMPLE_RUBRIC in prompt
        assert "TK-501" in prompt
        assert "Stall-detector race" in prompt
        assert "pending-approval" in prompt
        assert "recommend" in prompt

    def test_prompt_tolerates_missing_fields(self):
        prompt = _build_prompt(SAMPLE_RUBRIC, {})
        assert "Key: " in prompt
        assert "Title: " in prompt

    def test_prompt_truncates_long_description(self):
        huge = "x" * 20_000
        prompt = _build_prompt(SAMPLE_RUBRIC, {"key": "TK-1", "description": huge})
        assert "truncated" in prompt
        assert prompt.count("x") < 20_000

    def test_prompt_accepts_id_and_summary_aliases(self):
        story = {"id": "TK-77", "summary": "alt title"}
        prompt = _build_prompt(SAMPLE_RUBRIC, story)
        assert "TK-77" in prompt
        assert "alt title" in prompt


# ---------------------------------------------------------------------------
# _already_suggested
# ---------------------------------------------------------------------------


class TestAlreadySuggested:
    def test_returns_false_when_log_missing(self, tmp_decisions_log):
        assert not tmp_decisions_log.exists()
        assert _already_suggested("TK-999") is False

    def test_returns_false_for_unknown_key(self, tmp_decisions_log):
        aimm_decisions.log_decision("TK-1", SUGGEST_APPROVE, "r")
        assert _already_suggested("TK-999") is False

    def test_returns_true_for_suggest_approve(self, tmp_decisions_log):
        aimm_decisions.log_decision("TK-1", SUGGEST_APPROVE, "r")
        assert _already_suggested("TK-1") is True

    def test_returns_true_for_suggest_skip(self, tmp_decisions_log):
        aimm_decisions.log_decision("TK-2", SUGGEST_SKIP, "r")
        assert _already_suggested("TK-2") is True

    def test_ignores_non_suggest_actions(self, tmp_decisions_log):
        """Existing approve/archive/draft records don't block fresh scoring."""
        aimm_decisions.log_decision("TK-3", "approve", "r")
        aimm_decisions.log_decision("TK-3", "draft", "r")
        assert _already_suggested("TK-3") is False

    def test_empty_key_returns_false(self, tmp_decisions_log):
        aimm_decisions.log_decision("", SUGGEST_APPROVE, "r")
        assert _already_suggested("") is False

    def test_malformed_lines_do_not_break_scan(self, tmp_decisions_log):
        aimm_decisions.log_decision("TK-1", SUGGEST_APPROVE, "r")
        with tmp_decisions_log.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n\n")
        assert _already_suggested("TK-1") is True


# ---------------------------------------------------------------------------
# suggest_approval — end-to-end
# ---------------------------------------------------------------------------


class TestSuggestApproval:
    def test_acceptance_three_stories_three_findings_three_audit_rows(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        """Acceptance criterion: 3 pending stories → 3 findings entries +
        3 decisions.jsonl entries."""
        responses = [
            _claude_ok('{"recommend": true, "reasoning": "rubric pass 501"}'),
            _claude_ok('{"recommend": false, "reasoning": "pure rename 502"}'),
            _claude_ok('{"recommend": true, "reasoning": "rubric pass 503"}'),
        ]
        with patch(
            "aimm.suggester.run_claude_prompt", side_effect=responses
        ) as mock_run:
            results = [
                suggest_approval(
                    story, SAMPLE_RUBRIC, findings_path=tmp_findings
                )
                for story in three_pending_stories
            ]

        assert mock_run.call_count == 3
        assert [r.reason for r in results] == ["ok", "ok", "ok"]
        assert [r.recommend for r in results] == [True, False, True]

        # findings file: heading + one entry per story
        assert tmp_findings.exists()
        content = tmp_findings.read_text(encoding="utf-8")
        assert FINDINGS_HEADING in content
        assert content.count("### ") == 3
        for key in ("TK-501", "TK-502", "TK-503"):
            assert key in content
        assert "RECOMMEND APPROVE" in content
        assert "SKIP" in content

        # decisions.jsonl: one row per story
        lines = [
            line for line in tmp_decisions_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 3
        actions = [json.loads(line)["action"] for line in lines]
        assert actions == [SUGGEST_APPROVE, SUGGEST_SKIP, SUGGEST_APPROVE]
        story_keys = [json.loads(line)["story_key"] for line in lines]
        assert story_keys == ["TK-501", "TK-502", "TK-503"]

    def test_idempotent_second_call_is_noop(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        """Acceptance criterion: re-running on the same story does not
        duplicate the suggestion."""
        story = three_pending_stories[0]
        first_fake = _claude_ok(
            '{"recommend": true, "reasoning": "first run"}'
        )
        with patch(
            "aimm.suggester.run_claude_prompt", return_value=first_fake
        ) as mock_run:
            first = suggest_approval(
                story, SAMPLE_RUBRIC, findings_path=tmp_findings
            )
        assert first.reason == "ok"
        assert mock_run.call_count == 1

        # Snapshot findings + decisions state after first run.
        findings_after_first = tmp_findings.read_text(encoding="utf-8")
        decisions_after_first = tmp_decisions_log.read_text(encoding="utf-8")

        # Second call: must not invoke claude -p and must not append.
        with patch("aimm.suggester.run_claude_prompt") as mock_run2:
            second = suggest_approval(
                story, SAMPLE_RUBRIC, findings_path=tmp_findings
            )
        assert second.reason == "already_suggested"
        assert second.recommend is False
        mock_run2.assert_not_called()

        assert tmp_findings.read_text(encoding="utf-8") == findings_after_first
        assert tmp_decisions_log.read_text(encoding="utf-8") == decisions_after_first

    def test_missing_story_key_returns_missing_key(
        self, tmp_findings, tmp_decisions_log
    ):
        with patch("aimm.suggester.run_claude_prompt") as mock_run:
            s = suggest_approval(
                {"title": "no key"}, SAMPLE_RUBRIC, findings_path=tmp_findings
            )
        assert s.reason == "missing_story_key"
        mock_run.assert_not_called()
        assert not tmp_findings.exists()
        assert not tmp_decisions_log.exists()

    def test_missing_rubric_returns_rubric_unavailable(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch("aimm.suggester.run_claude_prompt") as mock_run:
            s = suggest_approval(
                three_pending_stories[0], "", findings_path=tmp_findings
            )
        assert s.reason == "rubric_unavailable"
        mock_run.assert_not_called()
        assert not tmp_findings.exists()
        assert not tmp_decisions_log.exists()

    def test_whitespace_only_rubric_returns_rubric_unavailable(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch("aimm.suggester.run_claude_prompt") as mock_run:
            s = suggest_approval(
                three_pending_stories[0], "   \n\t  ", findings_path=tmp_findings
            )
        assert s.reason == "rubric_unavailable"
        mock_run.assert_not_called()

    def test_claude_failure_returns_llm_error_and_writes_nothing(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch(
            "aimm.suggester.run_claude_prompt", return_value=_claude_err()
        ):
            s = suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        assert s.reason == "llm_error"
        assert s.recommend is False
        assert not tmp_findings.exists()
        assert not tmp_decisions_log.exists()

    def test_claude_raises_returns_llm_error(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch(
            "aimm.suggester.run_claude_prompt",
            side_effect=RuntimeError("subprocess exploded"),
        ):
            s = suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        assert s.reason == "llm_error"
        assert not tmp_findings.exists()

    def test_parse_failure_writes_nothing(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch(
            "aimm.suggester.run_claude_prompt",
            return_value=_claude_ok("definitely not json"),
        ):
            s = suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        assert s.reason == "parse_failure"
        assert not tmp_findings.exists()
        assert not tmp_decisions_log.exists()

    def test_non_dict_result_returns_llm_error(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch("aimm.suggester.run_claude_prompt", return_value="oops"):
            s = suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        assert s.reason == "llm_error"

    def test_findings_heading_created_only_once(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        responses = [
            _claude_ok(f'{{"recommend": true, "reasoning": "r{i}"}}')
            for i in range(3)
        ]
        with patch("aimm.suggester.run_claude_prompt", side_effect=responses):
            for story in three_pending_stories:
                suggest_approval(
                    story, SAMPLE_RUBRIC, findings_path=tmp_findings
                )
        content = tmp_findings.read_text(encoding="utf-8")
        assert content.count(FINDINGS_HEADING) == 1

    def test_cycle_id_propagated_to_decisions_log(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        with patch(
            "aimm.suggester.run_claude_prompt",
            return_value=_claude_ok('{"recommend": true, "reasoning": "r"}'),
        ):
            suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
                cycle_id="aimm-20260418-001",
            )
        record = json.loads(tmp_decisions_log.read_text(encoding="utf-8"))
        assert record["cycle_id"] == "aimm-20260418-001"

    def test_appends_to_existing_findings_preserving_content(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        """If raw_findings.md already has unrelated content, append the
        heading + entry at the end rather than clobbering it."""
        tmp_findings.write_text(
            "# Raw Findings\n\nSome pre-existing narrative.\n",
            encoding="utf-8",
        )
        with patch(
            "aimm.suggester.run_claude_prompt",
            return_value=_claude_ok('{"recommend": true, "reasoning": "r"}'),
        ):
            suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        content = tmp_findings.read_text(encoding="utf-8")
        assert "Some pre-existing narrative." in content
        assert FINDINGS_HEADING in content
        assert "TK-501" in content


# ---------------------------------------------------------------------------
# Contract: no Jira mutations anywhere in the module
# ---------------------------------------------------------------------------


class TestNoJiraMutations:
    """Suggester is a researcher, not a manager. It must not import any
    Jira-mutation code path, and a mocked provider passed through the
    suggestion flow must receive zero mutating calls.
    """

    def test_module_does_not_import_jira_sync(self):
        src = Path(suggester.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from idea_board.jira_sync",
            "import idea_board.jira_sync",
            "from idea_board import jira_sync",
            "jira_sync.",
            "JiraProvider",
        ):
            assert forbidden not in src, (
                f"suggester.py must not reference {forbidden!r} — AIMM "
                "does not mutate Jira, only appends to findings/audit."
            )

    def test_module_does_not_call_provider_mutators(self):
        """Belt-and-suspenders: even if a provider is imported later, the
        source must not call any mutating method on it."""
        src = Path(suggester.__file__).read_text(encoding="utf-8")
        for forbidden in (
            ".mark_executing(",
            ".mark_done(",
            ".mark_failed(",
            ".vote(",
            ".add_comment(",
            ".delete(",
            "add_label",
            "remove_label",
            "transition_issue",
            "put(",
            "PUT ",
        ):
            assert forbidden not in src, (
                f"suggester.py must not use mutating call {forbidden!r}."
            )

    def test_mocked_provider_receives_zero_put_calls(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        """Acceptance criterion: mocked provider asserts zero PUT calls.

        Wire a BoardProvider mock into the path even though suggester
        never receives one — and verify that end-to-end suggestion runs
        issue no mutating calls against it. This catches regressions
        where a future refactor accidentally threads a provider through.
        """
        provider = MagicMock(name="BoardProvider")

        with patch(
            "aimm.suggester.run_claude_prompt",
            return_value=_claude_ok(
                '{"recommend": true, "reasoning": "r"}'
            ),
        ):
            for story in three_pending_stories:
                suggest_approval(
                    story, SAMPLE_RUBRIC, findings_path=tmp_findings
                )

        # Every mutating method on the provider must have received zero
        # calls — suggester never threads a provider through.
        for attr in (
            "mark_executing",
            "mark_done",
            "mark_failed",
            "vote",
            "add",
            "add_comment",
            "delete",
            "set_execution_order",
            "set_epic_context",
        ):
            assert getattr(provider, attr).call_count == 0, (
                f"provider.{attr} was called by suggest_approval"
            )
        # And no PUT-shaped HTTP call was made on any attribute path.
        assert provider.put.call_count == 0
        assert provider.patch.call_count == 0

    def test_suggest_does_not_import_jira_at_runtime(
        self, three_pending_stories, tmp_findings, tmp_decisions_log
    ):
        before = {m for m in sys.modules if "jira" in m.lower()}
        with patch(
            "aimm.suggester.run_claude_prompt",
            return_value=_claude_ok(
                '{"recommend": false, "reasoning": "skip"}'
            ),
        ):
            suggest_approval(
                three_pending_stories[0],
                SAMPLE_RUBRIC,
                findings_path=tmp_findings,
            )
        after = {m for m in sys.modules if "jira" in m.lower()}
        assert after == before, (
            "suggest_approval must not import any jira module at runtime"
        )
