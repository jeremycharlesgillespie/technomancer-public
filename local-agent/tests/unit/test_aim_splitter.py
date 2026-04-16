"""Tests for aim.splitter — auto-decomposer for failed stories."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aim.splitter import (
    MIN_SPLITS,
    ProposedStory,
    SplitResult,
    SPLITTER_SOURCE,
    _apply_split,
    _build_context,
    _build_prompt,
    _count_prior_failures,
    _parse_split_response,
    _pick_relevant_files,
    _propose_splits,
    _should_split,
    evaluate_failure,
    scan_and_split,
)
from board.provider import Comment
from idea_board.models import Idea


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_item(
    item_id: str = "TK-100",
    title: str = "Do big thing",
    description: str = "Requires touching five modules and adding a new table.",
    source: str = "llm_analysis",
    category: str = "quality",
    idea_type: str = "story",
    state: str = "failed",
    parent_id: str | None = None,
    execution_log: str | None = None,
) -> Idea:
    return Idea(
        id=item_id,
        title=title,
        description=description,
        source=source,
        category=category,
        idea_type=idea_type,
        state=state,
        parent_id=parent_id,
        execution_log=execution_log,
    )


def failure_comments(n: int) -> list[Comment]:
    return [
        Comment(
            author="executor",
            text=f"[Execution Log - Failed]\nattempt {i}: tests blew up",
            created=f"2026-04-16T0{i}:00:00",
            marker="[Execution Log - Failed]",
        )
        for i in range(n)
    ]


class FakeProvider:
    """In-memory BoardProvider for splitter tests.

    Implements only the surface the splitter uses: get, load_all,
    list_by_state, get_comments, add, delete, add_comment, vote.
    """

    def __init__(self):
        self.items: dict[str, Idea] = {}
        self.comments: dict[str, list[Comment]] = {}
        self.votes: list[tuple[str, str, str]] = []
        self.add_side_effect = None
        self._counter = 0

    # seed / test helpers
    def seed(self, item: Idea, comments: list[Comment] | None = None) -> None:
        self.items[item.id] = item
        self.comments[item.id] = list(comments or [])

    # provider API
    def get(self, item_id: str) -> Idea | None:
        return self.items.get(item_id)

    def load_all(self) -> list[Idea]:
        return list(self.items.values())

    def list_by_state(self, state: str) -> list[Idea]:
        return [i for i in self.items.values() if i.state == state]

    def get_comments(self, item_id: str) -> list[Comment]:
        return list(self.comments.get(item_id, []))

    def add(
        self,
        title: str,
        description: str,
        source: str = "llm_analysis",
        category: str = "feature",
        idea_type: str = "story",
        parent_id: str | None = None,
    ) -> Idea:
        if self.add_side_effect is not None:
            exc = self.add_side_effect(title)
            if exc:
                raise exc
        self._counter += 1
        new = make_item(
            item_id=f"TK-NEW{self._counter}",
            title=title,
            description=description,
            source=source,
            category=category,
            idea_type=idea_type,
            state="approved",
            parent_id=parent_id,
        )
        self.items[new.id] = new
        return new

    def delete(self, item_id: str) -> bool:
        if item_id in self.items:
            del self.items[item_id]
            self.comments.pop(item_id, None)
            return True
        return False

    def add_comment(self, item_id: str, author: str, text: str) -> Idea | None:
        if item_id not in self.items:
            return None
        from board.provider import parse_marker
        self.comments.setdefault(item_id, []).append(
            Comment(author=author, text=text, created="", marker=parse_marker(text))
        )
        return self.items[item_id]

    def vote(self, item_id: str, voter: str, value: str) -> Idea | None:
        self.votes.append((item_id, voter, value))
        return self.items.get(item_id)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_seen_state(tmp_path, monkeypatch):
    """Redirect the persisted seen-state file to a temp path per-test."""
    monkeypatch.setattr("aim.splitter._STATE_FILE", tmp_path / "seen.json")


@pytest.fixture
def provider():
    return FakeProvider()


# ---------------------------------------------------------------------------
# _count_prior_failures
# ---------------------------------------------------------------------------


class TestCountPriorFailures:
    def test_zero_markers_no_log(self):
        assert _count_prior_failures([], None) == 1  # failed state w/o trace → 1

    def test_one_marker(self):
        assert _count_prior_failures(failure_comments(1)) == 1

    def test_two_markers(self):
        assert _count_prior_failures(failure_comments(2)) == 2

    def test_three_markers(self):
        assert _count_prior_failures(failure_comments(3)) == 3

    def test_markers_ignored_if_not_failed_type(self):
        comments = [
            Comment(author="a", text="[Execution Log]\nok", created="", marker="[Execution Log]"),
            Comment(author="a", text="[AIM Progress]\nwip", created="", marker="[AIM Progress]"),
        ]
        # No failed markers → falls back to execution_log heuristic (1).
        assert _count_prior_failures(comments, None) == 1

    def test_fallback_to_execution_log_separators(self):
        log = "=== Attempt 1 ===\noops\n=== Attempt 2 ===\nfailed again"
        assert _count_prior_failures([], log) == 2


# ---------------------------------------------------------------------------
# _should_split
# ---------------------------------------------------------------------------


class TestShouldSplit:
    def test_epic_exempt(self):
        item = make_item(idea_type="epic")
        ok, reason = _should_split(item, prior_failures=5)
        assert not ok
        assert reason == "exempt_epic"

    def test_splitter_child_skipped(self):
        item = make_item(source=SPLITTER_SOURCE)
        ok, reason = _should_split(item, prior_failures=2)
        assert not ok
        assert reason == "source_splitter_skip"

    def test_normal_story_fires(self):
        item = make_item()
        ok, reason = _should_split(item, prior_failures=1)
        assert ok
        assert reason == "split_due"

    def test_any_failure_count_fires(self):
        item = make_item()
        ok, _ = _should_split(item, prior_failures=3)
        assert ok


# ---------------------------------------------------------------------------
# _parse_split_response
# ---------------------------------------------------------------------------


class TestParseSplitResponse:
    def test_valid_three_splits(self):
        raw = json.dumps({
            "splits": [
                {"title": "A", "description": "do a thing in file.py"},
                {"title": "B", "description": "do b thing"},
                {"title": "C", "description": "do c thing"},
            ],
            "rationale": "split by file",
        })
        splits, rationale = _parse_split_response(raw)
        assert len(splits) == 3
        assert splits[0].title == "A"
        assert splits[0].description == "do a thing in file.py"
        assert rationale == "split by file"

    def test_json_wrapped_in_prose(self):
        raw = 'Here is my split: {"splits": [{"title": "X", "description": "y"}, {"title": "X2", "description": "y2"}], "rationale": "r"} Thanks.'
        splits, _ = _parse_split_response(raw)
        assert len(splits) == 2

    def test_empty_splits_ok(self):
        raw = '{"splits": [], "rationale": "tooling issue, not scope"}'
        splits, rationale = _parse_split_response(raw)
        assert splits == []
        assert "tooling" in rationale

    def test_missing_splits_key_raises(self):
        raw = '{"rationale": "nope"}'
        with pytest.raises(ValueError, match="missing 'splits'"):
            _parse_split_response(raw)

    def test_missing_description_raises(self):
        raw = '{"splits": [{"title": "only title"}], "rationale": ""}'
        with pytest.raises(ValueError, match="missing description"):
            _parse_split_response(raw)

    def test_missing_title_raises(self):
        raw = '{"splits": [{"description": "orphan desc"}], "rationale": ""}'
        with pytest.raises(ValueError, match="missing title"):
            _parse_split_response(raw)

    def test_empty_raw_raises(self):
        with pytest.raises(ValueError):
            _parse_split_response("")

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _parse_split_response("just prose, no braces at all")

    def test_handles_nested_objects(self):
        # Description contains braces — parser must not bail.
        raw = '{"splits": [{"title": "T1", "description": "use {json: 1} shape"}, {"title": "T2", "description": "normal"}], "rationale": ""}'
        splits, _ = _parse_split_response(raw)
        assert len(splits) == 2
        assert "{json: 1}" in splits[0].description


# ---------------------------------------------------------------------------
# _propose_splits (LLM integration seam)
# ---------------------------------------------------------------------------


class TestProposeSplits:
    def _ctx(self):
        return {
            "parent_key": "TK-100",
            "title": "t",
            "description": "d",
            "epic_context": "",
            "sibling_titles": [],
            "failure_tail": "",
            "files": [],
        }

    def test_valid_runner_returns_splits(self):
        def runner(_prompt, _timeout):
            return json.dumps({
                "splits": [
                    {"title": "a", "description": "x"},
                    {"title": "b", "description": "y"},
                ],
                "rationale": "ok",
            })
        splits, rationale = _propose_splits(self._ctx(), claude_runner=runner)
        assert len(splits) == 2
        assert rationale == "ok"

    def test_runner_returns_none(self):
        splits, rationale = _propose_splits(
            self._ctx(), claude_runner=lambda _p, _t: None,
        )
        assert splits == []
        assert rationale.startswith("error:")

    def test_runner_returns_bad_json(self):
        splits, rationale = _propose_splits(
            self._ctx(), claude_runner=lambda _p, _t: "nope",
        )
        assert splits == []
        assert "parse_failed" in rationale

    def test_hard_cap_truncates(self):
        many = {
            "splits": [{"title": f"t{i}", "description": f"d{i}"} for i in range(10)],
            "rationale": "too many",
        }
        splits, _ = _propose_splits(
            self._ctx(), claude_runner=lambda _p, _t: json.dumps(many),
        )
        assert len(splits) == 5  # MAX_SPLITS


# ---------------------------------------------------------------------------
# _apply_split
# ---------------------------------------------------------------------------


class TestApplySplit:
    def test_creates_and_deletes_parent(self, provider):
        parent = make_item("TK-100", parent_id="TK-EPIC")
        provider.seed(parent)
        splits = [
            ProposedStory(title="A", description="da"),
            ProposedStory(title="B", description="db"),
            ProposedStory(title="C", description="dc"),
        ]
        new_keys, err = _apply_split(parent, splits, provider)
        assert err is None
        assert len(new_keys) == 3
        assert provider.get("TK-100") is None
        for key in new_keys:
            child = provider.get(key)
            assert child is not None
            assert child.source == SPLITTER_SOURCE
            assert child.category == "quality"
            assert child.parent_id == "TK-EPIC"

    def test_rolls_back_when_too_few_succeed(self, provider):
        parent = make_item("TK-100")
        provider.seed(parent)

        def fail_on_second(title):
            if title == "B":
                return RuntimeError("Jira down")
            return None

        provider.add_side_effect = fail_on_second
        splits = [
            ProposedStory(title="A", description="da"),
            ProposedStory(title="B", description="db"),
        ]
        new_keys, err = _apply_split(parent, splits, provider)
        assert new_keys == []
        assert "add_failed" in err
        # Parent still exists, rollback deleted the one that succeeded.
        assert provider.get("TK-100") is not None
        assert len([i for i in provider.load_all() if i.id != "TK-100"]) == 0

    def test_partial_success_meets_minimum(self, provider):
        parent = make_item("TK-100")
        provider.seed(parent)

        def fail_last(title):
            if title == "D":
                return RuntimeError("boom")
            return None

        provider.add_side_effect = fail_last
        splits = [
            ProposedStory(title="A", description="da"),
            ProposedStory(title="B", description="db"),
            ProposedStory(title="C", description="dc"),
            ProposedStory(title="D", description="dd"),
        ]
        new_keys, err = _apply_split(parent, splits, provider)
        # 3 of 4 created, >= MIN_SPLITS, so we commit and delete parent.
        assert len(new_keys) == 3
        assert err is None
        assert provider.get("TK-100") is None


# ---------------------------------------------------------------------------
# evaluate_failure (integration-like, still unit-level)
# ---------------------------------------------------------------------------


class TestEvaluateFailure:
    def test_not_failed_short_circuits(self, provider):
        provider.seed(make_item(state="done"))
        result = evaluate_failure("TK-100", provider=provider)
        assert not result.fired
        assert result.reason == "not_failed"

    def test_missing_item(self, provider):
        result = evaluate_failure("TK-404", provider=provider)
        assert not result.fired
        assert result.reason == "not_found"

    def test_epic_exempt(self, provider):
        provider.seed(make_item(idea_type="epic"), comments=failure_comments(3))
        result = evaluate_failure("TK-100", provider=provider)
        assert not result.fired
        assert result.reason == "exempt_epic"

    def test_full_flow_creates_and_deletes(self, provider):
        provider.seed(make_item(), comments=failure_comments(2))
        runner = lambda _p, _t: json.dumps({
            "splits": [
                {"title": "step1", "description": "do first part"},
                {"title": "step2", "description": "do second part"},
                {"title": "step3", "description": "do third part"},
            ],
            "rationale": "split by concern",
        })
        notify = MagicMock()
        result = evaluate_failure(
            "TK-100", provider=provider, claude_runner=runner, notifier=notify,
        )
        assert result.fired
        assert result.reason == "split_done"
        assert len(result.new_keys) == 3
        assert provider.get("TK-100") is None
        notify.assert_called_once()
        # Rationale flows through.
        args = notify.call_args
        assert args.args[2] == "split by concern"

    def test_llm_returns_no_splits(self, provider):
        provider.seed(make_item(), comments=failure_comments(2))
        runner = lambda _p, _t: json.dumps({"splits": [], "rationale": "tooling"})
        result = evaluate_failure("TK-100", provider=provider, claude_runner=runner)
        assert not result.fired
        assert result.reason == "llm_declined"
        # Parent preserved.
        assert provider.get("TK-100") is not None

    def test_llm_unavailable_preserves_parent(self, provider):
        provider.seed(make_item(), comments=failure_comments(2))
        runner = lambda _p, _t: None
        result = evaluate_failure("TK-100", provider=provider, claude_runner=runner)
        assert not result.fired
        assert "llm_unavailable" in result.reason
        assert provider.get("TK-100") is not None

    def test_too_few_splits_preserves_parent(self, provider):
        provider.seed(make_item(), comments=failure_comments(2))
        runner = lambda _p, _t: json.dumps({
            "splits": [{"title": "only", "description": "solo"}],
            "rationale": "",
        })
        result = evaluate_failure("TK-100", provider=provider, claude_runner=runner)
        assert not result.fired
        assert result.reason == "too_few_splits"
        assert provider.get("TK-100") is not None


# ---------------------------------------------------------------------------
# Anti-loop (splitter-child protection)
# ---------------------------------------------------------------------------


class TestAntiLoop:
    def test_splitter_child_auto_vetoes_on_any_failure(self, provider):
        provider.seed(
            make_item(source=SPLITTER_SOURCE),
            comments=failure_comments(1),
        )
        result = evaluate_failure("TK-100", provider=provider)
        assert not result.fired
        assert result.reason == "auto_veto"
        assert ("TK-100", "owner", "veto") in provider.votes
        veto_comments = [
            c for c in provider.get_comments("TK-100")
            if "Auto-vetoed" in c.text
        ]
        assert len(veto_comments) == 1

    def test_splitter_child_multi_failure_also_vetoes(self, provider):
        provider.seed(
            make_item(source=SPLITTER_SOURCE),
            comments=failure_comments(3),
        )
        result = evaluate_failure("TK-100", provider=provider)
        assert not result.fired
        assert result.reason == "auto_veto"


# ---------------------------------------------------------------------------
# scan_and_split — dedup via seen-state file
# ---------------------------------------------------------------------------


class TestScanAndSplit:
    def test_skips_already_seen(self, provider, tmp_path, monkeypatch):
        state_file = tmp_path / "seen.json"
        state_file.write_text(json.dumps(["TK-100"]))
        monkeypatch.setattr("aim.splitter._STATE_FILE", state_file)
        provider.seed(make_item(), comments=failure_comments(3))

        called = []
        monkeypatch.setattr(
            "aim.splitter.evaluate_failure",
            lambda *a, **k: called.append(a) or SplitResult(True, "split_done"),
        )
        scan_and_split(provider=provider)
        assert called == []  # skipped because already in seen

    def test_handles_provider_exception(self, provider, monkeypatch):
        def boom(_state):
            raise RuntimeError("jira down")
        provider.list_by_state = boom  # type: ignore
        # Must not raise.
        scan_and_split(provider=provider)


# ---------------------------------------------------------------------------
# _build_prompt and _build_context smoke tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_story_fields(self):
        ctx = {
            "parent_key": "TK-100",
            "title": "Rebuild the thing",
            "description": "blah blah",
            "epic_context": "parent epic blurb",
            "sibling_titles": ["sib one", "sib two"],
            "failure_tail": "tests failed here",
            "files": [("agent/foo.py", "def foo(): pass")],
        }
        prompt = _build_prompt(ctx)
        assert "TK-100" in prompt
        assert "Rebuild the thing" in prompt
        assert "parent epic blurb" in prompt
        assert "- sib one" in prompt
        assert "tests failed here" in prompt
        assert "agent/foo.py" in prompt


class TestBuildContext:
    def test_assembles_pieces(self, provider, tmp_path):
        provider.seed(
            make_item(
                "TK-100",
                title="compaction atomicity",
                description="atomic writes and backups",
                execution_log="=== Attempt 1 ===\n...\n=== Attempt 2 ===\noops",
            ),
            comments=failure_comments(2),
        )
        ctx = _build_context(provider.get("TK-100"), provider, tmp_path)
        assert ctx["parent_key"] == "TK-100"
        assert ctx["title"] == "compaction atomicity"
        assert "atomic writes" in ctx["description"]
        assert ctx["failure_tail"]  # populated from comment


class TestPickRelevantFiles:
    def test_returns_empty_when_no_keywords(self, tmp_path):
        assert _pick_relevant_files(set(), tmp_path) == []

    def test_scores_and_returns_top(self, tmp_path):
        # Build a faux project layout.
        (tmp_path / "agent").mkdir()
        (tmp_path / "agent" / "memory_system.py").write_text(
            "def compaction(): pass  # atomic write backup"
        )
        (tmp_path / "agent" / "unrelated.py").write_text("def x(): pass")
        results = _pick_relevant_files({"compaction", "atomic"}, tmp_path, max_files=2)
        paths = [r[0] for r in results]
        assert "agent/memory_system.py" in paths
