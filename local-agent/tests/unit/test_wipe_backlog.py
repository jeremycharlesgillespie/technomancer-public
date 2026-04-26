"""Tests for scripts/wipe_backlog.py — backlog bulk-wipe classifier."""

from __future__ import annotations

import importlib

import pytest

from idea_board.models import Idea


# Import the module under test.
wipe_backlog = importlib.import_module("scripts.wipe_backlog")


def _idea(idea_id: str, *, state: str, idea_type: str = "story") -> Idea:
    return Idea(
        id=idea_id,
        title=f"Test {idea_id}",
        description="body",
        category="quality",
        source="test",
        idea_type=idea_type,
        state=state,
    )


class TestClassify:
    def test_keeps_epics_regardless_of_state(self) -> None:
        ideas = [
            _idea("e1", state="proposed", idea_type="epic"),
            _idea("e2", state="failed", idea_type="epic"),
            _idea("e3", state="approved", idea_type="epic"),
        ]
        to_keep, to_wipe = wipe_backlog._classify(ideas)
        assert {i.id for i in to_keep} == {"e1", "e2", "e3"}
        assert to_wipe == []

    def test_keeps_done_executing_vetoed_cancelled(self) -> None:
        ideas = [
            _idea("d1", state="done"),
            _idea("d2", state="executing"),
            _idea("d3", state="vetoed"),
            _idea("d4", state="cancelled"),
        ]
        to_keep, to_wipe = wipe_backlog._classify(ideas)
        assert len(to_keep) == 4
        assert to_wipe == []

    def test_wipes_proposed_refining_approved_failed_stories(self) -> None:
        ideas = [
            _idea("s1", state="proposed"),
            _idea("s2", state="refining"),
            _idea("s3", state="approved"),
            _idea("s4", state="failed"),
        ]
        to_keep, to_wipe = wipe_backlog._classify(ideas)
        assert to_keep == []
        assert {i.id for i in to_wipe} == {"s1", "s2", "s3", "s4"}

    def test_unknown_state_is_kept_conservatively(self) -> None:
        ideas = [_idea("u1", state="weird-unknown-state")]
        to_keep, to_wipe = wipe_backlog._classify(ideas)
        assert {i.id for i in to_keep} == {"u1"}
        assert to_wipe == []

    def test_mixed_set(self) -> None:
        ideas = [
            _idea("e1", state="proposed", idea_type="epic"),  # keep (epic)
            _idea("d1", state="done"),                         # keep (done)
            _idea("p1", state="proposed"),                     # wipe
            _idea("p2", state="failed"),                       # wipe
            _idea("p3", state="approved"),                     # wipe
        ]
        to_keep, to_wipe = wipe_backlog._classify(ideas)
        assert {i.id for i in to_keep} == {"e1", "d1"}
        assert {i.id for i in to_wipe} == {"p1", "p2", "p3"}


class TestMain:
    def test_main_with_yes_flag_calls_delete(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        ideas = [
            _idea("p1", state="proposed"),
            _idea("d1", state="done"),
        ]
        monkeypatch.setattr(wipe_backlog, "load_ideas", lambda: ideas)

        deleted_ids: list[str] = []
        def mock_delete(idea_id: str) -> bool:
            deleted_ids.append(idea_id)
            return True
        monkeypatch.setattr(wipe_backlog, "delete_idea", mock_delete)

        monkeypatch.setattr("sys.argv", ["wipe_backlog.py", "--yes"])
        rc = wipe_backlog.main()
        assert rc == 0
        assert deleted_ids == ["p1"]
        out = capsys.readouterr().out
        assert "Deleted 1" in out

    def test_main_dry_run_makes_no_changes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        ideas = [_idea("p1", state="proposed")]
        monkeypatch.setattr(wipe_backlog, "load_ideas", lambda: ideas)

        called = []
        monkeypatch.setattr(wipe_backlog, "delete_idea", lambda i: called.append(i) or True)

        monkeypatch.setattr("sys.argv", ["wipe_backlog.py", "--dry-run"])
        rc = wipe_backlog.main()
        assert rc == 0
        assert called == []  # delete_idea never invoked in dry-run
        assert "[dry-run]" in capsys.readouterr().out

    def test_main_without_yes_aborts_when_input_is_no(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        ideas = [_idea("p1", state="proposed")]
        monkeypatch.setattr(wipe_backlog, "load_ideas", lambda: ideas)

        called: list[str] = []
        monkeypatch.setattr(wipe_backlog, "delete_idea", lambda i: called.append(i) or True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")

        monkeypatch.setattr("sys.argv", ["wipe_backlog.py"])
        rc = wipe_backlog.main()
        assert rc == 1
        assert called == []
        assert "Aborted" in capsys.readouterr().out

    def test_main_returns_zero_when_nothing_to_delete(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        ideas = [_idea("e1", state="proposed", idea_type="epic")]
        monkeypatch.setattr(wipe_backlog, "load_ideas", lambda: ideas)
        monkeypatch.setattr("sys.argv", ["wipe_backlog.py", "--yes"])
        rc = wipe_backlog.main()
        assert rc == 0
        assert "Nothing to delete" in capsys.readouterr().out
