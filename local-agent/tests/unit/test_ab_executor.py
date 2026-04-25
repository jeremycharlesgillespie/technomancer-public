"""Tests for ``idea_board.ab_executor.execute_idea_ab`` (orchestrator).

The orchestrator drives two inner ``execute_idea`` calls back-to-back,
captures diffs, scores both, runs the LLM comparison, and merges the
winner. These tests mock everything below the orchestrator (the inner
runs, ``ab_repo`` push/merge subprocess calls, the scorer, and the
compare LLM) so we can verify the high-level decision tree:

- A succeeds, B succeeds → merge A (incumbent priority).
- A succeeds, B fails    → merge A.
- A fails,    B succeeds → merge B.
- A fails,    B fails    → no merge, story marked failed.
- Model B not pulled     → B short-circuits with failure_log; A still
  merges if it succeeded.
- Compare LLM returns sentinel → ``comparison_error`` populated, merge
  decision still made by the priority rule.

The async work runs on a background thread inside the orchestrator —
each test ``join()``s the state's thread before asserting.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import aiv_schema
from idea_board import ab_executor
from idea_board.executor import ExecutionState


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@dataclass
class _FakeIdea:
    id: str = "TK-1"
    title: str = "Add retry logic"
    description: str = "Add backoff to webhook delivery"
    category: str = "quality"


@pytest.fixture
def fake_idea_provider(monkeypatch):
    """Patch ``board.get_provider().get(idea_id)`` to return a fake idea."""
    provider = MagicMock()
    provider.get.return_value = _FakeIdea()
    fake_board = MagicMock()
    fake_board.get_provider.return_value = provider
    # board may not be importable; insert into sys.modules.
    import sys
    monkeypatch.setitem(sys.modules, "board", fake_board)
    return provider


@pytest.fixture
def patch_settings(monkeypatch):
    """Force the A/B model settings to known values + a tmp project root.

    Both host fields default to ``""`` (same-box, localhost) so existing
    eviction tests stay on the same-host code path. The cross-host tests
    in ``TestOrchestratorCrossHostBehavior`` re-patch host_b to a remote
    URL via their own monkeypatch."""
    from agent import config as _config
    monkeypatch.setattr(_config.settings, "aiw_ab_model_a", "model-a:tag", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_b", "model-b:tag", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_a_host", "", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_b_host", "", raising=False)
    monkeypatch.setattr(_config.settings, "project_root", "/tmp/fake-root", raising=False)


def _wait(state: ExecutionState, timeout: float = 5.0) -> None:
    if state.thread is not None:
        state.thread.join(timeout=timeout)


def _make_attempt_mock(*, success: bool, branch: str = "br", sha: str = "abc1234", log: str = "ok") -> tuple:
    return (success, branch if success else "", sha if success else "", log)


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def _patch_orchestrator_helpers(
    *,
    a_success: bool,
    b_success: bool,
    b_model_pulled: bool = True,
    comparison_winner: str = "model_a",
    comparison_error: str = "",
):
    """Return a list of context-manager patches that stub the executor's
    expensive work — diff, scoring, compare LLM, push, merge, ollama check."""
    from aiv.ab_compare import ABComparison

    comparison = ABComparison(
        winner=comparison_winner,
        reasoning="judgement reasoning",
        delta_axes={a: 0 for a in (
            "meets_requirements", "code_quality", "test_quality",
            "security_safety", "scope_discipline", "edge_cases", "product_impact",
        )},
        error=comparison_error,
    )

    do_attempt_results = [
        _make_attempt_mock(success=a_success, branch="br-a", sha="aaaaaaa"),
        _make_attempt_mock(success=b_success, branch="br-b", sha="bbbbbbb"),
    ]
    do_attempt_mock = MagicMock(side_effect=do_attempt_results)

    return {
        "_do_attempt": patch.object(ab_executor, "_do_attempt", do_attempt_mock),
        "_ollama_has_model": patch.object(ab_executor, "_ollama_has_model", return_value=b_model_pulled),
        "_unload_ollama_model": patch.object(ab_executor, "_unload_ollama_model", return_value=True),
        "_capture_diff": patch.object(ab_executor, "_capture_diff", return_value="diff text"),
        "_reset_to_main": patch.object(ab_executor, "_reset_to_main", return_value=True),
        "_push_branch": patch.object(ab_executor, "_push_branch", return_value=(True, "pushed")),
        "_merge_winner": patch.object(ab_executor, "_merge_winner", return_value=(True, "merged")),
        "compare": patch("aiv.ab_compare.compare", return_value=comparison),
        "score": patch("aiv.scorer.score", return_value=None),
        "mark_executing": patch("idea_board.ab_executor.mark_executing"),
        "mark_done": patch("idea_board.ab_executor.mark_done"),
        "mark_failed": patch("idea_board.ab_executor.mark_failed"),
    }


def _run_with_patches(idea_id: str, patches: dict) -> tuple[ExecutionState, dict]:
    """Apply all patches via ``ExitStack``-style nested ``with`` blocks."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        active_mocks = {name: stack.enter_context(p) for name, p in patches.items()}
        state = ab_executor.execute_idea_ab(idea_id)
        assert state is not None
        _wait(state)
        return state, active_mocks


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def test_both_succeed_a_wins_and_merges(fake_idea_provider, patch_settings) -> None:
    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, mocks = _run_with_patches("TK-1", patches)

    # _merge_winner called exactly once with A's branch.
    assert mocks["_merge_winner"].call_count == 1
    call = mocks["_merge_winner"].call_args
    # 2nd positional is the branch name (state, branch, idea_id, project_root).
    assert call.args[1] == "br-a"

    # mark_done invoked, mark_failed not.
    assert mocks["mark_done"].called
    assert not mocks["mark_failed"].called

    # ab_test_pairs row written with merged_run_id != None.
    conn = aiv_schema._get_conn()
    pair = conn.execute("SELECT * FROM ab_test_pairs WHERE story_key=?", ("TK-1",)).fetchone()
    assert pair is not None
    assert pair["merged_run_id"] is not None


def test_a_succeeds_b_fails_a_wins(fake_idea_provider, patch_settings) -> None:
    patches = _patch_orchestrator_helpers(a_success=True, b_success=False)
    state, mocks = _run_with_patches("TK-2", patches)
    assert mocks["_merge_winner"].call_count == 1
    assert mocks["_merge_winner"].call_args.args[1] == "br-a"
    assert mocks["mark_done"].called


def test_a_fails_b_succeeds_b_wins(fake_idea_provider, patch_settings) -> None:
    patches = _patch_orchestrator_helpers(a_success=False, b_success=True)
    state, mocks = _run_with_patches("TK-3", patches)
    assert mocks["_merge_winner"].call_count == 1
    assert mocks["_merge_winner"].call_args.args[1] == "br-b"
    assert mocks["mark_done"].called


def test_both_fail_no_merge(fake_idea_provider, patch_settings) -> None:
    patches = _patch_orchestrator_helpers(a_success=False, b_success=False)
    state, mocks = _run_with_patches("TK-4", patches)
    assert mocks["_merge_winner"].call_count == 0
    assert mocks["mark_failed"].called
    assert not mocks["mark_done"].called

    # ab_test_pairs has merged_run_id = NULL.
    conn = aiv_schema._get_conn()
    pair = conn.execute("SELECT * FROM ab_test_pairs WHERE story_key=?", ("TK-4",)).fetchone()
    assert pair is not None
    assert pair["merged_run_id"] is None


def test_b_model_not_pulled_short_circuits(fake_idea_provider, patch_settings) -> None:
    """When ollama doesn't list model B, B is recorded as failed without a run.
    A's success still results in a merge."""
    patches = _patch_orchestrator_helpers(
        a_success=True, b_success=False, b_model_pulled=False,
    )
    # _do_attempt is called only once (for A) when B is short-circuited.
    state, mocks = _run_with_patches("TK-5", patches)

    assert mocks["_do_attempt"].call_count == 1
    assert mocks["_merge_winner"].call_count == 1  # A still merges

    conn = aiv_schema._get_conn()
    rows = conn.execute(
        "SELECT model_label, status, failure_log FROM ab_test_runs "
        "WHERE story_key=? ORDER BY model_label", ("TK-5",),
    ).fetchall()
    by_label = {r["model_label"]: r for r in rows}
    assert by_label["model-b"]["status"] == "failed"
    assert "not pulled" in (by_label["model-b"]["failure_log"] or "")


def test_compare_error_persists_to_pair_row(fake_idea_provider, patch_settings) -> None:
    """When the compare LLM returns a sentinel, the pair row records the
    error string but the priority-rule merge still happens."""
    patches = _patch_orchestrator_helpers(
        a_success=True, b_success=True,
        comparison_winner="",
        comparison_error="llm_error",
    )
    state, mocks = _run_with_patches("TK-6", patches)
    assert mocks["_merge_winner"].call_count == 1  # priority still works

    conn = aiv_schema._get_conn()
    pair = conn.execute("SELECT * FROM ab_test_pairs WHERE story_key=?", ("TK-6",)).fetchone()
    assert pair is not None
    assert pair["comparison_error"] == "llm_error"
    assert pair["merged_run_id"] is not None


def test_returns_none_when_idea_missing(monkeypatch, patch_settings) -> None:
    """Unknown idea_id → orchestrator returns None without spawning a thread."""
    provider = MagicMock()
    provider.get.return_value = None
    fake_board = MagicMock()
    fake_board.get_provider.return_value = provider
    import sys
    monkeypatch.setitem(sys.modules, "board", fake_board)

    state = ab_executor.execute_idea_ab("TK-NOPE")
    assert state is None


def test_run_rows_persisted_with_branch_and_sha(fake_idea_provider, patch_settings) -> None:
    """Both ab_test_runs rows record branch_name + commit_sha when the
    inner attempts succeeded."""
    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, mocks = _run_with_patches("TK-7", patches)

    conn = aiv_schema._get_conn()
    rows = conn.execute(
        "SELECT branch_name, commit_sha FROM ab_test_runs WHERE story_key=?", ("TK-7",),
    ).fetchall()
    branches = {r["branch_name"] for r in rows}
    shas = {r["commit_sha"] for r in rows}
    assert branches == {"br-a", "br-b"}
    assert shas == {"aaaaaaa", "bbbbbbb"}


# ---------------------------------------------------------------------------
# Model-unload between A and B + at end-of-run
# ---------------------------------------------------------------------------

def test_model_a_unloaded_before_b_starts(fake_idea_provider, patch_settings) -> None:
    """A's model is evicted (keep_alive=0) before B's run begins.

    Why: both coder runs pin keep_alive=-1, so without an explicit unload
    the Ollama scheduler may try to keep both 25-30 GB models resident
    and thrash. The orchestrator must unload A between runs.
    """
    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, mocks = _run_with_patches("TK-UNLOAD-1", patches)

    unload = mocks["_unload_ollama_model"]
    # Called at least twice: once after A (before B), once at end-of-run.
    assert unload.call_count >= 2
    # First positional arg of the first call must be model A.
    first_call = unload.call_args_list[0]
    assert first_call.args[0] == "model-a:tag"


def test_model_b_unloaded_at_run_end(fake_idea_provider, patch_settings) -> None:
    """The final unload (in the orchestrator's finally block) targets model B
    so the next AIM cycle starts with clean VRAM."""
    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, mocks = _run_with_patches("TK-UNLOAD-2", patches)

    unload = mocks["_unload_ollama_model"]
    # Last call must be model B.
    last_call = unload.call_args_list[-1]
    assert last_call.args[0] == "model-b:tag"


def test_unload_runs_even_when_both_fail(fake_idea_provider, patch_settings) -> None:
    """End-of-run unload fires regardless of merge outcome — the finally
    block must always clean VRAM."""
    patches = _patch_orchestrator_helpers(a_success=False, b_success=False)
    state, mocks = _run_with_patches("TK-UNLOAD-3", patches)

    unload = mocks["_unload_ollama_model"]
    # Even with both failing, the final unload of B fires.
    final_calls = [c for c in unload.call_args_list if c.args[0] == "model-b:tag"]
    assert len(final_calls) >= 1


def test_no_unload_when_models_identical(fake_idea_provider, monkeypatch) -> None:
    """When A == B (degenerate config), the unload-between-runs is skipped
    since there's nothing distinct to evict."""
    from agent import config as _config
    monkeypatch.setattr(_config.settings, "aiw_ab_model_a", "same:tag", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_b", "same:tag", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_a_host", "", raising=False)
    monkeypatch.setattr(_config.settings, "aiw_ab_model_b_host", "", raising=False)
    monkeypatch.setattr(_config.settings, "project_root", "/tmp/fake-root", raising=False)

    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, mocks = _run_with_patches("TK-UNLOAD-4", patches)

    # No unload call should target the shared tag.
    unload = mocks["_unload_ollama_model"]
    same_tag_calls = [c for c in unload.call_args_list if c.args[0] == "same:tag"]
    assert same_tag_calls == []


def test_unload_helper_posts_keep_alive_zero(monkeypatch) -> None:
    """_unload_ollama_model posts keep_alive=0 to /api/generate with the
    target tag — that's the Ollama-recommended way to evict a runner."""
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    import requests as _requests
    monkeypatch.setattr(_requests, "post", fake_post)

    ok = ab_executor._unload_ollama_model("foo:bar")
    assert ok is True
    assert captured["json"]["model"] == "foo:bar"
    assert captured["json"]["keep_alive"] == 0
    assert captured["json"]["stream"] is False
    assert captured["url"].endswith("/api/generate")


def test_unload_helper_returns_false_on_empty_tag() -> None:
    """Empty / falsy tag → no-op, returns False (don't burn an HTTP call)."""
    assert ab_executor._unload_ollama_model("") is False


def test_unload_helper_swallows_network_errors(monkeypatch) -> None:
    """Network errors must NOT raise — orchestrator should proceed even if
    Ollama is briefly unreachable. Worst case Ollama auto-evicts when the
    next model is requested."""
    import requests as _requests

    def boom(*a, **kw):
        raise _requests.RequestException("connection refused")

    monkeypatch.setattr(_requests, "post", boom)
    assert ab_executor._unload_ollama_model("foo:bar") is False


# ---------------------------------------------------------------------------
# Orchestrator state visibility — get_execution must see the A/B state
# even while the inner per-attempt run is owning ``_active``.
# ---------------------------------------------------------------------------

def test_get_execution_finds_orchestrator_state_during_run(
    fake_idea_provider, patch_settings,
) -> None:
    """While the orchestrator is mid-run, ``get_execution(idea_id)`` must
    return the orchestrator's ``ExecutionState`` so callers like
    ``aim.worker.watch_execution`` don't get ``None`` between inner attempts.

    Why: the inner ``execute_idea`` calls churn ``_active[idea_id]`` (set on
    start, popped in their own ``finally``). Before the orchestrator fix,
    that left a window where neither dict held the state and the worker
    aborted with "No execution state found" within seconds of routing.
    """
    from idea_board import executor as _executor

    saw_state_during_run: list[bool] = []

    real_do_attempt = ab_executor._do_attempt

    def spy_do_attempt(idea_id, **kwargs):
        # During an inner attempt, _active is owned by the inner run; the
        # orchestrator's state must still be discoverable via the new fallback.
        seen = _executor.get_execution(idea_id)
        saw_state_during_run.append(seen is not None)
        return _make_attempt_mock(success=True, branch="br-a", sha="aaaaaaa")

    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    patches["_do_attempt"] = patch.object(
        ab_executor, "_do_attempt", side_effect=spy_do_attempt,
    )

    state, _mocks = _run_with_patches("TK-STATE-1", patches)

    assert state is not None
    # Spy ran twice (once per attempt), and both times get_execution
    # returned a non-None state.
    assert len(saw_state_during_run) == 2
    assert all(saw_state_during_run)


def test_orchestrator_state_cleaned_up_after_run(
    fake_idea_provider, patch_settings,
) -> None:
    """After the orchestrator's thread joins, the orchestrator registry
    must be empty for the idea so a subsequent run starts fresh."""
    from idea_board import executor as _executor

    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    state, _mocks = _run_with_patches("TK-STATE-2", patches)

    assert state is not None
    # Both registries should be cleared.
    assert "TK-STATE-2" not in _executor._ab_orchestrator_active
    assert "TK-STATE-2" not in _executor._active
    assert _executor.get_execution("TK-STATE-2") is None


def test_orchestrator_uses_ab_registry_not_active(
    fake_idea_provider, patch_settings,
) -> None:
    """The orchestrator must register its state in ``_ab_orchestrator_active``,
    not in ``_active`` — otherwise the inner ``execute_idea`` call would
    overwrite it on the dedup-and-set line and the orchestrator's outer
    state would be silently lost."""
    from idea_board import executor as _executor

    captured: dict[str, object] = {}

    real_do_attempt = ab_executor._do_attempt

    def spy_do_attempt(idea_id, **kwargs):
        # Snapshot which dict holds the orchestrator's state when an inner
        # attempt is about to begin.
        captured["in_ab_registry"] = idea_id in _executor._ab_orchestrator_active
        captured["in_active_at_attempt_start"] = idea_id in _executor._active
        return _make_attempt_mock(success=True, branch="br-a", sha="aaaaaaa")

    patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
    patches["_do_attempt"] = patch.object(
        ab_executor, "_do_attempt", side_effect=spy_do_attempt,
    )

    _state, _mocks = _run_with_patches("TK-STATE-3", patches)

    assert captured["in_ab_registry"] is True
    # The inner attempt is mocked, so _active is never touched in this test.
    assert captured["in_active_at_attempt_start"] is False


# ---------------------------------------------------------------------------
# AIV hand-off helpers — _diff_paths_from_text + _head_sha
# ---------------------------------------------------------------------------


class TestDiffPathsFromText:
    def test_extracts_paths_from_unified_diff(self):
        diff = (
            "diff --git a/agent/foo.py b/agent/foo.py\n"
            "index 1234..5678 100644\n"
            "--- a/agent/foo.py\n"
            "+++ b/agent/foo.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+pass\n"
            "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
            "index abcd..efgh 100644\n"
            "+def test_x(): pass\n"
        )
        assert ab_executor._diff_paths_from_text(diff) == [
            "agent/foo.py",
            "tests/test_foo.py",
        ]

    def test_empty_text_returns_empty_list(self):
        assert ab_executor._diff_paths_from_text("") == []
        assert ab_executor._diff_paths_from_text(None) == []  # type: ignore[arg-type]

    def test_dedupes_repeated_paths(self):
        """A single path appearing twice (rename → rename-back) only
        shows up once in the output."""
        diff = (
            "diff --git a/x.py b/x.py\n"
            "diff --git a/x.py b/x.py\n"
        )
        assert ab_executor._diff_paths_from_text(diff) == ["x.py"]

    def test_skips_malformed_headers(self):
        """Lines that look like headers but lack the ``b/`` half are skipped."""
        diff = (
            "diff --git malformed\n"
            "diff --git a/ok.py b/ok.py\n"
        )
        assert ab_executor._diff_paths_from_text(diff) == ["ok.py"]


class TestHeadSha:
    def test_returns_sha_on_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            ab_executor.subprocess, "run",
            lambda *a, **kw: MagicMock(returncode=0, stdout="abc123def\n"),
        )
        assert ab_executor._head_sha(tmp_path) == "abc123def"

    def test_returns_none_on_nonzero_exit(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            ab_executor.subprocess, "run",
            lambda *a, **kw: MagicMock(returncode=128, stdout=""),
        )
        assert ab_executor._head_sha(tmp_path) is None

    def test_returns_none_on_exception(self, monkeypatch, tmp_path):
        def _boom(*_a, **_kw):
            raise OSError("git missing")
        monkeypatch.setattr(ab_executor.subprocess, "run", _boom)
        assert ab_executor._head_sha(tmp_path) is None

    def test_returns_none_on_blank_stdout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            ab_executor.subprocess, "run",
            lambda *a, **kw: MagicMock(returncode=0, stdout="   \n"),
        )
        assert ab_executor._head_sha(tmp_path) is None


# ---------------------------------------------------------------------------
# Cross-host plumbing — _ollama_has_model, _unload_ollama_model accept a host
# argument so model A and model B can run on different Ollama instances
# (e.g. local Mac for A, remote 5090 box for B). The orchestrator passes
# settings.aiw_ab_model_a_host / aiw_ab_model_b_host through both helpers
# and through _do_attempt -> execute_idea(host_override=...).
# ---------------------------------------------------------------------------


class TestOllamaHasModelOverHttp:
    """``_ollama_has_model`` queries ``GET /api/tags`` over HTTP so the same
    code path works for local + remote Ollama instances."""

    def test_returns_true_when_model_listed(self, monkeypatch):
        captured: dict[str, object] = {}

        class _Resp:
            status_code = 200
            def json(self):
                return {"models": [
                    {"name": "qwen3-coder:30b-a3b-q4_K_M"},
                    {"name": "glm-4.7-flash:q4_K_M"},
                ]}

        def fake_get(url, timeout=None):
            captured["url"] = url
            captured["timeout"] = timeout
            return _Resp()

        import requests as _requests
        monkeypatch.setattr(_requests, "get", fake_get)

        assert ab_executor._ollama_has_model(
            "glm-4.7-flash:q4_K_M",
            host="http://192.168.1.150:11434",
        ) is True
        assert captured["url"] == "http://192.168.1.150:11434/api/tags"

    def test_returns_false_when_model_missing(self, monkeypatch):
        class _Resp:
            status_code = 200
            def json(self):
                return {"models": [{"name": "other:tag"}]}

        import requests as _requests
        monkeypatch.setattr(_requests, "get", lambda url, timeout=None: _Resp())

        assert ab_executor._ollama_has_model(
            "missing:tag",
            host="http://192.168.1.150:11434",
        ) is False

    def test_returns_false_on_network_error(self, monkeypatch):
        """Remote box unreachable → False so orchestrator short-circuits B
        with a clear ``failure_log`` rather than hanging on the model load."""
        import requests as _requests

        def boom(url, timeout=None):
            raise _requests.ConnectionError("no route to host")

        monkeypatch.setattr(_requests, "get", boom)
        assert ab_executor._ollama_has_model(
            "any:tag", host="http://192.168.1.150:11434"
        ) is False

    def test_returns_false_on_non_200(self, monkeypatch):
        class _Resp:
            status_code = 500
            def json(self):
                return {}

        import requests as _requests
        monkeypatch.setattr(_requests, "get", lambda url, timeout=None: _Resp())

        assert ab_executor._ollama_has_model("any:tag", host="http://x:11434") is False

    def test_exact_match_only_no_substring_collision(self, monkeypatch):
        """``qwen2.5-coder:32b`` must NOT match ``qwen2.5-coder:7b``."""
        class _Resp:
            status_code = 200
            def json(self):
                return {"models": [{"name": "qwen2.5-coder:7b"}]}

        import requests as _requests
        monkeypatch.setattr(_requests, "get", lambda url, timeout=None: _Resp())

        assert ab_executor._ollama_has_model(
            "qwen2.5-coder:32b", host="",
        ) is False

    def test_empty_host_falls_back_to_default(self, monkeypatch):
        """Empty host → uses ``OLLAMA_HOST`` from ``agent.ollama_client``."""
        captured: dict[str, object] = {}

        class _Resp:
            status_code = 200
            def json(self):
                return {"models": []}

        def fake_get(url, timeout=None):
            captured["url"] = url
            return _Resp()

        import requests as _requests
        monkeypatch.setattr(_requests, "get", fake_get)

        ab_executor._ollama_has_model("anything:tag", host="")
        # URL must include /api/tags and start with a real http(s) scheme,
        # not an empty string.
        url = captured["url"]
        assert url.endswith("/api/tags")
        assert url.startswith("http")


class TestUnloadOllamaModelWithHost:
    """``_unload_ollama_model`` accepts a host so we can evict a model on
    a remote Ollama box, not just localhost."""

    def test_remote_host_targets_remote_url(self, monkeypatch):
        captured: dict[str, object] = {}

        class _Resp:
            status_code = 200

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

        import requests as _requests
        monkeypatch.setattr(_requests, "post", fake_post)

        ok = ab_executor._unload_ollama_model(
            "glm-4.7-flash:q4_K_M",
            host="http://192.168.1.150:11434",
        )
        assert ok is True
        assert captured["url"] == "http://192.168.1.150:11434/api/generate"
        assert captured["json"]["model"] == "glm-4.7-flash:q4_K_M"
        assert captured["json"]["keep_alive"] == 0

    def test_empty_host_falls_back_to_default(self, monkeypatch):
        """Empty host → still posts to the default OLLAMA_HOST. Verifies
        existing localhost callers keep working."""
        captured: dict[str, object] = {}

        class _Resp:
            status_code = 200

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            return _Resp()

        import requests as _requests
        monkeypatch.setattr(_requests, "post", fake_post)

        ab_executor._unload_ollama_model("foo:bar", host="")
        assert captured["url"].endswith("/api/generate")
        assert captured["url"].startswith("http")


class TestOrchestratorCrossHostBehavior:
    """When ``host_a != host_b``, the orchestrator skips the
    between-runs eviction (no GPU contention to resolve) and threads
    each host through the matching ``_do_attempt`` call."""

    def test_skips_eviction_when_hosts_differ(
        self, fake_idea_provider, patch_settings, monkeypatch,
    ):
        from agent import config as _config
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_a_host", "", raising=False,
        )
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_b_host",
            "http://192.168.1.150:11434", raising=False,
        )

        patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
        state, mocks = _run_with_patches("TK-HOST-1", patches)

        # Eviction call between A and B is skipped on cross-host runs.
        # The end-of-run unload of B may still fire.
        unload = mocks["_unload_ollama_model"]
        # No call should target model A — the only legitimate unload is the
        # final unload of B.
        a_calls = [c for c in unload.call_args_list if c.args[0] == "model-a:tag"]
        assert a_calls == [], (
            "model A should not be unloaded when A and B run on different hosts"
        )

    def test_passes_host_a_and_host_b_to_do_attempt(
        self, fake_idea_provider, patch_settings, monkeypatch,
    ):
        from agent import config as _config
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_a_host", "", raising=False,
        )
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_b_host",
            "http://192.168.1.150:11434", raising=False,
        )

        patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
        state, mocks = _run_with_patches("TK-HOST-2", patches)

        do_attempt = mocks["_do_attempt"]
        assert do_attempt.call_count == 2
        # Call 1 is A; host kwarg must be "" (local).
        first_kwargs = do_attempt.call_args_list[0].kwargs
        assert first_kwargs.get("host") == ""
        # Call 2 is B; host kwarg must be the remote URL.
        second_kwargs = do_attempt.call_args_list[1].kwargs
        assert second_kwargs.get("host") == "http://192.168.1.150:11434"

    def test_passes_host_b_to_ollama_has_model_check(
        self, fake_idea_provider, patch_settings, monkeypatch,
    ):
        """The pre-flight ``_ollama_has_model`` check for B must target
        host B, not localhost — otherwise we'd false-positive on a local
        model name that doesn't exist on the remote box (or vice-versa)."""
        from agent import config as _config
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_b_host",
            "http://192.168.1.150:11434", raising=False,
        )

        patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
        state, mocks = _run_with_patches("TK-HOST-3", patches)

        has_model = mocks["_ollama_has_model"]
        # Find the call for model B (model name == "model-b:tag").
        b_calls = [c for c in has_model.call_args_list
                   if c.args and c.args[0] == "model-b:tag"]
        assert len(b_calls) == 1
        assert b_calls[0].kwargs.get("host") == "http://192.168.1.150:11434"

    def test_evicts_when_hosts_match(
        self, fake_idea_provider, patch_settings, monkeypatch,
    ):
        """Sanity: when both hosts are blank (the default same-box config),
        the between-runs eviction still fires for model A."""
        from agent import config as _config
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_a_host", "", raising=False,
        )
        monkeypatch.setattr(
            _config.settings, "aiw_ab_model_b_host", "", raising=False,
        )

        patches = _patch_orchestrator_helpers(a_success=True, b_success=True)
        state, mocks = _run_with_patches("TK-HOST-4", patches)

        unload = mocks["_unload_ollama_model"]
        a_calls = [c for c in unload.call_args_list if c.args[0] == "model-a:tag"]
        assert len(a_calls) >= 1, (
            "model A should be unloaded between runs when both hosts match"
        )
