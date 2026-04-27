"""Unit tests for ``aim.janitor.run_startup_janitor``.

The janitor is the worker's startup cleanup pass. It must:
  * Find and remove sibling ``technomancer-aiw-*`` worktree dirs.
  * Reap stranded ``ab_test_runs`` rows still marked ``running``.
  * Evict every configured coder model from VRAM via Ollama.
  * Never raise — every sweep is best-effort.

Tests use a temp dir as ``repo_root`` so the worktree scan is real (it
walks the parent directory) but no actual git operations happen — the
``remove_worktree`` import is mocked at the seam.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aim.janitor import (
    WORKTREE_PREFIX,
    _coder_model_targets,
    _find_orphan_worktrees,
    _sweep_ab_runs,
    _sweep_models,
    _sweep_worktrees,
    _unload_one,
    run_startup_janitor,
)


# ---------------------------------------------------------------------------
# Worktree sweep
# ---------------------------------------------------------------------------


class TestFindOrphanWorktrees:
    def test_returns_empty_when_parent_has_no_matches(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        # Sibling dir that doesn't match prefix
        (tmp_path / "unrelated").mkdir()
        (tmp_path / "another-thing").mkdir()

        result = _find_orphan_worktrees(repo_root)

        assert result == []

    def test_finds_only_matching_prefix(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        wt_a = tmp_path / f"{WORKTREE_PREFIX}aaaa1111"
        wt_b = tmp_path / f"{WORKTREE_PREFIX}bbbb2222"
        unrelated = tmp_path / "unrelated"
        wt_a.mkdir()
        wt_b.mkdir()
        unrelated.mkdir()

        result = _find_orphan_worktrees(repo_root)

        assert set(result) == {wt_a, wt_b}

    def test_skips_files_with_matching_prefix(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        # File (not dir) with the prefix — should be ignored
        bad_file = tmp_path / f"{WORKTREE_PREFIX}filething"
        bad_file.write_text("not a worktree")

        result = _find_orphan_worktrees(repo_root)

        assert result == []

    def test_returns_empty_when_parent_missing(self, tmp_path: Path) -> None:
        # repo_root.parent does not exist
        repo_root = tmp_path / "nope" / "repo"

        result = _find_orphan_worktrees(repo_root)

        assert result == []


class TestSweepWorktrees:
    def test_no_orphans_returns_zero(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        with patch("idea_board.ab_worktree.remove_worktree") as mock_remove:
            removed = _sweep_worktrees(repo_root)

        assert removed == 0
        mock_remove.assert_not_called()

    def test_removes_each_orphan(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        wt_a = tmp_path / f"{WORKTREE_PREFIX}aaaa"
        wt_b = tmp_path / f"{WORKTREE_PREFIX}bbbb"
        wt_a.mkdir()
        wt_b.mkdir()

        with patch(
            "idea_board.ab_worktree.remove_worktree", return_value=True,
        ) as mock_remove:
            removed = _sweep_worktrees(repo_root)

        assert removed == 2
        assert mock_remove.call_count == 2

    def test_counts_only_successful_removals(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (tmp_path / f"{WORKTREE_PREFIX}aaaa").mkdir()
        (tmp_path / f"{WORKTREE_PREFIX}bbbb").mkdir()

        with patch(
            "idea_board.ab_worktree.remove_worktree",
            side_effect=[True, False],
        ):
            removed = _sweep_worktrees(repo_root)

        assert removed == 1

    def test_swallows_remove_exception(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (tmp_path / f"{WORKTREE_PREFIX}aaaa").mkdir()

        with patch(
            "idea_board.ab_worktree.remove_worktree",
            side_effect=RuntimeError("git crashed"),
        ):
            # Must not raise
            removed = _sweep_worktrees(repo_root)

        assert removed == 0


# ---------------------------------------------------------------------------
# A/B sweep
# ---------------------------------------------------------------------------


class TestSweepAbRuns:
    def test_calls_reap_with_janitor_reason(self) -> None:
        with patch("idea_board.ab_repo.reap_stranded_runs", return_value=3) as mock_reap:
            count = _sweep_ab_runs()

        assert count == 3
        # Must use a janitor-specific reason so abandoned rows are
        # distinguishable in failure_log from the worker.run() call.
        mock_reap.assert_called_once()
        kwargs = mock_reap.call_args.kwargs
        assert "janitor" in kwargs["reason"].lower()

    def test_swallows_reap_exception(self) -> None:
        with patch(
            "idea_board.ab_repo.reap_stranded_runs",
            side_effect=RuntimeError("db locked"),
        ):
            count = _sweep_ab_runs()

        assert count == 0

    def test_zero_reaped_returns_zero(self) -> None:
        with patch("idea_board.ab_repo.reap_stranded_runs", return_value=0):
            count = _sweep_ab_runs()

        assert count == 0


# ---------------------------------------------------------------------------
# Model unload sweep
# ---------------------------------------------------------------------------


class TestCoderModelTargets:
    def test_includes_single_mode_model(self) -> None:
        fake_settings = MagicMock(
            aiw_ollama_coder_model="qwen3-coder:30b",
            aiw_ab_model_a="",
            aiw_ab_model_b="",
            aiw_ab_model_a_host="",
            aiw_ab_model_b_host="",
        )
        with patch("agent.config.settings", fake_settings):
            targets = _coder_model_targets()

        assert ("qwen3-coder:30b", "") in targets

    def test_includes_ab_models_with_hosts(self) -> None:
        fake_settings = MagicMock(
            aiw_ollama_coder_model="single:tag",
            aiw_ab_model_a="modelA:30b",
            aiw_ab_model_b="modelB:32b",
            aiw_ab_model_a_host="http://localhost:11434",
            aiw_ab_model_b_host="http://192.168.1.150:11434",
        )
        with patch("agent.config.settings", fake_settings):
            targets = _coder_model_targets()

        assert ("modelA:30b", "http://localhost:11434") in targets
        assert ("modelB:32b", "http://192.168.1.150:11434") in targets

    def test_skips_empty_tags(self) -> None:
        fake_settings = MagicMock(
            aiw_ollama_coder_model="",
            aiw_ab_model_a="",
            aiw_ab_model_b="modelB:32b",
            aiw_ab_model_a_host="",
            aiw_ab_model_b_host="",
        )
        with patch("agent.config.settings", fake_settings):
            targets = _coder_model_targets()

        assert targets == [("modelB:32b", "")]

    def test_dedupes_repeated_pairs(self) -> None:
        # Same model + host configured twice (single-mode tag also used as A)
        fake_settings = MagicMock(
            aiw_ollama_coder_model="shared:30b",
            aiw_ab_model_a="shared:30b",
            aiw_ab_model_b="other:32b",
            aiw_ab_model_a_host="",
            aiw_ab_model_b_host="",
        )
        with patch("agent.config.settings", fake_settings):
            targets = _coder_model_targets()

        # ("shared:30b", "") appears once, not twice
        shared_count = sum(1 for t in targets if t == ("shared:30b", ""))
        assert shared_count == 1


class TestUnloadOne:
    def test_posts_keep_alive_zero(self) -> None:
        mock_response = MagicMock(status_code=200)
        with patch("agent.ollama_client.OLLAMA_HOST", "http://default:11434"), \
             patch("requests.post", return_value=mock_response) as mock_post:
            ok = _unload_one("qwen3-coder:30b", "")

        assert ok is True
        call = mock_post.call_args
        assert call.args[0] == "http://default:11434/api/generate"
        body = call.kwargs["json"]
        assert body["model"] == "qwen3-coder:30b"
        assert body["keep_alive"] == 0
        assert body["stream"] is False

    def test_uses_explicit_host_when_provided(self) -> None:
        mock_response = MagicMock(status_code=200)
        with patch("agent.ollama_client.OLLAMA_HOST", "http://default:11434"), \
             patch("requests.post", return_value=mock_response) as mock_post:
            _unload_one("modelB:32b", "http://192.168.1.150:11434")

        url = mock_post.call_args.args[0]
        assert url.startswith("http://192.168.1.150:11434/")

    def test_empty_tag_returns_false(self) -> None:
        with patch("requests.post") as mock_post:
            ok = _unload_one("", "")

        assert ok is False
        mock_post.assert_not_called()

    def test_non_200_returns_false(self) -> None:
        mock_response = MagicMock(status_code=500)
        with patch("agent.ollama_client.OLLAMA_HOST", "http://h:1"), \
             patch("requests.post", return_value=mock_response):
            ok = _unload_one("modelA:30b", "")

        assert ok is False

    def test_swallows_network_exception(self) -> None:
        with patch("agent.ollama_client.OLLAMA_HOST", "http://h:1"), \
             patch("requests.post", side_effect=ConnectionError("nope")):
            ok = _unload_one("modelA:30b", "")

        assert ok is False


class TestSweepModels:
    def test_no_targets_returns_zero(self) -> None:
        count = _sweep_models(targets=[])
        assert count == 0

    def test_calls_unload_for_each_target(self) -> None:
        targets = [("a:1", ""), ("b:1", "host2")]
        with patch(
            "aim.janitor._unload_one", return_value=True,
        ) as mock_unload:
            count = _sweep_models(targets=targets)

        assert count == 2
        assert mock_unload.call_count == 2

    def test_counts_only_successful_unloads(self) -> None:
        targets = [("a:1", ""), ("b:1", "")]
        with patch("aim.janitor._unload_one", side_effect=[True, False]):
            count = _sweep_models(targets=targets)

        assert count == 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class TestRunStartupJanitor:
    def test_returns_stats_dict(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        with patch("aim.janitor._sweep_worktrees", return_value=2), \
             patch("aim.janitor._sweep_ab_runs", return_value=3), \
             patch("aim.janitor._sweep_models", return_value=1):
            stats = run_startup_janitor(repo_root)

        assert stats == {
            "worktrees_removed": 2,
            "ab_runs_reaped": 3,
            "models_unloaded": 1,
        }

    def test_swallows_sweep_exceptions(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        # Each sweep raises — janitor must not propagate.
        with patch("aim.janitor._sweep_worktrees", side_effect=RuntimeError("a")), \
             patch("aim.janitor._sweep_ab_runs", side_effect=RuntimeError("b")), \
             patch("aim.janitor._sweep_models", side_effect=RuntimeError("c")):
            stats = run_startup_janitor(repo_root)  # must not raise

        assert stats == {
            "worktrees_removed": 0,
            "ab_runs_reaped": 0,
            "models_unloaded": 0,
        }

    def test_partial_failure_records_zero_for_failed_only(
        self, tmp_path: Path,
    ) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        with patch("aim.janitor._sweep_worktrees", return_value=5), \
             patch("aim.janitor._sweep_ab_runs", side_effect=RuntimeError("x")), \
             patch("aim.janitor._sweep_models", return_value=2):
            stats = run_startup_janitor(repo_root)

        assert stats["worktrees_removed"] == 5
        assert stats["ab_runs_reaped"] == 0  # crashed sweep zeroed
        assert stats["models_unloaded"] == 2

    def test_runs_all_three_sweeps_in_order(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        call_order: list[str] = []

        with patch(
            "aim.janitor._sweep_worktrees",
            side_effect=lambda _: (call_order.append("wt"), 0)[1],
        ), patch(
            "aim.janitor._sweep_ab_runs",
            side_effect=lambda: (call_order.append("ab"), 0)[1],
        ), patch(
            "aim.janitor._sweep_models",
            side_effect=lambda: (call_order.append("models"), 0)[1],
        ):
            run_startup_janitor(repo_root)

        assert call_order == ["wt", "ab", "models"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
