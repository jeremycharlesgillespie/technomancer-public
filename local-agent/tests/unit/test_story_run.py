"""Tests for the StoryRun resource-lifecycle wrapper.

The wrapper's contract is:

1. Every acquired resource sees ``release()`` exactly once on context exit.
2. Resources are released in LIFO order.
3. ``release()`` is idempotent — calling it twice is safe.
4. ``release()`` never raises, even when the underlying teardown fails.
5. Body exceptions propagate; the wrapper does not swallow them.
6. Cleanup runs even when the body raises.

These tests pin all six.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.story_run import (
    Branch,
    ModelHandle,
    Resource,
    StoryRun,
    Worktree,
)


# ---------------------------------------------------------------------------
# A test-only Resource subclass that records calls — lets us inspect the
# release order and idempotency without depending on subprocess mocks.
# ---------------------------------------------------------------------------


class _RecordingResource(Resource):
    """A Resource that records every _do_release call. For test inspection only."""

    def __init__(self, label: str, *, raise_on_release: bool = False) -> None:
        super().__init__(label=label)
        self.release_count = 0
        self.raise_on_release = raise_on_release

    def _do_release(self) -> None:
        self.release_count += 1
        if self.raise_on_release:
            raise RuntimeError(f"intentional teardown failure for {self.label}")


# ===========================================================================
# StoryRun lifecycle behaviour
# ===========================================================================


class TestStoryRunLifecycle:
    """The wrapper's core guarantees."""

    def test_releases_all_resources_on_normal_exit(self):
        a = _RecordingResource("a")
        b = _RecordingResource("b")
        with StoryRun(label="test") as run:
            run.acquire(a)
            run.acquire(b)
        assert a.release_count == 1
        assert b.release_count == 1

    def test_releases_in_lifo_order(self):
        order: list[str] = []

        class _OrderedResource(Resource):
            def __init__(self, name: str) -> None:
                super().__init__(label=name)
                self.name = name

            def _do_release(self) -> None:
                order.append(self.name)

        with StoryRun() as run:
            run.acquire(_OrderedResource("first"))
            run.acquire(_OrderedResource("second"))
            run.acquire(_OrderedResource("third"))

        # LIFO: last acquired first released.
        assert order == ["third", "second", "first"]

    def test_releases_resources_when_body_raises(self):
        a = _RecordingResource("a")
        with pytest.raises(ValueError, match="boom"):
            with StoryRun() as run:
                run.acquire(a)
                raise ValueError("boom")
        assert a.release_count == 1, "release must run even when body raises"

    def test_body_exception_propagates(self):
        """Wrapper does not suppress body exceptions."""
        a = _RecordingResource("a")
        with pytest.raises(KeyError, match="missing"):
            with StoryRun() as run:
                run.acquire(a)
                raise KeyError("missing")

    def test_release_failure_does_not_block_other_releases(self):
        """If one resource's release raises, the others still run."""
        a = _RecordingResource("a")
        b = _RecordingResource("b", raise_on_release=True)  # this one fails
        c = _RecordingResource("c")

        with StoryRun() as run:
            run.acquire(a)
            run.acquire(b)
            run.acquire(c)

        # All three saw release exactly once, including the one that raised.
        assert a.release_count == 1
        assert b.release_count == 1
        assert c.release_count == 1

    def test_release_is_idempotent(self):
        a = _RecordingResource("a")
        a.release()
        a.release()
        a.release()
        assert a.release_count == 1, "subsequent releases must be no-ops"

    def test_manual_release_all_then_exit_does_not_double_release(self):
        a = _RecordingResource("a")
        b = _RecordingResource("b")
        with StoryRun() as run:
            run.acquire(a)
            run.acquire(b)
            run.release_all()  # release manually
            assert a.release_count == 1
            assert b.release_count == 1
        # Context exit calls release_all again — but each resource
        # records the second call as a no-op (count stays at 1).
        assert a.release_count == 1
        assert b.release_count == 1

    def test_acquire_returns_the_resource_for_chaining(self):
        with StoryRun() as run:
            r = run.acquire(_RecordingResource("a"))
        assert r.label == "a"

    def test_resources_property_returns_acquisition_order(self):
        a = _RecordingResource("a")
        b = _RecordingResource("b")
        c = _RecordingResource("c")
        with StoryRun() as run:
            run.acquire(a)
            run.acquire(b)
            run.acquire(c)
            assert run.resources == [a, b, c]

    def test_released_property_flips_on_release(self):
        a = _RecordingResource("a")
        assert a.released is False
        a.release()
        assert a.released is True


# ===========================================================================
# Branch resource
# ===========================================================================


class TestBranch:
    def test_default_release_is_noop(self, tmp_path):
        """Branches survive on purpose — release should not delete by default."""
        branch = Branch(name="feat-x", repo_root=tmp_path)
        with patch("idea_board.story_run.subprocess.run") as mock_run:
            branch.release()
        mock_run.assert_not_called()

    def test_delete_on_release_runs_branch_dash_d(self, tmp_path):
        branch = Branch(
            name="feat-x", repo_root=tmp_path, delete_on_release=True,
        )
        with patch("idea_board.story_run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            branch.release()

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["git", "branch", "-D"]
        assert cmd[-1] == "feat-x"

    def test_delete_failure_is_swallowed(self, tmp_path):
        """git failure must not raise out of release."""
        branch = Branch(
            name="feat-x", repo_root=tmp_path, delete_on_release=True,
        )
        with patch("idea_board.story_run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="not a branch", stdout="",
            )
            # Must not raise
            branch.release()

    def test_delete_subprocess_crash_is_swallowed(self, tmp_path):
        branch = Branch(
            name="feat-x", repo_root=tmp_path, delete_on_release=True,
        )
        with patch(
            "idea_board.story_run.subprocess.run",
            side_effect=OSError("git not found"),
        ):
            branch.release()  # must not raise


# ===========================================================================
# Worktree resource
# ===========================================================================


class TestWorktree:
    def test_release_when_path_already_gone(self, tmp_path):
        """If the worktree directory is already gone, prune still runs."""
        # Use a path that doesn't exist
        wt_path = tmp_path / "does-not-exist"
        wt = Worktree(repo_root=tmp_path, path=wt_path)

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("idea_board.story_run.subprocess.run", side_effect=fake_run):
            wt.release()

        # Only prune runs when path is missing.
        assert len(calls) == 1
        assert "prune" in calls[0]

    def test_release_runs_remove_then_prune(self, tmp_path):
        wt_path = tmp_path / "my-worktree"
        wt_path.mkdir()
        wt = Worktree(repo_root=tmp_path, path=wt_path)

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            # Simulate `git worktree remove --force` succeeding by deleting
            # the directory ourselves.
            if "remove" in cmd:
                import shutil as _sh
                _sh.rmtree(wt_path, ignore_errors=True)
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("idea_board.story_run.subprocess.run", side_effect=fake_run):
            wt.release()

        assert any("remove" in c for c in calls)
        assert any("prune" in c for c in calls)

    def test_rmtree_fallback_when_git_remove_leaves_directory(self, tmp_path):
        """If git refuses to remove and dir survives, rmtree fallback runs."""
        wt_path = tmp_path / "my-worktree"
        wt_path.mkdir()
        (wt_path / "leftover.txt").write_text("data")

        wt = Worktree(repo_root=tmp_path, path=wt_path)

        # git pretends to fail and the directory survives.
        with patch("idea_board.story_run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="refused", stdout="",
            )
            wt.release()

        # Fallback rmtree should have removed it.
        assert not wt_path.exists(), "rmtree fallback should have removed the dir"

    def test_release_swallows_subprocess_crash(self, tmp_path):
        wt_path = tmp_path / "my-worktree"
        wt_path.mkdir()
        wt = Worktree(repo_root=tmp_path, path=wt_path)

        with patch(
            "idea_board.story_run.subprocess.run",
            side_effect=OSError("git missing"),
        ):
            # Must not raise even when subprocess.run blows up
            wt.release()

    def test_release_is_idempotent(self, tmp_path):
        wt_path = tmp_path / "my-worktree"
        wt_path.mkdir()
        wt = Worktree(repo_root=tmp_path, path=wt_path)

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if "remove" in cmd:
                import shutil as _sh
                _sh.rmtree(wt_path, ignore_errors=True)
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("idea_board.story_run.subprocess.run", side_effect=fake_run):
            wt.release()
            calls_after_first = len(calls)
            wt.release()
            calls_after_second = len(calls)

        assert calls_after_second == calls_after_first, "second release must be no-op"


# ===========================================================================
# ModelHandle resource
# ===========================================================================


class TestModelHandle:
    def test_release_posts_keep_alive_zero(self):
        handle = ModelHandle(
            model_tag="qwen3-coder:30b-a3b-q4_K_M", host="http://1.2.3.4:11434",
        )

        post_calls: list[dict] = []

        def fake_post(url, json=None, timeout=None):
            post_calls.append({"url": url, "json": json})
            return MagicMock(status_code=200)

        # Need to patch where requests.post resolves at call-time.
        with patch.dict("sys.modules"):
            import requests as _req
            with patch.object(_req, "post", side_effect=fake_post):
                handle.release()

        assert len(post_calls) == 1
        assert post_calls[0]["url"] == "http://1.2.3.4:11434/api/generate"
        assert post_calls[0]["json"]["model"] == "qwen3-coder:30b-a3b-q4_K_M"
        assert post_calls[0]["json"]["keep_alive"] == 0

    def test_empty_model_tag_is_noop(self):
        handle = ModelHandle(model_tag="", host="http://1.2.3.4:11434")

        # No requests should fire — patch & confirm 0 calls.
        with patch.dict("sys.modules"):
            import requests as _req
            with patch.object(_req, "post") as mock_post:
                handle.release()
        mock_post.assert_not_called()

    def test_release_swallows_network_error(self):
        handle = ModelHandle(
            model_tag="some-model", host="http://nowhere.invalid:11434",
        )

        import requests as _req
        with patch.object(
            _req, "post", side_effect=_req.ConnectionError("dns fail"),
        ):
            handle.release()  # must not raise

    def test_release_swallows_non_200(self):
        handle = ModelHandle(model_tag="m", host="http://h:11434")
        import requests as _req
        with patch.object(
            _req, "post",
            return_value=MagicMock(status_code=500),
        ):
            handle.release()  # must not raise


# ===========================================================================
# Integration-ish: a realistic run with multiple resource types
# ===========================================================================


class TestIntegration:
    def test_realistic_run_releases_all_in_order(self, tmp_path):
        """Stand in for the actual A/B flow: branch + worktree + model."""
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        order: list[str] = []

        class _Tracking(Resource):
            def __init__(self, name: str) -> None:
                super().__init__(label=name)
                self.name = name

            def _do_release(self) -> None:
                order.append(self.name)

        with StoryRun(label="TK-9999") as run:
            run.acquire(_Tracking("branch"))
            run.acquire(_Tracking("worktree"))
            run.acquire(_Tracking("model_a"))
            # Simulated A/B flow: model_a released early before model_b
            # is acquired.
            run.resources[-1].release()
            assert order == ["model_a"]
            run.acquire(_Tracking("model_b"))
            # …work happens…

        # On context exit:
        # - model_b releases (LIFO)
        # - model_a is already released, no-op
        # - worktree releases
        # - branch releases
        assert order == ["model_a", "model_b", "worktree", "branch"]

    def test_failure_during_acquire_still_releases_prior(self, tmp_path):
        """If acquiring resource N fails, resources 1..N-1 must still release."""
        order: list[str] = []

        class _Tracking(Resource):
            def __init__(self, name: str) -> None:
                super().__init__(label=name)
                self.name = name

            def _do_release(self) -> None:
                order.append(self.name)

        class _BrokenAcquire(Resource):
            def __init__(self) -> None:
                super().__init__(label="broken")
                # Simulate setup failure: something in __init__ raises.
                raise RuntimeError("acquire failed")

            def _do_release(self) -> None:
                order.append("broken")

        with pytest.raises(RuntimeError, match="acquire failed"):
            with StoryRun() as run:
                run.acquire(_Tracking("first"))
                run.acquire(_Tracking("second"))
                # Constructor raises, so this never gets registered.
                run.acquire(_BrokenAcquire())

        # The two that were successfully acquired must have released.
        assert order == ["second", "first"]
