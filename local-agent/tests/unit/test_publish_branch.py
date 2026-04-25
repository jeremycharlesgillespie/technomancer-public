"""Tests for ``publish.py --branch <name>`` mode (A/B harness mirror).

The branch mode mirrors a feature branch from the private repo to the
public ``technomancer-public`` repo. These tests cover the pure helpers
(``_parse_branch_arg``, ``_public_remote_branch_exists``) and the
``_publish_branch`` orchestration with subprocess mocked out.

Scope:
- ``_parse_branch_arg`` recognises both ``--branch X`` and ``--branch=X``.
- Dirty public working tree → exit 1 without attempting checkout.
- Branch missing on remote → created from origin/main.
- Branch exists on remote → reset to remote tip (``checkout -B``).
- Empty diff after sync → exit 0, no commit, returns to main.
- Secret detected → exit 1, no push, returns to main.
- Successful path → commit, push, return to main, exit 0.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import publish


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _parse_branch_arg
# ---------------------------------------------------------------------------

def test_parse_branch_arg_space_form() -> None:
    assert publish._parse_branch_arg(
        ["publish.py", "--branch", "feature-x", "--push"]
    ) == "feature-x"


def test_parse_branch_arg_equals_form() -> None:
    assert publish._parse_branch_arg(
        ["publish.py", "--branch=feature-y"]
    ) == "feature-y"


def test_parse_branch_arg_absent() -> None:
    assert publish._parse_branch_arg(["publish.py", "--push"]) is None


def test_parse_branch_arg_trailing_branch_no_value() -> None:
    """``--branch`` at the very end with no value → treated as absent."""
    assert publish._parse_branch_arg(["publish.py", "--branch"]) is None


# ---------------------------------------------------------------------------
# _publish_branch — full orchestration
# ---------------------------------------------------------------------------

def test_publish_branch_aborts_on_dirty_public(capsys) -> None:
    with patch("publish.git_status", return_value="M  some_file.py"):
        rc = publish._publish_branch("feature-x", force=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "dirty" in out.lower()


def test_publish_branch_creates_branch_from_main_when_missing(capsys) -> None:
    """When origin/<branch> doesn't exist, checkout creates it from main."""
    with patch("publish.git_status") as mstatus, \
         patch("publish.subprocess.run") as mrun, \
         patch("publish.sync_files", return_value=(3, 0, 0)), \
         patch("publish.scan_for_secrets", return_value=[]):
        # First git_status: clean public. After sync: changes present.
        mstatus.side_effect = ["", "M file"]
        # Subprocess sequence:
        #  1. fetch origin <branch>           (in _public_remote_branch_exists)
        #  2. rev-parse origin/<branch>       → returncode != 0 (missing)
        #  3. fetch origin main               (in _checkout_public_branch)
        #  4. checkout -B <branch> origin/main → 0
        #  5. git add -A
        #  6. git commit -m ...
        #  7. git push origin <branch>
        #  8. git checkout main
        mrun.side_effect = [
            _cp(0),                               # 1. fetch origin branch
            _cp(1, stderr="not found"),           # 2. rev-parse miss
            _cp(0),                               # 3. fetch main
            _cp(0),                               # 4. checkout -B from main
            _cp(0),                               # 5. add
            _cp(0),                               # 6. commit
            _cp(0),                               # 7. push
            _cp(0),                               # 8. checkout main
        ]
        rc = publish._publish_branch("feature-x", force=True)
    assert rc == 0
    # Confirm checkout-from-main was invoked.
    checkout_calls = [c for c in mrun.call_args_list if "checkout" in c.args[0]]
    assert any("origin/main" in c.args[0] for c in checkout_calls)


def test_publish_branch_resets_to_remote_when_branch_exists(capsys) -> None:
    """When origin/<branch> exists, checkout -B uses origin/<branch>."""
    with patch("publish.git_status") as mstatus, \
         patch("publish.subprocess.run") as mrun, \
         patch("publish.sync_files", return_value=(3, 0, 0)), \
         patch("publish.scan_for_secrets", return_value=[]):
        mstatus.side_effect = ["", "M file"]
        mrun.side_effect = [
            _cp(0),                # fetch origin branch
            _cp(0),                # rev-parse hit (branch exists)
            _cp(0),                # checkout -B branch origin/branch
            _cp(0),                # add
            _cp(0),                # commit
            _cp(0),                # push
            _cp(0),                # checkout main
        ]
        rc = publish._publish_branch("existing-br", force=True)
    assert rc == 0
    checkout_calls = [c for c in mrun.call_args_list if "checkout" in c.args[0]]
    assert any("origin/existing-br" in c.args[0] for c in checkout_calls)


def test_publish_branch_no_changes_skips_commit(capsys) -> None:
    """Empty diff after sync → exit 0, no commit, no push."""
    with patch("publish.git_status") as mstatus, \
         patch("publish.subprocess.run") as mrun, \
         patch("publish.sync_files", return_value=(0, 0, 0)), \
         patch("publish.scan_for_secrets", return_value=[]):
        # Public clean, after sync still clean.
        mstatus.side_effect = ["", ""]
        mrun.side_effect = [
            _cp(0), _cp(0),    # fetch + rev-parse (existing)
            _cp(0),            # checkout -B
            _cp(0),            # final checkout main
        ]
        rc = publish._publish_branch("br", force=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No public-visible changes" in out
    # No commit / push attempted.
    cmds = [c.args[0] for c in mrun.call_args_list]
    assert not any("commit" in cmd for cmd in cmds)
    assert not any("push" in cmd for cmd in cmds)


def test_publish_branch_aborts_on_secret_finding(capsys) -> None:
    """Secret detection blocks the push, returns to main, exits 1."""
    with patch("publish.git_status") as mstatus, \
         patch("publish.subprocess.run") as mrun, \
         patch("publish.sync_files", return_value=(2, 0, 0)), \
         patch("publish.scan_for_secrets", return_value=["TOKEN found in foo.py"]):
        mstatus.side_effect = ["", ""]
        mrun.side_effect = [
            _cp(0), _cp(0),   # fetch + rev-parse
            _cp(0),           # checkout -B
            _cp(0),           # final checkout main
        ]
        rc = publish._publish_branch("br", force=True)
    assert rc == 1
    cmds = [c.args[0] for c in mrun.call_args_list]
    # Must NOT have pushed.
    assert not any("push" in cmd for cmd in cmds)
    # Must have returned to main.
    assert any("checkout" in cmd and "main" in cmd for cmd in cmds)


def test_publish_branch_handles_push_failure(capsys) -> None:
    """git push failure → exit 1, attempt to return to main."""
    with patch("publish.git_status") as mstatus, \
         patch("publish.subprocess.run") as mrun, \
         patch("publish.sync_files", return_value=(2, 0, 0)), \
         patch("publish.scan_for_secrets", return_value=[]):
        mstatus.side_effect = ["", "M file"]
        mrun.side_effect = [
            _cp(0), _cp(0),    # fetch + rev-parse
            _cp(0),            # checkout
            _cp(0),            # add
            _cp(0),            # commit
            subprocess.CalledProcessError(1, "git push"),  # push fails
            _cp(0),            # checkout main fallback
        ]
        rc = publish._publish_branch("br", force=True)
    assert rc == 1


def test_publish_branch_checkout_failure_returns_1(capsys) -> None:
    """If the public-side checkout fails, abort early."""
    with patch("publish.git_status", return_value=""), \
         patch("publish.subprocess.run") as mrun:
        mrun.side_effect = [
            _cp(0),                              # fetch origin branch
            _cp(1, stderr="boom"),               # rev-parse miss
            _cp(0),                              # fetch main
            _cp(1, stderr="checkout failed"),    # checkout -B fails
        ]
        rc = publish._publish_branch("br", force=True)
    assert rc == 1
