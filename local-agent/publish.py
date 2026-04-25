"""
Publish System — Sync code from private repo to technomancer-public.

Copies all publishable files from the private technomancer repo to the
public repo, runs safety checks (no secrets, no personal data), commits,
and pushes.

Usage:
    python publish.py                          # Preview what would be synced
    python publish.py --push                   # Sync and push to GitHub
    python publish.py --push --force           # Skip confirmation prompt
    python publish.py --branch <name> --push   # Mirror a feature branch
                                               # (used by the A/B harness)

The system:
1. Copies all code files (.py, .toml, .yaml, .md, .html, etc.)
2. Excludes runtime data (.env, ideas.json, profiling/, downloads/, etc.)
3. Scans for secrets and personal data before committing
4. Commits with a descriptive message
5. Pushes to the public repo

`--branch <name>` mode mirrors the named feature branch from private to
a like-named branch on `technomancer-public`. The same `local-agent/**`
filter and secret gate apply, so only the publishable subset reaches
the public branch. The public repo is left on `main` after the push.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Repo locations
PRIVATE_REPO = Path(__file__).parent.parent  # technomancer/
PUBLIC_REPO = Path(__file__).parent.parent.parent / "technomancer-public"

# Files/dirs to ALWAYS exclude from the public repo
EXCLUDE_PATTERNS = {
    # Runtime data
    ".env",
    ".bridge_token",
    "ideas.json",
    "sent_articles.json",
    "sent_topics.json",
    "service_state.json",
    "claude_queries.log",
    ".safe_update_state",
    "safe_update.log",
    "service.log",
    "bot.pid",
    "user_profile.json",
    "downloaded_videos.json",
    "quality_test.py",
    # Directories
    "profiling",
    "tool_results",
    "data",
    "downloads",
    "__pycache__",
    ".pytest_cache",
    "local_agent.egg-info",
    "context",
    # Personal content
    "docs/shared",
}

# Files/dirs to copy from private to public
SYNC_PATHS = [
    "local-agent/agent/",
    "local-agent/idea_board/",
    "local-agent/tests/",
    "local-agent/monitoring/",
    "local-agent/.env.example",
    "local-agent/.gitignore",
    "local-agent/Makefile",
    "local-agent/pyproject.toml",
    "local-agent/safe_update.py",
    "local-agent/bot_service.py",
    "local-agent/validate.py",
    "local-agent/publish.py",
    "local-agent/discord_cli.py",
    "local-agent/auto_improve.py",
    "local-agent/start_service.bat",
    "local-agent/README.md",
    "local-agent/generate_readme.py",
    "README.md",
    "CLAUDE.md",
    ".pre-commit-config.yaml",
    "docs/",
]

# Patterns that should NEVER appear in public code.
# Generic patterns are hardcoded; personal patterns loaded from
# PUBLISH_BLOCKLIST in .env (comma-separated).
_GENERIC_PATTERNS = [
    r"sk-ant-api\w+",
    r"discord\.com/api/webhooks/\d+/\w+",
]


def _load_secret_patterns() -> list[str]:
    """Load secret scan patterns from generic list + PUBLISH_BLOCKLIST env var."""
    patterns = list(_GENERIC_PATTERNS)
    blocklist = os.environ.get("PUBLISH_BLOCKLIST", "")
    for item in blocklist.split(","):
        item = item.strip()
        if item:
            patterns.append(item)
    return patterns


SECRET_PATTERNS = _load_secret_patterns()


def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from the public repo."""
    path_str = str(path).replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def scan_for_secrets(repo_path: Path) -> list[str]:
    """Scan all .py files for secrets or personal data.

    Returns list of findings (empty = clean).
    """
    findings: list[str] = []
    for f in repo_path.rglob("*.py"):
        if "__pycache__" in str(f) or "test_" in f.name or f.name == "publish.py":
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                try:
                    matches = re.findall(pattern, content)
                except re.error:
                    continue  # Skip invalid regex patterns
                if matches:
                    rel = f.relative_to(repo_path)
                    findings.append(f"  {rel}: {pattern} ({len(matches)} matches)")
        except (OSError, UnicodeDecodeError):
            pass
    return findings


def sync_files() -> tuple[int, int, int]:
    """Copy files from private to public repo.

    Returns (copied, skipped, deleted) counts.
    """
    copied = 0
    skipped = 0

    for sync_path in SYNC_PATHS:
        src = PRIVATE_REPO / sync_path
        dst = PUBLIC_REPO / sync_path

        if not src.exists():
            continue

        if src.is_file():
            if should_exclude(src):
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        elif src.is_dir():
            for f in src.rglob("*"):
                if f.is_dir():
                    continue
                if should_exclude(f):
                    skipped += 1
                    continue
                rel = f.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied += 1

    # Count files in public that don't exist in private (stale)
    deleted = 0
    for f in PUBLIC_REPO.rglob("*"):
        if f.is_dir() or ".git" in str(f):
            continue
        rel = f.relative_to(PUBLIC_REPO)
        src_check = PRIVATE_REPO / rel
        if not src_check.exists() and not should_exclude(f):
            f.unlink()
            deleted += 1

    return copied, skipped, deleted


def git_status(repo: Path) -> str:
    """Get git status summary for a repo."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()


def _get_recent_private_commits(since_last_publish: bool = True) -> list[str]:
    """Get recent commit messages from the private repo.

    If since_last_publish is True, gets commits since the last publish
    to the public repo. Otherwise gets the last 5 commits.
    """
    try:
        # Get the last publish timestamp from the public repo
        if since_last_publish and PUBLIC_REPO.exists():
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI"],
                cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=10,
            )
            last_publish = result.stdout.strip()
            if last_publish:
                # Get private commits since that timestamp
                result = subprocess.run(
                    ["git", "log", f"--since={last_publish}", "--format=%s", "--no-merges"],
                    cwd=PRIVATE_REPO, capture_output=True, text=True, timeout=10,
                )
                commits = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
                if commits:
                    return commits

        # Fallback: last 5 non-merge commits
        result = subprocess.run(
            ["git", "log", "-5", "--format=%s", "--no-merges"],
            cwd=PRIVATE_REPO, capture_output=True, text=True, timeout=10,
        )
        return [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
    except Exception:
        return []


# Patterns for generic/automated commits that should not become the public
# repo's commit title — these bury the real feature work.
_GENERIC_COMMIT_PATTERNS = [
    re.compile(r"^Update README with latest stats"),
    re.compile(r"^Merge branch '2026-"),
    re.compile(r"\[auto\]$"),
    re.compile(r"^\[TK-\d+\] Merge branch"),
    re.compile(r"^\[TK-\d+\] Deploy \+ update stats"),
]


def _is_generic_commit(message: str) -> bool:
    """Return True if the commit message is an auto-generated stats/deploy/merge commit."""
    return any(pat.search(message) for pat in _GENERIC_COMMIT_PATTERNS)


def _filter_meaningful_commits(commits: list[str]) -> list[str]:
    """Drop auto-generated stats/deploy/merge commits so real work is surfaced."""
    return [c for c in commits if not _is_generic_commit(c)]


def _build_publish_message(copied: int, deleted: int) -> str:
    """Build a descriptive commit message from private repo's recent changes."""
    commits = _filter_meaningful_commits(_get_recent_private_commits())

    if not commits:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"Update ({timestamp})\n\n{copied} files synced."

    # Use the most recent meaningful commit as the title
    title = commits[0]

    # If there are multiple commits, list them
    if len(commits) == 1:
        return title
    else:
        body = "\n".join(f"- {c}" for c in commits)
        return f"{title}\n\nChanges included:\n{body}"


def git_commit_and_push(repo: Path, message: str) -> bool:
    """Stage all, commit, and push."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo, check=True, timeout=10,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo, check=True, timeout=60,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        return False


def _public_remote_branch_exists(branch: str) -> bool:
    """Return True if origin/<branch> exists on the public remote."""
    try:
        subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"origin/{branch}"],
            cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _public_repo_clean() -> bool:
    """Return True if the public working tree has no uncommitted changes."""
    return git_status(PUBLIC_REPO) == ""


def _checkout_public_branch(branch: str) -> bool:
    """Switch the public repo to ``branch`` (creating from main if needed)."""
    if _public_remote_branch_exists(branch):
        result = subprocess.run(
            ["git", "checkout", "-B", branch, f"origin/{branch}"],
            cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=30,
        )
    else:
        # Branch doesn't exist on public — create it from main.
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "checkout", "-B", branch, "origin/main"],
            cwd=PUBLIC_REPO, capture_output=True, text=True, timeout=30,
        )
    if result.returncode != 0:
        print(f"  ERROR: checkout {branch} failed: {(result.stderr or result.stdout)[:300]}")
        return False
    return True


def _publish_branch(branch: str, force: bool) -> int:
    """Mirror feature ``branch`` from private to public. Returns exit code.

    Steps:
      1. Verify public working tree clean.
      2. Check out / create the public-side branch.
      3. Run sync_files() — same `local-agent/**` filter as main mode.
      4. Run scan_for_secrets() — same gate.
      5. Commit + push origin <branch>.
      6. Restore public to main.

    Skips the commit + push when sync produced no changes (exit 0). Any
    secret detection or push failure exits non-zero so callers
    (the A/B harness, ad-hoc operators) can surface it.
    """
    print(f"Mode: --branch {branch}")
    print()

    if not _public_repo_clean():
        print("  ERROR: public working tree is dirty — refusing to mirror.")
        print("  Resolve uncommitted changes in technomancer-public first.")
        return 1

    print(f"Step 1: Checkout public branch '{branch}'...")
    if not _checkout_public_branch(branch):
        return 1
    print(f"  On {branch}")
    print()

    print("Step 2: Syncing files...")
    copied, skipped, deleted = sync_files()
    print(f"  Copied: {copied} files")
    print(f"  Skipped: {skipped} (excluded)")
    print(f"  Deleted: {deleted} (stale)")
    print()

    print("Step 3: Scanning for secrets...")
    findings = scan_for_secrets(PUBLIC_REPO)
    if findings:
        print("  BLOCKED — secrets found in public repo:")
        for f in findings:
            print(f)
        # Reset to main so we don't leave a half-mirrored branch checked out.
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=PUBLIC_REPO, capture_output=True, timeout=15,
        )
        return 1
    print("  Clean — no secrets found")
    print()

    status = git_status(PUBLIC_REPO)
    if not status:
        print("Step 4: No public-visible changes for this branch — skipping commit.")
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=PUBLIC_REPO, capture_output=True, timeout=15,
        )
        return 0

    print(f"Step 4: Committing and pushing branch '{branch}'...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = f"[A/B run] {branch} ({timestamp})"
    try:
        subprocess.run(
            ["git", "add", "-A"], cwd=PUBLIC_REPO, check=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=PUBLIC_REPO, check=True, timeout=10,
        )
        subprocess.run(
            ["git", "push", "origin", branch],
            cwd=PUBLIC_REPO, check=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: git operation failed: {e}")
        # Best-effort: leave repo on main even if push failed.
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=PUBLIC_REPO, capture_output=True, timeout=15,
        )
        return 1

    print(f"  Pushed origin/{branch}")
    print()

    # Restore main so the next non-branch publish run has the expected state.
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=PUBLIC_REPO, capture_output=True, timeout=15,
    )
    return 0


def _parse_branch_arg(argv: list[str]) -> str | None:
    """Return the value following ``--branch`` in ``argv`` or None."""
    for i, arg in enumerate(argv):
        if arg == "--branch" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--branch="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    push = "--push" in sys.argv
    force = "--force" in sys.argv
    branch = _parse_branch_arg(sys.argv)

    print("=" * 60)
    print("TECHNOMANCER PUBLISH SYSTEM")
    print("=" * 60)
    print(f"Private: {PRIVATE_REPO}")
    print(f"Public:  {PUBLIC_REPO}")
    print()

    if not PUBLIC_REPO.exists():
        print(f"ERROR: Public repo not found at {PUBLIC_REPO}")
        print("Clone it first: git clone <url> technomancer-public")
        sys.exit(1)

    if branch:
        # --branch mode: mirror a feature branch instead of main. We
        # ignore --force here (commits are auto-named from the branch)
        # and require --push (no preview mode for branch mirroring).
        if not push:
            print("--branch requires --push (no dry-run mode for branch mirrors).")
            sys.exit(1)
        sys.exit(_publish_branch(branch, force))

    # Step 1: Sync files
    print("Step 1: Syncing files...")
    copied, skipped, deleted = sync_files()
    print(f"  Copied: {copied} files")
    print(f"  Skipped: {skipped} (excluded)")
    print(f"  Deleted: {deleted} (stale)")
    print()

    # Step 2: Security scan
    print("Step 2: Scanning for secrets...")
    findings = scan_for_secrets(PUBLIC_REPO)
    if findings:
        print("  BLOCKED — secrets found in public repo:")
        for f in findings:
            print(f)
        print()
        print("Fix these before publishing.")
        sys.exit(1)
    print("  Clean — no secrets found")
    print()

    # Step 3: Show diff
    status = git_status(PUBLIC_REPO)
    if not status:
        print("Step 3: No changes to publish.")
        sys.exit(0)

    print(f"Step 3: Changes to publish:")
    for line in status.split("\n")[:20]:
        print(f"  {line}")
    if status.count("\n") > 20:
        print(f"  ... and {status.count(chr(10)) - 20} more")
    print()

    if not push:
        print("DRY RUN — use --push to actually publish.")
        print(f"  python publish.py --push")
        sys.exit(0)

    # Step 4: Confirm
    if not force:
        resp = input("Publish these changes to technomancer-public? [y/N] ")
        if resp.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    # Step 5: Build commit message from private repo's recent commits
    message = _build_publish_message(copied, deleted)

    print("Step 4: Committing and pushing...")
    if git_commit_and_push(PUBLIC_REPO, message):
        print("  Published successfully!")
    else:
        print("  Publish failed — check git output above.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("PUBLISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
