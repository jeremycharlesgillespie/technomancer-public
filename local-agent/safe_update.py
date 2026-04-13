#!/usr/bin/env python3
"""
Safe Update Workflow - Automated branch, test, merge cycle.

Creates a timestamped git branch, runs tests, and merges to main
only if all tests pass. Restarts the Discord bot after merge.

Usage:
    python safe_update.py <short-name>     # Start new update workflow
    python safe_update.py continue         # Continue after making changes
    python safe_update.py abort            # Abort and return to main
    python safe_update.py status           # Show current status

Examples:
    python safe_update.py add-search-tests
    python safe_update.py fix-memory-bug
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from agent.logging_config import get_safe_update_logger

# Constants
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent  # technomancer/
MAIN_BRANCH = "main"
STATE_FILE = SCRIPT_DIR / ".safe_update_state"

# Initialize logger
_logger = get_safe_update_logger()


class SafeUpdateError(Exception):
    """Custom exception for workflow errors."""

    pass


def log(msg: str, level: str = "INFO"):
    """Log message using centralized logging."""
    level_map = {
        "INFO": _logger.info,
        "WARNING": _logger.warning,
        "ERROR": _logger.error,
        "DEBUG": _logger.debug,
    }
    log_func = level_map.get(level.upper(), _logger.info)
    log_func(msg)


def run_git(args: list, check: bool = True, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a git command and return result."""
    cmd = ["git"] + args
    log(f"Git: {' '.join(args)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or REPO_ROOT)
    if check and result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        raise SafeUpdateError(f"Git command failed: {error_msg}")
    return result


def verify_clean_state() -> bool:
    """Check that working directory is clean."""
    result = run_git(["status", "--porcelain"], check=False)
    if result.stdout.strip():
        log("Working directory has uncommitted changes:", "WARNING")
        for line in result.stdout.strip().split("\n")[:10]:
            log(f"  {line}")
        return False
    return True


def get_current_branch() -> str:
    """Get name of current branch."""
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def branch_exists(branch_name: str) -> bool:
    """Check if a branch exists."""
    result = run_git(["branch", "--list", branch_name], check=False)
    return bool(result.stdout.strip())


def create_branch(short_name: str) -> str:
    """Create and checkout new branch with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    branch_name = f"{timestamp}-{short_name}"

    # Ensure we're on main first
    current = get_current_branch()
    if current != MAIN_BRANCH:
        log(f"Currently on {current}, switching to {MAIN_BRANCH}")
        run_git(["checkout", MAIN_BRANCH])

    # Try to pull latest (non-fatal if no remote)
    try:
        run_git(["pull", "origin", MAIN_BRANCH], check=False)
    except Exception:
        log("Could not pull from remote (may not exist)", "WARNING")

    # Create and checkout new branch
    run_git(["checkout", "-b", branch_name])
    log(f"Created branch: {branch_name}")
    return branch_name


def run_tests() -> Tuple[bool, str]:
    """Run pytest and return (success, output)."""
    log("Running pytest...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
    )

    output = result.stdout + result.stderr

    # Check for actual test failures vs collection issues
    success = result.returncode == 0

    if success:
        log("All tests passed!")
    else:
        log("Tests failed!", "ERROR")
        # Show summary
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line or "passed" in line:
                log(f"  {line}")

    return success, output


def merge_to_main(branch_name: str) -> bool:
    """Merge branch to main and return success."""
    log(f"Merging {branch_name} to {MAIN_BRANCH}...")

    try:
        # Switch to main
        run_git(["checkout", MAIN_BRANCH])

        # Merge with --no-ff for clear history
        merge_msg = f"Merge branch '{branch_name}' - automated safe_update"
        run_git(["merge", "--no-ff", branch_name, "-m", merge_msg])

        log(f"Successfully merged {branch_name} to {MAIN_BRANCH}")
        return True

    except SafeUpdateError as e:
        log(f"Merge failed: {e}", "ERROR")
        # Try to abort merge if in progress
        run_git(["merge", "--abort"], check=False)
        # Return to the branch
        run_git(["checkout", branch_name], check=False)
        return False


def delete_branch(branch_name: str):
    """Delete the feature branch after merge."""
    try:
        run_git(["branch", "-d", branch_name])
        log(f"Deleted branch: {branch_name}")
    except SafeUpdateError:
        log(f"Could not delete branch {branch_name} (may need manual cleanup)", "WARNING")


def restart_bot():
    """Restart the Discord bot via bot_service.py."""
    log("Restarting Discord bot...")
    bot_service = SCRIPT_DIR / "bot_service.py"

    if not bot_service.exists():
        log("bot_service.py not found, skipping restart", "WARNING")
        return False

    try:
        # Stop the bot
        log("Stopping bot...")
        stop_result = subprocess.run(
            [sys.executable, str(bot_service), "stop"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if stop_result.returncode != 0:
            log(f"Stop output: {stop_result.stdout}{stop_result.stderr}", "WARNING")

        # Start the bot
        log("Starting bot...")
        start_result = subprocess.run(
            [sys.executable, str(bot_service), "start"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if start_result.returncode == 0:
            log("Bot restarted successfully")
            return True
        else:
            log(f"Bot start may have failed: {start_result.stderr}", "WARNING")
            return False

    except subprocess.TimeoutExpired:
        log("Bot restart timed out", "WARNING")
        return False
    except Exception as e:
        log(f"Bot restart error: {e}", "ERROR")
        return False


def push_to_remote() -> bool:
    """Push main branch to origin after merge."""
    try:
        run_git(["push", "origin", MAIN_BRANCH])
        log("Pushed to origin/main")
        return True
    except SafeUpdateError as e:
        log(f"Push failed: {e}", "WARNING")
        return False


def run_quality_tests() -> bool:
    """Run post-deploy quality tests using Claude to grade Ollama responses."""
    log("Running post-deploy quality tests...")
    quality_test = SCRIPT_DIR / "quality_test.py"

    if not quality_test.exists():
        log("quality_test.py not found, skipping QA", "WARNING")
        return True

    try:
        result = subprocess.run(
            [sys.executable, str(quality_test), "--quick"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SCRIPT_DIR,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            log("Quality tests passed")
            return True
        else:
            log("Quality tests had failures (non-blocking)", "WARNING")
            return False

    except subprocess.TimeoutExpired:
        log("Quality tests timed out (120s)", "WARNING")
        return False
    except Exception as e:
        log(f"Quality test error: {e}", "WARNING")
        return False


def check_bot_running() -> bool:
    """Check if the bot process is alive by reading its PID file.

    Uses ctypes on Windows (os.kill signal 0 doesn't work there) and
    os.kill on other platforms. Both are instant — no subprocess needed.
    """
    pid_file = SCRIPT_DIR / "bot.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False

    if sys.platform == "win32":
        # Windows: use kernel32 OpenProcess to check if PID exists
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def post_deploy_health_check(
    startup_grace: int = 20,
    check_interval: int = 5,
    total_duration: int = 30,
) -> bool:
    """Poll bot status after deploy to verify it stays alive.

    Waits for the bot to finish starting, then monitors for stability.

    Args:
        startup_grace: Seconds to wait before first check (bot needs time
            to initialize Discord connection, load models, etc.).
        check_interval: Seconds between status checks after grace period.
        total_duration: Total seconds to monitor after grace period.

    Returns:
        True if bot stayed alive for the full duration, False if it crashed.
    """
    log(f"Post-deploy health check: {startup_grace}s grace + "
        f"{total_duration}s monitoring...")

    # Grace period — let the bot finish starting
    log(f"  Waiting {startup_grace}s for bot to initialize...")
    time.sleep(startup_grace)

    if not check_bot_running():
        log("  Bot not running after grace period — likely crashed on startup", "ERROR")
        return False

    log("  Bot is alive after grace period, monitoring stability...")

    checks = total_duration // check_interval
    for i in range(checks):
        time.sleep(check_interval)
        alive = check_bot_running()
        log(f"  Health check {i + 1}/{checks}: {'alive' if alive else 'DEAD'}")
        if not alive:
            log("Bot crashed during post-deploy health check!", "ERROR")
            return False

    log("Post-deploy health check passed — bot is stable")
    return True


def rollback_deploy(reverted_branch: str = "") -> Tuple[bool, str]:
    """Revert the last merge commit, restart bot on previous version, notify Discord.

    Args:
        reverted_branch: Name of the branch that was merged (for logging).

    Returns:
        (success, message) tuple.
    """
    log("ROLLBACK: Reverting last deploy...", "ERROR")

    # Capture the commit we're reverting
    try:
        result = run_git(["rev-parse", "--short", "HEAD"])
        bad_commit = result.stdout.strip()
    except SafeUpdateError:
        bad_commit = "unknown"

    # Revert the merge commit
    try:
        run_git(["revert", "HEAD", "--no-edit", "-m", "1"])
        log(f"ROLLBACK: Reverted commit {bad_commit}")
    except SafeUpdateError as e:
        msg = f"ROLLBACK FAILED: Could not revert {bad_commit}: {e}"
        log(msg, "ERROR")
        _send_rollback_notification(bad_commit, reverted_branch, success=False, error=str(e))
        return False, msg

    # Push the revert
    try:
        run_git(["push", "origin", MAIN_BRANCH])
        log("ROLLBACK: Pushed revert to origin")
    except SafeUpdateError:
        log("ROLLBACK: Could not push revert (push manually)", "WARNING")

    # Restart bot on the reverted (good) code
    log("ROLLBACK: Restarting bot on previous version...")
    bot_ok = restart_bot()

    msg = (
        f"Auto-rollback complete. Reverted commit {bad_commit}"
        f"{f' (branch: {reverted_branch})' if reverted_branch else ''}. "
        f"Bot restart: {'OK' if bot_ok else 'FAILED'}."
    )
    log(msg)

    # Send Discord notification
    _send_rollback_notification(bad_commit, reverted_branch, success=True, bot_ok=bot_ok)

    return True, msg


def _send_rollback_notification(
    bad_commit: str,
    branch_name: str,
    success: bool,
    bot_ok: bool = False,
    error: str = "",
) -> None:
    """Send a Discord webhook notification about the rollback."""
    try:
        from agent.config import settings
        webhook_url = settings.discord_webhook_url
    except Exception:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not webhook_url:
        log("No DISCORD_WEBHOOK_URL configured — cannot send rollback notification", "WARNING")
        return

    try:
        import requests as req

        if success:
            message = (
                f"\u26a0\ufe0f **Auto-Rollback Triggered**\n\n"
                f"Deploy of `{bad_commit}`"
                f"{f' (branch `{branch_name}`)' if branch_name else ''}"
                f" crashed within 30s of restart.\n\n"
                f"**Action taken:** `git revert HEAD` — bot restarted on previous version.\n"
                f"**Bot status:** {'Running' if bot_ok else 'NOT running — manual intervention needed'}\n\n"
                f"Investigate the crash and re-deploy when fixed."
            )
        else:
            message = (
                f"\U0001f6a8 **Auto-Rollback FAILED**\n\n"
                f"Deploy of `{bad_commit}`"
                f"{f' (branch `{branch_name}`)' if branch_name else ''}"
                f" crashed, and the automatic revert failed.\n\n"
                f"**Error:** `{error[:300]}`\n\n"
                f"**Manual intervention required immediately.**"
            )

        req.post(webhook_url, json={"content": message}, timeout=10)
        log("Rollback notification sent to Discord")
    except Exception as e:
        log(f"Failed to send rollback notification: {e}", "WARNING")


def save_state(branch_name: str):
    """Save workflow state to file."""
    STATE_FILE.write_text(branch_name, encoding="utf-8")


def load_state() -> Optional[str]:
    """Load workflow state from file."""
    if STATE_FILE.exists():
        return STATE_FILE.read_text(encoding="utf-8").strip()
    return None


def clear_state():
    """Clear workflow state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def show_status():
    """Show current git status and workflow state."""
    print("=" * 60)
    print("Safe Update Status")
    print("=" * 60)

    # Current branch
    try:
        branch = get_current_branch()
        print(f"Current branch: {branch}")
    except Exception as e:
        print(f"Error getting branch: {e}")
        return

    # Workflow state
    saved_branch = load_state()
    if saved_branch:
        print(f"Active workflow: {saved_branch}")
        if branch == saved_branch:
            print("  -> You are on the workflow branch. Make changes and run 'continue'")
        else:
            print(f"  -> WARNING: Expected to be on {saved_branch}")
    else:
        print("No active workflow")

    # Working directory status
    result = run_git(["status", "--porcelain"], check=False)
    if result.stdout.strip():
        print("\nUncommitted changes:")
        for line in result.stdout.strip().split("\n")[:10]:
            print(f"  {line}")
    else:
        print("\nWorking directory clean")

    # Recent commits
    result = run_git(["log", "--oneline", "-5"], check=False)
    if result.returncode == 0:
        print("\nRecent commits:")
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")

    print("=" * 60)


def start_workflow(short_name: str):
    """Start a new update workflow."""
    log("=" * 60)
    log("Safe Update Workflow - Starting")
    log("=" * 60)

    # Check for existing workflow
    existing = load_state()
    if existing:
        log(f"Existing workflow found for branch: {existing}", "ERROR")
        log("Run 'python safe_update.py abort' first, or 'continue' to resume")
        sys.exit(1)

    # Validate short name
    clean_name = short_name.replace("-", "").replace("_", "")
    if not clean_name.isalnum():
        log("Short name should be alphanumeric with dashes/underscores only", "ERROR")
        sys.exit(1)

    try:
        # Step 1: Verify clean state
        log("Step 1: Checking working directory state...")
        if not verify_clean_state():
            log("Please commit or stash changes before running safe_update", "ERROR")
            sys.exit(1)

        # Step 2: Create branch
        log("Step 2: Creating feature branch...")
        branch_name = create_branch(short_name)

        # Save state
        save_state(branch_name)

        # Step 3: Prompt for code changes
        print()
        print("=" * 60)
        print("BRANCH CREATED SUCCESSFULLY")
        print("=" * 60)
        print()
        print(f"Branch: {branch_name}")
        print()
        print("Next steps:")
        print("  1. Make your code changes")
        print("  2. Stage and commit: git add . && git commit -m 'Description'")
        print("  3. Run: python safe_update.py continue")
        print()
        print("Or to abort: python safe_update.py abort")
        print("=" * 60)

    except SafeUpdateError as e:
        log(f"Workflow failed: {e}", "ERROR")
        clear_state()
        sys.exit(1)


def continue_workflow():
    """Continue workflow after code changes."""
    # Block execution from executor context — the executor handles deploy itself
    if os.environ.get("EXECUTOR_MODE"):
        log("BLOCKED: safe_update.py continue cannot run inside the executor.", "ERROR")
        log("The executor handles testing, merging, and deployment automatically.", "ERROR")
        sys.exit(1)

    log("=" * 60)
    log("Safe Update Workflow - Continuing")
    log("=" * 60)

    branch_name = load_state()
    if not branch_name:
        log("No active workflow. Run 'python safe_update.py <name>' first.", "ERROR")
        sys.exit(1)

    log(f"Resuming workflow for branch: {branch_name}")

    try:
        # Verify we're on the right branch
        current = get_current_branch()
        if current != branch_name:
            log(f"Expected branch {branch_name}, but on {current}")
            log(f"Switching to {branch_name}...")
            run_git(["checkout", branch_name])

        # Check for uncommitted changes
        if not verify_clean_state():
            log("Please commit your changes before continuing", "ERROR")
            log("  git add .")
            log("  git commit -m 'Your message'")
            sys.exit(1)

        # Step 3: Run tests
        log("Step 3: Running tests...")
        tests_passed, test_output = run_tests()

        if not tests_passed:
            print()
            print("=" * 60)
            print("TESTS FAILED")
            print("=" * 60)
            print()
            print("Fix the failing tests and run: python safe_update.py continue")
            print("Or abort: python safe_update.py abort")
            print("=" * 60)
            sys.exit(1)

        # Step 4: Merge to main
        log("Step 4: Merging to main...")
        if not merge_to_main(branch_name):
            log("Merge failed - manual intervention needed", "ERROR")
            print()
            print("=" * 60)
            print("MERGE FAILED")
            print("=" * 60)
            print("Resolve conflicts manually, then run continue again")
            print("=" * 60)
            sys.exit(1)

        # Step 5: Clean up branch
        log("Step 5: Cleaning up...")
        delete_branch(branch_name)
        clear_state()

        # Step 6: Restart bot
        log("Step 6: Restarting bot...")
        bot_ok = restart_bot()

        # Step 6b: Post-deploy health check (verify bot stays alive 30s)
        if bot_ok:
            log("Step 6b: Post-deploy health check...")
            healthy = post_deploy_health_check()
            if not healthy:
                log("Bot crashed after deploy — initiating auto-rollback", "ERROR")
                rollback_ok, rollback_msg = rollback_deploy(reverted_branch=branch_name)
                print()
                print("=" * 60)
                print("DEPLOY ROLLED BACK")
                print("=" * 60)
                print()
                print(rollback_msg)
                print()
                print("The bot crashed within 30 seconds of restart.")
                print("The merge has been reverted and the bot restarted")
                print("on the previous version.")
                print()
                print("Next steps:")
                print("  1. Check crash logs for the root cause")
                print("  2. Fix the issue on a new branch")
                print("  3. Re-deploy with safe_update.py")
                print("=" * 60)
                sys.exit(1)

        # Step 7: Push to remote
        log("Step 7: Pushing to origin...")
        push_ok = push_to_remote()

        # Step 8: Post-deploy quality test
        log("Step 8: Running post-deploy quality tests...")
        qa_ok = run_quality_tests()

        # Step 9: Update README with live stats
        readme_script = Path(__file__).parent / "generate_readme.py"
        if readme_script.exists():
            log("Step 9: Updating README...")
            try:
                result = subprocess.run(
                    [sys.executable, str(readme_script)],
                    capture_output=True, text=True, timeout=660,
                    cwd=Path(__file__).parent,
                )
                if result.returncode == 0:
                    # Commit the updated READMEs (local-agent + repo root)
                    run_git(["add", "README.md"], check=False)
                    root_readme = Path(__file__).parent.parent / "README.md"
                    if root_readme.exists():
                        subprocess.run(
                            ["git", "add", str(root_readme)],
                            capture_output=True, text=True, timeout=10,
                            cwd=Path(__file__).parent.parent,
                        )
                    try:
                        run_git(["commit", "-m", "Update README with latest stats [auto]"])
                        run_git(["push", "origin", MAIN_BRANCH], check=False)
                        log("README updated with latest stats")
                    except Exception:
                        log("README unchanged (no new stats)")
                else:
                    log(f"README generation failed: {result.stderr[:200]}")
            except Exception as e:
                log(f"README update error: {e}")

        # Step 10: Auto-publish to public repo
        publish_ok = False
        publish_script = Path(__file__).parent / "publish.py"
        public_repo = Path(__file__).parent.parent.parent / "technomancer-public"
        if publish_script.exists() and public_repo.exists():
            log("Step 10: Publishing to technomancer-public...")
            try:
                result = subprocess.run(
                    [sys.executable, str(publish_script), "--push", "--force"],
                    capture_output=True, text=True, timeout=120,
                    cwd=Path(__file__).parent,
                )
                if result.returncode == 0:
                    log("Published to technomancer-public")
                    publish_ok = True
                else:
                    log(f"Publish failed: {result.stderr[:200] or result.stdout[:200]}")
            except Exception as e:
                log(f"Publish error: {e}")
        else:
            log("Step 10: Skipping publish (no public repo found)")

        # Done!
        print()
        print("=" * 60)
        print("WORKFLOW COMPLETE!")
        print("=" * 60)
        print()
        print(f"Branch {branch_name} merged to {MAIN_BRANCH}")
        if bot_ok:
            print("Bot has been restarted")
        else:
            print("Bot restart may need manual attention")
        if push_ok:
            print("Pushed to origin/main")
        else:
            print("WARNING: Push to origin failed - push manually")
        if publish_ok:
            print("Published to technomancer-public")
        elif public_repo.exists():
            print("WARNING: Publish to technomancer-public failed")
        if qa_ok:
            print("Quality tests passed")
        else:
            print("WARNING: Quality tests had issues (see above)")
        print("=" * 60)

    except SafeUpdateError as e:
        log(f"Workflow failed: {e}", "ERROR")
        sys.exit(1)


def abort_workflow():
    """Abort current workflow and return to main."""
    log("Aborting workflow...")

    branch_name = load_state()
    if not branch_name:
        log("No active workflow to abort")
        return

    log(f"Aborting workflow for branch: {branch_name}")

    try:
        # Switch to main
        current = get_current_branch()
        if current != MAIN_BRANCH:
            run_git(["checkout", MAIN_BRANCH])

        # Delete the branch (force in case it has unmerged changes)
        if branch_exists(branch_name):
            run_git(["branch", "-D", branch_name], check=False)
            log(f"Deleted branch: {branch_name}")

        clear_state()
        log("Workflow aborted, returned to main")

    except Exception as e:
        log(f"Abort error: {e}", "WARNING")
        clear_state()


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "status":
        show_status()
    elif command == "continue":
        continue_workflow()
    elif command == "abort":
        abort_workflow()
    elif command in ("help", "-h", "--help"):
        print_usage()
    elif command.startswith("-"):
        print(f"Unknown option: {command}")
        print_usage()
        sys.exit(1)
    else:
        # Treat as short name for new workflow
        start_workflow(command)


if __name__ == "__main__":
    main()
