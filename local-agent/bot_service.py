#!/usr/bin/env python3
"""
Bot Service Controller - Manages Discord bot lifecycle with intelligent retry.

Features:
- Monitors bot process every 30 seconds
- Pre-flight import test catches syntax errors before starting
- Max 3 consecutive failures before alerting
- Discord webhook notification when human intervention needed
- 30-minute cooldown after persistent failures
- Persists state across service restarts

Run with: python bot_service.py
Or as background: pythonw bot_service.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from agent.config import settings
from agent.logging_config import get_bot_service_logger

# Configuration
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "service_state.json"
PID_FILE = SCRIPT_DIR / "bot.pid"

CHECK_INTERVAL = 30  # seconds
MAX_FAILURES = 3
COOLDOWN_MINUTES = 30
OLLAMA_URL = settings.ollama_host

# Initialize logger
_logger = get_bot_service_logger()


def log(msg: str):
    """Log message using centralized logging."""
    _logger.info(msg)


def load_state() -> dict:
    """Load service state from file."""
    default = {
        "bot_pid": None,
        "consecutive_failures": 0,
        "last_failure_time": None,
        "last_error": None,
        "cooldown_until": None,
        "total_restarts": 0,
    }
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            # Merge with defaults for any missing keys
            for key in default:
                if key not in state:
                    state[key] = default[key]
            return state
        except Exception as e:
            log(f"Error loading state: {e}")
    return default


def save_state(state: dict):
    """Save service state to file."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def is_bot_running() -> bool:
    """Check if the bot process is still alive."""
    if not PID_FILE.exists():
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        return False

    # Check if process exists
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return str(pid) in result.stdout
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def test_bot_import() -> tuple[bool, str]:
    """
    Pre-flight check: test if bot module can be imported.
    Catches syntax errors before attempting full start.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from agent.discord_memory_bot import main"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=SCRIPT_DIR,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Import test timed out"
    except Exception as e:
        return False, str(e)


def kill_bot():
    """Force kill the bot process."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            log(f"Killing bot (PID: {pid})")
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.kill(pid, 9)
        except Exception as e:
            log(f"Kill error (may already be dead): {e}")
        finally:
            if PID_FILE.exists():
                PID_FILE.unlink()
        time.sleep(2)


def unload_ollama_models():
    """Unload Ollama models from GPU to free memory."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                name = model.get("name", "unknown")
                log(f"Unloading Ollama model: {name}")
                requests.post(
                    f"{OLLAMA_URL}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30
                )
    except Exception:
        pass  # Ollama not running or no models loaded


def check_ollama_preflight() -> None:
    """Log a warning if Ollama isn't ready for the main chat model.

    Non-fatal: the bot still starts because Ollama often comes up a few
    seconds later, but the warning tells the operator why the first user
    message hangs (was the top user-reported visibility gap — TK-448).
    """
    try:
        from agent.ollama_health import check_ollama_ready

        ready, reason = check_ollama_ready(settings.ollama_model, timeout=5.0)
        if ready:
            log(f"Ollama pre-flight OK for {settings.ollama_model}: {reason}")
        else:
            log(
                f"WARNING: Ollama pre-flight failed for {settings.ollama_model}: "
                f"{reason} — first user message may hang until model loads"
            )
    except Exception as exc:
        log(f"Ollama pre-flight check errored (continuing): {exc}")


def start_bot() -> tuple[bool, str]:
    """
    Start the bot process.
    Returns (success, error_message).
    """
    os.chdir(SCRIPT_DIR)

    check_ollama_preflight()

    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "agent.discord_memory_bot"],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )

        # Wait a moment and check if it's still running
        time.sleep(3)

        if process.poll() is None:
            # Process still running
            PID_FILE.write_text(str(process.pid))
            log(f"Bot started successfully (PID: {process.pid})")
            return True, ""
        else:
            # Process exited immediately
            return False, f"Process exited with code {process.returncode}"

    except Exception as e:
        return False, str(e)


def send_discord_alert(error_msg: str):
    """Send alert to Discord via webhook."""
    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        log("No DISCORD_WEBHOOK_URL configured - cannot send alert")
        return

    # Truncate error message if too long
    if len(error_msg) > 500:
        error_msg = error_msg[:500] + "..."

    payload = {
        "content": (
            f"⚠️ **Bot Service Alert**\n\n"
            f"The Discord bot failed to start after {MAX_FAILURES} attempts.\n\n"
            f"**Error:**\n```\n{error_msg}\n```\n\n"
            f"Entering {COOLDOWN_MINUTES}-minute cooldown. Manual fix required."
        )
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            log("Discord alert sent successfully")
        else:
            log(f"Discord alert failed: {response.status_code}")
    except Exception as e:
        log(f"Failed to send Discord alert: {e}")


def in_cooldown(state: dict) -> bool:
    """Check if we're in cooldown period."""
    if not state.get("cooldown_until"):
        return False

    cooldown_until = datetime.fromisoformat(state["cooldown_until"])
    if datetime.now() < cooldown_until:
        return True

    # Cooldown expired - reset
    state["cooldown_until"] = None
    state["consecutive_failures"] = 0
    save_state(state)
    log("Cooldown expired - resuming monitoring")
    return False


def enter_cooldown(state: dict):
    """Enter cooldown period after persistent failures."""
    cooldown_until = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
    state["cooldown_until"] = cooldown_until.isoformat()
    save_state(state)
    log(f"Entering cooldown until {cooldown_until.strftime('%H:%M:%S')}")


def record_failure(state: dict, error: str):
    """Record a failure and update state."""
    state["consecutive_failures"] += 1
    state["last_failure_time"] = datetime.now().isoformat()
    state["last_error"] = error[:1000]  # Truncate
    save_state(state)
    log(f"Failure #{state['consecutive_failures']}: {error[:200]}")


def reset_failures(state: dict):
    """Reset failure counter after successful start."""
    state["consecutive_failures"] = 0
    state["last_error"] = None
    state["total_restarts"] += 1
    save_state(state)


def main_loop():
    """Main monitoring loop."""
    log("=" * 50)
    log("Bot Service Controller starting")
    log(f"Check interval: {CHECK_INTERVAL}s")
    log(f"Max failures before alert: {MAX_FAILURES}")
    log(f"Cooldown period: {COOLDOWN_MINUTES} minutes")
    log("=" * 50)

    # Check webhook configuration (loaded via settings)
    if not settings.discord_webhook_url:
        log("WARNING: DISCORD_WEBHOOK_URL not set - alerts will not be sent")

    state = load_state()
    log(
        f"Loaded state: {state['total_restarts']} total restarts, "
        f"{state['consecutive_failures']} consecutive failures"
    )

    while True:
        try:
            # Check if in cooldown
            if in_cooldown(state):
                time.sleep(CHECK_INTERVAL)
                continue

            # Check if bot is running
            if is_bot_running():
                time.sleep(CHECK_INTERVAL)
                continue

            log("Bot not running - attempting recovery")

            # Pre-flight import test
            import_ok, import_error = test_bot_import()
            if not import_ok:
                log(f"Import test failed: {import_error[:200]}")
                record_failure(state, import_error)

                if state["consecutive_failures"] >= MAX_FAILURES:
                    log(f"Max failures reached ({MAX_FAILURES}) - sending alert")
                    send_discord_alert(import_error)
                    enter_cooldown(state)

                time.sleep(CHECK_INTERVAL)
                continue

            # Start the bot
            success, start_error = start_bot()

            if success:
                log("Bot recovery successful")
                reset_failures(state)
            else:
                log(f"Start failed: {start_error}")
                record_failure(state, start_error)

                if state["consecutive_failures"] >= MAX_FAILURES:
                    log(f"Max failures reached ({MAX_FAILURES}) - sending alert")
                    send_discord_alert(start_error)
                    enter_cooldown(state)

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Service stopped by user")
            break
        except Exception as e:
            log(f"Unexpected error in main loop: {e}")
            time.sleep(CHECK_INTERVAL)


def main():
    """Entry point with command-line options."""
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == "status":
            state = load_state()
            running = is_bot_running()
            print(f"Bot running: {running}")
            print(f"Total restarts: {state['total_restarts']}")
            print(f"Consecutive failures: {state['consecutive_failures']}")
            if state.get("cooldown_until"):
                print(f"In cooldown until: {state['cooldown_until']}")
            if state.get("last_error"):
                print(f"Last error: {state['last_error'][:200]}")

        elif cmd == "reset":
            state = load_state()
            state["consecutive_failures"] = 0
            state["cooldown_until"] = None
            state["last_error"] = None
            save_state(state)
            print("State reset - failures cleared, cooldown cancelled")

        elif cmd == "start":
            # One-shot start (for manual use)
            kill_bot()
            success, error = start_bot()
            if not success:
                print(f"Failed: {error}")
                sys.exit(1)

        elif cmd == "stop":
            kill_bot()
            print("Bot stopped")

        elif cmd == "test-alert":
            send_discord_alert("This is a test alert from bot_service.py")

        else:
            print("Usage: python bot_service.py [status|reset|start|stop|test-alert]")
            print("  (no args) - Run the monitoring service")
            print("  status    - Show current state")
            print("  reset     - Clear failures and cooldown")
            print("  start     - One-shot start (kills existing)")
            print("  stop      - Stop the bot")
            print("  test-alert- Test Discord webhook")
    else:
        main_loop()


if __name__ == "__main__":
    main()
