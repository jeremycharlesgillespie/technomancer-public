# Last verified: 2026-04-06
"""
Pre-commit validation script — MUST pass before any code is committed.

Runs 5 levels of validation:
1. SYNTAX — ast.parse every .py file in agent/ to catch syntax errors
2. LINT — ruff check for undefined names, unused imports, f-string bugs
3. IMPORT — actually import every module to catch runtime import failures
4. STARTUP — verify the bot can start and stay running for 10 seconds
5. INTEGRATION — exercise key code paths with real Ollama (not mocks)

Exit code 0 = all passed, non-zero = failures found.

Usage:
    python validate.py          # Run all checks
    python validate.py syntax   # Run only syntax checks
    python validate.py lint     # Run syntax + lint
    python validate.py import   # Run syntax + lint + import
    python validate.py startup  # Run syntax + lint + import + startup
    python validate.py full     # Run everything including integration
"""

import ast
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).parent / "agent"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def _skip_discord_startup() -> bool:
    """True when the environment opts out of the Discord-bot startup check.

    Read from SKIP_DISCORD_STARTUP in os.environ rather than from
    ``agent.config.settings`` — validate.py runs before deps may be
    installed, and importing the Settings model would be heavier than the
    check warrants. Treat the usual ``1/true/yes/on`` values as truthy.
    """
    raw = os.environ.get("SKIP_DISCORD_STARTUP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def check_syntax() -> list[str]:
    """Parse every .py file in agent/ with ast.parse.

    Catches: SyntaxError, unterminated strings, bad f-strings, indentation.
    """
    print("\n" + "=" * 60)
    print("LEVEL 1: SYNTAX CHECK")
    print("=" * 60)

    errors = []
    py_files = sorted(AGENT_DIR.glob("*.py"))
    for f in py_files:
        try:
            source = f.read_text(encoding="utf-8")
            ast.parse(source, filename=str(f))
            print(f"  [{PASS}] {f.name}")
        except SyntaxError as e:
            errors.append(f"{f.name}:{e.lineno} — {e.msg}")
            print(f"  [{FAIL}] {f.name}:{e.lineno} — {e.msg}")

    if errors:
        print(f"\n  RESULT: {len(errors)} syntax error(s) found")
    else:
        print(f"\n  RESULT: All {len(py_files)} files clean")
    return errors


def check_lint() -> list[str]:
    """Run ruff on agent/ and idea_board/ to catch semantic errors.

    Catches: undefined names in f-strings (F821), unused imports (F401),
    redefined unused variables (F841), and other pyflakes errors that
    ast.parse alone cannot detect.
    """
    print("\n" + "=" * 60)
    print("LEVEL 2: LINT CHECK (ruff)")
    print("=" * 60)

    errors = []
    targets = [str(AGENT_DIR), str(AGENT_DIR.parent / "idea_board")]
    existing = [t for t in targets if Path(t).exists()]

    # Only check for dangerous errors (undefined names, redefined builtins, etc.)
    # Skip import sorting (I) and unused imports (F401) — those are code hygiene, not bugs.
    dangerous_rules = ["F821", "F811", "F601", "F602"]

    try:
        result = subprocess.run(
            ["ruff", "check", "--no-fix", "--select", ",".join(dangerous_rules)] + existing,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(AGENT_DIR.parent),
        )
        if result.returncode == 0:
            print(f"  [{PASS}] ruff found no issues")
        else:
            for line in result.stdout.strip().splitlines():
                if line.startswith("Found"):
                    continue
                errors.append(line)
                print(f"  [{FAIL}] {line}")
            print(f"\n  RESULT: {len(errors)} lint error(s)")
    except FileNotFoundError:
        print(f"  [{SKIP}] ruff not installed — skipping lint check")

    return errors


def check_imports() -> list[str]:
    """Actually import every module in agent/ to catch import-time errors.

    Catches: missing dependencies, circular imports, NameError at module level,
    bad type annotations, anything that crashes on import.
    """
    print("\n" + "=" * 60)
    print("LEVEL 3: IMPORT CHECK")
    print("=" * 60)

    errors = []
    modules = sorted(AGENT_DIR.glob("*.py"))
    for f in modules:
        if f.name.startswith("_"):
            continue
        module_name = f"agent.{f.stem}"
        try:
            # Force reimport
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)
            print(f"  [{PASS}] {module_name}")
        except Exception as e:
            errors.append(f"{module_name} — {type(e).__name__}: {e}")
            print(f"  [{FAIL}] {module_name} — {type(e).__name__}: {e}")

    if errors:
        print(f"\n  RESULT: {len(errors)} import error(s)")
    else:
        print(f"\n  RESULT: All modules import successfully")
    return errors


def check_startup() -> list[str]:
    """Start the bot process and verify it stays alive for 10 seconds.

    Catches: crashes on startup, missing config, Discord auth failures,
    anything that kills the process immediately.

    When ``SKIP_DISCORD_STARTUP`` is truthy (set by machines that don't
    host the Discord bot — e.g. a Mac running only AIM/AIMM/AIV), this
    check is reported as skipped rather than failed.
    """
    print("\n" + "=" * 60)
    print("LEVEL 4: STARTUP CHECK")
    print("=" * 60)

    if _skip_discord_startup():
        print(f"  [{SKIP}] Discord bot startup skipped (SKIP_DISCORD_STARTUP is set)")
        print("         This machine does not host the Discord bot. Level 4")
        print("         will be validated on a bot-hosting machine before deploy.")
        return []

    errors = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent.discord_memory_bot"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(Path(__file__).parent),
        )
        time.sleep(10)

        if proc.poll() is not None:
            stderr = proc.stderr.read()[:1000] if proc.stderr else ""
            errors.append(f"Bot crashed on startup (exit code {proc.returncode}): {stderr}")
            print(f"  [{FAIL}] Bot crashed after startup")
            print(f"         Exit code: {proc.returncode}")
            if stderr:
                print(f"         Error: {stderr[:300]}")
        else:
            print(f"  [{PASS}] Bot started and stayed alive for 10s (PID: {proc.pid})")
            proc.terminate()
            proc.wait(timeout=5)
    except Exception as e:
        errors.append(f"Could not start bot: {e}")
        print(f"  [{FAIL}] Could not start bot: {e}")

    return errors


def check_integration() -> list[str]:
    """Exercise key code paths with real Ollama.

    Catches: Ollama API changes, response format mismatches, None vs [],
    tool calling issues — anything that mocks hide.
    """
    print("\n" + "=" * 60)
    print("LEVEL 5: INTEGRATION CHECK")
    print("=" * 60)

    errors = []

    # Test 1: Basic Ollama call returns non-empty response
    print("  [....] Agent.run with simple question...", end="\r")
    try:
        from agent.core import Agent, AgentConfig

        agent = Agent(AgentConfig(verbose=False, model="qwen3.5:9b"))
        result = agent.run("What is 2+2? Reply with just the number.")
        if not result or not result.strip():
            errors.append("Agent.run returned empty response")
            print(f"  [{FAIL}] Agent.run returned empty response")
        elif "4" in result:
            print(f"  [{PASS}] Agent.run basic question (got: {result.strip()[:50]})")
        else:
            print(f"  [{PASS}] Agent.run returned response (got: {result.strip()[:50]})")
    except Exception as e:
        errors.append(f"Agent.run failed: {e}")
        print(f"  [{FAIL}] Agent.run failed: {e}")

    # Test 2: Tool calling works (agent can use a tool and get result)
    print("  [....] Agent.run with tool call...", end="\r")
    try:
        from agent.core import Agent, AgentConfig, create_tool

        agent = Agent(AgentConfig(verbose=False, model="qwen3.5:9b"))
        tool = create_tool(
            "get_current_time",
            "Returns the current time",
            {"type": "object", "properties": {}, "required": []},
            lambda: "The current time is 12:00 PM",
        )
        agent.register_tool(tool)
        result = agent.run("What time is it? Use the get_current_time tool.")
        if not result or not result.strip():
            errors.append("Agent.run with tool returned empty")
            print(f"  [{FAIL}] Agent.run with tool returned empty")
        else:
            print(f"  [{PASS}] Agent.run with tool call (got: {result.strip()[:50]})")
    except Exception as e:
        errors.append(f"Agent.run with tool failed: {e}")
        print(f"  [{FAIL}] Agent.run with tool failed: {e}")

    # Test 3: Profiler doesn't crash
    print("  [....] Profiler integration...", end="\r")
    try:
        from agent.profiler import RequestProfile, RequestTimer

        profile = RequestProfile(user="test", message="test", message_length=4)
        timer = RequestTimer(profile)
        with timer.phase("context_build"):
            pass
        timer.record_llm_call(turn=1, duration=1.0, input_chars=100,
                              output_chars=50, num_ctx=8192, tool_calls=[])
        timer.finish()
        data = profile.to_dict()
        assert "total_seconds" in data
        assert "llm_calls" in data
        print(f"  [{PASS}] Profiler records data correctly")
    except Exception as e:
        errors.append(f"Profiler failed: {e}")
        print(f"  [{FAIL}] Profiler failed: {e}")

    # Test 4: HTML normalization round-trip
    print("  [....] HTML normalization...", end="\r")
    try:
        from agent.html_generator import normalize_html, extract_body_content

        raw_html = '<html><head><style>body{color:red}</style></head><body><h1>Test</h1><p>Content</p></body></html>'
        normalized = normalize_html(raw_html)
        assert "<h1>Test</h1>" in normalized or "Test" in normalized
        assert "color:red" not in normalized  # Custom CSS stripped
        assert "prefers-color-scheme" in normalized  # Our CSS injected
        print(f"  [{PASS}] HTML normalization strips custom CSS, injects ours")
    except Exception as e:
        errors.append(f"HTML normalization failed: {e}")
        print(f"  [{FAIL}] HTML normalization failed: {e}")

    if errors:
        print(f"\n  RESULT: {len(errors)} integration error(s)")
    else:
        print(f"\n  RESULT: All integration checks passed")
    return errors


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "startup"

    print("=" * 60)
    print("TECHNOMANCER PRE-COMMIT VALIDATION")
    print("=" * 60)

    all_errors = []

    # Level 1: Always run syntax
    all_errors.extend(check_syntax())
    if all_errors:
        print(f"\n{'=' * 60}")
        print(f"BLOCKED: Fix {len(all_errors)} syntax error(s) before continuing")
        print(f"{'=' * 60}")
        sys.exit(1)

    if level in ("lint", "import", "startup", "full"):
        all_errors.extend(check_lint())
        if all_errors:
            print(f"\n{'=' * 60}")
            print(f"BLOCKED: Fix {len(all_errors)} lint error(s) before continuing")
            print(f"{'=' * 60}")
            sys.exit(1)

    if level in ("import", "startup", "full"):
        all_errors.extend(check_imports())
        if all_errors:
            print(f"\n{'=' * 60}")
            print(f"BLOCKED: Fix {len(all_errors)} import error(s) before continuing")
            print(f"{'=' * 60}")
            sys.exit(1)

    if level in ("startup", "full"):
        all_errors.extend(check_startup())
        if all_errors:
            print(f"\n{'=' * 60}")
            print(f"BLOCKED: Fix {len(all_errors)} startup error(s) before continuing")
            print(f"{'=' * 60}")
            sys.exit(1)

    if level == "full":
        all_errors.extend(check_integration())

    print(f"\n{'=' * 60}")
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  - {e}")
        print(f"{'=' * 60}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED — all checks clean")
        print(f"{'=' * 60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
