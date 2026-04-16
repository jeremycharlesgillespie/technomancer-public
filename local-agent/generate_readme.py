#!/usr/bin/env python3
"""
Auto-generate README.md with live project statistics.

Collects test count, code coverage, module inventory, line counts,
and architecture info to produce a professional README that stays
current with every deployment.

Called automatically by safe_update.py after successful merge.
Can also be run standalone: python generate_readme.py
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AGENT_DIR = SCRIPT_DIR / "agent"
TEST_DIR = SCRIPT_DIR / "tests"
README_PATH = SCRIPT_DIR / "README.md"
ROOT_README_PATH = SCRIPT_DIR.parent / "README.md"  # repo root
COVERAGE_JSON = SCRIPT_DIR / "profiling" / "coverage.json"


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )


def collect_test_stats(known_test_count: int | None = None) -> dict:
    """Collect test statistics.

    Args:
        known_test_count: If provided, skip pytest entirely and use this
            count. The executor and safe_update already ran the full suite
            — no need to run it again.
    """
    if known_test_count is not None and known_test_count > 0:
        # Use cached coverage if available, otherwise report 0
        coverage_pct = 0
        if COVERAGE_JSON.exists():
            try:
                data = json.loads(COVERAGE_JSON.read_text())
                coverage_pct = round(data.get("totals", {}).get("percent_covered", 0), 1)
            except Exception:
                pass
        return {"test_count": known_test_count, "coverage_pct": coverage_pct}

    # Fallback: run pytest --co to count tests (fast, ~2s)
    import re
    result = _run([
        sys.executable, "-m", "pytest",
        "--co", "-q",
    ])
    test_count = 0
    for line in result.stdout.splitlines():
        if "test" in line and "collected" in line:
            m = re.search(r"(\d+) tests? collected", line)
            if m:
                test_count = int(m.group(1))

    # Use cached coverage if available — don't rerun the full suite
    coverage_pct = 0
    if COVERAGE_JSON.exists():
        try:
            data = json.loads(COVERAGE_JSON.read_text())
            coverage_pct = round(data.get("totals", {}).get("percent_covered", 0), 1)
        except Exception:
            pass

    return {"test_count": test_count, "coverage_pct": coverage_pct}


def collect_code_stats() -> dict:
    """Count modules, lines, and categorize the codebase."""
    modules = sorted(AGENT_DIR.glob("*.py"))
    module_names = [m.stem for m in modules if m.stem != "__init__"]

    total_lines = 0
    for m in modules:
        try:
            total_lines += len(m.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass

    test_files = sorted(TEST_DIR.rglob("test_*.py"))

    # Categorize modules
    categories = {
        "Core": ["core", "config", "tools"],
        "Discord Bot": ["discord_memory_bot", "bot_commands", "bot_utils",
                        "discord_bridge", "discord_rate_limit", "discord_errors",
                        "message_validators", "fallback_responses", "command_suggestions"],
        "LLM Integration": ["claude_bridge", "claude_vault", "ask_claude",
                            "fallback_orchestrator", "reflection"],
        "Memory & Knowledge": ["memory_system", "conversation_context", "auto_memory",
                               "facts_db", "knowledge_gaps", "knowledge_fallback",
                               "knowledge_enrichment", "skill_gap_analysis"],
        "News & Learning": ["news_digest", "news_engagement", "dev_learning",
                            "learning_newsletter"],
        "Monitoring & Performance": ["perf_monitor", "metrics_db", "prometheus_metrics",
                                     "profiler", "api_usage_anomaly", "llm_optimizer",
                                     "infra_monitor"],
        "Content & Publishing": ["github_pages", "html_generator", "pdf_tools",
                                 "enhancements", "ref_enrichment"],
        "Idea Board": [],  # separate directory
    }

    return {
        "module_count": len(module_names),
        "total_lines": total_lines,
        "test_file_count": len(test_files),
        "categories": categories,
    }


def generate_badge(label: str, value: str, color: str) -> str:
    """Generate a shields.io-style badge in markdown."""
    label_enc = label.replace(" ", "%20").replace("-", "--")
    value_enc = value.replace(" ", "%20").replace("-", "--")
    return f"![{label}](https://img.shields.io/badge/{label_enc}-{value_enc}-{color})"


def generate_readme(test_stats: dict, code_stats: dict) -> str:
    """Generate the full README.md content."""
    tests = test_stats["test_count"]
    coverage = test_stats["coverage_pct"]
    modules = code_stats["module_count"]
    lines = code_stats["total_lines"]
    test_files = code_stats["test_file_count"]

    # Color for coverage badge (green at 75% — matches fail_under threshold)
    if coverage >= 75:
        cov_color = "brightgreen"
    elif coverage >= 60:
        cov_color = "yellow"
    else:
        cov_color = "red"

    badges = " ".join([
        generate_badge("tests", str(tests), "brightgreen"),
        generate_badge("coverage", f"{coverage}%25", cov_color),
        generate_badge("python", "3.10%2B", "blue"),
        generate_badge("modules", str(modules), "blue"),
        generate_badge("lines", f"{lines // 1000}k", "blue"),
        generate_badge("license", "MIT", "green"),
    ])

    # Module table
    cat_rows = ""
    for cat_name, cat_modules in code_stats["categories"].items():
        if cat_modules:
            mod_list = ", ".join(f"`{m}`" for m in cat_modules if (AGENT_DIR / f"{m}.py").exists())
            if mod_list:
                cat_rows += f"| {cat_name} | {mod_list} |\n"

    return f"""# Technomancer

{badges}

Technomancer is an **autonomous AI dev team** wrapped around an Ollama-powered
Discord bot. A human drags a story to the top of a Jira board; a daemon picks
it up, spins up Claude Code against it, streams the log, tests it, and merges
it. The Discord bot, Obsidian vault, and Claude API escalation are all still
here — but the headline feature is the engineering loop that turns a ranked
Jira story into a deployed commit without a human in the middle.

> **Auto-generated** — This README is updated automatically on every deployment
> via `safe_update.py`.

## AI Dev Team

Three moving parts turn a Jira ticket into a merged pull request:

- **AIM** (AI Manager) — a long-running daemon that watches the board, plans
  work, and decides when to dispatch the next story.
- **Worker** — a short-lived process AIM spawns per story. It prepares the
  git state, invokes Claude Code, and reports back.
- **Claude Code** — the hands. Given a ranked story's prompt, it edits files,
  runs tests, and commits.

The human's job shrinks to *ranking* the board. Drag a story to the top and
AIM treats that as "do this next."

### AIM — the Manager daemon

- Picks the next story from the Jira board using **rank** (drag-to-top ==
  next story picked, rank-aware since TK-384/TK-385).
- Spawns the Worker and monitors it for liveness and timeouts.
- Escalates to Discord when it gets stuck — surfaces the failing story, the
  error, and links to the live log.
- **Auto-approves safe-category stories** (`cat:quality`, doc-only changes)
  so low-risk work doesn't sit waiting on a human.
- Enforces a **one-in-progress-at-a-time mutex**: refuses ASSIGN while Jira
  already has an item In Progress, so two Workers can never fight over the
  same main branch.
- **Recovers orphaned In Progress items on startup** — if the daemon died
  mid-story, boot-time reconciliation either resumes or releases them.

### Worker — the executor

- Ensures a **clean main branch** before it starts (no dangling edits, no
  stale feature branch).
- Spawns Claude Code against the ranked story's prompt and streams every
  token of stdout.
- Detects success via **Jira state** (through the BoardProvider abstraction)
  rather than parsing log output — the ticket moving to Done is the ground
  truth.
- Emits structured lifecycle events to `aim/events.jsonl` so the dashboard
  and `/api/aim/status` snapshot always know what's happening.

### Jira integration — the single source of truth

Jira, when configured, is where work lives. The **BoardProvider** abstraction
(in `board/`) has two implementations:

- **`JiraProvider`** — reads and writes Jira directly. Used whenever
  `JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, and `JIRA_PROJECT_KEY` are set.
- **`LocalProvider`** — JSON fallback so the system runs fully standalone
  when Jira isn't configured. No features are lost — sync just doesn't fire.

Metadata travels as **labels**: category becomes `cat:quality`, source becomes
`src:llm_analysis`, and so on. That keeps the shape portable between the two
providers and survives round-trips through the Jira UI. JQL pagination is
handled for large boards, so ranking scales past the first page.

### Live log view — `/live/<item_id>`

While a Worker is running, anyone with the link can watch in real time:

- A dark-mode, **auto-scrolling** HTML page streams executor stdout over
  Server-Sent Events.
- On the Jira side, the same run writes a **single `[AIM Progress]` comment
  that's edited in place every 60s**. One comment per execution — not thirty —
  so the issue page stays readable but still shows live progress.
- Lifecycle transitions (spawned / succeeded / failed) land in
  `aim/events.jsonl` and surface on the `/aim` dashboard.

## Recent highlights

- **Jira-first refactor** with the `BoardProvider` abstraction — same code
  runs against Jira or the local JSON store depending on config.
- **`/aim` dashboard + `/api/aim/status` snapshot + `aim/events.jsonl`** —
  unified view of what the daemon is doing right now and what it's done
  recently.
- **Epic-vs-story prompt discipline** — epics get a planning prompt, stories
  get an implementation prompt; no more Claude Code trying to implement a
  whole epic in one pass.
- **Failure memory** — when a retry runs, the prior `[Execution Log - Failed]`
  comment is injected into the new prompt so Claude Code sees what went wrong
  last time instead of repeating the same mistake.
- **README idempotency** — this generator compares before writing and skips
  the write when content is identical, so consecutive `generate_readme.py`
  runs don't dirty the tree or produce no-op commits.
- **ASSIGN mutex + orphan recovery** — the one-in-progress invariant is
  enforced at dispatch time, and startup reconciles anything the last run
  left hanging.
- **JQL pagination fix** — ranking works correctly on large boards, not just
  the first page of results.

## Highlights

- **{tests} automated tests** with {coverage}% code coverage
- **{modules} Python modules** across {lines:,} lines of code
- **Self-improving knowledge base** — auto-fills gaps from Wikipedia and web search
- **Epic/Story/Task hierarchy** on the idea board with full lifecycle tracking
- **Local-first** — Ollama for primary inference, Claude API for escalation only
- **Discord-native** — all interaction through Discord with reaction tracking

## Architecture

```
Discord Bot (discord_memory_bot.py)
    |
    +-- Ollama LLM (local, tool-calling loop)
    |       |-- 50+ registered tools
    |       |-- Factual auto-search before answering
    |       |-- Knowledge gap detection + auto-enrichment
    |
    +-- Claude API (escalation for complex tasks)
    |       |-- Vault context with prompt caching (84% token savings)
    |       |-- Fallback orchestrator (auto-switches to Ollama on failure)
    |
    +-- Obsidian Vault (persistent memory)
    |       |-- Conversation context + summaries
    |       |-- Knowledge gap notes
    |       |-- Reference articles
    |       |-- Write-ahead logging for data safety
    |
    +-- Idea Board (Flask, port 8322)
    |       |-- Epic/Story/Task hierarchy
    |       |-- LLM-powered discussion threads
    |       |-- Copy Epic for Claude Code (sequential implementation)
    |
    +-- Background Tasks
            |-- News digest (hourly, 9am-9pm)
            |-- Developer learning articles (daily, 8am)
            |-- Knowledge enrichment (every 6 hours)
            |-- Gap frequency analysis (weekly)
            |-- Infrastructure monitoring (every 30 min)
            |-- Idea generation (hourly)
            |-- Learning newsletter (weekly, Sunday 9am)
```

## Modules

| Category | Modules |
|----------|---------|
{cat_rows}| Idea Board | `idea_board/web.py`, `idea_board/models.py`, `idea_board/executor.py` |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with `qwen3.5:27b` (or configure via `.env`)
- Discord bot token
- (Optional) Anthropic API key for Claude escalation

### Installation

```bash
cd local-agent
pip install -e ".[all]"    # All dependencies
pip install -e ".[dev]"    # Dev tools (pytest, coverage, linting)
```

### Configuration

Copy `.env.example` to `.env` and fill in:
```
DISCORD_BOT_TOKEN=your_token
ANTHROPIC_API_KEY=sk-ant-...
VAULT_PATH=C:\\Users\\you\\Documents\\ObsidianVault
OLLAMA_MODEL=qwen3.5:27b
```

### Running

```bash
# Start Discord bot
python -m agent.discord_memory_bot

# Or via bot service (with crash recovery)
python bot_service.py start

# Safe deployment workflow
python safe_update.py my-feature       # create branch
# ... make changes ...
python safe_update.py continue          # test, merge, restart, publish
```

### Testing & Quality Gates

Every code change passes through multiple quality gates before deployment:

```bash
pytest                        # Run all {tests} tests
pytest --cov=agent            # With coverage report
pytest tests/unit/            # Unit tests only
python validate.py startup    # Full pre-commit validation (5 levels)
```

#### validate.py — 5-Level Validation Pipeline

| Level | Check | What It Catches |
|-------|-------|-----------------|
| 1 | **Syntax** (`ast.parse`) | SyntaxError, unterminated strings, bad indentation |
| 2 | **Lint** (`ruff check`) | Undefined names in f-strings (F821), redefined variables (F811) |
| 3 | **Import** (actual module import) | Missing dependencies, circular imports, NameError at module level |
| 4 | **Startup** (10-second live test) | Config errors, Discord auth failures, runtime crashes |
| 5 | **Integration** (real Ollama calls) | API changes, response format mismatches, tool calling issues |

`validate.py startup` (levels 1-4) is **mandatory before every commit**. Level 5 runs via `validate.py full`.

#### safe_update.py — Deployment Pipeline

Every change follows the same branch → test → merge → restart flow:

```
safe_update.py <name>      →  Create isolated branch
validate.py startup        →  5-level validation (mandatory before commit)
git commit                 →  Pre-commit hooks (black, ruff, trailing whitespace)
safe_update.py continue    →  pytest (all {tests} tests) → mypy → merge → restart bot → quality tests → publish
```

No code reaches `main` without passing **all** of: pre-commit hooks, 5-level validation, the full test suite, type checking, and post-deploy quality tests.

## Discord Commands

| Command | Description |
|---------|-------------|
| `betterDev [topic]` | Generate a learning article |
| `techNews` | Latest tech news with analysis |
| `think` | Show what the bot knows about you |
| `suggest` | Context-aware command suggestions |
| `newsletter` | Weekly learning digest |
| `perf` | Session profiling data |
| `metrics` | Persistent LLM latency trends |
| `karen <complaint>` | Submit feedback (generates improvement ideas) |
| `commands` | Full command list |

## Idea Board

Access at `http://localhost:8322` — a web dashboard for managing improvement ideas:

- **Epics** group related stories into full value chains
- **Copy Epic for Claude Code** implements all stories sequentially
- **LLM discussion threads** on each idea
- **Archived view** hides completed work while keeping it for deduplication

## Key Design Decisions

- **safe_update.py** — Every code change goes through branch → test → merge → restart.
  No exceptions, even for "small" fixes.
- **validate.py** — Five-level validation (syntax → lint → import → startup → integration)
  catches what unit tests miss — including undefined names in f-strings, missing imports,
  and runtime crashes.
- **Pre-commit hooks** — black (formatting), ruff (linting), trailing whitespace cleanup,
  and private key detection run automatically on every commit.
- **Post-deploy quality tests** — After every deployment, the bot answers 3 live questions
  scored by an LLM judge. Failures are flagged immediately.
- **Crash notification cooldown** — If the bot hits a repeating error, only the first crash
  report is sent to Discord. Subsequent crashes are suppressed for 5 minutes to prevent
  channel spam, with a count of suppressed errors included in the next notification.
- **Local-first fallback** — If Claude API is down, the bot switches to Ollama
  automatically and recovers when Claude comes back.
- **Write-ahead logging** — Vault writes go through SQLite WAL first, so failed
  writes can be recovered.

## Project Stats

| Metric | Value |
|--------|-------|
| Test count | {tests} |
| Code coverage | {coverage}% |
| Python modules | {modules} |
| Lines of code | {lines:,} |
| Test files | {test_files} |

## License

MIT
"""


def write_if_changed(path: Path, content: str) -> bool:
    """Write content to path only if it differs from what's already there.

    Returns True when the file was written, False when the existing content
    matched and the write was skipped. Skipping keeps the working tree clean
    so safe_update.py doesn't produce a no-op README commit on every deploy.
    """
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except Exception:
            existing = None
        if existing == content:
            print(f"[README] {path} unchanged - skipping write")
            return False

    path.write_text(content, encoding="utf-8")
    print(f"[README] Written to {path}")
    return True


def main():
    # Accept --test-count N to skip running pytest
    known_count = None
    if "--test-count" in sys.argv:
        idx = sys.argv.index("--test-count")
        if idx + 1 < len(sys.argv):
            try:
                known_count = int(sys.argv[idx + 1])
            except ValueError:
                pass

    print("[README] Collecting test statistics...")
    test_stats = collect_test_stats(known_test_count=known_count)
    if known_count:
        print(f"  Tests: {test_stats['test_count']} (from caller), Coverage: {test_stats['coverage_pct']}%")
    else:
        print(f"  Tests: {test_stats['test_count']}, Coverage: {test_stats['coverage_pct']}%")

    print("[README] Collecting code statistics...")
    code_stats = collect_code_stats()
    print(f"  Modules: {code_stats['module_count']}, Lines: {code_stats['total_lines']}")

    print("[README] Generating README.md...")
    readme = generate_readme(test_stats, code_stats)
    write_if_changed(README_PATH, readme)

    # Also write to repo root so GitHub shows badges on the main page
    if ROOT_README_PATH.parent.exists():
        write_if_changed(ROOT_README_PATH, readme)


if __name__ == "__main__":
    main()
