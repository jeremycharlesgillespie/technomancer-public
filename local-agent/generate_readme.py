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
from datetime import datetime
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


def collect_test_stats() -> dict:
    """Run pytest with coverage and collect statistics."""
    result = _run([
        sys.executable, "-m", "pytest",
        "--co", "-q",  # collect-only, quiet
    ])
    test_count = 0
    for line in result.stdout.splitlines():
        if "test" in line and "collected" in line:
            import re
            m = re.search(r"(\d+) tests? collected", line)
            if m:
                test_count = int(m.group(1))

    # Run coverage (quick, just the summary)
    cov_result = _run([
        sys.executable, "-m", "pytest",
        "--cov=agent", "--cov-report=json:" + str(COVERAGE_JSON),
        "--cov-report=term-missing",
        "-q", "--tb=no",
    ])

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
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tests = test_stats["test_count"]
    coverage = test_stats["coverage_pct"]
    modules = code_stats["module_count"]
    lines = code_stats["total_lines"]
    test_files = code_stats["test_file_count"]

    # Color for coverage badge
    if coverage >= 80:
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

An Ollama-powered autonomous agent framework with Discord bot interface,
Obsidian vault integration, Claude API escalation, and a self-improving
knowledge base. Built for a Senior Software Engineer's daily workflow.

> **Auto-generated** — This README is updated automatically on every deployment
> via `safe_update.py`. Last updated: {now}

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


def main():
    print("[README] Collecting test statistics...")
    test_stats = collect_test_stats()
    print(f"  Tests: {test_stats['test_count']}, Coverage: {test_stats['coverage_pct']}%")

    print("[README] Collecting code statistics...")
    code_stats = collect_code_stats()
    print(f"  Modules: {code_stats['module_count']}, Lines: {code_stats['total_lines']}")

    print("[README] Generating README.md...")
    readme = generate_readme(test_stats, code_stats)
    README_PATH.write_text(readme, encoding="utf-8")
    print(f"[README] Written to {README_PATH}")

    # Also write to repo root so GitHub shows badges on the main page
    if ROOT_README_PATH.parent.exists():
        ROOT_README_PATH.write_text(readme, encoding="utf-8")
        print(f"[README] Also written to {ROOT_README_PATH}")


if __name__ == "__main__":
    main()
