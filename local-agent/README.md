# Technomancer

![tests](https://img.shields.io/badge/tests-1464-brightgreen) ![coverage](https://img.shields.io/badge/coverage-77.9%25-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![modules](https://img.shields.io/badge/modules-66-blue) ![lines](https://img.shields.io/badge/lines-26k-blue) ![license](https://img.shields.io/badge/license-MIT-green)

An Ollama-powered autonomous agent framework with Discord bot interface,
Obsidian vault integration, Claude API escalation, and a self-improving
knowledge base. Built for a Senior Software Engineer's daily workflow.

> **Auto-generated** — This README is updated automatically on every deployment
> via `safe_update.py`. Last updated: 2026-04-12 06:00

## Highlights

- **1464 automated tests** with 77.9% code coverage
- **66 Python modules** across 26,568 lines of code
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
| Core | `core`, `config`, `tools` |
| Discord Bot | `discord_memory_bot`, `bot_commands`, `bot_utils`, `discord_bridge`, `discord_rate_limit`, `discord_errors`, `message_validators`, `fallback_responses`, `command_suggestions` |
| LLM Integration | `claude_bridge`, `claude_vault`, `ask_claude`, `fallback_orchestrator`, `reflection` |
| Memory & Knowledge | `memory_system`, `conversation_context`, `auto_memory`, `facts_db`, `knowledge_gaps`, `knowledge_fallback`, `knowledge_enrichment`, `skill_gap_analysis` |
| News & Learning | `news_digest`, `news_engagement`, `dev_learning`, `learning_newsletter` |
| Monitoring & Performance | `perf_monitor`, `metrics_db`, `prometheus_metrics`, `profiler`, `api_usage_anomaly`, `llm_optimizer`, `infra_monitor` |
| Content & Publishing | `github_pages`, `html_generator`, `pdf_tools`, `enhancements`, `ref_enrichment` |
| Idea Board | `idea_board/web.py`, `idea_board/models.py`, `idea_board/executor.py` |

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
VAULT_PATH=C:\Users\you\Documents\ObsidianVault
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
pytest                        # Run all 1464 tests
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
safe_update.py continue    →  pytest (all 1464 tests) → mypy → merge → restart bot → quality tests → publish
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
| Test count | 1464 |
| Code coverage | 77.9% |
| Python modules | 66 |
| Lines of code | 26,568 |
| Test files | 64 |

## License

MIT
