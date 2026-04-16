# Technomancer

![tests](https://img.shields.io/badge/tests-2951-brightgreen) ![coverage](https://img.shields.io/badge/coverage-78.7%25-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![modules](https://img.shields.io/badge/modules-77-blue) ![lines](https://img.shields.io/badge/lines-32k-blue) ![license](https://img.shields.io/badge/license-MIT-green)

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

- **2951 automated tests** with 78.7% code coverage
- **77 Python modules** across 32,186 lines of code
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
pytest                        # Run all 2951 tests
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
safe_update.py continue    →  pytest (all 2951 tests) → mypy → merge → restart bot → quality tests → publish
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
| Test count | 2951 |
| Code coverage | 78.7% |
| Python modules | 77 |
| Lines of code | 32,186 |
| Test files | 106 |

## License

MIT
