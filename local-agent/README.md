# Technomancer

![tests](https://img.shields.io/badge/tests-5169-brightgreen) ![coverage](https://img.shields.io/badge/coverage-78.7%25-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![modules](https://img.shields.io/badge/modules-99-blue) ![lines](https://img.shields.io/badge/lines-41k-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Technomancer is a **fully autonomous AI dev team** that ships production code
the way humans do — from ranked Jira tickets, through feature branches, with
tests, passing CI, into `main`, and onto the live site. No human picks up the
story. No human writes the code. No human runs the merge.

**At full capacity: 425 commits to `main` in 24 hours** across two projects —
every commit branched, tested, merged, pushed, and README-regen'd through the
same safe-update workflow a human engineer would use.

Four specialized AI roles coordinate the pipeline like a real scrum team:

- **AIM** (AI Manager) — watches the Jira board, picks the next ranked story,
  dispatches a worker against it, and enforces the one-in-progress mutex so
  two workers never fight for `main`.
- **AIW** (AI Worker) — takes one ranked story, opens a feature branch,
  invokes Claude Code, runs the full pytest suite, auto-commits, merges when
  green, pushes, regenerates README, publishes to the public mirror.
- **AIMM** (AI Manager Manager) — the researcher above the managers. Observes
  what ships, scores stories for finding-worthiness, logs narrative-ready
  observations to `raw_findings.md`, and proposes research hypotheses to
  investigate. Suggests approvals but doesn't mutate Jira — the human still
  holds the approve button.
- **AIV** (AI Validator) — the DEMO step. After every merge, AIV opens the
  page, hits the API, or queries the DB and scores the shipped story on
  **seven axes** (meets_requirements, code_quality, test_quality,
  security_safety, scope_discipline, edge_cases, product_impact) plus red
  flags, against the original acceptance criteria. Senior-engineer-proxy
  code review at ship time. Self-audit beats self-repair.

The human's job shrinks to *priority*. Drag a story to the top of the board
and the team picks it up. Everything else is automatic.

> **Auto-generated** — This README is updated automatically on every deployment
> via `safe_update.py`.

## AI Dev Team — the details

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

### AIW — the AI Worker

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

- **5169 automated tests** with 78.7% code coverage
- **99 Python modules** across 41,139 lines of code
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
pytest                        # Run all 5169 tests
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
safe_update.py continue    →  pytest (all 5169 tests) → mypy → merge → restart bot → quality tests → publish
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
| Test count | 5169 |
| Code coverage | 78.7% |
| Python modules | 99 |
| Lines of code | 41,139 |
| Test files | 204 |

## License

MIT
