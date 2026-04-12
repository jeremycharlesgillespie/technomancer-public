# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Technomancer** is an Ollama-powered autonomous agent framework with:
- Discord bot for chat interface
- Obsidian vault integration for persistent memory
- Claude API escalation for complex tasks
- Web search, image identification, news digest
- **Test suite with 145 tests** for safe deployments
- **safe_update.py workflow** for branch-test-merge automation

All code lives in `local-agent/`.

## Owner Context

The bot owner's Discord username is configured via `BOT_OWNER` in `.env`.
Owner-only commands (reloadServer, evolve, #claude-code) check this setting.

**Default stack context**: Python, Django, PostgreSQL, AWS
(Customize the system prompt in discord_memory_bot.py for your own context)

## Current Status (March 2026)

### What's Working
- Discord bot fully operational with conversation memory
- Two-stage image identification (local llava → Claude API)
- Web search and news digest (hourly 9am-9pm)
- PDF/DOCX/TXT file processing and summarization
- Enhancement queue system reading/writing to Obsidian vault
- Accountability verification tools
- Claude escalation for complex tasks
- Crash logging with full stack traces to vault
- Memory system with hourly compaction
- **Test suite (145 tests) covering core, tools, memory, enhancements, accountability**
- **safe_update.py workflow for tested deployments**

### Recent Completed Enhancements
- **#7:** DevOps improvements - CI/CD, pre-commit, Makefile, Pydantic config, centralized logging
- **#6:** Test suite + safe_update workflow - 145 tests, automated branch-test-merge
- **#5:** Crash logging - full stack traces + local variables to crash_log.md
- **#4:** Numbered enhancements - auto-numbered with #N format
- **#3:** Vault-only enhancement reading - bot always reads fresh from vault
- **#2:** Accountability checks - verification tools for file/memory operations
- **#1:** Claude document passing - save_context_for_claude tool
- PDF summarization capability

### Pending Enhancements
None currently - queue is empty

## Build & Run Commands

### Local Agent (Discord Bot)
```bash
cd local-agent

# Install (editable)
pip install -e ".[all]"

# Run Discord bot
python -m agent.discord_memory_bot

# Restart bot (stops old process, unloads Ollama models, starts fresh)
python bot_service.py start

# Stop only
python bot_service.py stop
```

### Tests
```bash
cd local-agent
pytest              # Run all tests (145 tests)
pytest -v           # Verbose output
pytest -x           # Stop on first failure
pytest tests/unit/  # Unit tests only
pytest tests/integration/  # Integration tests only
mypy agent/         # Type check the codebase
```

### Safe Update Workflow
```bash
cd local-agent

# Start: creates timestamped branch (e.g., 2026-03-15-143022-fix-bug)
python safe_update.py <short-name>

# After making changes and committing:
python safe_update.py continue    # Runs tests → merges if pass → restarts bot

# Other commands:
python safe_update.py abort       # Cancel and return to main
python safe_update.py status      # Show current state
```

### Makefile Commands
```bash
cd local-agent
make install    # Install with all dependencies + pre-commit hooks
make test       # Run all tests
make test-cov   # Run tests with coverage report
make lint       # Check code style (ruff + black)
make format     # Auto-format code
make start      # Start Discord bot
make stop       # Stop Discord bot
make update name=fix-bug  # Start safe_update workflow
```

### Pre-commit Hooks
```bash
pre-commit install     # Install hooks (done by make install)
pre-commit run --all   # Run all hooks manually
```
Hooks: black, ruff, trailing whitespace, private key detection
(mypy: Claude manages this - enables when 0 type errors, disables otherwise)

## Architecture

### Local Agent (`local-agent/agent/`)

```
core.py              - Agent class with Ollama tool-calling loop
discord_memory_bot.py - Main Discord bot entry point, message handling, image/file processing
memory_system.py     - Obsidian vault-backed conversation memory with compaction
tools.py             - File/system tools (read, write, shell commands)
web_search.py        - DuckDuckGo search + URL fetch tools
news_digest.py       - Hourly tech news with LLM commentary (9am-9pm)
image_identification.py - Two-stage vision: local llava → Claude API fallback
enhancements.py      - Enhancement queue stored in Obsidian vault
ask_claude.py        - Claude API tools (ask_claude, web_search_claude, image_search_claude)
claude_bridge.py     - Lower-level Claude escalation (escalate_to_claude, report_to_claude)
claude_vault.py      - Claude API with vault access + prompt caching (84% token savings)
accountability.py    - Verification tools (verify_file_exists, verify_file_modified, verify_content, verify_memory_saved)
pdf_tools.py         - PDF extraction and summarization tools
capability_request.py - Self-improvement capability requests
notifications.py     - Discord webhook notifications for bot lifecycle
config.py            - Pydantic settings (loads from .env, typed config)
logging_config.py    - Centralized stdlib logging with file rotation
```

### Test Suite (`local-agent/tests/`)

```
conftest.py              - Shared fixtures (mock Ollama, Anthropic, temp vault)
pytest.ini               - Test configuration

unit/
  test_core.py           - Agent, Tool, strip_thinking_tags (20 tests)
  test_tools.py          - Command filtering, file operations (42 tests) **SECURITY CRITICAL**
  test_memory_system.py  - MemorySystem class (17 tests)
  test_enhancements.py   - Enhancement queue (14 tests)
  test_accountability.py - Verification tools (14 tests)
  test_claude_vault.py   - Claude vault integration + caching (20 tests)

integration/
  test_agent_workflow.py - Full agent runs with mocked Ollama (9 tests)
```

**Key test fixtures** (in conftest.py):
- `mock_ollama_client` - Patches `agent.core._ollama_client` with configurable responses
- `mock_anthropic_client` - Patches `anthropic.Anthropic`
- `temp_vault` - Creates temporary Obsidian vault structure
- `memory_system` - MemorySystem initialized with temp vault
- `patched_enhancements` - Patches VAULT_PATH for enhancements.py
- `patched_accountability` - Patches VAULT_PATH for accountability.py

### Bot Service (`local-agent/bot_service.py`)

```
python bot_service.py start    # Kill old bot, unload Ollama models, start fresh
python bot_service.py stop     # Stop bot and unload models
python bot_service.py status   # Show current state
python bot_service.py reset    # Clear failure count/cooldown
```

Features:
- Pre-flight import test (catches syntax errors before starting)
- Process monitoring (checks every 30 seconds)
- Max 3 consecutive failures before Discord alert + 30-min cooldown
- State persistence to `service_state.json`

### Safe Update Script (`local-agent/safe_update.py`)

Automated branch-test-merge workflow for safe deployments:

1. `safe_update.py <name>` - Creates branch `YYYY-MM-DD-HHMMSS-<name>`, checks it out
2. You make changes and commit them
3. `safe_update.py continue` - Runs pytest, merges to main if passing, restarts bot
4. If tests fail - stays on branch, you fix and retry

State stored in `.safe_update_state` (git-ignored).

**Legacy/Unused Files** (exist but not actively used):
- `discord_bot.py` - Older simple bot, replaced by discord_memory_bot.py
- `knowledge.py` - SQLite knowledge graph with embeddings (not integrated)
- `obsidian.py` - ObsidianVault utility class
- `obsidian_agent.py` - Obsidian-focused agent variant
- `cli.py` - Command-line interface (alternative to Discord)

**Key pattern**: Tools are registered via `agent.register_tool(tool)`. Create tools using `create_tool(name, description, params_schema, function)`.

**Tool Registration Order** (in discord_memory_bot.py on_ready):
1. get_all_tools() - file/system tools
2. get_memory_tools() - remember_permanently, get_full_profile, etc.
3. get_capability_tools() - request_capability
4. get_claude_tools() - ask_claude and variants
5. get_web_tools() - web_search, web_search_news, web_fetch
6. get_enhancement_tools() - add_enhancement, get_enhancements, update_enhancement_status
7. get_accountability_tools() - verify_* tools

**Obsidian Vault**: `C:\Users\razor\Documents\main\LLM Memory\`
- `Permanent/profile.md` - User profile for personalized news
- `Permanent/enhancements.md` - Feature request queue (THE source of truth)
- `Permanent/memories.md` - Long-term memory storage
- `Permanent/crash_log.md` - Crash reports with stack traces
- `Context/hourly.md`, `Context/daily.md` - Rolling conversation context

## Important Patterns

### MANDATORY: All Code Must Be Publishable
**This codebase is synced to a PUBLIC repo (technomancer-public). Every change you make will be visible to the public.**

Before writing or committing code, ask yourself:
- Does this contain any personal data (usernames, paths, API keys)?
- Does this reference hardcoded values that should be in config/settings?
- Would this look professional to someone reviewing the code on GitHub?

Use `settings.bot_owner` instead of hardcoded usernames. Use `settings.vault_path` instead of hardcoded paths. Use `settings.github_pages_url` instead of hardcoded URLs.

After every successful `safe_update.py continue`, run `python publish.py --push --force` to sync changes to the public repo. Or type `publish` in Discord.

### MANDATORY: Use safe_update.py for ALL Code Changes
**NEVER edit code directly on main. ALWAYS use the safe_update workflow for ANY code change.**

This is NON-NEGOTIABLE. Even for "small" changes, bugs can slip through (like Python scoping issues that tests don't catch). The branch workflow provides:
- Git history for rollback
- Isolated testing before merge
- Pre-commit hooks run on commit
- Clean separation of changes

**Workflow:**
```bash
cd local-agent

# 1. START: Create a branch (ALWAYS do this first)
python safe_update.py <short-description>

# 2. MAKE CHANGES: Edit files

# 3. VALIDATE: Run validate.py BEFORE committing (MANDATORY)
python validate.py startup
# Must show "VALIDATION PASSED" before proceeding

# 4. COMMIT: Only after validation passes
git add <files>
git commit -m "Description of changes"

# 5. FINISH: Test, merge, and restart
python safe_update.py continue

# 6. VERIFY: Confirm bot is running (MANDATORY)
python bot_service.py status
# Must show "Bot running: True"

# 7. PUBLISH & README: Generate fresh stats and push to both repos (MANDATORY)
python generate_readme.py               # Regenerate README with fresh test/coverage stats
git add README.md ../README.md
git commit -m "Update README with latest stats [auto]"
git push origin main                    # Private repo
python publish.py --push --force        # Public repo (technomancer-public)
```

### MANDATORY: Publish to Both Repos After Every Deploy
**After every successful deploy (whether via `safe_update.py continue` or manual merge), ALWAYS:**

1. **Regenerate README**: `python generate_readme.py` (updates test count, coverage badge, module stats)
2. **Commit the README**: `git add README.md ../README.md && git commit -m "Update README [auto]"`
3. **Push private**: `git push origin main`
4. **Push public**: `python publish.py --push --force`

**`safe_update.py continue` does steps 1-4 automatically.** For manual merges, you MUST do them yourself.

**Never tell the user a change is deployed without regenerating README and pushing to both repos.**

### MANDATORY: Run validate.py Before EVERY Commit
**Claude MUST run `python validate.py startup` before EVERY git commit. NO EXCEPTIONS.**

`validate.py` runs 3 levels of checks that catch the bugs unit tests miss:
1. **SYNTAX** — `ast.parse` every .py file (catches stray quotes, bad f-strings)
2. **IMPORT** — actually imports every module (catches NameError, missing deps)
3. **STARTUP** — starts the bot process, verifies it stays alive 10 seconds (catches runtime crashes)

For deeper validation, run `python validate.py full` which also tests real Ollama calls.

**If validate.py fails, DO NOT commit. Fix the errors first.**

**What `safe_update.py continue` does:**
1. Runs `pytest` - all tests must pass
2. Runs `mypy agent/` - type checks
3. Merges branch to main
4. Restarts the Discord bot
5. Cleans up the branch

### MANDATORY: Verify Bot is Running After Every Code Change
**After every `safe_update.py continue` or `bot_service.py start`, you MUST verify the bot is actually running:**

```bash
python bot_service.py status
```

Expected output: `Bot running: True`

If the bot is NOT running:
1. Check the crash log: `tail -50 "<vault_path>/LLM Memory/Permanent/crash_log.md"`
2. Try starting manually: `python bot_service.py start`
3. Check status again
4. If still not running, investigate the error before telling the user it's deployed

**Never tell the user a change is deployed without confirming `Bot running: True`.**

**DO NOT:**
- Commit code without running `python validate.py startup` first
- Edit files directly without starting a branch first
- Use `bot_service.py start` without going through safe_update
- Skip the workflow for "quick fixes" - those are exactly when bugs happen
- Tell the user something is deployed without verifying `Bot running: True`

### MANDATORY: Verify Bot is Running After Every Code Change
**After every `safe_update.py continue` or `bot_service.py start`, you MUST verify the bot is actually running:**

```bash
cd local-agent
python bot_service.py status
```

Expected output: `Bot running: True`

If the bot is NOT running:
1. Check the crash log: `tail -50 "<vault_path>/LLM Memory/Permanent/crash_log.md"`
2. Try starting manually: `python bot_service.py start`
3. Check status again: `python bot_service.py status`
4. If still not running, investigate the error before telling the user it's deployed

**Never tell the user a change is deployed without confirming `Bot running: True`.**

### MANDATORY: File Size Limits for Code Files
**Python files MUST stay under 5,000 lines. Target ~1,000 lines per file.**

When a file grows beyond ~1,000 lines, split it into logical modules:
- Extract utility functions into a `_utils.py` or `_helpers.py` file
- Extract command handlers, route handlers, etc. into separate files
- Keep tightly coupled code together, split at clean boundaries
- Use imports to maintain the same public API

This makes the codebase easier for AI ingestion and human review.

### Adding New Tools
```python
from .core import create_tool

tool = create_tool(
    "tool_name",
    "Description of what this tool does",
    {"type": "object", "properties": {...}, "required": [...]},
    your_function
)
# Register in discord_memory_bot.py's on_ready()
```

### Adding Tests for New Code
When adding new functionality:
1. Add unit tests in `tests/unit/test_<module>.py`
2. Use fixtures from `conftest.py` (mock_ollama_client, temp_vault, etc.)
3. Run `pytest -v` to verify
4. Tests must pass for `safe_update.py continue` to merge

Example test:
```python
def test_my_function(mock_ollama_client, temp_vault):
    """Test description."""
    # Setup
    mock_ollama_client.set_responses([...])

    # Execute
    result = my_function()

    # Assert
    assert result == expected
```

### Image Identification Flow
1. Local vision model (llava-llama3) responds first with quick analysis
2. Claude API (via ask_claude_with_image) follows with second opinion
3. Both responses sent to Discord sequentially

### News Digest
- Runs hourly from 9am-9pm
- Reads user profile from Obsidian vault for personalized "How it affects you" analysis
- Waits until next hour boundary on startup (doesn't send immediately)

### Enhancement Queue Rules
- THE ONLY source of truth is `/LLM Memory/Permanent/enhancements.md`
- The bot CANNOT implement code - it collects ideas only
- When user says "start working on enhancements" → bot lists them and directs to Claude Code
- Auto-numbered with #N format

### Accountability Pattern
Before reporting success on file/memory operations:
1. Call appropriate verify_* tool
2. Only confirm if "VERIFIED" returned
3. If verification fails, report the failure

## External Dependencies

- **Ollama**: Must be running at `http://127.0.0.1:11434` with models:
  - `qwen3.5:27b` - Main chat model (17GB VRAM)
  - `llava-llama3` - Vision model for image analysis
- **Anthropic API**: Set `ANTHROPIC_API_KEY` in `local-agent/.env` for Claude escalation
- **Discord**: Bot token in discord_memory_bot.py (line 29)
- **PyPDF2**: For PDF processing (`pip install PyPDF2`)
- **python-docx**: For Word doc processing (`pip install python-docx`)
- **pytest**: For running tests (`pip install pytest` or included in dev dependencies)
