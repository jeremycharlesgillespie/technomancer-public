# Technomancer

An autonomous AI agent framework powered by Ollama, with a Discord bot interface, Obsidian vault memory, Claude API escalation, and a self-improving idea board system.

## Architecture

```
Discord (#llm_chat)          Discord (#claude-code)        Idea Board (port 8322)
       |                            |                            |
       v                            v                            v
  Local LLM (Ollama)         Claude Code (Pro)            LLM Idea Generator
  qwen3.5:9b on GPU         claude.ai/code remote         Hourly analysis
       |                            |                            |
       +------- Obsidian Vault (persistent memory) -------------+
       |                            |                            |
       +------- GitHub Pages (HTML content delivery) -----------+
       |                            |                            |
       +------- Bridge API (port 8321, bidirectional) ----------+
```

## Features

### Core Bot
- **Discord Chat** — Conversational AI with 40+ registered tools
- **Obsidian Memory** — Persistent memory with hourly/daily/weekly LLM-powered compaction
- **Auto Memory Extraction** — Builds an identity database from conversations
- **Smart Context Injection** — Tiered context based on message complexity
- **Tool Result Truncation** — Large results stored and retrieved on demand

### Claude Integration
- **Claude API Escalation** — Auto-escalates to Claude when the local model is uncertain
- **Claude Code Remote** — Full Claude Code sessions via `claude.ai/code` from your phone
- **Discord Bridge API** — REST API (port 8321) for bidirectional Discord communication

### Self-Improvement
- **Idea Board** — Web dashboard (port 8322) where the LLM suggests improvements
- **LLM Discussion** — Chat with the LLM about ideas before implementing
- **Hourly Idea Generation** — Analyzes news, conversations, errors, and performance data
- **Auto-Improve** — Tests itself, diagnoses weaknesses, patches its own prompts

### Content & Learning
- **Developer Learning** — Daily AI-generated educational articles via Claude API
- **News Digest** — Hourly personalized tech news with relevance filtering (9am-9pm)
- **GitHub Pages** — Auto-deploys HTML content for mobile viewing
- **HTML Normalization** — Consistent styling across all generated pages

### Monitoring
- **Request Profiling** — Per-message timing breakdown (JSONL format)
- **Prometheus Metrics** — LLM call latency, token counts, error rates
- **Grafana Dashboard** — Visual performance monitoring
- **Dreaming** — Background memory consolidation (midnight-6am)

## Prerequisites

- **Python 3.12+**
- **Ollama** running at `http://127.0.0.1:11434` with `qwen3.5:9b` model
- **Discord Bot Token** (from Discord Developer Portal)
- **Anthropic API Key** (optional, for Claude escalation)
- **GPU** recommended (NVIDIA with 8GB+ VRAM for qwen3.5:9b)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/jeremycharlesgillespie/technomancer.git
cd technomancer/local-agent

# Configure environment
cp .env.example .env
# Edit .env with your Discord bot token and other settings

# Install dependencies
pip install -e ".[all]"

# Start the bot
python bot_service.py start

# Check status
python bot_service.py status
```

## Discord Commands

### Chat (`#llm_chat` channel)
| Command | Description |
|---|---|
| `betterDev [topic]` | Generate a learning article |
| `techNews` | Latest tech news with analysis |
| `idea` | Generate improvement ideas on demand |
| `think` | Show what the bot knows about you |
| `perf` | Show performance profiling stats |
| `showCommands` | List all commands |

### Claude Code (`#claude-code` channel)
Everything typed in this channel runs as a Claude Code session on your machine. Type `end` to close the session.

### Admin
| Command | Description |
|---|---|
| `reloadServer` | Restart the bot (owner only) |
| `evolve` | Run self-improvement cycle (owner only) |

## Configuration

All configuration is in `.env`. See `.env.example` for all available options.

Key settings:
- `DISCORD_BOT_TOKEN` — Required. Your Discord bot token
- `ANTHROPIC_API_KEY` — Optional. For Claude API features
- `VAULT_PATH` — Path to your Obsidian vault
- `OLLAMA_MODEL` — LLM model name (default: `qwen3.5:9b`)

## Development

```bash
# Run tests
pytest

# Run with verbose output
pytest -v

# Validate before committing (syntax + imports + bot startup)
python validate.py startup

# Safe update workflow (branch, test, merge, restart)
python safe_update.py <branch-name>
# ... make changes ...
python safe_update.py continue
```

## Project Structure

```
local-agent/
  agent/                    # Core bot modules
    discord_memory_bot.py   # Main Discord bot + message dispatcher
    bot_commands.py         # Discord command handlers
    bot_utils.py            # Utility functions
    core.py                 # Agent class with Ollama tool-calling loop
    memory_system.py        # Obsidian-backed conversation memory
    auto_memory.py          # Identity extraction from conversations
    dreaming.py             # Background memory consolidation
    news_digest.py          # Hourly tech news with relevance filtering
    dev_learning.py         # Daily developer learning articles
    discord_bridge.py       # REST API for Claude Code <-> Discord
    claude_code_runner.py   # Headless Claude Code execution
    idea_generator.py       # Hourly improvement idea generation
    itinerary.py            # Travel itinerary via Claude API
    profiler.py             # Request timing profiler
    html_generator.py       # Markdown -> HTML with dark mode
    config.py               # Centralized Pydantic settings
  idea_board/               # Self-improvement idea system
    web.py                  # Flask dashboard (port 8322)
    models.py               # Idea data model + Obsidian sync
    executor.py             # Claude Code execution manager
  tests/                    # Test suite (600+ tests)
  validate.py               # Pre-commit validation script
  safe_update.py            # Branch-test-merge automation
  bot_service.py            # Bot process manager
```

## Learning Articles

AI-generated developer learning articles:
**[Browse Articles](https://jeremycharlesgillespie.github.io/technomancer/learning/)**

## License

MIT
