# Local Agent

Ollama-powered autonomous agent with tools for file operations, memory, Obsidian integration, and Claude escalation.

## Learning Articles

AI-generated developer learning articles, created daily by the bot via Claude API:

**[Browse Learning Articles](<your-github-pages-url>/learning/)**

Topics include Python, Oracle, Neo4j, system design, and best practices. New articles are generated daily at 8:00 AM.

## Installation

```bash
# Basic install
pip install -e .

# With Claude API support
pip install -e ".[claude]"

# With embeddings for semantic search
pip install -e ".[embeddings]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### CLI

```bash
# Interactive mode
local-agent

# Single task
local-agent "list all python files"

# With different model
local-agent --model mistral "read config.json"
```

### Obsidian Agent

```bash
# Interactive mode with your vault
obsidian-agent --vault "C:\Users\razor\Documents\main"

# Single task
obsidian-agent "find notes about AI"
```

### Python

```python
from agent import Agent, AgentConfig, get_all_tools

# Create agent
agent = Agent(AgentConfig(model="llama3.1"))

# Add tools
for tool in get_all_tools():
    agent.register_tool(tool)

# Run tasks
result = agent.run("What files are in the current directory?")
print(result)
```

## Available Tools

### File Operations
- `read_file` - Read file contents
- `write_file` - Write/create files
- `append_file` - Append to files
- `list_directory` - List directory contents
- `search_files` - Search by filename/content

### System
- `run_command` - Execute shell commands
- `get_system_info` - Get system information

### Memory (Simple)
- `observe` - Record an observation
- `query_memory` - Search memories
- `recent_memories` - Get recent memories
- `forget` - Remove a memory

### Knowledge Graph (SQLite + optional embeddings)
- `kg_observe` - Record to knowledge graph
- `kg_query` - Search knowledge graph
- `kg_wander` - Random walk for serendipity
- `kg_stats` - Graph statistics

### Obsidian
- `obs_read` - Read a note
- `obs_write` - Create/update a note
- `obs_append` - Append to a note
- `obs_search` - Search notes
- `obs_search_tag` - Find notes by tag
- `obs_list` - List notes
- `obs_links` - Get links/backlinks
- `obs_tags` - Get all tags
- `obs_daily` - Create daily note
- `obs_recent` - Recent notes
- `obs_stats` - Vault statistics

### Claude Escalation
- `escalate_to_claude` - Escalate complex tasks
- `report_to_claude` - Send findings to Claude
- `ask_claude` - Ask Claude a question

## Custom Tools

```python
from agent import Agent, AgentConfig, create_tool

def my_custom_tool(param: str) -> str:
    return f"Did something with {param}"

tool = create_tool(
    name="my_tool",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Input parameter"}
        },
        "required": ["param"]
    },
    function=my_custom_tool
)

agent = Agent(AgentConfig(model="llama3.1"))
agent.register_tool(tool)
```

## Requirements

- Python 3.10+
- Ollama running locally with a model (llama3.1, mistral, etc.)

## License

MIT
