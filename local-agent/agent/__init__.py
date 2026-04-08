"""
Local Agent - Ollama-powered autonomous agent with tools.

A general-purpose agent that can:
- Read/write files
- Run commands
- Maintain memory across sessions (simple JSON or full knowledge graph)
- Interact with Obsidian vaults
- Escalate to Claude when needed

Quick Start:
    from agent import Agent, AgentConfig, get_all_tools

    agent = Agent(AgentConfig(model="llama3.1"))
    for tool in get_all_tools():
        agent.register_tool(tool)

    result = agent.run("List files in the current directory")
"""

from .claude_bridge import ClaudeBridge
from .core import Agent, AgentConfig, Tool, create_tool
from .notifications import discord_send
from .tools import (
    get_all_tools,
    get_file_tools,
    get_system_tools,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "Tool",
    "create_tool",
    "get_all_tools",
    "get_file_tools",
    "get_system_tools",
    "ClaudeBridge",
    "discord_send",
    "__version__",
]
