"""
Local Agent — autonomous agent with tools, routing every LLM call through
``claude -p`` subprocess (the real Ollama + Anthropic SDK are shimmed out).

Quick Start:
    from agent import Agent, AgentConfig, get_all_tools

    agent = Agent(AgentConfig(model="claude-p"))
    for tool in get_all_tools():
        agent.register_tool(tool)

    result = agent.run("List files in the current directory")
"""

# --- LLM shim installation --------------------------------------------------
# Must run BEFORE any ``import ollama`` or ``import anthropic`` in sibling
# modules so those imports resolve to our claude -p shims. Ollama was too
# flaky to keep as a dependency; the Anthropic SDK was gutted to remove
# API-credit burn and credential surface. Everything LLM-shaped now spawns
# ``claude -p`` as a subprocess under the user's Claude Pro session.
from .ollama_shim import install_as_ollama as _install_ollama_shim
from .anthropic_shim import install_as_anthropic as _install_anthropic_shim

_install_ollama_shim()
_install_anthropic_shim()

from .claude_bridge import ClaudeBridge
from .core import Agent, AgentConfig, Tool, create_tool
from .leak_counter import reset as leak_counter_reset, get_count, increment
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
    "leak_counter_reset",
    "get_count",
    "increment",
    "__version__",
]
