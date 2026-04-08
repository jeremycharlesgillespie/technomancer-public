"""
Ask Claude - Use Claude CLI to get responses from Claude.

This allows the local LLM to pass questions to Claude via the CLI,
leveraging an existing Claude Pro subscription.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

# Log file for tracking Claude queries
LOG_FILE = Path(__file__).parent.parent / "claude_queries.log"

# Claude CLI path - installed via npm
CLAUDE_CLI = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"


def log_query(question: str, response: str, success: bool) -> None:
    """Log Claude queries for debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        status = "SUCCESS" if success else "FAILED"
        f.write(f"\n[{timestamp}] {status}\n")
        f.write(f"Q: {question[:200]}...\n" if len(question) > 200 else f"Q: {question}\n")
        f.write(f"A: {response[:500]}...\n" if len(response) > 500 else f"A: {response}\n")
        f.write("-" * 80 + "\n")


def ask_claude(question: str, context: str = "") -> str:
    """
    Ask Claude a question using the Claude CLI.

    Args:
        question: The question to ask Claude
        context: Optional context to provide

    Returns:
        Claude's response as a string
    """
    try:
        # Build the prompt
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {question}"
        else:
            full_prompt = question

        # Check if CLI exists
        if not CLAUDE_CLI.exists():
            return f"Claude CLI not found at {CLAUDE_CLI}. Check npm installation."

        # Create env without CLAUDECODE to avoid "nested session" error
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        # Call Claude CLI with --print flag to get just the response
        # Using -p for print mode (non-interactive, just outputs response)
        result = subprocess.run(
            [str(CLAUDE_CLI), "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            encoding="utf-8",
            shell=True,  # Needed for .cmd files on Windows
            env=env,  # Use env without CLAUDECODE
        )

        if result.returncode == 0:
            response = result.stdout.strip()
            log_query(question, response, True)
            return response
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            log_query(question, f"Error: {error_msg}", False)
            return f"Claude CLI error: {error_msg}"

    except subprocess.TimeoutExpired:
        log_query(question, "Timeout", False)
        return "Claude took too long to respond (timeout after 2 minutes)"
    except FileNotFoundError:
        log_query(question, "CLI not found", False)
        return "Claude CLI not found. Make sure 'claude' is in your PATH."
    except Exception as e:
        log_query(question, str(e), False)
        return f"Error calling Claude: {e}"


def get_claude_tools() -> list:
    """Get the ask_claude tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="ask_claude",
            description=(
                "Ask Claude (the advanced AI) a question. Use this when you need "
                "Claude's expertise for complex reasoning, coding help, creative writing, "
                "or any question you want Claude to answer directly. "
                "The user can trigger this by saying 'ask claude to...' or 'hey claude...'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or request for Claude",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context to provide Claude (e.g., code snippets, background info)",
                    },
                },
                "required": ["question"],
            },
            function=ask_claude,
        ),
    ]
