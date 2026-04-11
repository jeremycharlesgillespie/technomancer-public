"""
Capability Request System - Self-improving agent via Claude.

When the local LLM can't fulfill a request, it can ask Claude to implement
the missing capability. Claude evaluates, implements, tests, and deploys.
"""

import os
import re
import time as _time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .config import settings
from .core import Agent, AgentConfig
from .perf_monitor import record_llm_call as _record_perf

# Path to tools file for hot modification
TOOLS_FILE = Path(__file__).parent / "tools.py"

# Log file for capability requests
LOG_FILE = Path(__file__).parent.parent / "capability_requests.log"


def log_request(request_type: str, description: str, result: str):
    """Log capability requests for debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] {request_type}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Result: {result[:500]}...\n")
        f.write("-" * 80 + "\n")


def read_current_tools() -> str:
    """Read the current tools.py file."""
    return TOOLS_FILE.read_text(encoding="utf-8")


def request_capability(
    capability_description: str, original_user_request: str, context: str = ""
) -> str:
    """
    Request a new capability from Claude.

    This is called by the local LLM when it cannot fulfill a user request.
    Claude will evaluate if the capability can be safely implemented,
    and if so, write the code, test it, and deploy it.

    Args:
        capability_description: What capability is needed (e.g., "get weather data")
        original_user_request: The original request from the user
        context: Any additional context about what was tried

    Returns:
        Success message with the new capability, or explanation of why it can't be done.
    """
    # Read current tools for context
    current_tools = read_current_tools()

    # Build the prompt for Claude
    prompt = f"""You are a code architect helping extend a local LLM agent's capabilities.

The local LLM (running on Ollama) received this user request:
"{original_user_request}"

The LLM determined it cannot fulfill this request because it's missing this capability:
"{capability_description}"

Additional context: {context or "None"}

## Current Tools (tools.py)
The agent currently has these tools available. Review them to understand the patterns used:

```python
{current_tools}
```

## Your Task

1. EVALUATE: Can this capability be safely implemented? Consider:
   - Is it technically feasible?
   - Is it safe (no security risks, no destructive operations without explicit consent)?
   - Does it fit the pattern of existing tools?
   - Does it require external APIs/services that may not be available?

2. If YES, provide the implementation:
   - Write a Python function following the existing tool patterns
   - Write the tool registration code (create_tool call)
   - Explain where to insert the code

3. If NO, explain:
   - Why it cannot be implemented
   - What would be needed to make it possible
   - Any alternative approaches the user could take

## Response Format

If implementing, respond with EXACTLY this format:
```
IMPLEMENT: YES
REASON: [brief explanation]
FUNCTION_CODE:
```python
[the function code]
```
REGISTRATION_CODE:
```python
[the create_tool registration code]
```
INSERT_AFTER: [the function name to insert after, or "END" for end of file]
```

If not implementing:
```
IMPLEMENT: NO
REASON: [detailed explanation of why not]
ALTERNATIVE: [any alternative approaches]
```
"""

    try:
        cap_agent = Agent(AgentConfig(
            model=settings.ollama_model,
            verbose=False,
            system_prompt="You evaluate capability requests for a software project.",
        ))

        cap_start = _time.perf_counter()
        ollama_response = cap_agent.run(prompt)
        cap_duration = _time.perf_counter() - cap_start
        _record_perf(
            "ollama", cap_duration, success=True,
            model=settings.ollama_model,
        )

        log_request("CAPABILITY_REQUEST", capability_description, ollama_response)

        # Parse response
        if "IMPLEMENT: YES" in ollama_response:
            return _implement_capability(ollama_response, capability_description)
        else:
            reason_match = re.search(
                r"REASON:\s*(.+?)(?=ALTERNATIVE:|$)", ollama_response, re.DOTALL
            )
            alt_match = re.search(r"ALTERNATIVE:\s*(.+?)$", ollama_response, re.DOTALL)

            reason = reason_match.group(1).strip() if reason_match else "Unknown reason"
            alternative = alt_match.group(1).strip() if alt_match else ""

            result = f"CANNOT IMPLEMENT: {reason}"
            if alternative:
                result += f"\n\nALTERNATIVE: {alternative}"

            return result

    except Exception as e:
        log_request("ERROR", capability_description, traceback.format_exc())
        return f"CANNOT IMPLEMENT: Error during capability request: {e}"


def _implement_capability(claude_response: str, description: str) -> str:
    """Parse Claude's response and implement the new capability."""
    try:
        # Extract function code
        func_match = re.search(r"FUNCTION_CODE:\s*```python\s*(.+?)```", claude_response, re.DOTALL)
        if not func_match:
            return "FAILED: Could not parse function code from Claude's response"

        function_code = func_match.group(1).strip()

        # Extract registration code
        reg_match = re.search(
            r"REGISTRATION_CODE:\s*```python\s*(.+?)```", claude_response, re.DOTALL
        )
        if not reg_match:
            return "FAILED: Could not parse registration code from Claude's response"

        registration_code = reg_match.group(1).strip()

        # Extract insertion point
        insert_match = re.search(r"INSERT_AFTER:\s*(\w+)", claude_response)
        insert_after = insert_match.group(1) if insert_match else "END"

        # Read current tools
        current_code = read_current_tools()

        # Validate the code (basic syntax check)
        try:
            compile(function_code, "<new_function>", "exec")
        except SyntaxError as e:
            return f"FAILED: Syntax error in generated function: {e}"

        # Insert the function code
        if insert_after == "END":
            # Add before get_all_tools
            marker = "def get_all_tools()"
            if marker in current_code:
                new_code = current_code.replace(marker, f"{function_code}\n\n\n{marker}")
            else:
                new_code = current_code + f"\n\n{function_code}\n"
        else:
            # Find the function to insert after
            pattern = rf"(def {insert_after}\([^)]*\):[^\n]*\n(?:(?!^def ).*\n)*)"
            match = re.search(pattern, current_code, re.MULTILINE)
            if match:
                insert_pos = match.end()
                new_code = (
                    current_code[:insert_pos] + f"\n\n{function_code}\n" + current_code[insert_pos:]
                )
            else:
                # Fallback: add before get_all_tools
                marker = "def get_all_tools()"
                new_code = current_code.replace(marker, f"{function_code}\n\n\n{marker}")

        # Add registration to appropriate get_*_tools function
        # For now, add to get_system_tools
        system_tools_match = re.search(
            r"(def get_system_tools\(\)[^:]*:\s*[^\[]*\[)(.*?)(\s*\])", new_code, re.DOTALL
        )
        if system_tools_match:
            tools_list = system_tools_match.group(2)
            # Add the new registration
            new_tools_list = tools_list.rstrip().rstrip(",") + ",\n        " + registration_code
            new_code = (
                new_code[: system_tools_match.start(2)]
                + new_tools_list
                + new_code[system_tools_match.end(2) :]
            )

        # Write the updated code
        TOOLS_FILE.write_text(new_code, encoding="utf-8")

        # Try to reload the module
        try:
            import importlib

            from . import tools

            importlib.reload(tools)
            log_request("IMPLEMENTED", description, "Successfully implemented and reloaded")
            return f"SUCCESS: Implemented new capability: {description}. You can now retry the original request."
        except Exception as e:
            # Rollback on reload failure
            TOOLS_FILE.write_text(current_code, encoding="utf-8")
            log_request("ROLLBACK", description, f"Reload failed: {e}")
            return f"FAILED: Code was generated but failed to load: {e}. Changes rolled back."

    except Exception as e:
        log_request("IMPLEMENT_ERROR", description, traceback.format_exc())
        return f"FAILED: Error implementing capability: {e}"


def test_capability(function_name: str, test_input: dict) -> Tuple[bool, str]:
    """Test a newly implemented capability."""
    try:
        from . import tools

        func = getattr(tools, function_name, None)
        if func is None:
            return False, f"Function {function_name} not found"

        result = func(**test_input)
        return True, f"Test passed: {result}"
    except Exception as e:
        return False, f"Test failed: {e}"


# Tool registration for the local LLM
def get_capability_tools() -> list:
    """Get the capability request tools."""
    from .core import create_tool

    return [
        create_tool(
            name="request_capability",
            description=(
                "Request a new capability when you cannot fulfill the user's request. "
                "Claude will evaluate and potentially implement the missing feature. "
                "Call this ONLY when you've determined you lack the tools to help the user."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capability_description": {
                        "type": "string",
                        "description": "What capability is needed (e.g., 'get current weather', 'calculate taxes')",
                    },
                    "original_user_request": {
                        "type": "string",
                        "description": "The exact request from the user that you cannot fulfill",
                    },
                    "context": {
                        "type": "string",
                        "description": "What you tried and why it didn't work",
                    },
                },
                "required": ["capability_description", "original_user_request"],
            },
            function=request_capability,
        ),
    ]
