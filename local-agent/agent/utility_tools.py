"""
Utility Tools - Calculator, Wikipedia, and Python execution.

Gives the LLM practical tools to answer questions accurately
instead of hallucinating math, facts, or data transformations.
"""

import json
import logging
import math
import re

import requests

from .core import create_tool

log = logging.getLogger(__name__)


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e, abs, round,
    min, max, int, float, and parentheses.

    Args:
        expression: Math expression to evaluate (e.g., "sqrt(144) + 2**3")

    Returns:
        The result as a string.
    """
    # Whitelist of safe names
    safe_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "int": int,
        "float": float,
        "pow": pow,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    # Reject anything that looks like code injection
    if any(kw in expression.lower() for kw in ["import", "exec", "eval", "open", "__", "os.", "sys."]):
        return "Error: Expression contains disallowed keywords."

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def wikipedia_summary(topic: str) -> str:
    """
    Get a Wikipedia summary for a topic.

    Args:
        topic: The topic to look up (e.g., "Blanchard Oklahoma")

    Returns:
        A summary from Wikipedia, or an error message.
    """
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        resp = requests.get(url, timeout=10, headers={"User-Agent": "TechnomancerBot/1.0"})

        if resp.status_code == 404:
            # Try search API as fallback
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "format": "json",
                "srlimit": 1,
            }
            search_resp = requests.get(search_url, params=params, timeout=10,
                                       headers={"User-Agent": "TechnomancerBot/1.0"})
            search_data = search_resp.json()
            results = search_data.get("query", {}).get("search", [])
            if results:
                # Try the first search result
                title = results[0]["title"]
                url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
                resp = requests.get(url, timeout=10, headers={"User-Agent": "TechnomancerBot/1.0"})
            else:
                return f"No Wikipedia article found for: {topic}"

        if resp.status_code != 200:
            return f"Wikipedia error (status {resp.status_code}) for: {topic}"

        data = resp.json()
        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"**{title}** (Wikipedia)\n\n{extract}"
        if page_url:
            result += f"\n\nSource: {page_url}"
        return result

    except requests.Timeout:
        return f"Wikipedia lookup timed out for: {topic}"
    except Exception as e:
        return f"Wikipedia error: {e}"


def run_python(code: str) -> str:
    """
    Execute a Python code snippet and return the output.

    Useful for data transformations, calculations, string manipulation,
    date math, and anything that's easier to compute than to reason about.

    Args:
        code: Python code to execute. Use print() for output.

    Returns:
        The stdout output, or error message.
    """
    import io
    import contextlib

    # Security: reject dangerous operations
    dangerous = ["import os", "import sys", "import subprocess", "open(", "__import__",
                  "exec(", "eval(", "compile(", "shutil", "pathlib", "glob"]
    if any(d in code for d in dangerous):
        return "Error: Code contains disallowed operations (file/system access not permitted)."

    stdout = io.StringIO()
    try:
        # Allow safe imports
        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "str": str, "int": int,
                "float": float, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "sorted": sorted, "reversed": reversed, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "sum": sum, "min": min, "max": max,
                "abs": abs, "round": round, "isinstance": isinstance, "type": type,
                "True": True, "False": False, "None": None, "bool": bool,
            },
            "math": math,
            "json": json,
            "re": re,
        }

        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals)

        output = stdout.getvalue().strip()
        return output if output else "(code executed successfully, no output)"
    except Exception as e:
        return f"Error: {e}"


def get_utility_tools() -> list:
    """Return utility tools for the agent."""
    return [
        create_tool(
            "calculate",
            "Evaluate a math expression. Use this instead of doing math in your head. "
            "Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e, etc. "
            "Example: calculate('sqrt(144) + 2**3')",
            {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            calculate,
        ),
        create_tool(
            "wikipedia",
            "Look up a topic on Wikipedia. Use this for factual information about places, "
            "people, events, science, history, etc. Returns a summary with source link.",
            {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to look up (e.g., 'Blanchard Oklahoma')",
                    }
                },
                "required": ["topic"],
            },
            wikipedia_summary,
        ),
        create_tool(
            "run_python",
            "Execute Python code and return the output. Use for calculations, data "
            "transformations, date math, string manipulation, or anything easier to "
            "compute than to reason about. Use print() to output results.",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() for output.",
                    }
                },
                "required": ["code"],
            },
            run_python,
        ),
    ]
