"""
Claude Bridge - Escalate tasks or report findings to Claude.

This allows the local Ollama agent to:
1. Escalate complex tasks to Claude
2. Send reports/findings to Claude for review
3. Ask Claude questions when stuck

Supports vault context integration for efficient token usage via prompt caching.
"""

import logging
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level daily cost tracker
_daily_cost_usd: float = 0.0
_daily_cost_date: str = ""

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .config import settings
from .logging_config import DEFAULT_REQUEST_ID, request_id_var
from .perf_monitor import record_llm_call as _record_perf


def _request_id_headers() -> dict[str, str]:
    """Return ``{"X-Request-ID": <id>}`` when a non-default rid is set."""
    rid = request_id_var.get()
    if rid and rid != DEFAULT_REQUEST_ID:
        return {"X-Request-ID": rid}
    return {}

# Approximate pricing per 1M tokens (Sonnet 4)
_PRICE_INPUT = 3.00  # $/1M input tokens
_PRICE_OUTPUT = 15.00  # $/1M output tokens
_PRICE_CACHE_READ = 0.30  # $/1M cached input tokens
_PRICE_CACHE_WRITE = 3.75  # $/1M cache creation tokens


def _estimate_cost(
    in_tokens: int, out_tokens: int,
    cache_read: int = 0, cache_write: int = 0,
) -> float:
    """Estimate USD cost from token counts."""
    return (
        in_tokens * _PRICE_INPUT / 1_000_000
        + out_tokens * _PRICE_OUTPUT / 1_000_000
        + cache_read * _PRICE_CACHE_READ / 1_000_000
        + cache_write * _PRICE_CACHE_WRITE / 1_000_000
    )


def _log_api_cost(
    source: str, model: str,
    in_tokens: int, out_tokens: int,
    cache_read: int = 0, cache_write: int = 0,
) -> None:
    """Log API cost and check daily threshold."""
    global _daily_cost_usd, _daily_cost_date

    cost = _estimate_cost(in_tokens, out_tokens, cache_read, cache_write)
    today = datetime.now().strftime("%Y-%m-%d")

    # Reset daily counter on new day
    if _daily_cost_date != today:
        _daily_cost_usd = 0.0
        _daily_cost_date = today

    _daily_cost_usd += cost

    logger.info(
        f"[API Cost] {source} | model={model} | "
        f"in={in_tokens} out={out_tokens} cache_r={cache_read} cache_w={cache_write} | "
        f"est=${cost:.4f} | daily_total=${_daily_cost_usd:.4f}"
    )

    # Alert if daily spend exceeds threshold
    if _daily_cost_usd >= settings.api_cost_alert_threshold:
        try:
            from .alerts import send_alert

            send_alert(
                f"Daily API spend ${_daily_cost_usd:.2f} exceeded "
                f"${settings.api_cost_alert_threshold:.2f} threshold",
                title="API Cost Alert",
                level="warning",
            )
        except Exception:
            pass


def get_daily_api_cost() -> float:
    """Return the current daily API cost estimate."""
    today = datetime.now().strftime("%Y-%m-%d")
    if _daily_cost_date != today:
        return 0.0
    return _daily_cost_usd


class ClaudeBridge:
    """Bridge to communicate with Claude."""

    def __init__(
        self,
        mode: str = "auto",
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
        use_vault_context: bool | None = None,
    ):
        self.timeout = timeout
        self.model = model
        self.api_key = api_key
        # Use vault context by default based on settings
        self.use_vault_context = (
            use_vault_context
            if use_vault_context is not None
            else settings.claude_use_vault_context
        )

        if mode == "auto":
            if api_key or HAS_ANTHROPIC:
                self.mode = "api"
            else:
                self.mode = "cli"
        else:
            self.mode = mode

        self.client = None
        if self.mode == "api" and HAS_ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def send(self, message: str, context: str = "", task_type: str = "general") -> str:
        """Send a message to Claude and get a response.

        Checks the fallback orchestrator first — if Claude API is degraded,
        routes the request to the local Ollama model instead.
        """
        prompt = self._build_prompt(message, context, task_type)

        if self.mode == "api" and self.client:
            # Check fallback orchestrator before calling Claude
            from .fallback_orchestrator import should_use_fallback

            if should_use_fallback():
                return self._send_ollama_fallback(message, context)
            return self._send_api(prompt)
        else:
            return self._send_cli(prompt)

    def _build_prompt(self, message: str, context: str, task_type: str) -> str:
        """Build a structured prompt based on task type."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if task_type == "escalate":
            return f"""[ESCALATED FROM LOCAL AI AGENT - {timestamp}]

The local Ollama agent has escalated this task to you.

TASK:
{message}

CONTEXT:
{context if context else "None"}

Please help complete this task."""

        elif task_type == "report":
            return f"""[REPORT FROM LOCAL AI AGENT - {timestamp}]

REPORT:
{message}

CONTEXT:
{context if context else "None"}

Please review and provide feedback."""

        elif task_type == "question":
            return f"""[QUESTION FROM LOCAL AI AGENT - {timestamp}]

QUESTION:
{message}

CONTEXT:
{context if context else "None"}

Please provide a clear answer."""

        else:
            return f"""[FROM LOCAL AI AGENT - {timestamp}]

{message}

{f"Context: {context}" if context else ""}"""

    def _send_api(self, prompt: str) -> str:
        """Send via Anthropic API, optionally using vault context with caching."""
        from .fallback_orchestrator import record_claude_result

        # Use vault session if enabled (provides cached context)
        # Note: vault session calls are already instrumented in claude_vault.py
        if self.use_vault_context:
            try:
                from .claude_vault import get_vault_session

                session = get_vault_session()
                api_start = _time.perf_counter()
                response = session.ask(prompt)
                latency = _time.perf_counter() - api_start

                is_error = response.startswith("Error:")
                record_claude_result(
                    success=not is_error, latency=latency,
                    error=response[:200] if is_error else "",
                )

                if is_error:
                    return self._maybe_ollama_fallback(prompt, response)

                # Append cost info if enabled
                if settings.claude_show_cost:
                    cost = session.format_last_cost()
                    return f"{response}\n\n`Cost: {cost}`"
                return response
            except ImportError:
                pass  # Fall back to direct API call
            except Exception as e:
                record_claude_result(success=False, error=str(e)[:200])
                print(f"[ClaudeBridge] Vault session error, using direct API: {e}")

        # Direct API call (no vault context)
        api_start = _time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=_request_id_headers(),
            )
            latency = _time.perf_counter() - api_start
            in_tokens = getattr(response.usage, "input_tokens", 0)
            out_tokens = getattr(response.usage, "output_tokens", 0)
            _record_perf(
                "claude_api", latency, success=True,
                model=self.model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            record_claude_result(success=True, latency=latency)
            _log_api_cost("ClaudeBridge", self.model, in_tokens, out_tokens)
            return getattr(response.content[0], "text", str(response.content[0]))
        except Exception as e:
            latency = _time.perf_counter() - api_start
            _record_perf(
                "claude_api", latency, success=False,
                model=self.model, error=str(e),
            )
            record_claude_result(success=False, latency=latency, error=str(e)[:200])
            return self._maybe_ollama_fallback(prompt, f"Error: {e}")

    def _send_cli(self, prompt: str) -> str:
        """Send via Claude CLI."""
        cli_start = _time.perf_counter()
        try:
            temp_file = Path(".claude_bridge_prompt.txt")
            temp_file.write_text(prompt, encoding="utf-8")

            if sys.platform == "win32":
                cmd = f'type "{temp_file}" | claude -p -'
            else:
                cmd = f'cat "{temp_file}" | claude -p -'

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            temp_file.unlink(missing_ok=True)
            success = result.returncode == 0
            _record_perf("claude_cli", _time.perf_counter() - cli_start,
                         success=success, model="claude-cli")
            return result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            _record_perf("claude_cli", _time.perf_counter() - cli_start,
                         success=False, model="claude-cli",
                         error=f"Timeout after {self.timeout}s")
            return f"Error: Timed out after {self.timeout}s"
        except Exception as e:
            _record_perf("claude_cli", _time.perf_counter() - cli_start,
                         success=False, model="claude-cli", error=str(e))
            return f"Error: {e}"

    def _send_ollama_fallback(self, message: str, context: str = "") -> str:
        """Route a request to local Ollama model as fallback for Claude."""
        from .core import Agent, AgentConfig

        fallback_agent = Agent(
            AgentConfig(
                model=settings.ollama_model,
                verbose=False,
                system_prompt=(
                    "You are acting as a fallback for Claude API which is currently "
                    "unavailable. Answer the following request as helpfully as you can."
                ),
            )
        )
        try:
            prompt = message if not context else f"{context}\n\n{message}"
            result = fallback_agent.run(prompt)
            return f"[Ollama fallback — Claude API unavailable]\n{result}"
        except Exception as e:
            return f"Error: Claude API unavailable and Ollama fallback failed: {e}"

    def _maybe_ollama_fallback(self, prompt: str, error_response: str) -> str:
        """After a Claude API error, check if fallback is now active and use Ollama."""
        from .fallback_orchestrator import should_use_fallback

        if should_use_fallback():
            return self._send_ollama_fallback(prompt)
        return error_response

    def escalate(self, task: str, context: str = "") -> str:
        return self.send(task, context, task_type="escalate")

    def report(self, findings: str, context: str = "") -> str:
        return self.send(findings, context, task_type="report")

    def ask(self, question: str, context: str = "") -> str:
        return self.send(question, context, task_type="question")


_bridge = None


def _get_bridge() -> ClaudeBridge:
    global _bridge
    if _bridge is None:
        _bridge = ClaudeBridge()
    return _bridge


def escalate_to_claude(task: str, context: str = "") -> str:
    """Escalate a task to Claude."""
    return _get_bridge().escalate(task, context)


def report_to_claude(findings: str, context: str = "") -> str:
    """Send a report to Claude."""
    return _get_bridge().report(findings, context)


def ask_claude(question: str, context: str = "") -> str:
    """Ask Claude a question."""
    return _get_bridge().ask(question, context)


def get_claude_tools():
    """Get tools for Claude communication."""
    from .core import create_tool

    return [
        create_tool(
            name="escalate_to_claude",
            description="Escalate a task to Claude when it's too complex",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to escalate"},
                    "context": {"type": "string", "description": "What you've tried"},
                },
                "required": ["task"],
            },
            function=escalate_to_claude,
        ),
        create_tool(
            name="report_to_claude",
            description="Send findings to Claude for review",
            parameters={
                "type": "object",
                "properties": {
                    "findings": {"type": "string", "description": "Your findings"},
                    "context": {"type": "string", "description": "Additional context"},
                },
                "required": ["findings"],
            },
            function=report_to_claude,
        ),
        create_tool(
            name="ask_claude",
            description="Ask Claude a question when stuck",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Your question"},
                    "context": {"type": "string", "description": "Relevant context"},
                },
                "required": ["question"],
            },
            function=ask_claude,
        ),
    ]
