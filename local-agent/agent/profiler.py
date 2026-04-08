"""
Request Profiler — Detailed timing data for every Discord message.

Captures end-to-end timing breakdown in JSON Lines format so Claude Code
can analyze performance patterns and make informed optimization decisions.

Data stored at: local-agent/profiling/requests.jsonl
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILING_DIR = Path(__file__).parent.parent / "profiling"
REQUESTS_LOG = PROFILING_DIR / "requests.jsonl"


@dataclass
class LLMCall:
    """Timing for a single LLM inference call."""

    turn: int = 0
    duration: float = 0.0
    input_chars: int = 0
    output_chars: int = 0
    num_ctx: int = 0
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class ToolExecution:
    """Timing for a single tool execution."""

    name: str = ""
    duration: float = 0.0
    result_chars: int = 0
    truncated: bool = False


@dataclass
class RequestProfile:
    """Full timing profile for one Discord message → response cycle."""

    timestamp: str = ""
    user: str = ""
    message: str = ""
    message_length: int = 0

    # Phase timings (seconds)
    context_build: float = 0.0
    pre_classification: float = 0.0
    reflection: float = 0.0
    post_processing: float = 0.0
    discord_send: float = 0.0
    total: float = 0.0

    # LLM call details
    llm_calls: list[LLMCall] = field(default_factory=list)

    # Tool execution details
    tool_executions: list[ToolExecution] = field(default_factory=list)

    # Classification results
    is_factual: bool = False
    question_type: str = ""
    reflection_mode: str = ""
    context_tier: str = ""
    context_injected_chars: int = 0
    num_ctx_used: int = 0

    # Response info
    response_length: int = 0
    claude_escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a dictionary for JSON output."""
        d = {
            "timestamp": self.timestamp,
            "user": self.user,
            "message": self.message[:200],
            "message_length": self.message_length,
            "total_seconds": round(self.total, 2),
            "phases": {
                "context_build": round(self.context_build, 3),
                "pre_classification": round(self.pre_classification, 3),
                "reflection": round(self.reflection, 3),
                "post_processing": round(self.post_processing, 3),
                "discord_send": round(self.discord_send, 3),
            },
            "llm_calls": [
                {
                    "turn": c.turn,
                    "duration": round(c.duration, 2),
                    "input_chars": c.input_chars,
                    "output_chars": c.output_chars,
                    "num_ctx": c.num_ctx,
                    "tool_calls": c.tool_calls,
                }
                for c in self.llm_calls
            ],
            "tool_executions": [
                {
                    "name": t.name,
                    "duration": round(t.duration, 3),
                    "result_chars": t.result_chars,
                    "truncated": t.truncated,
                }
                for t in self.tool_executions
            ],
            "llm_summary": {
                "total_calls": len(self.llm_calls),
                "total_llm_seconds": round(sum(c.duration for c in self.llm_calls), 2),
                "total_tool_seconds": round(sum(t.duration for t in self.tool_executions), 3),
                "avg_call_seconds": round(
                    sum(c.duration for c in self.llm_calls) / len(self.llm_calls), 2
                )
                if self.llm_calls
                else 0,
            },
            "classification": {
                "is_factual": self.is_factual,
                "question_type": self.question_type,
                "reflection_mode": self.reflection_mode,
                "context_tier": self.context_tier,
                "context_injected_chars": self.context_injected_chars,
                "num_ctx_used": self.num_ctx_used,
            },
            "response_length": self.response_length,
            "claude_escalated": self.claude_escalated,
        }
        return d


class RequestTimer:
    """Context-manager based timer for profiling request phases.

    Usage:
        profile = RequestProfile(user="testuser", message="hello")
        timer = RequestTimer(profile)

        with timer.phase("context_build"):
            # ... build context ...

        with timer.phase("pre_classification"):
            # ... classify ...

        timer.save()
    """

    def __init__(self, profile: RequestProfile) -> None:
        """Initialize the timer with a profile and start the clock."""
        self.profile = profile
        self._start = time.perf_counter()

    class _Phase:
        """Context manager that measures the duration of a named phase."""

        def __init__(self, timer: "RequestTimer", name: str) -> None:
            """Store the parent timer and phase name."""
            self.timer = timer
            self.name = name
            self.start = 0.0

        def __enter__(self) -> "_Phase":
            """Record the phase start time."""
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args: Any) -> None:
            """Record the phase duration on the profile."""
            duration = time.perf_counter() - self.start
            setattr(self.timer.profile, self.name, duration)

    def phase(self, name: str) -> "_Phase":
        """Return a context manager that times the named phase."""
        return self._Phase(self, name)

    def record_llm_call(self, turn: int, duration: float, input_chars: int,
                        output_chars: int, num_ctx: int, tool_calls: list[str]) -> None:
        """Append an LLM call timing record to the profile."""
        self.profile.llm_calls.append(LLMCall(
            turn=turn, duration=duration, input_chars=input_chars,
            output_chars=output_chars, num_ctx=num_ctx, tool_calls=tool_calls,
        ))

    def record_tool(self, name: str, duration: float, result_chars: int, truncated: bool = False) -> None:
        """Append a tool execution timing record to the profile."""
        self.profile.tool_executions.append(ToolExecution(
            name=name, duration=duration, result_chars=result_chars, truncated=truncated,
        ))

    def finish(self) -> None:
        """Finalize the profile with total elapsed time and timestamp."""
        self.profile.total = time.perf_counter() - self._start
        self.profile.timestamp = datetime.now().isoformat()

    def save(self) -> None:
        """Write the profile to the JSONL log."""
        self.finish()
        PROFILING_DIR.mkdir(parents=True, exist_ok=True)
        with open(REQUESTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.profile.to_dict()) + "\n")


def get_recent_profiles(n: int = 20) -> list[dict[str, Any]]:
    """Read the last N profiles from the log."""
    if not REQUESTS_LOG.exists():
        return []
    lines = REQUESTS_LOG.read_text(encoding="utf-8").strip().split("\n")
    profiles: list[dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            profiles.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return profiles


def get_performance_summary() -> str:
    """Generate a human-readable performance summary from recent profiles.

    Designed for both Discord display and Claude Code analysis.
    """
    profiles = get_recent_profiles(50)
    if not profiles:
        return "No profiling data yet."

    totals = [p["total_seconds"] for p in profiles]
    llm_totals = [p["llm_summary"]["total_llm_seconds"] for p in profiles]
    call_counts = [p["llm_summary"]["total_calls"] for p in profiles]
    ctx_sizes = [p["classification"]["context_injected_chars"] for p in profiles]

    avg_total = sum(totals) / len(totals)
    avg_llm = sum(llm_totals) / len(llm_totals)
    avg_calls = sum(call_counts) / len(call_counts)
    avg_ctx = sum(ctx_sizes) / len(ctx_sizes)

    # Find slowest requests
    sorted_by_time = sorted(profiles, key=lambda p: p["total_seconds"], reverse=True)

    lines = [
        f"**Performance Summary** ({len(profiles)} recent requests)",
        f"",
        f"**Averages:**",
        f"- Total response time: **{avg_total:.1f}s**",
        f"- LLM inference time: **{avg_llm:.1f}s** ({avg_llm/avg_total*100:.0f}% of total)" if avg_total > 0 else "- LLM inference time: 0s",
        f"- LLM calls per request: **{avg_calls:.1f}**",
        f"- Context injected: **{avg_ctx/1000:.1f}KB**",
        f"",
        f"**Slowest requests:**",
    ]

    for p in sorted_by_time[:5]:
        lines.append(
            f"- {p['total_seconds']:.1f}s | {p['llm_summary']['total_calls']} calls | "
            f"\"{p['message'][:60]}...\""
        )

    # Question type breakdown
    type_counts: dict[str, list[float]] = {}
    for p in profiles:
        qt = p["classification"]["question_type"] or "unknown"
        type_counts.setdefault(qt, []).append(p["total_seconds"])

    lines.append("")
    lines.append("**By question type:**")
    for qt, times in sorted(type_counts.items()):
        avg = sum(times) / len(times)
        lines.append(f"- {qt}: {len(times)} requests, avg {avg:.1f}s")

    return "\n".join(lines)
