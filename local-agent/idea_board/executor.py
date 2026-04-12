"""
Idea Executor — Manages Claude Code subprocess lifecycle for idea execution.

Provides:
- Live stdout streaming to the idea's execution_log
- Discord notifications as execution progresses
- Process PID tracking for cancel/health checks
- Auto-timeout recovery (no stuck "executing" states)
- Cancel support via PID kill

The execution log is updated in real-time as Claude Code works, so
the dashboard can poll and display progress line by line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import settings
from .models import add_comment, get_idea, load_ideas, mark_done, mark_executing, mark_failed

logger = logging.getLogger(__name__)

# Timeout for Claude Code execution (30 minutes — includes safe_update workflow)
EXECUTION_TIMEOUT: int = 1800

# Timeout for exploration pass (5 minutes — CLAUDE.md is large)
EXPLORATION_TIMEOUT: int = 300

# Minimum seconds between Discord webhook sends (rate limiting)
DISCORD_RATE_LIMIT: float = 10.0

# Category-specific implementation guidance
CATEGORY_GUIDANCE: dict[str, str] = {
    "performance": (
        "- Measure baseline metrics BEFORE making changes (use profiler.py or time commands)\n"
        "- Profile with agent/profiler.py to identify bottlenecks\n"
        "- Include before/after numbers in the commit message\n"
        "- Avoid premature optimization — measure first, optimize what matters"
    ),
    "feature": (
        "- Add unit tests in tests/unit/test_<module>.py\n"
        "- Follow existing test patterns from tests/conftest.py fixtures "
        "(mock_ollama_client, temp_vault, etc.)\n"
        "- Register new tools in discord_memory_bot.py on_ready() if applicable\n"
        "- Use create_tool() from agent/core.py for new tools"
    ),
    "quality": (
        "- Focus on readability and maintainability\n"
        "- Keep files under 1000 lines — extract utilities if needed\n"
        "- Run the FULL test suite, not just new tests\n"
        "- Don't add features — stick to the quality improvement scope"
    ),
    "security": (
        "- Check OWASP top 10 vulnerabilities\n"
        "- Validate all external input at system boundaries\n"
        "- Never hardcode secrets — use settings from agent/config.py\n"
        "- Check for command injection in any shell/subprocess calls"
    ),
    "ux": (
        "- Test the change from the Discord user's perspective\n"
        "- Ensure error messages are helpful and actionable\n"
        "- Keep Discord messages under 2000 chars\n"
        "- Use validate_discord_message() from message_validators.py"
    ),
}

# Bridge token file for Discord notifications
BRIDGE_TOKEN_FILE: Path = Path(__file__).parent.parent / ".bridge_token"


@dataclass
class ExecutionState:
    """Tracks a running Claude Code execution.

    Attributes:
        idea_id: The idea being executed
        pid: Claude Code subprocess PID
        started_at: Unix timestamp when execution started
        log_lines: Live buffer of stdout lines
        thread: The background thread running the execution
        cancelled: Whether cancellation was requested
    """

    idea_id: str
    pid: int | None = None
    started_at: float = field(default_factory=time.time)
    log_lines: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    cancelled: bool = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def is_alive(self) -> bool:
        """Check if the Claude Code process is still running."""
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @property
    def log_text(self) -> str:
        return "\n".join(self.log_lines)


# Active executions: idea_id -> ExecutionState
_active: dict[str, ExecutionState] = {}


def get_execution(idea_id: str) -> ExecutionState | None:
    """Get the active execution state for an idea.

    Args:
        idea_id: The idea ID

    Returns:
        ExecutionState if executing, None otherwise
    """
    return _active.get(idea_id)


def _notify_discord(message: str) -> None:
    """Send a notification to the #claude-code-updates channel via webhook.

    Falls back to the bridge API (main chat channel) if no webhook is configured.
    """
    try:
        import requests

        webhook_url = settings.discord_claude_code_webhook
        if webhook_url:
            requests.post(
                webhook_url,
                json={"content": message},
                timeout=5,
            )
            return

        # Fallback: bridge API to main channel
        if not BRIDGE_TOKEN_FILE.exists():
            return
        token = BRIDGE_TOKEN_FILE.read_text(encoding="utf-8").strip()
        requests.post(
            "http://127.0.0.1:8321/api/send",
            headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
            json={"message": message},
            timeout=5,
        )
    except Exception:
        pass


def _find_claude_binary() -> Path | None:
    """Find the Claude Code binary."""
    extensions_dir = Path.home() / ".vscode" / "extensions"
    if not extensions_dir.exists():
        return None
    candidates = sorted(
        extensions_dir.glob("anthropic.claude-code-*/resources/native-binary/claude.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Prompt builders — assemble rich context for Claude Code
# ---------------------------------------------------------------------------


def _build_discussion(idea: Any) -> str:
    """Format discussion thread from idea comments."""
    if not idea.comments:
        return ""
    lines = ["\n## Discussion (what was decided)"]
    for c in idea.comments:
        label = f"{settings.owner_name} (manager)" if c.author == "owner" else "LLM (engineer)"
        lines.append(f"- {label}: {c.text}")
    return "\n".join(lines)


def _build_epic_context(idea: Any) -> str:
    """Build parent epic and sibling context for a story."""
    if not idea.parent_id:
        return ""
    parent = get_idea(idea.parent_id)
    if not parent:
        return ""

    all_ideas = load_ideas()
    siblings = [i for i in all_ideas if i.parent_id == idea.parent_id]

    lines = [
        f"\n## Parent Epic: {parent.title}",
        f"**Epic Description:** {parent.description}",
        "",
        "**Stories in this epic:**",
    ]
    for s in siblings:
        if s.id == idea.id:
            lines.append(f"  - **[THIS] {s.id}: {s.title}** <-- you are implementing this one")
        elif s.state == "done":
            lines.append(f"  - [DONE] {s.id}: {s.title}")
        else:
            lines.append(f"  - {s.id}: {s.title}")
    lines.append(
        "\nBuild on what the completed stories created. "
        "Ensure your implementation integrates with the epic's full lifecycle goal."
    )
    return "\n".join(lines)


def _build_children_context(idea: Any) -> str:
    """Build child story list for an epic."""
    all_ideas = load_ideas()
    kids = [i for i in all_ideas if i.parent_id == idea.id]
    if not kids:
        return ""
    lines = ["\n**Stories in this epic:**"]
    for k in kids:
        done_marker = " [DONE]" if k.state == "done" else ""
        lines.append(f"  - {k.id}: {k.title}{done_marker}")
    return "\n".join(lines)


def _load_codebase_summary() -> str:
    """List all Python files in agent/ and idea_board/ with their first docstring line."""
    agent_dir = Path(__file__).parent.parent / "agent"
    lines = []
    for f in sorted(agent_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        desc = ""
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if '"""' in content:
                doc_start = content.index('"""') + 3
                doc_end = content.index('"""', doc_start)
                first_line = content[doc_start:doc_end].strip().split("\n")[0]
                desc = f" — {first_line}"
        except (ValueError, OSError):
            pass
        lines.append(f"- {f.name}{desc}")

    board_dir = Path(__file__).parent
    for f in sorted(board_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        lines.append(f"- idea_board/{f.name}")
    return "\n".join(lines)


def _find_relevant_test_file(idea: Any) -> str:
    """Detect which module the idea targets and include its test file as reference.

    Scans the idea description for module names (e.g. "agent/foo.py", "foo.py",
    or bare names matching files in agent/). If a corresponding test file exists,
    includes its first 80 lines. Falls back to conftest.py fixtures if no match.
    """
    import re

    tests_dir = Path(__file__).parent.parent / "tests" / "unit"
    agent_dir = Path(__file__).parent.parent / "agent"
    conftest = Path(__file__).parent.parent / "tests" / "conftest.py"

    description = f"{idea.title} {idea.description}"

    # Strategy 1: Look for explicit file references like "agent/foo.py" or "foo.py"
    file_refs = re.findall(r"(?:agent/)?(\w+)\.py", description)

    # Strategy 2: Look for module-like words that match actual agent/*.py files
    agent_modules = {f.stem for f in agent_dir.glob("*.py") if not f.name.startswith("_")}

    # Score candidates by how likely they are the target module
    candidates: list[str] = []
    for ref in file_refs:
        if ref in agent_modules and ref not in ("__init__", "config", "core"):
            candidates.append(ref)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # Try to find a matching test file
    for module_name in unique[:3]:
        test_file = tests_dir / f"test_{module_name}.py"
        if test_file.exists():
            try:
                lines = test_file.read_text(encoding="utf-8", errors="replace").split("\n")
                snippet = "\n".join(lines[:80])
                if len(snippet) > 2000:
                    snippet = snippet[:2000] + "\n... (truncated)"
                return (
                    f"\n## Test Pattern Reference (from test_{module_name}.py)\n"
                    f"```python\n{snippet}\n```"
                )
            except OSError:
                continue

    # Fallback: show conftest.py fixtures
    if conftest.exists():
        try:
            lines = conftest.read_text(encoding="utf-8", errors="replace").split("\n")
            snippet = "\n".join(lines[:50])
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "\n... (truncated)"
            return (
                f"\n## Test Pattern Reference (from conftest.py — available fixtures)\n"
                f"```python\n{snippet}\n```"
            )
        except OSError:
            pass

    return ""


def _get_category_guidance(category: str) -> str:
    """Get category-specific implementation guidance."""
    guidance = CATEGORY_GUIDANCE.get(category, "")
    if not guidance:
        return ""
    return f"\n## Category Guidance ({category})\n{guidance}"


def _load_similar_execution_logs(idea: Any) -> str:
    """Find completed ideas in the same category and include their execution logs."""
    try:
        all_ideas = load_ideas()
        similar = [
            i for i in all_ideas
            if i.state == "done"
            and i.category == idea.category
            and i.id != idea.id
            and i.execution_log
            and len(i.execution_log.strip()) > 50
        ]
        if not similar:
            return ""

        # Sort by creation date (most recent first) and take top 2
        similar.sort(key=lambda i: i.created, reverse=True)
        refs = similar[:2]

        lines = ["\n## Reference: How similar ideas were implemented"]
        for ref in refs:
            lines.append(f"\n**{ref.id}: {ref.title}**")
            # Last 1000 chars of execution log (completion summary)
            log_snippet = ref.execution_log[-1000:]
            if len(ref.execution_log) > 1000:
                log_snippet = "..." + log_snippet
            lines.append(f"```\n{log_snippet}\n```")

        return "\n".join(lines)
    except Exception:
        return ""


def _load_git_history() -> str:
    """Load recent git commit messages for context."""
    try:
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "log", "--oneline", "-15"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"\n## Recent Changes (git log)\n```\n{result.stdout.strip()}\n```"
    except Exception:
        pass
    return ""


def _load_recent_errors() -> str:
    """Load recent crash log entries for context."""
    crash_log = settings.llm_memory_path / "Permanent" / "crash_log.md"
    if not crash_log.exists():
        return ""
    try:
        content = crash_log.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return ""
        # Last 1500 chars (most recent errors)
        snippet = content[-1500:] if len(content) > 1500 else content
        return f"\n## Recent Errors (from crash log)\n```\n{snippet}\n```"
    except OSError:
        return ""


def _build_workflow_section(idea: Any) -> str:
    """Build the mandatory workflow and completion instructions."""
    short_name = idea.id.replace("idea-", "")
    return (
        f"\n## MANDATORY WORKFLOW\n"
        f"Follow these steps EXACTLY:\n"
        f"1. Read CLAUDE.md for project conventions\n"
        f"2. `cd local-agent`\n"
        f"3. `python safe_update.py {short_name}`\n"
        f"4. Make your code changes (with tests if adding new functionality)\n"
        f"5. `python validate.py startup` — MUST show VALIDATION PASSED\n"
        f"6. `git add <files>` && `git commit -m 'description'`\n"
        f"7. `python safe_update.py continue` — runs tests, merges, restarts bot\n"
        f"8. `python bot_service.py status` — MUST show Bot running: True\n"
        f"\nDo NOT skip any steps. Do NOT commit without validate.py passing.\n"
        f"\n## CRITICAL: How to run safe_update.py continue\n"
        f"The `safe_update.py continue` command takes 3-5 minutes to complete. "
        f"It runs pytest (~40s), merges, restarts the bot, runs quality tests (~60s), "
        f"generates README (~60s), and publishes. "
        f"Run it as a SINGLE BLOCKING Bash command with a 600 second timeout. "
        f"Do NOT run it in the background. Do NOT poll or check on it. "
        f"Do NOT run it multiple times. Just run it once and wait for the output. "
        f"If it fails, read the output to understand why.\n"
        f"\n## After Completion\n"
        f"When DONE and verified (bot running), mark the idea as complete:\n"
        f"```bash\n"
        f"curl -X POST http://localhost:8322/api/ideas/{idea.id}/done\n"
        f"```\n"
        f"If you CANNOT complete this, log the failure:\n"
        f"```bash\n"
        f'curl -X POST http://localhost:8322/api/ideas/{idea.id}/comment '
        f'-H "Content-Type: application/json" '
        f"-d '{{\"author\": \"claude\", \"text\": \"Execution failed: <describe what went wrong>\"}}'\n"
        f"```\n"
    )


def _enrich_stub_description(idea: Any) -> str:
    """If a story has only a stub description, pull the parent epic's full description.

    Auto-generated stories from idea_generator get placeholder descriptions like
    "Story under epic: <title>". These are useless for implementation. When detected,
    we pull the parent epic's full WHAT/WHY/HOW description and prepend it.
    """
    desc = idea.description or ""
    is_stub = (
        desc.startswith("Story under epic:")
        or len(desc.strip()) < 80
    )
    if not is_stub or not idea.parent_id:
        return desc

    parent = get_idea(idea.parent_id)
    if not parent or not parent.description:
        return desc

    return (
        f"**This story is part of:** {parent.title}\n\n"
        f"**Epic context (use this to guide your implementation):**\n"
        f"{parent.description}\n\n"
        f"**Your specific task:** {idea.title}\n"
        f"{desc}"
    )


def _build_story_prompt(idea: Any) -> str:
    """Build a rich prompt for executing a single story/task."""
    type_label = f"[{idea.idea_type.upper()}] " if idea.idea_type != "story" else ""
    description = _enrich_stub_description(idea)

    sections = [
        f"# Task: Implement {idea.title}\n",
        f"You are implementing a {idea.idea_type} for the Technomancer project.\n",
        f"## {type_label}Idea Details",
        f"- **ID:** {idea.id}",
        f"- **Category:** {idea.category}",
        f"- **Description:** {description}",
        _build_epic_context(idea),
        _build_discussion(idea),
        f"\n## Codebase (what already exists — don't duplicate)\n{_load_codebase_summary()}",
        _load_git_history(),
        _load_recent_errors(),
        _load_similar_execution_logs(idea),
        _get_category_guidance(idea.category),
        _find_relevant_test_file(idea),
        _build_workflow_section(idea),
    ]
    return "\n".join(s for s in sections if s)


def _build_epic_prompt(idea: Any) -> str:
    """Build a rich prompt for executing an entire epic sequentially."""
    all_ideas = load_ideas()
    stories = [i for i in all_ideas if i.parent_id == idea.id and i.state != "done"]
    done_stories = [i for i in all_ideas if i.parent_id == idea.id and i.state == "done"]

    if not stories and not done_stories:
        return _build_story_prompt(idea)

    # Done context
    done_context = ""
    if done_stories:
        done_lines = ["\n## Already Completed Stories"]
        for d in done_stories:
            done_lines.append(f"- {d.id}: {d.title} [DONE]")
        done_lines.append("\nThese are already implemented. Build on them, don't duplicate them.")
        done_context = "\n".join(done_lines)

    # Story sections
    story_sections = ""
    for idx, story in enumerate(stories, 1):
        discussion = ""
        if story.comments:
            discussion = "**Discussion:**\n"
            for c in story.comments:
                label = settings.owner_name if c.author == "owner" else "LLM"
                discussion += f"  - {label}: {c.text}\n"

        short_name = story.id.replace("idea-", "")
        story_sections += (
            f"\n{'=' * 70}\n"
            f"## Story {idx}/{len(stories)}: {story.title}\n"
            f"**ID:** {story.id}\n"
            f"**Category:** {story.category}\n\n"
            f"**Description:** {story.description}\n\n"
            f"{discussion}"
            f"**After completing this story**, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/done\n"
            f"```\n"
            f"If this story fails, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/comment "
            f'-H "Content-Type: application/json" '
            f"-d '{{\"author\": \"claude\", \"text\": \"Execution failed: <describe what went wrong>\"}}'\n"
            f"```\n"
            f"Then move to the next story.\n"
        )

    sections = [
        f"# EPIC: {idea.title}\n",
        f"You are implementing an entire epic for the Technomancer project.",
        f"This epic has **{len(stories)} stories** to implement sequentially.\n",
        f"## Epic Description\n{idea.description}",
        done_context,
        f"\n## Codebase (what already exists — don't duplicate)\n{_load_codebase_summary()}",
        _load_git_history(),
        _load_recent_errors(),
        f"\n## Implementation Process\n"
        f"For EACH story below, follow this exact cycle:\n"
        f"1. Read CLAUDE.md for project conventions\n"
        f"2. Run `python safe_update.py <short-name>` to create a branch\n"
        f"3. Implement the story (code, tests)\n"
        f"4. Run `python validate.py startup` before committing\n"
        f"5. Commit and run `python safe_update.py continue` to test, merge, restart\n"
        f"6. Verify `python bot_service.py status` shows Bot running: True\n"
        f"7. Mark the story done with the curl command provided\n"
        f"8. Move to the next story\n\n"
        f"IMPORTANT:\n"
        f"- Each story gets its OWN safe_update branch and commit\n"
        f"- Do NOT batch multiple stories into one branch\n"
        f"- If a story fails, log it and move to the next one\n"
        f"- Each story should build on what the previous stories created\n"
        f"- When ALL stories are complete, mark the epic done:\n"
        f"```bash\ncurl -X POST http://localhost:8322/api/ideas/{idea.id}/done\n```",
        f"\n# Stories to Implement\n{story_sections}",
    ]
    return "\n".join(s for s in sections if s)


def _build_exploration_prompt(idea: Any) -> str:
    """Build a read-only exploration prompt for the first pass.

    This prompt instructs Claude Code to explore the codebase and understand
    the architecture without making any changes. The session context from
    this pass carries forward into the implementation pass via --resume.
    """
    return (
        f"You are about to implement: {idea.title}\n\n"
        f"Description: {idea.description}\n\n"
        f"Category: {idea.category}\n\n"
        f"IMPORTANT: This is an EXPLORATION pass. Do NOT make any changes.\n"
        f"Your job is to build understanding by:\n"
        f"1. Read CLAUDE.md for project conventions and mandatory workflows\n"
        f"2. Explore the codebase files most relevant to this task\n"
        f"3. Read existing test files to understand test patterns and fixtures\n"
        f"4. Identify which files you will need to create or modify\n"
        f"5. Note any existing utilities or patterns you should reuse\n\n"
        f"After exploring, summarize your findings:\n"
        f"- Which files need to change\n"
        f"- What patterns to follow\n"
        f"- What test approach to use\n"
        f"- Any potential issues to watch for\n\n"
        f"Do NOT edit any files. Do NOT run any commands. Just read and plan."
    )


def _run_exploration_pass(
    binary: Path,
    idea: Any,
    state: ExecutionState,
    env: dict[str, str],
    project_root: Path,
) -> str | None:
    """Run the exploration pass and return the session_id for --resume.

    Args:
        binary: Path to the Claude Code binary
        idea: The idea being executed
        state: ExecutionState for logging
        env: Environment variables
        project_root: Working directory

    Returns:
        session_id string if successful, None if exploration failed
    """
    explore_prompt = _build_exploration_prompt(idea)

    state.log_lines.append("--- Phase 1: Exploration ---")
    state.log_lines.append("Reading CLAUDE.md, exploring relevant files, understanding patterns...")
    _notify_discord(f"[{idea.id}] Phase 1: Exploring codebase before implementation...")

    try:
        proc = subprocess.Popen(
            [
                str(binary), "-p", explore_prompt,
                "--output-format", "json",
                "--allowedTools", "Read,Glob,Grep",
                "--max-turns", "15",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
        )

        start = time.time()
        while proc.poll() is None:
            time.sleep(2)
            elapsed = time.time() - start

            if state.cancelled:
                proc.kill()
                return None

            if elapsed > EXPLORATION_TIMEOUT:
                proc.kill()
                state.log_lines.append(
                    f"Exploration timed out after {EXPLORATION_TIMEOUT}s — "
                    f"proceeding with single-pass execution"
                )
                return None

        stdout_bytes = proc.stdout.read() if proc.stdout else b""
        raw_output = stdout_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            state.log_lines.append("Exploration pass returned non-zero — falling back to single-pass")
            return None

        # Parse JSON to extract session_id
        try:
            result = json.loads(raw_output)
            session_id = result.get("session_id", "")
            if session_id:
                elapsed = time.time() - start
                state.log_lines.append(
                    f"Exploration complete ({elapsed:.0f}s) — "
                    f"session {session_id[:12]}... preserved for implementation"
                )
                _notify_discord(
                    f"[{idea.id}] Exploration complete ({elapsed:.0f}s). "
                    f"Starting implementation with full codebase context..."
                )
                return session_id
            else:
                state.log_lines.append("No session_id in exploration output — falling back")
                return None
        except (json.JSONDecodeError, TypeError):
            state.log_lines.append("Could not parse exploration JSON — falling back")
            return None

    except Exception as e:
        state.log_lines.append(f"Exploration error: {e} — falling back to single-pass")
        return None


def _parse_stream_event(line: str) -> tuple[str, str]:
    """Parse a stream-json line into (event_type, display_text).

    Claude Code --output-format stream-json emits one JSON object per line.
    Key event types:
      - {"type": "assistant", "message": {"content": [{"text": "..."}]}}
      - {"type": "tool_use", "tool": {"name": "Edit"}, ...}
      - {"type": "tool_result", ...}
      - {"type": "result", "result": "...", "session_id": "..."}

    Returns:
        (event_type, display_text) — display_text is empty if not worth showing.
    """
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ("unknown", "")

    event_type = event.get("type", "")

    if event_type == "assistant":
        # Extract text from content blocks
        message = event.get("message", {})
        content = message.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return ("assistant", "\n".join(texts))

    if event_type == "tool_use":
        tool_name = event.get("tool", {}).get("name", event.get("name", "tool"))
        return ("tool_use", f"Using tool: {tool_name}")

    if event_type == "tool_result":
        return ("tool_result", "")

    if event_type == "result":
        result_text = event.get("result", "")
        session_id = event.get("session_id", "")
        cost = event.get("total_cost_usd", 0)
        meta = f"(cost: ${cost:.4f})" if cost else ""
        return ("result", f"Final result {meta}")

    return (event_type, "")


def execute_idea(idea_id: str) -> ExecutionState | None:
    """Start executing an idea with Claude Code.

    Spawns a background thread that runs claude.exe and streams
    stdout line-by-line into the execution state. The dashboard
    polls this state for live updates.

    Args:
        idea_id: The idea to execute

    Returns:
        ExecutionState for tracking, or None if idea not found
    """
    idea = get_idea(idea_id)
    if not idea:
        return None

    # Don't start if already executing
    if idea_id in _active and _active[idea_id].is_alive:
        return _active[idea_id]

    mark_executing(idea_id)

    state = ExecutionState(idea_id=idea_id)
    _active[idea_id] = state

    # Build the rich prompt
    if idea.idea_type == "epic":
        prompt = _build_epic_prompt(idea)
    else:
        prompt = _build_story_prompt(idea)

    # Log prompt size for debugging context window issues
    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4
    logger.info(f"[Executor] {idea_id} prompt: {prompt_chars} chars (~{prompt_tokens_est} tokens)")

    def _run() -> None:
        """Background thread: two-pass Claude Code execution.

        Pass 1 (exploration): Read-only exploration of the codebase to build
        understanding of architecture, patterns, and test structure.

        Pass 2 (implementation): Full implementation with --resume to carry
        forward all context from the exploration pass.

        Falls back to single-pass if exploration fails.
        """
        binary = _find_claude_binary()
        if not binary:
            state.log_lines.append("ERROR: Claude Code binary not found")
            mark_failed(idea_id, "Claude Code binary not found")
            _active.pop(idea_id, None)
            return

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("ANTHROPIC_API_KEY", None)  # Force Pro subscription, not API credits
        project_root = Path(__file__).parent.parent.parent

        _notify_discord(f"Starting execution of {idea_id}: {idea.title}")

        try:
            # --- Phase 1: Exploration ---
            session_id = _run_exploration_pass(
                binary, idea, state, env, project_root
            )

            if state.cancelled:
                state.log_lines.append("CANCELLED by user")
                mark_failed(idea_id, state.log_text)
                _notify_discord(f"Execution of {idea_id} was cancelled.")
                _active.pop(idea_id, None)
                return

            # --- Phase 2: Implementation (streaming) ---
            state.log_lines.append("")
            state.log_lines.append("--- Phase 2: Implementation ---")
            state.log_lines.append(
                f"Prompt: {prompt_chars} chars (~{prompt_tokens_est} tokens)"
            )

            cmd = [
                str(binary), "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
                "--max-turns", "50",
            ]
            if session_id:
                cmd.extend(["--resume", session_id])
                state.log_lines.append(
                    f"Resuming session {session_id[:12]}... with full exploration context"
                )
            else:
                state.log_lines.append("Running single-pass (no exploration context)")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(project_root),
                env=env,
            )
            state.pid = proc.pid
            state.log_lines.append(f"Claude Code started (PID: {proc.pid})")
            state.log_lines.append(f"Working on: {idea.title}")
            logger.info(f"[Executor] {idea_id} started, PID {proc.pid}")

            # Stream stdout line-by-line, parsing JSON events as they arrive
            last_discord_time = 0.0
            final_result = ""

            while True:
                # Check cancellation and timeout before blocking on readline
                if state.cancelled:
                    proc.kill()
                    state.log_lines.append("CANCELLED by user")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(f"Execution of {idea_id} was cancelled.")
                    _active.pop(idea_id, None)
                    return

                if state.elapsed > EXECUTION_TIMEOUT:
                    proc.kill()
                    state.log_lines.append(f"TIMEOUT after {EXECUTION_TIMEOUT}s")
                    mark_failed(idea_id, state.log_text)
                    _notify_discord(
                        f"Execution of {idea_id} timed out after "
                        f"{EXECUTION_TIMEOUT // 60} minutes."
                    )
                    _active.pop(idea_id, None)
                    return

                raw_line = proc.stdout.readline() if proc.stdout else b""
                if not raw_line:
                    if proc.poll() is not None:
                        break  # Process exited and no more output
                    continue

                line_text = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line_text:
                    continue

                # Parse the stream-json event
                event_type, display_text = _parse_stream_event(line_text)

                if event_type == "result":
                    # Capture final result metadata and exit the stream loop
                    try:
                        result_data = json.loads(line_text)
                        final_result = result_data.get("result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    if display_text:
                        state.log_lines.append(display_text)
                    break  # Result event = Claude is done, stop reading

                if display_text:
                    # Log to dashboard
                    state.log_lines.append(display_text)

                    # Rate-limited Discord notification for meaningful events
                    now = time.time()
                    if event_type in ("assistant", "tool_use") and now - last_discord_time >= DISCORD_RATE_LIMIT:
                        # Truncate for Discord (keep it concise)
                        discord_msg = display_text[:300]
                        if len(display_text) > 300:
                            discord_msg += "..."
                        _notify_discord(f"[{idea_id}] {discord_msg}")
                        last_discord_time = now

            # Process finished — ensure cleanup
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            exit_code = proc.returncode or 0

            if exit_code == 0:
                state.log_lines.append(
                    f"Completed successfully (exit code 0, {state.elapsed:.0f}s)"
                )
                mark_done(idea_id, state.log_text[-5000:])
                _notify_discord(
                    f"Idea {idea_id} executed successfully "
                    f"({state.elapsed:.0f}s): {idea.title}"
                )
            else:
                state.log_lines.append(
                    f"Failed (exit code {exit_code}, {state.elapsed:.0f}s)"
                )
                mark_failed(idea_id, state.log_text[-5000:])
                _notify_discord(
                    f"Idea {idea_id} execution failed "
                    f"(exit code {exit_code}): {idea.title}"
                )

        except Exception as e:
            state.log_lines.append(f"ERROR: {e}")
            mark_failed(idea_id, state.log_text)
            _notify_discord(f"Idea {idea_id} execution error: {e}")

        finally:
            _active.pop(idea_id, None)

    thread = threading.Thread(target=_run, daemon=True, name=f"executor-{idea_id}")
    thread.start()
    state.thread = thread
    return state


def cancel_execution(idea_id: str) -> bool:
    """Cancel a running execution.

    Args:
        idea_id: The idea to cancel

    Returns:
        True if cancellation was initiated, False if not executing
    """
    state = _active.get(idea_id)
    if not state:
        return False

    state.cancelled = True

    # Also try to kill the process directly
    if state.pid and state.is_alive:
        try:
            os.kill(state.pid, signal.SIGTERM)
            logger.info(f"[Executor] Sent SIGTERM to PID {state.pid} for {idea_id}")
        except (OSError, ProcessLookupError):
            pass

    return True
