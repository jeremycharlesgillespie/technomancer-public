"""
Dreaming / Memory Consolidation - Background process that reviews and
consolidates memories during idle hours.

Inspired by Claude Code's Layer 6 "Dreaming" system. This runs as an
async background task and:

1. Reviews recent conversation logs for missed identity facts
2. Deduplicates and merges identity files
3. Updates the user profile based on accumulated patterns
4. Cleans stale or contradictory facts
5. Logs all consolidation activity

Runs during quiet hours (midnight-6am) when the bot is idle.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Obsidian vault location
from .config import settings

VAULT_PATH = settings.llm_memory_path
IDENTITY_DIR = VAULT_PATH / "Permanent" / "identity"
DREAM_LOG = VAULT_PATH / "Permanent" / "dream_log.md"
CONVERSATIONS_DIR = VAULT_PATH / "Conversations"

# Dreaming schedule
DREAM_START_HOUR = 0  # midnight
DREAM_END_HOUR = 6  # 6am
MIN_HOURS_BETWEEN_DREAMS = 4  # Don't dream more often than every 4 hours

# Lock file to prevent concurrent dreaming
DREAM_LOCK = VAULT_PATH / ".dream_lock"


def is_dream_hour() -> bool:
    """Check if current time is within dreaming hours."""
    hour = datetime.now().hour
    return DREAM_START_HOUR <= hour < DREAM_END_HOUR


def acquire_dream_lock() -> bool:
    """Try to acquire the dream lock. Returns True if acquired."""
    if DREAM_LOCK.exists():
        try:
            data = json.loads(DREAM_LOCK.read_text(encoding="utf-8"))
            pid = data.get("pid")
            started = data.get("started", "")

            # Check if the lock holder is still alive
            if pid and _pid_alive(pid):
                return False

            # Check if lock is stale (>2 hours old)
            if started:
                lock_time = datetime.fromisoformat(started)
                if datetime.now() - lock_time < timedelta(hours=2):
                    return False
                # Stale lock — reclaim
        except (json.JSONDecodeError, OSError):
            pass  # Corrupted lock — reclaim

    # Acquire the lock
    DREAM_LOCK.write_text(
        json.dumps({"pid": os.getpid(), "started": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    return True


def release_dream_lock() -> None:
    """Release the dream lock."""
    try:
        DREAM_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_recent_conversation_logs(days: int = 3) -> list[tuple[str, str]]:
    """Read recent conversation log files.

    Returns list of (date_str, content) tuples.
    """
    logs = []
    now = datetime.now()
    for i in range(days):
        day = now - timedelta(days=i)
        log_file = CONVERSATIONS_DIR / f"{day.strftime('%Y-%m-%d')}.md"
        if log_file.exists():
            try:
                content = log_file.read_text(encoding="utf-8")
                logs.append((day.strftime("%Y-%m-%d"), content))
            except OSError:
                continue
    return logs


def _load_all_identity_facts() -> dict[str, str]:
    """Load all identity files."""
    facts = {}
    if not IDENTITY_DIR.exists():
        return facts
    for f in IDENTITY_DIR.glob("*.md"):
        try:
            facts[f.stem] = f.read_text(encoding="utf-8")
        except OSError:
            continue
    return facts


def build_consolidation_prompt(
    conversation_logs: list[tuple[str, str]], identity_facts: dict[str, str]
) -> str:
    """Build the prompt for the dreaming/consolidation LLM call."""
    # Truncate logs to stay within context limits
    logs_text = ""
    for date_str, content in conversation_logs:
        # Take first 3000 chars per day
        logs_text += f"\n--- Conversations from {date_str} ---\n{content[:3000]}\n"

    identity_text = ""
    for category, content in identity_facts.items():
        identity_text += f"\n--- {category} ---\n{content}\n"

    return f"""You are a memory consolidation system performing "dreaming" — reviewing past
conversations and maintaining a clean, accurate identity database.

CURRENT IDENTITY DATABASE:
{identity_text if identity_text else "Empty — no facts stored yet."}

RECENT CONVERSATION LOGS:
{logs_text if logs_text else "No recent conversations available."}

Perform these consolidation tasks and output the results as JSON:

1. **NEW FACTS**: Extract any identity facts from conversations that aren't already in the database.
2. **DUPLICATES**: Identify facts that say the same thing in different words — mark one for removal.
3. **CONTRADICTIONS**: Find facts that conflict with each other or with conversation evidence.
4. **STALE**: Facts that conversations suggest are no longer true.

Output format:
```json
{{
  "new_facts": [
    {{"category": "CATEGORY", "fact": "Extracted fact", "confidence": "high|medium"}}
  ],
  "duplicates": [
    {{"category": "CATEGORY", "remove": "The duplicate fact text to remove", "keep": "The fact to keep"}}
  ],
  "contradictions": [
    {{"category": "CATEGORY", "old_fact": "The outdated fact", "correction": "What it should say"}}
  ],
  "stale": [
    {{"category": "CATEGORY", "fact": "The stale fact to remove"}}
  ],
  "summary": "1-2 sentence summary of what was consolidated"
}}
```

Rules:
- Only use valid categories: preferences, opinions, skills, experiences, goals, habits, personality, relationships
- Be conservative — only flag things you're confident about
- If nothing needs changing, return empty arrays
- Return ONLY the JSON, nothing else"""


def parse_consolidation_response(response: str) -> dict:
    """Parse the LLM's consolidation response."""
    # Try to find JSON in code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return {"new_facts": [], "duplicates": [], "contradictions": [], "stale": [], "summary": "Failed to parse response"}

    try:
        result = json.loads(raw)
        # Validate structure
        for key in ["new_facts", "duplicates", "contradictions", "stale"]:
            if key not in result:
                result[key] = []
        if "summary" not in result:
            result["summary"] = "No summary provided"
        return result
    except (json.JSONDecodeError, TypeError):
        return {"new_facts": [], "duplicates": [], "contradictions": [], "stale": [], "summary": "JSON parse error"}


def apply_consolidation(result: dict) -> dict[str, int]:
    """Apply consolidation results to identity files.

    Returns counts of changes made.
    """
    from .auto_memory import merge_facts_into_files, _load_identity_file, _save_identity_file

    counts = {"added": 0, "removed_duplicates": 0, "fixed_contradictions": 0, "removed_stale": 0}

    # 1. Add new facts
    if result.get("new_facts"):
        updated = merge_facts_into_files(result["new_facts"])
        counts["added"] = len(result["new_facts"])

    # 2. Remove duplicates
    for dup in result.get("duplicates", []):
        category = dup.get("category", "")
        remove_text = dup.get("remove", "")
        if category and remove_text:
            content = _load_identity_file(category)
            if remove_text in content:
                # Remove the line containing this fact
                lines = content.split("\n")
                lines = [l for l in lines if remove_text not in l]
                _save_identity_file(category, "\n".join(lines))
                counts["removed_duplicates"] += 1

    # 3. Fix contradictions (remove old, add corrected)
    for contradiction in result.get("contradictions", []):
        category = contradiction.get("category", "")
        old_fact = contradiction.get("old_fact", "")
        correction = contradiction.get("correction", "")
        if category and old_fact:
            content = _load_identity_file(category)
            if old_fact in content:
                lines = content.split("\n")
                lines = [l for l in lines if old_fact not in l]
                if correction:
                    timestamp = datetime.now().strftime("%Y-%m-%d")
                    lines.append(f"- {correction} [corrected, {timestamp}]")
                _save_identity_file(category, "\n".join(lines))
                counts["fixed_contradictions"] += 1

    # 4. Remove stale facts
    for stale in result.get("stale", []):
        category = stale.get("category", "")
        fact_text = stale.get("fact", "")
        if category and fact_text:
            content = _load_identity_file(category)
            if fact_text in content:
                lines = content.split("\n")
                lines = [l for l in lines if fact_text not in l]
                _save_identity_file(category, "\n".join(lines))
                counts["removed_stale"] += 1

    return counts


def _log_dream(summary: str, counts: dict, duration: float) -> None:
    """Append to dream log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## {timestamp} — Dream Consolidation\n"
        f"- **Duration**: {duration:.1f}s\n"
        f"- **Added**: {counts.get('added', 0)} facts\n"
        f"- **Removed duplicates**: {counts.get('removed_duplicates', 0)}\n"
        f"- **Fixed contradictions**: {counts.get('fixed_contradictions', 0)}\n"
        f"- **Removed stale**: {counts.get('removed_stale', 0)}\n"
        f"- **Summary**: {summary}\n"
    )

    if not DREAM_LOG.exists():
        DREAM_LOG.write_text("# Dream Log\n\nMemory consolidation history.\n", encoding="utf-8")

    with open(DREAM_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


async def run_dream_cycle(agent: Any) -> dict[str, Any]:
    """Run one dreaming/consolidation cycle.

    Args:
        agent: The LLM agent for analysis

    Returns:
        Stats dict with consolidation results
    """
    start = time.time()

    if not acquire_dream_lock():
        return {"status": "skipped", "reason": "lock held by another process"}

    try:
        # Gather data
        logs = _get_recent_conversation_logs(days=3)
        identity = _load_all_identity_facts()

        if not logs and not identity:
            return {"status": "skipped", "reason": "no data to consolidate"}

        # Build and run prompt
        prompt = build_consolidation_prompt(logs, identity)

        try:
            response = await asyncio.to_thread(agent.run, prompt)
        except Exception as e:
            print(f"[Dreaming] LLM call failed: {e}")
            return {"status": "error", "reason": str(e)}

        # Parse and apply
        result = parse_consolidation_response(response)
        counts = apply_consolidation(result)

        duration = time.time() - start
        summary = result.get("summary", "No summary")
        _log_dream(summary, counts, duration)

        total_changes = sum(counts.values())
        print(
            f"[Dreaming] Consolidation complete: {total_changes} changes in {duration:.1f}s "
            f"({summary})"
        )

        return {
            "status": "completed",
            "counts": counts,
            "summary": summary,
            "duration_seconds": round(duration, 1),
        }

    finally:
        release_dream_lock()


async def dream_loop(agent: Any) -> None:
    """Background loop that runs dreaming during idle hours."""
    print(f"[Dreaming] Started — will dream from {DREAM_START_HOUR}:00 to {DREAM_END_HOUR}:00")

    last_dream = datetime.min

    while True:
        try:
            now = datetime.now()

            # Check if we should dream
            if is_dream_hour() and (now - last_dream) >= timedelta(hours=MIN_HOURS_BETWEEN_DREAMS):
                print("[Dreaming] Starting dream cycle...")
                result = await run_dream_cycle(agent)
                last_dream = datetime.now()
                print(f"[Dreaming] Result: {result.get('status', 'unknown')}")

            # Check every 30 minutes
            await asyncio.sleep(1800)

        except Exception as e:
            print(f"[Dreaming] Loop error: {e}")
            await asyncio.sleep(600)


def start_dreaming(agent: Any) -> None:
    """Start the dreaming background task."""
    asyncio.create_task(dream_loop(agent))
    print("[Dreaming] Background task started")
