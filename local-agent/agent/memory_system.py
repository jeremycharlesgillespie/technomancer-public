"""
Memory System - Obsidian-backed conversation memory with time-based context.

Structure in Obsidian vault:
  LLM Memory/
    Conversations/
      2026-03-12.md    # Raw daily logs
    Context/
      hourly.md        # Rolling 1-hour summary
      daily.md         # Rolling 1-day summary
      weekly.md        # Rolling 1-week summary
    Permanent/
      memories.md      # Important permanent memories
"""

import json
import logging
import re
import shutil
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Number of compaction snapshots to retain in <vault>/Backups/memory/
# before pruning the oldest. Compaction is LLM-driven and destructive;
# 10 snapshots gives ~5 hours of recovery window at the default 30-min cadence.
MAX_COMPACTION_SNAPSHOTS = 10

log = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """A single conversation exchange."""

    timestamp: datetime
    user: str
    message: str
    response: str

    def to_markdown(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"### {ts} - {self.user}\n**Q:** {self.message}\n**A:** {self.response}\n"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "message": self.message,
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationEntry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user=data["user"],
            message=data["message"],
            response=data["response"],
        )


class MemorySystem:
    """
    Manages conversation memory with time-based compaction.

    - Logs all conversations to Obsidian daily notes
    - Maintains rolling summaries for hour/day/week
    - Supports permanent memories
    """

    def __init__(self, vault_path: str, memory_folder: str = "LLM Memory"):
        self.vault_path = Path(vault_path)
        self.memory_root = self.vault_path / memory_folder

        # Create folder structure
        (self.memory_root / "Conversations").mkdir(parents=True, exist_ok=True)
        (self.memory_root / "Context").mkdir(parents=True, exist_ok=True)
        (self.memory_root / "Permanent").mkdir(parents=True, exist_ok=True)

        # In-memory conversation buffer (for quick access)
        self.recent_conversations: deque[ConversationEntry] = deque(maxlen=1000)

        # Load today's conversations into memory
        self._load_today()

        # Compaction state
        self._last_hourly_compact = datetime.now()
        self._last_daily_compact = datetime.now()
        self._last_weekly_compact = datetime.now()

        # Background compaction thread
        self._compaction_thread: Optional[threading.Thread] = None
        self._running = False

        # Health-check state
        self._last_compaction_run: Optional[datetime] = None
        self._last_compaction_error: Optional[str] = None
        self._compaction_error_count: int = 0
        self._compaction_interval_minutes: int = 30

    def _load_today(self):
        """Load today's conversations into memory."""
        today_file = self._daily_log_path()
        if today_file.exists():
            # Parse existing entries from markdown
            content = today_file.read_text(encoding="utf-8")
            # Simple parse - look for ### HH:MM:SS patterns
            entries = re.findall(
                r"### (\d{2}:\d{2}:\d{2}) - (.+?)\n\*\*Q:\*\* (.+?)\n\*\*A:\*\* (.+?)(?=\n###|\n---|\Z)",
                content,
                re.DOTALL,
            )
            today = datetime.now().date()
            for time_str, user, msg, resp in entries:
                ts = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
                self.recent_conversations.append(
                    ConversationEntry(
                        timestamp=ts, user=user.strip(), message=msg.strip(), response=resp.strip()
                    )
                )

    def _daily_log_path(self, date: datetime = None) -> Path:
        """Get path to daily conversation log."""
        date = date or datetime.now()
        return self.memory_root / "Conversations" / f"{date.strftime('%Y-%m-%d')}.md"

    def log_conversation(self, user: str, message: str, response: str):
        """Log a conversation exchange."""
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user=user,
            message=message,
            response=response,
        )

        # Add to in-memory buffer
        self.recent_conversations.append(entry)

        # Append to daily log file
        log_path = self._daily_log_path()

        # Create header if new file
        if not log_path.exists():
            header = f"# Conversations - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            header += (
                f"[[{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}|← Previous]] | "
            )
            header += (
                f"[[{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}|Next →]]\n\n---\n\n"
            )
            log_path.write_text(header, encoding="utf-8")

        # Append entry
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry.to_markdown() + "\n---\n\n")

    def get_context(self, timeframe: str = "hour") -> str:
        """
        Get conversation context for a timeframe.

        Args:
            timeframe: "hour", "day", "week", or "permanent"
        """
        now = datetime.now()

        if timeframe == "permanent":
            return self._get_permanent_memories()

        # Calculate cutoff time
        if timeframe == "hour":
            cutoff = now - timedelta(hours=1)
        elif timeframe == "day":
            cutoff = now - timedelta(days=1)
        elif timeframe == "week":
            cutoff = now - timedelta(weeks=1)
        else:
            cutoff = now - timedelta(hours=1)

        # Filter recent conversations
        relevant = [e for e in self.recent_conversations if e.timestamp >= cutoff]

        if not relevant:
            return f"No conversations in the past {timeframe}."

        # Build context string
        lines = [f"## Conversations from the past {timeframe} ({len(relevant)} exchanges)\n"]
        for entry in relevant[-20:]:  # Limit to last 20 for context window
            lines.append(
                f"**{entry.timestamp.strftime('%H:%M')} {entry.user}:** {entry.message[:100]}"
            )
            lines.append(f"  → {entry.response[:150]}...")

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Get a combined context summary for the LLM."""
        parts = []

        # Recent (last hour) - detailed
        hour_convos = [
            e
            for e in self.recent_conversations
            if e.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        if hour_convos:
            parts.append("## Recent (last hour)")
            for e in hour_convos[-10:]:
                parts.append(f"- {e.user}: {e.message[:80]} → {e.response[:80]}...")

        # Check for compacted summaries
        _hourly_path = self.memory_root / "Context" / "hourly.md"  # Reserved for future use
        daily_path = self.memory_root / "Context" / "daily.md"
        weekly_path = self.memory_root / "Context" / "weekly.md"

        if daily_path.exists():
            parts.append("\n## Earlier today (summary)")
            parts.append(daily_path.read_text(encoding="utf-8")[:500])

        if weekly_path.exists():
            parts.append("\n## This week (summary)")
            parts.append(weekly_path.read_text(encoding="utf-8")[:300])

        # Permanent memories
        perm = self._get_permanent_memories()
        if perm and "No permanent" not in perm:
            parts.append("\n## Permanent memories")
            parts.append(perm[:500])

        return "\n".join(parts) if parts else "No conversation history yet."

    def _get_permanent_memories(self) -> str:
        """Get all permanent memories from every .md file in Permanent/ folder."""
        perm_dir = self.memory_root / "Permanent"
        if not perm_dir.exists():
            return "No permanent memories yet."

        parts = []
        for md_file in sorted(perm_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"--- {md_file.stem} ---\n{content}")
            except Exception:
                continue

        return "\n\n".join(parts) if parts else "No permanent memories yet."

    def save_permanent_memory(
        self, content: str, category: str = "general", replace_category: bool = False
    ) -> str:
        """Save something to permanent memory.

        Args:
            content: What to save
            category: Category/tag for this memory
            replace_category: If True, replace any existing entries with same category
        """
        perm_path = self.memory_root / "Permanent" / "memories.md"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## {timestamp} - {category}\n{content}\n"

        if perm_path.exists():
            existing = perm_path.read_text(encoding="utf-8")

            if replace_category:
                # Remove existing entries with this category
                # Pattern matches: ## YYYY-MM-DD HH:MM - category\n...content...\n## (next section)
                pattern = rf"\n## \d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}} - {re.escape(category)}\n.*?(?=\n## |\Z)"
                existing = re.sub(pattern, "", existing, flags=re.DOTALL)

            perm_path.write_text(existing + entry, encoding="utf-8")
        else:
            header = "# Permanent Memories\n\nImportant information to always remember.\n\n---\n"
            perm_path.write_text(header + entry, encoding="utf-8")

        return f"Saved to permanent memory: {content[:50]}..."

    # =========================================================================
    # COMPACTION — LLM-powered summarization with cost tracking
    # =========================================================================

    def _snapshot_before_compaction(self) -> Optional[Path]:
        """Copy compaction target files into a timestamped backup directory.

        Compaction rewrites hourly.md, daily.md, and (indirectly) memories.md
        from LLM output. A bad summary can silently drop important context
        with no recovery path (the vault is not under git). This snapshots
        the live files into <vault>/Backups/memory/<YYYYMMDD-HHMMSS>/ and
        prunes the backup directory to the MAX_COMPACTION_SNAPSHOTS most
        recent snapshots.

        Returns the snapshot directory path, or None if none of the target
        files existed (nothing to snapshot, e.g. first run on an empty vault).
        """
        targets = [
            self.memory_root / "Context" / "hourly.md",
            self.memory_root / "Context" / "daily.md",
            self.memory_root / "Permanent" / "memories.md",
        ]
        existing = [p for p in targets if p.exists()]
        if not existing:
            return None

        backups_dir = self.vault_path / "Backups" / "memory"
        backups_dir.mkdir(parents=True, exist_ok=True)

        # Collision handling: if the same-second timestamp dir already exists
        # (tests or a fast retry loop), append -1, -2, ... to guarantee a
        # fresh directory per call.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshot_dir = backups_dir / ts
        counter = 1
        while snapshot_dir.exists():
            snapshot_dir = backups_dir / f"{ts}-{counter}"
            counter += 1
        snapshot_dir.mkdir(parents=True)

        for src in existing:
            try:
                shutil.copy2(src, snapshot_dir / src.name)
            except OSError as e:
                log.warning("Snapshot copy failed for %s: %s", src, e)

        # Lexicographic sort matches chronological order because the
        # timestamp prefix is fixed-width YYYYMMDD-HHMMSS.
        snapshots = sorted(
            (d for d in backups_dir.iterdir() if d.is_dir()),
            key=lambda p: p.name,
        )
        while len(snapshots) > MAX_COMPACTION_SNAPSHOTS:
            oldest = snapshots.pop(0)
            try:
                shutil.rmtree(oldest)
            except OSError as e:
                log.warning("Snapshot prune failed for %s: %s", oldest, e)

        return snapshot_dir

    def _llm_summarize(self, summarizer, text: str, tier: str) -> tuple[str, dict]:
        """Run LLM summarization and track cost/time.

        Args:
            summarizer: Callable that takes a prompt string and returns summary
            text: The raw conversation text to summarize
            tier: "hourly", "daily", or "weekly" — affects prompt style

        Returns:
            (summary_text, stats_dict)
        """
        prompts = {
            "hourly": (
                "Summarize these recent conversations in 3-5 bullet points. "
                "Focus on: topics discussed, decisions made, user preferences revealed, "
                "and any unfinished tasks. Be specific — include names, tools, and technical details.\n\n"
            ),
            "daily": (
                "Summarize today's conversations into a structured daily briefing. Include:\n"
                "- **Key topics** discussed\n"
                "- **Decisions made** or preferences expressed\n"
                "- **Tasks** completed or still pending\n"
                "- **Mood/tone** of the day\n"
                "Keep it under 500 words. Be specific.\n\n"
            ),
            "weekly": (
                "Create a weekly summary of these conversation summaries. Include:\n"
                "- **Major themes** of the week\n"
                "- **Recurring topics** or interests\n"
                "- **Progress** on ongoing projects\n"
                "- **Notable changes** in preferences or priorities\n"
                "Keep it under 300 words.\n\n"
            ),
        }

        prompt = prompts.get(tier, prompts["hourly"]) + text

        start_time = time.time()
        input_chars = len(prompt)

        try:
            summary = summarizer(prompt)
        except Exception as e:
            log.error("LLM %s summarization failed: %s", tier, e)
            summary = None

        duration = time.time() - start_time
        output_chars = len(summary) if summary else 0

        stats = {
            "tier": tier,
            "timestamp": datetime.now().isoformat(),
            "input_chars": input_chars,
            "output_chars": output_chars,
            "duration_seconds": round(duration, 1),
            "success": summary is not None,
        }

        return summary, stats

    def _log_compaction_stats(self, stats: dict) -> None:
        """Append compaction stats to the tracking log."""
        log_path = self.memory_root / "Context" / "compaction_stats.json"

        existing = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(stats)
        # Keep last 500 entries
        existing = existing[-500:]
        log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def compact_hourly(self, summarizer=None) -> str:
        """Compact conversations older than 1 hour into hourly summary."""
        self._snapshot_before_compaction()
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Get conversations from 1-2 hours ago
        old_convos = [
            e
            for e in self.recent_conversations
            if cutoff - timedelta(hours=1) <= e.timestamp < cutoff
        ]

        if not old_convos:
            return "Nothing to compact."

        # Build conversation text
        text = "\n".join(
            [f"[{e.timestamp.strftime('%H:%M')}] {e.user}: {e.message}\n→ {e.response}" for e in old_convos]
        )

        # Create summary
        stats = {"tier": "hourly", "success": True, "duration_seconds": 0}
        if summarizer:
            summary, stats = self._llm_summarize(summarizer, text, "hourly")
            if summary is None:
                # Fallback to simple summary on LLM failure
                summary = self._simple_summary(old_convos)
                stats["fallback"] = True
        else:
            summary = self._simple_summary(old_convos)

        self._log_compaction_stats(stats)

        # Write to hourly context
        hourly_path = self.memory_root / "Context" / "hourly.md"
        content = f"# Hourly Context\n\nLast updated: {now.strftime('%Y-%m-%d %H:%M')}\n\n{summary}"
        hourly_path.write_text(content, encoding="utf-8")

        tier_info = f" (LLM, {stats['duration_seconds']}s)" if summarizer and stats.get("success") else ""
        return f"Compacted {len(old_convos)} conversations into hourly summary{tier_info}."

    def compact_daily(self, summarizer=None) -> str:
        """Compact hourly summaries into daily summary."""
        daily_path = self.memory_root / "Context" / "daily.md"
        hourly_path = self.memory_root / "Context" / "hourly.md"

        now = datetime.now()

        # Gather material for daily summary
        parts = []
        if hourly_path.exists():
            parts.append(hourly_path.read_text(encoding="utf-8"))

        today_convos = [
            e for e in self.recent_conversations if e.timestamp.date() == now.date()
        ]

        if today_convos:
            recent_text = "\n".join(
                [f"[{e.timestamp.strftime('%H:%M')}] {e.user}: {e.message[:80]}" for e in today_convos[-20:]]
            )
            parts.append(f"Recent conversations ({len(today_convos)} total):\n{recent_text}")

        if not parts:
            return "Nothing to compact for daily."

        text = "\n\n".join(parts)

        stats = {"tier": "daily", "success": True, "duration_seconds": 0}
        if summarizer:
            summary, stats = self._llm_summarize(summarizer, text, "daily")
            if summary is None:
                summary = self._simple_daily_summary(today_convos, now)
                stats["fallback"] = True
        else:
            summary = self._simple_daily_summary(today_convos, now)

        self._log_compaction_stats(stats)

        header = f"# Daily Context\n\nDate: {now.strftime('%Y-%m-%d')}\n\n"
        daily_path.write_text(header + summary, encoding="utf-8")

        tier_info = f" (LLM, {stats['duration_seconds']}s)" if summarizer and stats.get("success") else ""
        return f"Updated daily context{tier_info}."

    def compact_weekly(self, summarizer=None) -> str:
        """Compact daily summaries into weekly summary."""
        weekly_path = self.memory_root / "Context" / "weekly.md"
        daily_path = self.memory_root / "Context" / "daily.md"

        parts = []
        if daily_path.exists():
            parts.append(daily_path.read_text(encoding="utf-8"))

        # Include recent conversation log files
        conv_dir = self.memory_root / "Conversations"
        now = datetime.now()
        for i in range(7):
            day = now - timedelta(days=i)
            log_file = conv_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if log_file.exists():
                content = log_file.read_text(encoding="utf-8")
                # Take just first 2000 chars per day to keep prompt manageable
                parts.append(f"--- {day.strftime('%A %m/%d')} ---\n{content[:2000]}")

        if not parts:
            return "Nothing to compact for weekly."

        text = "\n\n".join(parts)

        stats = {"tier": "weekly", "success": True, "duration_seconds": 0}
        if summarizer:
            summary, stats = self._llm_summarize(summarizer, text, "weekly")
            if summary is None:
                summary = "Weekly summary unavailable — LLM summarization failed."
                stats["fallback"] = True
        else:
            summary = "Weekly summary requires LLM summarizer."

        self._log_compaction_stats(stats)

        header = f"# Weekly Context\n\nWeek of: {now.strftime('%Y-%m-%d')}\n\n"
        weekly_path.write_text(header + summary, encoding="utf-8")

        tier_info = f" (LLM, {stats['duration_seconds']}s)" if summarizer and stats.get("success") else ""
        return f"Updated weekly context{tier_info}."

    @staticmethod
    def _simple_summary(convos: list) -> str:
        """Fallback summary when LLM is unavailable."""
        summary = (
            f"**{len(convos)} conversations** from "
            f"{convos[0].timestamp.strftime('%H:%M')} to "
            f"{convos[-1].timestamp.strftime('%H:%M')}\n"
        )
        topics = set()
        for e in convos:
            words = e.message.lower().split()[:5]
            topics.update(w for w in words if len(w) > 3)
        summary += f"Topics: {', '.join(list(topics)[:10])}"
        return summary

    @staticmethod
    def _simple_daily_summary(today_convos: list, now: datetime) -> str:
        """Fallback daily summary when LLM is unavailable."""
        summary = f"Total conversations today: {len(today_convos)}\n\n"
        if today_convos:
            users = set(e.user for e in today_convos)
            summary += f"Users: {', '.join(users)}\n\n"
            recent_messages = [e.message for e in today_convos[-10:]]
            summary += "Recent topics:\n" + "\n".join(f"- {m[:60]}" for m in recent_messages)
        return summary

    def purge_old_conversations(self, retention_days: int = 60) -> str:
        """Delete conversation log files older than retention_days.

        Conversation data has already been compacted into daily/weekly
        summaries by the time it's this old, so the raw logs are redundant.

        Args:
            retention_days: Keep conversations newer than this (default 60 days).

        Returns:
            Summary of what was purged.
        """
        conv_dir = self.memory_root / "Conversations"
        if not conv_dir.exists():
            return "No conversations directory."

        cutoff = datetime.now() - timedelta(days=retention_days)
        purged = 0
        for log_file in sorted(conv_dir.glob("*.md")):
            try:
                # Files are named YYYY-MM-DD.md
                file_date_str = log_file.stem  # e.g. "2026-03-12"
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    log_file.unlink()
                    purged += 1
            except (ValueError, OSError):
                continue

        if purged:
            return f"Purged {purged} conversation log(s) older than {retention_days} days."
        return "No conversation logs old enough to purge."

    def get_compaction_stats(self, max_entries: int = 500) -> list[dict]:
        """Read compaction stats log, capping to a rotating window.

        If the stored entries exceed *max_entries*, the file is truncated
        on read so it doesn't grow without bound over months of operation.
        """
        log_path = self.memory_root / "Context" / "compaction_stats.json"
        if not log_path.exists():
            return []
        try:
            entries: list[dict] = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if len(entries) > max_entries:
            entries = entries[-max_entries:]
            try:
                log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            except OSError:
                pass  # non-critical — will be capped next cycle
        return entries

    def start_background_compaction(self, interval_minutes: int = 30, summarizer=None):
        """Start background thread for periodic compaction.

        Args:
            interval_minutes: How often to run compaction
            summarizer: Optional LLM summarizer callable. If provided, compaction
                       uses real LLM summarization instead of keyword extraction.
        """
        if self._running:
            return

        self._running = True
        self._compaction_interval_minutes = interval_minutes

        def compaction_loop():
            hourly_count = 0
            while self._running:
                time.sleep(interval_minutes * 60)
                if not self._running:
                    break
                try:
                    result = self.compact_hourly(summarizer)
                    log.info("Compaction hourly: %s", result)

                    result = self.compact_daily(summarizer)
                    log.info("Compaction daily: %s", result)

                    # Weekly runs every ~6 hours (12 cycles at 30min)
                    hourly_count += 1
                    if hourly_count % 12 == 0:
                        result = self.compact_weekly(summarizer)
                        log.info("Compaction weekly: %s", result)

                        # Purge old conversation logs (already summarized)
                        purge_result = self.purge_old_conversations(retention_days=60)
                        if "Purged" in purge_result:
                            log.info("Compaction purge: %s", purge_result)

                    self._last_compaction_run = datetime.now()
                    self._compaction_error_count = 0
                    self._last_compaction_error = None

                except Exception as e:
                    self._compaction_error_count += 1
                    self._last_compaction_error = str(e)
                    log.error(
                        "Compaction error (attempt %d): %s\n%s",
                        self._compaction_error_count,
                        e,
                        traceback.format_exc(),
                    )
                    # Sleep before retrying — don't exit the loop
                    retry_delay = min(60 * self._compaction_error_count, 300)
                    log.info("Compaction retrying in %ds", retry_delay)
                    time.sleep(retry_delay)

        self._compaction_thread = threading.Thread(target=compaction_loop, daemon=True)
        self._compaction_thread.start()

    def stop_background_compaction(self):
        """Stop background compaction."""
        self._running = False

    def compaction_health(self) -> dict:
        """Return health status of the compaction thread.

        Designed to be called by task_manager or any background monitor.

        Returns a dict with:
            running: bool — whether _running flag is set
            thread_alive: bool — whether the thread object is alive
            last_run: str | None — ISO timestamp of last successful cycle
            error_count: int — consecutive errors since last success
            last_error: str | None — most recent error message
            healthy: bool — overall health verdict
        """
        thread_alive = (
            self._compaction_thread is not None and self._compaction_thread.is_alive()
        )
        last_run_iso = (
            self._last_compaction_run.isoformat() if self._last_compaction_run else None
        )
        # Healthy = thread is alive and not stuck in error loop
        healthy = self._running and thread_alive and self._compaction_error_count < 5
        return {
            "running": self._running,
            "thread_alive": thread_alive,
            "last_run": last_run_iso,
            "error_count": self._compaction_error_count,
            "last_error": self._last_compaction_error,
            "healthy": healthy,
        }


# Global instance
_memory_system: Optional[MemorySystem] = None


def get_memory_system(vault_path: str = None) -> MemorySystem:
    """Get or create the memory system."""
    global _memory_system
    if _memory_system is None:
        if vault_path is None:
            raise ValueError("Vault path required for first initialization")
        _memory_system = MemorySystem(vault_path)
    return _memory_system


def init_memory_system(vault_path: str) -> MemorySystem:
    """Initialize the memory system."""
    global _memory_system
    _memory_system = MemorySystem(vault_path)
    return _memory_system


# Tool functions for the agent
def log_conversation(user: str, message: str, response: str) -> str:
    """Log a conversation (called automatically by bot)."""
    mem = get_memory_system()
    mem.log_conversation(user, message, response)
    return "Logged."


def get_context(timeframe: str = "hour") -> str:
    """
    Get conversation context.

    Args:
        timeframe: "hour", "day", "week", or "permanent"
    """
    return get_memory_system().get_context(timeframe)


def get_full_context() -> str:
    """Get combined context from all timeframes."""
    return get_memory_system().get_full_context()


def remember_permanently(content: str, category: str = "general") -> str:
    """Save something important to permanent memory."""
    return get_memory_system().save_permanent_memory(content, category)


def search_memories(query: str) -> str:
    """Search through conversation history."""
    mem = get_memory_system()
    query_lower = query.lower()

    matches = []
    for entry in mem.recent_conversations:
        if query_lower in entry.message.lower() or query_lower in entry.response.lower():
            matches.append(entry)

    if not matches:
        return f"No conversations found matching '{query}'."

    result = [f"Found {len(matches)} conversations matching '{query}':\n"]
    for e in matches[-10:]:
        result.append(f"- {e.timestamp.strftime('%m/%d %H:%M')} {e.user}: {e.message[:50]}...")

    return "\n".join(result)


def get_full_profile(save_to_file: bool = False) -> str:
    """
    Get EVERYTHING known about the user - full memory dump.

    This returns ALL permanent memories, recent conversations, and context summaries.
    Use save_to_file=True to save to a .txt file (for Discord's character limit).

    Args:
        save_to_file: If True, saves to a file and returns the file path

    Returns:
        Full memory dump as text, or file path if save_to_file=True
    """
    mem = get_memory_system()
    sections = []

    # Header
    sections.append("=" * 60)
    sections.append("FULL MEMORY PROFILE - EVERYTHING I KNOW ABOUT YOU")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("=" * 60)
    sections.append("")

    # 1. PERMANENT MEMORIES (full content, not truncated)
    sections.append("=" * 60)
    sections.append("SECTION 1: PERMANENT MEMORIES")
    sections.append("=" * 60)
    perm_dir = mem.memory_root / "Permanent"
    if perm_dir.exists():
        for md_file in sorted(perm_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(f"--- {md_file.stem} ---")
                    sections.append(content)
                    sections.append("")
            except Exception:
                continue
    else:
        sections.append("No permanent memories saved yet.")
    sections.append("")

    # 2. RECENT CONVERSATIONS (last 50)
    sections.append("=" * 60)
    sections.append("SECTION 2: RECENT CONVERSATIONS (Last 50)")
    sections.append("=" * 60)
    recent = list(mem.recent_conversations)[-50:]
    if recent:
        for entry in recent:
            sections.append(f"\n### {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {entry.user}")
            sections.append(f"**You said:** {entry.message}")
            sections.append(f"**I replied:** {entry.response}")
            sections.append("-" * 40)
    else:
        sections.append("No recent conversations in memory.")
    sections.append("")

    # 3. CONTEXT SUMMARIES
    sections.append("=" * 60)
    sections.append("SECTION 3: CONTEXT SUMMARIES")
    sections.append("=" * 60)

    # Hourly
    hourly_path = mem.memory_root / "Context" / "hourly.md"
    sections.append("\n### Hourly Context:")
    if hourly_path.exists():
        sections.append(hourly_path.read_text(encoding="utf-8"))
    else:
        sections.append("No hourly summary yet.")

    # Daily
    daily_path = mem.memory_root / "Context" / "daily.md"
    sections.append("\n### Daily Context:")
    if daily_path.exists():
        sections.append(daily_path.read_text(encoding="utf-8"))
    else:
        sections.append("No daily summary yet.")

    # Weekly
    weekly_path = mem.memory_root / "Context" / "weekly.md"
    sections.append("\n### Weekly Context:")
    if weekly_path.exists():
        sections.append(weekly_path.read_text(encoding="utf-8"))
    else:
        sections.append("No weekly summary yet.")
    sections.append("")

    # 4. CONVERSATION LOGS (list available days)
    sections.append("=" * 60)
    sections.append("SECTION 4: CONVERSATION LOG FILES")
    sections.append("=" * 60)
    conv_folder = mem.memory_root / "Conversations"
    log_files = sorted(conv_folder.glob("*.md"))
    if log_files:
        sections.append(f"Found {len(log_files)} daily conversation logs:")
        for f in log_files[-10:]:  # Show last 10 days
            size = f.stat().st_size
            sections.append(f"  - {f.name} ({size:,} bytes)")
        if len(log_files) > 10:
            sections.append(f"  ... and {len(log_files) - 10} more")
    else:
        sections.append("No conversation logs yet.")
    sections.append("")

    # 5. STATS
    sections.append("=" * 60)
    sections.append("SECTION 5: MEMORY STATISTICS")
    sections.append("=" * 60)
    total_convos = len(list(mem.recent_conversations))
    users = set(e.user for e in mem.recent_conversations)
    sections.append(f"Total conversations in memory: {total_convos}")
    sections.append(f"Unique users: {', '.join(users) if users else 'None'}")
    sections.append(f"Memory root: {mem.memory_root}")
    sections.append("")

    # Combine all sections
    full_dump = "\n".join(sections)

    # Save to file if requested
    if save_to_file:
        output_path = (
            mem.memory_root / f"memory_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        output_path.write_text(full_dump, encoding="utf-8")
        return f"FILE:{output_path}"

    return full_dump


def get_memory_tools(vault_path: str = None):
    """Get memory tools for the agent."""
    from .core import create_tool

    if vault_path:
        init_memory_system(vault_path)

    return [
        create_tool(
            "get_context",
            "Get conversation context from a timeframe (hour, day, week, or permanent)",
            {
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe: hour, day, week, or permanent",
                        "enum": ["hour", "day", "week", "permanent"],
                    }
                },
                "required": ["timeframe"],
            },
            get_context,
        ),
        create_tool(
            "remember_permanently",
            "Save important information to permanent memory",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "category": {
                        "type": "string",
                        "description": "Category (e.g., user_preferences, facts, instructions)",
                    },
                },
                "required": ["content"],
            },
            remember_permanently,
        ),
        create_tool(
            "search_memories",
            "Search through past conversations",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            search_memories,
        ),
        create_tool(
            "get_full_context",
            "Get a summary of all context (recent + daily + weekly + permanent)",
            {"type": "object", "properties": {}, "required": []},
            get_full_context,
        ),
        create_tool(
            "get_full_profile",
            "Get EVERYTHING known about the user - full memory dump including all permanent memories, recent conversations, and context summaries. Use save_to_file=True if the output is too large for Discord.",
            {
                "type": "object",
                "properties": {
                    "save_to_file": {
                        "type": "boolean",
                        "description": "If true, saves to a file and returns the file path (use for large outputs)",
                    }
                },
                "required": [],
            },
            get_full_profile,
        ),
    ]
