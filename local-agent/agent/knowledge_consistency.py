"""
Knowledge Consistency Monitor — Background audit of facts, memories, and
identity data for staleness, contradictions, and source conflicts.

Runs as a daily background task (3 AM). Does NOT check every query — that
would be too expensive. Instead, audits the knowledge base periodically
and generates a report of issues found.

Checks:
    1. Stale facts — facts_db entries older than 30 days
    2. Duplicate facts — same key with different values across categories
    3. Low-confidence facts — auto_enrichment entries that may be unreliable
    4. Identity staleness — user identity facts with no recent corroboration
    5. Orphaned gaps — knowledge gaps that were never resolved

Reports to: data/consistency_report.json (latest audit)
Alerts: Sends to alerts channel if critical issues found
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent.parent / "data" / "consistency_report.json"

# Thresholds
STALE_DAYS = 30  # Facts older than this are flagged
LOW_CONFIDENCE = 0.7  # Facts below this confidence are flagged
IDENTITY_STALE_DAYS = 60  # Identity facts older than this without refresh


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------

def _check_stale_facts() -> list[dict[str, Any]]:
    """Find facts_db entries older than STALE_DAYS with non-seed sources."""
    try:
        from .facts_db import _get_conn, init_db
        init_db()
        conn = _get_conn()
        cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).isoformat()
        rows = conn.execute(
            """SELECT id, category, key, value, source, confidence, created_at
               FROM facts WHERE source != 'seed' AND created_at < ?
               ORDER BY created_at ASC LIMIT 50""",
            (cutoff,),
        ).fetchall()
        return [
            {
                "issue": "stale_fact",
                "severity": "low",
                "fact_id": r["id"],
                "key": r["key"],
                "category": r["category"],
                "source": r["source"],
                "confidence": r["confidence"],
                "age_days": (datetime.now() - datetime.fromisoformat(r["created_at"])).days,
            }
            for r in rows
        ]
    except Exception as e:
        log.debug(f"[Consistency] stale facts check failed: {e}")
        return []


def _check_low_confidence_facts() -> list[dict[str, Any]]:
    """Find facts with below-threshold confidence scores."""
    try:
        from .facts_db import _get_conn, init_db
        init_db()
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, category, key, value, source, confidence, created_at
               FROM facts WHERE confidence < ? AND source != 'seed'
               ORDER BY confidence ASC LIMIT 30""",
            (LOW_CONFIDENCE,),
        ).fetchall()
        return [
            {
                "issue": "low_confidence",
                "severity": "medium",
                "fact_id": r["id"],
                "key": r["key"],
                "category": r["category"],
                "source": r["source"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]
    except Exception as e:
        log.debug(f"[Consistency] low confidence check failed: {e}")
        return []


def _check_duplicate_keys() -> list[dict[str, Any]]:
    """Find facts with the same key appearing in different categories."""
    try:
        from .facts_db import _get_conn, init_db
        init_db()
        conn = _get_conn()
        rows = conn.execute(
            """SELECT key, GROUP_CONCAT(category || ':' || SUBSTR(value, 1, 60), ' | ') AS entries,
                      COUNT(*) AS cnt
               FROM facts GROUP BY key HAVING cnt > 1
               ORDER BY cnt DESC LIMIT 20""",
        ).fetchall()
        return [
            {
                "issue": "duplicate_key",
                "severity": "medium",
                "key": r["key"],
                "count": r["cnt"],
                "entries": r["entries"],
            }
            for r in rows
        ]
    except Exception as e:
        log.debug(f"[Consistency] duplicate keys check failed: {e}")
        return []


def _check_identity_staleness() -> list[dict[str, Any]]:
    """Check identity files for aged entries without recent corroboration."""
    issues = []
    try:
        from .auto_memory import IDENTITY_DIR, MEMORY_CATEGORIES
        if not IDENTITY_DIR.exists():
            return []

        cutoff = datetime.now() - timedelta(days=IDENTITY_STALE_DAYS)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        for category in MEMORY_CATEGORIES:
            path = IDENTITY_DIR / f"{category}.md"
            if not path.exists():
                continue

            content = path.read_text(encoding="utf-8")
            # Count entries (lines with [confidence, date] format)
            import re
            entries = re.findall(r"\[(?:high|medium|low),\s*(\d{4}-\d{2}-\d{2})\]", content)
            old_entries = [d for d in entries if d < cutoff_str]

            if old_entries and len(old_entries) > len(entries) // 2:
                issues.append({
                    "issue": "identity_stale",
                    "severity": "low",
                    "category": category,
                    "total_entries": len(entries),
                    "stale_entries": len(old_entries),
                    "oldest": min(old_entries) if old_entries else "",
                })
    except Exception as e:
        log.debug(f"[Consistency] identity staleness check failed: {e}")
    return issues


def _check_unresolved_gaps() -> list[dict[str, Any]]:
    """Count knowledge gaps that have been open for more than 7 days."""
    issues = []
    try:
        from .config import settings
        gaps_dir = settings.llm_memory_path / "Permanent" / "Gaps"
        if not gaps_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=7)
        old_open = 0
        for gap_file in gaps_dir.glob("*.md"):
            content = gap_file.read_text(encoding="utf-8")
            if "status: open" in content.lower() or "status: pending" in content.lower():
                # Check creation date from frontmatter
                import re
                m = re.search(r"created:\s*(\d{4}-\d{2}-\d{2})", content)
                if m and m.group(1) < cutoff.strftime("%Y-%m-%d"):
                    old_open += 1

        if old_open > 0:
            issues.append({
                "issue": "unresolved_gaps",
                "severity": "low",
                "count": old_open,
                "detail": f"{old_open} knowledge gaps open > 7 days",
            })
    except Exception as e:
        log.debug(f"[Consistency] gap check failed: {e}")
    return issues


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def run_consistency_audit() -> dict[str, Any]:
    """Run all consistency checks and return the report.

    Returns a dict with issues grouped by type, summary stats, and timestamp.
    """
    log.info("[Consistency] Starting knowledge consistency audit...")

    all_issues: list[dict[str, Any]] = []
    all_issues.extend(_check_stale_facts())
    all_issues.extend(_check_low_confidence_facts())
    all_issues.extend(_check_duplicate_keys())
    all_issues.extend(_check_identity_staleness())
    all_issues.extend(_check_unresolved_gaps())

    # Severity summary
    by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for issue in all_issues:
        sev = issue.get("severity", "low")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    by_type: dict[str, int] = {}
    for issue in all_issues:
        t = issue["issue"]
        by_type[t] = by_type.get(t, 0) + 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_issues": len(all_issues),
        "by_severity": by_severity,
        "by_type": by_type,
        "issues": all_issues[:100],  # Cap at 100 for storage
    }

    # Persist report
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"[Consistency] Could not save report: {e}")

    log.info(
        f"[Consistency] Audit complete: {len(all_issues)} issues "
        f"({by_severity.get('medium', 0)} medium, {by_severity.get('high', 0)} high)"
    )

    # Alert if medium+ issues found
    medium_plus = by_severity.get("medium", 0) + by_severity.get("high", 0) + by_severity.get("critical", 0)
    if medium_plus > 0:
        try:
            from .alerts import send_alert
            summary_lines = [f"**{k}:** {v}" for k, v in by_type.items() if v > 0]
            send_alert(
                "\n".join(summary_lines),
                title=f"Knowledge Consistency: {len(all_issues)} issues ({medium_plus} need attention)",
                level="warning",
            )
        except Exception:
            pass

    return report


def get_consistency_report() -> str:
    """Get the latest consistency report as human-readable text."""
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return "No consistency report available (corrupt file)."
    else:
        # Run a fresh audit
        report = run_consistency_audit()

    lines = [
        f"**Knowledge Consistency Report** ({report['timestamp'][:16]})",
        f"Total issues: **{report['total_issues']}**",
        "",
    ]

    if report["by_type"]:
        lines.append("**By type:**")
        for t, cnt in report["by_type"].items():
            lines.append(f"  {t}: {cnt}")
        lines.append("")

    sev = report["by_severity"]
    lines.append(f"**Severity:** {sev.get('low', 0)} low, {sev.get('medium', 0)} medium, {sev.get('high', 0)} high")
    lines.append("")

    # Show top issues
    for issue in report["issues"][:10]:
        if issue["issue"] == "stale_fact":
            lines.append(f"- Stale fact: `{issue['key']}` ({issue['age_days']}d old, source: {issue['source']})")
        elif issue["issue"] == "low_confidence":
            lines.append(f"- Low confidence: `{issue['key']}` ({issue['confidence']}, source: {issue['source']})")
        elif issue["issue"] == "duplicate_key":
            lines.append(f"- Duplicate key: `{issue['key']}` ({issue['count']} entries)")
        elif issue["issue"] == "identity_stale":
            lines.append(f"- Identity stale: `{issue['category']}` ({issue['stale_entries']}/{issue['total_entries']} entries old)")
        elif issue["issue"] == "unresolved_gaps":
            lines.append(f"- {issue['detail']}")

    if not report["issues"]:
        lines.append("No issues found — knowledge base is consistent.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def start_consistency_monitor(client: Any, channel_name: str) -> None:
    """Start the daily consistency audit as a background task (3 AM)."""

    async def _run_loop() -> None:
        await asyncio.sleep(10)  # Wait for bot startup
        while True:
            now = datetime.now()
            # Next 3 AM
            target = now.replace(hour=3, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            log.info(f"[Consistency] Next audit at {target.strftime('%Y-%m-%d %H:%M')}")
            await asyncio.sleep(wait_seconds)

            try:
                report = run_consistency_audit()
                log.info(f"[Consistency] Daily audit: {report['total_issues']} issues found")
            except Exception as e:
                log.error(f"[Consistency] Audit failed: {e}")

    asyncio.create_task(_run_loop())


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

def get_consistency_tools() -> list:
    """Return consistency monitoring tools for the LLM agent."""
    from .core import create_tool

    return [
        create_tool(
            "knowledge_consistency",
            "Check knowledge base for stale facts, contradictions, and inconsistencies",
            parameters={
                "type": "object",
                "properties": {
                    "refresh": {
                        "type": "boolean",
                        "description": "Run a fresh audit instead of showing cached report (default false)",
                    },
                },
                "required": [],
            },
            function=lambda refresh=False: (
                get_consistency_report()
                if not refresh
                else _format_fresh_report()
            ),
        ),
    ]


def _format_fresh_report() -> str:
    """Run a fresh audit and return formatted report."""
    run_consistency_audit()
    return get_consistency_report()
