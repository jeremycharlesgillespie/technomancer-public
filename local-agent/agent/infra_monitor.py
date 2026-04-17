"""
Infrastructure Reliability Monitor — Vault backup, GPU health, API key
validation, write-ahead logging for vault writes, and security news alerts.

Runs as a background task checking system health every 30 minutes.
Covers epic stories: idea-065, idea-079, idea-085, idea-090, idea-092, idea-108.
"""

import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

VAULT_PATH = Path(settings.vault_path) / "LLM Memory"
DATA_DIR = Path(__file__).parent.parent / "data"
WAL_DB_PATH = DATA_DIR / "vault_wal.db"
BACKUP_DIR = Path(__file__).parent.parent / "backups"

CHECK_INTERVAL_MINUTES = 30
_local = threading.local()


# ---------------------------------------------------------------------------
# Write-ahead log for vault writes (story idea-090)
# ---------------------------------------------------------------------------

def _get_wal_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "wal_conn", None)
    if conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(WAL_DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.wal_conn = conn
    return conn


def init_wal_db() -> None:
    """Create WAL tables."""
    conn = _get_wal_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vault_wal (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath    TEXT NOT NULL,
            content     TEXT NOT NULL,
            operation   TEXT NOT NULL DEFAULT 'write',
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_vw_status ON vault_wal (status);

        CREATE TABLE IF NOT EXISTS infra_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT NOT NULL,
            message     TEXT NOT NULL,
            severity    TEXT NOT NULL DEFAULT 'info',
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ie_cat ON infra_events (category);
    """)
    conn.commit()


def wal_write(filepath: str, content: str) -> bool:
    """Write content to a file via write-ahead log.

    Logs the write to SQLite first, then attempts the actual file write.
    If the file write fails, the WAL entry remains pending for retry.
    Returns True if the file write succeeded.
    """
    init_wal_db()
    conn = _get_wal_conn()
    now = datetime.now().isoformat()

    # Log to WAL first
    cur = conn.execute(
        "INSERT INTO vault_wal (filepath, content, operation, status, created_at) VALUES (?, ?, 'write', 'pending', ?)",
        (filepath, content, now),
    )
    conn.commit()
    wal_id = cur.lastrowid

    # Attempt actual write
    try:
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        conn.execute(
            "UPDATE vault_wal SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), wal_id),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.execute(
            "UPDATE vault_wal SET status = 'failed' WHERE id = ?", (wal_id,)
        )
        conn.commit()
        # Log quietly — don't send Discord alert for write failures
        # (they'll be retried on next cycle)
        log.warning("WAL write failed for %s: %s", filepath, e)
        return False


def retry_pending_writes() -> int:
    """Retry any pending/failed WAL entries. Returns count of recovered writes."""
    init_wal_db()
    conn = _get_wal_conn()
    pending = conn.execute(
        "SELECT id, filepath, content FROM vault_wal WHERE status IN ('pending', 'failed') ORDER BY id"
    ).fetchall()

    recovered = 0
    for row in pending:
        try:
            target = Path(row["filepath"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(row["content"], encoding="utf-8")
            conn.execute(
                "UPDATE vault_wal SET status = 'completed', completed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"]),
            )
            conn.commit()
            recovered += 1
        except Exception:
            pass
    return recovered


# ---------------------------------------------------------------------------
# Vault backup (story idea-065)
# ---------------------------------------------------------------------------

def backup_vault() -> str | None:
    """Create a timestamped zip backup of the Obsidian vault.

    Saves to the backups/ directory. Returns the backup path or None on failure.
    """
    if not VAULT_PATH.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"vault_backup_{timestamp}"
    backup_path = BACKUP_DIR / backup_name

    try:
        shutil.make_archive(str(backup_path), "zip", str(VAULT_PATH))
        final_path = str(backup_path) + ".zip"

        # Clean old backups (keep last 7)
        backups = sorted(BACKUP_DIR.glob("vault_backup_*.zip"))
        for old in backups[:-7]:
            old.unlink()

        _log_event("vault_backup", f"Backup created: {final_path}", "info")
        return final_path
    except Exception as e:
        _log_event("vault_backup", f"Backup failed: {e}", "high")
        return None


# ---------------------------------------------------------------------------
# Vault change detection (story idea-079)
# ---------------------------------------------------------------------------

_vault_hashes: dict[str, str] = {}


def _hash_file(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def detect_vault_changes() -> list[str]:
    """Detect files that changed in the vault since last check.

    Returns list of changed file paths (relative to vault).
    """
    global _vault_hashes
    changed: list[str] = []

    if not VAULT_PATH.exists():
        return changed

    current: dict[str, str] = {}
    for md_file in VAULT_PATH.rglob("*.md"):
        rel = str(md_file.relative_to(VAULT_PATH))
        h = _hash_file(md_file)
        current[rel] = h
        if rel in _vault_hashes and _vault_hashes[rel] != h:
            changed.append(rel)
        elif rel not in _vault_hashes:
            changed.append(rel)  # new file

    _vault_hashes = current
    return changed


# ---------------------------------------------------------------------------
# GPU health monitoring (story idea-092)
# ---------------------------------------------------------------------------

def check_gpu_health() -> dict[str, Any]:
    """Check GPU health via nvidia-smi. Returns status dict."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip()}

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 5:
            temp = int(parts[1])
            mem_used = int(parts[2])
            mem_total = int(parts[3])
            util = int(parts[4])
            mem_pct = round(mem_used / mem_total * 100, 1) if mem_total else 0

            status = {
                "available": True,
                "name": parts[0],
                "temperature_c": temp,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": mem_pct,
                "utilization_percent": util,
            }

            # Alert on concerning values
            if temp > 85:
                _log_event("gpu", f"GPU temperature critical: {temp}C", "critical")
            elif temp > 75:
                _log_event("gpu", f"GPU temperature high: {temp}C", "high")
            if mem_pct > 90:
                _log_event("gpu", f"GPU memory near full: {mem_pct}% ({mem_used}/{mem_total} MB)", "high")

            return status
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except Exception as e:
        return {"available": False, "error": str(e)}

    return {"available": False, "error": "Unknown error"}


# ---------------------------------------------------------------------------
# API key validation (story idea-085)
# ---------------------------------------------------------------------------

def check_api_keys() -> dict[str, str]:
    """Check which API keys are configured.

    Returns dict of key_name -> status ("ok", "missing", "not_configured").
    Alerts go to the dedicated alerts channel (not llm_chat).
    """
    results: dict[str, str] = {}

    # Discord bot token — if the bot is running, this is valid
    token = settings.discord_bot_token
    results["discord_bot_token"] = "ok" if (token and len(token) > 20) else "missing"

    # Anthropic API key — optional
    api_key = settings.anthropic_api_key
    if api_key and len(api_key) > 10:
        results["anthropic_api_key"] = "ok"
    else:
        results["anthropic_api_key"] = "not_configured"
        _log_event("api_keys", "Anthropic API key not configured — Claude escalation disabled", "info")

    # Discord webhook — optional
    webhook = settings.discord_webhook_url
    results["discord_webhook"] = "ok" if webhook else "not_configured"

    return results


# ---------------------------------------------------------------------------
# Security news scanner (story idea-108)
# ---------------------------------------------------------------------------

SECURITY_KEYWORDS = frozenset({
    "vulnerability", "exploit", "attack", "breach", "hack", "malware",
    "ransomware", "cve", "zero-day", "backdoor", "supply chain attack",
    "rce", "privilege escalation", "authentication bypass",
})

STACK_KEYWORDS = frozenset({
    "nvidia", "gpu", "ollama", "llm", "python", "discord", "windows",
    "anthropic", "claude", "obsidian", "sqlite",
})


def scan_news_for_security(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Scan news articles for security-relevant content.

    Returns list of alerts with title, severity, and recommendation.
    """
    alerts: list[dict[str, str]] = []

    for article in articles:
        title = article.get("title", "").lower()
        summary = article.get("summary", "").lower()
        text = title + " " + summary

        sec_matches = [kw for kw in SECURITY_KEYWORDS if kw in text]
        stack_matches = [kw for kw in STACK_KEYWORDS if kw in text]

        if sec_matches:
            severity = "critical" if stack_matches else "info"
            alert = {
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "severity": severity,
                "security_keywords": ", ".join(sec_matches),
                "stack_relevance": ", ".join(stack_matches) if stack_matches else "general",
            }
            alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _log_event(category: str, message: str, severity: str = "info") -> None:
    """Log an infrastructure event."""
    try:
        init_wal_db()
        conn = _get_wal_conn()
        conn.execute(
            "INSERT INTO infra_events (category, message, severity, created_at) VALUES (?, ?, ?, ?)",
            (category, message[:500], severity, datetime.now().isoformat()),
        )
        conn.commit()
    except Exception:
        log.debug("Failed to log infra event", exc_info=True)

    # Send alerts to the dedicated alerts channel (not llm_chat)
    if severity in ("critical", "high"):
        try:
            from .alerts import send_alert as _send_alert
            _send_alert(message, title=f"Infra: {category}", level="error" if severity == "critical" else "warning")
        except Exception:
            pass




# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_infra_report() -> str:
    """Human-readable infrastructure health report."""
    lines = ["**Infrastructure Health Report**", ""]

    # GPU
    gpu = check_gpu_health()
    if gpu["available"]:
        lines.append(f"**GPU:** {gpu['name']}")
        lines.append(f"  Temp: {gpu['temperature_c']}C | Memory: {gpu['memory_percent']}% | Util: {gpu['utilization_percent']}%")
    else:
        lines.append(f"**GPU:** Unavailable ({gpu.get('error', '?')})")
    lines.append("")

    # API Keys
    keys = check_api_keys()
    lines.append("**API Keys:**")
    for name, status in keys.items():
        icon = "ok" if status == "ok" else "MISSING" if status == "missing" else status
        lines.append(f"  {name}: {icon}")
    lines.append("")

    # Vault
    if VAULT_PATH.exists():
        md_count = len(list(VAULT_PATH.rglob("*.md")))
        lines.append(f"**Vault:** {md_count} markdown files")
    else:
        lines.append("**Vault:** NOT FOUND")
    lines.append("")

    # WAL status
    try:
        init_wal_db()
        conn = _get_wal_conn()
        pending = conn.execute(
            "SELECT COUNT(*) AS cnt FROM vault_wal WHERE status IN ('pending', 'failed')"
        ).fetchone()["cnt"]
        if pending > 0:
            lines.append(f"**WAL:** {pending} pending/failed writes")
        else:
            lines.append("**WAL:** All writes completed")
    except Exception:
        lines.append("**WAL:** Could not check status")
    lines.append("")

    # Recent events
    try:
        init_wal_db()
        conn = _get_wal_conn()
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        events = conn.execute(
            "SELECT category, message, severity, created_at FROM infra_events WHERE created_at >= ? ORDER BY created_at DESC LIMIT 5",
            (since,),
        ).fetchall()
        if events:
            lines.append("**Recent Events (24h):**")
            for e in events:
                ts = e["created_at"][11:19] if len(e["created_at"]) > 11 else e["created_at"]
                lines.append(f"  [{ts}] {e['category']}: {e['message'][:80]}")
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def get_infra_tools() -> list:
    """Get infrastructure monitoring tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "infra_health",
            "Show infrastructure health: GPU status, API keys, vault integrity, and recent events",
            {"type": "object", "properties": {}, "required": []},
            lambda: get_infra_report(),
        ),
        create_tool(
            "vault_backup",
            "Create a zip backup of the Obsidian vault to the backups/ directory",
            {"type": "object", "properties": {}, "required": []},
            lambda: backup_vault() or "Backup failed — check vault path",
        ),
    ]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def infra_monitor_loop(client: Any, channel_name: str) -> None:
    """Background loop that checks infrastructure health every 30 minutes."""
    log.info("[InfraMonitor] Started — checks every %d minutes", CHECK_INTERVAL_MINUTES)

    # Initial delay to let everything start up
    await asyncio.sleep(60)

    # Initial vault hash snapshot
    detect_vault_changes()

    while True:
        try:
            # Check GPU health (alerts only on temp > 75C or memory > 90%)
            gpu = check_gpu_health()
            if gpu["available"]:
                log.debug("[InfraMonitor] GPU: %dC, %s%% mem", gpu["temperature_c"], gpu["memory_percent"])

            # Retry any pending WAL writes
            recovered = retry_pending_writes()
            if recovered > 0:
                log.info("[InfraMonitor] Recovered %d pending vault writes", recovered)

            # Detect vault changes
            changes = detect_vault_changes()
            if changes:
                log.debug("[InfraMonitor] %d vault files changed", len(changes))

            # SLO violation alerts (de-duped for 30min inside alerts.py)
            try:
                from . import alerts
                fired = alerts.check_slo_violations()
                if fired:
                    log.info("[InfraMonitor] SLO alerts fired: %s", fired)
            except Exception:
                log.exception("[InfraMonitor] SLO check failed")

            await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)

        except Exception:
            log.exception("[InfraMonitor] Loop error")
            await asyncio.sleep(300)


def start_infra_monitor(client: Any, channel_name: str) -> None:
    """Start the infrastructure monitor background task."""
    from .task_manager import create_monitored_task

    create_monitored_task(infra_monitor_loop(client, channel_name), "infra-monitor", critical=True)
    log.info("[InfraMonitor] Background task started")
