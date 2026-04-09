"""
Skill Gap Analysis — Identify knowledge gaps from conversations and recommend
targeted learning resources.

Pulls gap clusters from gap_frequency, maps them to dev_learning skill
categories, finds matching articles already published on GitHub Pages or
stored in the Obsidian vault, and tracks progress in SQLite.

Exposes agent tools so the bot can surface personalised recommendations.
"""

import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "skill_gaps.db"

_local = threading.local()


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS skill_gaps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_label TEXT NOT NULL,
            category    TEXT NOT NULL,
            domain      TEXT NOT NULL DEFAULT '',
            gap_count   INTEGER NOT NULL DEFAULT 1,
            first_seen  TEXT NOT NULL DEFAULT '',
            last_seen   TEXT NOT NULL DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'open',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_sg_category ON skill_gaps (category);
        CREATE INDEX IF NOT EXISTS idx_sg_status   ON skill_gaps (status);

        CREATE TABLE IF NOT EXISTS recommendations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            gap_id      INTEGER NOT NULL REFERENCES skill_gaps(id),
            resource_type TEXT NOT NULL,
            title       TEXT NOT NULL,
            url         TEXT NOT NULL DEFAULT '',
            completed   INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_rec_gap ON recommendations (gap_id);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Skill category mapping
# ---------------------------------------------------------------------------

def _build_topic_keywords() -> dict[str, set[str]]:
    """Build keyword sets per category from LEARNING_TOPICS.

    Lazily imported so dev_learning isn't loaded at module level.
    """
    from .dev_learning import LEARNING_TOPICS

    category_kw: dict[str, set[str]] = {}
    for cat, topics in LEARNING_TOPICS.items():
        words: set[str] = set()
        for topic in topics:
            words.update(
                w.lower() for w in re.findall(r"[a-zA-Z]{3,}", topic)
            )
        category_kw[cat] = words
    return category_kw


def classify_category(label: str, queries: list[str]) -> str:
    """Map a gap cluster to the best-matching LEARNING_TOPICS category.

    Falls back to 'best_practices' if no strong match is found.
    """
    topic_kw = _build_topic_keywords()
    text = " ".join([label] + queries).lower()
    text_words = set(re.findall(r"[a-zA-Z]{3,}", text))

    best_cat = "best_practices"
    best_score = 0.0
    for cat, kw_set in topic_kw.items():
        if not kw_set:
            continue
        overlap = len(text_words & kw_set)
        score = overlap / (len(text_words) + 1)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ---------------------------------------------------------------------------
# Resource matching
# ---------------------------------------------------------------------------

def _find_matching_articles(keywords: set[str], limit: int = 3) -> list[dict[str, str]]:
    """Search GitHub Pages articles and vault learning files for matches."""
    results: list[dict[str, str]] = []

    # 1) GitHub Pages articles
    try:
        from .github_pages import list_article_files, get_article_url

        for art in list_article_files():
            title_words = set(re.findall(r"[a-zA-Z]{3,}", art.get("topic", "").lower()))
            if keywords & title_words:
                results.append({
                    "type": "github_pages",
                    "title": art.get("topic", art.get("filename", "")),
                    "url": get_article_url(art["filename"]),
                })
                if len(results) >= limit:
                    return results
    except Exception:
        log.debug("Could not search GitHub Pages articles", exc_info=True)

    # 2) Obsidian vault Learning folder
    learning_dir = settings.llm_memory_path / "Learning"
    if learning_dir.exists():
        for md_file in sorted(learning_dir.glob("*.md"), reverse=True):
            title_words = set(re.findall(r"[a-zA-Z]{3,}", md_file.stem.lower()))
            if keywords & title_words:
                results.append({
                    "type": "vault_article",
                    "title": md_file.stem.replace("-", " ").replace("_", " "),
                    "url": str(md_file),
                })
                if len(results) >= limit:
                    return results

    return results


def _find_matching_topics(keywords: set[str], category: str, limit: int = 3) -> list[str]:
    """Find LEARNING_TOPICS entries that match the gap keywords."""
    from .dev_learning import LEARNING_TOPICS

    topics = LEARNING_TOPICS.get(category, [])
    scored: list[tuple[float, str]] = []
    for topic in topics:
        topic_words = set(re.findall(r"[a-zA-Z]{3,}", topic.lower()))
        overlap = len(keywords & topic_words)
        if overlap > 0:
            scored.append((overlap, topic))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit]]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_skill_gaps(days: int = 30) -> list[dict[str, Any]]:
    """Analyze recent knowledge gaps and produce skill-gap entries with recommendations.

    Pulls gap data from gap_frequency, clusters them, maps to categories,
    finds relevant learning resources, and persists to SQLite.

    Returns the list of gap analysis entries (newest first).
    """
    from .gap_frequency import cluster_gaps, _parse_all_gaps, _scan_conversations_for_gaps

    init_db()

    # Gather raw gaps
    logged = _parse_all_gaps()
    scanned = _scan_conversations_for_gaps(days=days)
    all_gaps = logged + scanned
    if not all_gaps:
        return []

    clusters = cluster_gaps(all_gaps)
    if not clusters:
        return []

    conn = _get_conn()
    entries: list[dict[str, Any]] = []

    for cl in clusters:
        label = cl["label"]
        queries = cl.get("queries", [])
        category = classify_category(label, queries)
        keywords = set(re.findall(r"[a-zA-Z]{3,}", " ".join([label] + queries).lower()))

        # Upsert into skill_gaps
        existing = conn.execute(
            "SELECT id, gap_count FROM skill_gaps WHERE cluster_label = ? AND status != 'completed'",
            (label,),
        ).fetchone()

        if existing:
            gap_id = existing["id"]
            conn.execute(
                "UPDATE skill_gaps SET gap_count = ?, last_seen = ?, category = ?, domain = ? WHERE id = ?",
                (cl["count"], cl.get("last_seen", ""), category, cl.get("domain", ""), gap_id),
            )
        else:
            cur = conn.execute(
                """INSERT INTO skill_gaps (cluster_label, category, domain, gap_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (label, category, cl.get("domain", ""), cl["count"],
                 cl.get("first_seen", ""), cl.get("last_seen", "")),
            )
            gap_id = cur.lastrowid

        # Find recommendations (articles + suggested topics)
        articles = _find_matching_articles(keywords)
        suggested = _find_matching_topics(keywords, category)

        # Persist recommendations (avoid duplicates)
        for art in articles:
            exists = conn.execute(
                "SELECT id FROM recommendations WHERE gap_id = ? AND title = ?",
                (gap_id, art["title"]),
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO recommendations (gap_id, resource_type, title, url) VALUES (?, ?, ?, ?)",
                    (gap_id, art["type"], art["title"], art.get("url", "")),
                )
        for topic in suggested:
            exists = conn.execute(
                "SELECT id FROM recommendations WHERE gap_id = ? AND title = ?",
                (gap_id, topic),
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO recommendations (gap_id, resource_type, title, url) VALUES (?, ?, ?, ?)",
                    (gap_id, "suggested_topic", topic, ""),
                )

        conn.commit()

        entries.append({
            "gap_id": gap_id,
            "label": label,
            "category": category,
            "domain": cl.get("domain", ""),
            "gap_count": cl["count"],
            "first_seen": cl.get("first_seen", ""),
            "last_seen": cl.get("last_seen", ""),
            "articles": articles,
            "suggested_topics": suggested,
        })

    return entries


def mark_recommendation_completed(rec_id: int) -> str:
    """Mark a recommendation as completed."""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT id FROM recommendations WHERE id = ?", (rec_id,)).fetchone()
    if not row:
        return f"Recommendation {rec_id} not found."
    conn.execute(
        "UPDATE recommendations SET completed = 1, completed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), rec_id),
    )
    conn.commit()

    # If all recommendations for a gap are complete, mark the gap completed
    gap_row = conn.execute(
        "SELECT gap_id FROM recommendations WHERE id = ?", (rec_id,)
    ).fetchone()
    if gap_row:
        gap_id = gap_row["gap_id"]
        pending = conn.execute(
            "SELECT COUNT(*) as cnt FROM recommendations WHERE gap_id = ? AND completed = 0",
            (gap_id,),
        ).fetchone()
        if pending and pending["cnt"] == 0:
            conn.execute("UPDATE skill_gaps SET status = 'completed' WHERE id = ?", (gap_id,))
            conn.commit()

    return f"Recommendation {rec_id} marked as completed."


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_skill_gap_report(limit: int = 10) -> str:
    """Human-readable skill gap report for Discord."""
    init_db()
    conn = _get_conn()

    rows = conn.execute(
        """SELECT id, cluster_label, category, domain, gap_count, first_seen, last_seen, status
           FROM skill_gaps
           WHERE status != 'completed'
           ORDER BY gap_count DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    if not rows:
        return "No skill gaps identified yet. Run `skill_gap_analyze` to scan recent conversations."

    lines = [f"**Skill Gap Analysis** ({len(rows)} active gaps)", ""]

    for row in rows:
        gid = row["id"]
        label = row["cluster_label"]
        cat = row["category"]
        count = row["gap_count"]
        domain = row["domain"]

        lines.append(f"**#{gid} {label}** ({cat}, {count}x)")
        if domain:
            lines.append(f"  Domain: {domain}")

        # Recommendations
        recs = conn.execute(
            "SELECT id, resource_type, title, url, completed FROM recommendations WHERE gap_id = ? ORDER BY id",
            (gid,),
        ).fetchall()
        if recs:
            for rec in recs:
                check = "x" if rec["completed"] else " "
                rtype = rec["resource_type"].replace("_", " ")
                title = rec["title"]
                url = rec["url"]
                if url and not url.startswith("/") and not url.startswith("C:"):
                    lines.append(f"  [{check}] ({rtype}) [{title}]({url})")
                else:
                    lines.append(f"  [{check}] ({rtype}) {title}")
        lines.append("")

    # Progress summary
    total = conn.execute("SELECT COUNT(*) as cnt FROM skill_gaps").fetchone()["cnt"]
    completed = conn.execute(
        "SELECT COUNT(*) as cnt FROM skill_gaps WHERE status = 'completed'"
    ).fetchone()["cnt"]
    total_recs = conn.execute("SELECT COUNT(*) as cnt FROM recommendations").fetchone()["cnt"]
    done_recs = conn.execute(
        "SELECT COUNT(*) as cnt FROM recommendations WHERE completed = 1"
    ).fetchone()["cnt"]

    lines.append(f"**Progress:** {completed}/{total} gaps resolved, {done_recs}/{total_recs} resources completed")

    return "\n".join(lines)


def get_progress_summary() -> dict[str, Any]:
    """Return progress stats as a dict."""
    init_db()
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) as cnt FROM skill_gaps").fetchone()["cnt"]
    completed = conn.execute(
        "SELECT COUNT(*) as cnt FROM skill_gaps WHERE status = 'completed'"
    ).fetchone()["cnt"]
    total_recs = conn.execute("SELECT COUNT(*) as cnt FROM recommendations").fetchone()["cnt"]
    done_recs = conn.execute(
        "SELECT COUNT(*) as cnt FROM recommendations WHERE completed = 1"
    ).fetchone()["cnt"]

    cats = conn.execute(
        """SELECT category, COUNT(*) as cnt FROM skill_gaps
           WHERE status != 'completed'
           GROUP BY category ORDER BY cnt DESC"""
    ).fetchall()

    return {
        "total_gaps": total,
        "completed_gaps": completed,
        "open_gaps": total - completed,
        "total_recommendations": total_recs,
        "completed_recommendations": done_recs,
        "categories": {r["category"]: r["cnt"] for r in cats},
    }


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def _tool_analyze(days: int = 30) -> str:
    """Run skill gap analysis and return a summary."""
    try:
        entries = analyze_skill_gaps(days=days)
        if not entries:
            return "No knowledge gaps found in the last {} days.".format(days)
        return get_skill_gap_report()
    except Exception as e:
        log.exception("Skill gap analysis failed")
        return f"Error running skill gap analysis: {e}"


def _tool_complete(recommendation_id: int) -> str:
    """Mark a recommendation as completed."""
    return mark_recommendation_completed(recommendation_id)


def get_skill_gap_tools() -> list:
    """Get skill gap analysis tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            name="skill_gap_analyze",
            description=(
                "Analyze knowledge gaps from recent conversations and recommend "
                "targeted learning resources. Returns prioritised gaps with "
                "matching articles and suggested dev_learning topics."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default 30)",
                    },
                },
                "required": [],
            },
            function=_tool_analyze,
        ),
        create_tool(
            name="skill_gap_report",
            description="Show current skill gap report with recommendations and progress",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: get_skill_gap_report(),
        ),
        create_tool(
            name="skill_gap_complete",
            description="Mark a learning recommendation as completed by its ID number",
            parameters={
                "type": "object",
                "properties": {
                    "recommendation_id": {
                        "type": "integer",
                        "description": "The recommendation ID to mark complete",
                    },
                },
                "required": ["recommendation_id"],
            },
            function=_tool_complete,
        ),
    ]
