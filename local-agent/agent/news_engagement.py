"""
News Engagement Tracking — Track reactions, replies, and source performance
for news digest articles.

Stores engagement data in SQLite alongside the existing metrics infrastructure.
Provides reports on which sources and topics get the most engagement so the
news digest can prioritise higher-value content.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "news_engagement.db"


def _utc_since(days: int) -> str:
    """Return a UTC cutoff string in SQLite's `datetime('now')` format.

    `sent_at` / `created_at` default to `datetime('now')` which is UTC and
    renders as `YYYY-MM-DD HH:MM:SS`. Comparing a local-time ISO string
    against that silently drops rows whenever the machine isn't on UTC.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
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
    """Create the engagement tables."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id      TEXT    NOT NULL UNIQUE,
            article_hash    TEXT    NOT NULL,
            title           TEXT    NOT NULL,
            source          TEXT    NOT NULL,
            link            TEXT    NOT NULL DEFAULT '',
            sent_at         TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_na_hash ON news_articles (article_hash);
        CREATE INDEX IF NOT EXISTS idx_na_source ON news_articles (source);

        CREATE TABLE IF NOT EXISTS engagement_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id      INTEGER NOT NULL REFERENCES news_articles(id),
            event_type      TEXT    NOT NULL,
            user_name       TEXT    NOT NULL DEFAULT '',
            detail          TEXT    NOT NULL DEFAULT '',
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ee_article ON engagement_events (article_id);
        CREATE INDEX IF NOT EXISTS idx_ee_type ON engagement_events (event_type);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_article_sent(
    message_id: str,
    article_hash: str,
    title: str,
    source: str,
    link: str = "",
) -> int:
    """Record a news article that was sent to Discord. Returns the article row ID."""
    init_db()
    conn = _get_conn()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO news_articles
               (message_id, article_hash, title, source, link)
               VALUES (?, ?, ?, ?, ?)""",
            (str(message_id), article_hash, title, source, link),
        )
        conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        # If IGNORE fired, look up the existing row
        row = conn.execute(
            "SELECT id FROM news_articles WHERE message_id = ?",
            (str(message_id),),
        ).fetchone()
        return row["id"] if row else 0
    except Exception:
        log.debug("Failed to record article", exc_info=True)
        return 0


def record_reaction(message_id: str, emoji: str, user_name: str = "") -> None:
    """Record a reaction to a news article."""
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM news_articles WHERE message_id = ?",
        (str(message_id),),
    ).fetchone()
    if not row:
        return
    conn.execute(
        "INSERT INTO engagement_events (article_id, event_type, user_name, detail) VALUES (?, ?, ?, ?)",
        (row["id"], "reaction", user_name, emoji),
    )
    conn.commit()


def record_reply(message_id: str, user_name: str = "", snippet: str = "") -> None:
    """Record a reply to a news article."""
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM news_articles WHERE message_id = ?",
        (str(message_id),),
    ).fetchone()
    if not row:
        return
    conn.execute(
        "INSERT INTO engagement_events (article_id, event_type, user_name, detail) VALUES (?, ?, ?, ?)",
        (row["id"], "reply", user_name, snippet[:200]),
    )
    conn.commit()


def is_news_message(message_id: str) -> bool:
    """Check if a Discord message ID is a tracked news article."""
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM news_articles WHERE message_id = ?",
        (str(message_id),),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_source_stats(days: int = 30) -> list[dict[str, Any]]:
    """Get engagement stats per news source."""
    init_db()
    conn = _get_conn()
    since = _utc_since(days)
    rows = conn.execute(
        """SELECT
                na.source,
                COUNT(DISTINCT na.id) AS articles_sent,
                COUNT(ee.id) AS total_engagements,
                SUM(CASE WHEN ee.event_type = 'reaction' THEN 1 ELSE 0 END) AS reactions,
                SUM(CASE WHEN ee.event_type = 'reply' THEN 1 ELSE 0 END) AS replies
           FROM news_articles na
           LEFT JOIN engagement_events ee ON ee.article_id = na.id
           WHERE na.sent_at >= ?
           GROUP BY na.source
           ORDER BY total_engagements DESC""",
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_top_articles(days: int = 30, limit: int = 5) -> list[dict[str, Any]]:
    """Get the most-engaged articles."""
    init_db()
    conn = _get_conn()
    since = _utc_since(days)
    rows = conn.execute(
        """SELECT
                na.title,
                na.source,
                na.link,
                COUNT(ee.id) AS engagements,
                SUM(CASE WHEN ee.event_type = 'reaction' THEN 1 ELSE 0 END) AS reactions,
                SUM(CASE WHEN ee.event_type = 'reply' THEN 1 ELSE 0 END) AS replies
           FROM news_articles na
           LEFT JOIN engagement_events ee ON ee.article_id = na.id
           WHERE na.sent_at >= ?
           GROUP BY na.id
           HAVING engagements > 0
           ORDER BY engagements DESC
           LIMIT ?""",
        (since, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_engagement_report(days: int = 30) -> str:
    """Human-readable engagement report for Discord."""
    init_db()
    conn = _get_conn()
    since = _utc_since(days)

    total_articles = conn.execute(
        "SELECT COUNT(*) AS cnt FROM news_articles WHERE sent_at >= ?", (since,)
    ).fetchone()["cnt"]
    total_events = conn.execute(
        """SELECT COUNT(*) AS cnt FROM engagement_events ee
           JOIN news_articles na ON na.id = ee.article_id
           WHERE na.sent_at >= ?""",
        (since,),
    ).fetchone()["cnt"]

    if total_articles == 0:
        return f"No news articles tracked in the last {days} days."

    engaged_articles = conn.execute(
        """SELECT COUNT(DISTINCT na.id) AS cnt FROM news_articles na
           JOIN engagement_events ee ON ee.article_id = na.id
           WHERE na.sent_at >= ?""",
        (since,),
    ).fetchone()["cnt"]

    rate = (engaged_articles / total_articles * 100) if total_articles else 0

    lines = [f"**News Engagement Report** (last {days} days)", ""]
    lines.append(f"Articles sent: **{total_articles}**")
    lines.append(f"Articles with engagement: **{engaged_articles}** ({rate:.0f}%)")
    lines.append(f"Total engagement events: **{total_events}**")
    lines.append("")

    # Source breakdown
    sources = get_source_stats(days)
    if sources:
        lines.append("**By Source:**")
        for s in sources[:10]:
            eng = s["total_engagements"]
            sent = s["articles_sent"]
            eng_rate = f" ({eng / sent:.1f}/article)" if sent else ""
            lines.append(f"  {s['source']}: {sent} sent, {eng} engagements{eng_rate}")
        lines.append("")

    # Top articles
    top = get_top_articles(days, limit=3)
    if top:
        lines.append("**Most Engaged Articles:**")
        for t in top:
            lines.append(f"  - **{t['title'][:60]}** ({t['source']}) — {t['reactions']}r {t['replies']}c")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def get_news_engagement_tools() -> list:
    """Get news engagement tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "news_engagement_report",
            "Show news article engagement metrics — which sources and articles get the most reactions and replies",
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to report on (default 30)",
                    },
                },
                "required": [],
            },
            function=lambda days=30: get_engagement_report(days),
        ),
    ]
