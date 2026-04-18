"""
SQLite-backed facts database for common reference queries.

Provides a persistent repository of structured reference data (definitions,
geographical locations, time zone mappings) that can be queried before
invoking the LLM, reducing token consumption and improving factual accuracy.

Database lives at ``local-agent/data/facts.db``.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Tool

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "facts.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (created on first use)."""
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
    """Create the facts table and FTS index if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT    NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'seed',
            confidence  REAL    NOT NULL DEFAULT 1.0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(category, key)
        )
    """)
    # Migrate existing databases that lack the confidence column
    try:
        conn.execute("SELECT confidence FROM facts LIMIT 1")
    except Exception:
        conn.execute("ALTER TABLE facts ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_category
        ON facts (category)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_key
        ON facts (key)
    """)
    # FTS5 virtual table for fuzzy text search on key and value
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            key, value, category,
            content='facts',
            content_rowid='id'
        )
    """)
    conn.commit()


def _sync_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the FTS index from the facts table."""
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
    conn.commit()


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_DATA: list[dict[str, str]] = [
    # Definitions - food
    {"category": "definition", "key": "spaghetti", "value": "A long, thin, cylindrical pasta of Italian origin, typically made from durum wheat semolina and water. It is a staple of traditional Italian cuisine."},
    {"category": "definition", "key": "pasta", "value": "A type of food made from an unleavened dough of wheat flour mixed with water or eggs, formed into various shapes, and cooked by boiling. Common types include spaghetti, penne, fusilli, and lasagna."},
    {"category": "definition", "key": "pizza", "value": "A dish of Italian origin consisting of a round, flat base of leavened wheat-based dough topped with tomato sauce, cheese, and various other ingredients, baked at high temperature."},
    {"category": "definition", "key": "sushi", "value": "A Japanese dish consisting of prepared vinegared rice combined with varied ingredients such as seafood, vegetables, and occasionally tropical fruits, often wrapped in seaweed (nori)."},
    {"category": "definition", "key": "bread", "value": "A staple food prepared from a dough of flour (usually wheat) and water, usually by baking. It is one of the oldest prepared foods, dating back thousands of years."},
    {"category": "definition", "key": "rice", "value": "The seed of the grass species Oryza sativa (Asian rice) or Oryza glaberrima (African rice). It is the most widely consumed staple food for over half the world's population."},
    {"category": "definition", "key": "chocolate", "value": "A food product made from roasted and ground cacao seeds (Theobroma cacao). It can be in liquid, paste, or block form, used as a flavoring ingredient or eaten as confection."},
    {"category": "definition", "key": "coffee", "value": "A brewed drink prepared from roasted coffee beans, the seeds of berries from the Coffea plant. It is one of the most popular beverages worldwide, valued for its stimulant effects due to caffeine."},
    {"category": "definition", "key": "tea", "value": "An aromatic beverage prepared by pouring hot or boiling water over cured or fresh leaves of Camellia sinensis. After water, it is the most widely consumed drink in the world."},
    # Definitions - technology
    {"category": "definition", "key": "algorithm", "value": "A finite sequence of well-defined instructions, typically used to solve a class of specific problems or to perform a computation."},
    {"category": "definition", "key": "API", "value": "Application Programming Interface — a set of defined rules and protocols that enable different software applications to communicate with each other."},
    {"category": "definition", "key": "database", "value": "An organized collection of structured information or data, typically stored electronically in a computer system and managed by a database management system (DBMS)."},
    {"category": "definition", "key": "machine learning", "value": "A subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed, using statistical techniques to identify patterns in data."},
    {"category": "definition", "key": "blockchain", "value": "A distributed, immutable ledger technology that records transactions across many computers so that any involved record cannot be altered retroactively."},
    {"category": "definition", "key": "cloud computing", "value": "The on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user, delivered over the internet."},
    {"category": "definition", "key": "LLM", "value": "Large Language Model — a type of artificial intelligence model trained on vast amounts of text data that can generate, understand, and manipulate human language."},
    {"category": "definition", "key": "Python", "value": "A high-level, general-purpose programming language known for its readability and versatility. Created by Guido van Rossum and first released in 1991."},
    {"category": "definition", "key": "JavaScript", "value": "A high-level, interpreted programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. It enables interactive web pages."},
    {"category": "definition", "key": "SQL", "value": "Structured Query Language — a domain-specific language used for managing and manipulating relational databases. It can insert, query, update, and delete data."},
    # Definitions - science
    {"category": "definition", "key": "photosynthesis", "value": "The biological process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose, using carbon dioxide and water, and releasing oxygen."},
    {"category": "definition", "key": "gravity", "value": "A fundamental force of nature that attracts any two objects with mass. On Earth, it gives weight to physical objects and causes them to fall toward the ground at approximately 9.8 m/s²."},
    {"category": "definition", "key": "DNA", "value": "Deoxyribonucleic acid — the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms."},
    {"category": "definition", "key": "atom", "value": "The smallest unit of ordinary matter that forms a chemical element. Every solid, liquid, gas, and plasma is composed of atoms. Atoms consist of protons, neutrons, and electrons."},
    {"category": "definition", "key": "evolution", "value": "The process of change in all forms of life over successive generations through variations in heritable characteristics, driven by natural selection, genetic drift, and mutation."},
    # Geography - countries and capitals
    {"category": "geography", "key": "United States capital", "value": "Washington, D.C."},
    {"category": "geography", "key": "United Kingdom capital", "value": "London"},
    {"category": "geography", "key": "France capital", "value": "Paris"},
    {"category": "geography", "key": "Germany capital", "value": "Berlin"},
    {"category": "geography", "key": "Japan capital", "value": "Tokyo"},
    {"category": "geography", "key": "China capital", "value": "Beijing"},
    {"category": "geography", "key": "India capital", "value": "New Delhi"},
    {"category": "geography", "key": "Brazil capital", "value": "Brasília"},
    {"category": "geography", "key": "Australia capital", "value": "Canberra"},
    {"category": "geography", "key": "Canada capital", "value": "Ottawa"},
    {"category": "geography", "key": "Russia capital", "value": "Moscow"},
    {"category": "geography", "key": "Italy capital", "value": "Rome"},
    {"category": "geography", "key": "Spain capital", "value": "Madrid"},
    {"category": "geography", "key": "Mexico capital", "value": "Mexico City"},
    {"category": "geography", "key": "South Korea capital", "value": "Seoul"},
    {"category": "geography", "key": "Egypt capital", "value": "Cairo"},
    {"category": "geography", "key": "South Africa capital", "value": "Pretoria (executive), Cape Town (legislative), Bloemfontein (judicial)"},
    {"category": "geography", "key": "Argentina capital", "value": "Buenos Aires"},
    {"category": "geography", "key": "Turkey capital", "value": "Ankara"},
    {"category": "geography", "key": "Nigeria capital", "value": "Abuja"},
    # Geography - continents
    {"category": "geography", "key": "continents", "value": "Africa, Antarctica, Asia, Europe, North America, Oceania (Australia), South America"},
    {"category": "geography", "key": "largest continent", "value": "Asia — approximately 44.58 million km² (17.21 million mi²)"},
    {"category": "geography", "key": "smallest continent", "value": "Australia/Oceania — approximately 8.53 million km² (3.29 million mi²)"},
    {"category": "geography", "key": "largest ocean", "value": "Pacific Ocean — approximately 165.25 million km² (63.8 million mi²)"},
    {"category": "geography", "key": "longest river", "value": "The Nile (approximately 6,650 km / 4,130 mi) or the Amazon (approximately 6,400 km / 3,976 mi), depending on measurement methodology."},
    {"category": "geography", "key": "highest mountain", "value": "Mount Everest — 8,849 meters (29,032 ft) above sea level, located in the Himalayas on the border of Nepal and Tibet."},
    # Time zones
    {"category": "timezone", "key": "EST", "value": "Eastern Standard Time — UTC-5. Used by: New York, Washington D.C., Miami, Atlanta, Toronto."},
    {"category": "timezone", "key": "EDT", "value": "Eastern Daylight Time — UTC-4. Daylight saving time for the Eastern time zone (March–November)."},
    {"category": "timezone", "key": "CST", "value": "Central Standard Time — UTC-6. Used by: Chicago, Houston, Dallas, Mexico City."},
    {"category": "timezone", "key": "CDT", "value": "Central Daylight Time — UTC-5. Daylight saving time for the Central time zone (March–November)."},
    {"category": "timezone", "key": "MST", "value": "Mountain Standard Time — UTC-7. Used by: Denver, Phoenix, Salt Lake City."},
    {"category": "timezone", "key": "PST", "value": "Pacific Standard Time — UTC-8. Used by: Los Angeles, San Francisco, Seattle, Vancouver."},
    {"category": "timezone", "key": "PDT", "value": "Pacific Daylight Time — UTC-7. Daylight saving time for the Pacific time zone (March–November)."},
    {"category": "timezone", "key": "GMT", "value": "Greenwich Mean Time — UTC+0. Used by: London (winter), Lisbon, Accra."},
    {"category": "timezone", "key": "BST", "value": "British Summer Time — UTC+1. Daylight saving time for the UK (March–October)."},
    {"category": "timezone", "key": "CET", "value": "Central European Time — UTC+1. Used by: Berlin, Paris, Rome, Madrid."},
    {"category": "timezone", "key": "CEST", "value": "Central European Summer Time — UTC+2. Daylight saving time for Central Europe (March–October)."},
    {"category": "timezone", "key": "JST", "value": "Japan Standard Time — UTC+9. Used by: Tokyo, Osaka. Japan does not observe daylight saving time."},
    {"category": "timezone", "key": "IST", "value": "Indian Standard Time — UTC+5:30. Used throughout India. India does not observe daylight saving time."},
    {"category": "timezone", "key": "CST (China)", "value": "China Standard Time — UTC+8. Used throughout China. China does not observe daylight saving time."},
    {"category": "timezone", "key": "AEST", "value": "Australian Eastern Standard Time — UTC+10. Used by: Sydney, Melbourne, Brisbane."},
    {"category": "timezone", "key": "UTC", "value": "Coordinated Universal Time — the primary time standard by which the world regulates clocks and time. Successor to GMT."},
    # Unit conversions
    {"category": "conversion", "key": "miles to kilometers", "value": "1 mile = 1.60934 kilometers"},
    {"category": "conversion", "key": "kilometers to miles", "value": "1 kilometer = 0.621371 miles"},
    {"category": "conversion", "key": "pounds to kilograms", "value": "1 pound = 0.453592 kilograms"},
    {"category": "conversion", "key": "kilograms to pounds", "value": "1 kilogram = 2.20462 pounds"},
    {"category": "conversion", "key": "fahrenheit to celsius", "value": "°C = (°F − 32) × 5/9. Common: 32°F=0°C, 72°F=22°C, 100°F=38°C, 212°F=100°C"},
    {"category": "conversion", "key": "celsius to fahrenheit", "value": "°F = (°C × 9/5) + 32. Common: 0°C=32°F, 20°C=68°F, 37°C=98.6°F, 100°C=212°F"},
    {"category": "conversion", "key": "inches to centimeters", "value": "1 inch = 2.54 centimeters"},
    {"category": "conversion", "key": "feet to meters", "value": "1 foot = 0.3048 meters"},
    {"category": "conversion", "key": "gallons to liters", "value": "1 US gallon = 3.78541 liters"},
    {"category": "conversion", "key": "ounces to grams", "value": "1 ounce = 28.3495 grams"},
]


def seed_db() -> int:
    """Insert seed data, skipping rows that already exist. Returns count inserted."""
    conn = _get_conn()
    init_db()
    inserted = 0
    for row in _SEED_DATA:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO facts (category, key, value, source) VALUES (?, ?, ?, ?)",
                (row["category"], row["key"], row["value"], "seed"),
            )
            if conn.total_changes:
                inserted += 1
        except Exception:
            log.exception("Failed to seed fact: %s", row["key"])
    conn.commit()
    _sync_fts(conn)
    return inserted


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def lookup_fact(query: str, category: str | None = None) -> list[dict[str, Any]]:
    """
    Look up facts matching a query.

    First tries exact key match, then falls back to FTS search.
    Returns list of {category, key, value, source} dicts.
    """
    conn = _get_conn()
    init_db()

    # 1. Exact key match (case-insensitive)
    if category:
        rows = conn.execute(
            "SELECT category, key, value, source, confidence FROM facts WHERE LOWER(key) = LOWER(?) AND category = ?",
            (query, category),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT category, key, value, source, confidence FROM facts WHERE LOWER(key) = LOWER(?)",
            (query,),
        ).fetchall()

    if rows:
        return [dict(r) for r in rows]

    # 2. FTS search (join back to facts table for confidence + source)
    fts_query = query.replace('"', '""')
    try:
        if category:
            rows = conn.execute(
                """SELECT f.category, f.key, f.value, f.source, f.confidence
                   FROM facts_fts fts
                   JOIN facts f ON f.rowid = fts.rowid
                   WHERE fts.facts_fts MATCH ? AND fts.category = ? LIMIT 10""",
                (f'"{fts_query}"', category),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT f.category, f.key, f.value, f.source, f.confidence
                   FROM facts_fts fts
                   JOIN facts f ON f.rowid = fts.rowid
                   WHERE fts.facts_fts MATCH ? LIMIT 10""",
                (f'"{fts_query}"',),
            ).fetchall()
        if rows:
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        pass

    # 3. LIKE fallback for partial matches
    like_pattern = f"%{query}%"
    if category:
        rows = conn.execute(
            "SELECT category, key, value, source, confidence FROM facts WHERE (LOWER(key) LIKE LOWER(?) OR LOWER(value) LIKE LOWER(?)) AND category = ? LIMIT 10",
            (like_pattern, like_pattern, category),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT category, key, value, source, confidence FROM facts WHERE LOWER(key) LIKE LOWER(?) OR LOWER(value) LIKE LOWER(?) LIMIT 10",
            (like_pattern, like_pattern),
        ).fetchall()
    return [dict(r) for r in rows]


def source_confidence(source: str) -> float:
    """Return a default confidence score for a given source type."""
    scores = {
        "seed": 1.0,
        "user": 0.95,
        "wikipedia": 0.9,
        "auto_enrichment": 0.8,
        "web_search": 0.6,
    }
    return scores.get(source, 0.7)


def add_fact(
    category: str,
    key: str,
    value: str,
    source: str = "user",
    confidence: float | None = None,
) -> str:
    """Add or update a fact in the database.

    If ``confidence`` is not provided, a default is assigned based on the
    source type (seed=1.0, wikipedia=0.9, web_search=0.6, etc.).
    """
    conn = _get_conn()
    init_db()
    conf = confidence if confidence is not None else source_confidence(source)
    conn.execute(
        "INSERT OR REPLACE INTO facts (category, key, value, source, confidence) VALUES (?, ?, ?, ?, ?)",
        (category, key, value, source, round(conf, 2)),
    )
    conn.commit()
    _sync_fts(conn)
    return f"Saved fact: [{category}] {key} (confidence: {conf:.0%})"


def get_categories() -> list[str]:
    """Return all distinct categories in the facts database."""
    conn = _get_conn()
    init_db()
    rows = conn.execute("SELECT DISTINCT category FROM facts ORDER BY category").fetchall()
    return [r["category"] for r in rows]


def get_stats() -> dict[str, Any]:
    """Return statistics about the facts database."""
    conn = _get_conn()
    init_db()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM facts").fetchone()["cnt"]
    cats = conn.execute(
        "SELECT category, COUNT(*) AS cnt FROM facts GROUP BY category ORDER BY category"
    ).fetchall()
    return {
        "total_facts": total,
        "categories": {r["category"]: r["cnt"] for r in cats},
    }


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------


def _tool_lookup_fact(query: str, category: str = "") -> str:
    """Tool wrapper: look up a fact."""
    results = lookup_fact(query, category=category or None)
    if not results:
        return f"No facts found for '{query}'."
    lines = []
    for r in results:
        lines.append(f"[{r.get('category', '?')}] **{r.get('key', '?')}**: {r.get('value', '?')}")
    return "\n".join(lines)


def _tool_add_fact(category: str, key: str, value: str) -> str:
    """Tool wrapper: add a fact."""
    return add_fact(category, key, value, source="user")


def _tool_facts_stats() -> str:
    """Tool wrapper: get facts stats."""
    stats = get_stats()
    return json.dumps(stats, indent=2)


def get_facts_tools() -> list["Tool"]:
    """Get tools for the facts database."""
    from .core import create_tool

    return [
        create_tool(
            name="lookup_fact",
            description=(
                "Look up a fact from the local reference database. Use this BEFORE asking "
                "the LLM for common definitions, geography, time zones, or unit conversions. "
                "Returns exact matches first, then fuzzy search results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The term or topic to look up (e.g. 'spaghetti', 'Japan capital', 'EST')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter: definition, geography, timezone, conversion",
                    },
                },
                "required": ["query"],
            },
            function=_tool_lookup_fact,
        ),
        create_tool(
            name="add_fact",
            description="Add a new fact to the local reference database for future lookups.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Fact category: definition, geography, timezone, conversion, or custom",
                    },
                    "key": {
                        "type": "string",
                        "description": "The lookup key (e.g. 'spaghetti', 'France capital')",
                    },
                    "value": {
                        "type": "string",
                        "description": "The factual content",
                    },
                },
                "required": ["category", "key", "value"],
            },
            function=_tool_add_fact,
        ),
        create_tool(
            name="facts_stats",
            description="Show statistics about the local facts reference database.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=_tool_facts_stats,
        ),
    ]
