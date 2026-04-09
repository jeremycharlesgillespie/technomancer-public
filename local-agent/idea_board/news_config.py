"""
News Configuration — JSON-backed preferences for news digest filtering.

Manages RSS feeds, topic likes/dislikes, and schedule settings.
Provides a Flask Blueprint with API routes and a web UI for configuration.

Storage: news_config.json (same directory as ideas.json)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

CONFIG_FILE: Path = Path(__file__).parent / "news_config.json"
_lock = threading.Lock()

# Default feeds — migrated from news_digest.py hardcoded list
DEFAULT_FEEDS: list[dict[str, Any]] = [
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "general", "enabled": True},
    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "category": "general", "enabled": True},
    {"name": "Hacker News", "url": "https://hnrss.org/frontpage", "category": "general", "enabled": True},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "general", "enabled": True},
    {"name": "Dev.to", "url": "https://dev.to/feed", "category": "dev-tools", "enabled": True},
    {"name": "InfoQ", "url": "https://feed.infoq.com/", "category": "dev-tools", "enabled": True},
    {"name": "The New Stack", "url": "https://thenewstack.io/feed/", "category": "dev-tools", "enabled": True},
    {"name": "Real Python", "url": "https://realpython.com/atom.xml", "category": "python", "enabled": True},
    {"name": "Python Insider", "url": "https://blog.python.org/feeds/posts/default?alt=rss", "category": "python", "enabled": True},
    {"name": "AWS Blog", "url": "https://aws.amazon.com/blogs/aws/feed/", "category": "cloud", "enabled": True},
    {"name": "AWS Architecture", "url": "https://aws.amazon.com/blogs/architecture/feed/", "category": "cloud", "enabled": True},
    {"name": "MIT Tech Review AI", "url": "https://www.technologyreview.com/feed/", "category": "ai", "enabled": True},
    {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml", "category": "ai", "enabled": True},
]

FEED_CATEGORIES: list[str] = ["general", "ai", "cloud", "dev-tools", "python", "security", "other"]


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class NewsConfig:
    """News digest configuration with feeds, preferences, and schedule."""

    feeds: list[dict[str, Any]] = field(default_factory=lambda: [dict(f) for f in DEFAULT_FEEDS])
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    start_hour: int = 9
    end_hour: int = 21
    max_articles_per_hour: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "feeds": self.feeds,
            "likes": self.likes,
            "dislikes": self.dislikes,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "max_articles_per_hour": self.max_articles_per_hour,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NewsConfig:
        return cls(
            feeds=data.get("feeds", [dict(f) for f in DEFAULT_FEEDS]),
            likes=data.get("likes", []),
            dislikes=data.get("dislikes", []),
            start_hour=data.get("start_hour", 9),
            end_hour=data.get("end_hour", 21),
            max_articles_per_hour=data.get("max_articles_per_hour", 1),
        )


# ============================================================================
# JSON PERSISTENCE
# ============================================================================

def load_news_config() -> NewsConfig:
    """Load news config from JSON file. Thread-safe."""
    with _lock:
        if not CONFIG_FILE.exists():
            return NewsConfig()
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return NewsConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading news config: {e}")
            return NewsConfig()


def save_news_config(config: NewsConfig) -> None:
    """Save news config to JSON file. Thread-safe."""
    with _lock:
        CONFIG_FILE.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ============================================================================
# CONVENIENCE ACCESSORS (used by news_digest.py)
# ============================================================================

def get_enabled_feeds() -> list[tuple[str, str]]:
    """Return list of (name, url) tuples for enabled feeds only."""
    config = load_news_config()
    return [(f["name"], f["url"]) for f in config.feeds if f.get("enabled", True)]


def get_topic_preferences() -> tuple[list[str], list[str]]:
    """Return (likes, dislikes) topic lists."""
    config = load_news_config()
    return config.likes, config.dislikes


def get_schedule() -> tuple[int, int]:
    """Return (start_hour, end_hour) for digest schedule."""
    config = load_news_config()
    return config.start_hour, config.end_hour


# ============================================================================
# FLASK BLUEPRINT — API ROUTES
# ============================================================================

news_bp = Blueprint("news", __name__)


@news_bp.route("/api/news/config")
def api_get_config() -> tuple:
    """GET /api/news/config — return full news configuration."""
    config = load_news_config()
    return jsonify(config.to_dict())


@news_bp.route("/api/news/config", methods=["PUT"])
def api_update_config() -> tuple:
    """PUT /api/news/config — update schedule settings."""
    data = request.get_json(silent=True) or {}
    config = load_news_config()
    if "start_hour" in data:
        config.start_hour = max(0, min(23, int(data["start_hour"])))
    if "end_hour" in data:
        config.end_hour = max(1, min(24, int(data["end_hour"])))
    if "max_articles_per_hour" in data:
        config.max_articles_per_hour = max(1, min(10, int(data["max_articles_per_hour"])))
    save_news_config(config)
    return jsonify(config.to_dict())


@news_bp.route("/api/news/feeds", methods=["POST"])
def api_add_feed() -> tuple:
    """POST /api/news/feeds — add a new RSS feed."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    url = data.get("url", "").strip()
    category = data.get("category", "other").strip()
    if not name or not url:
        return jsonify({"error": "Need name and url"}), 400
    config = load_news_config()
    # Check for duplicate URL
    if any(f["url"] == url for f in config.feeds):
        return jsonify({"error": "Feed URL already exists"}), 409
    config.feeds.append({"name": name, "url": url, "category": category, "enabled": True})
    save_news_config(config)
    return jsonify({"status": "added", "feed": config.feeds[-1]}), 201


@news_bp.route("/api/news/feeds/<int:index>", methods=["DELETE"])
def api_delete_feed(index: int) -> tuple:
    """DELETE /api/news/feeds/<index> — remove a feed by index."""
    config = load_news_config()
    if index < 0 or index >= len(config.feeds):
        return jsonify({"error": "Invalid feed index"}), 404
    removed = config.feeds.pop(index)
    save_news_config(config)
    return jsonify({"status": "deleted", "feed": removed})


@news_bp.route("/api/news/feeds/<int:index>/toggle", methods=["POST"])
def api_toggle_feed(index: int) -> tuple:
    """POST /api/news/feeds/<index>/toggle — enable/disable a feed."""
    config = load_news_config()
    if index < 0 or index >= len(config.feeds):
        return jsonify({"error": "Invalid feed index"}), 404
    config.feeds[index]["enabled"] = not config.feeds[index].get("enabled", True)
    save_news_config(config)
    return jsonify({"status": "toggled", "feed": config.feeds[index]})


@news_bp.route("/api/news/likes", methods=["POST"])
def api_add_like() -> tuple:
    """POST /api/news/likes — add a liked topic."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Need topic"}), 400
    config = load_news_config()
    if topic not in config.likes:
        config.likes.append(topic)
        # Remove from dislikes if present
        config.dislikes = [d for d in config.dislikes if d != topic]
        save_news_config(config)
    return jsonify({"status": "added", "likes": config.likes})


@news_bp.route("/api/news/likes", methods=["DELETE"])
def api_remove_like() -> tuple:
    """DELETE /api/news/likes — remove a liked topic."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    config = load_news_config()
    config.likes = [l for l in config.likes if l != topic]
    save_news_config(config)
    return jsonify({"status": "removed", "likes": config.likes})


@news_bp.route("/api/news/dislikes", methods=["POST"])
def api_add_dislike() -> tuple:
    """POST /api/news/dislikes — add a disliked topic."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Need topic"}), 400
    config = load_news_config()
    if topic not in config.dislikes:
        config.dislikes.append(topic)
        # Remove from likes if present
        config.likes = [l for l in config.likes if l != topic]
        save_news_config(config)
    return jsonify({"status": "added", "dislikes": config.dislikes})


@news_bp.route("/api/news/dislikes", methods=["DELETE"])
def api_remove_dislike() -> tuple:
    """DELETE /api/news/dislikes — remove a disliked topic."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    config = load_news_config()
    config.dislikes = [d for d in config.dislikes if d != topic]
    save_news_config(config)
    return jsonify({"status": "removed", "dislikes": config.dislikes})


@news_bp.route("/api/news/reset", methods=["POST"])
def api_reset_config() -> tuple:
    """POST /api/news/reset — reset configuration to defaults."""
    config = NewsConfig()
    save_news_config(config)
    return jsonify(config.to_dict())


# ============================================================================
# NEWS CONFIG WEB PAGE
# ============================================================================

NEWS_CONFIG_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --orange: #ff9800;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.6; }
h1 { margin-bottom: 0.5rem; color: var(--accent); }
h2 { font-size: 1.1rem; color: var(--accent); margin: 1.5rem 0 0.8rem;
     border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
.nav { margin-bottom: 1.5rem; display: flex; gap: 12px; flex-wrap: wrap; }
.nav a { color: var(--accent); text-decoration: none; padding: 6px 14px;
         border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }
.nav a:hover, .nav a.active { background: var(--accent); color: #000; }
.section { background: var(--surface); border-radius: 8px; padding: 1.2rem;
           margin-bottom: 1rem; }
.feed-row { display: flex; align-items: center; gap: 10px; padding: 8px 0;
            border-bottom: 1px solid var(--border); }
.feed-row:last-child { border-bottom: none; }
.feed-name { font-weight: 600; min-width: 140px; }
.feed-url { color: var(--muted); font-size: 0.85rem; flex: 1; overflow: hidden;
            text-overflow: ellipsis; white-space: nowrap; }
.feed-cat { font-size: 0.75rem; padding: 2px 8px; background: #333;
            border-radius: 4px; color: var(--accent); }
.toggle { cursor: pointer; padding: 4px 10px; border-radius: 4px; border: none;
          font-size: 0.8rem; font-weight: 500; }
.toggle.on { background: var(--green); color: white; }
.toggle.off { background: #555; color: #aaa; }
.btn-sm { padding: 4px 10px; border: none; border-radius: 4px; cursor: pointer;
          font-size: 0.8rem; }
.btn-del { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn-del:hover { background: var(--red); color: white; border-color: var(--red); }
.btn-add { background: var(--accent); color: #000; font-weight: 500; padding: 6px 16px; }
.btn-add:hover { opacity: 0.85; }
.add-form { display: flex; gap: 8px; margin-top: 0.8rem; flex-wrap: wrap; }
.add-form input, .add-form select { padding: 6px 10px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 6px; color: var(--text);
    font-size: 0.9rem; }
.add-form input:focus, .add-form select:focus { outline: none; border-color: var(--accent); }
.add-form input[name="name"] { width: 140px; }
.add-form input[name="url"] { flex: 1; min-width: 200px; }
.tag-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 0.8rem; }
.tag { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
       border-radius: 16px; font-size: 0.85rem; }
.tag.like { background: #1b3a1b; color: var(--green); }
.tag.dislike { background: #3a1b1b; color: var(--red); }
.tag .remove { cursor: pointer; font-weight: bold; opacity: 0.6; }
.tag .remove:hover { opacity: 1; }
.tag-form { display: flex; gap: 8px; }
.tag-form input { padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 0.9rem; flex: 1; }
.tag-form input:focus { outline: none; border-color: var(--accent); }
.schedule-row { display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem; }
.schedule-row label { color: var(--muted); font-size: 0.9rem; min-width: 140px; }
.schedule-row input, .schedule-row select { padding: 6px 10px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 6px; color: var(--text);
    font-size: 0.9rem; width: 80px; }
.toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
    background: var(--surface); color: var(--text); padding: 12px 24px;
    border-radius: 8px; border: 1px solid var(--accent); font-size: 0.9rem;
    z-index: 9999; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateX(-50%) translateY(20px); }
                    to { opacity: 1; transform: translateX(-50%) translateY(0); } }
"""


def render_news_config_page() -> str:
    """Render the news configuration page HTML."""
    config = load_news_config()

    # Build feed rows
    feed_rows = ""
    for i, feed in enumerate(config.feeds):
        enabled = feed.get("enabled", True)
        toggle_cls = "on" if enabled else "off"
        toggle_txt = "ON" if enabled else "OFF"
        cat = feed.get("category", "other")
        feed_rows += f"""
        <div class="feed-row" id="feed-{i}">
            <button class="toggle {toggle_cls}" onclick="toggleFeed({i})">{toggle_txt}</button>
            <span class="feed-name">{feed['name']}</span>
            <span class="feed-cat">{cat}</span>
            <span class="feed-url">{feed['url']}</span>
            <button class="btn-sm btn-del" onclick="deleteFeed({i})">Remove</button>
        </div>"""

    # Build category options
    cat_options = "".join(f'<option value="{c}">{c}</option>' for c in FEED_CATEGORIES)

    # Build like/dislike tags
    like_tags = "".join(
        f'<span class="tag like">{t}<span class="remove" onclick="removeLike(\'{t}\')">x</span></span>'
        for t in config.likes
    )
    dislike_tags = "".join(
        f'<span class="tag dislike">{t}<span class="remove" onclick="removeDislike(\'{t}\')">x</span></span>'
        for t in config.dislikes
    )

    feed_count = len([f for f in config.feeds if f.get("enabled", True)])
    total_feeds = len(config.feeds)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Config - Technomancer Hub</title>
    <style>{NEWS_CONFIG_CSS}</style>
</head>
<body>
    <h1>Technomancer Hub</h1>
    <div class="nav">
        <a href="/">Hub</a>
        <a href="/ideas">Ideas</a>
        <a href="/news" class="active">News Config</a>
        <a href="/karen">KAREN</a>
    </div>

    <p style="color:var(--muted);margin-bottom:1rem">{feed_count}/{total_feeds} feeds active &bull; {len(config.likes)} likes &bull; {len(config.dislikes)} dislikes &bull; Schedule: {config.start_hour}:00-{config.end_hour}:00</p>

    <h2>RSS Feeds</h2>
    <div class="section" id="feeds-section">
        {feed_rows if feed_rows else '<p style="color:var(--muted)">No feeds configured.</p>'}
        <form class="add-form" onsubmit="addFeed(event)">
            <input name="name" placeholder="Feed name" required>
            <input name="url" placeholder="RSS feed URL" required>
            <select name="category">{cat_options}</select>
            <button type="submit" class="btn-sm btn-add">Add Feed</button>
        </form>
    </div>

    <h2>Topics I Like (boost relevance)</h2>
    <div class="section">
        <div class="tag-list" id="likes-list">
            {like_tags if like_tags else '<span style="color:var(--muted)">No liked topics yet.</span>'}
        </div>
        <form class="tag-form" onsubmit="addLike(event)">
            <input name="topic" placeholder="e.g. Python, AI, Kubernetes..." required>
            <button type="submit" class="btn-sm btn-add">Add</button>
        </form>
    </div>

    <h2>Topics I Dislike (reduce relevance)</h2>
    <div class="section">
        <div class="tag-list" id="dislikes-list">
            {dislike_tags if dislike_tags else '<span style="color:var(--muted)">No disliked topics yet.</span>'}
        </div>
        <form class="tag-form" onsubmit="addDislike(event)">
            <input name="topic" placeholder="e.g. celebrity gossip, sports..." required>
            <button type="submit" class="btn-sm btn-add">Add</button>
        </form>
    </div>

    <h2>Schedule</h2>
    <div class="section">
        <div class="schedule-row">
            <label>Start hour (24h):</label>
            <input type="number" id="start-hour" min="0" max="23" value="{config.start_hour}" onchange="updateSchedule()">
        </div>
        <div class="schedule-row">
            <label>End hour (24h):</label>
            <input type="number" id="end-hour" min="1" max="24" value="{config.end_hour}" onchange="updateSchedule()">
        </div>
        <div class="schedule-row">
            <label>Articles per hour:</label>
            <input type="number" id="max-articles" min="1" max="10" value="{config.max_articles_per_hour}" onchange="updateSchedule()">
        </div>
    </div>

    <script>
    function showToast(msg) {{
        const old = document.getElementById('toast');
        if (old) old.remove();
        const t = document.createElement('div');
        t.id = 'toast'; t.className = 'toast'; t.textContent = msg;
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 3000);
    }}

    async function toggleFeed(i) {{
        await fetch(`/api/news/feeds/${{i}}/toggle`, {{method:'POST'}});
        location.reload();
    }}

    async function deleteFeed(i) {{
        if (!confirm('Remove this feed?')) return;
        await fetch(`/api/news/feeds/${{i}}`, {{method:'DELETE'}});
        location.reload();
    }}

    async function addFeed(e) {{
        e.preventDefault();
        const f = e.target;
        const resp = await fetch('/api/news/feeds', {{
            method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{name: f.name.value, url: f.url.value, category: f.category.value}})
        }});
        if (resp.ok) {{ showToast('Feed added'); location.reload(); }}
        else {{ const d = await resp.json(); showToast(d.error || 'Failed'); }}
    }}

    async function addLike(e) {{
        e.preventDefault();
        const input = e.target.topic;
        await fetch('/api/news/likes', {{
            method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{topic: input.value}})
        }});
        showToast('Like added'); location.reload();
    }}

    async function removeLike(topic) {{
        await fetch('/api/news/likes', {{
            method:'DELETE', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{topic}})
        }});
        location.reload();
    }}

    async function addDislike(e) {{
        e.preventDefault();
        const input = e.target.topic;
        await fetch('/api/news/dislikes', {{
            method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{topic: input.value}})
        }});
        showToast('Dislike added'); location.reload();
    }}

    async function removeDislike(topic) {{
        await fetch('/api/news/dislikes', {{
            method:'DELETE', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{topic}})
        }});
        location.reload();
    }}

    async function updateSchedule() {{
        await fetch('/api/news/config', {{
            method:'PUT', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{
                start_hour: parseInt(document.getElementById('start-hour').value),
                end_hour: parseInt(document.getElementById('end-hour').value),
                max_articles_per_hour: parseInt(document.getElementById('max-articles').value)
            }})
        }});
        showToast('Schedule updated');
    }}
    </script>
</body>
</html>"""


@news_bp.route("/news")
def news_config_page() -> str:
    """Serve the news configuration page."""
    return render_news_config_page()
