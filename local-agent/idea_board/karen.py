"""
KAREN — Kinetic Aggression Routing Enhancement Network.

A complaint-to-idea pipeline: users voice frustrations via Discord or the
web UI, and the LLM immediately generates 1-3 actionable improvement ideas
that land on the idea board.

Complaint lifecycle:
  pending → processed (ideas generated) → resolved (idea accepted) / dismissed

Storage: complaints.json (same directory as ideas.json)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

# ============================================================================
# STORAGE
# ============================================================================

COMPLAINTS_FILE: Path = Path(__file__).parent / "complaints.json"
_lock = threading.Lock()


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass
class Complaint:
    """A user complaint that feeds the idea generation pipeline.

    Attributes:
        id: Unique identifier (e.g. "complaint-001")
        text: The raw complaint text
        author: Discord username or "web"
        timestamp: ISO format timestamp
        state: pending | processed | resolved | dismissed
        generated_idea_ids: IDs of ideas created from this complaint
    """

    id: str
    text: str
    author: str = "web"
    timestamp: str = ""
    state: str = "pending"
    generated_idea_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "timestamp": self.timestamp,
            "state": self.state,
            "generated_idea_ids": self.generated_idea_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Complaint:
        return cls(
            id=data["id"],
            text=data["text"],
            author=data.get("author", "web"),
            timestamp=data.get("timestamp", ""),
            state=data.get("state", "pending"),
            generated_idea_ids=data.get("generated_idea_ids", []),
        )


# ============================================================================
# PERSISTENCE
# ============================================================================


def load_complaints() -> list[Complaint]:
    """Load all complaints from disk."""
    with _lock:
        if not COMPLAINTS_FILE.exists():
            return []
        try:
            data = json.loads(COMPLAINTS_FILE.read_text(encoding="utf-8"))
            return [Complaint.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[KAREN] Failed to load complaints: {e}")
            return []


def save_complaints(complaints: list[Complaint]) -> None:
    """Persist complaints to disk."""
    with _lock:
        COMPLAINTS_FILE.write_text(
            json.dumps([c.to_dict() for c in complaints], indent=2),
            encoding="utf-8",
        )


def _next_complaint_id(complaints: list[Complaint]) -> str:
    """Generate the next sequential complaint ID."""
    if not complaints:
        return "complaint-001"
    nums = []
    for c in complaints:
        try:
            nums.append(int(c.id.split("-")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums, default=0) + 1
    return f"complaint-{next_num:03d}"


# ============================================================================
# COMPLAINT OPERATIONS
# ============================================================================


def add_complaint(text: str, author: str = "web") -> Complaint:
    """Create and persist a new complaint.

    Args:
        text: The complaint text
        author: Discord username or "web"

    Returns:
        The created Complaint
    """
    complaints = load_complaints()
    complaint = Complaint(
        id=_next_complaint_id(complaints),
        text=text,
        author=author,
    )
    complaints.append(complaint)
    save_complaints(complaints)
    logger.info(f"[KAREN] New complaint {complaint.id} from {author}: {text[:80]}")
    return complaint


def dismiss_complaint(complaint_id: str) -> Complaint | None:
    """Dismiss a complaint (hides from default view).

    Returns:
        The updated Complaint, or None if not found
    """
    complaints = load_complaints()
    complaint = next((c for c in complaints if c.id == complaint_id), None)
    if not complaint:
        return None
    complaint.state = "dismissed"
    save_complaints(complaints)
    logger.info(f"[KAREN] Dismissed {complaint_id}")
    return complaint


def resolve_complaint_for_idea(idea_id: str) -> None:
    """Resolve any complaint that generated a given idea.

    Called from models.py when a karen-sourced idea is accepted or completed.
    """
    complaints = load_complaints()
    changed = False
    for complaint in complaints:
        if idea_id in complaint.generated_idea_ids and complaint.state in ("pending", "processed"):
            complaint.state = "resolved"
            changed = True
            logger.info(f"[KAREN] Auto-resolved {complaint.id} (idea {idea_id} accepted)")
    if changed:
        save_complaints(complaints)


# ============================================================================
# IDEA GENERATION FROM COMPLAINTS
# ============================================================================

COMPLAINT_PROMPT = """You are KAREN (Kinetic Aggression Routing Enhancement Network), \
an improvement analyst for the Technomancer Discord bot project.

A user has submitted a complaint. Analyze their frustration and generate 1-3 \
concrete, actionable improvement ideas that would address it.

COMPLAINT: {complaint_text}
SUBMITTED BY: {author}

EXISTING IDEAS (do NOT duplicate these):
{existing}

Output ONLY a JSON array of idea objects. Each object must have:
- "title": Short descriptive title (under 80 chars)
- "description": Structured text with WHAT/WHY/HOW/BENEFITS/COST/UNLOCKS sections
- "category": One of: performance, feature, quality, security, ux

Example output:
```json
[{{"title": "Example improvement", "description": "WHAT: ...", "category": "ux"}}]
```"""


def _load_existing_ideas() -> str:
    """Load existing idea titles for dedup context."""
    try:
        from board import get_provider

        ideas = get_provider().load_all()
        if not ideas:
            return "No existing ideas."
        return "\n".join(
            f"- [{i.state}] {i.title}"
            for i in ideas
            if i.state not in ("vetoed", "failed")
        )
    except Exception:
        return "No existing ideas."


def generate_ideas_from_complaint(complaint: Complaint) -> list[str]:
    """Generate 1-3 ideas from a complaint using Ollama.

    Calls the LLM, parses the response, adds ideas to the board, and
    updates the complaint with the generated idea IDs.

    Args:
        complaint: The Complaint to generate ideas from

    Returns:
        List of created idea IDs
    """
    import ollama

    from board import get_provider

    provider = get_provider()

    prompt = COMPLAINT_PROMPT.format(
        complaint_text=complaint.text,
        author=complaint.author,
        existing=_load_existing_ideas(),
    )

    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        response = client.chat(
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_ctx": 8192},
        )
        content = response.get("message", {}).get("content", "") or ""

        # Strip thinking tags if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    except Exception as e:
        logger.error(f"[KAREN] LLM call failed: {e}")
        return []

    # Reuse the proven parser from idea_generator
    from agent.idea_generator import _parse_ideas

    parsed = _parse_ideas(content)
    if not parsed:
        logger.warning("[KAREN] No ideas parsed from LLM response")
        return []

    idea_ids: list[str] = []
    for idea_data in parsed:
        idea = provider.add(
            title=idea_data["title"],
            description=idea_data["description"],
            source="karen",
            category=idea_data.get("category", "feature"),
        )
        # Link complaint to idea via a comment
        provider.add_comment(
            idea.id,
            "claude",
            f"Generated from KAREN complaint {complaint.id}: {complaint.text[:200]}",
        )
        idea_ids.append(idea.id)

    # Update complaint with results
    complaints = load_complaints()
    for c in complaints:
        if c.id == complaint.id:
            c.generated_idea_ids = idea_ids
            c.state = "processed"
            break
    save_complaints(complaints)

    logger.info(f"[KAREN] Generated {len(idea_ids)} idea(s) from {complaint.id}: {idea_ids}")
    return idea_ids


# ============================================================================
# FLASK BLUEPRINT — API + WEB PAGE
# ============================================================================

karen_bp = Blueprint("karen", __name__)


@karen_bp.route("/api/karen/complain", methods=["POST"])
def api_complain() -> tuple:
    """Submit a complaint and trigger immediate idea generation."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Complaint text is required"}), 400

    author = data.get("author", "web")
    complaint = add_complaint(text=text, author=author)
    idea_ids = generate_ideas_from_complaint(complaint)

    # Reload complaint to get updated state
    complaints = load_complaints()
    updated = next((c for c in complaints if c.id == complaint.id), complaint)

    return jsonify({
        "complaint": updated.to_dict(),
        "idea_ids": idea_ids,
    })


@karen_bp.route("/api/karen/complaints")
def api_complaints() -> tuple:
    """List complaints. Default hides resolved/dismissed; ?state=all shows everything."""
    complaints = load_complaints()
    state_filter = request.args.get("state", "active")
    if state_filter == "all":
        pass  # No filtering
    elif state_filter == "active":
        complaints = [c for c in complaints if c.state not in ("resolved", "dismissed")]
    else:
        complaints = [c for c in complaints if c.state == state_filter]

    return jsonify([c.to_dict() for c in complaints]), 200


@karen_bp.route("/api/karen/<complaint_id>", methods=["DELETE"])
def api_dismiss(complaint_id: str) -> tuple:
    """Dismiss a complaint."""
    complaint = dismiss_complaint(complaint_id)
    if not complaint:
        return jsonify({"error": "Complaint not found"}), 404
    return jsonify({"status": "dismissed", "complaint": complaint.to_dict()})


@karen_bp.route("/karen")
def karen_page() -> str:
    """Render the KAREN complaints page."""
    complaints = load_complaints()
    active = [c for c in complaints if c.state not in ("resolved", "dismissed")]
    resolved_count = len([c for c in complaints if c.state == "resolved"])
    return _render_karen_page(active, len(complaints), resolved_count)


# ============================================================================
# HTML PAGE
# ============================================================================

KAREN_CSS = """
:root {
    --bg: #1a1a2e; --surface: #16213e; --card: #0f3460;
    --text: #e0e0e0; --muted: #888; --accent: #e94560;
    --green: #4ecca3; --red: #e94560; --orange: #f39c12; --blue: #3498db;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 2rem; max-width: 900px; margin: 0 auto; }
h1 { margin-bottom: 0.25rem; }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; }
.nav { margin-bottom: 1.5rem; display: flex; gap: 1rem; }
.nav a { color: var(--muted); text-decoration: none; padding: 0.4rem 0.8rem;
         border-radius: 6px; transition: all 0.2s; }
.nav a:hover { color: var(--text); background: var(--surface); }
.nav a.active { color: var(--accent); background: var(--surface); }
.stats { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
.form-card { background: var(--surface); border-radius: 10px; padding: 1.5rem;
             margin-bottom: 2rem; border-left: 4px solid var(--accent); }
.form-card h2 { font-size: 1.1rem; margin-bottom: 0.75rem; }
.form-card textarea { width: 100%; min-height: 80px; padding: 0.75rem;
    background: var(--bg); color: var(--text); border: 1px solid #333;
    border-radius: 6px; font-family: inherit; font-size: 0.95rem; resize: vertical; }
.form-card textarea:focus { outline: none; border-color: var(--accent); }
.btn { padding: 0.6rem 1.2rem; border: none; border-radius: 6px; cursor: pointer;
       font-size: 0.95rem; font-weight: 600; transition: all 0.2s; margin-top: 0.75rem; }
.btn-submit { background: var(--accent); color: white; }
.btn-submit:hover { filter: brightness(1.15); }
.btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-dismiss { background: transparent; color: var(--muted); border: 1px solid #333;
               font-size: 0.8rem; padding: 0.3rem 0.7rem; }
.btn-dismiss:hover { border-color: var(--red); color: var(--red); }
.complaint { background: var(--surface); border-radius: 10px; padding: 1.25rem;
             margin-bottom: 1rem; border-left: 4px solid var(--orange); }
.complaint.processed { border-left-color: var(--green); }
.complaint .meta { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.5rem; }
.complaint .text { margin-bottom: 0.75rem; line-height: 1.5; }
.complaint .ideas { font-size: 0.9rem; }
.complaint .ideas a { color: var(--blue); text-decoration: none; }
.complaint .ideas a:hover { text-decoration: underline; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
         font-size: 0.75rem; font-weight: 600; }
.badge.pending { background: var(--orange); color: #000; }
.badge.processed { background: var(--green); color: #000; }
.spinner { display: none; margin-top: 0.75rem; color: var(--muted); font-style: italic; }
.empty { color: var(--muted); text-align: center; padding: 3rem; }
.toast { position: fixed; bottom: 2rem; right: 2rem; background: var(--green);
         color: #000; padding: 0.75rem 1.25rem; border-radius: 8px;
         font-weight: 600; display: none; z-index: 100; }
"""


def _render_karen_page(active: list[Complaint], total: int, resolved: int) -> str:
    """Render the KAREN complaints page HTML."""
    complaints_html = ""
    if not active:
        complaints_html = '<div class="empty">No active complaints. Everything is sunshine and rainbows.</div>'
    else:
        for c in reversed(active):  # Newest first
            state_class = c.state
            ideas_html = ""
            if c.generated_idea_ids:
                links = ", ".join(
                    f'<a href="/ideas">{iid}</a>' for iid in c.generated_idea_ids
                )
                ideas_html = f'<div class="ideas">Ideas generated: {links}</div>'

            complaints_html += f"""
            <div class="complaint {state_class}" id="{c.id}">
                <div class="meta">
                    <span class="badge {c.state}">{c.state}</span>
                    &bull; {c.author} &bull; {c.timestamp}
                    <button class="btn btn-dismiss" onclick="dismissComplaint('{c.id}')"
                            style="float: right;">Dismiss</button>
                </div>
                <div class="text">{c.text}</div>
                {ideas_html}
            </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KAREN — Complaint Board</title>
    <style>{KAREN_CSS}</style>
</head>
<body>
    <h1>K.A.R.E.N.</h1>
    <p class="subtitle">Kinetic Aggression Routing Enhancement Network</p>
    <div class="nav">
        <a href="/">Hub</a>
        <a href="/ideas">Ideas</a>
        <a href="/news">News Config</a>
        <a href="/karen" class="active">KAREN</a>
    </div>
    <p class="stats">{len(active)} active &bull; {resolved} resolved &bull; {total} total</p>

    <div class="form-card">
        <h2>Submit a Complaint</h2>
        <textarea id="complaint-text" placeholder="What's bothering you? Be as specific as you like..."></textarea>
        <button class="btn btn-submit" id="submit-btn" onclick="submitComplaint()">
            Submit Complaint
        </button>
        <div class="spinner" id="spinner">Generating improvement ideas from your frustration...</div>
    </div>

    <div id="complaints-list">
        {complaints_html}
    </div>

    <div class="toast" id="toast"></div>

    <script>
    async function submitComplaint() {{
        const text = document.getElementById('complaint-text').value.trim();
        if (!text) return;

        const btn = document.getElementById('submit-btn');
        const spinner = document.getElementById('spinner');
        btn.disabled = true;
        spinner.style.display = 'block';

        try {{
            const resp = await fetch('/api/karen/complain', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{text: text, author: 'web'}})
            }});
            const data = await resp.json();

            if (resp.ok) {{
                showToast(`Generated ${{data.idea_ids.length}} idea(s) from your complaint!`);
                document.getElementById('complaint-text').value = '';
                // Reload to show new complaint
                setTimeout(() => location.reload(), 1500);
            }} else {{
                showToast(data.error || 'Something went wrong', true);
            }}
        }} catch(e) {{
            showToast('Network error: ' + e.message, true);
        }} finally {{
            btn.disabled = false;
            spinner.style.display = 'none';
        }}
    }}

    async function dismissComplaint(id) {{
        if (!confirm('Dismiss this complaint?')) return;
        try {{
            await fetch(`/api/karen/${{id}}`, {{method: 'DELETE'}});
            document.getElementById(id).remove();
            showToast('Complaint dismissed');
        }} catch(e) {{
            showToast('Error dismissing complaint', true);
        }}
    }}

    function showToast(msg, isError) {{
        const t = document.getElementById('toast');
        t.textContent = msg;
        t.style.background = isError ? 'var(--red)' : 'var(--green)';
        t.style.display = 'block';
        setTimeout(() => t.style.display = 'none', 3000);
    }}
    </script>
</body>
</html>"""
