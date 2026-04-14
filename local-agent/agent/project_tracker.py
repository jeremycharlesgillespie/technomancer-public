"""
Project Tracker — SQLite-backed personal project tracking.

Stores project metadata (name, repo URL, status, blockers, notes) in a local
SQLite database.  Exposes CRUD operations as LLM tools so the agent can
register, update, list, and remove tracked projects on behalf of the owner.

Includes periodic GitHub REST API sync that pulls open PR count, issue count,
last commit date, and repo description for each tracked project with a repo URL.

Database lives at ``local-agent/data/projects.db``.
"""

import asyncio
import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

from .config import settings

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
SYNC_INTERVAL_MINUTES = 30

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "projects.db"

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
    """Create the projects table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            repo_url    TEXT    NOT NULL DEFAULT '',
            status      TEXT    NOT NULL DEFAULT 'active',
            blockers    TEXT    NOT NULL DEFAULT '',
            notes       TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL,
            last_synced TEXT    NOT NULL DEFAULT '',
            github_data TEXT    NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_name
        ON projects (name)
    """)
    conn.commit()
    # Migrate existing DBs: add github_data column if missing
    _migrate_add_github_data(conn)


def _migrate_add_github_data(conn: sqlite3.Connection) -> None:
    """Add github_data column to existing DBs that lack it."""
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
        if "github_data" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN github_data TEXT NOT NULL DEFAULT ''")
            conn.commit()
            log.info("[ProjectTracker] Migrated: added github_data column")
    except Exception:
        log.exception("[ProjectTracker] Migration error")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

_GITHUB_URL_RE = re.compile(
    r"(?:https?://)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/\s#?]+)"
)


def parse_github_repo(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL, or None if not a GitHub URL."""
    m = _GITHUB_URL_RE.search(url)
    if not m:
        return None
    repo = m.group("repo")
    # Strip trailing .git
    if repo.endswith(".git"):
        repo = repo[:-4]
    return m.group("owner"), repo


async def github_fetch(repo_url: str, token: str | None = None) -> dict[str, Any] | None:
    """Fetch GitHub data for a repo URL.

    Returns a dict with keys: description, stars, open_issues, open_prs,
    last_commit_sha, last_commit_date.  Returns None on error or non-GitHub URL.
    """
    parsed = parse_github_repo(repo_url)
    if not parsed:
        return None
    owner, repo = parsed
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            # 1. Repo metadata (description, stars, open_issues_count)
            async with session.get(f"{GITHUB_API}/repos/{owner}/{repo}") as resp:
                if resp.status != 200:
                    log.warning("[GitHubSync] Repo API returned %d for %s/%s", resp.status, owner, repo)
                    return None
                repo_data = await resp.json()

            # 2. Open PRs count
            async with session.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
                params={"state": "open", "per_page": "1"},
            ) as resp:
                pr_count = 0
                if resp.status == 200:
                    pr_body = await resp.json()
                    if pr_body:
                        pr_count = _parse_last_page(resp) or len(pr_body)

            # 3. Last commit
            async with session.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                params={"per_page": "1"},
            ) as resp:
                last_sha = ""
                last_date = ""
                if resp.status == 200:
                    commits = await resp.json()
                    if commits:
                        last_sha = commits[0].get("sha", "")[:7]
                        commit_info = commits[0].get("commit", {})
                        last_date = commit_info.get("committer", {}).get("date", "")

        return {
            "description": repo_data.get("description", "") or "",
            "stars": repo_data.get("stargazers_count", 0),
            "open_issues": repo_data.get("open_issues_count", 0),
            "open_prs": pr_count,
            "last_commit_sha": last_sha,
            "last_commit_date": last_date,
        }
    except Exception:
        log.exception("[GitHubSync] Error fetching %s/%s", owner, repo)
        return None


def _parse_last_page(resp: aiohttp.ClientResponse) -> int:
    """Extract the last page number from a GitHub pagination Link header.

    When requesting per_page=1, the last page number equals the total count.
    Returns 0 if no Link header or no 'last' relation found.
    """
    link = resp.headers.get("Link", "")
    if 'rel="last"' not in link:
        return 0
    for part in link.split(","):
        if 'rel="last"' in part:
            m = re.search(r"[?&]page=(\d+)", part)
            if m:
                return int(m.group(1))
    return 0


async def sync_project_github(name: str, repo_url: str) -> bool:
    """Sync a single project's GitHub data. Returns True on success."""
    token = settings.github_token
    data = await github_fetch(repo_url, token)
    if data is None:
        return False
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE projects SET github_data = ?, last_synced = ? WHERE name = ?",
            (json.dumps(data), datetime.now().isoformat(), name),
        )
        conn.commit()
        log.info("[GitHubSync] Synced %s: %d PRs, %d issues", name, data["open_prs"], data["open_issues"])
        return True
    except Exception:
        log.exception("[GitHubSync] DB update failed for %s", name)
        return False


async def sync_all_projects() -> dict[str, bool]:
    """Sync GitHub data for all projects that have a repo_url.

    Returns a dict mapping project name to success/failure.
    """
    projects = get_all_projects_raw()
    results: dict[str, bool] = {}
    for p in projects:
        if not p["repo_url"] or not parse_github_repo(p["repo_url"]):
            continue
        results[p["name"]] = await sync_project_github(p["name"], p["repo_url"])
    return results


async def _github_sync_loop() -> None:
    """Background loop: sync all projects every SYNC_INTERVAL_MINUTES."""
    log.info("[GitHubSync] Background sync starting (every %d min)", SYNC_INTERVAL_MINUTES)
    while True:
        try:
            results = await sync_all_projects()
            if results:
                ok = sum(1 for v in results.values() if v)
                log.info("[GitHubSync] Sync complete: %d/%d succeeded", ok, len(results))
        except Exception:
            log.exception("[GitHubSync] Sync loop error")
        await asyncio.sleep(SYNC_INTERVAL_MINUTES * 60)


def start_github_sync() -> None:
    """Start the GitHub sync background task (call from on_ready)."""
    from .task_manager import create_monitored_task

    token = settings.github_token
    if not token:
        log.warning("[GitHubSync] No GITHUB_TOKEN configured — sync disabled")
        return
    create_monitored_task(_github_sync_loop(), "github-sync", critical=True)
    log.info("[GitHubSync] Background task started")


def _get_github_data(project: dict[str, Any]) -> dict[str, Any] | None:
    """Parse github_data JSON from a project row, or None if empty."""
    raw = project.get("github_data", "")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def add_project(name: str, repo_url: str = "", notes: str = "") -> str:
    """Register a new project. Returns confirmation or error message."""
    try:
        conn = _get_conn()
        init_db()
        conn.execute(
            """INSERT INTO projects (name, repo_url, notes, created_at)
               VALUES (?, ?, ?, ?)""",
            (name.strip(), repo_url.strip(), notes.strip(), datetime.now().isoformat()),
        )
        conn.commit()
        return f"Project **{name}** tracked successfully."
    except sqlite3.IntegrityError:
        return f"Project **{name}** is already tracked."
    except Exception:
        log.exception("Failed to add project")
        return "Error adding project."


def remove_project(name: str) -> str:
    """Remove a tracked project by name."""
    try:
        conn = _get_conn()
        init_db()
        cur = conn.execute("DELETE FROM projects WHERE name = ?", (name.strip(),))
        conn.commit()
        if cur.rowcount:
            return f"Project **{name}** removed."
        return f"No project named **{name}** found."
    except Exception:
        log.exception("Failed to remove project")
        return "Error removing project."


def list_projects() -> str:
    """Return a formatted list of all tracked projects."""
    try:
        conn = _get_conn()
        init_db()
        rows = conn.execute(
            "SELECT name, repo_url, status, blockers, github_data FROM projects ORDER BY name"
        ).fetchall()
        if not rows:
            return "No projects tracked yet. Use `track <name> <url>` to add one."
        lines = ["**Tracked Projects**\n"]
        for r in rows:
            status_icon = {"active": "\u2705", "paused": "\u23f8\ufe0f", "done": "\u2714\ufe0f"}.get(
                r["status"], "\u2753"
            )
            line = f"{status_icon} **{r['name']}**  \u2014  {r['status']}"
            if r["repo_url"]:
                line += f"  |  <{r['repo_url']}>"
            gh = _get_github_data(dict(r))
            if gh:
                parts = []
                if gh.get("open_prs"):
                    parts.append(f"{gh['open_prs']} PRs")
                if gh.get("open_issues"):
                    parts.append(f"{gh['open_issues']} issues")
                if gh.get("stars"):
                    parts.append(f"\u2b50 {gh['stars']}")
                if parts:
                    line += f"  |  {', '.join(parts)}"
            if r["blockers"]:
                line += f"\n   Blockers: {r['blockers']}"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        log.exception("Failed to list projects")
        return "Error listing projects."


def get_project(name: str) -> str:
    """Return detailed info for a single project."""
    try:
        conn = _get_conn()
        init_db()
        row = conn.execute(
            "SELECT * FROM projects WHERE name = ?", (name.strip(),)
        ).fetchone()
        if not row:
            return f"No project named **{name}** found."
        d = dict(row)
        lines = [
            f"**{d['name']}**",
            f"Status: {d['status']}",
            f"Repo: {d['repo_url'] or 'none'}",
            f"Blockers: {d['blockers'] or 'none'}",
            f"Notes: {d['notes'] or 'none'}",
            f"Created: {d['created_at']}",
            f"Last synced: {d['last_synced'] or 'never'}",
        ]
        gh = _get_github_data(d)
        if gh:
            lines.append("**GitHub**")
            if gh.get("description"):
                lines.append(f"  Description: {gh['description']}")
            lines.append(f"  Open PRs: {gh.get('open_prs', 0)}")
            lines.append(f"  Open Issues: {gh.get('open_issues', 0)}")
            lines.append(f"  Stars: {gh.get('stars', 0)}")
            if gh.get("last_commit_sha"):
                lines.append(f"  Last commit: {gh['last_commit_sha']} ({gh.get('last_commit_date', '')})")
        return "\n".join(lines)
    except Exception:
        log.exception("Failed to get project")
        return "Error fetching project details."


def add_blocker(name: str, blocker_text: str) -> str:
    """Append a blocker to a project."""
    try:
        conn = _get_conn()
        init_db()
        row = conn.execute(
            "SELECT blockers FROM projects WHERE name = ?", (name.strip(),)
        ).fetchone()
        if not row:
            return f"No project named **{name}** found."
        existing = row["blockers"]
        updated = f"{existing}; {blocker_text.strip()}" if existing else blocker_text.strip()
        conn.execute(
            "UPDATE projects SET blockers = ? WHERE name = ?", (updated, name.strip())
        )
        conn.commit()
        return f"Blocker added to **{name}**: {blocker_text.strip()}"
    except Exception:
        log.exception("Failed to add blocker")
        return "Error adding blocker."


def update_project(name: str, **fields: Any) -> str:
    """Update one or more fields on a project (status, notes, blockers, repo_url)."""
    allowed = {"status", "notes", "blockers", "repo_url", "last_synced", "github_data"}
    to_set = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not to_set:
        return "Nothing to update."
    try:
        conn = _get_conn()
        init_db()
        set_clause = ", ".join(f"{k} = ?" for k in to_set)
        values = list(to_set.values()) + [name.strip()]
        cur = conn.execute(
            f"UPDATE projects SET {set_clause} WHERE name = ?", values  # noqa: S608
        )
        conn.commit()
        if cur.rowcount:
            return f"Project **{name}** updated."
        return f"No project named **{name}** found."
    except Exception:
        log.exception("Failed to update project")
        return "Error updating project."


def get_all_projects_raw() -> list[dict[str, Any]]:
    """Return all projects as a list of dicts (for programmatic use)."""
    try:
        conn = _get_conn()
        init_db()
        rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
        return [dict(r) for r in rows]
    except Exception:
        log.exception("Failed to query projects")
        return []


# ---------------------------------------------------------------------------
# Health summary (for daily briefing)
# ---------------------------------------------------------------------------


def get_project_health_summary() -> str:
    """Return a concise project health section for the daily briefing.

    Highlights:
    - Active projects with blockers
    - Projects with no sync in 7+ days (stale)
    - Open PR / issue counts if synced (future: populated by GitHub sync)
    - Skips healthy/quiet projects to keep output short (3-5 lines max per project)

    Returns:
        Formatted markdown string, or a short note if no projects tracked.
    """
    projects = get_all_projects_raw()
    if not projects:
        return "No projects tracked."

    now = datetime.now()
    stale_threshold = now - timedelta(days=7)
    lines: list[str] = []

    for p in projects:
        # Skip completed projects
        if p["status"] == "done":
            continue

        issues: list[str] = []

        # Check for blockers
        if p["blockers"]:
            issues.append(f"Blockers: {p['blockers']}")

        # Check for stale sync (no sync in 7+ days, only if repo_url is set)
        if p["repo_url"] and p["last_synced"]:
            try:
                last_sync = datetime.fromisoformat(p["last_synced"])
                if last_sync < stale_threshold:
                    days_ago = (now - last_sync).days
                    issues.append(f"No sync in {days_ago} days")
            except (ValueError, TypeError):
                issues.append("Last sync date invalid")
        elif p["repo_url"] and not p["last_synced"]:
            issues.append("Never synced")

        # Check GitHub data for open PRs / issues
        gh = _get_github_data(p)
        if gh:
            if gh.get("open_prs", 0) > 0:
                issues.append(f"Open PRs: {gh['open_prs']}")
            if gh.get("open_issues", 0) >= 5:
                issues.append(f"Open issues: {gh['open_issues']}")

        # Check for paused status
        if p["status"] == "paused":
            issues.append("Status: paused")

        # Only include projects that have something to report
        if not issues:
            continue

        status_icon = {
            "active": "\u2705",
            "paused": "\u23f8\ufe0f",
        }.get(p["status"], "\u2753")
        lines.append(f"{status_icon} **{p['name']}**")
        for issue in issues:
            lines.append(f"   - {issue}")

    if not lines:
        active_count = sum(1 for p in projects if p["status"] != "done")
        return f"All {active_count} project(s) healthy. No blockers or stale repos."

    header = f"**Project Health** ({len(projects)} tracked)\n"
    return header + "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM tool wrappers
# ---------------------------------------------------------------------------

def _tool_track_project(name: str, repo_url: str = "", notes: str = "") -> str:
    """Tool wrapper: track a new project."""
    return add_project(name, repo_url, notes)


def _tool_untrack_project(name: str) -> str:
    """Tool wrapper: remove a tracked project."""
    return remove_project(name)


def _tool_list_projects() -> str:
    """Tool wrapper: list all tracked projects."""
    return list_projects()


def _tool_get_project(name: str) -> str:
    """Tool wrapper: get project detail."""
    return get_project(name)


def _tool_add_blocker(name: str, blocker: str) -> str:
    """Tool wrapper: add a blocker to a project."""
    return add_blocker(name, blocker)


def _tool_update_project(name: str, status: str = "", notes: str = "", blockers: str = "") -> str:
    """Tool wrapper: update project fields."""
    fields: dict[str, Any] = {}
    if status:
        fields["status"] = status
    if notes:
        fields["notes"] = notes
    if blockers:
        fields["blockers"] = blockers
    return update_project(name, **fields)


def get_project_tracker_tools() -> list:
    """Return LLM tools for the project tracker."""
    from .core import create_tool

    return [
        create_tool(
            name="track_project",
            description=(
                "Register a new project to track. Use when the owner wants to add a "
                "project to the tracker. Stores name, repo URL, and optional notes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short project name (e.g. 'technomancer', 'myapp')",
                    },
                    "repo_url": {
                        "type": "string",
                        "description": "GitHub repo URL (e.g. 'https://github.com/user/repo')",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the project",
                    },
                },
                "required": ["name"],
            },
            function=_tool_track_project,
        ),
        create_tool(
            name="untrack_project",
            description="Remove a project from the tracker.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name to remove",
                    },
                },
                "required": ["name"],
            },
            function=_tool_untrack_project,
        ),
        create_tool(
            name="list_tracked_projects",
            description=(
                "List all tracked projects with their status and blockers. "
                "Use when the owner asks 'what am I working on?' or wants a project overview."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            function=_tool_list_projects,
        ),
        create_tool(
            name="get_project_detail",
            description="Get full details for a specific tracked project.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name to look up",
                    },
                },
                "required": ["name"],
            },
            function=_tool_get_project,
        ),
        create_tool(
            name="add_project_blocker",
            description="Add a blocker or obstacle to a tracked project.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name",
                    },
                    "blocker": {
                        "type": "string",
                        "description": "Description of the blocker",
                    },
                },
                "required": ["name", "blocker"],
            },
            function=_tool_add_blocker,
        ),
        create_tool(
            name="update_project",
            description=(
                "Update a tracked project's status, notes, or blockers. "
                "Status can be: active, paused, done."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name to update",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status: active, paused, or done",
                    },
                    "notes": {
                        "type": "string",
                        "description": "New notes (replaces existing)",
                    },
                    "blockers": {
                        "type": "string",
                        "description": "New blockers text (replaces existing)",
                    },
                },
                "required": ["name"],
            },
            function=_tool_update_project,
        ),
    ]
