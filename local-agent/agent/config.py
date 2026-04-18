"""
Centralized configuration management using Pydantic Settings.

All configuration is loaded from environment variables and .env file.
This provides type safety, validation, and a single source of truth for config.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Discord settings
    discord_bot_token: str = Field(description="Discord bot token for authentication")
    discord_webhook_url: Optional[str] = Field(
        default=None, description="Discord webhook URL for bot lifecycle notifications"
    )
    discord_allowed_channel: str = Field(
        default="llm_chat", description="Discord channel name the bot responds in"
    )
    discord_alerts_channel: str = Field(
        default="bot_alerts", description="Discord channel for infrastructure and error alerts"
    )
    discord_alerts_webhook: str = Field(
        default="", description="Discord webhook URL for the alerts channel"
    )
    discord_claude_code_webhook: str = Field(
        default="", description="Discord webhook URL for claude-code-updates channel"
    )
    bot_owner: str = Field(
        default="", description="Discord username of the bot owner (for admin commands)"
    )
    owner_name: str = Field(
        default="Owner",
        description="Display name of the bot owner (used in idea board UI and prompts)",
    )

    # Anthropic/Claude settings
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for Claude escalation"
    )
    claude_cache_ttl: str = Field(default="5m", description="Claude prompt cache TTL: '5m' or '1h'")
    claude_use_vault_context: bool = Field(
        default=True, description="Use vault context with Claude API calls"
    )
    claude_show_cost: bool = Field(
        default=True, description="Show cost per Claude API request in responses"
    )

    # Ollama settings
    ollama_host: str = Field(default="http://127.0.0.1:11434", description="Ollama server URL")
    ollama_model: str = Field(default="qwen3.5:9b", description="Default Ollama model for chat")
    ollama_fast_model: str = Field(
        default="",
        description=(
            "Smaller/faster Ollama model for simple queries (e.g. 'qwen2.5:3b'). "
            "Empty string falls back to ollama_model."
        ),
    )
    ollama_vision_model: str = Field(
        default="llava-llama3", description="Ollama model for vision/image analysis"
    )
    ollama_request_timeout: float = Field(
        default=600.0,
        description=(
            "Per-request timeout (seconds) for Ollama HTTP calls. Prevents "
            "indefinite hangs when the server stalls. Default 600s (10 min)."
        ),
    )
    ollama_max_retries: int = Field(
        default=3,
        description=(
            "Max retries on transient Ollama failures (connection refused, "
            "5xx, read timeouts). Permanent errors (unknown model, 4xx) are "
            "never retried. Total attempts = retries + 1."
        ),
    )
    ollama_retry_base_delay: float = Field(
        default=1.0,
        description=(
            "Base delay (seconds) for exponential backoff between Ollama "
            "retries. Actual delay = base * 2**attempt + jitter."
        ),
    )
    ollama_retry_max_delay: float = Field(
        default=30.0,
        description="Cap (seconds) on the exponential backoff delay between retries.",
    )
    ollama_health_check_interval: int = Field(
        default=30,
        description=(
            "Seconds between Ollama /api/tags health probes. Lets the bot "
            "flip to 'degraded' before user calls start failing."
        ),
    )
    ollama_health_check_timeout: float = Field(
        default=5.0,
        description="HTTP timeout (seconds) for a single Ollama health probe.",
    )
    ollama_recovery_successes: int = Field(
        default=3,
        description=(
            "Hysteresis: consecutive successful probes required to leave the "
            "'down' state. Prevents flapping when Ollama is mid-recovery "
            "(e.g. a GPU driver is still stabilising)."
        ),
    )
    ollama_degraded_max_retries: int = Field(
        default=1,
        description=(
            "Max retries for Ollama calls while the gate is in 'degraded' "
            "state. Shorter than healthy-mode retries so we don't hammer "
            "a struggling server."
        ),
    )
    ollama_degraded_base_delay: float = Field(
        default=0.5,
        description="Base retry delay (seconds) when the gate is in 'degraded' state.",
    )
    ollama_degraded_max_delay: float = Field(
        default=2.0,
        description="Cap on retry delay (seconds) when the gate is in 'degraded' state.",
    )
    ollama_escalation_max_per_hour: int = Field(
        default=20,
        description=(
            "Rate limit on Ollama→Claude auto-escalations per rolling hour. "
            "Protects against runaway API spend when Ollama is flapping."
        ),
    )

    # Obsidian vault settings
    vault_path: Path = Field(
        default=Path.home() / "Documents" / "ObsidianVault",
        description="Path to Obsidian vault root (set VAULT_PATH in .env)",
    )

    # GitHub settings
    github_token: Optional[str] = Field(
        default=None, description="GitHub personal access token for API sync"
    )

    # Jira integration
    jira_url: Optional[str] = Field(
        default=None, description="Jira instance URL (e.g., https://myorg.atlassian.net)"
    )
    jira_email: Optional[str] = Field(
        default=None, description="Jira account email for API auth"
    )
    jira_api_token: Optional[str] = Field(
        default=None, description="Jira API token"
    )
    jira_project_key: Optional[str] = Field(
        default=None, description="Jira project key (e.g., TK)"
    )

    # GitHub Pages settings
    github_pages_enabled: bool = Field(
        default=True, description="Enable GitHub Pages deployment for learning articles"
    )
    github_pages_url: str = Field(
        default="https://yourusername.github.io/technomancer",
        description="Base URL for GitHub Pages site",
    )

    # API usage anomaly detection settings
    anomaly_spike_multiplier: float = Field(
        default=2.0, description="Alert when usage exceeds Nx the baseline"
    )
    anomaly_window_seconds: int = Field(
        default=3600, description="Rolling window size in seconds for rate tracking"
    )
    anomaly_baseline_hours: int = Field(
        default=24, description="Hours of history used to compute baselines"
    )
    anomaly_cooldown_seconds: int = Field(
        default=1800, description="Seconds between repeat alerts per endpoint"
    )

    # Server host for user-facing URLs (set to Tailscale IP for remote access)
    server_host: str = Field(
        default="localhost",
        description="Hostname/IP for service URLs in hub page and Discord messages",
    )

    # API cost alerting
    api_cost_alert_threshold: float = Field(
        default=1.0, description="Daily API spend ($) threshold for Discord alert"
    )

    # Daily briefing settings
    briefing_hour: int = Field(
        default=7, description="Hour (0-23) to send daily briefing"
    )
    briefing_enabled: bool = Field(
        default=True, description="Enable/disable the daily briefing"
    )

    # Multi-project settings (defaults = Technomancer, backward compatible)
    project_name: str = Field(
        default="technomancer", description="Active project name for AIM/Worker context"
    )
    project_root: str = Field(
        default="", description="Root directory of the target project repo (empty = auto-detect)"
    )
    deploy_cmd: str = Field(
        default="", description="Deploy command for external projects (empty = safe_update workflow)"
    )
    aim_state_dir: str = Field(
        default="", description="Directory for AIM state/lock/pid files (empty = aim/ default)"
    )
    test_command: str = Field(
        default="",
        description=(
            "Shell-style command the executor runs to validate a branch "
            "before merging. Parsed with shlex. Empty = Technomancer "
            "default (pytest from local-agent/). Example for Django: "
            "'python manage.py test'."
        ),
    )
    test_cwd: str = Field(
        default="",
        description=(
            "Working directory for test_command, relative to project_root "
            "(or absolute). Empty = project_root/local-agent for the "
            "Technomancer default, or project_root when test_command is set."
        ),
    )
    aim_story_generation_enabled: bool = Field(
        default=True,
        description=(
            "Enable autonomous story generation (evergreen + CREATE_WORK). "
            "Set False for projects where stories are created manually only."
        ),
    )
    aim_story_generation_categories: str = Field(
        default="quality,performance,test",
        description=(
            "Comma-separated categories the evergreen generator may produce. "
            "Empty string = all categories allowed. Only checked when "
            "aim_story_generation_enabled is True."
        ),
    )

    # AIM (AI Manager) settings
    aim_cycle_interval: int = Field(
        default=180, description="Seconds between AIM decision cycles"
    )
    aim_board_low_threshold: int = Field(
        default=10, description="Generate new work when TODO count drops below this"
    )
    aim_board_high_threshold: int = Field(
        default=100, description="Stop generating work when TODO count exceeds this"
    )
    aim_worker_heartbeat_timeout: int = Field(
        default=300, description="Seconds without heartbeat before Worker is considered dead"
    )
    aim_execution_timeout: int = Field(
        default=2700, description="Seconds before Worker considers an execution stuck (45 min)"
    )
    aim_max_worker_failures: int = Field(
        default=3, description="Consecutive Worker failures before Discord escalation"
    )
    aim_auto_approve_categories: str = Field(
        default="quality,performance,test",
        description="Comma-separated categories that AIM auto-approves",
    )
    aim_status_report_interval: int = Field(
        default=20, description="Send Discord status report every N cycles"
    )
    aim_queue_review_interval: int = Field(
        default=10, description="Run queue review (dedup + failure detection) every N cycles"
    )
    aim_brain_use_ollama: bool = Field(
        default=True,
        description=(
            "Route AIM brain's decide_next_action through local Ollama first. "
            "Falls back to claude -p on error, empty output, or unparseable JSON. "
            "generate_work_ideas still uses claude -p regardless."
        ),
    )
    stale_worktree_hours: float = Field(
        default=2.0,
        description=(
            "Worktrees not claimed by a live WorkerSlot.pid and older than this "
            "many hours are removed by the AIM housekeeping loop."
        ),
    )
    aim_worktree_cleanup_dry_run: bool = Field(
        default=False,
        description=(
            "If True, _cleanup_stale_worktrees logs intended removals without "
            "actually calling worktree_manager.remove_worktree."
        ),
    )

    # Claude rate-limit retry settings (used by the AI Worker)
    rate_limit_wait_minutes: int = Field(
        default=15,
        description=(
            "Initial wait (minutes) after a Claude rate-limit is detected "
            "before the Worker retries. Doubles on each retry up to 60 min."
        ),
    )
    rate_limit_max_retries: int = Field(
        default=3,
        description=(
            "Max rate-limit retries before the Worker gives up and marks "
            "the idea failed."
        ),
    )

    # Idea generator settings
    idea_generator_skip_threshold: int = Field(
        default=15,
        description=(
            "Skip the hourly idea_generator run when the board already has "
            "at least this many approved + proposed items ready to work. "
            "Saves the ~5000-token prompt when the backlog is healthy."
        ),
    )

    # Executor runtime settings
    executor_max_runtime_seconds: int = Field(
        default=1800,
        description=(
            "Wall-clock timeout (seconds) for a single claude -p executor run. "
            "On timeout the subprocess is sent SIGTERM, waits the grace period, "
            "then escalates to SIGKILL. Default 30 minutes. "
            "A hung Claude Code run would otherwise block the executor queue."
        ),
    )
    executor_sigterm_grace_seconds: int = Field(
        default=30,
        description=(
            "Seconds to wait after SIGTERM for the executor subprocess to exit "
            "cleanly before escalating to SIGKILL."
        ),
    )
    executor_summary_webhook: str = Field(
        default="",
        description=(
            "Discord webhook URL for per-run executor summary messages posted "
            "on completion. Empty string falls back to discord_webhook_url."
        ),
    )

    # AIMM daemon settings (env vars prefixed with AIMM_)
    aimm_cycle_interval: int = Field(
        default=600,
        description="Seconds between AIMM daemon decision cycles (env: AIMM_CYCLE_INTERVAL)",
    )
    aimm_max_approvals_per_cycle: int = Field(
        default=5,
        description="Maximum approvals AIMM issues in a single cycle (env: AIMM_MAX_APPROVALS_PER_CYCLE)",
    )
    aimm_max_approvals_per_day: int = Field(
        default=50,
        description="Rolling 24h cap on AIMM approvals (env: AIMM_MAX_APPROVALS_PER_DAY)",
    )
    aimm_approve_feature_security: bool = Field(
        default=True,
        description=(
            "Whether AIMM may auto-approve feature and security categories "
            "(env: AIMM_APPROVE_FEATURE_SECURITY)"
        ),
    )
    aimm_state_dir: str = Field(
        default="aimm",
        description="Directory for AIMM state/lock files (env: AIMM_STATE_DIR)",
    )

    # Fallback orchestrator settings
    fallback_max_errors: int = Field(
        default=5, description="Consecutive Claude API errors before fallback to Ollama"
    )
    fallback_latency_threshold: float = Field(
        default=30.0, description="Avg Claude API latency (seconds) that triggers fallback"
    )
    fallback_recovery_cooldown: int = Field(
        default=300, description="Seconds before probing Claude again after fallback activates"
    )

    @property
    def llm_memory_path(self) -> Path:
        """Path to LLM Memory folder within vault."""
        return self.vault_path / "LLM Memory"

    @property
    def permanent_path(self) -> Path:
        """Path to Permanent folder for persistent data."""
        return self.llm_memory_path / "Permanent"

    @property
    def context_path(self) -> Path:
        """Path to Context folder for rolling context."""
        return self.llm_memory_path / "Context"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings.
    """
    return Settings()  # type: ignore


# Convenience alias
settings = get_settings()
