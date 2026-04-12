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
    ollama_vision_model: str = Field(
        default="llava-llama3", description="Ollama model for vision/image analysis"
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
