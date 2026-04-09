"""Tests for the infra_monitor module — vault backup, GPU health, WAL, security news."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.infra_monitor import (
    backup_vault,
    check_api_keys,
    check_gpu_health,
    detect_vault_changes,
    get_infra_report,
    get_infra_tools,
    init_wal_db,
    retry_pending_writes,
    scan_news_for_security,
    wal_write,
)


@pytest.fixture(autouse=True)
def _use_temp_dirs(tmp_path, monkeypatch):
    """Redirect all paths to temp."""
    monkeypatch.setattr("agent.infra_monitor.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("agent.infra_monitor.WAL_DB_PATH", tmp_path / "data" / "vault_wal.db")
    monkeypatch.setattr("agent.infra_monitor.BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr("agent.infra_monitor.VAULT_PATH", tmp_path / "vault")
    import agent.infra_monitor as mod
    if hasattr(mod._local, "wal_conn"):
        try:
            mod._local.wal_conn.close()
        except Exception:
            pass
        del mod._local.wal_conn
    mod._vault_hashes.clear()


class TestWalWrite:
    def test_successful_write(self, tmp_path):
        init_wal_db()
        target = tmp_path / "vault" / "test.md"
        result = wal_write(str(target), "hello world")
        assert result is True
        assert target.read_text() == "hello world"

    def test_failed_write_stays_pending(self, tmp_path):
        init_wal_db()
        # Write to an impossible path
        result = wal_write("/nonexistent/path/\x00/file.md", "content")
        assert result is False

    def test_retry_recovers(self, tmp_path):
        init_wal_db()
        target = tmp_path / "vault" / "retry.md"
        # Simulate a failed write by inserting a pending entry manually
        import agent.infra_monitor as mod
        conn = mod._get_wal_conn()
        conn.execute(
            "INSERT INTO vault_wal (filepath, content, status, created_at) VALUES (?, ?, 'failed', ?)",
            (str(target), "recovered content", "2026-04-09T12:00:00"),
        )
        conn.commit()

        recovered = retry_pending_writes()
        assert recovered == 1
        assert target.read_text() == "recovered content"


class TestVaultBackup:
    def test_creates_backup(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "test.md").write_text("content")

        result = backup_vault()
        assert result is not None
        assert result.endswith(".zip")
        assert Path(result).exists()

    def test_no_vault_returns_none(self, tmp_path):
        # vault doesn't exist
        result = backup_vault()
        assert result is None


class TestVaultChangeDetection:
    def test_detects_new_files(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("hello")

        changes = detect_vault_changes()
        assert "note.md" in changes

    def test_no_change_on_second_scan(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("hello")

        detect_vault_changes()  # first scan
        changes = detect_vault_changes()  # second scan
        assert changes == []

    def test_detects_modification(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        f = vault / "note.md"
        f.write_text("v1")
        detect_vault_changes()

        f.write_text("v2")
        changes = detect_vault_changes()
        assert "note.md" in changes


class TestGpuHealth:
    @patch("subprocess.run")
    def test_parses_nvidia_smi(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5080, 55, 4096, 16384, 30",
        )
        health = check_gpu_health()
        assert health["available"] is True
        assert health["temperature_c"] == 55
        assert health["memory_used_mb"] == 4096
        assert health["utilization_percent"] == 30

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_no_nvidia_smi(self, mock_run):
        health = check_gpu_health()
        assert health["available"] is False


class TestApiKeyCheck:
    def test_keys_present(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.discord_bot_token = "a" * 50
        mock_settings.anthropic_api_key = "sk-ant-" + "b" * 50
        mock_settings.discord_webhook_url = "https://discord.com/api/webhooks/..."
        monkeypatch.setattr("agent.infra_monitor.settings", mock_settings)

        keys = check_api_keys()
        assert keys["discord_bot_token"] == "ok"
        assert keys["anthropic_api_key"] == "ok"
        assert keys["discord_webhook"] == "ok"

    def test_missing_keys(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.discord_bot_token = ""
        mock_settings.anthropic_api_key = None
        mock_settings.discord_webhook_url = ""
        monkeypatch.setattr("agent.infra_monitor.settings", mock_settings)

        keys = check_api_keys()
        assert keys["discord_bot_token"] == "missing"
        assert keys["anthropic_api_key"] == "missing"


class TestSecurityNewsScan:
    def test_detects_security_article(self):
        articles = [
            {"title": "New GPU vulnerability allows RCE on Nvidia cards", "summary": "exploit found", "source": "Ars"},
        ]
        alerts = scan_news_for_security(articles)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"  # affects nvidia (our stack)

    def test_general_security_is_info(self):
        articles = [
            {"title": "Apple patches zero-day vulnerability", "summary": "iOS exploit", "source": "TC"},
        ]
        alerts = scan_news_for_security(articles)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"  # doesn't affect our stack

    def test_no_security_articles(self):
        articles = [
            {"title": "New JavaScript framework released", "summary": "Another one", "source": "HN"},
        ]
        alerts = scan_news_for_security(articles)
        assert alerts == []


class TestInfraReport:
    @patch("agent.infra_monitor.check_gpu_health", return_value={"available": False, "error": "not found"})
    @patch("agent.infra_monitor.check_api_keys", return_value={"discord_bot_token": "ok"})
    def test_generates_report(self, mock_keys, mock_gpu):
        report = get_infra_report()
        assert "Infrastructure Health Report" in report
        assert "GPU" in report
        assert "API Keys" in report


class TestGetTools:
    def test_returns_tools(self):
        tools = get_infra_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "infra_health" in names
        assert "vault_backup" in names
