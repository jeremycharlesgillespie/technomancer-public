"""Tests for idea_board.gpu_monitor — GPU sample logging + query."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board import gpu_monitor


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "gpu.db"
    gpu_monitor._init_db(db)
    return db


def test_init_db_creates_schema(tmp_path: Path) -> None:
    db = tmp_path / "gpu.db"
    gpu_monitor._init_db(db)
    assert db.exists()
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "gpu_samples" in tables


def test_record_and_retrieve_sample(tmp_db: Path) -> None:
    gpu_monitor.record_sample(tmp_db, 1000, 42.5, 8000, 16384)
    samples = gpu_monitor.get_samples(tmp_db)
    assert len(samples) == 1
    assert samples[0]["ts"] == 1000
    assert samples[0]["utilization"] == 42.5
    assert samples[0]["mem_used_mb"] == 8000
    assert samples[0]["mem_total_mb"] == 16384


def test_get_samples_filters_by_range(tmp_db: Path) -> None:
    for ts in (100, 200, 300, 400, 500):
        gpu_monitor.record_sample(tmp_db, ts, 10.0, 1000, 16384)
    samples = gpu_monitor.get_samples(tmp_db, start_ts=200, end_ts=400)
    assert [s["ts"] for s in samples] == [200, 300, 400]


def test_get_samples_downsamples_when_over_max_points(tmp_db: Path) -> None:
    for ts in range(1000, 1100):
        gpu_monitor.record_sample(tmp_db, ts, 50.0, 1000, 16384)
    samples = gpu_monitor.get_samples(tmp_db, max_points=10)
    assert len(samples) <= 20
    assert samples[-1]["ts"] == 1099


def test_get_samples_empty_when_no_db() -> None:
    assert gpu_monitor.get_samples(Path("/nonexistent/gpu.db")) == []


def test_prune_old_deletes_stale(tmp_db: Path) -> None:
    for ts in (100, 200, 1_000_000):
        gpu_monitor.record_sample(tmp_db, ts, 10.0, 1000, 16384)
    deleted = gpu_monitor.prune_old(tmp_db, now_ts=1_000_000, retention_days=1)
    assert deleted == 2
    samples = gpu_monitor.get_samples(tmp_db)
    assert [s["ts"] for s in samples] == [1_000_000]


def test_sample_gpu_parses_csv() -> None:
    mock_result = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0,
        stdout="55, 8192, 16384\n", stderr="",
    )
    with patch("idea_board.gpu_monitor.subprocess.run", return_value=mock_result):
        assert gpu_monitor.sample_gpu() == (55.0, 8192, 16384)


def test_sample_gpu_returns_none_on_missing_binary() -> None:
    with patch("idea_board.gpu_monitor.subprocess.run", side_effect=FileNotFoundError()):
        assert gpu_monitor.sample_gpu() is None


def test_sample_gpu_returns_none_on_nonzero_exit() -> None:
    mock_result = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=9, stdout="", stderr="no gpu",
    )
    with patch("idea_board.gpu_monitor.subprocess.run", return_value=mock_result):
        assert gpu_monitor.sample_gpu() is None


def test_sample_gpu_returns_none_on_bad_output() -> None:
    mock_result = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="not,valid\n", stderr="",
    )
    with patch("idea_board.gpu_monitor.subprocess.run", return_value=mock_result):
        assert gpu_monitor.sample_gpu() is None


def test_insert_or_replace_preserves_primary_key(tmp_db: Path) -> None:
    gpu_monitor.record_sample(tmp_db, 100, 10.0, 1000, 16384)
    gpu_monitor.record_sample(tmp_db, 100, 90.0, 2000, 16384)
    samples = gpu_monitor.get_samples(tmp_db)
    assert len(samples) == 1
    assert samples[0]["utilization"] == 90.0
