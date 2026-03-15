from pathlib import Path

import pytest

from sl_rag.core.encryption_manager import EncryptionManager
from sl_rag.monitoring.monitoring_system import MonitoringSystem


def test_encrypted_monitoring_roundtrip(tmp_path):
    key_path = tmp_path / "master.key"
    db_path = tmp_path / "monitoring.db"
    enc = EncryptionManager(master_key_path=str(key_path))

    monitor = MonitoringSystem(
        db_path=str(db_path),
        encryption_manager=enc,
        encrypt_at_rest=True,
    )
    monitor.log_query(
        query="phase2 test query",
        domains=["security"],
        num_results=2,
        latency_ms=25.0,
        confidence=0.9,
    )

    assert db_path.exists()
    raw = db_path.read_bytes()
    # sqlite plaintext files start with "SQLite format 3"
    assert b"SQLite format 3" not in raw

    # Re-open with same key and verify data is readable.
    reopened = MonitoringSystem(
        db_path=str(db_path),
        encryption_manager=enc,
        encrypt_at_rest=True,
    )
    stats = reopened.get_query_stats(last_n=10)
    assert stats["total_queries"] >= 1


def test_encrypted_monitoring_requires_encryption_manager(tmp_path):
    with pytest.raises(ValueError):
        MonitoringSystem(
            db_path=str(tmp_path / "monitoring.db"),
            encrypt_at_rest=True,
        )
