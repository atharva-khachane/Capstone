import sqlite3

from sl_rag.monitoring.monitoring_system import MonitoringSystem


def _security_events(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT event_type FROM security_events ORDER BY id")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def test_drift_baseline_then_no_drift(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))

    baseline = {"domain_a": [1.0, 0.0], "domain_b": [0.0, 1.0]}
    res1 = monitor.check_domain_drift(baseline, threshold=0.1)
    assert res1["drift_detected"] is False

    res2 = monitor.check_domain_drift(baseline, threshold=0.1)
    assert res2["drift_detected"] is False
    assert res2["domains_compared"] == 2


def test_drift_detection_emits_security_event(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))

    monitor.check_domain_drift({"domain_a": [1.0, 0.0]}, threshold=0.1)
    # Opposite vector -> cosine distance ~= 2.0
    result = monitor.check_domain_drift({"domain_a": [-1.0, 0.0]}, threshold=0.1)

    assert result["drift_detected"] is True
    assert "domain_embedding_drift_detected" in _security_events(db_path)
