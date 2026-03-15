import sqlite3

from sl_rag.monitoring.monitoring_system import MonitoringSystem


def _security_event_types(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT event_type FROM security_events ORDER BY id")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def test_query_rate_spike_detection(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))

    for _ in range(3):
        monitor.log_query(
            query="q",
            domains=["finance"],
            num_results=1,
            latency_ms=5.0,
            confidence=0.5,
            user_id="user-a",
            role="analyst",
        )

    anomalies = monitor.analyze_query_patterns(
        user_id="user-a",
        domains=["finance"],
        window_seconds=120,
        rate_threshold=2,
    )
    assert any(a["type"] == "query_rate_spike" for a in anomalies)
    assert "query_rate_spike" in _security_event_types(db_path)


def test_repeated_sensitive_doc_access_detection(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))
    monitor.log_query("q1", ["security"], 1, 5.0, user_id="user-b")

    # Trigger repeated access count crossing threshold.
    monitor.analyze_query_patterns(
        user_id="user-b",
        domains=["security"],
        sensitive_doc_ids=["doc-sensitive-1"],
        sensitive_repeat_threshold=2,
    )
    anomalies = monitor.analyze_query_patterns(
        user_id="user-b",
        domains=["security"],
        sensitive_doc_ids=["doc-sensitive-1"],
        sensitive_repeat_threshold=2,
    )
    assert any(a["type"] == "repeated_sensitive_doc_access" for a in anomalies)
    assert "repeated_sensitive_doc_access" in _security_event_types(db_path)


def test_unusual_domain_access_detection(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))

    # Historical behavior: finance-only.
    for i in range(5):
        monitor.log_query(
            query=f"historical-{i}",
            domains=["finance"],
            num_results=1,
            latency_ms=10.0,
            confidence=0.7,
            user_id="user-c",
        )

    anomalies = monitor.analyze_query_patterns(
        user_id="user-c",
        domains=["aerospace"],
        min_history_for_domain_anomaly=3,
    )
    assert any(a["type"] == "unusual_domain_access_pattern" for a in anomalies)
    assert "unusual_domain_access_pattern" in _security_event_types(db_path)
