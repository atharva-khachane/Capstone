import sqlite3

from sl_rag.core.schemas import Chunk
from sl_rag.monitoring.monitoring_system import MonitoringSystem
from sl_rag.security.auth import RBACManager


def test_rbac_document_access_matrix():
    rbac = RBACManager()
    restricted_meta = {"sensitivity": "restricted"}
    public_meta = {"sensitivity": "public"}

    assert rbac.can_query("admin")
    assert rbac.can_query("guest")
    assert not rbac.can_query("unknown_role")

    assert rbac.can_access_document("admin", restricted_meta)
    assert rbac.can_access_document("auditor", restricted_meta)
    assert not rbac.can_access_document("analyst", restricted_meta)
    assert not rbac.can_access_document("guest", restricted_meta)
    assert rbac.can_access_document("guest", public_meta)


def test_monitoring_logs_identity_fields(tmp_path):
    db_path = tmp_path / "monitoring.db"
    monitor = MonitoringSystem(str(db_path))

    monitor.log_query(
        query="test query",
        domains=["finance"],
        num_results=1,
        latency_ms=10.0,
        confidence=0.8,
        user_id="u1",
        role="analyst",
        session_id="s1",
    )
    monitor.log_document_access(
        doc_id="doc-1",
        chunk_ids=["chunk-1"],
        user_id="u1",
        role="analyst",
        session_id="s1",
    )

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT user_id, role, session_id FROM query_log ORDER BY id DESC LIMIT 1")
    q_row = cur.fetchone()
    cur.execute("SELECT user_id, role, session_id FROM document_access_log ORDER BY id DESC LIMIT 1")
    d_row = cur.fetchone()
    conn.close()

    assert q_row == ("u1", "analyst", "s1")
    assert d_row == ("u1", "analyst", "s1")


def test_rbac_filters_restricted_sources_for_guest():
    restricted_chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        content="Restricted content",
        chunk_index=0,
        start_char=0,
        end_char=18,
        token_count=3,
        domain="security",
        metadata={"source_document": "restricted.pdf", "sensitivity": "restricted"},
    )
    public_chunk = Chunk(
        chunk_id="c2",
        doc_id="d2",
        content="Public content",
        chunk_index=1,
        start_char=0,
        end_char=13,
        token_count=2,
        domain="public",
        metadata={"source_document": "public.pdf", "sensitivity": "public"},
    )

    rbac = RBACManager()
    results = [(restricted_chunk, 0.91), (public_chunk, 0.72)]
    filtered = rbac.filter_accessible_results("guest", results)

    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "d2"
