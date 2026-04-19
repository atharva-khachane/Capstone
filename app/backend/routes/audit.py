"""
Audit route — GET /api/audit
Returns paginated query log + audit chain integrity status.
"""
import sqlite3
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from app.backend.auth_dependencies import require_min_role
from app.backend.pipeline_instance import pipe

router = APIRouter()

_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "storage", "audit_logs", "monitoring.db")
)


def _get_db_path() -> str:
    """Use the monitoring system's resolved DB path if available."""
    monitor = getattr(pipe, "monitor", None) or getattr(pipe, "_monitor", None)
    if monitor:
        rt = str(getattr(monitor, "_runtime_db_path", ""))
        if rt and os.path.exists(rt):
            return rt
    return _DB_PATH


@router.get("/audit")
def get_audit(
    _auth: dict = Depends(require_min_role("auditor")),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Return paginated query log entries and audit chain status."""
    monitor = getattr(pipe, "monitor", None) or getattr(pipe, "_monitor", None)

    # Audit chain integrity
    chain_status: Dict = {"valid": False, "entries": 0, "broken_at": None}
    if monitor:
        try:
            chain_status = monitor.verify_audit_chain()
        except Exception:
            pass

    # Query stats
    query_stats: Dict = {}
    if monitor:
        try:
            query_stats = monitor.get_query_stats(last_n=200)
        except Exception:
            pass

    # Raw paginated query log
    entries: List[Dict] = []
    db_path = _get_db_path()
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(
                "SELECT id, timestamp, query_text, user_id, role, domains, "
                "num_results, latency_ms, confidence FROM query_log "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = c.fetchall()
            c.execute("SELECT COUNT(*) FROM query_log")
            total = c.fetchone()[0]
            conn.close()
            entries = [dict(r) for r in rows]
        except Exception as exc:
            entries = []
            total = 0
    else:
        total = 0

    return {
        "audit_chain": chain_status,
        "query_stats": query_stats,
        "entries": entries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
