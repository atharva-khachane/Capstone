"""
Ingest routes:
  POST /api/ingest  — trigger re-ingestion in a background thread
  GET  /api/status  — return pipeline ready state + last ingest stats
"""
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from app.backend.auth_dependencies import require_min_role
from app.backend.pipeline_instance import pipe

router = APIRouter()

# Module-level state shared across requests
_ingest_state: Dict[str, Any] = {
    "running": False,
    "ingested_at": None,
    "docs": 0,
    "chunks": 0,
    "domains": {},
    "error": None,
}
_lock = threading.Lock()


def _run_ingest():
    with _lock:
        _ingest_state["running"] = True
        _ingest_state["error"] = None
    try:
        result = pipe.ingest()
        with _lock:
            _ingest_state["running"] = False
            _ingest_state["ingested_at"] = datetime.now().isoformat()
            _ingest_state["docs"] = result.get("documents", 0)
            _ingest_state["chunks"] = result.get("chunks", 0)
            _ingest_state["domains"] = result.get("domains", {})
    except Exception as exc:
        with _lock:
            _ingest_state["running"] = False
            _ingest_state["error"] = str(exc)


@router.post("/ingest")
def trigger_ingest(_auth: dict = Depends(require_min_role("admin"))):
    """Start a background ingest. Returns immediately."""
    with _lock:
        if _ingest_state["running"]:
            return {"status": "already_running", "message": "Ingest already in progress."}

    thread = threading.Thread(target=_run_ingest, daemon=True)
    thread.start()
    return {"status": "started", "message": "Ingest started in background."}


@router.get("/status")
def get_status():
    """Return pipeline readiness + last ingest stats."""
    with _lock:
        state = dict(_ingest_state)

    # Extract domain distribution from nested domain result
    domain_dist: Optional[Dict] = None
    domains_raw = state.get("domains")
    if isinstance(domains_raw, dict):
        domain_dist = domains_raw.get("domain_distribution", domains_raw)

    return {
        "ready": pipe._ready,
        "ingest_running": state["running"],
        "ingested_at": state["ingested_at"],
        "docs": state["docs"],
        "chunks": state["chunks"],
        "domains": domain_dist,
        "error": state["error"],
    }
