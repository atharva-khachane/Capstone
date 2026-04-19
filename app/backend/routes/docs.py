"""
Document management routes:
  GET    /api/docs               — list PDFs in ./data
  POST   /api/docs/upload        — upload a PDF (admin only) → auto re-ingest
  DELETE /api/docs/{filename}    — delete a PDF  (admin only) → auto re-ingest
"""
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.backend.auth_dependencies import require_min_role

router = APIRouter()

DATA_DIR = Path("data")
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


def _safe_filename(name: str) -> str:
    """Reject names with path traversal components; normalise separators."""
    name = os.path.basename(name)
    if not name or name.startswith("."):
        raise HTTPException(400, "Invalid filename.")
    if not re.match(r"^[\w\s\-().]+\.pdf$", name, re.IGNORECASE):
        raise HTTPException(400, "Only PDF files are supported.")
    return name


def _trigger_reingest():
    """Kick off a background re-ingest (reuses the existing ingest machinery)."""
    from app.backend.routes.ingest import _run_ingest, _ingest_state, _lock
    with _lock:
        if _ingest_state["running"]:
            return  # already running — the ongoing ingest will pick up the new file
    thread = threading.Thread(target=_run_ingest, daemon=True, name="ingest-on-change")
    thread.start()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/docs")
def list_docs(_auth: dict = Depends(require_min_role("analyst"))):
    """Return metadata for every PDF in ./data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() == ".pdf" and path.is_file():
            stat = path.stat()
            docs.append({
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })
    return {"docs": docs}


@router.post("/docs/upload")
async def upload_doc(
    _auth: dict = Depends(require_min_role("admin")),
    file: UploadFile = File(...),
):
    """Upload a PDF and trigger re-ingestion. Admin only."""

    filename = _safe_filename(file.filename or "upload.pdf")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds the {MAX_FILE_SIZE // (1024*1024)} MB limit.")
    if not contents[:4] == b"%PDF":
        raise HTTPException(400, "File does not appear to be a valid PDF.")

    dest.write_bytes(contents)
    _trigger_reingest()
    return {
        "status": "uploaded",
        "filename": filename,
        "size_bytes": len(contents),
        "message": f"'{filename}' uploaded. Re-ingestion started in background.",
    }


@router.delete("/docs/{filename}")
def delete_doc(
    filename: str,
    _auth: dict = Depends(require_min_role("admin")),
):
    """Delete a PDF from ./data and trigger re-ingestion. Admin only."""

    safe_name = _safe_filename(filename)
    target = DATA_DIR / safe_name
    if not target.exists():
        raise HTTPException(404, f"'{safe_name}' not found in the data directory.")

    target.unlink()
    _trigger_reingest()
    return {
        "status": "deleted",
        "filename": safe_name,
        "message": f"'{safe_name}' deleted. Re-ingestion started in background.",
    }
