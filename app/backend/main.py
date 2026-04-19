"""
SL-RAG FastAPI backend.

Startup sequence (lifespan):
  1. pipeline_instance.pipe is imported (models load here).
  2. pipe.ingest() indexes all PDFs in ./data.

Then the server accepts requests on :8000.
"""
from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sl_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Loading SL-RAG pipeline …")
    from app.backend.pipeline_instance import pipe  # noqa: F401
    from app.backend.routes.ingest import _run_ingest
    import threading
    # Start ingest in a daemon thread so the server accepts requests immediately.
    # Clients can poll GET /api/status to track readiness.
    t = threading.Thread(target=_run_ingest, daemon=True, name="ingest-startup")
    t.start()
    logger.info("[STARTUP] Ingest started in background — server is ready for requests.")
    yield
    logger.info("[SHUTDOWN] Bye.")


app = FastAPI(
    title="SL-RAG API",
    description="Secure RAG pipeline for ISRO government documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.backend.routes import auth, query, ingest, audit, docs  # noqa: E402

app.include_router(auth.router,   prefix="/api/auth",   tags=["auth"])
app.include_router(query.router,  prefix="/api",        tags=["query"])
app.include_router(ingest.router, prefix="/api",        tags=["ingest"])
app.include_router(audit.router,  prefix="/api",        tags=["audit"])
app.include_router(docs.router,   prefix="/api",        tags=["docs"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
