"""
Query routes:
  POST /api/query         — standard (waits for full answer)
  POST /api/query/stream  — SSE streaming (yields tokens in real-time)
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.backend.auth_dependencies import get_auth_context
from app.backend.pipeline_instance import pipe

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    user_id: str = "anonymous"
    role: str = "guest"
    session_id: str = ""
    top_k: int = Field(default=5, ge=1, le=10)
    enable_reranking: bool = True
    generate_answer: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    hallucination_risk: str
    citation_quality: str
    injection_blocked: bool
    latency: dict
    num_results: int
    sources: list
    domain: Optional[str] = None


@router.post("/query", response_model=QueryResponse)
def run_query(
    req: QueryRequest,
    auth_context: dict = Depends(get_auth_context),
):
    if not pipe._ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready — ingest has not completed yet.",
        )

    result = pipe.query(
        question=req.question,
        top_k=req.top_k,
        enable_reranking=req.enable_reranking,
        generate_answer=req.generate_answer,
        auth_context=auth_context,
    )

    if "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])

    # Derive top domain from sources list for the badge
    sources = result.get("sources", [])
    domain = None
    if sources:
        domain_counts: dict = {}
        for src in sources:
            d = src.get("domain", "")
            if d:
                domain_counts[d] = domain_counts.get(d, 0) + 1
        if domain_counts:
            domain = max(domain_counts, key=domain_counts.get)

    return QueryResponse(
        query=result.get("query", req.question),
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        hallucination_risk=result.get("hallucination_risk", "unknown"),
        citation_quality=result.get("citation_quality", ""),
        injection_blocked=result.get("injection_blocked", False),
        latency=result.get("latency", {}),
        num_results=result.get("num_results", 0),
        sources=sources,
        domain=domain,
    )


@router.post("/query/stream")
def run_query_stream(
    req: QueryRequest,
    auth_context: dict = Depends(get_auth_context),
):
    """SSE endpoint — yields tokens as they are generated, then metadata."""
    if not pipe._ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready — ingest has not completed yet.",
        )

    def event_generator():
        for event in pipe.query_stream(
            question=req.question,
            top_k=req.top_k,
            enable_reranking=req.enable_reranking,
            auth_context=auth_context,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
