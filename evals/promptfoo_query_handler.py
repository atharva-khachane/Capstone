"""
Promptfoo Python provider for evaluating the initialized SL-RAG pipeline.

Configured via promptfooconfig.yaml as:
  providers:
    - id: file://promptfoo_query_handler.py
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent UnicodeEncodeError on Windows terminals when upstream modules print
# non-ASCII log characters.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Keep parity with the main eval pipeline configuration.
os.environ.setdefault("ENTAILMENT_SYNC", "true")
os.environ.setdefault("LLM_BACKEND", "api")

from app.backend.pipeline_instance import pipe  # noqa: E402

_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None


def get_or_init_pipeline():
    """Return the shared pipeline, ingesting once if needed."""
    global _PIPELINE
    if (
        _PIPELINE is not None
        and getattr(_PIPELINE, "_ready", False)
        and getattr(_PIPELINE, "pipeline", None) is not None
    ):
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = pipe
        if not getattr(_PIPELINE, "_ready", False):
            print("[PROMPTFOO_PROVIDER] Initializing pipeline ingest...", file=sys.stderr)
            _PIPELINE.ingest()

        if not getattr(_PIPELINE, "_ready", False):
            raise RuntimeError("Pipeline ingest did not reach ready state.")
        if getattr(_PIPELINE, "pipeline", None) is None:
            raise RuntimeError("Retriever not initialized: pipeline.pipeline is None")

    return _PIPELINE


def _extract_query(prompt: Any, context: Dict[str, Any]) -> str:
    """Prefer test vars.question; fall back to prompt text parsing."""
    vars_payload = context.get("vars", {}) if isinstance(context, dict) else {}
    if isinstance(vars_payload, dict):
        question = vars_payload.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()

    if isinstance(prompt, str):
        candidate = prompt.strip()
        if not candidate:
            return ""

        try:
            parsed = json.loads(candidate)
        except Exception:
            return candidate

        if isinstance(parsed, list):
            for message in reversed(parsed):
                if not isinstance(message, dict):
                    continue
                if str(message.get("role", "")).lower() != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

        return candidate

    return str(prompt or "").strip()


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Promptfoo Python provider entrypoint."""
    del options  # Config not used right now.
    started = time.time()

    try:
        query = _extract_query(prompt, context)
        if not query:
            return {"output": "", "error": "No query text was provided to provider."}

        print(f"[PROMPTFOO_PROVIDER] Received query: {query[:180]}", file=sys.stderr)

        pipeline = get_or_init_pipeline()
        retriever_loaded = getattr(pipeline, "pipeline", None) is not None
        print(f"[PROMPTFOO_PROVIDER] Vector store loaded: {retriever_loaded}", file=sys.stderr)

        if not retriever_loaded:
            return {"output": "", "error": "Vector store is not loaded."}

        response = pipeline.query(
            query,
            top_k=5,
            enable_reranking=True,
            generate_answer=True,
            auth_context={
                "user_id": "promptfoo_eval",
                "role": "analyst",
                "session_id": "promptfoo",
            },
        )

        if isinstance(response, dict) and response.get("error"):
            return {"output": "", "error": str(response["error"])}

        answer = ""
        confidence = 0.0
        injection_blocked = False
        if isinstance(response, dict):
            answer = str(response.get("answer") or "")
            injection_blocked = bool(response.get("injection_blocked"))
            try:
                confidence = float(response.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0

        latency_ms = int((time.time() - started) * 1000)
        return {
            "output": answer,
            "latencyMs": latency_ms,
            "metadata": {
                "injection_blocked": injection_blocked,
                "confidence": confidence,
            },
        }

    except Exception as exc:
        return {"output": "", "error": f"Promptfoo pipeline provider failure: {exc}"}
