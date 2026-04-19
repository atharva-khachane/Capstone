"""
TruLens-style RAG Triad Evaluation for SL-RAG + Trust-RAG Pipeline.

Reads pipeline outputs from eval_paper_results.json (produced by eval_extended.py)
and computes the RAG Triad using dense embedding cosine similarity via
sentence-transformers (all-MiniLM-L6-v2).

RAG Triad Metrics:
    1. Groundedness      — cosine_sim(answer, combined_context)
    2. Context Relevance — cosine_sim(question, each context chunk) averaged
    3. Answer Relevance  — cosine_sim(question, answer)

Using embedding-based scoring (not LLM-as-a-judge) so that the evaluation
works reliably with any local model and is reproducible.

Usage:
    python eval_trulens.py

Requirements:
    pip install sentence-transformers
    eval_paper_results.json must exist (run eval_extended.py first)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Validate prerequisite ─────────────────────────────────────────────────────
if not Path("eval_paper_results.json").exists():
    print("[ERROR] eval_paper_results.json not found.")
    print("        Run eval_extended.py first to generate pipeline outputs.")
    sys.exit(1)

with open("eval_paper_results.json", encoding="utf-8") as f:
    paper_results = json.load(f)

per_query = paper_results.get("per_query", [])
if not per_query:
    print("[ERROR] No per_query records found in eval_paper_results.json")
    sys.exit(1)

print(f"[TRULENS] Loaded {len(per_query)} queries from eval_paper_results.json")

# ── Load embedding model ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = "all-MiniLM-L6-v2"
    print(f"[TRULENS] Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[TRULENS] Embedding model ready OK")
except ImportError:
    print("[ERROR] sentence-transformers not installed.")
    print("        Run: pip install sentence-transformers")
    sys.exit(1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(texts: list) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ── Evaluate each query ───────────────────────────────────────────────────────
print(f"\n[TRULENS] Evaluating {len(per_query)} queries (RAG Triad, embedding-based)...\n")

per_sample_results   = []
groundedness_scores:      list = []
context_relevance_scores: list = []
answer_relevance_scores:  list = []

t_start = time.time()

for i, item in enumerate(per_query, 1):
    question = item["question"]
    answer   = (item.get("answer") or "").strip() or "No answer generated."
    contexts = item.get("contexts") or []
    item_id  = item["id"]

    if not contexts:
        contexts = ["No context retrieved."]

    print(f"  [{i}/{len(per_query)}] {item_id}...", end="", flush=True)

    q_emb = embed(question)
    a_emb = embed(answer)

    # 1. Groundedness — how well the answer is supported by the context
    combined_ctx = " ".join(contexts)
    ctx_emb      = embed(combined_ctx)
    gnd          = round(cosine_sim(a_emb, ctx_emb), 4)
    groundedness_scores.append(gnd)

    # 2. Context Relevance — avg similarity of each chunk to the question
    ctx_embs   = embed_batch(contexts)
    cr_scores  = [cosine_sim(q_emb, ce) for ce in ctx_embs]
    avg_cr     = round(float(np.mean(cr_scores)), 4)
    context_relevance_scores.append(avg_cr)

    # 3. Answer Relevance — how well the answer addresses the question
    ar = round(cosine_sim(q_emb, a_emb), 4)
    answer_relevance_scores.append(ar)

    print(f"  Gnd={gnd:.3f}  CR={avg_cr:.3f}  AR={ar:.3f}")

    per_sample_results.append({
        "id":               item_id,
        "question":         question,
        "groundedness":     gnd,
        "context_relevance": avg_cr,
        "answer_relevance": ar,
    })

eval_time = round(time.time() - t_start, 1)


def _avg(lst: list):
    return round(sum(lst) / len(lst), 4) if lst else None


aggregate = {
    "groundedness":      _avg(groundedness_scores),
    "context_relevance": _avg(context_relevance_scores),
    "answer_relevance":  _avg(answer_relevance_scores),
}

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "eval_timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    "method":            f"embedding-cosine ({EMBED_MODEL})",
    "num_samples":       len(per_query),
    "eval_time_seconds": eval_time,
    "aggregate":         aggregate,
    "per_sample":        per_sample_results,
}

with open("eval_trulens_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  TRULENS RAG TRIAD SUMMARY  (embedding: {EMBED_MODEL})")
print("=" * 60)
print(f"  {'Groundedness':<35}: {aggregate['groundedness']}")
print(f"  {'Context Relevance':<35}: {aggregate['context_relevance']}")
print(f"  {'Answer Relevance':<35}: {aggregate['answer_relevance']}")
print(f"\n  Samples evaluated : {len(per_query)}")
print(f"  Eval time         : {eval_time}s")
print("=" * 60)
print("\n[TRULENS] Results saved -> eval_trulens_results.json")
print("[TRULENS] Run eval_visualize.py next to generate all figures.")
