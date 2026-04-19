"""
Fresh Comprehensive Evaluation for SL-RAG + Trust-RAG Pipeline.

Loads evaluation queries from slrag_expanded_queries_v1.csv (235 queries).

Metrics computed (all fresh — no cached results used):
  Generation Quality:
    1.  BLEU-1, BLEU-4          — n-gram overlap with ground truth
    2.  ROUGE-L                  — longest common subsequence overlap
    3.  BERTScore F1             — semantic similarity via contextual embeddings
    4.  Cosine Similarity        — embedding-space similarity (all-MiniLM-L6-v2)
  Retrieval Quality:
    5.  MRR                      — Mean Reciprocal Rank
    6.  NDCG@3, NDCG@5           — Normalised Discounted Cumulative Gain
    7.  Precision@3, Precision@5 — Fraction of top-k results that are relevant
    8.  Recall@3, Recall@5       — Whether any top-k result is relevant
  Trust & Safety:
    9.  ECE                      — Expected Calibration Error of confidence
    10. Hallucination Rate        — % answers flagged as hallucination_risk=high
    11. Avg Trust Score           — mean composite trust across sources
    12. Avg Faithfulness          — mean claim-level faithfulness (Trust-RAG)
    13. ASI Detection Accuracy    — adversarial query detection (dynamic threshold)
  Performance:
    14. Avg Latency               — mean total and retrieval latency
  Ablation:
    15. BM25-only / Dense-only / Hybrid MRR, NDCG@5, P@5, R@5

Usage:
    python eval_extended.py

Requirements:
    pip install bert-score rouge-score sentence-transformers nltk
"""

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ['ENTAILMENT_SYNC'] = 'true'
os.environ['LLM_BACKEND'] = 'api'

assert os.environ.get("ENTAILMENT_SYNC") == "true", \
    "ENTAILMENT_SYNC must be 'true' for eval runs -- set it before importing pipeline"

logger = logging.getLogger(__name__)
logger.info(
    f"Eval config: ENTAILMENT_SYNC={os.environ['ENTAILMENT_SYNC']}, "
    f"LLM_BACKEND={os.environ.get('LLM_BACKEND', 'api')}, queries=234"
)

# Silence heavy startup noise
sys.stderr = open(os.devnull, "w")

try:
    import torch
    _torch_ok = True
except ImportError:
    _torch_ok = False

from sl_rag.pipeline import SLRAGPipeline
from sl_rag.calibrated_confidence import compute_ece as _compute_ece_helper

sys.stderr = sys.__stderr__

# ── Optional dep imports ───────────────────────────────────────────────────────
try:
    from bert_score import score as bert_score_fn
    _bert_ok = True
    print("[EVAL] BERTScore [OK]")
except ImportError:
    _bert_ok = False
    print("[EVAL] bert-score not installed — run: pip install bert-score")

try:
    from rouge_score import rouge_scorer as rouge_lib
    _rouge_ok = True
    _rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    print("[EVAL] ROUGE [OK]")
except ImportError:
    _rouge_ok = False
    print("[EVAL] rouge-score not installed — run: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _bleu_ok = True
    print("[EVAL] BLEU [OK]")
except ImportError:
    _bleu_ok = False
    print("[EVAL] nltk not installed — run: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    _cos_ok = True
    print("[EVAL] Cosine Similarity (all-MiniLM-L6-v2) [OK]")
except ImportError:
    _cos_ok = False
    print("[EVAL] sentence-transformers not installed")

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("\n[EVAL] Loading SL-RAG pipeline (with LLM for generation)...")
sys.stderr = open(os.devnull, "w")

pipe = SLRAGPipeline(
    data_dir="./data",
    storage_dir="./storage",
    config_path="./config/config.yaml",
    use_gpu=True,
    encryption=True,
    load_llm=True,
)
pipe.ingest()
sys.stderr = sys.__stderr__
print("[EVAL] Pipeline ready [OK]")

# ── Load dataset from CSV ─────────────────────────────────────────────────────
EVAL_CSV = "slrag_expanded_queries_v1.csv"
if not Path(EVAL_CSV).exists():
    print(f"[ERROR] {EVAL_CSV} not found. Place the evaluation CSV in the project root.")
    sys.exit(1)

eval_data: list = []
with open(EVAL_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_data.append({
            "id": row["query_id"],
            "question": row["query_text"],
            "ground_truth": row["reference_answer_brief"],
            "domain": row["category"],
            "subcategory": row.get("subcategory", ""),
            "difficulty": row.get("difficulty", ""),
            "source_document": row.get("source_document", ""),
            "query_type": row.get("query_type", "factual"),
            "is_adversarial": row.get("query_type", "") == "adversarial",
        })

normal_items     = [x for x in eval_data if not x.get("is_adversarial", False)]
adversarial_items = [x for x in eval_data if x.get("is_adversarial", False)]

# Optional focused regeneration for selected IDs, e.g. EVAL_ONLY_IDS=Q004,Q005
_only_ids_raw = os.getenv("EVAL_ONLY_IDS", "").strip()
if _only_ids_raw:
    only_ids = [x.strip() for x in _only_ids_raw.split(",") if x.strip()]
    if only_ids:
        wanted = set(only_ids)
        normal_items = [x for x in normal_items if x.get("id") in wanted]
        adversarial_items = [x for x in adversarial_items if x.get("id") in wanted]
        found_ids = {x.get("id") for x in (normal_items + adversarial_items)}
        missing_ids = [qid for qid in only_ids if qid not in found_ids]
        print(
            f"[EVAL] Focus mode via EVAL_ONLY_IDS: "
            f"{len(normal_items)} normal, {len(adversarial_items)} adversarial"
        )
        if missing_ids:
            print(f"[WARN] EVAL_ONLY_IDS not found in dataset: {', '.join(missing_ids)}")

print(f"[EVAL] Loaded {len(eval_data)} queries from {EVAL_CSV}")
print(f"[EVAL] {len(normal_items)} normal queries, {len(adversarial_items)} adversarial")

# ── Domain → source filename keyword map ─────────────────────────────────────
# Maps CSV category values to keywords expected in retrieved chunk source filenames.
DOMAIN_SOURCE_MAP = {
    "GFR-Budget":               ["gfr", "financial", "budget"],
    "GFR-Advances":             ["gfr", "financial", "advance"],
    "GFR-Delegation":           ["gfr", "financial", "dfpr", "delegation"],
    "GFR-Grants":               ["gfr", "financial", "grant"],
    "Proc-Goods":               ["goods", "procurement_goods", "procurement"],
    "Proc-QCBS":                ["consultancy", "procurement_consultancy", "qcbs"],
    "Proc-Consultancy":         ["consultancy", "procurement_consultancy"],
    "Proc-Security":            ["goods", "procurement_goods", "procurement"],
    "Proc-EMD":                 ["goods", "procurement_goods", "procurement"],
    "Proc-NoTender":            ["goods", "procurement_goods", "procurement"],
    "Proc-SingleBid":           ["goods", "procurement_goods", "procurement"],
    "Tech-Telemetry-DSN":       ["telemetry", "208b", "dsn", "deep_space"],
    "Tech-Telemetry-Dict":      ["telemetry", "20180000774", "dictionary"],
    "Tech-Arecibo":             ["arecibo", "20210017934", "nesc", "observatory"],
    "Tech-HelicopterPylon":     ["helicopter", "pylon", "20020050369", "fuel"],
    "Mixed-CrossDomain":        ["gfr", "financial", "goods", "consultancy", "telemetry",
                                 "arecibo", "helicopter", "20210017934", "20180000774",
                                 "20020050369", "208b", "procurement"],
}

def _source_matches(source_text: str, expected_domain: str) -> bool:
    txt = source_text.lower()
    for kw in DOMAIN_SOURCE_MAP.get(expected_domain, []):
        if kw in txt:
            return True
    exp_tokens = set(expected_domain.lower().replace("_", " ").split())
    generic    = {"rules", "report", "general", "document", "services"}
    sig_tokens = exp_tokens - generic or exp_tokens
    return bool(sig_tokens & set(txt.replace("_", " ").split()))


def _compute_ndcg(relevance_labels: list, k: int) -> float:
    """Binary NDCG@k where relevance_labels is ordered by retrieval rank."""
    labels = relevance_labels[:k]
    dcg  = sum(rel / math.log2(i + 2) for i, rel in enumerate(labels))
    n_rel = sum(relevance_labels)
    ideal = [1] * min(int(n_rel), k) + [0] * max(0, k - int(n_rel))
    idcg  = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


AUTH = {"user_id": "eval_extended", "role": "analyst", "session_id": "eval_ext"}

# ══════════════════════════════════════════════════════════════════════════════
# PART A — Normal queries with full generation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[EVAL] Running {len(normal_items)} normal queries with LLM generation...")

responses: list    = []
answers: list      = []
ground_truths: list = []

for i, item in enumerate(normal_items, 1):
    q = item["question"]
    print(f"  [{i}/{len(normal_items)}] {item['id']}...", end="", flush=True)

    if _torch_ok and torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        resp = pipe.query(
            q, top_k=5, enable_reranking=True,
            generate_answer=True, auth_context=AUTH,
        )
        answers.append(resp.get("answer") or "")
        ground_truths.append(item["ground_truth"])
        responses.append(resp)
        print(f" conf={resp.get('confidence', 0):.3f} "
              f"faith={resp.get('faithfulness_score', 'N/A')} "
              f"asi={resp.get('asi_score', 0):.3f}")
    except RuntimeError as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            print(" [CUDA OOM — retrying without generation]")
            if _torch_ok and torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                resp = pipe.query(
                    q, top_k=5, enable_reranking=True,
                    generate_answer=False, auth_context=AUTH,
                )
                resp["answer"] = "[CUDA error — answer unavailable]"
            except Exception:
                resp = {}
        else:
            print(f" ERROR: {e}")
            resp = {}
        answers.append(resp.get("answer") or "")
        ground_truths.append(item["ground_truth"])
        responses.append(resp)
    except Exception as e:
        print(f" ERROR: {e}")
        responses.append({})
        answers.append("")
        ground_truths.append(item["ground_truth"])

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Adversarial queries
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[EVAL] Running {len(adversarial_items)} adversarial queries...")
asi_results: list = []
for item in adversarial_items:
    try:
        resp = pipe.query(
            item["question"], top_k=3, enable_reranking=False,
            generate_answer=False, auth_context=AUTH,
        )
        blocked = resp.get("injection_blocked", False) or resp.get("asi_flagged", False)
        asi_results.append({
            "id": item["id"],
            "question": item["question"],
            "blocked": blocked,
            "asi_score": resp.get("asi_score", 0),
            "asi_flagged": resp.get("asi_flagged", False),
            "injection_blocked": resp.get("injection_blocked", False),
        })
    except Exception as e:
        asi_results.append({"id": item["id"], "blocked": False, "error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE METRICS
# ══════════════════════════════════════════════════════════════════════════════
metrics: dict = {}

# 1. BLEU
if _bleu_ok:
    smooth = SmoothingFunction().method1
    bleu1_scores, bleu4_scores = [], []
    for ans, gt in zip(answers, ground_truths):
        ref = [gt.lower().split()]
        hyp = ans.lower().split() if ans else [""]
        bleu1_scores.append(sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    metrics["bleu_1"] = round(sum(bleu1_scores) / len(bleu1_scores), 4)
    metrics["bleu_4"] = round(sum(bleu4_scores) / len(bleu4_scores), 4)
    print(f"  BLEU-1 = {metrics['bleu_1']:.4f}  BLEU-4 = {metrics['bleu_4']:.4f}")
else:
    metrics["bleu_1"] = metrics["bleu_4"] = None

# 2. ROUGE-L
if _rouge_ok and any(answers):
    rl_scores = [_rouge.score(gt, ans)["rougeL"].fmeasure for ans, gt in zip(answers, ground_truths) if ans]
    metrics["rouge_l"] = round(sum(rl_scores) / len(rl_scores), 4) if rl_scores else None
    print(f"  ROUGE-L = {metrics['rouge_l']:.4f}")
else:
    metrics["rouge_l"] = None

# 3. BERTScore
if _bert_ok and any(answers):
    print("  Computing BERTScore...")
    P, R, F1 = bert_score_fn(answers, ground_truths, lang="en", verbose=False)
    metrics["bertscore_f1"]        = round(float(F1.mean()), 4)
    metrics["bertscore_precision"] = round(float(P.mean()), 4)
    metrics["bertscore_recall"]    = round(float(R.mean()), 4)
    print(f"  BERTScore F1 = {metrics['bertscore_f1']:.4f}")
else:
    metrics["bertscore_f1"] = metrics["bertscore_precision"] = metrics["bertscore_recall"] = None

# 4. Cosine Similarity
if _cos_ok and any(answers):
    print("  Computing cosine similarity...")
    valid_pairs = [(a, g) for a, g in zip(answers, ground_truths) if a]
    ans_embs = _st_model.encode([p[0] for p in valid_pairs], convert_to_tensor=True)
    gt_embs  = _st_model.encode([p[1] for p in valid_pairs], convert_to_tensor=True)
    cos_scores = [float(cos_sim(a.unsqueeze(0), g.unsqueeze(0))) for a, g in zip(ans_embs, gt_embs)]
    metrics["cosine_similarity"] = round(sum(cos_scores) / len(cos_scores), 4)
    print(f"  Cosine Similarity = {metrics['cosine_similarity']:.4f}")
else:
    metrics["cosine_similarity"] = None

# 5–8. MRR, NDCG, Precision, Recall
mrr_scores, ndcg3, ndcg5, p3, p5, r3, r5 = [], [], [], [], [], [], []
per_query_bleu1  = [None] * len(normal_items)
per_query_bleu4  = [None] * len(normal_items)
per_query_rouge  = [None] * len(normal_items)
per_query_bert   = [None] * len(normal_items)
per_query_cos    = [None] * len(normal_items)

for idx, (item, resp) in enumerate(zip(normal_items, responses)):
    domain = item.get("domain", "")
    sources = resp.get("sources", [])
    src_texts = [(s.get("source", "") + " " + s.get("domain", "")).lower() for s in sources]
    relevance = [1 if _source_matches(t, domain) else 0 for t in src_texts]

    # MRR
    rr = 0.0
    for rank, rel in enumerate(relevance, 1):
        if rel:
            rr = 1.0 / rank
            break
    mrr_scores.append(rr)

    # NDCG
    ndcg3.append(_compute_ndcg(relevance, 3))
    ndcg5.append(_compute_ndcg(relevance, 5))

    # Precision@k
    p3.append(sum(relevance[:3]) / 3.0 if len(relevance) >= 3 else (sum(relevance) / max(len(relevance), 1)))
    p5.append(sum(relevance[:5]) / 5.0 if len(relevance) >= 5 else (sum(relevance) / max(len(relevance), 1)))

    # Recall@k
    r3.append(1.0 if any(relevance[:3]) else 0.0)
    r5.append(1.0 if any(relevance[:5]) else 0.0)

    # Per-query text metrics
    ans, gt = answers[idx], ground_truths[idx]
    if _bleu_ok and ans:
        smooth = SmoothingFunction().method1
        ref = [gt.lower().split()]
        hyp = ans.lower().split()
        per_query_bleu1[idx] = round(sentence_bleu(ref, hyp, weights=(1,0,0,0), smoothing_function=smooth), 4)
        per_query_bleu4[idx] = round(sentence_bleu(ref, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth), 4)
    if _rouge_ok and ans:
        per_query_rouge[idx] = round(_rouge.score(gt, ans)["rougeL"].fmeasure, 4)

metrics["mrr"]           = round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None
metrics["ndcg_at_3"]     = round(sum(ndcg3) / len(ndcg3), 4) if ndcg3 else None
metrics["ndcg_at_5"]     = round(sum(ndcg5) / len(ndcg5), 4) if ndcg5 else None
metrics["precision_at_3"] = round(sum(p3) / len(p3), 4) if p3 else None
metrics["precision_at_5"] = round(sum(p5) / len(p5), 4) if p5 else None
metrics["recall_at_3"]   = round(sum(r3) / len(r3), 4) if r3 else None
metrics["recall_at_5"]   = round(sum(r5) / len(r5), 4) if r5 else None

print(f"  MRR      = {metrics['mrr']:.4f}")
print(f"  NDCG@3   = {metrics['ndcg_at_3']:.4f}  NDCG@5   = {metrics['ndcg_at_5']:.4f}")
print(f"  P@3      = {metrics['precision_at_3']:.4f}  P@5      = {metrics['precision_at_5']:.4f}")
print(f"  Recall@3 = {metrics['recall_at_3']:.4f}  Recall@5 = {metrics['recall_at_5']:.4f}")

# Compute per-query cosine/BERTScore in a second pass (vectorised above)
if _cos_ok and any(answers):
    valid_idx = [i for i, a in enumerate(answers) if a]
    ans_embs = _st_model.encode([answers[i] for i in valid_idx], convert_to_tensor=True)
    gt_embs  = _st_model.encode([ground_truths[i] for i in valid_idx], convert_to_tensor=True)
    for j, i in enumerate(valid_idx):
        per_query_cos[i] = round(float(cos_sim(ans_embs[j].unsqueeze(0), gt_embs[j].unsqueeze(0))), 4)

if _bert_ok and any(answers):
    valid_idx = [i for i, a in enumerate(answers) if a]
    _, _, F1 = bert_score_fn(
        [answers[i] for i in valid_idx],
        [ground_truths[i] for i in valid_idx],
        lang="en", verbose=False,
    )
    for j, i in enumerate(valid_idx):
        per_query_bert[i] = round(float(F1[j]), 4)

# 9. ECE — rule-based confidence vs ROUGE-L accuracy (Fix 1)
N_BINS = 5
confs = [r.get("confidence", 0.5) for r in responses]
if _rouge_ok and any(answers):
    correctness = [
        _rouge.score(gt, ans)["rougeL"].fmeasure
        for ans, gt in zip(answers, ground_truths)
    ]

    metrics["ece"] = _compute_ece_helper(confs, correctness, N_BINS)

    conf_range = max(confs) - min(confs) if confs else 0
    print(f"  ECE (rule-based) = {metrics['ece']}")
    print(f"  Confidence range = {conf_range:.3f}  (min={min(confs):.3f}, max={max(confs):.3f})")
    if metrics["ece"] < 0.35:
        print(f"  [CALIBRATION] [OK] ECE < 0.35 target met")
    else:
        print(f"  [CALIBRATION] WARNING: ECE={metrics['ece']:.4f} still >= 0.35")
    if conf_range < 0.4:
        print(f"  [CALIBRATION] WARNING: confidence range {conf_range:.3f} < 0.4 target")
    else:
        print(f"  [CALIBRATION] [OK] confidence range >= 0.4")
else:
    metrics["ece"] = None
    confs = [0.5] * len(responses)
    print(f"  ECE = {metrics['ece']}")

# 10. Hallucination rate
high_hall = [r for r in responses if r.get("hallucination_risk") == "high"]
metrics["hallucination_rate"] = round(len(high_hall) / max(len(responses), 1), 4)
print(f"  Hallucination Rate = {metrics['hallucination_rate']:.4f}")

# 11. Avg Trust Score
trust_vals = []
for resp in responses:
    for src in resp.get("sources", []):
        tb = src.get("trust_breakdown", {})
        if "trust_score" in tb:
            trust_vals.append(tb["trust_score"])
metrics["avg_trust_score"] = round(sum(trust_vals) / len(trust_vals), 4) if trust_vals else None
print(f"  Avg Trust Score = {metrics['avg_trust_score']}")

# 12. Avg Faithfulness
faith_vals = [r.get("faithfulness_score") for r in responses if r.get("faithfulness_score") is not None]
metrics["avg_faithfulness"] = round(sum(faith_vals) / len(faith_vals), 4) if faith_vals else None
print(f"  Avg Faithfulness = {metrics['avg_faithfulness']}")

# 13. ASI Detection Accuracy (dynamic threshold)
normal_asi = [r.get("asi_score", 0) for r in responses if r.get("asi_score") is not None]
if normal_asi:
    import numpy as _np
    p95 = float(_np.percentile(normal_asi, 95))
    dyn_thresh = max(p95, max(normal_asi) * 1.02)
else:
    dyn_thresh = 1.8
for r in asi_results:
    r["dynamic_threshold"] = round(dyn_thresh, 4)
    r["asi_flagged_dynamic"] = r.get("asi_score", 0) > dyn_thresh
    if r["asi_flagged_dynamic"]:
        r["blocked"] = True
blocked_count = sum(1 for r in asi_results if r.get("blocked", False))
metrics["asi_detection_accuracy"] = round(blocked_count / max(len(asi_results), 1), 4)
metrics["asi_dynamic_threshold"]  = round(dyn_thresh, 4)
print(f"  ASI Accuracy = {metrics['asi_detection_accuracy']} ({blocked_count}/{len(asi_results)})")

# 14. Latency
lat_total = [r.get("latency", {}).get("total_ms", 0) for r in responses if r.get("latency")]
lat_retr  = [r.get("latency", {}).get("retrieval_ms", 0) for r in responses if r.get("latency")]
metrics["avg_latency_total_ms"]    = round(sum(lat_total) / len(lat_total), 1) if lat_total else None
metrics["avg_latency_retrieval_ms"] = round(sum(lat_retr) / len(lat_retr), 1) if lat_retr else None
print(f"  Avg Total Latency = {metrics['avg_latency_total_ms']} ms")
print(f"  Avg Retrieval Latency = {metrics['avg_latency_retrieval_ms']} ms")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — Retrieval Ablation: BM25 / Dense / Hybrid
# ══════════════════════════════════════════════════════════════════════════════
print("\n[EVAL] Running retrieval ablation (BM25 / Dense / Hybrid)...")

ablation_modes = {
    "bm25_only":  "BM25 only (alpha=0.0)",
    "dense_only": "Dense only (alpha=1.0)",
    "hybrid":     "Hybrid (alpha=0.7)",
}
ablation: dict = {}


def _eval_retrieval_mode(items, retrieval_fn, embedder=None, alpha_label=""):
    """Helper: compute MRR/NDCG/P/R for one retrieval mode."""
    mrr_, nd5_, p5_, r5_ = [], [], [], []
    for item in items:
        q = item["question"]
        domain = item.get("domain", "")
        try:
            if alpha_label == "bm25":
                results = pipe.bm25.search(q, top_k=5)
            elif alpha_label == "dense":
                qemb = pipe.embedder.generate_query_embedding(q, normalize=True)
                results = pipe.faiss_index.search(qemb, top_k=5)
            else:
                results = pipe.hybrid.search(q, top_k=5)
        except Exception:
            results = []

        src_texts = [(c.metadata.get("source_document", "") + " " + (c.domain or "")).lower()
                     for c, _ in results]
        relevance = [1 if _source_matches(t, domain) else 0 for t in src_texts]

        rr = 0.0
        for rank, rel in enumerate(relevance, 1):
            if rel:
                rr = 1.0 / rank
                break
        mrr_.append(rr)
        nd5_.append(_compute_ndcg(relevance, 5))
        p5_.append(sum(relevance[:5]) / 5.0 if len(relevance) >= 5 else sum(relevance) / max(len(relevance), 1))
        r5_.append(1.0 if any(relevance[:5]) else 0.0)

    n = max(len(mrr_), 1)
    return {
        "mrr":           round(sum(mrr_) / n, 4),
        "ndcg_at_5":     round(sum(nd5_) / n, 4),
        "precision_at_5": round(sum(p5_) / n, 4),
        "recall_at_5":   round(sum(r5_) / n, 4),
    }


ablation["bm25_only"]  = _eval_retrieval_mode(normal_items, None, alpha_label="bm25")
print(f"  BM25:   {ablation['bm25_only']}")
ablation["dense_only"] = _eval_retrieval_mode(normal_items, None, alpha_label="dense")
print(f"  Dense:  {ablation['dense_only']}")
ablation["hybrid"]     = _eval_retrieval_mode(normal_items, None, alpha_label="hybrid")
print(f"  Hybrid: {ablation['hybrid']}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PER-QUERY RECORDS
# ══════════════════════════════════════════════════════════════════════════════
per_query_records = []
for idx, (item, resp) in enumerate(zip(normal_items, responses)):
    sources = resp.get("sources", [])
    # Save full contexts for eval_deepeval.py / eval_trulens.py
    contexts = [s.get("content", s.get("preview", "")) for s in sources]

    src_texts = [(s.get("source", "") + " " + s.get("domain", "")).lower() for s in sources]
    relevance  = [1 if _source_matches(t, item.get("domain", "")) else 0 for t in src_texts]

    avg_trust = (
        round(sum(s.get("trust_breakdown", {}).get("trust_score", 0) for s in sources)
              / max(len(sources), 1), 4)
        if sources else None
    )

    conf = resp.get("confidence", 0.0)
    raw_conf = resp.get("raw_confidence", conf)

    per_query_records.append({
        "id":                item["id"],
        "question":          item["question"],
        "ground_truth":      item["ground_truth"],
        "domain":            item.get("domain", ""),
        "subcategory":       item.get("subcategory", ""),
        "difficulty":        item.get("difficulty", ""),
        "query_type":        item.get("query_type", ""),
        "source_document":   item.get("source_document", ""),
        "answer":            resp.get("answer", ""),
        "contexts":          contexts,
        "confidence":        conf,
        "raw_confidence":    raw_conf,
        "faithfulness_score": resp.get("faithfulness_score"),
        "hallucination_risk": resp.get("hallucination_risk", "n/a"),
        "asi_score":         resp.get("asi_score"),
        "avg_trust":         avg_trust,
        "bleu_1":            per_query_bleu1[idx],
        "bleu_4":            per_query_bleu4[idx],
        "rouge_l":           per_query_rouge[idx],
        "bertscore_f1":      per_query_bert[idx],
        "cosine_similarity": per_query_cos[idx],
        "mrr":               round(mrr_scores[idx], 4),
        "ndcg_at_5":         round(ndcg5[idx], 4),
        "precision_at_5":    round(p5[idx], 4),
        "recall_at_5":       round(r5[idx], 4),
        "latency_total_ms":  resp.get("latency", {}).get("total_ms"),
        "latency_retrieval_ms": resp.get("latency", {}).get("retrieval_ms"),
    })

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output = {
    "eval_timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    "num_normal_queries":     len(normal_items),
    "num_adversarial_queries": len(adversarial_items),
    "metrics":                metrics,
    "ablation":               ablation,
    "per_query":              per_query_records,
    "adversarial": {
        "detection_accuracy": metrics["asi_detection_accuracy"],
        "dynamic_threshold":  metrics["asi_dynamic_threshold"],
        "detail":             asi_results,
    },
}

with open("eval_paper_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EVAL_EXTENDED SUMMARY")
print("=" * 65)
for k, v in metrics.items():
    if k.startswith("asi_detail"):
        continue
    label = k.replace("_", " ").title()
    suffix = ""
    if k == "ece":
        target_met = isinstance(v, float) and v < 0.35
        suffix = "  [OK] target met" if target_met else "  [FAIL] target >= 0.35"
    print(f"  {label:<40}: {v}{suffix}")
print("=" * 65)
print("\n  Retrieval Ablation:")
for mode, vals in ablation.items():
    print(f"    {mode:<14}: MRR={vals['mrr']:.4f}  NDCG@5={vals['ndcg_at_5']:.4f}  "
          f"P@5={vals['precision_at_5']:.4f}  R@5={vals['recall_at_5']:.4f}")
print("=" * 65)
print("\n[EVAL] Results saved → eval_paper_results.json")
print("[EVAL] Run eval_deepeval.py and eval_trulens.py next (they reuse these results).")
