"""
Publication-Quality Visualization for SL-RAG / Trust-RAG Evaluation.

Reads all evaluation result JSON files and generates publication figures suitable
for a research paper (IEEE / ACM two-column format), saved as both
PNG (300 DPI) and PDF (vector) in the eval_figures/ directory.

Figures (each saved as its own standalone image; no multi-panel “clubbed” dashboards):
    fig1_aggregate_metrics        — Master metrics summary (all categories)
    fig2_deepeval_radar           — DeepEval metrics radar chart
    fig3_framework_comparison     — DeepEval vs TruLens overlapping metrics
    fig4_retrieval_ablation       — BM25 vs Dense vs Hybrid ablation
    fig5_per_query_heatmap        — Per-query score heatmap (11+ columns)
    fig6_answer_quality           — BLEU / ROUGE / BERTScore / Cosine per query
    fig7a_trust_distribution      — Trust score distribution
    fig7b_hallucination_risk      — Hallucination risk breakdown
    fig7c_promptfoo_pass_fail     — Promptfoo adversarial pass/fail
    fig7d_hallucination_per_query — DeepEval hallucination per query (or faithfulness fallback)
    fig7e_hallucination_distribution — DeepEval hallucination score distribution
    fig8a_trulens_triad_radar     — TruLens RAG triad (aggregate radar)
    fig8b_trulens_groundedness    — Groundedness per query
    fig9a_reliability_diagram     — Confidence calibration (reliability)
    fig9b_confidence_distribution — Confidence distribution
    fig10a_latency_breakdown      — Per-query latency breakdown (retrieval vs generation)
    fig10b_latency_composition    — Average latency composition (donut)
    fig10c_latency_distribution   — Total latency distribution

Usage:
    python eval_visualize.py
"""

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

C_RETRIEVAL  = "#2563EB"
C_GENERATION = "#16A34A"
C_FRAMEWORK  = "#7C3AED"
C_TRUST      = "#DC2626"
C_BM25       = "#F59E0B"
C_DENSE      = "#06B6D4"
C_HYBRID     = "#10B981"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

OUT_DIR = PROJECT_DIR / "eval_figures"
OUT_DIR.mkdir(exist_ok=True)


def _save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved: eval_figures/{name}.png + .pdf")


def _normalise_hallucination_risk(value, default: str = "low") -> str:
    """Normalise hallucination risk values into {low, medium, high}.

    Some eval artifacts contain missing values ("n/a"/"unknown"/empty) or
    unexpected labels. Those are coerced to `default` so the breakdown sums
    to the total number of samples.
    """
    default = (default or "low").strip().lower()
    if default not in {"low", "medium", "high"}:
        default = "low"

    if value is None:
        return default

    # Numeric (or numeric strings) -> bucket via novelty thresholds used in validation.
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 0.30:
            return "high"
        if v > 0.15:
            return "medium"
        return "low"

    s = str(value).strip().lower()
    if s in {"", "n/a", "na", "none", "null", "unknown", "nan"}:
        return default
    if s in {"low", "l"}:
        return "low"
    if s in {"medium", "med", "m"}:
        return "medium"
    if s in {"high", "h"}:
        return "high"

    # Fallback: coerce unexpected labels to the default bucket.
    return default


# ── Load result files ─────────────────────────────────────────────────────────
def _load(path: str | Path, warn: bool = True) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if warn:
        print(f"  [WARN] {path} not found -- some figures will be partial.")
    return {}


paper     = _load(BASE_DIR / "eval_paper_results.json")
deepeval  = _load(BASE_DIR / "eval_deepeval_results.json")
trulens   = _load(BASE_DIR / "eval_trulens_results.json")
promptfoo = _load(BASE_DIR / "eval_promptfoo_results.json")

if not paper:
    print("[ERROR] eval_paper_results.json is required. Run eval_extended.py first.")
    sys.exit(1)

metrics   = paper.get("metrics", {})
ablation  = paper.get("ablation", {})
per_query = paper.get("per_query", [])

# Fix Promptfoo adversarial flags -- parser marks all as non-adversarial;
# re-detect by question text.
ADVERSARIAL_KEYWORDS = ["ignore all", "dan", "reveal", "system prompt",
                        "confidential", "forget", "jailbreak", "bypass", "override"]
if promptfoo.get("tests"):
    for t in promptfoo["tests"]:
        q_lower = t.get("question", "").lower()
        t["is_adversarial"] = any(kw in q_lower for kw in ADVERSARIAL_KEYWORDS)

QUERY_LABELS = [q["id"] for q in per_query]

print(f"\n[VIZ] Loaded data: {len(per_query)} queries | "
      f"DeepEval={'yes' if deepeval else 'no'} | "
      f"TruLens={'yes' if trulens else 'no'} | "
      f"Promptfoo={'yes' if promptfoo else 'no'}")
print(f"[VIZ] Saving figures to {OUT_DIR}/\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 -- Master Metrics Summary
# ══════════════════════════════════════════════════════════════════════════════
def fig1_aggregate_metrics():
    categories = {
        "Retrieval": [
            ("MRR",          metrics.get("mrr")),
            ("NDCG@3",       metrics.get("ndcg_at_3")),
            ("NDCG@5",       metrics.get("ndcg_at_5")),
            ("Precision@3",  metrics.get("precision_at_3")),
            ("Precision@5",  metrics.get("precision_at_5")),
            ("Recall@3",     metrics.get("recall_at_3")),
            ("Recall@5",     metrics.get("recall_at_5")),
        ],
        "Generation": [
            ("BLEU-1",       metrics.get("bleu_1")),
            ("BLEU-4",       metrics.get("bleu_4")),
            ("ROUGE-L",      metrics.get("rouge_l")),
            ("BERTScore F1", metrics.get("bertscore_f1")),
            ("Cosine Sim",   metrics.get("cosine_similarity")),
        ],
        "DeepEval / TruLens": [],
        "Trust & Safety": [
            ("Avg Trust",     metrics.get("avg_trust_score")),
            ("Faithfulness",  metrics.get("avg_faithfulness")),
            ("ASI Detect.",   metrics.get("asi_detection_accuracy")),
            ("Halluc. Rate",  metrics.get("hallucination_rate")),
        ],
    }

    de_agg  = deepeval.get("aggregate", {})
    tru_agg = trulens.get("aggregate", {})
    fw_entries = []
    for k, label in [
        ("hallucination",            "DE Hallucination"),
        ("answer_relevancy",         "DE Ans Relevancy"),
        ("contextual_precision",     "DE Ctx Precision"),
        ("contextual_recall",        "DE Ctx Recall"),
        ("faithfulness_geval",       "DE Faithfulness"),
        ("answer_correctness_geval", "DE Correctness"),
    ]:
        v = de_agg.get(k)
        if v is not None:
            fw_entries.append((label, v))
    for k, label in [
        ("groundedness",      "TruL Groundedness"),
        ("context_relevance", "TruL Ctx Relevance"),
        ("answer_relevance",  "TruL Ans Relevance"),
    ]:
        v = tru_agg.get(k)
        if v is not None:
            fw_entries.append((label, v))
    categories["DeepEval / TruLens"] = fw_entries
    categories = {k: v for k, v in categories.items() if v}

    cat_colours = {
        "Retrieval":          C_RETRIEVAL,
        "Generation":         C_GENERATION,
        "DeepEval / TruLens": C_FRAMEWORK,
        "Trust & Safety":     C_TRUST,
    }

    rows, colours = [], []
    for cat, items in categories.items():
        for label, val in items:
            rows.append((label, val if val is not None else 0.0))
            colours.append(cat_colours.get(cat, "#888888"))

    labels, values = zip(*rows) if rows else ([], [])
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, max(4.5, len(labels) * 0.38)))
    bars = ax.barh(y_pos, values, color=colours, height=0.7,
                   edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score (0-1)")
    ax.set_title("SL-RAG Evaluation -- Aggregate Metrics by Category",
                 fontweight="bold", pad=8)
    ax.invert_yaxis()

    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in cat_colours.items() if k in categories]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig1_aggregate_metrics")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 -- DeepEval Radar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig2_deepeval_radar():
    de_agg = deepeval.get("aggregate", {})
    labels_map = [
        ("faithfulness_geval",       "Faithfulness\n(G-Eval)"),
        ("answer_correctness_geval", "Answer\nCorrectness"),
        ("answer_relevancy",         "Answer\nRelevancy"),
        ("contextual_precision",     "Contextual\nPrecision"),
        ("contextual_recall",        "Contextual\nRecall"),
        ("hallucination",            "Hallucination\n(inverted)"),
    ]

    values, axis_labels = [], []
    for key, label in labels_map:
        v = de_agg.get(key) or 0.0
        if "hallucination" in key:
            v = 1.0 - v
        values.append(v)
        axis_labels.append(label)

    N = len(axis_labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=6.5)
    ax.set_rlabel_position(20)
    ax.plot(angles, values_plot, "o-", color=C_FRAMEWORK, linewidth=2)
    ax.fill(angles, values_plot, color=C_FRAMEWORK, alpha=0.20)
    for angle, value in zip(angles[:-1], values):
        ax.text(angle, value + 0.08, f"{value:.2f}", ha="center", va="center",
                fontsize=7.5, color=C_FRAMEWORK, fontweight="bold")

    ax.set_title("DeepEval Metrics -- Radar Chart\n(higher = better; hallucination inverted)",
                 fontweight="bold", pad=18, size=9)
    fig.tight_layout()
    _save(fig, "fig2_deepeval_radar")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 -- Framework Comparison: DeepEval vs TruLens
# ══════════════════════════════════════════════════════════════════════════════
def fig3_framework_comparison():
    de_agg  = deepeval.get("aggregate", {})
    tru_agg = trulens.get("aggregate", {})

    comparisons = [
        ("Faithfulness /\nGroundedness",
         de_agg.get("faithfulness_geval"), tru_agg.get("groundedness")),
        ("Answer\nRelevance",
         de_agg.get("answer_relevancy"), tru_agg.get("answer_relevance")),
        ("Context Precision /\nRelevance",
         de_agg.get("contextual_precision"), tru_agg.get("context_relevance")),
        ("Contextual\nRecall",
         de_agg.get("contextual_recall"), None),
        ("Answer\nCorrectness",
         de_agg.get("answer_correctness_geval"), None),
    ]
    comparisons = [(l, d, t) for l, d, t in comparisons if d is not None or t is not None]

    if not comparisons:
        print("  [SKIP] fig3 -- no data")
        return

    labels  = [c[0] for c in comparisons]
    de_vals  = [c[1] if c[1] is not None else 0 for c in comparisons]
    tru_vals = [c[2] if c[2] is not None else 0 for c in comparisons]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    b1 = ax.bar(x - width / 2, de_vals,  width, label="DeepEval",
                color=C_FRAMEWORK, alpha=0.87)
    b2 = ax.bar(x + width / 2, tru_vals, width, label="TruLens",
                color=C_TRUST, alpha=0.87)

    for bar, val in zip(list(b1) + list(b2), de_vals + tru_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Framework Comparison: DeepEval vs TruLens",
                 fontweight="bold", pad=8)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save(fig, "fig3_framework_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 -- Retrieval Ablation: BM25 / Dense / Hybrid
# ══════════════════════════════════════════════════════════════════════════════
def fig4_retrieval_ablation():
    if not ablation:
        print("  [SKIP] fig4 -- no ablation data")
        return

    metric_keys  = ["mrr", "ndcg_at_5", "precision_at_5", "recall_at_5"]
    metric_names = ["MRR", "NDCG@5", "Precision@5", "Recall@5"]
    modes        = ["bm25_only", "dense_only", "hybrid"]
    mode_labels  = ["BM25 Only", "Dense Only", "Hybrid"]
    mode_colours = [C_BM25, C_DENSE, C_HYBRID]

    x      = np.arange(len(metric_names))
    width  = 0.25
    offset = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, (mode, label, colour) in enumerate(zip(modes, mode_labels, mode_colours)):
        mode_data = ablation.get(mode, {})
        vals = [mode_data.get(k, 0) or 0 for k in metric_keys]
        bars = ax.bar(x + offset[i], vals, width, label=label, color=colour,
                      alpha=0.87, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Retrieval Ablation: BM25 vs Dense vs Hybrid\n(higher = better)",
                 fontweight="bold", pad=8)
    ax.legend(framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig4_retrieval_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 -- Expanded Per-Query Score Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig5_per_query_heatmap():
    de_per  = {s["id"]: s for s in deepeval.get("per_sample", [])}
    tru_per = {s["id"]: s for s in trulens.get("per_sample", [])}

    col_defs = [
        ("bleu_4",            "BLEU-4"),
        ("rouge_l",           "ROUGE-L"),
        ("bertscore_f1",      "BERTScore"),
        ("cosine_similarity", "Cosine Sim"),
        ("mrr",               "MRR"),
        ("ndcg_at_5",         "NDCG@5"),
        ("precision_at_5",    "P@5"),
        ("recall_at_5",       "R@5"),
        ("avg_trust",         "Trust"),
        ("faithfulness_score","Faithfulness"),
        ("confidence",        "Confidence"),
    ]
    col_labels = [c[1] for c in col_defs]

    if de_per:
        col_labels += ["DE Halluc.", "DE Ans.Rel.", "DE Ctx.Prec."]
    if tru_per:
        col_labels += ["TruL Grnd.", "TruL Ctx.Rel.", "TruL Ans.Rel."]

    data = []
    for q in per_query:
        row = [float(q.get(c[0]) or 0.0) for c in col_defs]
        if de_per:
            de = de_per.get(q["id"], {})
            row.append(float(de.get("hallucination") if de.get("hallucination") is not None else 0.0))
            row.append(float(de.get("answer_relevancy") if de.get("answer_relevancy") is not None else 0.0))
            row.append(float(de.get("contextual_precision") if de.get("contextual_precision") is not None else 0.0))
        if tru_per:
            tru = tru_per.get(q["id"], {})
            row.append(float(tru.get("groundedness") or 0.0))
            row.append(float(tru.get("context_relevance") or 0.0))
            row.append(float(tru.get("answer_relevance") or 0.0))
        data.append(row)

    mat = np.array(data, dtype=float)

    fig_h = max(5.0, len(QUERY_LABELS) * 0.48)
    fig_w = max(6.0, len(col_labels) * 0.80)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Score (0-1)")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(QUERY_LABELS)))
    ax.set_yticklabels(QUERY_LABELS, fontsize=7.5)

    if len(QUERY_LABELS) <= 30:
        for i in range(len(QUERY_LABELS)):
            for j in range(len(col_labels)):
                val = mat[i, j]
                tc = "white" if val < 0.30 or val > 0.82 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.0, color=tc, fontweight="bold")

    ax.set_title("Per-Query Score Heatmap -- All Metrics\n(green = high, red = low)",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "fig5_per_query_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 -- Answer Quality Per Query
# ══════════════════════════════════════════════════════════════════════════════
def fig6_answer_quality():
    metric_cols = [
        ("bleu_4",            "BLEU-4",       C_BM25),
        ("rouge_l",           "ROUGE-L",      C_HYBRID),
        ("bertscore_f1",      "BERTScore F1", C_RETRIEVAL),
        ("cosine_similarity", "Cosine Sim",   C_FRAMEWORK),
    ]

    x = np.arange(len(per_query))
    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    for key, label, colour in metric_cols:
        vals = [float(q.get(key) or 0.0) for q in per_query]
        ax.plot(x, vals, "o-", label=label, color=colour,
                linewidth=1.8, markersize=4.5, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Answer Quality Metrics per Query", fontweight="bold", pad=8)
    ax.legend(framealpha=0.9, ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig6_answer_quality")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 -- Trust & Safety Dashboard (2x2)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_trust_safety():
    saved = []

    # (a) Trust Score distribution
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    trust_vals = [float(q.get("avg_trust") or 0.0) for q in per_query]
    if trust_vals:
        ax.hist(trust_vals, bins=8, color=C_TRUST, edgecolor="white", alpha=0.85)
        mean_t = np.mean(trust_vals)
        ax.axvline(mean_t, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_t:.3f}")
        ax.legend(fontsize=7.5)
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Trust Score Distribution", fontweight="bold", pad=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, "fig7a_trust_distribution")
    saved.append("fig7a_trust_distribution")

    # (b) Hallucination risk pie
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    hall_counts = {"low": 0, "medium": 0, "high": 0}
    for q in per_query:
        risk = _normalise_hallucination_risk(q.get("hallucination_risk"), default="low")
        hall_counts[risk] += 1

    sizes   = [hall_counts["low"], hall_counts["medium"], hall_counts["high"]]
    slabels = [f"Low ({hall_counts['low']})", f"Medium ({hall_counts['medium']})",
               f"High ({hall_counts['high']})"]
    cpie    = ["#16A34A", "#F59E0B", "#DC2626"]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, slabels, cpie) if s > 0]
    if non_zero:
        sz, sl, cc = zip(*non_zero)
        wedges, _, ats = ax.pie(sz, labels=sl, colors=cc, autopct="%1.0f%%",
                                startangle=90,
                                textprops={"fontsize": 8},
                                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        for at in ats:
            at.set_fontsize(7.5)
    else:
        ax.text(0.5, 0.5, "No hallucination risk data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.axis("off")

    ax.set_title("Hallucination Risk Breakdown", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig7b_hallucination_risk")
    saved.append("fig7b_hallucination_risk")

    # (c) Promptfoo adversarial pass/fail
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    pf_tests = promptfoo.get("tests", [])
    if pf_tests:
        adv_tests  = [t for t in pf_tests if t.get("is_adversarial")]
        norm_tests = [t for t in pf_tests if not t.get("is_adversarial")]
        cats_pf, pass_v, fail_v = [], [], []
        for lbl, tgroup in [("Adversarial", adv_tests), ("Normal", norm_tests)]:
            if tgroup:
                p = sum(1 for t in tgroup if t.get("passed"))
                f = len(tgroup) - p
                cats_pf.append(f"{lbl}\n(n={len(tgroup)})")
                pass_v.append(p)
                fail_v.append(f)
        if cats_pf:
            x_pf = np.arange(len(cats_pf))
            ax.bar(x_pf, pass_v, label="Pass", color="#16A34A", alpha=0.87, edgecolor="white")
            ax.bar(x_pf, fail_v, bottom=pass_v, label="Fail",
                   color="#DC2626", alpha=0.87, edgecolor="white")
            for xi, (p, f) in enumerate(zip(pass_v, fail_v)):
                if p > 0:
                    ax.text(xi, p / 2, str(p), ha="center", va="center",
                            fontsize=11, color="white", fontweight="bold")
                if f > 0:
                    ax.text(xi, p + f / 2, str(f), ha="center", va="center",
                            fontsize=11, color="white", fontweight="bold")
            ax.set_xticks(x_pf)
            ax.set_xticklabels(cats_pf, fontsize=9)
            ax.set_ylabel("Number of Tests")
            ax.set_ylim(0, max(pass_v) + 3)
            ax.legend(fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "Promptfoo results\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.axis("off")
    ax.set_title("Promptfoo Test Pass/Fail", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig7c_promptfoo_pass_fail")
    saved.append("fig7c_promptfoo_pass_fail")

    # (d) DeepEval hallucination per query
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    de_per = {s["id"]: s for s in deepeval.get("per_sample", [])}
    if de_per:
        hall_scores, q_short = [], []
        for q in per_query:
            de = de_per.get(q["id"], {})
            hs = de.get("hallucination")
            hall_scores.append(float(hs) if hs is not None else 0.0)
            q_short.append(q["id"].replace("Procurement-", "P-")
                                   .replace("Technical-", "T-")
                                   .replace("GFR-", "G-")
                                   .replace("Mixed-", "M-"))
        x_de = np.arange(len(hall_scores))
        bar_cols = [C_TRUST if s > 0.5 else C_HYBRID for s in hall_scores]
        ax.bar(x_de, hall_scores, color=bar_cols, alpha=0.87, edgecolor="white")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Threshold 0.5")
        ax.set_xticks(x_de)
        ax.set_xticklabels(q_short, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Hallucination Score")
        ax.legend(fontsize=7.5)
    else:
        faith_vals = [float(q.get("faithfulness_score") or 0.0) for q in per_query]
        x_f = np.arange(len(faith_vals))
        ax.bar(x_f, faith_vals, color=C_FRAMEWORK, alpha=0.87, edgecolor="white")
        ax.set_xticks(x_f)
        ax.set_xticklabels([q["id"].split("-")[-1][:6] for q in per_query],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Faithfulness")
    ax.set_title("Hallucination per Query\n(lower = better)",
                 fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig7d_hallucination_per_query")
    saved.append("fig7d_hallucination_per_query")

    # (e) Hallucination distribution (safe vs flagged)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    de_scores = []
    if deepeval.get("per_sample"):
        for s in deepeval.get("per_sample", []):
            hs = s.get("hallucination")
            if hs is not None:
                de_scores.append(float(hs))

    if de_scores:
        bins = np.arange(0, 1.1, 0.1)
        safe = [s for s in de_scores if s < 0.5]
        flagged = [s for s in de_scores if s >= 0.5]
        ax.hist(safe, bins=bins, color="#1D9E75", label=f"Safe ({len(safe)})", alpha=0.9)
        ax.hist(flagged, bins=bins, color="#E24B4A", label=f"Flagged ({len(flagged)})", alpha=0.9)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold 0.5")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Hallucination score")
        ax.set_ylabel("Queries")
        ax.legend(framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "DeepEval hallucination scores\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.axis("off")

    ax.set_title("Hallucination Score Distribution", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig7e_hallucination_distribution")
    saved.append("fig7e_hallucination_distribution")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 -- TruLens RAG Triad
# ══════════════════════════════════════════════════════════════════════════════
def fig8_trulens_triad():
    tru_agg = trulens.get("aggregate", {})
    tru_per = trulens.get("per_sample", [])

    radar_labels = ["Groundedness", "Context\nRelevance", "Answer\nRelevance"]
    radar_values = [
        tru_agg.get("groundedness", 0) or 0,
        tru_agg.get("context_relevance", 0) or 0,
        tru_agg.get("answer_relevance", 0) or 0,
    ]

    saved = []

    # (a) Radar chart (aggregate)
    fig, ax_r = plt.subplots(figsize=(4.8, 4.6), subplot_kw={"polar": True})
    N      = 3
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    vp     = radar_values + radar_values[:1]

    ax_r.set_theta_offset(math.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_labels, size=8.5)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=6.5)
    ax_r.plot(angles, vp, "o-", color=C_TRUST, linewidth=2.2)
    ax_r.fill(angles, vp, color=C_TRUST, alpha=0.18)
    for angle, val in zip(angles[:-1], radar_values):
        ax_r.text(angle, val + 0.1, f"{val:.2f}", ha="center", va="center",
                  fontsize=8, color=C_TRUST, fontweight="bold")
    ax_r.set_title("TruLens RAG Triad (Aggregate)", fontweight="bold", pad=16, size=9)
    fig.tight_layout()
    _save(fig, "fig8a_trulens_triad_radar")
    saved.append("fig8a_trulens_triad_radar")

    # (b) Groundedness per query (bars)
    fig, ax_b = plt.subplots(figsize=(6.8, 4.6))
    if tru_per:
        gnd_all = []
        for q in per_query:
            match = next((s for s in tru_per if s.get("id") == q["id"]), {})
            gnd_all.append((q, float(match.get("groundedness") or 0.0)))

        if len(gnd_all) > 50:
            from collections import defaultdict
            by_cat = defaultdict(list)
            for q, gnd in gnd_all:
                cat = q.get("domain", "Other")
                by_cat[cat].append((q, gnd))

            sampled = []
            for cat in sorted(by_cat):
                items = sorted(by_cat[cat], key=lambda x: x[1])
                n_pick = max(1, min(3, len(items)))
                step = max(1, (len(items) - 1) // (n_pick - 1)) if n_pick > 1 else 1
                picked = [items[min(i * step, len(items) - 1)] for i in range(n_pick)]
                for p in picked:
                    if p not in sampled:
                        sampled.append(p)

            gnd_vals = [g for _, g in sampled]
            q_short = [
                q["id"].replace("Procurement-", "P-")
                       .replace("Technical-", "T-")
                       .replace("GFR-", "G-")
                       .replace("Mixed-", "M-")
                for q, _ in sampled
            ]
            x_b   = np.arange(len(gnd_vals))
            bcols = [C_HYBRID if v >= 0.5 else C_TRUST for v in gnd_vals]
            ax_b.barh(x_b, gnd_vals, color=bcols, alpha=0.87, edgecolor="white")
            ax_b.set_yticks(x_b)
            ax_b.set_yticklabels(q_short, fontsize=7.5)
            ax_b.set_xlim(0, 1.1)
            ax_b.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="0.5")
            ax_b.set_xlabel("Groundedness Score")
            all_scores = [g for _, g in gnd_all]
            agg = np.mean(all_scores)
            ax_b.set_title(
                f"Groundedness per Query\n"
                f"(showing {len(sampled)}/{len(gnd_all)}, "
                f"agg={agg:.3f})",
                fontweight="bold",
            )
            ax_b.invert_yaxis()
            ax_b.legend(fontsize=7.5)
        else:
            gnd_vals = [g for _, g in gnd_all]
            q_short = [
                q["id"].replace("Procurement-", "P-")
                       .replace("Technical-", "T-")
                       .replace("GFR-", "G-")
                       .replace("Mixed-", "M-")
                for q, _ in gnd_all
            ]
            x_b   = np.arange(len(gnd_vals))
            bcols = [C_HYBRID if v >= 0.5 else C_TRUST for v in gnd_vals]
            ax_b.barh(x_b, gnd_vals, color=bcols, alpha=0.87, edgecolor="white")
            ax_b.set_yticks(x_b)
            ax_b.set_yticklabels(q_short, fontsize=7.5)
            ax_b.set_xlim(0, 1.1)
            ax_b.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="0.5")
            ax_b.set_xlabel("Groundedness Score")
            ax_b.set_title("Groundedness\nper Query", fontweight="bold")
            ax_b.invert_yaxis()
            ax_b.legend(fontsize=7.5)
    else:
        ax_b.text(0.5, 0.5, "TruLens per-sample results\nnot available",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=9, color="gray")
        ax_b.axis("off")

    ax_b.set_title("Groundedness per Query", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig8b_trulens_groundedness")
    saved.append("fig8b_trulens_groundedness")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 -- Confidence Calibration (Reliability Diagram)
# ══════════════════════════════════════════════════════════════════════════════
def fig9_confidence_calibration():
    confs = [float(q.get("confidence") or 0.5) for q in per_query]
    rouge = [float(q.get("rouge_l") or 0.0)    for q in per_query]
    ece   = metrics.get("ece")

    N_BINS   = 5
    bin_size = 1.0 / N_BINS
    bin_confs, bin_accs = [], []

    for b in range(N_BINS):
        lo, hi = b * bin_size, (b + 1) * bin_size
        idx = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            continue
        bin_confs.append(sum(confs[i] for i in idx) / len(idx))
        bin_accs.append(sum(rouge[i] for i in idx) / len(idx))

    saved = []

    # (a) Reliability diagram
    fig, ax_m = plt.subplots(figsize=(6.6, 4.0))

    ax_m.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", alpha=0.6)
    if bin_confs:
        ax_m.bar(bin_confs, bin_accs, width=bin_size * 0.8, alpha=0.65,
                 color=C_RETRIEVAL, edgecolor="white", linewidth=0.8, label="Bin accuracy")
        ax_m.plot(bin_confs, bin_accs, "D-", color=C_RETRIEVAL, markersize=6, linewidth=1.5)

    ax_m.scatter(confs, rouge, s=28, color=C_GENERATION, alpha=0.7, zorder=5,
                 label="Per-query (conf, ROUGE-L)")

    if ece is not None:
        ax_m.text(0.97, 0.04, f"ECE = {ece:.4f}", transform=ax_m.transAxes,
                  ha="right", va="bottom", fontsize=8.5, color=C_TRUST,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=C_TRUST, alpha=0.9))

    ax_m.set_xlim(-0.02, 1.02)
    ax_m.set_ylim(-0.02, 1.02)
    ax_m.set_xlabel("Predicted Confidence")
    ax_m.set_ylabel("Observed Accuracy (ROUGE-L)")
    ax_m.set_title("Confidence Calibration\n(Reliability Diagram)", fontweight="bold", pad=8)
    ax_m.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig9a_reliability_diagram")
    saved.append("fig9a_reliability_diagram")

    # (b) Confidence distribution
    fig, ax_h = plt.subplots(figsize=(5.4, 3.8))
    ax_h.hist(confs, bins=6, color=C_RETRIEVAL,
              alpha=0.75, edgecolor="white")
    ax_h.set_xlabel("Predicted Confidence")
    ax_h.set_ylabel("Count")
    ax_h.set_xlim(-0.02, 1.02)
    ax_h.set_title("Confidence Distribution", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig9b_confidence_distribution")
    saved.append("fig9b_confidence_distribution")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 -- Latency Analysis (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_latency():
    total_ms     = [float(q.get("latency_total_ms") or 0.0)    for q in per_query]
    retrieval_ms = [float(q.get("latency_retrieval_ms") or 0.0) for q in per_query]
    gen_ms       = [max(0.0, t - r) for t, r in zip(total_ms, retrieval_ms)]

    avg_total    = np.mean(total_ms)
    avg_retrieval = np.mean(retrieval_ms)
    avg_gen      = np.mean(gen_ms)

    saved = []

    # (a) Stacked bars per query
    fig, ax_bar = plt.subplots(figsize=(8.8, 4.2))

    # Left: stacked bar -- retrieval + generation per query (in seconds)
    x     = np.arange(len(per_query))
    ret_s = [r / 1000.0 for r in retrieval_ms]
    gen_s = [g / 1000.0 for g in gen_ms]

    ax_bar.bar(x, ret_s, label="Retrieval", color=C_DENSE,    alpha=0.88, edgecolor="white")
    ax_bar.bar(x, gen_s, bottom=ret_s, label="LLM Generation",
               color=C_FRAMEWORK, alpha=0.88, edgecolor="white")

    ax_bar.axhline(avg_total / 1000.0, color="black", linestyle="--",
                   linewidth=1.1, alpha=0.7,
                   label=f"Avg total = {avg_total/1000:.1f}s")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7.0)
    ax_bar.set_ylabel("Latency (seconds)")
    ax_bar.set_title("Per-Query Latency Breakdown\n(Retrieval vs LLM Generation)",
                     fontweight="bold", pad=8)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.set_title(f"Per-Query Latency Breakdown ({len(per_query)} queries)",
                     fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig10a_latency_breakdown")
    saved.append("fig10a_latency_breakdown")

    # (b) Donut chart -- avg latency composition
    fig, ax_pie = plt.subplots(figsize=(5.2, 4.2))
    donut_sizes  = [avg_retrieval, avg_gen]
    denom = avg_total if avg_total and avg_total > 0 else 0.0
    pct_r = (avg_retrieval / denom * 100.0) if denom else 0.0
    pct_g = (avg_gen / denom * 100.0) if denom else 0.0
    donut_labels = [
        f"Retrieval\n{avg_retrieval:.0f} ms\n({pct_r:.1f}%)",
        f"LLM Gen\n{avg_gen/1000:.1f}s\n({pct_g:.1f}%)",
    ]
    donut_colours = [C_DENSE, C_FRAMEWORK]

    wedges, _ = ax_pie.pie(
        donut_sizes,
        colors=donut_colours,
        startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 1.5},
    )
    ax_pie.legend(wedges, donut_labels, loc="lower center", fontsize=7.0,
                  bbox_to_anchor=(0.5, -0.22), framealpha=0.9)
    ax_pie.set_title("Average Latency Composition", fontweight="bold", pad=8)
    ax_pie.text(0, 0, f"Avg\n{avg_total/1000:.1f}s",
                ha="center", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig10b_latency_composition")
    saved.append("fig10b_latency_composition")

    # (c) Total latency distribution (seconds)
    fig, ax_h = plt.subplots(figsize=(8.0, 4.0))
    latencies_s = [t / 1000.0 for t in total_ms if t is not None]
    if latencies_s:
        ax_h.hist(latencies_s, bins=20, color="#7F77DD", alpha=0.9, edgecolor="white")
        mean_s = float(np.mean(latencies_s))
        ax_h.axvline(mean_s, color="#E24B4A", linestyle="--", linewidth=1.2,
                     label=f"Mean {mean_s:.1f}s")
        ax_h.set_xlabel("Total latency (s)")
        ax_h.set_ylabel("Queries")
        ax_h.legend(framealpha=0.9)
    else:
        ax_h.text(0.5, 0.5, "No latency data",
                  ha="center", va="center", transform=ax_h.transAxes,
                  fontsize=9, color="gray")
        ax_h.axis("off")

    ax_h.set_title("Total Latency Distribution", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "fig10c_latency_distribution")
    saved.append("fig10c_latency_distribution")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL FIGURES
# ══════════════════════════════════════════════════════════════════════════════
figure_fns = [
    ("fig1_aggregate_metrics",      fig1_aggregate_metrics),
    ("fig2_deepeval_radar",         fig2_deepeval_radar),
    ("fig3_framework_comparison",   fig3_framework_comparison),
    ("fig4_retrieval_ablation",     fig4_retrieval_ablation),
    ("fig5_per_query_heatmap",      fig5_per_query_heatmap),
    ("fig6_answer_quality",         fig6_answer_quality),
    ("fig7_trust_safety",           fig7_trust_safety),
    ("fig8_trulens_triad",          fig8_trulens_triad),
    ("fig9_confidence_calibration", fig9_confidence_calibration),
    ("fig10_latency",               fig10_latency),
]

expected_outputs = [
    "fig1_aggregate_metrics",
    "fig2_deepeval_radar",
    "fig3_framework_comparison",
    "fig4_retrieval_ablation",
    "fig5_per_query_heatmap",
    "fig6_answer_quality",
    "fig7a_trust_distribution",
    "fig7b_hallucination_risk",
    "fig7c_promptfoo_pass_fail",
    "fig7d_hallucination_per_query",
    "fig7e_hallucination_distribution",
    "fig8a_trulens_triad_radar",
    "fig8b_trulens_groundedness",
    "fig9a_reliability_diagram",
    "fig9b_confidence_distribution",
    "fig10a_latency_breakdown",
    "fig10b_latency_composition",
    "fig10c_latency_distribution",
]

for name, fn in figure_fns:
    print(f"  Generating {name}...")
    try:
        out = fn()
        if isinstance(out, list):
            expected_outputs.extend([x for x in out if x not in expected_outputs])
    except Exception as e:
        import traceback
        print(f"  [ERROR] {name} failed: {e}")
        traceback.print_exc()

print(f"\n[VIZ] All figures saved to {OUT_DIR}/")
print("[VIZ] Each figure exists as .png (300 DPI) and .pdf (vector for LaTeX).\n")
print("Figure summary:")
for name in expected_outputs:
    png_ok = (OUT_DIR / f"{name}.png").exists()
    pdf_ok = (OUT_DIR / f"{name}.pdf").exists()
    status = "OK" if png_ok and pdf_ok else ("PNG only" if png_ok else "MISSING")
    print(f"  {name:<44} {status}")
