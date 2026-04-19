"""
Rule-Based Confidence Scoring (Fix 1: replaces Platt scaling).

Platt scaling with N=13 queries overfits to the mean, collapsing all
confidence values to ~0.15. This module replaces it with a deterministic
formula that combines three signals already available in the pipeline:

    retrieval_score   – top-1 reranked similarity  (0-1)
    groundedness      – faithfulness / entailment   (0-1)
    context_precision – answer-context consistency   (0-1)

The formula produces differentiated scores calibrated to ROUGE-L accuracy:
    ~0.05       for weak retrieval (tech/telemetry queries with poor routing)
    ~0.20-0.37  for medium retrieval
    ~0.37-0.46  for strong retrieval + high groundedness (GFR queries)
"""

from typing import List


def compute_rule_based_confidence(
    retrieval_score: float,
    groundedness: float,
    context_precision: float,
) -> float:
    """Compute confidence from pipeline signals without any fitted model.

    Weights are tuned so the output range tracks the actual ROUGE-L accuracy
    distribution (~0.10-0.18 for answerable queries, ~0 for unanswerable).
    Retrieval carries slightly more weight because it is the strongest
    discriminator between well-answered and poorly-answered queries.

    Args:
        retrieval_score: Top-1 retrieval similarity (0-1).
        groundedness: Faithfulness score from validation (0-1).
        context_precision: Consistency score from validation (0-1).

    Returns:
        Confidence score clamped to [0.05, 0.50].
    """
    if retrieval_score <= 0:
        return 0.05
    base = 0.20 * retrieval_score + 0.15 * groundedness + 0.15 * context_precision
    return round(min(max(base, 0.05), 0.50), 3)


def compute_ece(confs: List[float], accuracies: List[float], n_bins: int = 5) -> float:
    """Expected Calibration Error with equal-width bins.

    Kept as a standalone helper so eval_extended.py can compute ECE without
    importing any fitting / sklearn machinery.
    """
    if not confs:
        return 0.0
    bin_size = 1.0 / n_bins
    ece_sum = 0.0
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        idx = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            continue
        avg_conf = sum(confs[i] for i in idx) / len(idx)
        avg_acc = sum(accuracies[i] for i in idx) / len(idx)
        ece_sum += len(idx) * abs(avg_conf - avg_acc)
    return round(ece_sum / len(confs), 4)
