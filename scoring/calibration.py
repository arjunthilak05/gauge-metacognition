"""Calibration metrics for metacognitive monitoring evaluation.

Implements ECE, Brier, AUROC, and AUGRC following best practices from:
- Equal-mass binning for ECE (ICLR 2025 recommendations)
- Brier Skill Score vs naive baseline
- AUROC with single-class guard
- AUGRC (NeurIPS 2024) for abstention quality
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def ece(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 15,
    strategy: str = "equal_mass",
) -> float:
    """Expected Calibration Error with equal-mass (adaptive) binning.

    Args:
        confidences: Model confidence values in [0, 1].
        correctness: Binary correctness labels (0 or 1).
        n_bins: Number of bins.
        strategy: "equal_mass" (recommended) or "equal_width".

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    n = len(conf)
    if n == 0:
        return float("nan")

    if strategy == "equal_mass":
        sorted_idx = np.argsort(conf)
        bin_size = max(1, n // n_bins)
        bins = [sorted_idx[i * bin_size: (i + 1) * bin_size] for i in range(n_bins)]
        # Last bin absorbs remainder
        if len(bins) > 1:
            remainder = sorted_idx[n_bins * bin_size:]
            if len(remainder) > 0:
                bins[-1] = np.concatenate([bins[-1], remainder])
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi == 1.0:
                mask = (conf >= lo) & (conf <= hi)
            else:
                mask = (conf >= lo) & (conf < hi)
            bins.append(np.where(mask)[0])

    weighted_error = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        avg_conf = conf[bin_idx].mean()
        avg_acc = corr[bin_idx].mean()
        weighted_error += len(bin_idx) / n * abs(avg_conf - avg_acc)

    return float(weighted_error)


def brier_score(confidences: list[float], correctness: list[int]) -> float:
    """Brier score: mean squared error between confidence and correctness.

    Args:
        confidences: Model confidence in [0, 1].
        correctness: Binary correctness (0 or 1).

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - corr) ** 2))


def brier_skill_score(confidences: list[float], correctness: list[int]) -> float:
    """Brier Skill Score relative to a naive baseline (always predict base rate).

    Returns:
        BSS in (-inf, 1]. 1 = perfect, 0 = no better than baseline, <0 = worse.
    """
    bs = brier_score(confidences, correctness)
    base_rate = np.mean(correctness)
    bs_baseline = float(np.mean((base_rate - np.array(correctness, dtype=float)) ** 2))
    if bs_baseline == 0:
        return float("nan")
    return float(1.0 - bs / bs_baseline)


def auroc(confidences: list[float], correctness: list[int]) -> float:
    """AUROC: confidence as a classifier for correctness.

    Returns NaN if only one class is present (AUROC is undefined).
    """
    corr = np.array(correctness)
    if len(np.unique(corr)) < 2:
        return float("nan")

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(corr, confidences))


def augrc(confidences: list[float], correctness: list[int]) -> float:
    """Area Under the Generalized Risk-Coverage Curve (NeurIPS 2024).

    Measures abstention quality independent of base classifier accuracy.
    Lower is better.
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    n = len(conf)
    if n == 0:
        return float("nan")

    # Sort by confidence descending (most confident first)
    order = np.argsort(-conf)
    sorted_corr = corr[order]

    # Cumulative risk at each coverage level
    cum_errors = np.cumsum(1 - sorted_corr)
    coverages = np.arange(1, n + 1) / n
    risks = cum_errors / np.arange(1, n + 1)

    # Generalized risk: risk * coverage (normalizes out base accuracy)
    gen_risks = risks * coverages

    # Integrate using trapezoidal rule
    area = float(np.trapz(gen_risks, coverages))
    return area


def eol_difficulty_correlation(
    predicted_difficulties: list[int],
    actual_difficulties: list[int],
) -> float:
    """Spearman correlation between predicted and actual difficulty.

    Measures Ease-of-Learning judgment accuracy.

    Returns:
        Spearman rho in [-1, 1]. Higher is better.
    """
    if len(predicted_difficulties) < 3:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(predicted_difficulties, actual_difficulties)
    return float(rho)


def abstention_optimality(
    decisions: list[str],
    correctness: list[int],
) -> dict[str, float]:
    """Evaluate quality of submit/abstain decisions.

    Args:
        decisions: List of "submit" or "abstain" strings.
        correctness: Binary correctness of the answer (0 or 1).

    Returns:
        Dict with optimality metrics.
    """
    decisions = [d.lower().strip() for d in decisions]
    corr = np.array(correctness, dtype=int)

    n_abstain = sum(1 for d in decisions if d == "abstain")
    n_submit = sum(1 for d in decisions if d == "submit")

    # Abstention on wrong answers = good metacognitive control
    abstain_on_wrong = sum(
        1 for d, c in zip(decisions, corr) if d == "abstain" and c == 0
    )
    # Abstention on correct answers = overly cautious
    abstain_on_correct = sum(
        1 for d, c in zip(decisions, corr) if d == "abstain" and c == 1
    )

    abstain_precision = (
        abstain_on_wrong / n_abstain if n_abstain > 0 else float("nan")
    )

    # Game score: +3 correct submit, -1 wrong submit, +1 abstain wrong, 0 abstain correct
    total_score = 0
    for d, c in zip(decisions, corr):
        if d == "submit" and c == 1:
            total_score += 3
        elif d == "submit" and c == 0:
            total_score -= 1
        elif d == "abstain" and c == 0:
            total_score += 1
        # abstain + correct = 0

    # Optimal score: submit all correct (+3 each), abstain all wrong (+1 each)
    n_correct = int(corr.sum())
    n_wrong = len(corr) - n_correct
    optimal_score = n_correct * 3 + n_wrong * 1

    # Worst score: submit all wrong (-1 each), abstain all correct (0 each)
    worst_score = n_wrong * -1

    score_efficiency = (
        (total_score - worst_score) / (optimal_score - worst_score)
        if optimal_score != worst_score
        else float("nan")
    )

    return {
        "total_score": total_score,
        "optimal_score": optimal_score,
        "score_efficiency": score_efficiency,
        "abstain_count": n_abstain,
        "submit_count": n_submit,
        "abstain_precision": abstain_precision,
        "abstain_on_wrong": abstain_on_wrong,
        "abstain_on_correct": abstain_on_correct,
    }
