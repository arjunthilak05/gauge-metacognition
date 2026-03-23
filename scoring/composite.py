"""Composite scoring: M-Score, C-Score, L-Score for the MetaCog benchmark.

Aggregates normalized metrics from all three paradigms (staircase, strategy_switch,
offloading) into three interpretable scores plus an overall composite.

Scores:
  M-Score (monitoring):  Can the model assess its own accuracy/difficulty?
  C-Score (control):     Does the model act on its assessments?
  L-Score (coupling):    Does good monitoring actually drive good control?
  Overall:               0.4*M + 0.4*C + 0.2*L

All scores are in [0, 1], higher = better.  NaN components are skipped;
the score is the average of whatever is available.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """Average of non-NaN values.  Returns NaN if nothing is available."""
    clean = [v for v in values if not pd.isna(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _clamp01(x: float) -> float:
    """Clamp a value to [0, 1], preserving NaN."""
    if pd.isna(x):
        return float("nan")
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Per-paradigm extraction helpers
# ---------------------------------------------------------------------------

def _staircase_m(m: dict) -> list[float]:
    """Extract normalized monitoring components from staircase metrics."""
    components: list[float] = []

    # (1 - ECE): lower ECE is better, so invert
    if "ece" in m and not pd.isna(m["ece"]):
        components.append(1.0 - m["ece"])

    # (1 - Brier): lower Brier is better, so invert
    if "brier" in m and not pd.isna(m["brier"]):
        components.append(1.0 - m["brier"])

    # AUROC: already [0, 1], higher = better
    if "auroc" in m and not pd.isna(m["auroc"]):
        components.append(m["auroc"])

    # EOL correlation: Spearman rho in [-1, 1] → normalize to [0, 1]
    if "eol_correlation" in m and not pd.isna(m["eol_correlation"]):
        components.append((m["eol_correlation"] + 1.0) / 2.0)

    return components


def _staircase_c(m: dict) -> list[float]:
    """Extract normalized control components from staircase metrics."""
    components: list[float] = []

    if "score_efficiency" in m and not pd.isna(m["score_efficiency"]):
        components.append(_clamp01(m["score_efficiency"]))

    return components


def _staircase_l(m: dict) -> list[float]:
    """Extract coupling components from staircase metrics."""
    components: list[float] = []

    # l_score is point-biserial r(confidence, submit), already clamped to [0, 1]
    if "l_score" in m and not pd.isna(m["l_score"]):
        components.append(_clamp01(m["l_score"]))

    return components


def _strategy_m(m: dict) -> list[float]:
    """Extract normalized monitoring components from strategy_switch metrics."""
    components: list[float] = []

    # Confidence-keep alignment: point-biserial r, roughly [-1, 1]
    # Normalize to [0, 1] via (x+1)/2
    if "confidence_alignment" in m and not pd.isna(m["confidence_alignment"]):
        components.append(_clamp01((m["confidence_alignment"] + 1.0) / 2.0))

    return components


def _strategy_c(m: dict) -> list[float]:
    """Extract normalized control components from strategy_switch metrics."""
    components: list[float] = []

    if "score_efficiency" in m and not pd.isna(m["score_efficiency"]):
        components.append(_clamp01(m["score_efficiency"]))

    # (1 - sycophancy_rate): lower sycophancy is better
    if "sycophancy_rate" in m and not pd.isna(m["sycophancy_rate"]):
        components.append(_clamp01(1.0 - m["sycophancy_rate"]))

    if "correction_rate" in m and not pd.isna(m["correction_rate"]):
        components.append(_clamp01(m["correction_rate"]))

    return components


def _strategy_l(m: dict) -> list[float]:
    """Extract coupling components from strategy_switch metrics."""
    components: list[float] = []

    # Phi coefficient: correlation between correctness and keep decision
    # Range roughly [-1, 1] → normalize to [0, 1], clamp
    if "phi_coefficient" in m and not pd.isna(m["phi_coefficient"]):
        components.append(_clamp01((m["phi_coefficient"] + 1.0) / 2.0))

    return components


def _offloading_m(m: dict) -> list[float]:
    """Extract normalized monitoring components from offloading metrics."""
    components: list[float] = []

    # Memory calibration: (1 - memory_ECE)
    if "memory_ece" in m and not pd.isna(m["memory_ece"]):
        components.append(1.0 - m["memory_ece"])

    # Rarity→lookup AUROC: higher rarity questions get lookups = good monitoring
    if "rarity_auroc" in m and not pd.isna(m["rarity_auroc"]):
        components.append(m["rarity_auroc"])

    return components


def _offloading_c(m: dict) -> list[float]:
    """Extract normalized control components from offloading metrics."""
    components: list[float] = []

    # Memory accuracy as proxy for allocation quality: if you only answered
    # from memory what you actually know, memory_accuracy will be high.
    if "memory_accuracy" in m and not pd.isna(m["memory_accuracy"]):
        components.append(_clamp01(m["memory_accuracy"]))

    return components


def _offloading_l(m: dict) -> list[float]:
    """Extract coupling components from offloading metrics."""
    components: list[float] = []

    # Rarity→lookup AUROC: higher rarity → more lookups = good coupling
    if "rarity_auroc" in m and not pd.isna(m["rarity_auroc"]):
        components.append(_clamp01(m["rarity_auroc"]))

    return components


# ---------------------------------------------------------------------------
# Main composite function
# ---------------------------------------------------------------------------

def compute_composite(
    staircase_metrics: Optional[dict] = None,
    strategy_metrics: Optional[dict] = None,
    offloading_metrics: Optional[dict] = None,
) -> dict:
    """Compute M-Score, C-Score, L-Score, and overall composite.

    Args:
        staircase_metrics: Output of staircase.compute_metrics(), or None.
        strategy_metrics: Output of strategy_switch.compute_metrics(), or None.
        offloading_metrics: Output of offloading.compute_offloading_metrics(), or None.

    Returns:
        Dict with keys: m_score, c_score, l_score, overall.
        Each score is in [0, 1] (or NaN if no data was available for that score).
    """
    m_components: list[float] = []
    c_components: list[float] = []
    l_components: list[float] = []

    if staircase_metrics is not None:
        m_components.extend(_staircase_m(staircase_metrics))
        c_components.extend(_staircase_c(staircase_metrics))
        l_components.extend(_staircase_l(staircase_metrics))

    if strategy_metrics is not None:
        m_components.extend(_strategy_m(strategy_metrics))
        c_components.extend(_strategy_c(strategy_metrics))
        l_components.extend(_strategy_l(strategy_metrics))

    if offloading_metrics is not None:
        m_components.extend(_offloading_m(offloading_metrics))
        c_components.extend(_offloading_c(offloading_metrics))
        l_components.extend(_offloading_l(offloading_metrics))

    m_score = _clamp01(_safe_mean(m_components))
    c_score = _clamp01(_safe_mean(c_components))
    l_score = _clamp01(_safe_mean(l_components))

    # Overall: weighted average (0.4*M + 0.4*C + 0.2*L)
    # Skip any NaN scores in the weighted average, re-normalizing weights.
    weighted_parts: list[tuple[float, float]] = []  # (weight, value)
    if not pd.isna(m_score):
        weighted_parts.append((0.4, m_score))
    if not pd.isna(c_score):
        weighted_parts.append((0.4, c_score))
    if not pd.isna(l_score):
        weighted_parts.append((0.2, l_score))

    if weighted_parts:
        total_weight = sum(w for w, _ in weighted_parts)
        overall = sum(w * v for w, v in weighted_parts) / total_weight
    else:
        overall = float("nan")

    return {
        "m_score": m_score,
        "c_score": c_score,
        "l_score": l_score,
        "overall": _clamp01(overall),
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_composite_report(scores: dict) -> None:
    """Print a formatted summary of the composite MetaCog scores."""

    def _fmt(v: float) -> str:
        return f"{v:.3f}" if not pd.isna(v) else "N/A"

    print()
    print("=" * 60)
    print("  METACOG COMPOSITE SCORES")
    print("=" * 60)
    print()
    print(f"  M-Score (monitoring quality):       {_fmt(scores['m_score'])}")
    print(f"  C-Score (control quality):          {_fmt(scores['c_score'])}")
    print(f"  L-Score (monitoring-control link):  {_fmt(scores['l_score'])}")
    print()
    print(f"  Overall (0.4M + 0.4C + 0.2L):      {_fmt(scores['overall'])}")
    print()
    print("  Interpretation guide:")
    print("    0.90+  Excellent metacognition")
    print("    0.75+  Good — monitors and acts on uncertainty well")
    print("    0.60+  Moderate — some metacognitive ability")
    print("    <0.60  Weak — poor self-assessment or poor use of it")
    print("=" * 60)
