"""Item analysis using Classical Test Theory for benchmark item selection.

Computes per-item and test-level statistics to identify items that maximize
measurement quality across LLM respondents.

Theoretical grounding:
  - Kelley (1939): 27% extreme-group split for D-index (but we use 50% split
    for small N per Beuchert & Mendoza 1979)
  - Guilford (1954): Corrected point-biserial (item-remainder correlation)
    removes spurious part-whole overlap
  - Ebel & Frisbie (1991): D-index interpretation thresholds
  - Kline (2005), Nunnally & Bernstein (1994): r >= 0.20 minimum for
    item-total correlation
  - Crocker & Algina (2006): SE of point-biserial ~ 1/sqrt(N), so with
    N=5 models, SE ~ 0.45 — all correlations are exploratory flags, not
    definitive filters

Small-sample caveats (N = 3-8 LLMs):
  - P-values are coarsely quantized (N=5 → p in {0, 0.2, 0.4, 0.6, 0.8, 1})
  - Point-biserial SE ~ 1/sqrt(N) ~ 0.35-0.58; use as soft flags only
  - D-index with 50% split (not 27%) for stability
  - Alpha-if-deleted is the most actionable metric for identifying bad items
  - All thresholds are soft flags, not hard cutoffs

Recent LLM benchmark psychometrics:
  - Madaan et al. (2024): CTT/IRT "struggle to meaningfully reduce variance"
    in LLM benchmarks — item analysis is exploratory, not definitive
  - Zhou et al. (2025): PSN-IRT framework reveals "significant shortcomings
    in measurement quality" across 11 LLM benchmarks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# 1. Per-item statistics
# ---------------------------------------------------------------------------

def compute_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Classical Test Theory statistics for each item.

    Uses corrected point-biserial (item-remainder correlation) to avoid
    spurious part-whole inflation, and 50% split for D-index to handle
    small respondent counts (Beuchert & Mendoza 1979).

    Args:
        df: DataFrame with columns: model, item_id, correct (0/1).
            Optional columns: difficulty_level, confidence, paradigm.

    Returns:
        DataFrame indexed by item_id with columns:
          - n_models: number of models tested on this item
          - p_value: proportion correct (item difficulty; 0=hard, 1=easy)
          - rpb_corrected: corrected point-biserial (item-remainder r)
          - discrimination_index: upper-half accuracy minus lower-half accuracy
          - item_variance: p * (1 - p)
          - difficulty_level, paradigm: from input if available
    """
    if df.empty:
        return pd.DataFrame()

    required = {"model", "item_id", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Total score per model (sum of correct across all items)
    model_totals = df.groupby("model")["correct"].sum().rename("total_score")
    df = df.merge(model_totals, on="model", how="left")

    # Corrected total: total_score minus this item's score (removes part-whole overlap)
    df["total_minus_item"] = df["total_score"] - df["correct"]

    results = []
    for item_id, group in df.groupby("item_id"):
        n_models = len(group)
        p_value = group["correct"].mean()
        item_variance = p_value * (1 - p_value)

        # Corrected point-biserial: correlate item score with total_minus_item
        # (Guilford 1954 correction for spurious part-whole overlap)
        if n_models >= 3 and group["correct"].nunique() > 1:
            rpb, _ = scipy_stats.pointbiserialr(
                group["correct"].values, group["total_minus_item"].values
            )
            rpb_corrected = float(rpb)
        else:
            rpb_corrected = float("nan")

        # Discrimination index with 50% split (not 27%) for small N
        # With N=5, 27% split = 1.35 → comparing 1 vs 1 respondent = meaningless
        # 50% split uses all data and is stable even with small N
        sorted_group = group.sort_values("total_score")
        n_half = n_models // 2
        if n_half >= 1:
            bottom = sorted_group.head(n_half)["correct"].mean()
            top = sorted_group.tail(n_half)["correct"].mean()
            discrimination_index = float(top - bottom)
        else:
            discrimination_index = float("nan")

        row = {
            "item_id": item_id,
            "n_models": n_models,
            "p_value": float(p_value),
            "rpb_corrected": rpb_corrected,
            "discrimination_index": discrimination_index,
            "item_variance": float(item_variance),
        }

        if "difficulty_level" in group.columns:
            row["difficulty_level"] = group["difficulty_level"].iloc[0]
        if "paradigm" in group.columns:
            row["paradigm"] = group["paradigm"].iloc[0]

        results.append(row)

    return pd.DataFrame(results).set_index("item_id")


# ---------------------------------------------------------------------------
# Test-level reliability
# ---------------------------------------------------------------------------

def compute_reliability(df: pd.DataFrame) -> dict:
    """Compute test-level reliability statistics.

    Args:
        df: DataFrame with columns: model, item_id, correct (0/1).

    Returns:
        Dict with:
          - kr20: Kuder-Richardson 20 (= Cronbach's alpha for binary items)
          - sem: Standard Error of Measurement
          - alpha_if_deleted: dict of item_id → alpha when that item is dropped
          - mean_inter_item_r: average pairwise item correlation
          - n_items, n_models: counts
    """
    if df.empty:
        return {}

    # Pivot to items × models matrix
    pivot = df.pivot_table(
        index="model", columns="item_id", values="correct", aggfunc="first"
    )
    pivot = pivot.dropna(axis=1, how="all").fillna(0)

    k = pivot.shape[1]  # number of items
    n = pivot.shape[0]  # number of models

    if k < 2 or n < 2:
        return {"kr20": float("nan"), "sem": float("nan"),
                "alpha_if_deleted": {}, "mean_inter_item_r": float("nan"),
                "n_items": k, "n_models": n}

    # KR-20 (Kuder-Richardson formula 20) = Cronbach's alpha for binary items
    # alpha = (k / (k-1)) * (1 - sum(p_i * q_i) / var_total)
    total_scores = pivot.sum(axis=1)
    var_total = float(total_scores.var(ddof=1))

    item_p = pivot.mean(axis=0)
    sum_pq = float((item_p * (1 - item_p)).sum())

    if var_total > 0:
        kr20 = float((k / (k - 1)) * (1 - sum_pq / var_total))
    else:
        kr20 = float("nan")

    # Standard Error of Measurement
    sd_total = float(total_scores.std(ddof=1))
    if not pd.isna(kr20) and kr20 < 1:
        sem = sd_total * np.sqrt(1 - max(0, kr20))
    else:
        sem = float("nan")

    # Alpha-if-deleted: most actionable metric for small N
    # If alpha increases when an item is dropped, that item hurts reliability
    alpha_if_deleted = {}
    for item in pivot.columns:
        reduced = pivot.drop(columns=item)
        k_r = reduced.shape[1]
        if k_r < 2:
            alpha_if_deleted[item] = float("nan")
            continue
        total_r = reduced.sum(axis=1)
        var_r = float(total_r.var(ddof=1))
        p_r = reduced.mean(axis=0)
        sum_pq_r = float((p_r * (1 - p_r)).sum())
        if var_r > 0:
            alpha_if_deleted[item] = float(
                (k_r / (k_r - 1)) * (1 - sum_pq_r / var_r)
            )
        else:
            alpha_if_deleted[item] = float("nan")

    # Mean inter-item correlation
    corr_matrix = pivot.corr()
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_vals = corr_matrix.where(mask).stack().values
    mean_inter_item_r = float(np.nanmean(upper_vals)) if len(upper_vals) > 0 else float("nan")

    return {
        "kr20": kr20,
        "sem": sem,
        "alpha_if_deleted": alpha_if_deleted,
        "mean_inter_item_r": mean_inter_item_r,
        "n_items": k,
        "n_models": n,
    }


# ---------------------------------------------------------------------------
# 2. Goldilocks filtering (soft flags for small N)
# ---------------------------------------------------------------------------

def filter_goldilocks(
    item_stats: pd.DataFrame,
    min_p: float = 0.15,
    max_p: float = 0.85,
    min_disc: float = 0.2,
    min_rpb: float = 0.10,
    strict: bool = False,
) -> pd.DataFrame:
    """Filter to 'Goldilocks' items: adequate difficulty and discrimination.

    With small N (3-8 models), thresholds are soft flags. The `strict`
    parameter controls whether min_rpb is enforced (default: no, because
    SE ~ 1/sqrt(N) makes point-biserial unreliable at small N).

    Thresholds per Ebel & Frisbie (1991) and Kline (2005):
      - p in [0.15, 0.85]: adequate variance (liberal bounds for small N)
      - D >= 0.20: acceptable discrimination
      - rpb >= 0.10: minimal item-total correlation (soft flag)

    Args:
        item_stats: Output of compute_item_stats().
        min_p: Minimum p-value (default 0.15).
        max_p: Maximum p-value (default 0.85).
        min_disc: Minimum discrimination index (default 0.2).
        min_rpb: Minimum corrected point-biserial (default 0.10).
        strict: If True, also enforce min_rpb. If False (default), only
                use p-value and D-index filters (safer for small N).

    Returns:
        Filtered DataFrame.
    """
    mask = (
        (item_stats["p_value"] >= min_p)
        & (item_stats["p_value"] <= max_p)
        & (item_stats["discrimination_index"] >= min_disc)
    )

    if strict:
        rpb = item_stats["rpb_corrected"].fillna(0)
        mask = mask & (rpb >= min_rpb)

    return item_stats[mask].copy()


# ---------------------------------------------------------------------------
# 3. Optimal item selection
# ---------------------------------------------------------------------------

def select_optimal_items(
    item_stats: pd.DataFrame,
    n_target: int = 150,
) -> list[str]:
    """Select items that maximize total test information.

    Scoring composite (weights informed by CTT theory):
      - 40% discrimination index (primary driver of test reliability)
      - 35% p-value quality (distance from 0.5; max info at p=0.5)
      - 25% corrected point-biserial (item-remainder correlation)

    Items with p = 0.0 or p = 1.0 are always excluded (zero variance).

    Args:
        item_stats: Output of compute_item_stats().
        n_target: Target number of items to select.

    Returns:
        List of selected item_id strings.
    """
    if item_stats.empty:
        return []

    df = item_stats.copy()

    # Exclude zero-variance items (all correct or all wrong)
    df = df[(df["p_value"] > 0) & (df["p_value"] < 1)]
    if df.empty:
        return []

    # p-value quality: 1.0 at p=0.5, 0.0 at p=0.0 or p=1.0
    df["p_quality"] = 1.0 - 2.0 * abs(df["p_value"] - 0.5)
    df["p_quality"] = df["p_quality"].clip(0, 1)

    # Discrimination (clip to [0, 1])
    df["disc_norm"] = df["discrimination_index"].clip(0, 1)

    # Corrected point-biserial (handle NaN, clip to [0, 1])
    df["rpb_norm"] = df["rpb_corrected"].fillna(0).clip(0, 1)

    # Composite score
    df["selection_score"] = (
        0.40 * df["disc_norm"]
        + 0.35 * df["p_quality"]
        + 0.25 * df["rpb_norm"]
    )

    df = df.sort_values("selection_score", ascending=False)
    selected = df.head(n_target).index.tolist()
    return selected


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_difficulty_distribution(
    item_stats: pd.DataFrame,
    selected_ids: list[str] | None = None,
    title: str = "Item Difficulty Distribution (p-values)",
) -> "matplotlib.figure.Figure":
    """Histogram of p-values, optionally highlighting selected items.

    Args:
        item_stats: Output of compute_item_stats().
        selected_ids: If provided, overlay selected items in a different color.
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_p = item_stats["p_value"].values
    bins = np.linspace(0, 1, 21)

    # Left: p-value histogram
    ax = axes[0]
    if selected_ids is not None:
        selected_mask = item_stats.index.isin(selected_ids)
        rejected_p = item_stats.loc[~selected_mask, "p_value"].values
        selected_p = item_stats.loc[selected_mask, "p_value"].values

        ax.hist(rejected_p, bins=bins, alpha=0.4, color="gray",
                label=f"Rejected ({len(rejected_p)})", edgecolor="white")
        ax.hist(selected_p, bins=bins, alpha=0.7, color="steelblue",
                label=f"Selected ({len(selected_p)})", edgecolor="white")
    else:
        ax.hist(all_p, bins=bins, alpha=0.7, color="steelblue", edgecolor="white")

    ax.axvspan(0.3, 0.7, alpha=0.08, color="green")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("p-value (proportion correct)")
    ax.set_ylabel("Number of items")
    ax.set_title(title)
    ax.legend()

    # Right: discrimination vs p-value scatter
    ax2 = axes[1]
    disc = item_stats["discrimination_index"].values
    if selected_ids is not None:
        sel = item_stats.index.isin(selected_ids)
        ax2.scatter(all_p[~sel], disc[~sel], alpha=0.4, c="gray",
                    label="Rejected", s=30)
        ax2.scatter(all_p[sel], disc[sel], alpha=0.7, c="steelblue",
                    label="Selected", s=30)
    else:
        ax2.scatter(all_p, disc, alpha=0.7, c="steelblue", s=30)

    ax2.axhline(0.2, color="orange", linestyle="--", alpha=0.5, label="D=0.2 threshold")
    ax2.axvspan(0.3, 0.7, alpha=0.08, color="green")
    ax2.set_xlabel("p-value")
    ax2.set_ylabel("Discrimination index (D)")
    ax2.set_title("Item Discrimination vs Difficulty")
    ax2.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_item_analysis(
    item_stats: pd.DataFrame,
    reliability: dict | None = None,
    selected_ids: list[str] | None = None,
) -> None:
    """Print a summary of item analysis results."""
    n_total = len(item_stats)
    print("\n" + "=" * 60)
    print("  ITEM ANALYSIS (Classical Test Theory)")
    print("=" * 60)

    n_models = item_stats["n_models"].iloc[0] if n_total > 0 else 0
    print(f"\n  Items: {n_total} | Models: {n_models}")

    if n_models <= 8:
        print(f"  WARNING: N={n_models} is below recommended minimum (30+).")
        print(f"  All statistics are exploratory flags, not definitive filters.")

    print(f"\n  -- ITEM DIFFICULTY (p-value) --")
    print(f"  Range: [{item_stats['p_value'].min():.2f}, "
          f"{item_stats['p_value'].max():.2f}]")
    print(f"  Mean:  {item_stats['p_value'].mean():.3f}")

    too_easy = (item_stats["p_value"] > 0.85).sum()
    too_hard = (item_stats["p_value"] < 0.15).sum()
    floor_ceil = ((item_stats["p_value"] == 0) | (item_stats["p_value"] == 1)).sum()
    print(f"  Too easy (p > 0.85): {too_easy} | Too hard (p < 0.15): {too_hard}")
    print(f"  Floor/ceiling (p=0 or 1): {floor_ceil}")

    print(f"\n  -- DISCRIMINATION --")
    print(f"  D-index mean: {item_stats['discrimination_index'].mean():.3f} "
          f"(50% split)")
    rpb = item_stats["rpb_corrected"].dropna()
    if len(rpb) > 0:
        print(f"  Corrected r_pb mean: {rpb.mean():.3f}")

    d_excellent = (item_stats["discrimination_index"] >= 0.4).sum()
    d_good = ((item_stats["discrimination_index"] >= 0.3)
              & (item_stats["discrimination_index"] < 0.4)).sum()
    d_acceptable = ((item_stats["discrimination_index"] >= 0.2)
                    & (item_stats["discrimination_index"] < 0.3)).sum()
    d_poor = (item_stats["discrimination_index"] < 0.2).sum()
    print(f"  Excellent (D>=0.4): {d_excellent} | Good (0.3-0.4): {d_good} "
          f"| Acceptable (0.2-0.3): {d_acceptable} | Poor (<0.2): {d_poor}")

    neg_disc = (item_stats["discrimination_index"] < 0).sum()
    if neg_disc > 0:
        print(f"  ALERT: {neg_disc} items have NEGATIVE discrimination (remove these)")

    # Reliability
    if reliability:
        print(f"\n  -- TEST RELIABILITY --")
        kr20 = reliability.get("kr20", float("nan"))
        kr20_str = f"{kr20:.3f}" if not pd.isna(kr20) else "N/A"
        print(f"  KR-20 (Cronbach's alpha): {kr20_str}")
        sem = reliability.get("sem", float("nan"))
        sem_str = f"{sem:.3f}" if not pd.isna(sem) else "N/A"
        print(f"  SEM: {sem_str}")
        mir = reliability.get("mean_inter_item_r", float("nan"))
        mir_str = f"{mir:.3f}" if not pd.isna(mir) else "N/A"
        print(f"  Mean inter-item r: {mir_str} (target: 0.15-0.50)")

        # Flag items that hurt reliability (alpha increases when dropped)
        aid = reliability.get("alpha_if_deleted", {})
        if aid and not pd.isna(kr20):
            hurting = [item for item, a in aid.items()
                       if not pd.isna(a) and a > kr20 + 0.01]
            if hurting:
                print(f"  Items hurting reliability ({len(hurting)}):")
                for item in hurting[:5]:
                    print(f"    {item}: alpha_if_deleted = {aid[item]:.3f}")

    # Goldilocks summary
    goldilocks = (
        (item_stats["p_value"] >= 0.15)
        & (item_stats["p_value"] <= 0.85)
        & (item_stats["discrimination_index"] >= 0.2)
    ).sum()
    print(f"\n  -- GOLDILOCKS ITEMS --")
    print(f"  Meeting all criteria: {goldilocks} / {n_total} "
          f"({goldilocks/n_total:.0%})" if n_total > 0 else "  No items")

    if selected_ids is not None:
        selected = item_stats.loc[item_stats.index.isin(selected_ids)]
        print(f"\n  -- SELECTED ITEMS ({len(selected_ids)}) --")
        print(f"  Mean p-value: {selected['p_value'].mean():.3f}")
        print(f"  Mean D-index: {selected['discrimination_index'].mean():.3f}")
        srpb = selected["rpb_corrected"].dropna()
        if len(srpb) > 0:
            print(f"  Mean corrected r_pb: {srpb.mean():.3f}")

    # By paradigm
    if "paradigm" in item_stats.columns:
        print(f"\n  -- BY PARADIGM --")
        for paradigm, group in item_stats.groupby("paradigm"):
            print(f"  {paradigm}: {len(group)} items, "
                  f"p={group['p_value'].mean():.3f}, "
                  f"D={group['discrimination_index'].mean():.3f}")

    # By difficulty level
    if "difficulty_level" in item_stats.columns:
        print(f"\n  -- BY DIFFICULTY LEVEL --")
        for diff, group in item_stats.groupby("difficulty_level"):
            print(f"  Level {diff}: {len(group)} items, "
                  f"p={group['p_value'].mean():.3f}, "
                  f"D={group['discrimination_index'].mean():.3f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    models = ["gemini-flash", "claude-sonnet", "llama-70b", "deepseek-r1",
              "gemini-pro", "qwen-235b"]
    n_items = 60

    rows = []
    for i in range(n_items):
        true_p = rng.beta(2, 2)
        paradigm = ["staircase", "strategy_switch", "offloading"][i % 3]
        diff_level = (i % 5) + 1

        for model in models:
            ability = {"gemini-flash": 0.15, "claude-sonnet": 0.1,
                       "llama-70b": -0.15, "deepseek-r1": 0.05,
                       "gemini-pro": 0.2, "qwen-235b": -0.05}[model]
            model_p = np.clip(true_p + ability, 0, 1)
            correct = int(rng.random() < model_p)
            rows.append({
                "model": model, "item_id": f"item_{i:03d}",
                "correct": correct, "difficulty_level": diff_level,
                "paradigm": paradigm,
            })

    df = pd.DataFrame(rows)
    print(f"Synthetic: {len(df)} responses, {n_items} items, {len(models)} models\n")

    stats = compute_item_stats(df)
    reliability = compute_reliability(df)
    gold = filter_goldilocks(stats)
    selected = select_optimal_items(stats, n_target=30)

    print_item_analysis(stats, reliability=reliability, selected_ids=selected)

    fig = plot_difficulty_distribution(stats, selected_ids=selected)
    fig.savefig("item_difficulty_distribution.png", dpi=150)
    print("\nPlot saved to item_difficulty_distribution.png")
