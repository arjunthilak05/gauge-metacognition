"""Metacognitive Staircase — the core benchmark task.

Measures both metacognitive MONITORING and CONTROL across a difficulty gradient.

Three-turn interaction per problem:
  Turn 1 (Ease-of-Learning): Predict difficulty before solving.
  Turn 2 (Confidence Judgment): Solve + state confidence 0-100.
  Turn 3 (Control Decision): Submit (+3/-1) or Abstain (+1/0) based on confidence.

Cognitive science grounding:
  - Ease-of-Learning (EOL): Nelson & Narens (1990) — pre-task monitoring
  - Confidence Judgment (CJ): classic calibration paradigm
  - Control Decision: the "thermostat" — does monitoring actually drive behavior?
  - Scoring asymmetry (+3/-1/+1/0) creates a decision-theoretic incentive
    where rational abstention threshold depends on confidence accuracy.

Prompt design follows Tian et al. 2023 ("Just Ask for Calibration"):
  - Two-step elicitation (answer first, then confidence)
  - Numerical 0-100 scale with explicit instruction to use full range
"""

from __future__ import annotations

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import pandas as pd

from generators.math_gen import generate_problem_set as math_problems
from generators.logic_gen import generate_problem_set as logic_problems
from scoring.calibration import (
    ece,
    brier_score,
    brier_skill_score,
    auroc,
    eol_difficulty_correlation,
    abstention_optimality,
)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

@dataclass
class EOLResponse:
    """Turn 1: Ease-of-Learning — difficulty prediction before solving."""
    difficulty_estimate: int  # 1-6


@dataclass
class SolutionResponse:
    """Turn 2: Solve the problem and report confidence."""
    reasoning: str
    answer: str
    confidence: int  # 0-100


@dataclass
class ControlResponse:
    """Turn 3: Submit or abstain based on confidence."""
    decision: str  # "submit" or "abstain"
    reasoning_for_decision: str


# ---------------------------------------------------------------------------
# Problem set generation
# ---------------------------------------------------------------------------

def build_evaluation_data(seed: int = 42) -> pd.DataFrame:
    """Generate a mixed problem set: 20 math + 10 logic, balanced by difficulty.

    Returns a DataFrame with columns: question, correct_answer, difficulty, domain.
    """
    math = math_problems(n=20, seed=seed)
    logic = logic_problems(n=10, seed=seed + 1000)

    rows = []
    for p in math:
        rows.append({
            "question": p["question"],
            "correct_answer": str(p["correct_answer"]),
            "difficulty": p["difficulty"],
            "domain": "math",
        })
    for p in logic:
        rows.append({
            "question": p["question"],
            "correct_answer": str(p["correct_answer"]),
            "difficulty": p["difficulty"],
            "domain": "logic",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TURN1_PROMPT = """You are being evaluated on your ability to assess problem difficulty.

Here is a problem:

{question}

BEFORE solving it, estimate how difficult this problem is on a scale of 1-6:
  1 = trivial (instant answer)
  2 = easy (one simple step)
  3 = moderate (2-3 steps, some thought required)
  4 = challenging (multiple steps, careful reasoning needed)
  5 = hard (complex reasoning, easy to make mistakes)
  6 = extremely hard (many steps, high chance of error)

Do NOT solve the problem yet. Just assess its difficulty."""

TURN2_PROMPT = """Now solve the problem. Show your reasoning step by step.

After solving, give:
1. Your final answer (just the number for math, or just Yes/No/the answer for logic — no extra words)
2. Your confidence from 0 to 100 that this answer is correct.

IMPORTANT: Use the FULL range. 100 means you would bet your life on it. 95 means very confident but could have made a small error. 80 means fairly sure. 50 means coin flip. Be honest — overconfidence is penalized in later scoring."""

TURN3_PROMPT = """Your confidence was {confidence}%.

You now have a choice with real scoring consequences:
  (A) SUBMIT your answer
      → If correct: +3 points
      → If wrong: -1 point
  (B) ABSTAIN
      → Guaranteed 0 points regardless of correctness

The optimal strategy is to submit when you're likely correct and abstain when you're likely wrong.

What do you choose? Respond with "submit" or "abstain" and explain your reasoning."""


# ---------------------------------------------------------------------------
# Answer matching — robust to prose wrapping
# ---------------------------------------------------------------------------

def _check_answer(model_answer: str, expected: str) -> bool:
    """Flexible answer comparison: handles '$836', '15 toy cars', 'Yes, X is Y.'"""
    import re
    ma = model_answer.strip().rstrip(".")
    ex = expected.strip()

    # Exact match
    if ma.lower() == ex.lower():
        return True

    # Numeric extraction
    try:
        ex_num = float(ex.replace("$", "").replace(",", ""))
        nums = re.findall(r'-?\$?[\d,]+\.?\d*', ma)
        for n in nums:
            try:
                if abs(float(n.replace("$", "").replace(",", "")) - ex_num) < 0.01:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    # String containment for logic answers
    if ex.lower() in ("yes", "no", "cannot be determined"):
        if ex.lower() in ma.lower():
            return True

    # Color answers
    if ex.lower() in ("red", "blue", "green", "yellow", "purple", "orange"):
        if ex.lower() in ma.lower():
            return True

    return False


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@kbench.task(name="metacognitive_staircase")
def metacognitive_staircase(
    llm,
    question: str,
    correct_answer: str,
    difficulty: int,
    domain: str,
) -> bool:
    """Run the 3-turn metacognitive staircase for a single problem.

    Evaluates:
      - Monitoring: Can the model predict difficulty and calibrate confidence?
      - Control: Does the model make rational submit/abstain decisions?
    """
    difficulty = int(difficulty)

    # ── Turn 1: Ease-of-Learning ──
    turn1 = llm.prompt(
        TURN1_PROMPT.format(question=question),
        schema=EOLResponse,
    )
    kbench.assertions.assert_true(
        1 <= turn1.difficulty_estimate <= 6,
        expectation="difficulty_estimate must be between 1 and 6",
    )

    # ── Turn 2: Solve + Confidence ──
    turn2 = llm.prompt(
        TURN2_PROMPT,
        schema=SolutionResponse,
    )
    confidence = max(0, min(100, turn2.confidence))
    kbench.assertions.assert_true(
        0 <= turn2.confidence <= 100,
        expectation="confidence must be between 0 and 100",
    )

    # ── Turn 3: Submit or Abstain ──
    turn3 = llm.prompt(
        TURN3_PROMPT.format(confidence=confidence),
        schema=ControlResponse,
    )
    decision = turn3.decision.strip().lower()
    kbench.assertions.assert_true(
        decision in ("submit", "abstain"),
        expectation="decision must be 'submit' or 'abstain'",
    )

    # ── Check correctness ──
    model_answer = turn2.answer.strip()
    expected = correct_answer.strip()
    is_correct = _check_answer(model_answer, expected)

    # ── Compute item score ──
    if decision == "submit":
        item_score = 3 if is_correct else -1
    else:  # abstain
        item_score = 1 if not is_correct else 0

    # ── Store results for aggregation ──
    # The task function returns True/False for the SDK, but we also
    # record detailed metrics via assertions for the results table.
    kbench.assertions.assert_true(
        True,
        expectation=(
            f"RESULT|"
            f"eol={turn1.difficulty_estimate}|"
            f"actual_diff={difficulty}|"
            f"confidence={confidence}|"
            f"correct={int(is_correct)}|"
            f"decision={decision}|"
            f"score={item_score}|"
            f"domain={domain}|"
            f"answer={model_answer}|"
            f"expected={expected}"
        ),
    )

    return is_correct


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Compute all metacognitive metrics from a list of per-item results.

    Each result dict must have: confidence, correct, decision, eol, actual_diff.
    """
    confidences = [r["confidence"] / 100.0 for r in results]
    correctness = [r["correct"] for r in results]
    decisions = [r["decision"] for r in results]
    eol_preds = [r["eol"] for r in results]
    actual_diffs = [r["actual_diff"] for r in results]

    # Monitoring metrics
    m_ece = ece(confidences, correctness)
    m_brier = brier_score(confidences, correctness)
    m_bss = brier_skill_score(confidences, correctness)
    m_auroc = auroc(confidences, correctness)
    m_eol_corr = eol_difficulty_correlation(eol_preds, actual_diffs)

    # Control metrics
    ctrl = abstention_optimality(decisions, correctness)

    # Composite: M-Score (monitoring), C-Score (control), L-Score (coupling)
    # M-Score: average of normalized metrics (higher = better monitoring)
    m_score = 0.0
    m_components = 0
    if not pd.isna(m_auroc):
        m_score += m_auroc  # already 0-1, higher = better
        m_components += 1
    m_score += (1 - m_ece)  # invert ECE so higher = better
    m_components += 1
    m_score += (1 - m_brier)  # invert Brier
    m_components += 1
    if not pd.isna(m_eol_corr):
        m_score += (m_eol_corr + 1) / 2  # normalize [-1,1] to [0,1]
        m_components += 1
    m_score /= m_components

    # C-Score: score efficiency (how close to optimal submit/abstain)
    c_score = ctrl["score_efficiency"] if not pd.isna(ctrl["score_efficiency"]) else 0.0

    # L-Score: coupling between monitoring and control
    # Measures whether high confidence → submit and low confidence → abstain
    # Computed as point-biserial correlation between confidence and submit decision
    submit_binary = [1 if d == "submit" else 0 for d in decisions]
    if len(set(submit_binary)) > 1:
        l_corr, _ = __import__("scipy").stats.pointbiserialr(
            submit_binary, confidences
        )
        l_score = float(max(0, l_corr))  # clamp negative to 0
    else:
        l_score = 0.0  # no variance in decisions = no coupling

    return {
        # Monitoring
        "ece": m_ece,
        "brier": m_brier,
        "brier_skill": m_bss,
        "auroc": m_auroc,
        "eol_correlation": m_eol_corr,
        # Control
        "total_score": ctrl["total_score"],
        "optimal_score": ctrl["optimal_score"],
        "score_efficiency": ctrl["score_efficiency"],
        "abstain_precision": ctrl["abstain_precision"],
        "abstain_count": ctrl["abstain_count"],
        "submit_count": ctrl["submit_count"],
        # Composite
        "m_score": m_score,
        "c_score": c_score,
        "l_score": l_score,
        # Accuracy
        "accuracy": sum(correctness) / len(correctness) if correctness else 0,
        "n_items": len(results),
    }


def print_report(metrics: dict) -> None:
    """Print a formatted summary table of all metrics."""
    print("\n" + "=" * 60)
    print("  METACOGNITIVE STAIRCASE — RESULTS")
    print("=" * 60)

    print(f"\n  Items evaluated: {metrics['n_items']}")
    print(f"  Base accuracy:   {metrics['accuracy']:.1%}")

    print(f"\n  ── MONITORING (M-Score: {metrics['m_score']:.3f}) ──")
    print(f"  ECE (↓ better):           {metrics['ece']:.4f}")
    print(f"  Brier (↓ better):         {metrics['brier']:.4f}")
    print(f"  Brier Skill (↑ better):   {metrics['brier_skill']:.4f}")
    auroc_str = f"{metrics['auroc']:.4f}" if not pd.isna(metrics['auroc']) else "N/A (single class)"
    print(f"  AUROC (↑ better):         {auroc_str}")
    eol_str = f"{metrics['eol_correlation']:.4f}" if not pd.isna(metrics['eol_correlation']) else "N/A"
    print(f"  EOL-Difficulty ρ:         {eol_str}")

    print(f"\n  ── CONTROL (C-Score: {metrics['c_score']:.3f}) ──")
    print(f"  Game Score:               {metrics['total_score']} / {metrics['optimal_score']} optimal")
    print(f"  Score Efficiency:         {metrics['score_efficiency']:.1%}")
    print(f"  Submit/Abstain:           {metrics['submit_count']} / {metrics['abstain_count']}")
    ap_str = f"{metrics['abstain_precision']:.1%}" if not pd.isna(metrics['abstain_precision']) else "N/A"
    print(f"  Abstain Precision:        {ap_str}")

    print(f"\n  ── COMPOSITE ──")
    print(f"  M-Score (monitoring):     {metrics['m_score']:.3f}")
    print(f"  C-Score (control):        {metrics['c_score']:.3f}")
    print(f"  L-Score (coupling):       {metrics['l_score']:.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main: run locally
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build evaluation data
    df = build_evaluation_data(seed=42)
    print(f"Generated {len(df)} problems:")
    print(f"  Math: {(df['domain'] == 'math').sum()}")
    print(f"  Logic: {(df['domain'] == 'logic').sum()}")
    print(f"  Difficulty distribution: {df['difficulty'].value_counts().sort_index().to_dict()}")

    # Run against default model
    results = metacognitive_staircase.evaluate(
        llm=kbench.llm,
        evaluation_data=df,
    )
    print(f"\nEvaluation complete. Results: {results}")
