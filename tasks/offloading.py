"""Cognitive Offloading — Paradigm 3: Optimal tool-use under resource constraints.

Measures whether models can optimally allocate limited resources (lookup tokens)
by accurately assessing what they know vs. what they don't.

Cognitive science grounding:
  - Risko & Gilbert (2016): Cognitive offloading = using external actions to reduce
    cognitive demand. The decision to offload is governed by metacognition.
  - Kadavath et al. (2022): P(IK) — "probability I know" — models can partially
    assess their own knowledge boundaries.
  - INTENT (Liu et al. 2026): Budget-constrained tool use as sequential
    decision-making with priced, stochastic tool executions.

Design:
  20 factual questions spanning rarity 1-5.
  Budget of 10 lookup tokens (guaranteed correct answers).
  Model must decide for each: MEMORY (free) or LOOKUP (costs 1 token).

  This paradigm is powerful because it requires JOINT monitoring + control:
  - Monitoring alone (knowing what you know) doesn't help without good allocation
  - Control alone (spending tokens) doesn't help without knowing where to spend them
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import json
import re
import pandas as pd
import numpy as np

from generators.factual_gen import (
    generate_question_set,
    check_factual_answer,
    FactualQuestion,
)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

@dataclass
class AllocationResponse:
    """Turn 1: Budget allocation decisions for all questions."""
    decisions: str  # JSON string: list of {qid, choice}


@dataclass
class MemoryAnswers:
    """Turn 2: Answers for MEMORY questions."""
    answers: str  # JSON string: list of {qid, answer, confidence}


# ---------------------------------------------------------------------------
# Problem set generation
# ---------------------------------------------------------------------------

def build_evaluation_data(
    n_questions: int = 25,
    budget: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate factual questions and package as evaluation data.

    Returns DataFrame with: questions_json, budget, seed
    (single row — the task processes all 20 questions as one unit).
    """
    questions = generate_question_set(n=n_questions, seed=seed)

    return pd.DataFrame([{
        "questions_json": json.dumps([
            {"qid": q["qid"], "question": q["question"],
             "correct_answer": q["correct_answer"],
             "aliases": q["aliases"],
             "category": q["category"], "rarity": q["rarity"]}
            for q in questions
        ]),
        "budget": budget,
        "seed": seed,
    }])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TURN1_PROMPT = """You are being evaluated on your ability to know what you know and what you don't.

You have {n_questions} factual questions and {budget} LOOKUP tokens.

For each question, you must decide:
  - MEMORY: Answer from your own knowledge (FREE, but you might be wrong)
  - LOOKUP: Use a lookup token (GUARANTEED correct, costs 1 token)

You have exactly {budget} tokens — you cannot use more. Allocate them wisely:
  - Use LOOKUP on questions you're UNSURE about
  - Use MEMORY on questions you're CONFIDENT about
  - Your goal is to maximize total correct answers

Here are the questions:

{questions_list}

For each question, decide MEMORY or LOOKUP.

Respond with a JSON object:
{{"decisions": [{{"qid": "question_id", "choice": "memory" or "lookup"}}]}}

Make sure you include a decision for ALL {n_questions} questions and use EXACTLY {budget} lookup tokens."""

TURN2_PROMPT = """Now answer the questions you chose MEMORY for.

For each MEMORY question, provide:
1. Your answer (be concise and specific)
2. Your confidence (0-100) that your answer is correct

Questions to answer from memory:
{memory_questions}

Respond with a JSON object:
{{"answers": [{{"qid": "question_id", "answer": "your answer", "confidence": <0-100>}}]}}"""


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------

def compute_offloading_metrics(
    questions: list[dict],
    decisions: dict[str, str],  # qid → "memory" or "lookup"
    memory_answers: dict[str, dict],  # qid → {answer, confidence}
    budget: int,
) -> dict:
    """Compute all cognitive offloading metrics.

    Args:
        questions: List of question dicts with qid, correct_answer, aliases, rarity.
        decisions: Map from qid to "memory" or "lookup".
        memory_answers: Map from qid to {answer, confidence} for memory questions.
        budget: Total lookup budget.
    """
    n = len(questions)
    tokens_spent = sum(1 for d in decisions.values() if d == "lookup")

    # Score each question
    results = []
    for q in questions:
        qid = q["qid"]
        decision = decisions.get(qid, "memory")
        rarity = q["rarity"]

        if decision == "lookup":
            is_correct = True  # lookups are always correct
            confidence = 100
            answer = q["correct_answer"]
        else:
            ma = memory_answers.get(qid, {})
            answer = ma.get("answer", "")
            confidence = ma.get("confidence", 50)
            is_correct = check_factual_answer(answer, q)

        results.append({
            "qid": qid,
            "rarity": rarity,
            "category": q["category"],
            "decision": decision,
            "answer": answer,
            "correct": int(is_correct),
            "confidence": confidence / 100.0,
            "expected": q["correct_answer"],
        })

    total_correct = sum(r["correct"] for r in results)
    memory_items = [r for r in results if r["decision"] == "memory"]
    lookup_items = [r for r in results if r["decision"] == "lookup"]

    # ── Net utility ──
    net_utility = total_correct  # maximize correct answers given budget constraint

    # ── Memory accuracy ──
    memory_correct = sum(r["correct"] for r in memory_items)
    memory_accuracy = memory_correct / len(memory_items) if memory_items else float("nan")

    # ── Optimal allocation ──
    # Oracle: if we knew which memory answers would be wrong, we'd lookup those
    # Simulate: answer all from memory, then lookup the ones we'd get wrong
    memory_would_be_wrong = []
    for q in questions:
        qid = q["qid"]
        # Check if the model would get it right from memory
        ma = memory_answers.get(qid, {})
        answer = ma.get("answer", "")
        would_be_correct = check_factual_answer(answer, q) if answer else False
        if not would_be_correct:
            memory_would_be_wrong.append(qid)

    # For items we don't have memory answers for (because they were looked up),
    # we don't know if memory would have been correct — treat as unknown
    # Conservative: assume lookup items would have been wrong (justify the lookup)
    n_wrong_from_memory = len(memory_would_be_wrong)

    # Optimal: lookup exactly min(budget, n_wrong) of the hardest questions
    optimal_lookups = min(budget, n_wrong_from_memory)
    optimal_correct = (n - n_wrong_from_memory) + optimal_lookups  # memory correct + looked up
    # But we can't compute true optimal since we don't know lookup items' memory accuracy
    # Use actual results instead
    actual_memory_wrong = sum(1 for r in memory_items if not r["correct"])
    wasted_lookups = 0  # lookups on items the model would have gotten right anyway
    # We can't know this perfectly, but we can estimate

    # ── Offloading calibration ──
    # Did the model lookup high-rarity items and memory low-rarity items?
    lookup_rarities = [r["rarity"] for r in lookup_items]
    memory_rarities = [r["rarity"] for r in memory_items]
    mean_lookup_rarity = np.mean(lookup_rarities) if lookup_rarities else float("nan")
    mean_memory_rarity = np.mean(memory_rarities) if memory_rarities else float("nan")

    # ── AUROC: rarity as predictor of lookup decision ──
    if lookup_items and memory_items:
        from sklearn.metrics import roc_auc_score
        binary_lookup = [1 if r["decision"] == "lookup" else 0 for r in results]
        rarities = [r["rarity"] for r in results]
        try:
            rarity_auroc = float(roc_auc_score(binary_lookup, rarities))
        except ValueError:
            rarity_auroc = float("nan")
    else:
        rarity_auroc = float("nan")

    # ── Memory confidence calibration ──
    if memory_items:
        memory_confs = [r["confidence"] for r in memory_items]
        memory_corrs = [r["correct"] for r in memory_items]
        from scoring.calibration import ece, brier_score, auroc as auroc_fn
        m_ece = ece(memory_confs, memory_corrs)
        m_brier = brier_score(memory_confs, memory_corrs)
        m_auroc = auroc_fn(memory_confs, memory_corrs)
    else:
        m_ece = float("nan")
        m_brier = float("nan")
        m_auroc = float("nan")

    # ── Rarity-stratified accuracy ──
    rarity_acc = {}
    for rarity in range(1, 6):
        items = [r for r in results if r["rarity"] == rarity]
        if items:
            rarity_acc[rarity] = sum(r["correct"] for r in items) / len(items)

    # ── Allocation efficiency ──
    # What fraction of tokens were "well-spent" (on items model would get wrong from memory)?
    # We only know for memory items (we have their answers)
    # For lookup items, we'd need a counterfactual

    return {
        # Core
        "total_correct": total_correct,
        "total_questions": n,
        "accuracy": total_correct / n,
        "tokens_spent": tokens_spent,
        "budget": budget,
        "net_utility": net_utility,
        # Memory performance
        "memory_count": len(memory_items),
        "memory_correct": memory_correct,
        "memory_accuracy": memory_accuracy,
        "lookup_count": len(lookup_items),
        # Allocation quality
        "mean_lookup_rarity": mean_lookup_rarity,
        "mean_memory_rarity": mean_memory_rarity,
        "rarity_auroc": rarity_auroc,
        # Memory calibration
        "memory_ece": m_ece,
        "memory_brier": m_brier,
        "memory_auroc": m_auroc,
        # Rarity breakdown
        "rarity_accuracy": rarity_acc,
        # Per-item results
        "item_results": results,
    }


def print_report(metrics: dict) -> None:
    """Print formatted summary of offloading metrics."""
    print("\n" + "=" * 60)
    print("  COGNITIVE OFFLOADING — RESULTS")
    print("=" * 60)

    print(f"\n  Total correct:    {metrics['total_correct']} / {metrics['total_questions']}"
          f" ({metrics['accuracy']:.1%})")
    print(f"  Tokens spent:     {metrics['tokens_spent']} / {metrics['budget']}")

    print(f"\n  ── ALLOCATION QUALITY ──")
    print(f"  Memory items:     {metrics['memory_count']} (accuracy: "
          f"{metrics['memory_accuracy']:.1%})" if not pd.isna(metrics['memory_accuracy']) else
          f"  Memory items:     {metrics['memory_count']}")
    print(f"  Lookup items:     {metrics['lookup_count']} (always correct)")
    lr = metrics['mean_lookup_rarity']
    mr = metrics['mean_memory_rarity']
    lr_str = f"{lr:.1f}" if not pd.isna(lr) else "N/A"
    mr_str = f"{mr:.1f}" if not pd.isna(mr) else "N/A"
    print(f"  Mean lookup rarity: {lr_str} (higher = better allocation)")
    print(f"  Mean memory rarity: {mr_str} (lower = better allocation)")
    ra_str = f"{metrics['rarity_auroc']:.4f}" if not pd.isna(metrics['rarity_auroc']) else "N/A"
    print(f"  Rarity→Lookup AUROC: {ra_str}")

    print(f"\n  ── MEMORY CALIBRATION ──")
    ece_str = f"{metrics['memory_ece']:.4f}" if not pd.isna(metrics['memory_ece']) else "N/A"
    brier_str = f"{metrics['memory_brier']:.4f}" if not pd.isna(metrics['memory_brier']) else "N/A"
    auroc_str = f"{metrics['memory_auroc']:.4f}" if not pd.isna(metrics['memory_auroc']) else "N/A"
    print(f"  ECE (lower=better):    {ece_str}")
    print(f"  Brier (lower=better):  {brier_str}")
    print(f"  AUROC (higher=better): {auroc_str}")

    print(f"\n  ── ACCURACY BY RARITY ──")
    for rarity in range(1, 6):
        acc = metrics['rarity_accuracy'].get(rarity)
        if acc is not None:
            print(f"  Rarity {rarity}: {acc:.1%}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@kbench.task(name="cognitive_offloading")
def cognitive_offloading(
    llm,
    questions_json: str,
    budget: int,
    seed: int,
) -> bool:
    """Run the cognitive offloading paradigm.

    Evaluates:
      - Monitoring: Does the model know what it knows?
      - Control: Does the model allocate lookups to maximize accuracy?
      - Coupling: Does monitoring accuracy drive allocation quality?
    """
    budget = int(budget)
    questions = json.loads(questions_json)
    n = len(questions)

    # Format questions for prompt
    q_list = "\n".join(
        f"  [{q['qid']}] {q['question']}"
        for q in questions
    )

    # ── Turn 1: Allocation ──
    turn1 = llm.prompt(
        TURN1_PROMPT.format(
            n_questions=n,
            budget=budget,
            questions_list=q_list,
        ),
        schema=AllocationResponse,
    )

    # Parse decisions
    try:
        dec_list = json.loads(turn1.decisions)
    except (json.JSONDecodeError, TypeError):
        dec_list = []

    decisions = {}
    for d in dec_list:
        qid = d.get("qid", "")
        choice = d.get("choice", "memory").lower().strip()
        if choice not in ("memory", "lookup"):
            choice = "memory"
        decisions[qid] = choice

    # Enforce budget: if too many lookups, convert extras to memory
    lookup_count = sum(1 for v in decisions.values() if v == "lookup")
    if lookup_count > budget:
        lookup_qids = [k for k, v in decisions.items() if v == "lookup"]
        for qid in lookup_qids[budget:]:
            decisions[qid] = "memory"

    # Ensure all questions have decisions
    for q in questions:
        if q["qid"] not in decisions:
            decisions[q["qid"]] = "memory"

    # ── Turn 2: Answer memory questions ──
    memory_qids = [q["qid"] for q in questions if decisions.get(q["qid"]) == "memory"]
    memory_q_list = "\n".join(
        f"  [{q['qid']}] {q['question']}"
        for q in questions if q["qid"] in memory_qids
    )

    if memory_qids:
        turn2 = llm.prompt(
            TURN2_PROMPT.format(memory_questions=memory_q_list),
            schema=MemoryAnswers,
        )
        try:
            ans_list = json.loads(turn2.answers)
        except (json.JSONDecodeError, TypeError):
            ans_list = []

        memory_answers = {}
        for a in ans_list:
            qid = a.get("qid", "")
            memory_answers[qid] = {
                "answer": str(a.get("answer", "")),
                "confidence": max(0, min(100, int(a.get("confidence", 50)))),
            }
    else:
        memory_answers = {}

    # ── Compute metrics ──
    q_lookup = {q["qid"]: q for q in questions}
    metrics = compute_offloading_metrics(questions, decisions, memory_answers, budget)

    # Store results
    kbench.assertions.assert_true(
        True,
        expectation=(
            f"RESULT|"
            f"total_correct={metrics['total_correct']}|"
            f"accuracy={metrics['accuracy']:.4f}|"
            f"tokens_spent={metrics['tokens_spent']}|"
            f"memory_accuracy={metrics['memory_accuracy']:.4f}|"
            f"rarity_auroc={metrics['rarity_auroc']:.4f}|"
            f"memory_ece={metrics['memory_ece']:.4f}"
        ),
    )

    return metrics["accuracy"] > 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_evaluation_data(n_questions=20, budget=10, seed=42)
    print(f"Generated evaluation data: {len(df)} rows")
    questions = json.loads(df.iloc[0]["questions_json"])
    print(f"  Questions: {len(questions)}")
    print(f"  Budget: {df.iloc[0]['budget']}")

    results = cognitive_offloading.evaluate(
        llm=kbench.llm,
        evaluation_data=df,
    )
    print(f"\nEvaluation complete. Results: {results}")
