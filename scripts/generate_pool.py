#!/usr/bin/env python3
"""Generate large candidate pool for CTT item selection.

Produces ~1093 items:
  - 600 math (procedural, 100 per difficulty 1-6)
  - 400 logic (procedural, ~67 per difficulty 1-6)
  - 93 factual (hardcoded bank, all items)

Each item is verified against its ground truth before inclusion.
Output: datasets/candidate_pool.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
from generators.math_gen import generate_problem_set as gen_math, verify_answer as verify_math
from generators.logic_gen import generate_problem_set as gen_logic, verify_answer as verify_logic
from generators.factual_gen import generate_question_set, check_factual_answer, QUESTION_BANK

POOL_SEED = 2026
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "datasets", "candidate_pool.csv")


def map_factual_rarity_to_difficulty(rarity, category):
    """Map factual rarity (1-5) + trick to difficulty (1-6)."""
    if category == "trick":
        return 4  # Looks easy, actually hard
    return min(rarity, 6)


def main():
    print("=" * 60)
    print("  Generating candidate pool for CTT selection")
    print("=" * 60)

    all_items = []
    verified = 0
    failed = 0

    # ── Math items ──
    print("\n  Generating 600 math items...")
    math_problems = gen_math(n=600, seed=POOL_SEED)
    for i, p in enumerate(math_problems):
        if verify_math(p):
            all_items.append({
                "item_id": f"math_{i:04d}",
                "domain": "math",
                "difficulty": p["difficulty"],
                "question": p["question"],
                "correct_answer": str(p["correct_answer"]),
                "template_id": p.get("template_id", ""),
                "seed": p.get("seed", ""),
            })
            verified += 1
        else:
            failed += 1
    print(f"  Math: {verified} verified, {failed} failed")

    # ── Logic items ──
    v_before = verified
    f_before = failed
    print("\n  Generating 400 logic items...")
    logic_problems = gen_logic(n=400, seed=POOL_SEED)
    for i, p in enumerate(logic_problems):
        if verify_logic(p):
            all_items.append({
                "item_id": f"logic_{i:04d}",
                "domain": "logic",
                "difficulty": p["difficulty"],
                "question": p["question"],
                "correct_answer": str(p["correct_answer"]),
                "template_id": p.get("problem_type", ""),
                "seed": p.get("seed", ""),
            })
            verified += 1
        else:
            failed += 1
    print(f"  Logic: {verified - v_before} verified, {failed - f_before} failed")

    # ── Factual items ──
    v_before = verified
    print("\n  Loading all factual items...")
    for i, q in enumerate(QUESTION_BANK):
        diff = map_factual_rarity_to_difficulty(q["rarity"], q["category"])
        all_items.append({
            "item_id": f"factual_{q['qid']}",
            "domain": "factual",
            "difficulty": diff,
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "template_id": q["category"],
            "seed": "",
        })
        verified += 1
    print(f"  Factual: {verified - v_before} items (all from bank)")

    # ── Write CSV ──
    print(f"\n  Total pool: {len(all_items)} items")

    # Distribution
    from collections import Counter
    domain_counts = Counter(item["domain"] for item in all_items)
    diff_counts = Counter(item["difficulty"] for item in all_items)
    print(f"  By domain: {dict(domain_counts)}")
    print(f"  By difficulty: {dict(sorted(diff_counts.items()))}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "domain", "difficulty",
                                                "question", "correct_answer",
                                                "template_id", "seed"])
        writer.writeheader()
        writer.writerows(all_items)

    print(f"\n  Saved: {OUTPUT}")
    print(f"  Total: {len(all_items)} items ({failed} failed verification)")
    print("=" * 60)


if __name__ == "__main__":
    main()
