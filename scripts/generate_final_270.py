#!/usr/bin/env python3
"""Generate final 270-item benchmark set.

Strategy:
- Keep the original 60 CTT-selected staircase items (proven quality)
- Add 210 new items: 90 math + 70 logic + 50 factual
- Target: 45 items per difficulty level (1-6)
- All items verified against ground truth

Output: datasets/final_items_v2.csv + a Python ITEMS list for embedding
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random
from collections import Counter
from generators.math_gen import generate_problem_set as gen_math, verify_answer as verify_math
from generators.logic_gen import generate_problem_set as gen_logic, verify_answer as verify_logic
from generators.factual_gen import QUESTION_BANK

# ── Load original 60 items ──
ORIGINAL_ITEMS_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_items.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_items_v2.csv")
OUTPUT_PY = os.path.join(os.path.dirname(__file__), "..", "datasets", "items_270.py")

TARGET_PER_DIFFICULTY = 45
DIFFICULTIES = [1, 2, 3, 4, 5, 6]
TARGET_TOTAL = TARGET_PER_DIFFICULTY * len(DIFFICULTIES)  # 270


def load_original_items():
    """Load original 60 staircase items from final_items.csv."""
    items = []
    with open(ORIGINAL_ITEMS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["paradigm"] == "staircase":
                items.append({
                    "item_id": row["item_id"],
                    "domain": row["domain"],
                    "difficulty": int(row["difficulty_level"]),
                    "question": row["question"],
                    "correct_answer": row["correct_answer"],
                })
    return items


def map_rarity_to_difficulty(rarity, category):
    if category == "trick":
        return 4
    return min(rarity, 6)


def main():
    print("=" * 60)
    print("  Generating final 270-item benchmark set")
    print("=" * 60)

    # ── Step 1: Load original 60 items ──
    original = load_original_items()
    print(f"\n  Original staircase items: {len(original)}")

    # Track existing questions to avoid duplicates
    existing_questions = set(item["question"].strip().lower()[:80] for item in original)

    # Count what we have per difficulty
    orig_by_diff = Counter(item["difficulty"] for item in original)
    print(f"  By difficulty: {dict(sorted(orig_by_diff.items()))}")

    # ── Step 2: Calculate how many more we need per difficulty ──
    needed = {}
    for d in DIFFICULTIES:
        needed[d] = TARGET_PER_DIFFICULTY - orig_by_diff.get(d, 0)
    print(f"\n  Needed per difficulty: {dict(sorted(needed.items()))}")
    print(f"  Total needed: {sum(needed.values())}")

    # ── Step 3: Generate new math items ──
    print("\n  Generating new math items...")
    # Generate more than needed to allow filtering duplicates
    new_math = gen_math(n=300, seed=7777)
    math_by_diff = {}
    for p in new_math:
        if not verify_math(p):
            continue
        q_key = p["question"].strip().lower()[:80]
        if q_key in existing_questions:
            continue
        d = p["difficulty"]
        if d not in math_by_diff:
            math_by_diff[d] = []
        math_by_diff[d].append({
            "item_id": f"new_math_{len(math_by_diff[d]):04d}_d{d}",
            "domain": "math",
            "difficulty": d,
            "question": p["question"],
            "correct_answer": str(p["correct_answer"]),
        })
        existing_questions.add(q_key)

    # ── Step 4: Generate new logic items ──
    print("  Generating new logic items...")
    new_logic = gen_logic(n=250, seed=8888)
    logic_by_diff = {}
    for p in new_logic:
        if not verify_logic(p):
            continue
        q_key = p["question"].strip().lower()[:80]
        if q_key in existing_questions:
            continue
        d = p["difficulty"]
        if d not in logic_by_diff:
            logic_by_diff[d] = []
        logic_by_diff[d].append({
            "item_id": f"new_logic_{len(logic_by_diff[d]):04d}_d{d}",
            "domain": "logic",
            "difficulty": d,
            "question": p["question"],
            "correct_answer": str(p["correct_answer"]),
        })
        existing_questions.add(q_key)

    # ── Step 5: Add factual items ──
    print("  Loading factual items...")
    factual_by_diff = {}
    for q in QUESTION_BANK:
        d = map_rarity_to_difficulty(q["rarity"], q["category"])
        if d not in factual_by_diff:
            factual_by_diff[d] = []
        factual_by_diff[d].append({
            "item_id": f"factual_{q['qid']}",
            "domain": "factual",
            "difficulty": d,
            "question": q["question"],
            "correct_answer": q["correct_answer"],
        })

    # ── Step 6: Fill each difficulty level to 45 ──
    final_items = list(original)  # Start with original 60
    rng = random.Random(2026)

    for d in DIFFICULTIES:
        current = sum(1 for item in final_items if item["difficulty"] == d)
        still_need = TARGET_PER_DIFFICULTY - current

        if still_need <= 0:
            continue

        # Priority: factual first (adds domain diversity), then math, then logic
        sources = [
            ("factual", factual_by_diff.get(d, [])),
            ("math", math_by_diff.get(d, [])),
            ("logic", logic_by_diff.get(d, [])),
        ]

        # Allocate: ~20% factual, ~45% math, ~35% logic of remaining slots
        factual_target = min(len(sources[0][1]), max(3, int(still_need * 0.20)))
        math_target = min(len(sources[1][1]), max(5, int(still_need * 0.45)))
        logic_target = still_need - factual_target - math_target

        allocations = [
            (factual_target, sources[0][1]),
            (math_target, sources[1][1]),
            (logic_target, sources[2][1]),
        ]

        added = 0
        for target_n, pool in allocations:
            rng.shuffle(pool)
            for item in pool[:target_n]:
                if added >= still_need:
                    break
                # Skip if question already in final set
                q_key = item["question"].strip().lower()[:80]
                if any(q_key == fi["question"].strip().lower()[:80] for fi in final_items):
                    continue
                final_items.append(item)
                added += 1

        # If still short, fill from any remaining pool
        if added < still_need:
            for _, pool in allocations:
                for item in pool:
                    if added >= still_need:
                        break
                    q_key = item["question"].strip().lower()[:80]
                    if any(q_key == fi["question"].strip().lower()[:80] for fi in final_items):
                        continue
                    final_items.append(item)
                    added += 1

    # ── Step 7: Report ──
    print(f"\n  Final items: {len(final_items)}")
    domain_counts = Counter(item["domain"] for item in final_items)
    diff_counts = Counter(item["difficulty"] for item in final_items)
    print(f"  By domain: {dict(sorted(domain_counts.items()))}")
    print(f"  By difficulty: {dict(sorted(diff_counts.items()))}")

    # ── Step 8: Write CSV ──
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "domain", "difficulty",
                                                "question", "correct_answer"])
        writer.writeheader()
        writer.writerows(final_items)
    print(f"\n  Saved CSV: {OUTPUT_CSV}")

    # ── Step 9: Write Python ITEMS list for embedding ──
    with open(OUTPUT_PY, "w") as f:
        f.write("# Auto-generated: 270 items for GAUGE benchmark\n")
        f.write("# Original 60 CTT-selected + 210 new verified items\n\n")
        f.write("ITEMS = [\n")
        for item in final_items:
            q = item["question"].replace("'", "\\'").replace('"', '\\"')
            a = item["correct_answer"].replace("'", "\\'")
            f.write(f"    ('{q}', '{a}', {item['difficulty']}, '{item['domain']}'),\n")
        f.write("]\n")
    print(f"  Saved Python: {OUTPUT_PY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
