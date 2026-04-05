"""
GAUGE CTT Pilot v2 — Lightweight 1-turn evaluation for item selection.
=====================================================================
Runs ~1093 items on 3 fast models with a simplified 1-turn protocol
(solve + confidence only, no EOL or submit/abstain turns).

Purpose: Collect per-item accuracy data across models to compute
CTT statistics (p-value, discrimination, point-biserial) for
selecting the best 270 items.

Upload this to Kaggle and run. Output is printed as CSV rows
in the notebook output (copy to pilot_v2_results.csv).
"""

import kaggle_benchmarks as kbench
import pandas as pd
import re
from dataclasses import dataclass


# ── Structured output schema (1-turn: solve + confidence) ──

@dataclass
class PilotResponse:
    reasoning: str
    answer: str
    confidence: int


PILOT_PROMPT = """Solve this problem step by step.

{question}

Give:
1. Your final answer (just the number for math, or Yes/No/the answer for logic/factual -- no extra words)
2. Your confidence from 0 to 100 that your answer is correct.
  100 = certain, 50 = coin flip, 15 = guessing."""


# ── Answer checking (same as benchmark) ──

def _check_answer(model_answer, expected):
    ma = model_answer.strip().rstrip(".")
    ex = expected.strip()
    if ma.lower() == ex.lower():
        return True
    try:
        ex_num = float(ex.replace("$", "").replace(",", ""))
        for n in re.findall(r'-?\$?[\d,]+\.?\d*', ma):
            try:
                if abs(float(n.replace("$", "").replace(",", "")) - ex_num) < 0.01:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass
    if ex.lower() in ("yes", "no", "cannot be determined"):
        if ex.lower() in ma.lower():
            return True
    if ex.lower() in ("red", "blue", "green", "yellow", "purple", "orange"):
        if ex.lower() in ma.lower():
            return True
    # Factual: check containment for longer answers
    if len(ex) > 3 and ex.lower() in ma.lower():
        return True
    # Factual: check if any word-level match for short answers
    ex_words = set(ex.lower().split())
    ma_words = set(ma.lower().split())
    if len(ex_words) == 1 and ex_words.issubset(ma_words):
        return True
    return False


# ── Pilot task (1-turn only) ──

@kbench.task(name="gauge_pilot_v2")
def gauge_pilot_v2(llm, item_id: str, question: str, correct_answer: str,
                   difficulty: int, domain: str):
    """Simplified 1-turn pilot for CTT item analysis."""
    difficulty = int(difficulty)

    response = llm.prompt(PILOT_PROMPT.format(question=question), schema=PilotResponse)
    confidence = max(0, min(100, response.confidence))

    is_correct = _check_answer(response.answer.strip(), correct_answer.strip())

    # Store result in assertion string for parsing
    kbench.assertions.assert_true(True, expectation=(
        f"PILOT|item_id={item_id}|difficulty={difficulty}|domain={domain}|"
        f"confidence={confidence}|correct={int(is_correct)}|"
        f"answer={response.answer.strip()}|expected={correct_answer.strip()}"))


# ── Load candidate pool (embedded as CSV string for Kaggle) ──
# NOTE: You must paste the candidate_pool.csv content here before uploading.
# Use the helper at the bottom to generate the embedded data.

import io

# === PASTE CANDIDATE POOL CSV HERE ===
# Run: python -c "print(open('datasets/candidate_pool.csv').read())" > paste this below
POOL_CSV = """item_id,domain,difficulty,question,correct_answer,template_id,seed
"""
# === END PASTE ===

# If POOL_CSV has only the header, load from file (local dev)
if POOL_CSV.strip().count("\n") < 2:
    try:
        eval_df = pd.read_csv("datasets/candidate_pool.csv")
    except FileNotFoundError:
        eval_df = pd.read_csv("../datasets/candidate_pool.csv")
else:
    eval_df = pd.read_csv(io.StringIO(POOL_CSV))

# Rename columns for task parameters
eval_df = eval_df.rename(columns={"correct_answer": "correct_answer"})
eval_df["difficulty"] = eval_df["difficulty"].astype(int)

print(f"Loaded {len(eval_df)} items")
print(f"  Domains: {eval_df['domain'].value_counts().to_dict()}")
print(f"  Difficulties: {eval_df['difficulty'].value_counts().sort_index().to_dict()}")

# ── Run on 3 fast models ──
PILOT_MODELS = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["anthropic/claude-haiku-4-5@20251001"],
    kbench.llms["qwen/qwen3-235b-a22b-instruct-2507"],
]

pilot_results = gauge_pilot_v2.evaluate(
    llm=PILOT_MODELS,
    evaluation_data=eval_df
)

# ── Parse and print results as CSV ──
print("\n=== PILOT RESULTS CSV ===")
print("item_id,model,difficulty,domain,confidence,correct,answer,expected")

for run in pilot_results.runs:
    model_name = "unknown"
    for msg in run.chat.history:
        sender_name = getattr(msg.sender, "name", "")
        if sender_name and sender_name not in ("User", "Assertion", "unknown"):
            model_name = sender_name
            break
    for ar in run.assertion_results:
        if ar.expectation.startswith("PILOT|"):
            fields = {}
            for kv in ar.expectation.split("|")[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    fields[k] = v
            print(f"{fields.get('item_id','')},{model_name},"
                  f"{fields.get('difficulty','')},{fields.get('domain','')},"
                  f"{fields.get('confidence','')},{fields.get('correct','')},"
                  f"{fields.get('answer','')},{fields.get('expected','')}")

print("=== END PILOT RESULTS ===")

# %choose gauge_pilot_v2
