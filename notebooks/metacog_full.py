"""
GAUGE: Grading Awareness of Uncertainty and Grounded Epistemics
================================================================
Kaggle Community Benchmark — "Measuring Progress Toward AGI" Hackathon

Do LLMs know what they know? Do they act on it?

Theoretical grounding:
  Nelson & Narens (1990): metacognition = monitoring + control
  Tian et al. (2023): "Just Ask for Calibration" — 0-100 confidence scale

Primary paradigm: Metacognitive Staircase
  - 3-turn interaction per problem across 6 difficulty levels
  - Turn 1: Predict difficulty BEFORE solving (Ease-of-Learning)
  - Turn 2: Solve + report confidence 0-100
  - Turn 3: Submit (+3/-1) or Abstain (+1/0) based on confidence

Scoring: ECE, Brier, Score Efficiency, M-Score, C-Score, L-Score
Items: 60 IRT-selected items (Classical Test Theory, Kelley 1939)

MODEL EVALUATION RESILIENCE:
  - 10 models across 4 families (Gemini, Claude, Qwen, DeepSeek)
  - Batch evaluation preferred (faster). Falls back to per-model if batch fails.
  - If a model crashes (e.g., JSON parsing error), benchmark continues with others.
  - See Section 8B for resilience logic. Test individual models with test_models.py
  - Known issues: DeepSeek-R1 crashes with <think> tags; using V3.1 instead.
"""

# ============================================================
#  SECTION 1: IMPORTS
# ============================================================

import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
import re
import random
from dataclasses import dataclass

print("=" * 60)
print("  GAUGE Benchmark v1.0")
print("  Grading Awareness of Uncertainty and Grounded Epistemics")
print("  60 IRT-selected items | Nelson & Narens (1990)")
print("=" * 60)


# ============================================================
#  SECTION 2: EMBEDDED ITEM DATA (IRT-selected, 60 items)
# ============================================================
# Format: (question, correct_answer, difficulty_level, domain)
# Items selected via Classical Test Theory item analysis:
#   - p-value in [0.15, 0.85] (adequate variance)
#   - Discrimination index >= 0.2 (50% split, Beuchert & Mendoza 1979)
#   - Selection score: 40% discrimination + 35% p-quality + 25% rpb

ITEMS = [
    # --- Math items (30) ---
    ('David earned money over 2 days. On Monday, David earned $119. On Tuesday, David earned $300. David was wearing a red hat that day. How much did David earn in total?', '419', 3, 'math'),
    ("Tom wants to buy a bicycle that costs $2844. The store is offering a 50% discount today. The store's loyalty program showed Tom had 40 points, which weren't redeemable yet. The cashier mentioned they had sold 36 items earlier that day to someone else. How much does Tom pay after the discount?", '1422', 6, 'math'),
    ('Maria has 39 bananas. Maria gives 17 to Rachel and buys 42 more. How many bananas does Maria have now?', '64', 2, 'math'),
    ('Sarah has 199 pears. The store had been open since 1987. Sarah gives 75 to Aisha and buys 113 more. How many pears does Sarah have now?', '237', 3, 'math'),
    ('Carlos earned money over 3 days. On Monday, Carlos earned $294. On Tuesday, Carlos earned $513. On Wednesday, Carlos earned $189. The cashier mentioned they had sold 2 items earlier that day to someone else. How much did Carlos earn in total?', '996', 4, 'math'),
    ('Tom has 20 oranges. Tom was wearing a red hat that day. Tom gives 8 to Sarah and buys 136 more. How many oranges does Tom have now?', '148', 3, 'math'),
    ('Rachel goes to a store to buy notebooks. Each notebook costs $1. Rachel buys 17 of them. How much does Rachel spend in total?', '17', 1, 'math'),
    ("Aisha earned a total of $1280 over three days. There were 6 items in the clearance bin, but Aisha wasn't interested. On the first day, Aisha earned $580. On the second day, Aisha earned $402. How much did Aisha earn on the third day?", '298', 4, 'math'),
    ("Rachel bought 10 markers and paid $1690 in total. Each marker cost the same amount. The store's loyalty program showed Rachel had 21 points, which weren't redeemable yet. How much did each marker cost?", '169', 5, 'math'),
    ('Nadia has 661 bananas and 247 apples. Nadia gives 178 bananas to a friend. Nadia noticed that 27 other customers were in the store. How many bananas does Nadia have now?', '483', 4, 'math'),
    ('Sarah has $31. Sarah spends $7 on a book. Sarah spends $10 on a gift. How much money does Sarah have left?', '14', 2, 'math'),
    ("Sarah goes shopping with $938. Sarah buys 5 markers at $79 each. There were 29 items in the clearance bin, but Sarah wasn't interested. Sarah buys 1 pens at $48 each. Sarah buys 4 notebooks at $56 each. How much money does Sarah have left?", '271', 4, 'math'),
    ("Kenji has 118 cookies. Kenji's favorite color is blue. Lin has 3 times as many cookies as Kenji. How many cookies do they have together?", '472', 3, 'math'),
    ('Aisha has $6163. Aisha spends $1799 on a gift. Aisha spends $228 on a book. Aisha had 48 coupons in a drawer at home but forgot to bring them. Aisha spends $2314 on coffee. How much money does Aisha have left?', '1822', 5, 'math'),
    ('Priya has 416 stickers and wants to share them equally among 4 friends. Priya was wearing a red hat that day. How many stickers does each friend get?', '104', 3, 'math'),
    ('David has 28 mangoes. David gives 11 to Maria and buys 27 more. How many mangoes does David have now?', '44', 1, 'math'),
    ('James has 12 cookies and wants to share them equally among 6 friends. How many cookies does each friend get?', '2', 1, 'math'),
    ("James has 1566 cookies. The store also had 25 envelopes on display, but James didn't buy any. Nadia has 2 times as many cookies as James. James's friend had mentioned wanting 45 cookies, but James didn't get any for them. How many cookies do they have together?", '4698', 6, 'math'),
    ('Nadia builds 38 widgets per day. The store had been open since 1987. If Nadia works for 8 days, how many widgets does Nadia produce?', '304', 3, 'math'),
    ("Omar has 548 bananas and 417 mangoes. Omar gives 200 bananas to a friend. Omar's friend had mentioned wanting 46 bananas, but Omar didn't get any for them. How many bananas does Omar have now?", '348', 4, 'math'),
    ("Kenji saves $109 every week for 5 weeks. During that time, Kenji spends $5 on concert tickets. Kenji's friend had mentioned wanting 39 items, but Kenji didn't get any for them. How much money does Kenji have at the end?", '540', 4, 'math'),
    ('Tom earned money over 3 days. On Monday, Tom earned $701. On Tuesday, Tom earned $4744. On Wednesday, Tom earned $5221. The shelf had 37 empty spots where items used to be. How much did Tom earn in total?', '10666', 5, 'math'),
    ('Lin bought 3 markers and paid $441 in total. On the way there, Lin passed 29 houses. Each marker cost the same amount. How much did each marker cost?', '147', 4, 'math'),
    ('Fatima goes shopping with $4509. Fatima buys 8 erasers at $156 each. Fatima had 46 coupons in a drawer at home but forgot to bring them. Fatima buys 5 markers at $113 each. Fatima buys 6 folders at $65 each. How much money does Fatima have left?', '2306', 5, 'math'),
    ("Alex earned a total of $15117 over three days. On the way there, Alex passed 10 houses. The store also had 29 batteries on display, but Alex didn't buy any. On the first day, Alex earned $2220. On the second day, Alex earned $6424. How much did Alex earn on the third day?", '6473', 6, 'math'),
    ('Kenji bought 9 notebooks and paid $315 in total. The shelf had 35 empty spots where notebooks used to be. Each notebook cost the same amount. How much did each notebook cost?', '35', 4, 'math'),
    ("Priya has 188 bananas and 180 oranges. There were 38 items in the clearance bin, but Priya wasn't interested. Priya gives 77 bananas to a friend. How many bananas does Priya have now?", '111', 4, 'math'),
    ('Aisha wants to buy a camera that costs $748. The store is offering a 50% discount today. Aisha noticed that 49 other customers were in the store. How much does Aisha pay after the discount?', '374', 4, 'math'),
    ('Nadia earned money over 4 days. On Monday, Nadia earned $17. On Tuesday, Nadia earned $18. On Wednesday, Nadia earned $26. On Thursday, Nadia earned $16. How much did Nadia earn in total?', '77', 2, 'math'),
    ('Tom has 340 mangoes. The receipt was printed on recycled paper. Tom gives 131 to Aisha and buys 472 more. How many mangoes does Tom have now?', '681', 3, 'math'),
    # --- Logic items (30) ---
    ('Finn is a Snazzles. Every Snazzles is a Murnips. Every Murnips is a Flimbers. All Blorps are wise. Every Flimbers is a Wumpuses. Is Finn a Quibbles?', 'Cannot be determined', 4, 'logic'),
    ("Mina's favorite color is orange. Jake enjoys gardening. Finn's favorite color is yellow. Gina's favorite color is blue. Iris does not like blue or orange. Iris's favorite color is red. Noel does not like blue. What is Noel's favorite color?", 'purple', 3, 'logic'),
    ('If something is a Snazzles, it might be wise. Ora is a Quibbles. Every Quibbles is a Tazzles. Every Tazzles is a Grumpkins. Every Grumpkins is a Fizzgigs. Is Ora a Fizzgigs?', 'Yes', 3, 'logic'),
    ('All Snazzles are kind. Ora is a Murnips. All Murnips are quiet. All quiet things are kind. Is Ora not kind?', 'No', 4, 'logic'),
    ('Hugo is a Drazzles. All Drazzles are fast. If something is fast, then it is wise. Some Quibbles are tall. If something is wise, then it is kind. Is Hugo clever?', 'Cannot be determined', 5, 'logic'),
    ('All Vexlings are wise. Kira is a Drazzles. All Drazzles are quiet. If something is quiet, then it is wise. Is Kira wise?', 'Yes', 3, 'logic'),
    ('Paul is a Vexlings. Some Tazzles are clever. All Vexlings are kind. If something is kind, then it is friendly. Is Paul friendly?', 'Yes', 4, 'logic'),
    ('Paul is a Murnips. All Murnips are wise. If something is wise, then it is fast. If something is fast, then it is clever. All Fizzgigs are wise. Some Drogons are happy. Is Paul clever?', 'Yes', 6, 'logic'),
    ('Cora is a Vexlings. All Vexlings are brave. Some Flimbers are quiet. If something is brave, then it is fast. Is Cora wise?', 'Cannot be determined', 3, 'logic'),
    ('Paul is a Drogons. Some Plonkers are friendly. All Drogons are fast. If something is fast, then it is happy. Is Paul happy?', 'Yes', 3, 'logic'),
    ("Hugo's favorite color is red. Cora enjoys gardening. Finn's favorite color is blue. Ora does not like blue or purple. Ora's favorite color is orange. Paul does not like orange. What is Paul's favorite color?", 'purple', 5, 'logic'),
    ("Gina's favorite color is purple. Eve does not like blue or purple. Eve's favorite color is green. Iris does not like blue or purple. Noel enjoys painting. Iris's favorite color is orange. Paul does not like purple. What is Paul's favorite color?", 'blue', 3, 'logic'),
    ('Dan is a Murnips. Every Murnips is a Wumpuses. Every Wumpuses is a Snazzles. Is Dan a Wumpuses?', 'Yes', 2, 'logic'),
    ('Leo is a Wumpuses. All Grumpkins are kind. Every Wumpuses is a Zephlins. Every Zephlins is a Snazzles. Every Snazzles is a Drazzles. Is Leo a Drogons?', 'Cannot be determined', 4, 'logic'),
    ("Leo's favorite color is blue. Cora does not like blue or red. Eve enjoys cooking. Gina enjoys painting. Cora's favorite color is yellow. Ava does not like purple or blue. Ava's favorite color is green. Jake's favorite color is red. Paul does not like yellow. What is Paul's favorite color?", 'purple', 6, 'logic'),
    ('Mina is a Fizzgigs. All Fizzgigs are tall. If something is tall, then it is clever. Is Mina clever?', 'Yes', 2, 'logic'),
    ('Some Plonkers are fast. Paul is a Grumpkins. Every Grumpkins is a Drazzles. Every Drazzles is a Kelpoids. Every Kelpoids is a Blorps. Every Blorps is a Wumpuses. Is Paul a Flimbers?', 'Cannot be determined', 5, 'logic'),
    ('Paul is a Vexlings. All Flimbers are fast. All Vexlings are kind. If something is kind, then it is wise. If something is wise, then it is fast. Some Drogons are strong. Is Paul fast?', 'Yes', 6, 'logic'),
    ('Mina is a Plonkers. All Plonkers are quiet. All quiet things are clever. If something is a Tazzles, it might be happy. Is Mina not clever?', 'No', 4, 'logic'),
    ('Some Drogons are brave. Cora is a Grumpkins. All Grumpkins are tall. If something is tall, then it is fast. Is Cora fast?', 'Yes', 3, 'logic'),
    ("Eve's favorite color is yellow. Hugo's favorite color is red. Ora enjoys chess. Leo's favorite color is blue. Kira does not like yellow. What is Kira's favorite color?", 'green', 4, 'logic'),
    ("Dan's favorite color is blue. Mina enjoys painting. Eve enjoys gardening. Ora does not like purple or green. Ora's favorite color is red. Iris does not like red or green. Iris's favorite color is yellow. Noel's favorite color is green. Finn does not like blue. What is Finn's favorite color?", 'purple', 6, 'logic'),
    ("Noel's favorite color is red. Finn's favorite color is purple. Leo does not like green or red. Hugo enjoys cooking. Leo's favorite color is orange. Dan does not like purple. What is Dan's favorite color?", 'green', 3, 'logic'),
    ('Some Drazzles are wise. Hugo is a Plonkers. All Plonkers are friendly. If something is friendly, then it is clever. Is Hugo strong?', 'Cannot be determined', 3, 'logic'),
    ('Finn is a Wumpuses. Some Drazzles are kind. Every Wumpuses is a Plonkers. Every Plonkers is a Zephlins. Every Zephlins is a Murnips. Is Finn a Grumpkins?', 'Cannot be determined', 3, 'logic'),
    ('Noel is a Zephlins. All Zephlins are strong. All Drazzles are clever. If something is strong, then it is quiet. Is Noel wise?', 'Cannot be determined', 4, 'logic'),
    ('If something is a Drazzles, it might be friendly. Kira is a Flimbers. All Flimbers are quiet. If something is quiet, then it is strong. Is Kira strong?', 'Yes', 3, 'logic'),
    ("Ava's favorite color is purple. Finn does not like purple or green. Finn's favorite color is yellow. Jake's favorite color is red. Kira does not like yellow or red. Kira's favorite color is green. Gina enjoys painting. Iris does not like red. What is Iris's favorite color?", 'blue', 5, 'logic'),
    ("Eve's favorite color is orange. Kira's favorite color is yellow. Ora's favorite color is purple. Cora does not like yellow or purple. Mina enjoys cooking. Cora's favorite color is blue. Noel does not like blue. Ava enjoys painting. What is Noel's favorite color?", 'red', 6, 'logic'),
    ('Paul is a Grumpkins. All Flimbers are brave. All Grumpkins are quiet. All quiet things are wise. Is Paul wise?', 'Yes', 4, 'logic'),
]

N_ITEMS = len(ITEMS)
print(f"\n  Items loaded: {N_ITEMS}")
print(f"  Math: {sum(1 for _,_,_,d in ITEMS if d=='math')}")
print(f"  Logic: {sum(1 for _,_,_,d in ITEMS if d=='logic')}")
print(f"  Difficulty: { {d: sum(1 for _,_,dd,_ in ITEMS if dd==d) for d in sorted(set(dd for _,_,dd,_ in ITEMS))} }")


# ============================================================
#  SECTION 3: ANSWER CHECKING
# ============================================================

def _check_answer(model_answer, expected):
    """Robust answer comparison: exact, numeric, containment, colors."""
    ma = model_answer.strip().rstrip(".")
    ex = expected.strip()
    if ma.lower() == ex.lower():
        return True
    # Numeric extraction
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
    # Logic: Yes/No/Cannot be determined
    if ex.lower() in ("yes", "no", "cannot be determined"):
        if ex.lower() in ma.lower():
            return True
    # Colors
    if ex.lower() in ("red", "blue", "green", "yellow", "purple", "orange"):
        if ex.lower() in ma.lower():
            return True
    return False


# ============================================================
#  SECTION 4: SCORING FUNCTIONS
# ============================================================

def score_ece(confidences, correctness, n_bins=15):
    """Expected Calibration Error (equal-mass binning)."""
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    n = len(conf)
    if n == 0:
        return float("nan")
    idx = np.argsort(conf)
    bsz = max(1, n // n_bins)
    err = 0.0
    for i in range(n_bins):
        b = idx[i * bsz: (i + 1) * bsz] if i < n_bins - 1 else idx[i * bsz:]
        if len(b) == 0:
            continue
        err += len(b) / n * abs(conf[b].mean() - corr[b].mean())
    return float(err)


def score_brier(confidences, correctness):
    """Brier score: MSE between confidence and correctness."""
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    return float(np.mean((conf - corr) ** 2)) if len(conf) > 0 else float("nan")


def score_abstention(decisions, correctness):
    """Abstention optimality: game score, efficiency, precision."""
    corr = np.array(correctness, dtype=int)
    total = 0
    for d, c in zip(decisions, corr):
        if d == "submit":
            total += 3 if c else -1
        else:
            total += 1 if not c else 0
    n_correct = int(corr.sum())
    n_wrong = len(corr) - n_correct
    optimal = n_correct * 3 + n_wrong * 1
    worst = n_wrong * -1
    efficiency = (total - worst) / (optimal - worst) if optimal != worst else float("nan")
    n_submit = sum(1 for d in decisions if d == "submit")
    n_abstain = sum(1 for d in decisions if d == "abstain")
    abstain_wrong = sum(1 for d, c in zip(decisions, corr) if d == "abstain" and c == 0)
    abstain_prec = abstain_wrong / n_abstain if n_abstain > 0 else float("nan")
    return {
        "total_score": total, "optimal_score": optimal,
        "score_efficiency": efficiency,
        "submit_count": n_submit, "abstain_count": n_abstain,
        "abstain_precision": abstain_prec,
    }


# ============================================================
#  SECTION 5: STRUCTURED OUTPUT SCHEMAS
# ============================================================

@dataclass
class EOLResponse:
    """Turn 1: Ease-of-Learning judgment (Nelson & Narens 1990)."""
    difficulty_estimate: int  # 1-6

@dataclass
class SolutionResponse:
    """Turn 2: Solve + confidence (Tian et al. 2023)."""
    reasoning: str
    answer: str
    confidence: int  # 0-100

@dataclass
class ControlResponse:
    """Turn 3: Submit/Abstain control decision."""
    decision: str  # "submit" or "abstain"
    reasoning_for_decision: str


# ============================================================
#  SECTION 6: PROMPTS
# ============================================================

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
1. Your final answer (just the number for math, or just Yes/No/the answer for logic -- no extra words)
2. Your confidence from 0 to 100 that this answer is correct.

IMPORTANT: Use the FULL 0-100 range honestly. Research shows AI models are systematically overconfident — please calibrate carefully.
  100 = certain, would bet your life on it
  95 = very confident, but could have made a small error
  80 = fairly sure, checked reasoning once
  60 = more right than wrong, but real chance of error
  50 = coin flip, genuinely unsure
  30 = more likely wrong than right
  15 = guessing, very low confidence
Your confidence will directly affect your score in the next step, so accuracy matters more than optimism."""

TURN3_PROMPT = """Your confidence was {confidence}%.

You now have a choice with real scoring consequences:
  (A) SUBMIT your answer
      -> If correct: +3 points
      -> If wrong:   -1 point
  (B) ABSTAIN (decline to answer)
      -> If your answer would have been wrong:   +1 point (rewarded caution)
      -> If your answer would have been correct:  0 points (missed opportunity)

Game theory: if your true probability of being correct is p, then
  EV(submit) = 4p - 1, and EV(abstain) = 1 - p.
  These are equal at p = 0.40. Below 40% confidence, abstaining maximizes expected score.

What do you choose? Respond with "submit" or "abstain" and explain your reasoning."""


# ============================================================
#  SECTION 7: GAUGE STAIRCASE TASK (PRIMARY — goes on leaderboard)
# ============================================================

@kbench.task(name="gauge_staircase")
def gauge_staircase(llm, question: str, correct_answer: str,
                    difficulty: int, domain: str):
    """GAUGE Metacognitive Staircase: EOL -> Solve+Confidence -> Submit/Abstain.

    Measures both MONITORING (can the model assess its own accuracy?)
    and CONTROL (does the model act on that assessment?).
    """
    difficulty = int(difficulty)

    # Turn 1: Ease-of-Learning prediction
    turn1 = llm.prompt(TURN1_PROMPT.format(question=question), schema=EOLResponse)
    kbench.assertions.assert_true(
        1 <= turn1.difficulty_estimate <= 6,
        expectation="difficulty_estimate must be between 1 and 6")

    # Turn 2: Solve + Confidence judgment
    turn2 = llm.prompt(TURN2_PROMPT, schema=SolutionResponse)
    confidence = max(0, min(100, turn2.confidence))
    kbench.assertions.assert_true(
        0 <= turn2.confidence <= 100,
        expectation="confidence must be between 0 and 100")

    # Turn 3: Submit or Abstain (metacognitive control)
    turn3 = llm.prompt(TURN3_PROMPT.format(confidence=confidence), schema=ControlResponse)
    decision = turn3.decision.strip().lower()
    kbench.assertions.assert_true(
        decision in ("submit", "abstain"),
        expectation="decision must be 'submit' or 'abstain'")

    # Check correctness
    is_correct = _check_answer(turn2.answer.strip(), correct_answer.strip())

    # Compute item score
    if decision == "submit":
        item_score = 3 if is_correct else -1
    else:
        item_score = 1 if not is_correct else 0

    # Store detailed results via assertion string
    kbench.assertions.assert_true(True, expectation=(
        f"RESULT|eol={turn1.difficulty_estimate}|actual_diff={difficulty}|"
        f"confidence={confidence}|correct={int(is_correct)}|decision={decision}|"
        f"score={item_score}|domain={domain}|"
        f"answer={turn2.answer.strip()}|expected={correct_answer.strip()}"))


# ============================================================
#  SECTION 8: BUILD DATA & RUN
# ============================================================

print("\n" + "-" * 60)
print("  Building evaluation data...")
print("-" * 60)

eval_df = pd.DataFrame([
    {"question": q, "correct_answer": a, "difficulty": d, "domain": dom}
    for q, a, d, dom in ITEMS
])

print(f"  {len(eval_df)} items ready")

# Run against ALL available models in one shot
# The SDK evaluates every model × every item automatically
# NOTE: DeepSeek R1 outputs <think>...</think> tags that crash the SDK's JSON parser.
# Run models that support clean structured output only.
# To add more models, pick from: kbench.llms.keys()

# 9 models across 4 families for maximum statistical power
# Gemini: 4 models (does L=0.0 hold across all generations?)
# Claude: 3 models (does strategic abstention persist in newest?)
# Qwen: 1 model (Qwen3-Coder removed — 0 parseable results, same as Gemma)
# DeepSeek: 1 model (V3.1 — R1's <think> tags crash SDK JSON parser)
ALL_MODELS = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["google/gemini-2.5-pro"],
    kbench.llms["google/gemini-3-flash-preview"],
    kbench.llms["google/gemini-3-pro-preview"],
    kbench.llms["anthropic/claude-sonnet-4@20250514"],
    kbench.llms["anthropic/claude-haiku-4-5@20251001"],
    kbench.llms["anthropic/claude-sonnet-4-6@default"],
    kbench.llms["qwen/qwen3-235b-a22b-instruct-2507"],
    kbench.llms["deepseek-ai/deepseek-v3.1"],
]

print(f"  Models: {len(ALL_MODELS)} (4 families: 4 Gemini, 3 Claude, 1 Qwen, 1 DeepSeek)")
print(f"  Total runs: {len(eval_df)} items x {len(ALL_MODELS)} models = {len(eval_df) * len(ALL_MODELS)} evals")
print(f"  Running 3-turn staircase on each item per model...\n")

# ============================================================
#  SECTION 8B: ROBUST MODEL EVALUATION WITH FALLBACK
# ============================================================
# Strategy: Try batch evaluation first. If any model fails (e.g., JSON parsing),
# fall back to per-model evaluation and skip the failing model.
# This prevents the entire benchmark from crashing due to one problematic model.

staircase_results = None
successful_models = []
failed_models = []

# Attempt 1: Batch evaluation (faster, preferred)
print(f"  Attempting batch evaluation with all {len(ALL_MODELS)} models...")
try:
    staircase_results = gauge_staircase.evaluate(
        llm=ALL_MODELS, evaluation_data=eval_df
    )
    # If batch succeeds, record all models as successful
    successful_models = [m.name if hasattr(m, 'name') else str(m) for m in ALL_MODELS]
    print(f"  ✅ Batch evaluation succeeded.\n")
except Exception as batch_error:
    print(f"  ⚠️  Batch evaluation failed: {type(batch_error).__name__}: {str(batch_error)[:100]}")
    print(f"  Falling back to per-model evaluation...\n")

    # Attempt 2: Per-model evaluation (slower, more resilient)
    all_runs = []
    for idx, model in enumerate(ALL_MODELS, 1):
        model_name = model.name if hasattr(model, 'name') else f"Model_{idx}"
        print(f"  [{idx}/{len(ALL_MODELS)}] Evaluating {model_name}...", end=" ")

        try:
            model_result = gauge_staircase.evaluate(
                llm=model, evaluation_data=eval_df, timeout=300  # per-model timeout
            )
            all_runs.extend(model_result.runs)
            successful_models.append(model_name)
            print(f"✅")
        except Exception as model_error:
            error_msg = str(model_error)[:80]
            print(f"❌ {type(model_error).__name__}: {error_msg}")
            failed_models.append({
                'model': model_name,
                'error_type': type(model_error).__name__,
                'error_msg': str(model_error)[:200]
            })

    # Wrap per-model results in a results object
    if all_runs:
        class ResultsWrapper:
            def __init__(self, runs):
                self.runs = runs
        staircase_results = ResultsWrapper(all_runs)
        print(f"\n  ✅ Per-model evaluation complete.")
    else:
        print(f"\n  ❌ All models failed. Cannot continue.")
        raise RuntimeError(f"All {len(ALL_MODELS)} models failed evaluation. No results available.")

# Summary of model success/failure
print(f"\n  Model Evaluation Summary:")
print(f"  ├─ Successful: {len(successful_models)} models")
for m in successful_models:
    print(f"  │  ✓ {m}")
if failed_models:
    print(f"  ├─ Failed: {len(failed_models)} models")
    for fail in failed_models:
        print(f"  │  ✗ {fail['model']} ({fail['error_type']})")
print(f"\n")


# ============================================================
#  SECTION 9: PARSE RESULTS & COMPUTE METRICS (ALL MODELS)
# ============================================================

# ── Pure-numpy replacements for scipy/sklearn (not available on Kaggle) ──

def _rankdata(x):
    """Rank data (average ranks for ties), replacing scipy.stats.rankdata."""
    arr = np.asarray(x, dtype=float)
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(arr) + 1, dtype=float)
    # Average ties
    for val in np.unique(arr):
        mask = arr == val
        ranks[mask] = ranks[mask].mean()
    return ranks

def spearmanr(x, y):
    """Spearman rank correlation (numpy-only)."""
    rx, ry = _rankdata(x), _rankdata(y)
    n = len(rx)
    if n < 3:
        return float("nan"), 1.0
    mx, my = rx.mean(), ry.mean()
    cov = np.sum((rx - mx) * (ry - my))
    sx = np.sqrt(np.sum((rx - mx) ** 2))
    sy = np.sqrt(np.sum((ry - my) ** 2))
    if sx == 0 or sy == 0:
        return float("nan"), 1.0
    return float(cov / (sx * sy)), 0.0  # (rho, pvalue_placeholder)

def pointbiserialr(binary, continuous):
    """Point-biserial correlation (numpy-only)."""
    b = np.asarray(binary, dtype=float)
    c = np.asarray(continuous, dtype=float)
    n = len(b)
    if n < 3 or b.std() == 0 or c.std() == 0:
        return 0.0, 1.0
    return float(np.corrcoef(b, c)[0, 1]), 0.0

def kendalltau(x, y):
    """Kendall tau-b correlation (numpy-only, O(n^2) but fine for n=60)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), 1.0
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1; ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    denom = np.sqrt((total_pairs - ties_x) * (total_pairs - ties_y))
    if denom == 0:
        return float("nan"), 1.0
    tau = (concordant - discordant) / denom
    return float(tau), 0.0

def roc_auc_score(y_true, y_score):
    """AUROC (numpy-only, via Mann-Whitney U statistic)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Count how often positive scores exceed negative scores
    auc = 0.0
    for p in pos:
        auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc / (len(pos) * len(neg)))

print("=" * 60)
print("  PARSING RESULTS & FILTERING TO SUCCESSFUL MODELS")
print("=" * 60)

# Extract structured results from assertion strings, tagging each with its model
parsed = []
for run in staircase_results.runs:
    # Get model name from the run's chat history
    model_name = "unknown"
    for msg in run.chat.history:
        sender_name = getattr(msg.sender, "name", "")
        if sender_name and sender_name not in ("User", "Assertion", "unknown"):
            model_name = sender_name
            break
    for ar in run.assertion_results:
        if ar.expectation.startswith("RESULT|"):
            fields = {"model": model_name}
            for kv in ar.expectation.split("|")[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    fields[k] = v
            parsed.append(fields)

if not parsed:
    print("  WARNING: No results parsed. Check task execution.")
    # If any models succeeded (in fallback mode), they should have results
    if successful_models:
        print(f"  Expected results from: {successful_models}")
        print("  But got none. This may indicate a result parsing issue.")
else:
    results_df = pd.DataFrame(parsed)
    for col in ["eol", "actual_diff", "confidence", "correct", "score"]:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    all_models = sorted(results_df["model"].unique())
    print(f"  Parsed {len(results_df)} item results across {len(all_models)} model(s)")
    print(f"  Models with results:")
    for m in all_models:
        count = len(results_df[results_df["model"] == m])
        print(f"    • {m}: {count} items")

    # Warn if expected models are missing from results
    if failed_models:
        print(f"\n  ⚠️  {len(failed_models)} model(s) had no results (failed during evaluation)")
    if len(all_models) < len(successful_models):
        missing = set(successful_models) - set(all_models)
        print(f"  ⚠️  Results missing for these otherwise-successful models: {missing}")

    # ── Per-Difficulty Breakdown ──
    def compute_difficulty_breakdown(mdf):
        """
        Compute calibration metrics broken down by task difficulty level.

        Research (hard-easy effect): Models tend to be overconfident on hard tasks,
        underconfident on easy tasks. This function quantifies WHERE calibration breaks.

        Returns dict with keys like "difficulty_1", "difficulty_2", etc. with:
          - accuracy: % correct at this difficulty
          - mean_conf: mean confidence (0-1 scale) at this difficulty
          - ece: ECE computed only for items at this difficulty
          - n: count of items at this difficulty
          - abstain_rate: fraction of items abstained at this difficulty
          - calibration_gap: mean_conf - accuracy (positive = overconfident)
        """
        breakdown = {}

        # Group by difficulty (1-6)
        for diff_level in range(1, 7):
            diff_df = mdf[mdf["actual_diff"] == diff_level]
            if len(diff_df) < 3:  # Skip levels with <3 items
                continue

            n = len(diff_df)
            correctness = diff_df["correct"].astype(int).tolist()
            confidences_raw = (diff_df["confidence"] / 100.0).tolist()
            decisions = diff_df["decision"].tolist()

            # Accuracy
            accuracy = sum(correctness) / n if n > 0 else 0

            # Mean confidence (0-1 scale)
            mean_conf = np.mean(confidences_raw) if n > 0 else 0

            # ECE for this difficulty level only (equal-mass binning)
            if len(set(correctness)) > 1:  # Need both correct and incorrect
                ece_diff = score_ece(confidences_raw, correctness)
            else:
                ece_diff = float("nan")

            # Abstention rate
            abstain_count = sum(1 for d in decisions if d == "abstain")
            abstain_rate = abstain_count / n if n > 0 else 0

            # Calibration gap (positive = overconfident)
            calibration_gap = mean_conf - accuracy

            breakdown[f"difficulty_{diff_level}"] = {
                "n": n,
                "accuracy": accuracy,
                "mean_conf": mean_conf,
                "ece": ece_diff,
                "abstain_rate": abstain_rate,
                "calibration_gap": calibration_gap,
            }

        return breakdown

    # ── Compute metrics per model ──
    def compute_model_metrics(mdf):
        """Compute all metrics for a single model's results."""
        n = len(mdf)
        confidences = (mdf["confidence"] / 100.0).tolist()
        correctness = mdf["correct"].astype(int).tolist()
        decisions = mdf["decision"].tolist()
        eol_preds = mdf["eol"].astype(int).tolist()
        actual_diffs = mdf["actual_diff"].astype(int).tolist()

        accuracy = sum(correctness) / n if n > 0 else 0
        m_ece = score_ece(confidences, correctness)
        m_brier = score_brier(confidences, correctness)
        ctrl = score_abstention(decisions, correctness)

        eol_corr = spearmanr(eol_preds, actual_diffs)[0] if n >= 3 else float("nan")

        if len(set(correctness)) > 1:
            try:
                m_auroc = roc_auc_score(correctness, confidences)
            except Exception:
                m_auroc = float("nan")
        else:
            m_auroc = float("nan")

        submit_binary = [1 if d == "submit" else 0 for d in decisions]
        n_abstain = sum(1 for d in decisions if d == "abstain")
        # L-Score requires >=3 abstentions to be meaningful
        # 1-2 abstentions can produce spuriously high point-biserial correlations
        if len(set(submit_binary)) > 1 and n_abstain >= 3:
            l_corr, _ = pointbiserialr(submit_binary, confidences)
            l_score = float(max(0, l_corr))
        else:
            l_score = 0.0

        # ─── NEW METRICS ────────────────────────────────────────────
        # 1. CONFIDENCE VARIANCE: assess calibration spread
        conf_array = np.array(confidences)
        conf_std = float(np.std(conf_array)) if len(conf_array) > 0 else 0.0
        conf_range = float(np.max(conf_array) - np.min(conf_array)) if len(conf_array) > 0 else 0.0

        # 2. KENDALL TAU: robust correlation (resistant to response bias)
        # Preempts Fleming & Lau (2014) critique of point-biserial
        if len(set(submit_binary)) > 1 and n >= 3:
            try:
                l_kendall, _ = kendalltau(submit_binary, confidences)
                l_score_kendall = float(max(0, l_kendall))
            except Exception:
                l_score_kendall = float("nan")
        else:
            l_score_kendall = 0.0 if all(b == submit_binary[0] for b in submit_binary) else float("nan")

        # 3. RATIONAL THRESHOLD: optimal threshold is 40% given payoff +3/-1/+1/0
        # EV_submit = 4p-1, EV_abstain = 1-p → threshold at p=0.40
        # For models that abstain: find empirical threshold (min conf at submit)
        submit_mask = np.array(submit_binary, dtype=bool)
        if np.any(submit_mask):
            empirical_threshold = float(np.min(conf_array[submit_mask]))
        else:
            empirical_threshold = float("nan")  # Never submits

        m_components = [(1 - m_ece), (1 - m_brier)]
        if not np.isnan(m_auroc):
            m_components.append(m_auroc)
        if not np.isnan(eol_corr):
            m_components.append((eol_corr + 1) / 2)
        m_score = float(np.mean(m_components))

        c_score = ctrl["score_efficiency"] if not np.isnan(ctrl["score_efficiency"]) else 0.0
        overall = 0.4 * m_score + 0.4 * c_score + 0.2 * l_score

        # Compute per-difficulty breakdown
        difficulty_breakdown = compute_difficulty_breakdown(mdf)

        return {
            "n": n, "accuracy": accuracy,
            "ece": m_ece, "brier": m_brier, "auroc": m_auroc,
            "eol_corr": eol_corr, "ctrl": ctrl,
            "m_score": m_score, "c_score": c_score, "l_score": l_score,
            "overall": overall,
            "confidences": confidences, "correctness": correctness,
            "decisions": decisions, "eol_preds": eol_preds,
            "actual_diffs": actual_diffs,
            "difficulty_breakdown": difficulty_breakdown,
            # New metrics for calibration and robustness
            "conf_std": conf_std, "conf_range": conf_range,
            "l_score_kendall": l_score_kendall,
            "empirical_threshold": empirical_threshold,
        }

    model_metrics = {}
    for model in all_models:
        mdf = results_df[results_df["model"] == model]
        model_metrics[model] = compute_model_metrics(mdf)

    # ============================================================
    #  SECTION 10: RESULTS TABLE (PER MODEL)
    # ============================================================

    for model in all_models:
        m = model_metrics[model]
        ctrl = m["ctrl"]
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}")
        print(f"  Items: {m['n']} | Accuracy: {m['accuracy']:.1%}")
        print(f"\n  --- MONITORING (M-Score: {m['m_score']:.3f}) ---")
        print(f"  ECE:    {m['ece']:.4f}")
        print(f"  Brier:  {m['brier']:.4f}")
        auroc_s = f"{m['auroc']:.4f}" if not np.isnan(m['auroc']) else "N/A"
        print(f"  AUROC:  {auroc_s}")
        eol_s = f"{m['eol_corr']:.4f}" if not np.isnan(m['eol_corr']) else "N/A"
        print(f"  EOL rho:{eol_s}")
        print(f"\n  --- CONTROL (C-Score: {m['c_score']:.3f}) ---")
        print(f"  Game:   {ctrl['total_score']} / {ctrl['optimal_score']}")
        print(f"  Eff:    {ctrl['score_efficiency']:.1%}")
        print(f"  Sub/Abs:{ctrl['submit_count']} / {ctrl['abstain_count']}")
        print(f"\n  --- DECISION QUALITY (L-Score: {m['l_score']:.3f}) ---")
        print(f"  Ptbiserial: {m['l_score']:.4f}  (point-biserial correlation)")
        kendall_s = f"{m['l_score_kendall']:.4f}" if not np.isnan(m.get('l_score_kendall', float('nan'))) else "N/A  "
        print(f"  Kendall τ:  {kendall_s}  (robust to response bias)")
        print(f"\n  --- CONFIDENCE CALIBRATION VARIANCE ---")
        print(f"  σ(conf):    {m.get('conf_std', float('nan')):.4f}  (degenerate if ≈0)")
        print(f"  Range:      {m.get('conf_range', float('nan')):.4f}  (max - min)")
        thresh_s = f"{m.get('empirical_threshold', float('nan')):.2f}" if not np.isnan(m.get('empirical_threshold', float('nan'))) else "Never"
        print(f"  Emp. Thresh:{thresh_s:>6s}  (rational = 0.40)")
        print(f"\n  --- COMPOSITE ---")
        print(f"  M={m['m_score']:.3f} | C={m['c_score']:.3f} | L={m['l_score']:.3f} | Overall={m['overall']:.3f}")

        # Per-difficulty breakdown (shows WHERE calibration breaks)
        if m.get("difficulty_breakdown"):
            print(f"\n  --- DIFFICULTY BREAKDOWN (Hard-Easy Effect Analysis) ---")
            print(f"  {'Diff':<5s} {'Acc':>6s} {'Conf':>6s} {'ECE':>7s} {'Gap':>7s} {'Abstain%':>8s} {'N':>4s}")
            print(f"  {'-'*50}")
            for level in range(1, 7):
                key = f"difficulty_{level}"
                if key in m["difficulty_breakdown"]:
                    db = m["difficulty_breakdown"][key]
                    ece_str = f"{db['ece']:.4f}" if not np.isnan(db['ece']) else "N/A  "
                    gap_str = f"{db['calibration_gap']:+.4f}"
                    abstain_pct = f"{db['abstain_rate']:>6.1%}"
                    print(f"  {level:<5d} {db['accuracy']:>6.1%} {db['mean_conf']:>6.2f} "
                          f"{ece_str:>7s} {gap_str:>7s} {abstain_pct:>8s} {db['n']:>4d}")
                else:
                    print(f"  {level:<5d} {'—':>6s} {'—':>6s} {'—':>7s} {'—':>7s} {'—':>8s} {'<3':>4s}")

    # ============================================================
    #  SECTION 11: CROSS-MODEL COMPARISON TABLE
    # ============================================================

    print(f"\n{'='*60}")
    print("  CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    header = f"  {'Model':<20s} {'Acc':>5s} {'ECE':>6s} {'σ(C)':>6s} {'M':>5s} {'C':>5s} {'L-PB':>5s} {'L-τ':>5s} {'All':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in sorted(model_metrics, key=lambda m: -model_metrics[m]["overall"]):
        mm = model_metrics[model]
        conf_std_s = f"{mm.get('conf_std', float('nan')):.3f}"
        l_kendall_s = f"{mm.get('l_score_kendall', float('nan')):.3f}" if not np.isnan(mm.get('l_score_kendall', float('nan'))) else "—    "
        print(f"  {model:<20s} {mm['accuracy']:>5.1%} {mm['ece']:>6.3f} "
              f"{conf_std_s:>6s} {mm['m_score']:>5.3f} {mm['c_score']:>5.3f} "
              f"{mm['l_score']:>5.3f} {l_kendall_s:>5s} {mm['overall']:>5.3f}")

    # ============================================================
    #  SECTION 11B: SAVE METRICS TO CSV
    # ============================================================

    # -- Summary CSV: one row per model with all key metrics --
    summary_rows = []
    for model in sorted(model_metrics, key=lambda m: -model_metrics[m]["overall"]):
        mm = model_metrics[model]
        ctrl = mm["ctrl"]
        summary_rows.append({
            "model": model,
            "n_items": mm["n"],
            "accuracy": round(mm["accuracy"], 4),
            "ece": round(mm["ece"], 4),
            "brier": round(mm["brier"], 4),
            "auroc": round(mm["auroc"], 4) if not np.isnan(mm["auroc"]) else None,
            "eol_corr": round(mm["eol_corr"], 4) if not np.isnan(mm["eol_corr"]) else None,
            "m_score": round(mm["m_score"], 4),
            "c_score": round(mm["c_score"], 4),
            "l_score": round(mm["l_score"], 4),
            "l_score_kendall": round(mm["l_score_kendall"], 4) if not np.isnan(mm.get("l_score_kendall", float("nan"))) else None,
            "overall": round(mm["overall"], 4),
            "conf_std": round(mm.get("conf_std", 0), 4),
            "conf_range": round(mm.get("conf_range", 0), 4),
            "empirical_threshold": round(mm["empirical_threshold"], 4) if not np.isnan(mm.get("empirical_threshold", float("nan"))) else None,
            "game_score": ctrl["total_score"],
            "optimal_score": ctrl["optimal_score"],
            "score_efficiency": round(ctrl["score_efficiency"], 4) if not np.isnan(ctrl["score_efficiency"]) else None,
            "submit_count": ctrl["submit_count"],
            "abstain_count": ctrl["abstain_count"],
            "abstain_precision": round(ctrl["abstain_precision"], 4) if not np.isnan(ctrl["abstain_precision"]) else None,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("gauge_metrics_summary.csv", index=False)
    print(f"\n  Saved: gauge_metrics_summary.csv ({len(summary_rows)} models)")

    # -- Per-difficulty CSV: one row per (model, difficulty_level) --
    diff_rows = []
    for model in all_models:
        mm = model_metrics[model]
        for level in range(1, 7):
            key = f"difficulty_{level}"
            if key in mm.get("difficulty_breakdown", {}):
                db = mm["difficulty_breakdown"][key]
                diff_rows.append({
                    "model": model,
                    "difficulty": level,
                    "n": db["n"],
                    "accuracy": round(db["accuracy"], 4),
                    "mean_conf": round(db["mean_conf"], 4),
                    "ece": round(db["ece"], 4) if not np.isnan(db["ece"]) else None,
                    "calibration_gap": round(db["calibration_gap"], 4),
                    "abstain_rate": round(db["abstain_rate"], 4),
                })
    if diff_rows:
        diff_df = pd.DataFrame(diff_rows)
        diff_df.to_csv("gauge_difficulty_breakdown.csv", index=False)
        print(f"  Saved: gauge_difficulty_breakdown.csv ({len(diff_rows)} rows)")

    # -- Raw item-level CSV: every parsed result --
    results_df.to_csv("gauge_raw_results.csv", index=False)
    print(f"  Saved: gauge_raw_results.csv ({len(results_df)} items)")

    # ============================================================
    #  SECTION 12: VISUALIZATIONS
    # ============================================================

    try:
        import subprocess, sys
        try:
            import matplotlib
        except ImportError:
            print("  Installing matplotlib...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
            import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_models = len(all_models)

        # --- Fig 1: Reliability diagrams (one per model) ---
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        if n_models == 1:
            axes1 = np.array([[axes1]])
        elif nrows == 1:
            axes1 = axes1[np.newaxis, :]
        fig1.suptitle("GAUGE — Reliability Diagrams per Model", fontsize=14, fontweight="bold")

        for idx, model in enumerate(all_models):
            r, c = divmod(idx, ncols)
            ax = axes1[r, c]
            mm = model_metrics[model]
            conf_arr = np.array(mm["confidences"])
            corr_arr = np.array(mm["correctness"], dtype=float)
            n_bins = 15
            sorted_idx = np.argsort(conf_arr)
            bsz = max(1, len(sorted_idx) // n_bins)
            bc, ba = [], []
            for i in range(n_bins):
                b = sorted_idx[i*bsz:(i+1)*bsz] if i < n_bins-1 else sorted_idx[i*bsz:]
                if len(b) == 0: continue
                bc.append(conf_arr[b].mean())
                ba.append(corr_arr[b].mean())
            ax.bar(bc, ba, width=0.08, alpha=0.7, color="#56B4E9", edgecolor="white")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_title(f"{model}\nECE={mm['ece']:.3f}", fontsize=9, fontweight="bold")
            if r == nrows - 1: ax.set_xlabel("Confidence")
            if c == 0: ax.set_ylabel("Accuracy")
        for idx in range(n_models, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes1[r, c].set_visible(False)
        fig1.tight_layout()
        fig1.savefig("reliability_diagrams.png", dpi=150, bbox_inches="tight")
        plt.show()

        # --- Fig 2: Money chart (M/C/L grouped bar) ---
        # Use short model names for readability
        short_names = [m.split("/")[-1].split("@")[0] for m in all_models]
        fig2, ax2 = plt.subplots(figsize=(max(10, n_models * 1.8), 5))
        x = np.arange(n_models)
        w = 0.22
        m_vals = [model_metrics[m]["m_score"] for m in all_models]
        c_vals = [model_metrics[m]["c_score"] for m in all_models]
        l_vals = [model_metrics[m]["l_score"] for m in all_models]
        o_vals = [model_metrics[m]["overall"] for m in all_models]
        ax2.bar(x - w, m_vals, w, label="M-Score", color="#0072B2", alpha=0.85)
        ax2.bar(x, c_vals, w, label="C-Score", color="#D55E00", alpha=0.85)
        ax2.bar(x + w, l_vals, w, label="L-Score", color="#009E73", alpha=0.85)
        ax2.scatter(x, o_vals, marker="D", color="black", s=40, zorder=5, label="Overall")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, fontsize=8, rotation=30, ha="right")
        ax2.set_ylabel("Score (0-1)")
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=8)
        ax2.set_title("GAUGE Composite Scores", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        fig2.savefig("gauge_scores.png", dpi=150, bbox_inches="tight")
        plt.show()

        # --- Fig 3: Difficulty-calibration (two-panel: accuracy + confidence) ---
        colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9",
                  "#CC79A7", "#882255", "#332288", "#44AA99"]
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle("Difficulty-Calibration by Model", fontsize=13, fontweight="bold")
        for idx, model in enumerate(all_models):
            mdf = results_df[results_df["model"] == model]
            diffs = sorted(mdf["actual_diff"].unique())
            accs = [mdf[mdf["actual_diff"] == d]["correct"].mean() for d in diffs]
            confs = [mdf[mdf["actual_diff"] == d]["confidence"].mean() / 100 for d in diffs]
            col = colors[idx % len(colors)]
            short = model.split("/")[-1].split("@")[0]
            ax3a.plot(diffs, accs, "s-", color=col, alpha=0.8, linewidth=1.5,
                      label=short, markersize=5)
            ax3b.plot(diffs, confs, "o-", color=col, alpha=0.8, linewidth=1.5,
                      label=short, markersize=5)
        ax3a.set_xlabel("Difficulty Level"); ax3a.set_ylabel("Accuracy")
        ax3a.set_title("Accuracy by Difficulty"); ax3a.set_ylim(0, 1.05)
        ax3a.legend(fontsize=6, loc="lower left")
        ax3b.set_xlabel("Difficulty Level"); ax3b.set_ylabel("Mean Confidence")
        ax3b.set_title("Confidence by Difficulty"); ax3b.set_ylim(0, 1.05)
        ax3b.legend(fontsize=6, loc="lower left")
        fig3.tight_layout()
        fig3.savefig("difficulty_calibration.png", dpi=150, bbox_inches="tight")
        plt.show()

        # --- Fig 4: Abstention Decision Boundary ---
        # Shows: X=sorted confidence, Y=confidence, colored by outcome+decision
        # GREEN=submit+correct, RED=submit+wrong, BLUE=abstain+correct (overcautious),
        # ORANGE=abstain+wrong (good caution). Rational threshold at 40%.
        # Visual story: Claude shows blue/orange clusters at low conf (strategic abstention),
        # Gemini is all green/red (always submit, no abstention region)
        ncols_fig4 = min(3, n_models)
        nrows_fig4 = (n_models + ncols_fig4 - 1) // ncols_fig4
        fig4, axes4 = plt.subplots(nrows_fig4, ncols_fig4, figsize=(ncols_fig4 * 5, nrows_fig4 * 4))
        if n_models == 1:
            axes4 = np.array([[axes4]])
        elif nrows_fig4 == 1:
            axes4 = axes4[np.newaxis, :]
        fig4.suptitle("Abstention Decision Boundary: Strategic vs Always-Submit Behavior",
                      fontsize=14, fontweight="bold")

        for idx, model in enumerate(all_models):
            r, c = divmod(idx, ncols_fig4)
            ax = axes4[r, c]
            mm = model_metrics[model]
            mdf = results_df[results_df["model"] == model]

            # Get model data
            confidences = (mdf["confidence"] / 100.0).values  # 0-1 scale
            correctness = mdf["correct"].astype(int).values
            decisions = (mdf["decision"] == "submit").astype(int).values  # 1=submit, 0=abstain

            # Sort by confidence (ascending) for left-to-right visualization
            sort_idx = np.argsort(confidences)
            x_pos = np.arange(len(sort_idx))
            conf_sorted = confidences[sort_idx]
            corr_sorted = correctness[sort_idx]
            dec_sorted = decisions[sort_idx]

            # Color code by outcome + decision:
            # GREEN (0): submit + correct
            # RED (1): submit + wrong
            # BLUE (2): abstain + correct (overcautious)
            # ORANGE (3): abstain + wrong (good caution)
            colors_array = np.zeros(len(sort_idx), dtype=int)
            for i in range(len(sort_idx)):
                if dec_sorted[i] == 1:  # submit
                    colors_array[i] = 0 if corr_sorted[i] == 1 else 1  # green if correct, red if wrong
                else:  # abstain
                    colors_array[i] = 2 if corr_sorted[i] == 1 else 3  # blue if correct, orange if wrong

            color_map = {0: "#009E73", 1: "#D55E00", 2: "#56B4E9", 3: "#F8AD09"}  # green, red, blue, orange
            labels_map = {0: "Submit+Correct", 1: "Submit+Wrong", 2: "Abstain+Correct (overcautious)",
                         3: "Abstain+Wrong (good)"}

            # Scatter plot with colors
            for color_id in [0, 1, 2, 3]:
                mask = colors_array == color_id
                if np.any(mask):
                    ax.scatter(x_pos[mask], conf_sorted[mask] * 100, color=color_map[color_id],
                              s=30, alpha=0.7, label=labels_map[color_id], edgecolors="black", linewidth=0.5)

            # Rational threshold line (40% confidence: where EV_submit = EV_abstain given +3/-1/+1/0)
            ax.axhline(y=40, color="black", linestyle="--", linewidth=2, alpha=0.5, label="Rational Threshold (40%)")

            # Labels and formatting
            ax.set_xlabel("Item Index (sorted by confidence ascending)")
            ax.set_ylabel("Confidence (%)")
            ax.set_ylim(0, 105)
            l_score_val = mm.get("l_score", 0)
            ax.set_title(f"{model}\nL-Score={l_score_val:.3f}", fontsize=10, fontweight="bold")
            if idx == 0 or c == 0:
                ax.legend(fontsize=7, loc="upper left")

        # Hide unused subplots
        for idx in range(n_models, nrows_fig4 * ncols_fig4):
            r, c = divmod(idx, ncols_fig4)
            axes4[r, c].set_visible(False)

        fig4.tight_layout()
        fig4.savefig("abstention_boundary.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("\n  Figures saved: reliability_diagrams.png, gauge_scores.png, difficulty_calibration.png, abstention_boundary.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    #  SECTION 13: FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("  GAUGE — FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<25s} {'Acc':>5s} {'ECE':>6s} {'M':>5s} {'C':>5s} {'L':>5s} {'Overall':>7s}")
    print("  " + "-" * 60)
    for model in sorted(model_metrics, key=lambda m: -model_metrics[m]["overall"]):
        mm = model_metrics[model]
        print(f"  {model:<25s} {mm['accuracy']:>5.1%} {mm['ece']:>6.3f} "
              f"{mm['m_score']:>5.3f} {mm['c_score']:>5.3f} "
              f"{mm['l_score']:>5.3f} {mm['overall']:>7.3f}")
    print("  " + "-" * 60)
    best = max(model_metrics, key=lambda m: model_metrics[m]["overall"])
    print(f"\n  Best overall: {best} ({model_metrics[best]['overall']:.3f})")
    print("=" * 60)


# ============================================================
#  SECTION 13: CHOOSE PRIMARY TASK FOR LEADERBOARD
# ============================================================

# %choose gauge_staircase
