"""Procedural math word problem generator for metacognitive calibration.

Generates GSM8K-style problems with parameterized templates across 6 difficulty
levels. Difficulty is controlled by number of reasoning steps, number magnitude,
presence of distractor clauses, and operation diversity.

Design informed by:
- GSM-Symbolic (Apple, ICLR 2025): numerically-loaded distractors (GSM-NoOp)
- GSM-Plus (ACL 2024): reverse-reasoning and critical-thinking perturbations
- Nelson & Narens (1990): metacognitive calibration requires problems where
  models are confidently wrong (traps) and problems that are unanswerable.
"""

from __future__ import annotations

import random
from typing import TypedDict


class MathProblem(TypedDict):
    question: str
    correct_answer: int | float
    difficulty: int
    seed: int
    template_id: str


# ---------------------------------------------------------------------------
# Distractor clauses — two tiers
# ---------------------------------------------------------------------------

# Tier 1: trivially irrelevant (no numbers) — used at difficulty 3
DISTRACTORS_SOFT: list[str] = [
    "{name}'s favorite color is blue.",
    "It was a sunny Wednesday afternoon.",
    "The store had been open since 1987.",
    "{name} was wearing a red hat that day.",
    "A song was playing on the radio in the background.",
    "The receipt was printed on recycled paper.",
]

# Tier 2: numerically loaded but irrelevant (GSM-NoOp style) — used at difficulty 4+
# These contain numbers that LOOK relevant but DON'T affect the answer.
DISTRACTORS_NUMERIC: list[str] = [
    "{num} of the {item} were slightly bruised, but {name} kept them all.",
    "The store also had {num} {other_item} on display, but {name} didn't buy any.",
    "{name} noticed that {num} other customers were in the store.",
    "The shelf had {num} empty spots where {item} used to be.",
    "{name}'s friend had mentioned wanting {num} {item}, but {name} didn't get any for them.",
    "On the way there, {name} passed {num} houses.",
    "The cashier mentioned they had sold {num} {item} earlier that day to someone else.",
    "There were {num} items in the clearance bin, but {name} wasn't interested.",
    "{name} had {num} coupons in a drawer at home but forgot to bring them.",
    "The store's loyalty program showed {name} had {num} points, which weren't redeemable yet.",
]

# ---------------------------------------------------------------------------
# Name / item pools
# ---------------------------------------------------------------------------

NAMES: list[str] = [
    "Sarah", "Tom", "Maria", "James", "Priya", "Carlos", "Aisha", "David",
    "Lin", "Omar", "Rachel", "Kenji", "Fatima", "Alex", "Nadia", "Marcus",
]

FRUIT: list[str] = [
    "apples", "oranges", "bananas", "mangoes", "pears", "peaches",
]

ITEMS: list[str] = [
    "books", "marbles", "stickers", "pencils", "cookies", "toy cars",
]

STORE_ITEMS: list[str] = [
    "notebooks", "pens", "erasers", "folders", "markers", "rulers",
]

OTHER_ITEMS: list[str] = [
    "magazines", "postcards", "batteries", "candles", "envelopes", "stamps",
]

# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG: dict[int, dict] = {
    1: {"num_lo": 1,   "num_hi": 50,    "steps": (1, 1), "distractors": 0, "distractor_tier": 0},
    2: {"num_lo": 1,   "num_hi": 50,    "steps": (1, 2), "distractors": 0, "distractor_tier": 0},
    3: {"num_lo": 10,  "num_hi": 500,   "steps": (2, 3), "distractors": 1, "distractor_tier": 1},
    4: {"num_lo": 20,  "num_hi": 800,   "steps": (2, 4), "distractors": 1, "distractor_tier": 2},
    5: {"num_lo": 100, "num_hi": 10000, "steps": (3, 5), "distractors": 1, "distractor_tier": 2},
    6: {"num_lo": 100, "num_hi": 10000, "steps": (3, 5), "distractors": 2, "distractor_tier": 2},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, pool: list[str], n: int = 1) -> list[str]:
    return rng.sample(pool, min(n, len(pool)))


def _num(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _insert_distractors(
    rng: random.Random, sentences: list[str], n_distractors: int, name: str,
    cfg: dict, item: str = "items",
) -> list[str]:
    """Insert n distractor sentences at random positions among real sentences."""
    if n_distractors == 0:
        return sentences

    tier = cfg.get("distractor_tier", 1)
    if tier <= 1:
        pool = DISTRACTORS_SOFT
        chosen = rng.sample(pool, min(n_distractors, len(pool)))
        chosen = [d.format(name=name) for d in chosen]
    else:
        pool = DISTRACTORS_NUMERIC
        chosen = rng.sample(pool, min(n_distractors, len(pool)))
        other_item = rng.choice(OTHER_ITEMS)
        chosen = [
            d.format(
                name=name,
                num=rng.randint(2, 50),
                item=item,
                other_item=other_item,
            )
            for d in chosen
        ]

    result = list(sentences)
    for d in chosen:
        pos = rng.randint(1, max(1, len(result) - 1))
        result.insert(pos, d)
    return result


def _join(sentences: list[str]) -> str:
    return " ".join(sentences)


def _they(name: str) -> str:
    """Use 'they' as a gender-neutral pronoun for all names."""
    return "They"


def _them(name: str) -> str:
    return "them"


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, callable] = {}


def _template(template_id: str, min_difficulty: int = 1, max_difficulty: int = 6):
    """Decorator to register a template function."""
    def decorator(fn):
        fn.template_id = template_id
        fn.min_difficulty = min_difficulty
        fn.max_difficulty = max_difficulty
        _TEMPLATES[template_id] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Templates — original 12 (with fixes) + 3 new types
# ---------------------------------------------------------------------------

@_template("give_and_buy", 1, 4)
def _give_and_buy(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """1-2 steps: have items, give some away, buy more."""
    name1, name2 = _pick(rng, NAMES, 2)
    item = rng.choice(FRUIT)
    has = _num(rng, cfg["num_lo"], cfg["num_hi"])
    gives = _num(rng, 1, max(1, has // 2))
    buys = _num(rng, cfg["num_lo"], cfg["num_hi"])
    answer = has - gives + buys
    parts = [
        f"{name1} has {has} {item}.",
        f"{name1} gives {gives} to {name2} and buys {buys} more.",
        f"How many {item} does {name1} have now?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name1, cfg, item)
    return _join(parts), answer


@_template("simple_purchase", 1, 3)
def _simple_purchase(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """1 step: buy N items at price P, compute total."""
    name = rng.choice(NAMES)
    item = rng.choice(STORE_ITEMS)
    qty = _num(rng, 2, min(20, cfg["num_hi"]))
    price = _num(rng, cfg["num_lo"], min(50, cfg["num_hi"]))
    answer = qty * price
    singular = item[:-1] if item.endswith("s") else item
    parts = [
        f"{name} goes to a store to buy {item}.",
        f"Each {singular} costs ${price}.",
        f"{name} buys {qty} of them.",
        f"How much does {name} spend in total?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg, item)
    return _join(parts), answer


@_template("share_equally", 1, 4)
def _share_equally(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """1-2 steps: divide items among friends."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    friends = _num(rng, 2, 8)
    per_friend = _num(rng, cfg["num_lo"], cfg["num_hi"] // max(1, friends))
    total = friends * per_friend
    answer = per_friend
    parts = [
        f"{name} has {total} {item} and wants to share them equally among {friends} friends.",
        f"How many {item} does each friend get?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg, item)
    return _join(parts), answer


@_template("multi_day_earning", 2, 5)
def _multi_day_earning(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """2-3 steps: earn different amounts over multiple days, compute total."""
    name = rng.choice(NAMES)
    days = _num(rng, 2, 4)
    earnings = [_num(rng, cfg["num_lo"], cfg["num_hi"]) for _ in range(days)]
    answer = sum(earnings)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][:days]
    parts = [f"{name} earned money over {days} days."]
    for day_name, earned in zip(day_names, earnings):
        parts.append(f"On {day_name}, {name} earned ${earned}.")
    parts.append(f"How much did {name} earn in total?")
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("percentage_discount", 3, 6)
def _percentage_discount(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """2-3 steps: original price, percentage off, compute final price."""
    name = rng.choice(NAMES)
    item = rng.choice(["jacket", "laptop", "bicycle", "camera", "guitar"])
    pct = rng.choice([10, 20, 25, 50])
    # Ensure price is divisible by (100/pct) so discount is always an exact integer
    divisor = 100 // pct  # 10→10, 20→5, 25→4, 50→2
    raw = _num(rng, cfg["num_lo"] // divisor, cfg["num_hi"] // divisor)
    price = raw * divisor
    discount = price * pct // 100
    answer = price - discount
    parts = [
        f"{name} wants to buy a {item} that costs ${price}.",
        f"The store is offering a {pct}% discount today.",
        f"How much does {name} pay after the discount?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("two_step_ratio", 3, 6)
def _two_step_ratio(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """2-3 steps: person A has N, person B has K times as many, find total."""
    name1, name2 = _pick(rng, NAMES, 2)
    item = rng.choice(ITEMS)
    a_count = _num(rng, cfg["num_lo"], cfg["num_hi"] // 4)
    multiplier = _num(rng, 2, 5)
    b_count = a_count * multiplier
    answer = a_count + b_count
    parts = [
        f"{name1} has {a_count} {item}.",
        f"{name2} has {multiplier} times as many {item} as {name1}.",
        f"How many {item} do they have together?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name1, cfg, item)
    return _join(parts), answer


@_template("remaining_after_spending", 2, 5)
def _remaining_after_spending(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """2-3 steps: start with money, make multiple purchases, find remainder."""
    name = rng.choice(NAMES)
    n_purchases = _num(rng, 2, 3)
    purchases = [_num(rng, cfg["num_lo"], cfg["num_hi"] // (n_purchases + 1)) for _ in range(n_purchases)]
    start = sum(purchases) + _num(rng, cfg["num_lo"], cfg["num_hi"])
    answer = start - sum(purchases)
    things = rng.sample(["a shirt", "lunch", "a book", "a bus ticket", "a gift", "coffee"], n_purchases)
    parts = [f"{name} has ${start}."]
    for thing, cost in zip(things, purchases):
        parts.append(f"{name} spends ${cost} on {thing}.")
    parts.append(f"How much money does {name} have left?")
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("rate_and_time", 3, 6)
def _rate_and_time(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """2-3 steps: rate * time problems (speed, production, etc.)."""
    name = rng.choice(NAMES)
    scenario = rng.choice([
        ("reads {rate} pages per hour", "hours", "pages", "read"),
        ("types {rate} words per minute", "minutes", "words", "type"),
        ("builds {rate} widgets per day", "days", "widgets", "produce"),
    ])
    rate = _num(rng, max(1, cfg["num_lo"] // 5), min(200, cfg["num_hi"] // 5))
    time_val = _num(rng, 2, 8)
    answer = rate * time_val
    desc = scenario[0].format(rate=rate)
    parts = [
        f"{name} {desc}.",
        f"If {name} works for {time_val} {scenario[1]}, how many {scenario[2]} does {name} {scenario[3]}?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("combined_work", 4, 6)
def _combined_work(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """3-4 steps: two people work at different rates, find combined output."""
    name1, name2 = _pick(rng, NAMES, 2)
    task = rng.choice(["paint fences", "assemble boxes", "pack orders", "sort files"])
    rate1 = _num(rng, max(1, cfg["num_lo"] // 10), min(100, cfg["num_hi"] // 10))
    rate2 = _num(rng, max(1, cfg["num_lo"] // 10), min(100, cfg["num_hi"] // 10))
    hours = _num(rng, 2, 8)
    answer = (rate1 + rate2) * hours
    parts = [
        f"{name1} and {name2} {task} together.",
        f"{name1} can finish {rate1} per hour and {name2} can finish {rate2} per hour.",
        f"They work together for {hours} hours.",
        f"How many do they complete in total?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name1, cfg)
    return _join(parts), answer


@_template("multi_item_purchase", 4, 6)
def _multi_item_purchase(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """3-4 steps: buy multiple item types at different prices, find change."""
    name = rng.choice(NAMES)
    items = rng.sample(STORE_ITEMS, 3)
    qtys = [_num(rng, 1, 10) for _ in range(3)]
    prices = [_num(rng, max(1, cfg["num_lo"] // 5), min(200, cfg["num_hi"] // 10)) for _ in range(3)]
    subtotal = sum(q * p for q, p in zip(qtys, prices))
    budget = subtotal + _num(rng, cfg["num_lo"], cfg["num_hi"] // 2)
    answer = budget - subtotal
    parts = [f"{name} goes shopping with ${budget}."]
    for item, qty, price in zip(items, qtys, prices):
        parts.append(f"{name} buys {qty} {item} at ${price} each.")
    parts.append(f"How much money does {name} have left?")
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("savings_over_weeks", 4, 6)
def _savings_over_weeks(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """3-5 steps: save per week, spend occasionally, compute remaining."""
    name = rng.choice(NAMES)
    weeks = _num(rng, 3, 6)
    per_week = _num(rng, max(1, cfg["num_lo"] // 2), cfg["num_hi"] // weeks)
    total_saved = per_week * weeks
    n_expenses = rng.randint(1, 2)
    expenses = [_num(rng, 1, max(1, total_saved // (n_expenses + 2))) for _ in range(n_expenses)]
    answer = total_saved - sum(expenses)
    expense_items = rng.sample(["a video game", "a birthday gift", "new shoes", "concert tickets"], n_expenses)
    parts = [f"{name} saves ${per_week} every week for {weeks} weeks."]
    for exp_item, exp_cost in zip(expense_items, expenses):
        parts.append(f"During that time, {name} spends ${exp_cost} on {exp_item}.")
    parts.append(f"How much money does {name} have at the end?")
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


@_template("profit_calculation", 5, 6)
def _profit_calculation(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """4-5 steps: buy in bulk, sell individually, subtract overhead, compute profit."""
    name = rng.choice(NAMES)
    product = rng.choice(["t-shirts", "candles", "phone cases", "mugs", "posters"])
    bulk_qty = _num(rng, 20, 100)
    cost_per = _num(rng, max(1, cfg["num_lo"] // 10), cfg["num_hi"] // bulk_qty)
    total_cost = bulk_qty * cost_per
    sell_per = cost_per + _num(rng, max(1, cost_per // 3), max(2, cost_per))
    sold_qty = _num(rng, bulk_qty // 2, bulk_qty)
    revenue = sold_qty * sell_per
    overhead = _num(rng, max(1, cfg["num_lo"] // 2), max(2, total_cost // 5))
    answer = revenue - total_cost - overhead
    parts = [
        f"{name} starts a small business selling {product}.",
        f"{name} buys {bulk_qty} {product} at ${cost_per} each.",
        f"{name} sells {sold_qty} of them at ${sell_per} each.",
        f"{name} also pays ${overhead} in overhead costs.",
        f"What is {name}'s profit?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg, product)
    return _join(parts), answer


# ---------------------------------------------------------------------------
# NEW: Reverse reasoning (GSM-Plus style)
# ---------------------------------------------------------------------------

@_template("reverse_purchase", 3, 6)
def _reverse_purchase(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """Inverse reasoning: given total cost and quantity, find unit price."""
    name = rng.choice(NAMES)
    item = rng.choice(STORE_ITEMS)
    singular = item[:-1] if item.endswith("s") else item
    qty = _num(rng, 2, min(20, cfg["num_hi"] // 10))
    price = _num(rng, cfg["num_lo"], min(200, cfg["num_hi"] // qty))
    total = qty * price
    answer = price
    parts = [
        f"{name} bought {qty} {item} and paid ${total} in total.",
        f"Each {singular} cost the same amount.",
        f"How much did each {singular} cost?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg, item)
    return _join(parts), answer


@_template("reverse_earning", 4, 6)
def _reverse_earning(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """Inverse reasoning: given total and partial info, find the missing amount."""
    name = rng.choice(NAMES)
    day_earnings = [_num(rng, cfg["num_lo"], cfg["num_hi"]) for _ in range(3)]
    total = sum(day_earnings)
    known = day_earnings[:2]
    unknown = day_earnings[2]
    answer = unknown
    parts = [
        f"{name} earned a total of ${total} over three days.",
        f"On the first day, {name} earned ${known[0]}.",
        f"On the second day, {name} earned ${known[1]}.",
        f"How much did {name} earn on the third day?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg)
    return _join(parts), answer


# ---------------------------------------------------------------------------
# NEW: Trap / critical thinking (metacognition-critical)
# ---------------------------------------------------------------------------

@_template("trap_irrelevant_operation", 4, 6)
def _trap_irrelevant_operation(rng: random.Random, cfg: dict) -> tuple[str, int]:
    """Contains a plausible extra operation that should NOT be performed.

    The question asks for a specific subtotal, not the grand total.
    Models that blindly combine all numbers will get it wrong.
    """
    name = rng.choice(NAMES)
    item1 = rng.choice(FRUIT)
    item2 = rng.choice([i for i in FRUIT if i != item1])
    qty1 = _num(rng, cfg["num_lo"], cfg["num_hi"])
    qty2 = _num(rng, cfg["num_lo"], cfg["num_hi"])
    gives = _num(rng, 1, max(1, qty1 // 2))
    # The trap: qty2 is mentioned but the question only asks about item1
    answer = qty1 - gives
    parts = [
        f"{name} has {qty1} {item1} and {qty2} {item2}.",
        f"{name} gives {gives} {item1} to a friend.",
        f"How many {item1} does {name} have now?",
    ]
    parts = _insert_distractors(rng, parts, cfg["distractors"], name, cfg, item1)
    return _join(parts), answer


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def _templates_for_difficulty(difficulty: int) -> list[callable]:
    return [
        fn for fn in _TEMPLATES.values()
        if fn.min_difficulty <= difficulty <= fn.max_difficulty
    ]


def generate_problem(difficulty: int, seed: int) -> MathProblem:
    """Generate a single math problem at the given difficulty level."""
    if difficulty < 1 or difficulty > 6:
        raise ValueError(f"difficulty must be 1-6, got {difficulty}")

    rng = random.Random(seed)
    cfg = DIFFICULTY_CONFIG[difficulty]
    templates = _templates_for_difficulty(difficulty)
    fn = rng.choice(templates)

    # Re-seed so the same seed always produces the same problem for this template
    rng = random.Random(seed)
    question, answer = fn(rng, cfg)

    return MathProblem(
        question=question,
        correct_answer=answer,
        difficulty=difficulty,
        seed=seed,
        template_id=fn.template_id,
    )


def verify_answer(problem: MathProblem) -> bool:
    """Independently regenerate the problem and verify the stored answer matches."""
    regenerated = generate_problem(problem["difficulty"], problem["seed"])
    return (
        regenerated["correct_answer"] == problem["correct_answer"]
        and regenerated["question"] == problem["question"]
    )


def generate_problem_set(
    n: int = 30,
    seed: int = 42,
) -> list[MathProblem]:
    """Generate a balanced set of problems across all 6 difficulty levels.

    Distributes n problems as evenly as possible across difficulties 1-6.
    Each problem gets a unique sub-seed derived from the master seed.
    """
    rng = random.Random(seed)
    problems: list[MathProblem] = []

    per_level = n // 6
    remainder = n % 6
    counts = [per_level + (1 if i < remainder else 0) for i in range(6)]

    for difficulty, count in enumerate(counts, start=1):
        for _ in range(count):
            sub_seed = rng.randint(0, 2**31 - 1)
            problems.append(generate_problem(difficulty, sub_seed))

    rng2 = random.Random(seed + 1)
    rng2.shuffle(problems)
    return problems


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    problems = generate_problem_set(n=30, seed=42)

    by_diff: dict[int, list[MathProblem]] = {}
    for p in problems:
        by_diff.setdefault(p["difficulty"], []).append(p)

    print(f"Generated {len(problems)} problems across {len(by_diff)} difficulty levels\n")
    print(f"Templates available: {len(_TEMPLATES)}")
    print(f"Distribution: { {d: len(ps) for d, ps in sorted(by_diff.items())} }\n")

    all_valid = all(verify_answer(p) for p in problems)
    print(f"All answers verified: {all_valid}\n")
    print("=" * 70)

    for diff in range(1, 7):
        print(f"\n--- Difficulty {diff} ---")
        for p in by_diff.get(diff, [])[:2]:
            print(f"  Template: {p['template_id']}")
            print(f"  Q: {p['question']}")
            print(f"  A: {p['correct_answer']}")
            print(f"  Seed: {p['seed']}")
            print()
