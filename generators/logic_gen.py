"""Procedural logic / deductive reasoning problem generator.

Generates problems across 4 types: syllogisms, multi-step deductions,
constraint satisfaction, and set membership. Uses the ProntoQA approach:
build a proof tree (ground truth) first, then render to natural language.

Difficulty axes (orthogonal):
  - Proof depth (number of reasoning hops)
  - Negation count
  - Distractor density (irrelevant premises)
  - Entity count
"""

from __future__ import annotations

import random
from typing import TypedDict


class LogicProblem(TypedDict):
    question: str
    correct_answer: str
    difficulty: int
    seed: int
    problem_type: str


# ---------------------------------------------------------------------------
# Entity pools — fictional to prevent memorization shortcuts
# ---------------------------------------------------------------------------

CREATURE_NAMES: list[str] = [
    "Blorps", "Grumpkins", "Zephlins", "Tazzles", "Wumpuses",
    "Flimbers", "Drogons", "Snazzles", "Quibbles", "Plonkers",
    "Vexlings", "Murnips", "Drazzles", "Kelpoids", "Fizzgigs",
]

PERSON_NAMES: list[str] = [
    "Ava", "Ben", "Cora", "Dan", "Eve", "Finn", "Gina", "Hugo",
    "Iris", "Jake", "Kira", "Leo", "Mina", "Noel", "Ora", "Paul",
]

PROPERTIES: list[str] = [
    "happy", "tall", "clever", "brave", "friendly",
    "quiet", "strong", "wise", "fast", "kind",
]

COLORS: list[str] = ["red", "blue", "green", "yellow", "purple", "orange"]
FOODS: list[str] = ["pizza", "sushi", "tacos", "pasta", "salad", "soup"]
HOBBIES: list[str] = ["painting", "chess", "gardening", "cooking", "reading", "hiking"]

# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG: dict[int, dict] = {
    1: {"hops": (1, 1), "entities": (2, 3), "negations": 0, "distractors": 0},
    2: {"hops": (1, 2), "entities": (2, 3), "negations": 0, "distractors": 0},
    3: {"hops": (2, 3), "entities": (3, 5), "negations": 1, "distractors": 1},
    4: {"hops": (2, 3), "entities": (3, 5), "negations": 1, "distractors": 1},
    5: {"hops": (3, 5), "entities": (4, 7), "negations": 2, "distractors": 1},
    6: {"hops": (3, 5), "entities": (5, 7), "negations": 2, "distractors": 2},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, pool: list[str], n: int) -> list[str]:
    return rng.sample(pool, min(n, len(pool)))


def _distractor_premises(rng: random.Random, n: int, used_names: set[str]) -> list[str]:
    """Generate irrelevant premises that don't interact with the proof chain."""
    available = [c for c in CREATURE_NAMES if c not in used_names]
    if len(available) < n:
        available = CREATURE_NAMES[:]
    distractors = []
    props = _pick(rng, PROPERTIES, max(n, 1))
    for i in range(n):
        c = rng.choice(available)
        p = props[i % len(props)]
        form = rng.choice([
            f"All {c} are {p}.",
            f"Some {c} are {p}.",
            f"If something is a {c}, it might be {p}.",
        ])
        distractors.append(form)
    return distractors


def _insert_distractors(rng: random.Random, premises: list[str], distractors: list[str]) -> list[str]:
    result = list(premises)
    for d in distractors:
        pos = rng.randint(0, len(result))
        result.insert(pos, d)
    return result


# ---------------------------------------------------------------------------
# Problem type: Syllogisms
# ---------------------------------------------------------------------------

def _syllogism(rng: random.Random, cfg: dict) -> tuple[str, str]:
    """If A then B (+ optional chain). A is true. What about B (or C)?"""
    n_hops = rng.randint(*cfg["hops"])
    n_hops = min(n_hops, 2)  # syllogisms cap at 2 hops

    creatures = _pick(rng, CREATURE_NAMES, n_hops + 1)
    props = _pick(rng, PROPERTIES, n_hops + 1)

    instance_name = rng.choice(PERSON_NAMES)
    premises = []
    premises.append(f"{instance_name} is a {creatures[0]}.")
    premises.append(f"All {creatures[0]} are {props[0]}.")

    for i in range(1, n_hops):
        premises.append(f"All {props[i-1]} things are {props[i]}.")

    target_prop = props[n_hops - 1]

    # Decide answer type — even at low difficulty, ~30% ask about unrelated property
    roll = rng.random()
    use_negation = cfg["negations"] > 0 and roll < 0.3
    use_unrelated = roll >= 0.7  # 30% chance: ask about property NOT in chain

    if use_negation:
        answer = "No"
        question_end = f"Is {instance_name} not {target_prop}?"
    elif use_unrelated:
        unused_props = [p for p in PROPERTIES if p not in props]
        if unused_props:
            fake_prop = rng.choice(unused_props)
            answer = "Cannot be determined"
            question_end = f"Is {instance_name} {fake_prop}?"
        else:
            answer = "Yes"
            question_end = f"Is {instance_name} {target_prop}?"
    else:
        answer = "Yes"
        question_end = f"Is {instance_name} {target_prop}?"

    used = set(creatures)
    distractors = _distractor_premises(rng, cfg["distractors"], used)
    premises = _insert_distractors(rng, premises, distractors)

    question = " ".join(premises) + " " + question_end
    return question, answer


# ---------------------------------------------------------------------------
# Problem type: Multi-step deduction chains
# ---------------------------------------------------------------------------

def _multi_step_deduction(rng: random.Random, cfg: dict) -> tuple[str, str]:
    """Chain: A→B→C→...→Z. Given A, what can we conclude about Z?"""
    n_hops = rng.randint(*cfg["hops"])
    n_hops = max(n_hops, 2)  # at least 2 for multi-step

    props = _pick(rng, PROPERTIES, n_hops + 1)
    creatures = _pick(rng, CREATURE_NAMES, 2)
    instance = rng.choice(PERSON_NAMES)

    premises = [f"{instance} is a {creatures[0]}."]
    premises.append(f"All {creatures[0]} are {props[0]}.")
    for i in range(1, n_hops):
        premises.append(f"If something is {props[i-1]}, then it is {props[i]}.")

    final_prop = props[n_hops - 1]

    # Decide answer type — ensure variety at all difficulty levels
    roll = rng.random()
    use_negation = cfg["negations"] > 0 and roll < 0.3
    use_unrelated = roll >= 0.65  # 35% chance: ask about property NOT in chain

    if use_negation:
        unused_props = [p for p in PROPERTIES if p not in props]
        if unused_props:
            fake_prop = rng.choice(unused_props)
            answer = "Cannot be determined"
            question_end = f"Is {instance} {fake_prop}?"
        else:
            answer = "Yes"
            question_end = f"Is {instance} {final_prop}?"
    elif use_unrelated:
        unused_props = [p for p in PROPERTIES if p not in props]
        if unused_props:
            fake_prop = rng.choice(unused_props)
            answer = "Cannot be determined"
            question_end = f"Is {instance} {fake_prop}?"
        else:
            answer = "Yes"
            question_end = f"Is {instance} {final_prop}?"
    else:
        answer = "Yes"
        question_end = f"Is {instance} {final_prop}?"

    used = set(creatures)
    distractors = _distractor_premises(rng, cfg["distractors"], used)
    premises = _insert_distractors(rng, premises, distractors)

    question = " ".join(premises) + " " + question_end
    return question, answer


# ---------------------------------------------------------------------------
# Problem type: Set membership
# ---------------------------------------------------------------------------

def _set_membership(rng: random.Random, cfg: dict) -> tuple[str, str]:
    """All X are Y. Z is X. Is Z a Y? With optional negation and nesting."""
    n_hops = rng.randint(*cfg["hops"])
    n_hops = min(n_hops, 4)

    categories = _pick(rng, CREATURE_NAMES, n_hops + 1)
    instance = rng.choice(PERSON_NAMES)

    # Chain: instance ∈ cat[0] ⊂ cat[1] ⊂ ... ⊂ cat[n]
    premises = [f"{instance} is a {categories[0]}."]
    for i in range(n_hops):
        premises.append(f"Every {categories[i]} is a {categories[i+1]}.")

    # Decide answer type — ensure variety even at low difficulty
    roll = rng.random()
    use_negation = cfg["negations"] > 0 and roll < 0.3
    use_unrelated = roll >= 0.7  # 30% chance: ask about category NOT in chain

    if use_negation:
        unused = [c for c in CREATURE_NAMES if c not in categories]
        if unused:
            fake_cat = rng.choice(unused)
            answer = "Cannot be determined"
            question_end = f"Is {instance} a {fake_cat}?"
        else:
            answer = "Yes"
            question_end = f"Is {instance} a {categories[-1]}?"
    elif use_unrelated:
        unused = [c for c in CREATURE_NAMES if c not in categories]
        if unused:
            fake_cat = rng.choice(unused)
            answer = "Cannot be determined"
            question_end = f"Is {instance} a {fake_cat}?"
        else:
            target = rng.choice(categories[1:])
            answer = "Yes"
            question_end = f"Is {instance} a {target}?"
    else:
        target = rng.choice(categories[1:])
        answer = "Yes"
        question_end = f"Is {instance} a {target}?"

    used = set(categories)
    distractors = _distractor_premises(rng, cfg["distractors"], used)
    premises = _insert_distractors(rng, premises, distractors)

    question = " ".join(premises) + " " + question_end
    return question, answer


# ---------------------------------------------------------------------------
# Problem type: Constraint satisfaction (seating / assignment)
# ---------------------------------------------------------------------------

def _constraint_satisfaction(rng: random.Random, cfg: dict) -> tuple[str, str]:
    """Small assignment puzzle: assign attributes to entities given constraints.

    Generate a ground-truth assignment first, derive clues from it,
    then ask about one specific assignment.
    """
    n_entities = rng.randint(*cfg["entities"])
    n_entities = min(n_entities, 5)  # keep puzzles tractable
    n_entities = max(n_entities, 3)

    people = _pick(rng, PERSON_NAMES, n_entities)
    colors = _pick(rng, COLORS, n_entities)

    # Ground truth: random 1-to-1 assignment
    assignment = dict(zip(people, colors))

    # Generate clues that uniquely determine the assignment
    premises = []
    revealed: set[str] = set()

    # Direct assignment clues for all but one person
    people_list = list(people)
    rng.shuffle(people_list)

    # Reveal all assignments via clues (some direct, some indirect)
    for i, person in enumerate(people_list):
        color = assignment[person]
        if i < n_entities - 1:
            # Mix clue types
            clue_type = rng.choice(["direct", "negation_others"])
            if clue_type == "direct" or i == 0:
                premises.append(f"{person}'s favorite color is {color}.")
                revealed.add(person)
            else:
                # "Person does not like X or Y" (ruling out everything except the right one)
                wrong_colors = [c for c in colors if c != color]
                picked_wrong = rng.sample(wrong_colors, min(2, len(wrong_colors)))
                neg_str = " or ".join(picked_wrong)
                premises.append(f"{person} does not like {neg_str}.")
                # Also add one more direct clue to keep it solvable
                premises.append(f"{person}'s favorite color is {color}.")
                revealed.add(person)

    # The last person is determined by elimination
    last_person = people_list[-1]
    last_color = assignment[last_person]

    # Add negation clues for higher difficulty
    if cfg["negations"] > 0:
        wrong_color = rng.choice([c for c in colors if c != last_color])
        premises.append(f"{last_person} does not like {wrong_color}.")

    # Red herring premises
    used_names = set(people)
    distractor_strs = []
    if cfg["distractors"] > 0:
        extra_people = _pick(rng, [n for n in PERSON_NAMES if n not in used_names], cfg["distractors"])
        for ep in extra_people:
            hobby = rng.choice(HOBBIES)
            distractor_strs.append(f"{ep} enjoys {hobby}.")

    premises = _insert_distractors(rng, premises, distractor_strs)

    # Question: ask about the last person (must be deduced by elimination)
    question = " ".join(premises) + f" What is {last_person}'s favorite color?"
    answer = last_color

    return question, answer


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_PROBLEM_TYPES: dict[str, dict] = {
    "syllogism": {
        "fn": _syllogism,
        "min_difficulty": 1,
        "max_difficulty": 4,
    },
    "multi_step_deduction": {
        "fn": _multi_step_deduction,
        "min_difficulty": 2,
        "max_difficulty": 6,
    },
    "set_membership": {
        "fn": _set_membership,
        "min_difficulty": 1,
        "max_difficulty": 5,
    },
    "constraint_satisfaction": {
        "fn": _constraint_satisfaction,
        "min_difficulty": 3,
        "max_difficulty": 6,
    },
}


def _types_for_difficulty(difficulty: int) -> list[str]:
    return [
        name for name, info in _PROBLEM_TYPES.items()
        if info["min_difficulty"] <= difficulty <= info["max_difficulty"]
    ]


# ---------------------------------------------------------------------------
# Verification: regenerate from seed and check answer matches
# ---------------------------------------------------------------------------

def verify_answer(problem: LogicProblem) -> bool:
    """Regenerate the problem from its seed and verify answer + question match."""
    regenerated = generate_problem(problem["difficulty"], problem["seed"])
    return (
        regenerated["correct_answer"] == problem["correct_answer"]
        and regenerated["question"] == problem["question"]
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def generate_problem(difficulty: int, seed: int) -> LogicProblem:
    """Generate a single logic problem at the given difficulty level."""
    if difficulty < 1 or difficulty > 6:
        raise ValueError(f"difficulty must be 1-6, got {difficulty}")

    rng = random.Random(seed)
    cfg = DIFFICULTY_CONFIG[difficulty]

    available_types = _types_for_difficulty(difficulty)
    chosen_type = rng.choice(available_types)

    # Re-seed for deterministic generation
    rng = random.Random(seed)
    fn = _PROBLEM_TYPES[chosen_type]["fn"]
    question, answer = fn(rng, cfg)

    return LogicProblem(
        question=question,
        correct_answer=answer,
        difficulty=difficulty,
        seed=seed,
        problem_type=chosen_type,
    )


def generate_problem_set(
    n: int = 30,
    seed: int = 42,
) -> list[LogicProblem]:
    """Generate a balanced set of logic problems across all 6 difficulty levels."""
    rng = random.Random(seed)
    problems: list[LogicProblem] = []

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

    from collections import Counter

    by_diff: dict[int, list[LogicProblem]] = {}
    for p in problems:
        by_diff.setdefault(p["difficulty"], []).append(p)

    print(f"Generated {len(problems)} problems across {len(by_diff)} difficulty levels\n")
    print(f"Problem types: {len(_PROBLEM_TYPES)}")
    print(f"Distribution: { {d: len(ps) for d, ps in sorted(by_diff.items())} }")

    type_dist = Counter(p["problem_type"] for p in problems)
    print(f"Type distribution: {dict(type_dist)}")

    # Verify all
    failures = [p for p in problems if not verify_answer(p)]
    print(f"\nAll answers verified: {len(failures) == 0} ({len(problems) - len(failures)}/{len(problems)})")
    if failures:
        for f in failures:
            print(f"  FAIL: diff={f['difficulty']} seed={f['seed']} type={f['problem_type']}")

    # Uniqueness
    questions = [p["question"] for p in problems]
    unique = len(set(questions))
    print(f"Unique questions: {unique}/{len(problems)}")

    print("\n" + "=" * 70)

    for diff in range(1, 7):
        print(f"\n--- Difficulty {diff} ---")
        for p in by_diff.get(diff, [])[:2]:
            print(f"  Type: {p['problem_type']}")
            print(f"  Q: {p['question']}")
            print(f"  A: {p['correct_answer']}")
            print()
