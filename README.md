# GAUGE: Grading Awareness of Uncertainty and Grounded Epistemics

A benchmark for measuring metacognitive monitoring and control in large language models. Built for the Kaggle "Measuring Progress Toward AGI" Hackathon (Metacognition Track).

Most benchmarks ask whether a model can answer correctly. GAUGE asks whether it knows when it can't, and whether it does anything about it.

## The finding

Gemini 2.5 Pro: 94% accuracy. Never abstained once across 270 items. L-Score = 0.0.

Qwen3-235B: 82% accuracy. Abstained 22 times with 82% precision. L-Score = 0.813.

Claude Haiku 4.5: 78% accuracy. Abstained 25 times with 84% precision. L-Score = 0.870.

The best test-taker has the worst self-awareness. The model that knows when to step back has the highest overall metacognitive score.

## How it works

270 items. Three turns per item. Grounded in Nelson & Narens (1990).

| Turn | What happens | What it measures |
|------|-------------|-----------------|
| 1 | Predict difficulty before solving | Ease-of-Learning (prospective monitoring) |
| 2 | Solve + report confidence 0-100 | Calibration (retrospective monitoring) |
| 3 | Submit (+3/-1) or Abstain (+1/0) | Metacognitive control |

Below 40% confidence, abstaining is the rational move. Gemini never gets there because it reports near-100% confidence on everything.

## Results (8 models, 4 families)

| Model | Accuracy | ECE | L-Score | Abstentions | Overall |
|-------|----------|-----|---------|-------------|---------|
| Qwen3-235B | 81.6% | 0.117 | 0.813 | 22/261 | 0.868 |
| Claude Haiku 4.5 | 78.2% | 0.151 | 0.870 | 25/270 | 0.857 |
| DeepSeek V3.1 | 83.9% | 0.137 | 0.587 | 6/218 | 0.794 |
| Claude Sonnet 4.6 | 81.5% | 0.148 | 0.574 | 4/270 | 0.759 |
| Claude Sonnet 4 | 81.1% | 0.138 | 0.241 | 4/270 | 0.732 |
| Gemini 2.5 Pro | 94.1% | 0.053 | 0.000 | 0/270 | 0.708 |
| Gemini 3 Flash | 82.5% | 0.159 | 0.000 | 0/269 | 0.686 |
| Gemini 2.5 Flash | 87.0% | 0.125 | 0.000 | 1/270 | 0.673 |

## Scoring

| Score | What it measures | How |
|-------|-----------------|-----|
| M-Score | Monitoring accuracy | Average of (1-ECE), (1-Brier), AUROC, EOL correlation |
| C-Score | Control efficiency | Game score / oracle optimal score |
| L-Score | Monitoring-control coupling | Point-biserial correlation (confidence vs submit/abstain) |

Overall = 0.4M + 0.4C + 0.2L

## Dataset

270 items across math (122), logic (115), and factual knowledge (33). Six difficulty levels, 45 items each. All procedurally generated with verified ground truth. No LLM involved in generation.

Selected from 1,093 candidates via Classical Test Theory (p-value 0.15-0.85, discrimination > 0.2).

## Project structure

```
metacog-benchmark/
├── notebooks/
│   ├── metacog_full.py          # Full analytics notebook (270 items, 8 models, scoring + viz)
│   └── benchmark_task.py        # Kaggle "Add Models" task (270 items, minimal)
├── generators/
│   ├── math_gen.py              # 15 templates, GSM-NoOp distractors, 6 difficulty levels
│   ├── logic_gen.py             # Syllogisms, deduction, constraints, fictional entities
│   └── factual_gen.py           # 5 rarity levels + trick questions
├── scoring/
│   ├── calibration.py           # ECE, Brier, AUROC, AUGRC
│   ├── composite.py             # M-Score, C-Score, L-Score
│   └── irt.py                   # Classical Test Theory item analysis
├── scripts/
│   ├── generate_pool.py         # Generate 1,093 candidate items
│   └── generate_final_270.py    # CTT selection down to 270
├── datasets/
│   ├── candidate_pool.csv       # Full 1,093-item pool
│   ├── final_items_v2.csv       # Selected 270 items
│   └── items_270.py             # Python list for embedding in notebooks
├── figures/                     # All visualization PNGs
└── writeup/
    └── writeup.md               # Competition writeup (1,065 words)
```

## References

- Nelson & Narens (1990). Metamemory: A Theoretical Framework. Psychology of Learning and Motivation.
- Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Tian et al. (2023). Just Ask for Calibration. EMNLP.
- Fleming & Lau (2014). How to measure metacognition. Frontiers in Human Neuroscience.
- Wang et al. (2026). Are LLM Decisions Faithful to Verbal Confidence? arXiv:2601.07767.
- Tomani et al. (2024). Uncertainty-Based Abstention in LLMs Improves Safety. arXiv:2404.10960.
- Mirzadeh et al. (2024). GSM-Symbolic. arXiv:2410.05229.
- Burnell et al. (2026). Measuring Progress Toward AGI. Google DeepMind.
