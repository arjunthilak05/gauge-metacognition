## What does GAUGE measure?

Most benchmarks ask whether a model can answer correctly. We ask something different: does it know when it can't?

A model that reports 95% confidence and never abstains gives the operator no warning signal. If you're deploying that model for medical triage or legal review, you have a problem. Not because it gets things wrong, but because it never tells you when it might.

GAUGE measures monitoring (can the model assess its own accuracy?), control (does it act on that assessment?), and whether the two are actually connected. That last part is the L-Score, and it's what separates this benchmark from everything else in this competition.

## How it works

270 items. Three turns per item. Grounded in Nelson & Narens (1990).

| Turn | What happens | What it measures |
|------|-------------|-----------------|
| 1 | Predict difficulty BEFORE solving | Ease-of-Learning (prospective monitoring) |
| 2 | Solve + report confidence 0-100 | Calibration (retrospective monitoring) |
| 3 | Submit (+3/-1) or Abstain (+1/0) | Metacognitive control |

The payoffs are asymmetric on purpose. You get +3 for a correct submission, -1 for a wrong one, +1 for abstaining on something you'd have gotten wrong, and 0 for abstaining on something you'd have nailed. The math works out so that below 40% confidence, abstaining is the rational move.

Items span math (122), logic (115), and factual knowledge (33) across 6 difficulty levels. We started with 1,093 candidates, ran Classical Test Theory analysis (p-value 0.15-0.85, discrimination > 0.2), and kept 270, stratified to 45 per difficulty level. Every item has a verified ground-truth answer. No LLM was involved in generation.

`[IMAGE: gallery2_final.png]`

## Results

8 models, 4 families, 2,160 total evaluations.

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

Look at the bottom three rows. All Gemini. All L-Score = 0.000. At most one abstention across 270 items.

Gemini 2.5 Pro got 94% accuracy with an ECE of 0.053. If you stopped there, you'd call it the best model in the sample. But it reported near-100% confidence on everything, including the items it got wrong. It never once considered stepping back.

Claude Haiku got 78% accuracy. Lower. But it abstained 25 times, and 84% of those were on items it would have gotten wrong. Qwen3-235B abstained 22 times with 82% precision. They're actually using their uncertainty estimates to make decisions.

Gemini is the better test-taker. Qwen and Claude know when they don't know.

`[IMAGE: fig4_abstention.png]`

`[IMAGE: fig1_reliability.png]`

## Where calibration breaks

At difficulty levels 5 and 6, Gemini models report 97%+ confidence. Their accuracy at those levels is 73-84%. That's a 20+ percentage point gap between how sure they are and how often they're right.

Claude and Qwen drop their confidence as problems get harder and start abstaining on the hardest items. Left panel below: accuracy falls with difficulty (everyone does this). Right panel: which models adjust confidence to match. Gemini doesn't.

`[IMAGE: fig3_difficulty.png]`

`[IMAGE: fig2_composite.png]`

## What we did that nobody else did

L-Score is a new metric. It's the point-biserial correlation between stated confidence and the submit/abstain decision, with a minimum of 3 abstentions to avoid spurious values. We validate it with Kendall's tau and get consistent rankings. Nobody else in this competition measures whether monitoring actually drives behavior.

This is the only multi-turn metacognition benchmark among 17 submissions. Everyone else does single-shot evaluation. Metacognition is about self-assessment changing behavior over time, which you can't measure in one turn.

We're also the only benchmark with formal psychometric item selection. 1,093 candidates down to 270 via CTT. And the game-theoretic payoff structure means confidence isn't just a number the model reports and forgets. It has consequences.

## References

- Nelson & Narens (1990). Metamemory: A Theoretical Framework. Psychology of Learning and Motivation.
- Tian et al. (2023). Just Ask for Calibration. EMNLP.
- Fleming & Lau (2014). How to measure metacognition. Frontiers in Human Neuroscience.
- Wang et al. (2026). Are LLM Decisions Faithful to Verbal Confidence? arXiv:2601.07767.
- Tomani et al. (2024). Uncertainty-Based Abstention in LLMs Improves Safety. arXiv:2404.10960.
