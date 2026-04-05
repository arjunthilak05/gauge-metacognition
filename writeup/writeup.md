# GAUGE: Measuring the gap between what LLMs know and what they do about it

## Team
Arjun Thilak — Eros Gen AI, AI Research Scientist
Ramkumar MV - Eros Gen AI, AI Research Scientist

## Problem statement

A model that reports 95% confidence and never abstains is more dangerous than one that gets questions wrong. It gives the human operator no signal that something might be off.

Nelson and Narens (1990) split metacognition into monitoring and control. Monitoring is knowing how well you're doing. Control is acting on that knowledge. A doctor who recognizes uncertainty orders more tests. A pilot unsure of an instrument reading cross-checks before acting. Both halves matter, and they have to work together.

Kadavath et al. (2022) showed that large language models can partially distinguish questions they will answer correctly from ones they won't. That is monitoring. The open question is whether that self-assessment changes behavior. Does the model abstain when unsure? Does it hold its ground under pressure when it should?

Recent work confirms the gap is real. Wang et al. (2026) found that models produce calibrated verbal confidence but almost never adjust abstention behavior, even when extreme penalties make abstention the mathematically optimal strategy. MUAZU's Cognitive Assessment Benchmark independently observed "strong monitoring paired with weak control." We formalize this observation with a coupling metric: the L-Score.

A system that knows it's uncertain but acts certain anyway is worse than one that's simply wrong, because the operator trusts the confidence signal. GAUGE measures monitoring, control, and whether they talk to each other.

## Task and benchmark construction

Single-shot benchmarks cannot measure metacognition. Metacognition is about the relationship between self-assessment and subsequent behavior. That requires multiple turns.

GAUGE uses a three-turn staircase protocol testing the Nelson and Narens monitoring-control loop as a dynamic process. All 17 other metacognition submissions in this competition use single-shot evaluation; ours is the only multi-turn protocol.

**Metacognitive Staircase** (270 items across 3 domains and 6 difficulty levels). The model first predicts problem difficulty before solving (an Ease-of-Learning judgment). It then solves and reports confidence 0-100 following Tian et al. (2023). Finally, it chooses: submit (+3 if correct, -1 if wrong) or abstain (+1 if the answer would have been wrong, 0 if correct). Turn 1 measures monitoring. Turn 3 measures control. The game-theoretic payoff makes confidence consequential: below 40% confidence, abstention maximizes expected score.

Three composite scores capture the picture. M-Score averages monitoring metrics (ECE, Brier, AUROC, EOL correlation). C-Score measures control efficiency in the staircase game. L-Score is the point-biserial correlation between stated confidence and the binary submit/abstain decision, requiring at least 3 abstentions to avoid spurious inflation. We confirm robustness via Kendall's tau, with consistent results across both measures. Fleming and Lau (2014) recommend Signal Detection Theory measures for human metacognition; we use point-biserial for interpretability at benchmark scale, where the binary dependent variable makes SDT's Type 2 framework less natural.

![fig1_reliability.png](fig1_reliability.png)

## Dataset

All 270 items are procedurally generated from parameterized templates with deterministic seeds across three domains: mathematics (122 items), logic (115 items), and factual knowledge (33 items). No LLM is involved in generation, and every item has a verifiable ground-truth answer. Math problems use 15 templates with GSM-NoOp-style distractors (Mirzadeh et al., 2024) across six difficulty levels. Logic problems use proof-tree-first generation with fictional entities to block memorization.

From an initial pool of 1,093 candidate items, Classical Test Theory item analysis selected the final 270: p-value between 0.15 and 0.85, discrimination index above 0.2, stratified to exactly 45 items per difficulty level for balanced measurement. CTT provides transparent, assumption-free item quality metrics.

## Results

We tested eight models from four families: Gemini (3), Claude (3), Qwen (1), and DeepSeek (1), totaling 2,160 model-item evaluations.

Gemini 2.5 Pro achieved 94.1% accuracy and ECE of 0.053 — the best raw performance and calibration in our sample. Its L-Score was 0.0. It never abstained once across 270 items. All three Gemini models share this behavior: zero abstentions, L-Score of exactly 0.0, despite confidence standard deviations as low as 0.018 (Gemini 2.5 Pro reports near-100% confidence on virtually everything).

![fig4_abstention.png](fig4_abstention.png)

Three models crossed the threshold for meaningful strategic abstention. Claude Haiku 4.5 abstained on 25 of 270 items with 84% precision — most abstained items would have been wrong. Its L-Score was 0.870 and its overall metacognitive score (0.857) was the highest in the sample despite having the lowest accuracy (78.2%). Qwen3-235B abstained on 22 items (L = 0.813, overall = 0.868). DeepSeek V3.1 abstained on 6 items (L = 0.587). Strategic abstention is not an Anthropic-specific behavior.

The per-difficulty breakdown makes the failure mode concrete. At difficulty levels 5-6, Gemini 3 Flash reported 97% mean confidence while accuracy was 73-84% — a calibration gap exceeding 20 percentage points. Claude Haiku reported 83-90% confidence at the same levels with similar accuracy but abstained on 13-27% of those items. The difference is structural: monitoring without control versus monitoring driving control.

![fig3_difficulty.png](fig3_difficulty.png)

Tomani et al. (2024) showed that uncertainty-based abstention reduces unsafe responses by 70-99%. Our results identify which models would benefit: Gemini's monitoring is accurate enough to support abstention, but its control pipeline ignores it. This is recoverable through fine-tuning, but only if the gap is measured.

![fig2_composite.png](fig2_composite.png)

Gemini 2.5 Pro is the better test-taker. Qwen3-235B and Claude Haiku are the most self-aware. For any deployment where the operator needs to know when to override — medical diagnosis, legal analysis, autonomous systems — monitoring without control is a failure mode that scaling alone will not fix.

## Organizational affiliations
Eros Gen AI - AI Research Scientist

## References

- Burnell, R., et al. (2026). Measuring Progress Toward AGI. Google DeepMind.
- Fleming, S. M. and Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8, 443.
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Koriat, A. and Goldsmith, M. (1996). Monitoring and Control Processes in Memory Accuracy. Psychological Review, 103(3).
- Mirzadeh, I., et al. (2024). GSM-Symbolic. arXiv:2410.05229.
- Nelson, T. O. and Narens, L. (1990). Metamemory: A Theoretical Framework. Psychology of Learning and Motivation, 26.
- Tian, K., et al. (2023). Just Ask for Calibration. EMNLP.
- Tomani, C., et al. (2024). Uncertainty-Based Abstention in LLMs Improves Safety. arXiv:2404.10960.
- Wang, J., et al. (2026). Are LLM Decisions Faithful to Verbal Confidence? arXiv:2601.07767.
