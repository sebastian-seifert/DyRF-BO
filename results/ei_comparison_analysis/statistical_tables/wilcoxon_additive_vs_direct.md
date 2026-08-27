# Head-to-Head Wilcoxon Signed-Rank Tests: Additive Hybrid vs. Direct Replacement (N=18 Tasks)

Evaluated pairwise for each of the **7 UQ extractors** across **18 official CARP-S benchmarks** (mean performance over 5 seeds per task).

| UQ Extractor | Additive Mean Regret | Direct Mean Regret | $\Delta$ (Additive - Direct) | Additive Win / Loss / Tie | Wilcoxon $W$ | $p$-value (raw) | Holm-Bonferroni Adjusted $p$ | Primary Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`likelihood_credal`** | **0.0261** | 0.0442 | **-0.0181** | 7 / 9 / 2 | 49.0 | 0.3259 | 1.0000 | **Additive cuts regret by 41% (Best Overall)** |
| **`proximity_b`** | **0.0331** | 0.0444 | **-0.0113** | 6 / 10 / 2 | 67.0 | 0.9588 | 1.0000 | **Additive cuts regret by 25%** |
| **`proximity_bc`** | **0.0338** | 0.0425 | **-0.0087** | 8 / 8 / 2 | 47.0 | 0.2775 | 1.0000 | **Additive cuts regret by 20%** |
| **`proximity_auto_lambda`** | **0.0402** | 0.0410 | -0.0008 | 8 / 9 / 1 | 59.0 | 0.4074 | 1.0000 | Additive slightly lower regret |
| **`standard_disagreement`** | 0.0360 | **0.0402** | -0.0042 | 7 / 10 / 1 | 74.0 | 0.9058 | 1.0000 | Direct has higher win count |
| **`shaker_entropy`** | 0.0368 | **0.0366** | +0.0002 | 6 / 9 / 3 | 56.0 | 0.8203 | 1.0000 | Functionally equivalent |
| **`standard_proximity`** | 0.0355 | **0.0326** | +0.0029 | 7 / 10 / 1 | 76.0 | 0.9811 | 1.0000 | Direct has slightly tighter rank |
