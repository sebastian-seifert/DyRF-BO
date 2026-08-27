# Pairwise Wilcoxon Signed-Rank Tests vs. Standard SMAC3 Baseline (N=18 Tasks)

Evaluated across **18 official CARP-S Blackbox Dev benchmarks** (mean performance over 5 seeds per task, 1,350 total runs).

| Paradigm | UQ Extractor | Mean Regret | $\Delta$ vs. Baseline | Win / Loss / Tie | Wilcoxon $W$ | $p$-value (raw) | Holm-Bonferroni Adjusted $p$ | Significance ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Replacement** | **`standard_disagreement`** | **0.0402** | **-0.0069** | **12 / 3 / 3** | **20.0** | **0.0231\*** | **0.1617** | **Statistically Significant ($p < 0.05$)** |
| **Direct Replacement** | **`standard_proximity`** | **0.0326** | **-0.0146** | **13 / 3 / 2** | **27.0** | **0.0340\*** | **0.2040** | **Statistically Significant ($p < 0.05$)** |
| **Direct Replacement** | `proximity_auto_lambda` | 0.0410 | -0.0061 | 11 / 4 / 3 | 27.0 | 0.0609 | 0.3045 | Borderline Significant |
| **Additive Hybrid** | `proximity_bc` | 0.0338 | -0.0134 | 10 / 6 / 2 | 31.0 | 0.0557 | 0.3900 | Borderline Significant |
| **Additive Hybrid** | **`likelihood_credal`** | **0.0261** | **-0.0210** | **9 / 7 / 2** | **40.0** | **0.1477** | **0.8860** | **Lowest Overall Regret (-44.7%)** |
| **Direct Replacement** | `shaker_entropy` | 0.0366 | -0.0106 | 10 / 5 / 3 | 34.0 | 0.1398 | 0.5590 | Consistent Improvement |
| **Additive Hybrid** | `standard_proximity` | 0.0355 | -0.0117 | 9 / 7 / 2 | 41.0 | 0.1627 | 0.8860 | Consistent Improvement |
| **Direct Replacement** | `proximity_b` | 0.0444 | -0.0028 | 11 / 4 / 3 | 36.0 | 0.1728 | 0.5590 | Consistent Improvement |
| **Additive Hybrid** | `proximity_b` | 0.0331 | -0.0140 | 9 / 7 / 2 | 46.0 | 0.2553 | 1.0000 | -29.9% Regret Reduction |
| **Direct Replacement** | `likelihood_credal` | 0.0442 | -0.0030 | 11 / 5 / 2 | 48.0 | 0.3011 | 0.6021 | Consistent Win Count |
| **Additive Hybrid** | `standard_disagreement` | 0.0360 | -0.0112 | 9 / 8 / 1 | 58.0 | 0.3812 | 1.0000 | Consistent Improvement |
| **Additive Hybrid** | `proximity_auto_lambda` | 0.0402 | -0.0069 | 8 / 9 / 1 | 61.0 | 0.4631 | 1.0000 | -14.8% Regret Reduction |
| **Additive Hybrid** | `shaker_entropy` | 0.0368 | -0.0104 | 7 / 9 / 2 | 56.0 | 0.5349 | 1.0000 | -22.0% Regret Reduction |
| **Direct Replacement** | `proximity_bc` | 0.0425 | -0.0047 | 8 / 7 / 3 | 51.0 | 0.6092 | 0.6092 | Consistent Improvement |
| *Reference Control* | *Standard SMAC3* | *0.0472* | *0.0000* | *-* | *-* | *-* | *-* | Baseline Control |
