# Benchmark Sweep Scorecard

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityBC_LCB    | 80.0%          |       0.0122 |         -0.824 | 65.0%         |      0.6018 |        -0.426 |
| SMAC20_ProximityAC_LCB    | 80.0%          |       0.0122 |         -0.794 | 50.0%         |      0.6018 |        -0.294 |
| SMAC20_ProximityAC_LCB_CV | 80.0%          |       0.0122 |         -0.838 | 57.5%         |      0.6018 |        -0.417 |
| SMAC20_ProximityA_LCB     | 85.0%          |       0.0122 |         -0.853 | 50.0%         |      0.6018 |        -0.412 |
| SMAC20_ProximityBC_LCB_CV | 85.0%          |       0.0122 |         -0.853 | 60.0%         |      0.6018 |        -0.397 |
| SMAC20_ProximityB_LCB_CV  | 85.0%          |       0.0122 |         -0.853 | 50.0%         |      0.6018 |        -0.279 |
| SMAC20_ProximityB_LCB     | 75.0%          |       0.0125 |         -0.75  | 57.5%         |      0.6018 |        -0.367 |
| SMAC20_ProximityA_LCB_CV  | 75.0%          |       0.0125 |         -0.721 | 55.0%         |      0.6018 |        -0.324 |
| SMAC20_Entropy_LCB        | 75.0%          |       0.0125 |         -0.676 | 55.0%         |      0.6018 |        -0.353 |
