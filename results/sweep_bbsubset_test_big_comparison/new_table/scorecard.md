# Scorecard: sweep_bbsubset_test_big_comparison

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityB_LCB_CV  | 85.0%          |       0.0122 |         -0.853 | 60.0%         |      0.5438 |        -0.426 |
| SMAC20_ProximityAC_LCB_CV | 75.0%          |       0.0523 |         -0.706 | 60.0%         |      0.5438 |        -0.441 |
| SMAC20_ProximityBC_LCB_CV | 75.0%          |       0.0528 |         -0.691 | 65.0%         |      0.5438 |        -0.441 |
| SMAC20_ProximityAC_LCB    | 75.0%          |       0.0786 |         -0.632 | 55.0%         |      0.704  |        -0.265 |
| SMAC20_ProximityB_LCB     | 65.0%          |       0.0786 |         -0.588 | 65.0%         |      0.6382 |        -0.324 |
| SMAC20_ProximityA_LCB     | 70.0%          |       0.0786 |         -0.618 | 65.0%         |      0.5438 |        -0.397 |
| SMAC20_ProximityA_LCB_CV  | 65.0%          |       0.0836 |         -0.544 | 50.0%         |      0.704  |        -0.265 |
| SMAC20_ProximityBC_LCB    | 65.0%          |       0.0879 |         -0.485 | 60.0%         |      0.704  |        -0.147 |
| SMAC20_Entropy_LCB        | 60.0%          |       0.0879 |         -0.471 | 50.0%         |      0.704  |         0.029 |
