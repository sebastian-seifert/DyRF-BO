# Scorecard: sweep_bbob_dim2

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB    | 70.8%          |       0.1115 |         -0.304 | 43.1%         |           1 |         0.431 |
| SMAC20_ProximityBC_LCB_CV | 63.9%          |       0.2901 |         -0.244 | 45.8%         |           1 |         0.311 |
| SMAC20_ProximityBC_LCB    | 65.3%          |       0.7742 |         -0.166 | 45.8%         |           1 |         0.355 |
| SMAC20_ProximityB_LCB     | 63.9%          |       0.7742 |         -0.164 | 43.1%         |           1 |         0.364 |
| SMAC20_ProximityAC_LCB_CV | 58.3%          |       0.9231 |         -0.122 | 38.9%         |           1 |         0.468 |
| SMAC20_ProximityB_LCB_CV  | 58.3%          |       1      |         -0.054 | 35.4%         |           1 |         0.563 |
| SMAC20_ProximityA_LCB_CV  | 54.2%          |       1      |          0.043 | 32.6%         |           1 |         0.569 |
| SMAC20_ProximityA_LCB     | 51.4%          |       1      |          0.156 | 34.7%         |           1 |         0.562 |
| SMAC20_Entropy_LCB        | 50.0%          |       1      |          0.202 | 34.0%         |           1 |         0.644 |
