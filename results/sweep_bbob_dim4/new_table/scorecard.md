# Scorecard: sweep_bbob_dim4

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB    | 87.5%          |     9.91e-09 |         -0.811 | 68.1%         |      0.0127 |        -0.405 |
| SMAC20_ProximityB_LCB     | 84.7%          |     1.57e-07 |         -0.745 | 56.9%         |      0.1711 |        -0.275 |
| SMAC20_ProximityBC_LCB    | 81.9%          |     4.24e-07 |         -0.718 | 61.1%         |      0.1874 |        -0.262 |
| SMAC20_ProximityB_LCB_CV  | 80.6%          |     4.78e-07 |         -0.711 | 51.4%         |      1      |        -0.079 |
| SMAC20_ProximityBC_LCB_CV | 76.4%          |     6.08e-07 |         -0.7   | 55.6%         |      0.743  |        -0.157 |
| SMAC20_ProximityAC_LCB_CV | 81.9%          |     1.02e-06 |         -0.681 | 55.6%         |      0.8361 |        -0.131 |
| SMAC20_ProximityA_LCB_CV  | 79.2%          |     1.87e-06 |         -0.658 | 50.0%         |      1      |         0.037 |
| SMAC20_ProximityA_LCB     | 73.6%          |     1.55e-05 |         -0.586 | 41.7%         |      1      |         0.231 |
| SMAC20_Entropy_LCB        | 72.2%          |     2.05e-05 |         -0.556 | 40.3%         |      1      |         0.194 |
