# Scorecard: sweep_yahpo_rbv2_ranger

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB_CV | 95.0%          |     2.86e-18 |         -0.972 | 83.6%         |    4.78e-14 |        -0.849 |
| SMAC20_ProximityB_LCB_CV  | 94.1%          |     3.4e-18  |         -0.968 | 78.6%         |    6.94e-13 |        -0.807 |
| SMAC20_Entropy_LCB        | 93.3%          |     3.4e-18  |         -0.968 | 82.8%         |    5.13e-12 |        -0.764 |
| SMAC20_ProximityAC_LCB    | 93.3%          |     3.41e-18 |         -0.964 | 78.6%         |    1.85e-12 |        -0.784 |
| SMAC20_ProximityBC_LCB_CV | 93.3%          |     3.41e-18 |         -0.961 | 86.1%         |    2.24e-13 |        -0.826 |
| SMAC20_ProximityB_LCB     | 93.3%          |     3.41e-18 |         -0.963 | 82.8%         |    7.11e-13 |        -0.805 |
| SMAC20_ProximityBC_LCB    | 94.1%          |     3.41e-18 |         -0.965 | 87.0%         |    1.23e-12 |        -0.794 |
| SMAC20_ProximityA_LCB_CV  | 94.1%          |     3.41e-18 |         -0.961 | 76.1%         |    5.59e-10 |        -0.681 |
| SMAC20_ProximityA_LCB     | 89.9%          |     1.78e-17 |         -0.921 | 72.7%         |    7.84e-08 |        -0.576 |
