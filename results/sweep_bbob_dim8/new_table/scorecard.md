# Scorecard: sweep_bbob_dim8

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityA_LCB     | 98.6%          |     8.12e-13 |         -0.998 | 77.8%         |    2.05e-07 |        -0.709 |
| SMAC20_ProximityAC_LCB    | 95.8%          |     1.01e-12 |         -0.992 | 79.2%         |    1.08e-08 |        -0.807 |
| SMAC20_ProximityB_LCB     | 95.8%          |     1.01e-12 |         -0.992 | 79.2%         |    1.73e-07 |        -0.728 |
| SMAC20_ProximityAC_LCB_CV | 95.8%          |     1.06e-12 |         -0.986 | 79.9%         |    1.91e-08 |        -0.797 |
| SMAC20_ProximityBC_LCB_CV | 94.4%          |     1.06e-12 |         -0.986 | 80.6%         |    7.26e-09 |        -0.818 |
| SMAC20_ProximityB_LCB_CV  | 93.1%          |     1.06e-12 |         -0.983 | 80.6%         |    2.1e-08  |        -0.785 |
| SMAC20_Entropy_LCB        | 95.8%          |     1.06e-12 |         -0.976 | 79.2%         |    1.73e-07 |        -0.731 |
| SMAC20_ProximityBC_LCB    | 91.7%          |     2.22e-12 |         -0.95  | 77.8%         |    2.05e-07 |        -0.702 |
| SMAC20_ProximityA_LCB_CV  | 93.1%          |     2.22e-12 |         -0.952 | 79.2%         |    2.05e-07 |        -0.715 |
