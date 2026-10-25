# Scorecard: sweep_bbob_dim32

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityBC_LCB_CV | 93.1%          |     1.23e-12 |         -0.979 | 91.7%         |    2.62e-12 |        -0.975 |
| SMAC20_ProximityB_LCB     | 93.1%          |     1.23e-12 |         -0.986 | 91.7%         |    4.4e-12  |        -0.963 |
| SMAC20_ProximityAC_LCB_CV | 94.4%          |     1.23e-12 |         -0.985 | 87.5%         |    6.38e-12 |        -0.947 |
| SMAC20_ProximityBC_LCB    | 94.4%          |     1.23e-12 |         -0.989 | 89.6%         |    6.38e-12 |        -0.958 |
| SMAC20_ProximityB_LCB_CV  | 94.4%          |     1.23e-12 |         -0.99  | 93.1%         |    2.6e-12  |        -0.977 |
| SMAC20_ProximityAC_LCB    | 95.8%          |     1.23e-12 |         -0.986 | 88.2%         |    6.38e-12 |        -0.952 |
| SMAC20_Entropy_LCB        | 95.8%          |     1.23e-12 |         -0.988 | 93.1%         |    6.38e-12 |        -0.952 |
| SMAC20_ProximityA_LCB     | 98.6%          |     1.23e-12 |         -0.991 | 90.3%         |    6.38e-12 |        -0.944 |
| SMAC20_ProximityA_LCB_CV  | 94.4%          |     1.23e-12 |         -0.978 | 90.3%         |    8.58e-12 |        -0.912 |
