# Scorecard: sweep_bbob_dim16

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB    | 97.2%          |     8.47e-13 |         -0.998 | 88.9%         |    1.82e-11 |        -0.938 |
| SMAC20_ProximityBC_LCB    | 95.8%          |     1.1e-12  |         -0.989 | 88.9%         |    1.55e-11 |        -0.944 |
| SMAC20_ProximityBC_LCB_CV | 95.8%          |     1.1e-12  |         -0.97  | 87.5%         |    1.05e-10 |        -0.865 |
| SMAC20_ProximityAC_LCB_CV | 94.4%          |     1.1e-12  |         -0.989 | 84.7%         |    7.85e-11 |        -0.904 |
| SMAC20_ProximityB_LCB     | 95.8%          |     1.1e-12  |         -0.989 | 87.5%         |    4.95e-11 |        -0.916 |
| SMAC20_ProximityB_LCB_CV  | 95.8%          |     1.1e-12  |         -0.986 | 87.5%         |    1.05e-10 |        -0.884 |
| SMAC20_Entropy_LCB        | 95.8%          |     1.1e-12  |         -0.986 | 83.3%         |    7.85e-11 |        -0.896 |
| SMAC20_ProximityA_LCB     | 97.2%          |     1.1e-12  |         -0.991 | 81.9%         |    1.05e-10 |        -0.882 |
| SMAC20_ProximityA_LCB_CV  | 95.8%          |     1.1e-12  |         -0.989 | 86.1%         |    7.85e-11 |        -0.904 |
