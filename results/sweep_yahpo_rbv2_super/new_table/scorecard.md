# Scorecard: sweep_yahpo_rbv2_super

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB    | 79.6%          |     2.29e-07 |         -0.618 | 66.0%         |      0.0039 |        -0.381 |
| SMAC20_ProximityBC_LCB    | 80.1%          |     9.75e-07 |         -0.589 | 63.1%         |      0.0544 |        -0.28  |
| SMAC20_ProximityBC_LCB_CV | 78.6%          |     1.26e-06 |         -0.578 | 62.1%         |      0.1604 |        -0.227 |
| SMAC20_ProximityAC_LCB_CV | 75.7%          |     2.1e-05  |         -0.515 | 60.2%         |      0.3012 |        -0.176 |
| SMAC20_ProximityB_LCB_CV  | 71.8%          |     3.41e-05 |         -0.494 | 53.9%         |      0.4448 |        -0.139 |
| SMAC20_ProximityA_LCB     | 73.3%          |     7.99e-05 |         -0.468 | 56.3%         |      0.4448 |        -0.062 |
| SMAC20_Entropy_LCB        | 70.4%          |     0.000182 |         -0.438 | 60.7%         |      0.1604 |        -0.223 |
| SMAC20_ProximityB_LCB     | 67.5%          |     0.000204 |         -0.41  | 53.9%         |      0.4448 |        -0.133 |
| SMAC20_ProximityA_LCB_CV  | 71.4%          |     0.000204 |         -0.424 | 60.2%         |      0.4448 |        -0.111 |
