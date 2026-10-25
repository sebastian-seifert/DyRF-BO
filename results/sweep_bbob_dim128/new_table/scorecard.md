# Scorecard: sweep_bbob_dim128

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer              | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:-----------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityAC_LCB | 90.3%          |     5.68e-10 |         -0.849 | 65.3%         |      0.0036 |        -0.412 |
| SMAC20_ProximityBC_LCB | 77.8%          |     3.56e-09 |         -0.801 | 58.3%         |      0.029  |        -0.264 |
| SMAC20_Entropy_LCB     | 80.6%          |     7e-09    |         -0.769 | 61.1%         |      0.029  |        -0.296 |
