# Scorecard: sweep_hpobench_ml

**Statistical Notes**:
- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.
- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.
- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \text{cand} < \text{baseline}$) with step-down Holm-Bonferroni correction.
- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\Delta| \le 10^{-6}$) strictly filtered out prior to ranking.

| Optimizer                 | LCB Win Rate   |   LCB Adj. p |   LCB $r_{rb}$ | EI Win Rate   |   EI Adj. p |   EI $r_{rb}$ |
|:--------------------------|:---------------|-------------:|---------------:|:--------------|------------:|--------------:|
| SMAC20_ProximityBC_LCB    | 53.4%          |            1 |         -0.104 | 47.2%         |           1 |         0.131 |
| SMAC20_ProximityAC_LCB_CV | 50.6%          |            1 |          0.049 | 48.3%         |           1 |         0.226 |
| SMAC20_ProximityB_LCB     | 46.0%          |            1 |          0.171 | 43.2%         |           1 |         0.439 |
| SMAC20_ProximityAC_LCB    | 51.7%          |            1 |          0.092 | 47.2%         |           1 |         0.361 |
| SMAC20_ProximityA_LCB     | 48.3%          |            1 |          0.198 | 44.3%         |           1 |         0.476 |
| SMAC20_ProximityB_LCB_CV  | 49.4%          |            1 |          0.159 | 46.0%         |           1 |         0.343 |
| SMAC20_ProximityBC_LCB_CV | 48.3%          |            1 |          0.232 | 43.2%         |           1 |         0.501 |
| SMAC20_ProximityA_LCB_CV  | 47.7%          |            1 |          0.295 | 41.5%         |           1 |         0.608 |
| SMAC20_Entropy_LCB        | 45.5%          |            1 |          0.358 | 39.8%         |           1 |         0.693 |
