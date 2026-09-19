# CARP-S BBsubset Held-Out Test Evaluation Scorecard

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Tuned Proximity LCB: k=25, decay_lambda=1.345, eps=0.1678, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_lcb` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (20 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 600
- **Seed-level Record (W / T / L)**: **313 / 128 / 159** (52.2% win rate)
- **Task-level Empirical Record (W / T / L)**: **12 / 4 / 4**
- **Task-level Statistically Significant Record (alpha=0.05) (W / T / L)**: **6 / 13 / 1**
- **Mean Normalized Regret**: Proposed = **0.1845** vs Baseline = **0.2571** (+28.2% relative reduction)
- **Task-level Wilcoxon (Demšar) p-value (two-sided)**: `0.0151` (Significant (p < 0.05))
- **Task-level Wilcoxon (Demšar) p-value (one-sided, proposed < baseline)**: `0.0075`
- **Mean Per-Task Cliff's Delta**: `-0.2226` (Favors Proposed)
- **Task-Level Cliff's Delta (on normalized regret)**: `-0.1500`
- **Legacy Pooled Wilcoxon p-value (unnormalized scale-sensitive)**: `2.8467e-11`

## High-Dimensional Stratification Analysis

| Stratum | N Tasks | Mean Regret Proposed | Mean Regret Baseline | Rel. Reduction | Mean Cliff's Delta | Wilcoxon p (1-sided) | Wilcoxon p (2-sided) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBOB High-D (D >= 16) | 4 | 0.2356 | 0.3658 | +35.6% | -0.371 | 0.0625 | 0.1250 |
| BBOB High-D (D >= 8) | 5 | 0.2777 | 0.4028 | +31.1% | -0.339 | 0.0312 | 0.0625 |
| All High-D (D >= 8) | 10 | 0.1955 | 0.3075 | +36.4% | -0.290 | 0.0078 | 0.0156 |
| Low-D (D <= 3) | 6 | 0.1703 | 0.1286 | -32.5% | +0.021 | 0.9375 | 0.1875 |

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 30 | 131.5896 | 153.0966 | 0.4024 | 0.6332 | 24 | 0 | 6 | -0.564 | 6.918e-06 | 1.245e-04 | WIN |
| blackbox/20/test/bbob/16/11/0 | 30 | 823.0777 | 4241.6068 | 0.0298 | 0.1555 | 22 | 1 | 7 | -0.501 | 1.053e-03 | 1.685e-02 | WIN |
| blackbox/20/test/bbob/2/12/2 | 30 | 7331.2405 | 2888.6510 | 0.0416 | 0.0172 | 24 | 0 | 6 | -0.384 | 1.205e-02 | 1.446e-01 | LOSS |
| blackbox/20/test/bbob/2/6/1 | 30 | 40.0180 | 38.5855 | 0.3259 | 0.2107 | 6 | 1 | 23 | +0.243 | 9.754e-04 | 1.658e-02 | LOSS |
| blackbox/20/test/bbob/2/9/0 | 30 | -356.8329 | -357.5412 | 0.2552 | 0.1858 | 15 | 0 | 15 | +0.111 | 4.771e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/bbob/32/11/0 | 30 | 1186.6195 | 1864.6336 | 0.0718 | 0.1191 | 16 | 2 | 12 | -0.078 | 3.164e-01 | 1.000e+00 | WIN |
| blackbox/20/test/bbob/32/9/0 | 30 | 180780.8766 | 208452.0667 | 0.4385 | 0.5554 | 22 | 0 | 8 | -0.342 | 1.341e-03 | 2.011e-02 | WIN |
| blackbox/20/test/bbob/8/22/0 | 30 | 75.5161 | 82.7051 | 0.4458 | 0.5510 | 17 | 0 | 13 | -0.211 | 1.241e-01 | 1.000e+00 | WIN |
| blackbox/20/test/hpobench/blackbox/tabular/ml/svm/12 | 30 | 0.0327 | 0.0327 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.200 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/lcbench/167184/None | 30 | -84.0988 | -83.6046 | 0.2436 | 0.3379 | 20 | 0 | 10 | -0.224 | 1.966e-02 | 2.163e-01 | WIN |
| blackbox/20/test/yahpo/rbv2_glmnet/32/None | 30 | -0.9683 | -0.9683 | 0.2302 | 0.2649 | 20 | 0 | 10 | -0.156 | 2.452e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_glmnet/375/None | 30 | -0.9608 | -0.9609 | 0.1691 | 0.0929 | 12 | 3 | 15 | +0.114 | 3.676e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 30 | -0.9186 | -0.9172 | 0.2786 | 0.4131 | 23 | 0 | 7 | -0.358 | 2.020e-03 | 2.828e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/18/None | 30 | -0.8390 | -0.8270 | 0.0510 | 0.3381 | 27 | 0 | 3 | -0.800 | 2.552e-07 | 5.104e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/4534/None | 30 | -0.9609 | -0.9598 | 0.2114 | 0.3877 | 21 | 0 | 9 | -0.503 | 5.776e-03 | 7.509e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_svm/1493/None | 30 | -0.9027 | -0.9021 | 0.2075 | 0.2331 | 17 | 1 | 12 | -0.157 | 6.266e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 30 | -0.9875 | -0.9816 | 0.2879 | 0.6472 | 27 | 0 | 3 | -0.841 | 2.552e-07 | 5.104e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
