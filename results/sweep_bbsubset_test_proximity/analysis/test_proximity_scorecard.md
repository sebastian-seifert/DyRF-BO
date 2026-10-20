# CARP-S BBsubset Held-Out Test Evaluation Scorecard

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Proximity LCB, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_lcb` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (20 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 600
- **Seed-level Record (W / T / L)**: **270 / 128 / 202** (45.0% win rate)
- **Task-level Empirical Record (W / T / L)**: **9 / 5 / 6**
- **Task-level Statistically Significant Record (alpha=0.05) (W / T / L)**: **6 / 11 / 3**
- **Mean Normalized Regret**: Proposed = **0.1863** vs Baseline = **0.2368** (+21.3% relative reduction)
- **Task-level Wilcoxon (Demšar) p-value (two-sided)**: `0.2146` (Not Significant (p >= 0.05))
- **Task-level Wilcoxon (Demšar) p-value (one-sided, proposed < baseline)**: `0.1073`
- **Mean Per-Task Cliff's Delta**: `-0.1082` (Favors Proposed)
- **Task-Level Cliff's Delta (on normalized regret)**: `-0.0950`
- **Legacy Pooled Wilcoxon p-value (unnormalized scale-sensitive)**: `2.7295e-06`

## High-Dimensional Stratification Analysis

| Stratum | N Tasks | Mean Regret Proposed | Mean Regret Baseline | Rel. Reduction | Mean Cliff's Delta | Wilcoxon p (1-sided) | Wilcoxon p (2-sided) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBOB High-D (D >= 16) | 4 | 0.1950 | 0.3283 | +40.6% | -0.511 | 1.0000 | 1.0000 |
| BBOB High-D (D >= 8) | 5 | 0.2398 | 0.3721 | +35.6% | -0.434 | 0.0312 | 0.0625 |
| All High-D (D >= 8) | 10 | 0.1689 | 0.2754 | +38.7% | -0.292 | 0.0078 | 0.0156 |
| Low-D (D <= 3) | 6 | 0.1249 | 0.0657 | -89.9% | +0.231 | 0.9688 | 0.1250 |

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 30 | 134.3430 | 153.1340 | 0.3140 | 0.5875 | 28 | 0 | 2 | -0.593 | 5.718e-07 | 1.086e-05 | WIN |
| blackbox/20/test/bbob/16/11/0 | 30 | 494.6388 | 5018.8670 | 0.0027 | 0.0189 | 23 | 0 | 7 | -0.413 | 6.084e-04 | 9.126e-03 | WIN |
| blackbox/20/test/bbob/2/12/2 | 30 | 25522.6885 | 5696.3218 | 0.0003 | 0.0003 | 13 | 2 | 15 | +0.042 | 8.352e-02 | 7.517e-01 | TIE |
| blackbox/20/test/bbob/2/6/1 | 30 | 40.9667 | 38.9685 | 0.2751 | 0.1899 | 5 | 0 | 25 | +0.304 | 6.287e-05 | 1.069e-03 | LOSS |
| blackbox/20/test/bbob/2/9/0 | 30 | -354.9448 | -357.8648 | 0.0239 | 0.0098 | 9 | 0 | 21 | +0.262 | 8.705e-03 | 9.576e-02 | LOSS |
| blackbox/20/test/bbob/32/11/0 | 30 | 438.5605 | 1239.2513 | 0.0381 | 0.1126 | 21 | 5 | 4 | -0.479 | 9.804e-04 | 1.373e-02 | WIN |
| blackbox/20/test/bbob/32/9/0 | 30 | 182475.9886 | 228382.0350 | 0.4251 | 0.5939 | 26 | 0 | 4 | -0.558 | 1.192e-06 | 2.146e-05 | WIN |
| blackbox/20/test/bbob/8/22/0 | 30 | 80.5662 | 84.3971 | 0.4188 | 0.5476 | 16 | 0 | 14 | -0.124 | 4.771e-01 | 1.000e+00 | WIN |
| blackbox/20/test/hpobench/blackbox/tabular/ml/svm/12 | 30 | 0.0327 | 0.0327 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/lcbench/167184/None | 30 | -84.1025 | -83.9662 | 0.7110 | 0.6595 | 16 | 0 | 14 | +0.042 | 7.922e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/yahpo/rbv2_glmnet/32/None | 30 | -0.9678 | -0.9683 | 0.1664 | 0.1236 | 9 | 1 | 20 | +0.301 | 2.253e-02 | 2.253e-01 | LOSS |
| blackbox/20/test/yahpo/rbv2_glmnet/375/None | 30 | -0.9608 | -0.9608 | 0.2836 | 0.0709 | 6 | 0 | 24 | +0.476 | 1.483e-03 | 1.780e-02 | LOSS |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 30 | -0.9175 | -0.9169 | 0.2279 | 0.3916 | 20 | 0 | 10 | -0.187 | 2.801e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/18/None | 30 | -0.8379 | -0.8263 | 0.0472 | 0.4803 | 28 | 0 | 2 | -0.844 | 3.148e-07 | 6.296e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/4534/None | 30 | -0.9608 | -0.9604 | 0.1180 | 0.3386 | 21 | 0 | 9 | -0.309 | 1.706e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_svm/1493/None | 30 | -0.8972 | -0.9027 | 0.4123 | 0.1091 | 6 | 0 | 24 | +0.484 | 1.529e-04 | 2.446e-03 | LOSS |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 30 | -0.9846 | -0.9808 | 0.2626 | 0.5021 | 23 | 0 | 7 | -0.569 | 1.038e-03 | 1.373e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
