# CARP-S BBsubset Held-Out Test Evaluation Scorecard

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Tuned Proximity LCB: k=25, decay_lambda=1.345, eps=0.1678, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_lcb` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (10 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 300
- **Seed-level Record (W / T / L)**: **157 / 95 / 48** (52.3% win rate)
- **Task-level Empirical Record (W / T / L)**: **7 / 3 / 0**
- **Task-level Statistically Significant Record (alpha=0.05) (W / T / L)**: **5 / 5 / 0**
- **Mean Normalized Regret**: Proposed = **0.1901** vs Baseline = **0.2970** (+36.0% relative reduction)
- **Task-level Wilcoxon (Demšar) p-value (two-sided)**: `0.0156` (Significant (p < 0.05))
- **Task-level Wilcoxon (Demšar) p-value (one-sided, proposed < baseline)**: `0.0078`
- **Mean Per-Task Cliff's Delta**: `-0.2923` (Favors Proposed)
- **Task-Level Cliff's Delta (on normalized regret)**: `-0.2500`
- **Legacy Pooled Wilcoxon p-value (unnormalized scale-sensitive)**: `2.9769e-14`

## High-Dimensional Stratification Analysis

| Stratum | N Tasks | Mean Regret Proposed | Mean Regret Baseline | Rel. Reduction | Mean Cliff's Delta | Wilcoxon p (1-sided) | Wilcoxon p (2-sided) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBOB High-D (D >= 16) | 4 | 0.2042 | 0.3874 | +47.3% | -0.511 | 0.0625 | 0.1250 |
| BBOB High-D (D >= 8) | 5 | 0.2594 | 0.4175 | +37.9% | -0.434 | 0.0312 | 0.0625 |
| All High-D (D >= 8) | 10 | 0.1901 | 0.2970 | +36.0% | -0.292 | 0.0078 | 0.0156 |

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 30 | 134.3430 | 153.1340 | 0.3361 | 0.5695 | 28 | 0 | 2 | -0.593 | 5.718e-07 | 5.718e-06 | WIN |
| blackbox/20/test/bbob/16/11/0 | 30 | 494.6388 | 5018.8670 | 0.0097 | 0.0951 | 23 | 0 | 7 | -0.413 | 6.084e-04 | 4.867e-03 | WIN |
| blackbox/20/test/bbob/32/11/0 | 30 | 438.5605 | 1239.2513 | 0.0619 | 0.2726 | 21 | 5 | 4 | -0.479 | 9.804e-04 | 6.863e-03 | WIN |
| blackbox/20/test/bbob/32/9/0 | 30 | 182475.9886 | 228382.0350 | 0.4092 | 0.6122 | 26 | 0 | 4 | -0.558 | 1.192e-06 | 1.073e-05 | WIN |
| blackbox/20/test/bbob/8/22/0 | 30 | 80.5662 | 84.3971 | 0.4799 | 0.5381 | 16 | 0 | 14 | -0.124 | 4.771e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 30 | -0.9175 | -0.9169 | 0.3131 | 0.3708 | 20 | 0 | 10 | -0.187 | 2.801e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 30 | -0.9846 | -0.9808 | 0.2910 | 0.5119 | 23 | 0 | 7 | -0.569 | 1.038e-03 | 6.863e-03 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
