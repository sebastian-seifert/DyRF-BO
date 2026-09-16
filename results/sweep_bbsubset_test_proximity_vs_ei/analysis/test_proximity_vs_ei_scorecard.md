# CARP-S BBsubset Held-Out Test Evaluation Scorecard: Proximity LCB vs SMAC3 EI

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Tuned Proximity LCB: k=25, decay_lambda=1.345, eps=0.1678, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_ei` (Standard SMAC3 with native Expected Improvement)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (20 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 600
- **Seed-level Record (W / T / L)**: **220 / 130 / 250** (36.7% win rate)
- **Task-level Empirical Record (W / T / L)**: **8 / 4 / 8**
- **Task-level Statistically Significant Record (Holm alpha=0.05) (W / T / L)**: **1 / 15 / 4**
- **Mean Normalized Regret**: Proposed = **0.2257** vs Baseline = **0.2073** (-8.9% relative reduction)
- **Demšar Task-Level Wilcoxon p-value (two-sided)**: `5.6949e-01` (Not Significant (p >= 0.05))
- **Demšar Task-Level Wilcoxon p-value (one-sided, proposed < baseline)**: `7.1525e-01`
- **Exact Binomial Sign Test on Task Wins**: `5.9819e-01` (8/16 non-tied tasks won)
- **Mean Within-Task Cliff's Delta**: `+0.0161` (Favors Baseline)
- **Task-Level Cliff's Delta (on normalized regret)**: `+0.0600`

## Stratified Scorecard Breakdown

| Stratum | N Tasks | Record (W/T/L) | Mean Regret Proposed | Mean Regret Baseline | Rel. Reduction | Mean Cliff's Delta | Demšar Wilcoxon p (2-sided) | Demšar Wilcoxon p (1-sided) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Tasks (Full Suite) | 20 | 8/4/8 | 0.2257 | 0.2073 | -8.9% | +0.016 | 0.5695 | 0.7153 |
| All High-D (D >= 8) | 10 | 7/3/0 | 0.1947 | 0.2491 | +21.9% | -0.178 | 0.0156 | 0.0078 |
| All Low-D (D < 8) | 10 | 1/1/8 | 0.2567 | 0.1654 | -55.2% | +0.210 | 0.0273 | 0.9902 |
| BBOB Continuous Subset | 8 | 5/0/3 | 0.2496 | 0.2497 | +0.0% | -0.034 | 0.9453 | 0.4727 |
| Real-World ML (YAHPO + HPOBench) | 12 | 3/4/5 | 0.2098 | 0.1790 | -17.2% | +0.049 | 0.3828 | 0.8438 |

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | dim | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 16 | 30 | 134.4526 | 139.4285 | 0.3289 | 0.4074 | 23 | 0 | 7 | -0.182 | 6.356e-02 | 6.356e-01 | WIN |
| blackbox/20/test/bbob/16/11/0 | 16 | 30 | 611.4976 | 1432.3282 | 0.0656 | 0.1501 | 19 | 2 | 9 | -0.369 | 5.019e-02 | 5.521e-01 | WIN |
| blackbox/20/test/bbob/2/12/2 | 2 | 30 | 29664.3791 | 573.2969 | 0.1742 | 0.0048 | 8 | 0 | 22 | +0.540 | 7.979e-04 | 1.356e-02 | LOSS |
| blackbox/20/test/bbob/2/6/1 | 2 | 30 | 40.8680 | 38.9132 | 0.3593 | 0.2163 | 3 | 0 | 27 | +0.351 | 5.548e-04 | 9.987e-03 | LOSS |
| blackbox/20/test/bbob/2/9/0 | 2 | 30 | -354.5966 | -357.3417 | 0.0616 | 0.0267 | 13 | 1 | 16 | +0.046 | 4.688e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/bbob/32/11/0 | 32 | 30 | 520.4696 | 1520.2150 | 0.0218 | 0.0900 | 19 | 3 | 8 | -0.334 | 4.583e-03 | 6.875e-02 | WIN |
| blackbox/20/test/bbob/32/9/0 | 32 | 30 | 186987.1067 | 206615.7787 | 0.4882 | 0.5848 | 18 | 1 | 11 | -0.286 | 3.892e-02 | 4.671e-01 | WIN |
| blackbox/20/test/bbob/8/22/0 | 8 | 30 | 78.2098 | 79.4993 | 0.4973 | 0.5175 | 14 | 1 | 15 | -0.034 | 9.914e-01 | 1.000e+00 | WIN |
| blackbox/20/test/hpobench/blackbox/tabular/ml/svm/12 | 0 | 30 | 0.0327 | 0.0327 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/lcbench/167184/None | 7 | 30 | -84.1746 | -84.3084 | 0.4880 | 0.4653 | 13 | 1 | 16 | +0.069 | 6.733e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/yahpo/rbv2_glmnet/32/None | 3 | 30 | -0.9679 | -0.9683 | 0.3702 | 0.2079 | 8 | 0 | 22 | +0.374 | 1.454e-02 | 1.890e-01 | LOSS |
| blackbox/20/test/yahpo/rbv2_glmnet/375/None | 3 | 30 | -0.9608 | -0.9609 | 0.2611 | 0.0479 | 2 | 1 | 27 | +0.700 | 1.604e-05 | 3.207e-04 | LOSS |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 8 | 30 | -0.9177 | -0.9171 | 0.2934 | 0.3406 | 16 | 0 | 14 | -0.129 | 6.702e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/18/None | 5 | 30 | -0.8384 | -0.8333 | 0.0681 | 0.1997 | 22 | 0 | 8 | -0.551 | 2.766e-03 | 4.426e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/4534/None | 5 | 30 | -0.9605 | -0.9607 | 0.3236 | 0.2730 | 15 | 0 | 15 | +0.013 | 6.702e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/yahpo/rbv2_svm/1493/None | 6 | 30 | -0.8972 | -0.9029 | 0.4614 | 0.2126 | 6 | 0 | 24 | +0.560 | 2.367e-05 | 4.497e-04 | LOSS |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 14 | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 14 | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 14 | 30 | -0.9850 | -0.9822 | 0.2516 | 0.4008 | 21 | 0 | 9 | -0.447 | 8.705e-03 | 1.219e-01 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 14 | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
