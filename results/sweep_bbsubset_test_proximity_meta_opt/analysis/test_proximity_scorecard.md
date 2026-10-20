# CARP-S BBsubset Held-Out Test Evaluation Scorecard

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Proximity LCB, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_lcb` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (20 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 600
- **Seed-level Record (W / T / L)**: **313 / 128 / 159** (52.2% win rate)
- **Task-level Empirical Record (W / T / L)**: **12 / 4 / 4**
- **Task-level Statistically Significant Record (alpha=0.05) (W / T / L)**: **6 / 13 / 1**
- **Mean Normalized Regret**: Proposed = **0.1450** vs Baseline = **0.2212** (+34.4% relative reduction)
- **Task-level Wilcoxon (Demšar) p-value (two-sided)**: `0.0097` (Significant (p < 0.05))
- **Task-level Wilcoxon (Demšar) p-value (one-sided, proposed < baseline)**: `0.0049`
- **Mean Per-Task Cliff's Delta**: `-0.2226` (Favors Proposed)
- **Task-Level Cliff's Delta (on normalized regret)**: `-0.1550`
- **Legacy Pooled Wilcoxon p-value (unnormalized scale-sensitive)**: `2.8467e-11`

## High-Dimensional Stratification Analysis

| Stratum | N Tasks | Mean Regret Proposed | Mean Regret Baseline | Rel. Reduction | Mean Cliff's Delta | Wilcoxon p (1-sided) | Wilcoxon p (2-sided) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBOB High-D (D >= 16) | 4 | 0.2264 | 0.3153 | +28.2% | -0.371 | 1.0000 | 1.0000 |
| BBOB High-D (D >= 8) | 5 | 0.2578 | 0.3672 | +29.8% | -0.339 | 0.0625 | 0.1250 |
| All High-D (D >= 8) | 10 | 0.1820 | 0.2927 | +37.8% | -0.290 | 0.0156 | 0.0312 |
| Low-D (D <= 3) | 6 | 0.0988 | 0.0890 | -11.0% | +0.021 | 0.7812 | 0.6250 |

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 30 | 131.5896 | 153.0966 | 0.4173 | 0.5865 | 24 | 0 | 6 | -0.564 | 6.918e-06 | 1.245e-04 | WIN |
| blackbox/20/test/bbob/16/11/0 | 30 | 823.0777 | 4241.6068 | 0.0094 | 0.0917 | 22 | 1 | 7 | -0.501 | 1.053e-03 | 1.685e-02 | WIN |
| blackbox/20/test/bbob/2/12/2 | 30 | 7331.2405 | 2888.6510 | 0.0000 | 0.0003 | 24 | 0 | 6 | -0.384 | 1.205e-02 | 1.446e-01 | WIN |
| blackbox/20/test/bbob/2/6/1 | 30 | 40.0180 | 38.5855 | 0.2405 | 0.1545 | 6 | 1 | 23 | +0.243 | 9.754e-04 | 1.658e-02 | LOSS |
| blackbox/20/test/bbob/2/9/0 | 30 | -356.8329 | -357.5412 | 0.1243 | 0.1224 | 15 | 0 | 15 | +0.111 | 4.771e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/bbob/32/11/0 | 30 | 1186.6195 | 1864.6336 | 0.0368 | 0.0328 | 16 | 2 | 12 | -0.078 | 3.164e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/bbob/32/9/0 | 30 | 180780.8766 | 208452.0667 | 0.4422 | 0.5503 | 22 | 0 | 8 | -0.342 | 1.341e-03 | 2.011e-02 | WIN |
| blackbox/20/test/bbob/8/22/0 | 30 | 75.5161 | 82.7051 | 0.3834 | 0.5748 | 17 | 0 | 13 | -0.211 | 1.241e-01 | 1.000e+00 | WIN |
| blackbox/20/test/hpobench/blackbox/tabular/ml/svm/12 | 30 | 0.0327 | 0.0327 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.200 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/lcbench/167184/None | 30 | -84.0988 | -83.6046 | 0.2887 | 0.2952 | 20 | 0 | 10 | -0.224 | 1.966e-02 | 2.163e-01 | WIN |
| blackbox/20/test/yahpo/rbv2_glmnet/32/None | 30 | -0.9683 | -0.9683 | 0.1820 | 0.2139 | 20 | 0 | 10 | -0.156 | 2.452e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_glmnet/375/None | 30 | -0.9608 | -0.9609 | 0.0458 | 0.0428 | 12 | 3 | 15 | +0.114 | 3.676e-01 | 1.000e+00 | LOSS |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 30 | -0.9186 | -0.9172 | 0.2338 | 0.4035 | 23 | 0 | 7 | -0.358 | 2.020e-03 | 2.828e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/18/None | 30 | -0.8390 | -0.8270 | 0.0162 | 0.2409 | 27 | 0 | 3 | -0.800 | 2.552e-07 | 5.104e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/4534/None | 30 | -0.9609 | -0.9598 | 0.0839 | 0.3222 | 21 | 0 | 9 | -0.503 | 5.776e-03 | 7.509e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_svm/1493/None | 30 | -0.9027 | -0.9021 | 0.0983 | 0.1040 | 17 | 1 | 12 | -0.157 | 6.266e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 30 | -0.9875 | -0.9816 | 0.2973 | 0.6878 | 27 | 0 | 3 | -0.841 | 2.552e-07 | 5.104e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
