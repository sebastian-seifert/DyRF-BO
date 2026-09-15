# CARP-S BBsubset Held-Out Test Evaluation Scorecard

## Executive Summary

- **Proposed Method**: `SMAC20_ProximityLCB_tuned` (Tuned Proximity LCB: k=25, decay_lambda=1.345, eps=0.1678, level=0.95)
- **Baseline Method**: `SMAC3_HPOFacade_lcb` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)
- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set (20 tasks, 30 seeds, T=100 budget)
- **Total Paired Runs**: 600
- **Seed-level Record (W / T / L)**: **270 / 128 / 202** (45.0% win rate)
- **Task-level Empirical Record (W / T / L)**: **10 / 4 / 6**
- **Task-level Statistically Significant Record (alpha=0.05) (W / T / L)**: **6 / 11 / 3**
- **Mean Normalized Regret**: Proposed = **0.2271** vs Baseline = **0.2646**
- **Overall Cliff's Delta**: `+0.0053` (Favors Baseline)
- **Macro Wilcoxon p-value**: `3.1576e-06` (Significant (p < 0.05))

## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)

| task | n_seeds | mean_proposed | mean_baseline | norm_regret_proposed | norm_regret_baseline | wins | ties | losses | cliffs_delta | p_raw | p_holm | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackbox/20/test/bbob/16/1/1 | 30 | 134.3430 | 153.1340 | 0.3361 | 0.5695 | 28 | 0 | 2 | -0.593 | 5.718e-07 | 1.086e-05 | WIN |
| blackbox/20/test/bbob/16/11/0 | 30 | 494.6388 | 5018.8670 | 0.0097 | 0.0951 | 23 | 0 | 7 | -0.413 | 6.084e-04 | 9.126e-03 | WIN |
| blackbox/20/test/bbob/2/12/2 | 30 | 25522.6885 | 5696.3218 | 0.1065 | 0.0246 | 13 | 2 | 15 | +0.042 | 8.352e-02 | 7.517e-01 | LOSS |
| blackbox/20/test/bbob/2/6/1 | 30 | 40.9667 | 38.9685 | 0.3680 | 0.2221 | 5 | 0 | 25 | +0.304 | 6.287e-05 | 1.069e-03 | LOSS |
| blackbox/20/test/bbob/2/9/0 | 30 | -354.9448 | -357.8648 | 0.0739 | 0.0259 | 9 | 0 | 21 | +0.262 | 8.705e-03 | 9.576e-02 | LOSS |
| blackbox/20/test/bbob/32/11/0 | 30 | 438.5605 | 1239.2513 | 0.0619 | 0.2726 | 21 | 5 | 4 | -0.479 | 9.804e-04 | 1.373e-02 | WIN |
| blackbox/20/test/bbob/32/9/0 | 30 | 182475.9886 | 228382.0350 | 0.4092 | 0.6122 | 26 | 0 | 4 | -0.558 | 1.192e-06 | 2.146e-05 | WIN |
| blackbox/20/test/bbob/8/22/0 | 30 | 80.5662 | 84.3971 | 0.4799 | 0.5381 | 16 | 0 | 14 | -0.124 | 4.771e-01 | 1.000e+00 | WIN |
| blackbox/20/test/hpobench/blackbox/tabular/ml/svm/12 | 30 | 0.0327 | 0.0327 | 0.0000 | 0.0000 | 0 | 30 | 0 | -0.033 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/lcbench/167184/None | 30 | -84.1025 | -83.9662 | 0.6748 | 0.7006 | 16 | 0 | 14 | +0.042 | 7.922e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_glmnet/32/None | 30 | -0.9678 | -0.9683 | 0.2873 | 0.1656 | 9 | 1 | 20 | +0.301 | 2.253e-02 | 2.253e-01 | LOSS |
| blackbox/20/test/yahpo/rbv2_glmnet/375/None | 30 | -0.9608 | -0.9608 | 0.3123 | 0.1442 | 6 | 0 | 24 | +0.476 | 1.431e-03 | 1.717e-02 | LOSS |
| blackbox/20/test/yahpo/rbv2_ranger/29/None | 30 | -0.9175 | -0.9169 | 0.3131 | 0.3708 | 20 | 0 | 10 | -0.187 | 2.801e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/18/None | 30 | -0.8379 | -0.8263 | 0.0998 | 0.4544 | 28 | 0 | 2 | -0.844 | 3.148e-07 | 6.296e-06 | WIN |
| blackbox/20/test/yahpo/rbv2_rpart/4534/None | 30 | -0.9608 | -0.9604 | 0.2893 | 0.3769 | 21 | 0 | 9 | -0.309 | 1.706e-01 | 1.000e+00 | WIN |
| blackbox/20/test/yahpo/rbv2_svm/1493/None | 30 | -0.8972 | -0.9027 | 0.4283 | 0.2069 | 6 | 0 | 24 | +0.484 | 1.529e-04 | 2.446e-03 | LOSS |
| blackbox/20/test/yahpo/rbv2_xgboost/1457/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1493/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
| blackbox/20/test/yahpo/rbv2_xgboost/1510/None | 30 | -0.9846 | -0.9808 | 0.2910 | 0.5119 | 23 | 0 | 7 | -0.569 | 1.038e-03 | 1.373e-02 | WIN |
| blackbox/20/test/yahpo/rbv2_xgboost/41027/None | 30 | -1.0000 | -1.0000 | 0.0000 | 0.0000 | 0 | 30 | 0 | +0.000 | 1.000e+00 | 1.000e+00 | TIE |
