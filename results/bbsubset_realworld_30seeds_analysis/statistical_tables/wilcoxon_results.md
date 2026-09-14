# Wilcoxon Signed-Rank Test: Real-World ML Benchmark Suite (30 Seeds)

- Reference Baseline: `SMAC3_HPOFacade_ei`
- Proposed Model: `CARPSDynamicRF_DAEHRF_AdditiveEI`
- Evaluated Real-World Tasks: 14
- Metric: `trial_value__cost_inc` (Final Incumbent Cost, Minimization)

### Summary Statistics

| Proposed Optimizer | Baseline Optimizer | N Tasks | Proposed Mean Incumbent | Baseline Mean Incumbent | Mean Diff vs Base | Rel Reduction (%) | Win / Loss / Tie | Wilcoxon W | p_val | Cliff's delta | Significance (p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CARPSDynamicRF_DAEHRF_AdditiveEI | SMAC3_HPOFacade_ei | 14 | -7.5802 | -7.4937 | -0.0865 | +1.2% | 7 / 6 / 1 | 27.0 | 0.1961 | -0.056 | Non-significant |

### Per-Task Breakdown

| task_id | CARPSDynamicRF_DAEHRF_AdditiveEI | SMAC3_HPOFacade_ei | Diff (Prop - Base) | Outcome |
| --- | --- | --- | --- | --- |
| blackbox/20/dev/hpobench/blackbox/tabular/ml/lr/146818 | 0.15593 | 0.15577 | 0.00016 | Loss (Baseline) |
| blackbox/20/dev/hpobench/blackbox/tabular/ml/rf/146212 | 0.00015 | 5e-05 | 0.0001 | Loss (Baseline) |
| blackbox/20/dev/hpobench/blackbox/tabular/ml/xgboost/146212 | 0.0 | 0.0 | 0.0 | Loss (Baseline) |
| blackbox/20/dev/yahpo/lcbench/168335/None | -97.07958 | -95.97644 | -1.10313 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_aknn/1462/None | -0.9999 | -0.99996 | 7e-05 | Loss (Baseline) |
| blackbox/20/dev/yahpo/rbv2_aknn/312/None | -0.96643 | -0.96691 | 0.00049 | Loss (Baseline) |
| blackbox/20/dev/yahpo/rbv2_aknn/40498/None | -0.72228 | -0.69134 | -0.03095 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_aknn/458/None | -0.998 | -0.99864 | 0.00064 | Loss (Baseline) |
| blackbox/20/dev/yahpo/rbv2_glmnet/41157/None | -0.64308 | -0.62963 | -0.01345 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_ranger/40927/None | -1.0 | -1.0 | 0.0 | Tie |
| blackbox/20/dev/yahpo/rbv2_svm/182/None | -0.92535 | -0.92376 | -0.00159 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_svm/24/None | -1.0 | -1.0 | -0.0 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_xgboost/23512/None | -1.0 | -0.96407 | -0.03593 | Win (Proposed) |
| blackbox/20/dev/yahpo/rbv2_xgboost/42/None | -0.94418 | -0.91689 | -0.02729 | Win (Proposed) |
