# CARP-S 1v1 Paired Wilcoxon Signed-Rank Evaluation Report

**Significance Criterion:** Two-sided Paired Wilcoxon Signed-Rank test with $\alpha = 0.05$.
**Effect Size:** Matched-pairs Rank-Biserial Correlation $r_{\text{rb}} \in [-1.0, 1.0]$ (negative values indicate lower objective cost for the candidate).

## 1. Global Cross-Task Aggregate Summary

| Candidate Approach | Tasks | Wins | Ties | Losses | Win Rate (%) | Mean Rank (Cand / Base) | Global Wilcoxon W | Global p-value | Global Sig (α=0.05) | Macro $r_{\text{rb}}$ | Rel Imp (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | 18 | **4** | 8 | 6 | 22.2% | 1.53 / 1.47 | 81.0 | 0.8617 | NO | -0.059 | -354.98% |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | 18 | **0** | 18 | 0 | 0.0% | 1.44 / 1.56 | 72.0 | 0.6008 | NO | -0.147 | -89.83% |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | 18 | **1** | 17 | 0 | 5.6% | 1.47 / 1.53 | 84.0 | 0.9653 | NO | -0.007 | -10.14% |

## 2. Per-Approach & Per-Task Breakdown

### Approach: `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` vs `SMAC3_HPOFacade_ei`

| Task ID | Seeds | Candidate Cost (Mean ± SEM) | Baseline Cost (Mean ± SEM) | $r_{\text{rb}}$ | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `blackbox/20/dev/bbob/2/12/0` | 30 | 75307.0942 ± 28112.1838 | 17228.3569 ± 7804.5605 | +0.53 | 0.0144 | 0.1579 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/bbob/2/12/1` | 30 | 1232.3851 ± 525.7501 | -534.2651 ± 53.9428 | +0.90 | 0.0000 | 0.0003 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/bbob/2/20/0` | 30 | 205.3311 ± 12.8922 | 185.4541 ± 0.5519 | +0.80 | 0.0000 | 0.0006 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/bbob/4/6/1` | 30 | 85.0121 ± 5.3488 | 88.4691 ± 4.5504 | -0.17 | 0.4400 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/lr/146818` | 30 | 0.1557 ± 0.0001 | 0.1559 ± 0.0002 | -0.30 | 0.4722 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/rf/146212` | 30 | 0.0001 ± 0.0000 | 0.0000 ± 0.0000 | +0.98 | 0.0000 | 0.0001 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/xgboost/146212` | 30 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.50 | 0.1573 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/lcbench/168335/None` | 30 | -97.6098 ± 0.2347 | -96.2711 ± 0.3820 | -0.59 | 0.0040 | 0.0564 | ✓ **YES** | **WIN** |
| `blackbox/20/dev/yahpo/rbv2_aknn/1462/None` | 30 | -0.9999 ± 0.0000 | -1.0000 ± 0.0000 | +0.80 | 0.0003 | 0.0042 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/yahpo/rbv2_aknn/312/None` | 30 | -0.9669 ± 0.0002 | -0.9667 ± 0.0003 | -0.29 | 0.1642 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/40498/None` | 30 | -0.7149 ± 0.0077 | -0.6812 ± 0.0100 | -0.58 | 0.0047 | 0.0606 | ✓ **YES** | **WIN** |
| `blackbox/20/dev/yahpo/rbv2_aknn/458/None` | 30 | -0.9983 ± 0.0001 | -0.9986 ± 0.0002 | +0.57 | 0.0054 | 0.0646 | ✓ **YES** | *LOSS* |
| `blackbox/20/dev/yahpo/rbv2_glmnet/41157/None` | 30 | -0.6432 ± 0.0062 | -0.6269 ± 0.0095 | -0.29 | 0.1772 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_ranger/40927/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +0.00 | 1.0000 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/182/None` | 30 | -0.9246 ± 0.0009 | -0.9248 ± 0.0014 | +0.05 | 0.8236 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/24/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +0.30 | 0.1680 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_xgboost/23512/None` | 30 | -0.9944 ± 0.0056 | -0.9635 ± 0.0119 | -0.64 | 0.0363 | 0.3264 | ✓ **YES** | **WIN** |
| `blackbox/20/dev/yahpo/rbv2_xgboost/42/None` | 30 | -0.9381 ± 0.0126 | -0.8752 ± 0.0220 | -0.51 | 0.0224 | 0.2239 | ✓ **YES** | **WIN** |

### Approach: `SMAC20_CustomUncertainty_ei_standard_disagreement` vs `SMAC3_HPOFacade_ei`

| Task ID | Seeds | Candidate Cost (Mean ± SEM) | Baseline Cost (Mean ± SEM) | $r_{\text{rb}}$ | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `blackbox/20/dev/bbob/2/12/0` | 30 | 32423.7251 ± 13173.1520 | 17228.3569 ± 7804.5605 | +0.35 | 0.0957 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/2/12/1` | 30 | -583.1746 ± 9.3290 | -534.2651 ± 53.9428 | +0.14 | 0.5028 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/2/20/0` | 30 | 185.1801 ± 0.1647 | 185.4541 ± 0.5519 | +0.16 | 0.4522 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/4/6/1` | 30 | 90.3937 ± 3.9642 | 88.4691 ± 4.5504 | +0.11 | 0.5928 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/lr/146818` | 30 | 0.1559 ± 0.0002 | 0.1559 ± 0.0002 | -0.11 | 0.7553 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/rf/146212` | 30 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.16 | 0.5048 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/xgboost/146212` | 30 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.00 | 1.0000 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/lcbench/168335/None` | 30 | -96.5196 ± 0.3930 | -96.2711 ± 0.3820 | -0.18 | 0.3990 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/1462/None` | 30 | -0.9999 ± 0.0000 | -1.0000 ± 0.0000 | +0.36 | 0.1406 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/312/None` | 30 | -0.9666 ± 0.0003 | -0.9667 ± 0.0003 | +0.11 | 0.5978 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/40498/None` | 30 | -0.6799 ± 0.0091 | -0.6812 ± 0.0100 | +0.27 | 0.1981 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/458/None` | 30 | -0.9987 ± 0.0002 | -0.9986 ± 0.0002 | +0.02 | 0.9263 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_glmnet/41157/None` | 30 | -0.6407 ± 0.0090 | -0.6269 ± 0.0095 | -0.31 | 0.1386 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_ranger/40927/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +0.00 | 1.0000 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/182/None` | 30 | -0.9217 ± 0.0015 | -0.9248 ± 0.0014 | +0.35 | 0.1132 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/24/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | -0.28 | 0.1675 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_xgboost/23512/None` | 30 | -0.9837 ± 0.0089 | -0.9635 ± 0.0119 | -0.55 | 0.1170 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_xgboost/42/None` | 30 | -0.9121 ± 0.0178 | -0.8752 ± 0.0220 | -0.37 | 0.1353 | 1.0000 | ✗ NO | TIE |

### Approach: `SMAC20_CustomUncertainty_ei_standard_proximity` vs `SMAC3_HPOFacade_ei`

| Task ID | Seeds | Candidate Cost (Mean ± SEM) | Baseline Cost (Mean ± SEM) | $r_{\text{rb}}$ | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `blackbox/20/dev/bbob/2/12/0` | 30 | 18884.6531 ± 11437.0355 | 17228.3569 ± 7804.5605 | +0.09 | 0.6959 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/2/12/1` | 30 | -480.6913 ± 80.0149 | -534.2651 ± 53.9428 | +0.17 | 0.4400 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/2/20/0` | 30 | 185.2454 ± 0.1878 | 185.4541 ± 0.5519 | +0.22 | 0.3085 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/bbob/4/6/1` | 30 | 88.7822 ± 4.0996 | 88.4691 ± 4.5504 | +0.02 | 0.9018 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/lr/146818` | 30 | 0.1558 ± 0.0002 | 0.1559 ± 0.0002 | -0.28 | 0.4977 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/rf/146212` | 30 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | -0.07 | 0.5220 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/hpobench/blackbox/tabular/ml/xgboost/146212` | 30 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | -0.33 | 0.5637 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/lcbench/168335/None` | 30 | -96.6039 ± 0.3391 | -96.2711 ± 0.3820 | -0.19 | 0.3818 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/1462/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +0.33 | 0.1492 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/312/None` | 30 | -0.9667 ± 0.0002 | -0.9667 ± 0.0003 | +0.05 | 0.7892 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/40498/None` | 30 | -0.6934 ± 0.0093 | -0.6812 ± 0.0100 | -0.29 | 0.1642 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_aknn/458/None` | 30 | -0.9986 ± 0.0002 | -0.9986 ± 0.0002 | +0.06 | 0.7655 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_glmnet/41157/None` | 30 | -0.6188 ± 0.0077 | -0.6269 ± 0.0095 | +0.13 | 0.5561 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_ranger/40927/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +0.00 | 1.0000 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/182/None` | 30 | -0.9234 ± 0.0018 | -0.9248 ± 0.0014 | +0.10 | 0.6215 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_svm/24/None` | 30 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | -0.23 | 0.3049 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_xgboost/23512/None` | 30 | -0.9799 ± 0.0096 | -0.9635 ± 0.0119 | -0.39 | 0.1477 | 1.0000 | ✗ NO | TIE |
| `blackbox/20/dev/yahpo/rbv2_xgboost/42/None` | 30 | -0.9222 ± 0.0167 | -0.8752 ± 0.0220 | -0.47 | 0.0424 | 0.7639 | ✓ **YES** | **WIN** |

