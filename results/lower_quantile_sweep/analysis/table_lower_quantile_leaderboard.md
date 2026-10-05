# Lower-Quantile UQ Absolute Performance Leaderboard

Absolute ranking of all evaluated UQ estimators sorted by mean Spearman rank correlation with distance.

### Leaderboard: All Dimensions

| Rank | Estimator | Method Description | Category | N Runs | Spearman d_norm | Spearman d_inf | Norm Monotonicity |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **1** | `hutter_between` | Hutter Between (Epistemic) | Baseline (Tree Disagreement) | 9600 | **+0.1595** | +0.1778 | 0.5797 |
| **2** | `shaker` | Shaker Epistemic | Baseline (Numerical Integration) | 9600 | **+0.1341** | +0.1407 | 0.5670 |
| **3** | `slcb` | SLCB / Hutter Total | Baseline (Variance) | 9600 | **+0.1308** | +0.1434 | 0.5654 |
| **4** | `hutter_within` | Hutter Within (Aleatoric) | Baseline (Leaf Variance) | 9600 | **+0.1200** | +0.1136 | 0.5600 |
| **5** | `shaker_total` | Shaker Total | Baseline (Numerical Integration) | 9600 | **+0.1166** | +0.1210 | 0.5583 |
| **6** | `shaker_total_entropy` | Shaker Total Entropy | Baseline (Information Theoretic) | 9600 | **+0.1166** | +0.1210 | 0.5583 |
| **7** | `shaker_mi` | Shaker Mutual Information | Baseline (Information Theoretic) | 9600 | **+0.0831** | +0.0931 | 0.5415 |
| **8** | `prox_bc` | Proximity BC (TWQ + Density) | Proximity (Unweighted) | 9600 | **-0.1874** | -0.1757 | 0.4063 |
| **9** | `prox_b` | Proximity B (TWQ Continuous) | Proximity (Unweighted) | 9600 | **-0.3985** | -0.3638 | 0.3007 |
| **10** | `plcb` | PLCB (Adaptive Floor) | Proximity (Unweighted) | 9600 | **-0.4327** | -0.3967 | 0.2837 |
| **11** | `prox_ac` | Proximity AC (TNS + Density) | Proximity (Unweighted) | 9600 | **-0.4332** | -0.3985 | 0.2834 |
| **12** | `prox_a` | Proximity A (TNS Top-k) | Proximity (Unweighted) | 9600 | **-0.4391** | -0.4022 | 0.2804 |
| **13** | `rf_fire` | RF-FIRE Lower | Baseline (Volume Expansion) | 9600 | **-0.4396** | -0.4049 | 0.2802 |

### Leaderboard: Low-D (D <= 5)

| Rank | Estimator | Method Description | Category | N Runs | Spearman d_norm | Spearman d_inf | Norm Monotonicity |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **1** | `hutter_between` | Hutter Between (Epistemic) | Baseline (Tree Disagreement) | 4800 | **+0.0997** | +0.1241 | 0.5498 |
| **2** | `shaker` | Shaker Epistemic | Baseline (Numerical Integration) | 4800 | **+0.0990** | +0.1118 | 0.5495 |
| **3** | `shaker_total` | Shaker Total | Baseline (Numerical Integration) | 4800 | **+0.0770** | +0.0870 | 0.5385 |
| **4** | `shaker_total_entropy` | Shaker Total Entropy | Baseline (Information Theoretic) | 4800 | **+0.0770** | +0.0870 | 0.5385 |
| **5** | `shaker_mi` | Shaker Mutual Information | Baseline (Information Theoretic) | 4800 | **+0.0758** | +0.0901 | 0.5379 |
| **6** | `hutter_within` | Hutter Within (Aleatoric) | Baseline (Leaf Variance) | 4800 | **+0.0651** | +0.0658 | 0.5325 |
| **7** | `slcb` | SLCB / Hutter Total | Baseline (Variance) | 4800 | **+0.0553** | +0.0745 | 0.5277 |
| **8** | `prox_bc` | Proximity BC (TWQ + Density) | Proximity (Unweighted) | 4800 | **-0.1140** | -0.1092 | 0.4430 |
| **9** | `prox_b` | Proximity B (TWQ Continuous) | Proximity (Unweighted) | 4800 | **-0.3709** | -0.3441 | 0.3145 |
| **10** | `plcb` | PLCB (Adaptive Floor) | Proximity (Unweighted) | 4800 | **-0.4313** | -0.4019 | 0.2843 |
| **11** | `prox_ac` | Proximity AC (TNS + Density) | Proximity (Unweighted) | 4800 | **-0.4423** | -0.4125 | 0.2788 |
| **12** | `prox_a` | Proximity A (TNS Top-k) | Proximity (Unweighted) | 4800 | **-0.4436** | -0.4125 | 0.2782 |
| **13** | `rf_fire` | RF-FIRE Lower | Baseline (Volume Expansion) | 4800 | **-0.4547** | -0.4235 | 0.2727 |

### Leaderboard: High-D (D >= 16)

| Rank | Estimator | Method Description | Category | N Runs | Spearman d_norm | Spearman d_inf | Norm Monotonicity |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **1** | `hutter_between` | Hutter Between (Epistemic) | Baseline (Tree Disagreement) | 3200 | **+0.2548** | +0.2608 | 0.6274 |
| **2** | `slcb` | SLCB / Hutter Total | Baseline (Variance) | 3200 | **+0.2484** | +0.2492 | 0.6242 |
| **3** | `hutter_within` | Hutter Within (Aleatoric) | Baseline (Leaf Variance) | 3200 | **+0.1958** | +0.1799 | 0.5979 |
| **4** | `shaker` | Shaker Epistemic | Baseline (Numerical Integration) | 3200 | **+0.1830** | +0.1808 | 0.5915 |
| **5** | `shaker_total` | Shaker Total | Baseline (Numerical Integration) | 3200 | **+0.1750** | +0.1716 | 0.5875 |
| **6** | `shaker_total_entropy` | Shaker Total Entropy | Baseline (Information Theoretic) | 3200 | **+0.1750** | +0.1716 | 0.5875 |
| **7** | `shaker_mi` | Shaker Mutual Information | Baseline (Information Theoretic) | 3200 | **+0.0994** | +0.1026 | 0.5497 |
| **8** | `prox_bc` | Proximity BC (TWQ + Density) | Proximity (Unweighted) | 3200 | **-0.2507** | -0.2332 | 0.3747 |
| **9** | `rf_fire` | RF-FIRE Lower | Baseline (Volume Expansion) | 3200 | **-0.3886** | -0.3555 | 0.3057 |
| **10** | `prox_ac` | Proximity AC (TNS + Density) | Proximity (Unweighted) | 3200 | **-0.3962** | -0.3600 | 0.3019 |
| **11** | `prox_b` | Proximity B (TWQ Continuous) | Proximity (Unweighted) | 3200 | **-0.4048** | -0.3648 | 0.2976 |
| **12** | `plcb` | PLCB (Adaptive Floor) | Proximity (Unweighted) | 3200 | **-0.4083** | -0.3690 | 0.2958 |
| **13** | `prox_a` | Proximity A (TNS Top-k) | Proximity (Unweighted) | 3200 | **-0.4088** | -0.3693 | 0.2956 |
