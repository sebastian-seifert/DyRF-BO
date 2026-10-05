# Lower-Quantile UQ: Surrogate Architecture Breakdown

Empirical calibration performance of each UQ estimator stratified across the 5 Random Forest surrogates:
`smac_default`, `mature`, `shallow`, `coarse`, `breiman`.

| Surrogate | Estimator | Category | N Runs | Spearman d_norm | Norm Monotonicity |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `breiman` | `prox_a` | Proximity (Unweighted) | 1920 | -0.3090 | 0.3455 |
| `breiman` | `prox_b` | Proximity (Unweighted) | 1920 | -0.2454 | 0.3773 |
| `breiman` | `prox_ac` | Proximity (Unweighted) | 1920 | -0.3477 | 0.3261 |
| `breiman` | `prox_bc` | Proximity (Unweighted) | 1920 | -0.3537 | 0.3231 |
| `breiman` | `plcb` | Proximity (Unweighted) | 1920 | -0.3088 | 0.3456 |
| `breiman` | `slcb` | Baseline (Variance) | 1920 | +0.2285 | 0.6142 |
| `breiman` | `hutter_between` | Baseline (Tree Disagreement) | 1920 | +0.1862 | 0.5931 |
| `breiman` | `hutter_within` | Baseline (Leaf Variance) | 1920 | +0.3495 | 0.6748 |
| `breiman` | `shaker` | Baseline (Numerical Integration) | 1920 | +0.1022 | 0.5511 |
| `breiman` | `shaker_total` | Baseline (Numerical Integration) | 1920 | +0.1668 | 0.5834 |
| `breiman` | `shaker_mi` | Baseline (Information Theoretic) | 1920 | -0.0603 | 0.4699 |
| `breiman` | `shaker_total_entropy` | Baseline (Information Theoretic) | 1920 | +0.1668 | 0.5834 |
| `breiman` | `rf_fire` | Baseline (Volume Expansion) | 1920 | -0.3226 | 0.3387 |
| `coarse` | `prox_a` | Proximity (Unweighted) | 1920 | -0.4864 | 0.2568 |
| `coarse` | `prox_b` | Proximity (Unweighted) | 1920 | -0.4412 | 0.2794 |
| `coarse` | `prox_ac` | Proximity (Unweighted) | 1920 | -0.4785 | 0.2608 |
| `coarse` | `prox_bc` | Proximity (Unweighted) | 1920 | -0.2114 | 0.3943 |
| `coarse` | `plcb` | Proximity (Unweighted) | 1920 | -0.4773 | 0.2614 |
| `coarse` | `slcb` | Baseline (Variance) | 1920 | +0.0688 | 0.5344 |
| `coarse` | `hutter_between` | Baseline (Tree Disagreement) | 1920 | +0.0781 | 0.5390 |
| `coarse` | `hutter_within` | Baseline (Leaf Variance) | 1920 | +0.1023 | 0.5511 |
| `coarse` | `shaker` | Baseline (Numerical Integration) | 1920 | +0.0916 | 0.5458 |
| `coarse` | `shaker_total` | Baseline (Numerical Integration) | 1920 | +0.0982 | 0.5491 |
| `coarse` | `shaker_mi` | Baseline (Information Theoretic) | 1920 | +0.0581 | 0.5291 |
| `coarse` | `shaker_total_entropy` | Baseline (Information Theoretic) | 1920 | +0.0982 | 0.5491 |
| `coarse` | `rf_fire` | Baseline (Volume Expansion) | 1920 | -0.4864 | 0.2568 |
| `mature` | `prox_a` | Proximity (Unweighted) | 1920 | -0.5157 | 0.2421 |
| `mature` | `prox_b` | Proximity (Unweighted) | 1920 | -0.4769 | 0.2615 |
| `mature` | `prox_ac` | Proximity (Unweighted) | 1920 | -0.5168 | 0.2416 |
| `mature` | `prox_bc` | Proximity (Unweighted) | 1920 | -0.4379 | 0.2811 |
| `mature` | `plcb` | Proximity (Unweighted) | 1920 | -0.5023 | 0.2488 |
| `mature` | `slcb` | Baseline (Variance) | 1920 | +0.1520 | 0.5760 |
| `mature` | `hutter_between` | Baseline (Tree Disagreement) | 1920 | +0.1303 | 0.5652 |
| `mature` | `hutter_within` | Baseline (Leaf Variance) | 1920 | +0.2187 | 0.6093 |
| `mature` | `shaker` | Baseline (Numerical Integration) | 1920 | +0.0861 | 0.5430 |
| `mature` | `shaker_total` | Baseline (Numerical Integration) | 1920 | +0.1148 | 0.5574 |
| `mature` | `shaker_mi` | Baseline (Information Theoretic) | 1920 | +0.0245 | 0.5122 |
| `mature` | `shaker_total_entropy` | Baseline (Information Theoretic) | 1920 | +0.1148 | 0.5574 |
| `mature` | `rf_fire` | Baseline (Volume Expansion) | 1920 | -0.5008 | 0.2496 |
| `shallow` | `prox_a` | Proximity (Unweighted) | 1920 | -0.5019 | 0.2490 |
| `shallow` | `prox_b` | Proximity (Unweighted) | 1920 | -0.4728 | 0.2636 |
| `shallow` | `prox_ac` | Proximity (Unweighted) | 1920 | -0.4317 | 0.2841 |
| `shallow` | `prox_bc` | Proximity (Unweighted) | 1920 | +0.4048 | 0.7024 |
| `shallow` | `plcb` | Proximity (Unweighted) | 1920 | -0.4928 | 0.2536 |
| `shallow` | `slcb` | Baseline (Variance) | 1920 | +0.0782 | 0.5391 |
| `shallow` | `hutter_between` | Baseline (Tree Disagreement) | 1920 | +0.2919 | 0.6460 |
| `shallow` | `hutter_within` | Baseline (Leaf Variance) | 1920 | -0.2108 | 0.3946 |
| `shallow` | `shaker` | Baseline (Numerical Integration) | 1920 | +0.3286 | 0.6643 |
| `shallow` | `shaker_total` | Baseline (Numerical Integration) | 1920 | +0.1097 | 0.5548 |
| `shallow` | `shaker_mi` | Baseline (Information Theoretic) | 1920 | +0.3784 | 0.6892 |
| `shallow` | `shaker_total_entropy` | Baseline (Information Theoretic) | 1920 | +0.1097 | 0.5548 |
| `shallow` | `rf_fire` | Baseline (Volume Expansion) | 1920 | -0.4964 | 0.2518 |
| `smac_default` | `prox_a` | Proximity (Unweighted) | 1920 | -0.3825 | 0.3088 |
| `smac_default` | `prox_b` | Proximity (Unweighted) | 1920 | -0.3563 | 0.3219 |
| `smac_default` | `prox_ac` | Proximity (Unweighted) | 1920 | -0.3914 | 0.3043 |
| `smac_default` | `prox_bc` | Proximity (Unweighted) | 1920 | -0.3390 | 0.3305 |
| `smac_default` | `plcb` | Proximity (Unweighted) | 1920 | -0.3824 | 0.3088 |
| `smac_default` | `slcb` | Baseline (Variance) | 1920 | +0.1265 | 0.5632 |
| `smac_default` | `hutter_between` | Baseline (Tree Disagreement) | 1920 | +0.1109 | 0.5555 |
| `smac_default` | `hutter_within` | Baseline (Leaf Variance) | 1920 | +0.1402 | 0.5701 |
| `smac_default` | `shaker` | Baseline (Numerical Integration) | 1920 | +0.0618 | 0.5309 |
| `smac_default` | `shaker_total` | Baseline (Numerical Integration) | 1920 | +0.0936 | 0.5468 |
| `smac_default` | `shaker_mi` | Baseline (Information Theoretic) | 1920 | +0.0147 | 0.5073 |
| `smac_default` | `shaker_total_entropy` | Baseline (Information Theoretic) | 1920 | +0.0936 | 0.5468 |
| `smac_default` | `rf_fire` | Baseline (Volume Expansion) | 1920 | -0.3919 | 0.3041 |