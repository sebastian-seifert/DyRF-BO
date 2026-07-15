# CARP-S Optimization Sweep Summary Report

This report summarizes the performance (minimum cost reached in 50 trials) for 7 dynamic Random Forest UQ approaches across 6 HPOBench tasks, aggregated across 5 seeds.

## Task: cfg_ml_svm_12

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| proximity_b | 0.03286 | 0.00027 | 5/5 | 0.03266 |
| shaker_entropy | 0.03300 | 0.00041 | 4/5 | 0.03266 |
| proximity_bc | 0.03387 | 0.00226 | 5/5 | 0.03266 |
| standard_proximity | 0.03455 | 0.00218 | 5/5 | 0.03266 |
| standard_disagreement | 0.03529 | 0.00281 | 5/5 | 0.03300 |
| chen_variance | 0.03589 | 0.00297 | 5/5 | 0.03333 |
| likelihood_credal | 0.03657 | 0.00352 | 5/5 | 0.03266 |

## Task: cfg_ml_svm_3

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| proximity_bc | 0.01457 | 0.00028 | 5/5 | 0.01411 |
| shaker_entropy | 0.01469 | 0.00016 | 5/5 | 0.01453 |
| proximity_b | 0.01478 | 0.00025 | 5/5 | 0.01453 |
| standard_proximity | 0.01649 | 0.00263 | 3/5 | 0.01453 |
| likelihood_credal | 0.01811 | 0.00337 | 2/5 | 0.01474 |
| standard_disagreement | 0.01987 | 0.00723 | 5/5 | 0.01453 |
| chen_variance | 0.02029 | 0.00736 | 5/5 | 0.01453 |

## Task: cfg_ml_svm_31

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| likelihood_credal | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| standard_disagreement | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| chen_variance | 0.25441 | 0.00108 | 5/5 | 0.25387 |
| proximity_b | 0.25441 | 0.00108 | 5/5 | 0.25387 |
| shaker_entropy | 0.25567 | 0.00254 | 3/5 | 0.25387 |
| standard_proximity | 0.25670 | 0.00471 | 5/5 | 0.25387 |
| proximity_bc | 0.25684 | 0.00348 | 5/5 | 0.25387 |

## Task: cfg_ml_xgboost_12

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| likelihood_credal | 0.00889 | 0.00016 | 5/5 | 0.00875 |
| chen_variance | 0.00902 | 0.00013 | 5/5 | 0.00875 |
| proximity_b | 0.00902 | 0.00013 | 5/5 | 0.00875 |
| shaker_entropy | 0.00902 | 0.00013 | 5/5 | 0.00875 |
| standard_disagreement | 0.00909 | 0.00021 | 5/5 | 0.00875 |
| proximity_bc | 0.00916 | 0.00025 | 5/5 | 0.00875 |
| standard_proximity | 0.00916 | 0.00025 | 5/5 | 0.00875 |

## Task: cfg_ml_xgboost_3

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| proximity_bc | 0.00232 | 0.00013 | 5/5 | 0.00211 |
| standard_proximity | 0.00232 | 0.00000 | 5/5 | 0.00232 |
| chen_variance | 0.00236 | 0.00008 | 5/5 | 0.00232 |
| proximity_b | 0.00236 | 0.00008 | 5/5 | 0.00232 |
| likelihood_credal | 0.00240 | 0.00010 | 5/5 | 0.00232 |
| shaker_entropy | 0.00240 | 0.00010 | 5/5 | 0.00232 |
| standard_disagreement | 0.00240 | 0.00010 | 5/5 | 0.00232 |

## Task: cfg_ml_xgboost_31

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| standard_proximity | 0.10088 | 0.00066 | 5/5 | 0.09966 |
| proximity_bc | 0.10101 | 0.00085 | 5/5 | 0.09966 |
| proximity_b | 0.10141 | 0.00101 | 5/5 | 0.09966 |
| likelihood_credal | 0.10182 | 0.00178 | 5/5 | 0.09966 |
| standard_disagreement | 0.10209 | 0.00179 | 5/5 | 0.09966 |
| chen_variance | 0.10249 | 0.00162 | 5/5 | 0.09966 |
| shaker_entropy | 0.10263 | 0.00243 | 5/5 | 0.09966 |
