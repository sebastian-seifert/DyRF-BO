# CARP-S Optimization Sweep Summary Report

This report summarizes the performance (minimum cost reached in 50 trials) for 7 dynamic Random Forest UQ approaches across 6 HPOBench tasks, aggregated across 5 seeds.

## Task: cfg_ml_svm_12

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| standard_proximity | 0.03266 | 0.00000 | 5/5 | 0.03266 |
| chen_variance | 0.03266 | 0.00000 | 4/5 | 0.03266 |
| proximity_b | 0.03266 | 0.00000 | 5/5 | 0.03266 |
| shaker_entropy | 0.03266 | 0.00000 | 5/5 | 0.03266 |
| proximity_bc | 0.03266 | 0.00000 | 5/5 | 0.03266 |
| standard_disagreement | 0.03274 | 0.00015 | 4/5 | 0.03266 |
| likelihood_credal | 0.03279 | 0.00027 | 5/5 | 0.03266 |

## Task: cfg_ml_svm_3

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| standard_proximity | 0.01453 | 0.00023 | 5/5 | 0.01411 |
| likelihood_credal | 0.01457 | 0.00039 | 5/5 | 0.01411 |
| shaker_entropy | 0.01460 | 0.00010 | 3/5 | 0.01453 |
| proximity_bc | 0.01461 | 0.00029 | 5/5 | 0.01411 |
| proximity_b | 0.01469 | 0.00016 | 5/5 | 0.01453 |
| chen_variance | 0.01474 | 0.00000 | 1/5 | 0.01474 |
| standard_disagreement | 0.02358 | 0.00000 | 1/5 | 0.02358 |

## Task: cfg_ml_svm_31

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| proximity_b | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| proximity_bc | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| standard_proximity | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| likelihood_credal | 0.25387 | 0.00000 | 5/5 | 0.25387 |
| shaker_entropy | 0.25428 | 0.00081 | 5/5 | 0.25387 |
| chen_variance | 0.25441 | 0.00108 | 5/5 | 0.25387 |
| standard_disagreement | 0.25468 | 0.00099 | 5/5 | 0.25387 |

## Task: cfg_ml_xgboost_12

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| chen_variance | 0.00889 | 0.00016 | 5/5 | 0.00875 |
| shaker_entropy | 0.00896 | 0.00016 | 5/5 | 0.00875 |
| standard_disagreement | 0.00896 | 0.00027 | 5/5 | 0.00875 |
| standard_proximity | 0.00896 | 0.00027 | 5/5 | 0.00875 |
| proximity_b | 0.00902 | 0.00013 | 5/5 | 0.00875 |
| likelihood_credal | 0.00902 | 0.00013 | 5/5 | 0.00875 |
| proximity_bc | 0.00902 | 0.00013 | 5/5 | 0.00875 |

## Task: cfg_ml_xgboost_3

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| standard_disagreement | 0.00227 | 0.00008 | 5/5 | 0.00211 |
| chen_variance | 0.00232 | 0.00000 | 5/5 | 0.00232 |
| proximity_bc | 0.00232 | 0.00000 | 5/5 | 0.00232 |
| standard_proximity | 0.00232 | 0.00000 | 5/5 | 0.00232 |
| shaker_entropy | 0.00232 | 0.00013 | 5/5 | 0.00211 |
| likelihood_credal | 0.00236 | 0.00008 | 5/5 | 0.00232 |
| proximity_b | 0.00236 | 0.00008 | 5/5 | 0.00232 |

## Task: cfg_ml_xgboost_31

| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |
| --- | --- | --- | --- | --- |
| standard_disagreement | 0.10034 | 0.00085 | 5/5 | 0.09899 |
| proximity_b | 0.10088 | 0.00188 | 5/5 | 0.09899 |
| chen_variance | 0.10088 | 0.00162 | 5/5 | 0.09899 |
| shaker_entropy | 0.10128 | 0.00193 | 5/5 | 0.09899 |
| standard_proximity | 0.10141 | 0.00215 | 5/5 | 0.09899 |
| proximity_bc | 0.10155 | 0.00131 | 5/5 | 0.09966 |
| likelihood_credal | 0.10168 | 0.00200 | 5/5 | 0.09899 |
