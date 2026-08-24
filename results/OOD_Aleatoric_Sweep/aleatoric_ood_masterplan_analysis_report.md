# Aleatoric OOD Masterplan Sweep Report: In-Distribution vs. Out-of-Distribution Analysis

**Total Records Evaluated**: 15750 runs across 15 benchmark target functions, 7 noise regimes, 5 RF configs, and 5 seeds.

## 1. Grand Summary: In-Distribution (ID) vs. Out-of-Distribution (OOD) Scopes

| approach | id_only_spearman_true | ood_only_spearman_true | id_only_mse_var | ood_only_mse_var | id_only_nlpd_aleatoric | ood_only_nlpd_aleatoric | ood_id_variance_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.4394 | 0.0222 | 3.7628 | 3.9000 | 1616171.3601 | 2879599.8884 | -106400557.7041 |
| shaker_geom_std | 0.4394 | 0.0222 | 0.0300 | 0.0472 | 0.4985 | 1.0683 | 1.0389 |
| shaker_geom_var | 0.4394 | 0.0222 | 0.0058 | 0.0464 | 27.0218 | 54.7622 | 1.1002 |
| standard_ari_std | 0.4585 | 0.0255 | 0.0344 | 0.0377 | 0.3079 | 0.4686 | 1.0500 |
| standard_ari_var | 0.4585 | 0.0255 | 0.0029 | 0.0334 | 0.0082 | 0.7866 | 1.1260 |
| standard_disagreement | 0.3896 | 0.0193 | 0.0093 | 0.0531 | 27.0192 | 41.1647 | 1.1320 |

## 2. OOD / ID Variance Ratio (Epistemic Explosion vs. Aleatoric Extrapolation)

| approach | ratio_mean | ratio_std |
| --- | --- | --- |
| shaker_entropy | -106400557.7041 | 156110873.8625 |
| shaker_geom_std | 1.0389 | 0.3121 |
| shaker_geom_var | 1.1002 | 0.6896 |
| standard_ari_std | 1.0500 | 0.3331 |
| standard_ari_var | 1.1260 | 0.7393 |
| standard_disagreement | 1.1320 | 0.9768 |

## 3. Breakdown by Noise Regime (Spearman Rank Correlation)

### ID-Only Spearman Rank Correlation vs. True Noise:

| approach | hetero_linear | hetero_localized | hetero_ood_step_double | hetero_quadratic | hetero_sinusoidal | homoscedastic_high | homoscedastic_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.8763 | 0.7989 | 0.0000 | 0.8186 | 0.5817 | 0.0000 | 0.0000 |
| shaker_geom_std | 0.8763 | 0.7989 | 0.0000 | 0.8186 | 0.5817 | 0.0000 | 0.0000 |
| shaker_geom_var | 0.8763 | 0.7989 | 0.0000 | 0.8186 | 0.5817 | 0.0000 | 0.0000 |
| standard_ari_std | 0.9127 | 0.8125 | 0.0000 | 0.8629 | 0.6214 | 0.0000 | 0.0000 |
| standard_ari_var | 0.9127 | 0.8125 | 0.0000 | 0.8629 | 0.6214 | 0.0000 | 0.0000 |
| standard_disagreement | 0.8305 | 0.6962 | 0.0000 | 0.7759 | 0.4249 | 0.0000 | 0.0000 |

### OOD-Only Spearman Rank Correlation vs. True Noise:

| approach | hetero_linear | hetero_localized | hetero_ood_step_double | hetero_quadratic | hetero_sinusoidal | homoscedastic_high | homoscedastic_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.1490 | -0.0068 | 0.0000 | 0.0051 | 0.0083 | 0.0000 | 0.0000 |
| shaker_geom_std | 0.1490 | -0.0068 | 0.0000 | 0.0051 | 0.0083 | 0.0000 | 0.0000 |
| shaker_geom_var | 0.1490 | -0.0068 | 0.0000 | 0.0051 | 0.0083 | 0.0000 | 0.0000 |
| standard_ari_std | 0.1775 | -0.0018 | 0.0000 | -0.0014 | 0.0043 | 0.0000 | 0.0000 |
| standard_ari_var | 0.1775 | -0.0018 | 0.0000 | -0.0014 | 0.0043 | 0.0000 | 0.0000 |
| standard_disagreement | 0.1332 | 0.0008 | 0.0000 | -0.0008 | 0.0017 | 0.0000 | 0.0000 |

## 4. Breakdown by Random Forest Configuration

| approach | RF_DeepEnsemble300 | RF_Default | RF_Overfit_Leaf1 | RF_Shallow | RF_Smoothed_Leaf15 |
| --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.4836 | 0.5064 | 0.3288 | 0.3674 | 0.5106 |
| shaker_geom_std | 0.4836 | 0.5064 | 0.3288 | 0.3674 | 0.5106 |
| shaker_geom_var | 0.4836 | 0.5064 | 0.3288 | 0.3674 | 0.5106 |
| standard_ari_std | 0.4826 | 0.4954 | 0.4488 | 0.3635 | 0.5022 |
| standard_ari_var | 0.4826 | 0.4954 | 0.4488 | 0.3635 | 0.5022 |
| standard_disagreement | 0.3764 | 0.4667 | 0.4371 | 0.2172 | 0.4509 |

## 5. Statistical Significance & Win/Tie/Loss Matrix vs. Standard Arithmetic Variance

| approach | baseline | metric | mean_diff | p_value | w_statistic | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | standard_ari_var | id_only_spearman_true | -0.0191 | 0.3602 | 547524.0000 | 0 | 2625 | 0 |
| shaker_entropy | standard_ari_var | ood_only_spearman_true | -0.0033 | 0.0194 | 456980.0000 | 652 | 1232 | 741 |
| shaker_entropy | standard_ari_var | ood_id_variance_ratio | -106400558.8301 | 0.0000 | 470225.0000 | 710 | 27 | 1888 |
| shaker_geom_var | standard_ari_var | id_only_spearman_true | -0.0191 | 0.3602 | 547524.0000 | 0 | 2625 | 0 |
| shaker_geom_var | standard_ari_var | ood_only_spearman_true | -0.0033 | 0.0194 | 456980.0000 | 652 | 1232 | 741 |
| shaker_geom_var | standard_ari_var | ood_id_variance_ratio | -0.0258 | 0.0000 | 1520101.0000 | 1576 | 79 | 970 |
| shaker_geom_std | standard_ari_var | id_only_spearman_true | -0.0191 | 0.3602 | 547524.0000 | 0 | 2625 | 0 |
| shaker_geom_std | standard_ari_var | ood_only_spearman_true | -0.0033 | 0.0194 | 456980.0000 | 652 | 1232 | 741 |
| shaker_geom_std | standard_ari_var | ood_id_variance_ratio | -0.0870 | 0.0000 | 1310179.0000 | 1667 | 61 | 897 |
| standard_ari_std | standard_ari_var | id_only_spearman_true | 0.0000 | 1.0000 | 0.0000 | 0 | 2625 | 0 |
| standard_ari_std | standard_ari_var | ood_only_spearman_true | 0.0000 | 1.0000 | 0.0000 | 0 | 2625 | 0 |
| standard_ari_std | standard_ari_var | ood_id_variance_ratio | -0.0760 | 0.0000 | 1361980.0000 | 1565 | 49 | 1011 |
| standard_disagreement | standard_ari_var | id_only_spearman_true | -0.0689 | 0.0000 | 4994.0000 | 24 | 1126 | 1475 |
| standard_disagreement | standard_ari_var | ood_only_spearman_true | -0.0062 | 0.0000 | 419104.5000 | 613 | 1207 | 805 |
| standard_disagreement | standard_ari_var | ood_id_variance_ratio | 0.0060 | 0.0000 | 1186263.0000 | 841 | 2 | 1782 |

