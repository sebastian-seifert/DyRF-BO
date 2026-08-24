# Aleatoric OOD Masterplan Sweep Report: In-Distribution vs. Out-of-Distribution Analysis

**Total Records Evaluated**: 24 runs across 2 benchmark target functions, 2 noise regimes, 1 RF configs, and 1 seeds.

## 1. Grand Summary: Global vs. ID vs. OOD Scope Performance

| approach | id_only_spearman_true | ood_only_spearman_true | id_only_mse_var | ood_only_mse_var | id_only_nlpd_aleatoric | ood_only_nlpd_aleatoric | ood_id_variance_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.3535 | -0.0055 | 0.9327 | 0.3542 | 1424206.5825 | 5840160.0625 | -40480332.7240 |
| shaker_geom_std | 0.3535 | -0.0055 | 0.0204 | 0.0298 | 0.0817 | 0.4706 | 1.2043 |
| shaker_geom_var | 0.3535 | -0.0055 | 0.0003 | 0.0010 | -0.2295 | 1.7075 | 1.4054 |
| standard_ari_std | 0.3592 | -0.0079 | 0.0234 | 0.0332 | 0.1037 | 0.4732 | 1.1934 |
| standard_ari_var | 0.3592 | -0.0079 | 0.0004 | 0.0012 | -0.2543 | 1.4907 | 1.3796 |
| standard_disagreement | 0.2762 | -0.0255 | 0.0009 | 0.0006 | 1.5445 | 9.7180 | 1.1304 |

## 2. OOD / ID Variance Ratio (Epistemic Explosion vs. Aleatoric Extrapolation)

| approach | ratio_mean | ratio_std |
| --- | --- | --- |
| shaker_entropy | -40480332.7240 | 38643789.4633 |
| shaker_geom_std | 1.2043 | 0.1811 |
| shaker_geom_var | 1.4054 | 0.4419 |
| standard_ari_std | 1.1934 | 0.1687 |
| standard_ari_var | 1.3796 | 0.4163 |
| standard_disagreement | 1.1304 | 0.2892 |

