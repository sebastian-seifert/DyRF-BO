# Aleatoric Noise Masterplan Sweep Report: Shaker Entropy vs. Arithmetic Leaf Variance

**Total Records Evaluated**: 11215 runs across 270 benchmark target functions, 3 noise regimes, 5 RF hyperparameter configurations, and 5 seeds.

## 1. Grand Summary Performance Across All Experiments

| approach | spearman_true | spearman_resid | log_pearson_true | mse_var | rmse_var | nlpd_aleatoric |
| --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.6245 | 0.3329 | 0.3979 | 13.4878 | 2.1920 | 2262332.4502 |
| shaker_geom_std | 0.6245 | 0.3329 | 0.6001 | 0.0333 | 0.1581 | 13.0943 |
| shaker_geom_var | 0.6245 | 0.3329 | 0.6001 | 0.0173 | 0.0940 | 13314.5450 |
| standard_ari_std | 0.6193 | 0.3308 | 0.5947 | 0.0374 | 0.1708 | 13.1173 |
| standard_ari_var | 0.6193 | 0.3308 | 0.5947 | 0.0157 | 0.0884 | 13314.4714 |

## 2. Breakdown by Noise Regime (Spearman Rank Correlation vs. True Noise)

| approach | Overfit | RF | Smoothed |
| --- | --- | --- | --- |
| shaker_entropy | 0.0000 | 0.7462 | 0.8840 |
| shaker_geom_std | 0.0000 | 0.7462 | 0.8840 |
| shaker_geom_var | 0.0000 | 0.7462 | 0.8840 |
| standard_ari_std | 0.0000 | 0.7452 | 0.8610 |
| standard_ari_var | 0.0000 | 0.7452 | 0.8610 |

## 3. Breakdown by Target Function & Dimensionality (1D-15D)

| approach | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.7309 | 0.6074 | 0.5344 | 0.5687 | 0.6001 | 0.6209 | 0.6393 | 0.6394 | 0.6565 | 0.6342 | 0.6470 | 0.6220 | 0.6299 | 0.6251 | 0.6122 |
| shaker_geom_std | 0.7309 | 0.6074 | 0.5344 | 0.5687 | 0.6001 | 0.6209 | 0.6393 | 0.6394 | 0.6565 | 0.6342 | 0.6470 | 0.6220 | 0.6299 | 0.6251 | 0.6122 |
| shaker_geom_var | 0.7309 | 0.6074 | 0.5344 | 0.5687 | 0.6001 | 0.6209 | 0.6393 | 0.6394 | 0.6565 | 0.6342 | 0.6470 | 0.6220 | 0.6299 | 0.6251 | 0.6122 |
| standard_ari_std | 0.7347 | 0.6031 | 0.5226 | 0.5711 | 0.5942 | 0.6139 | 0.6341 | 0.6345 | 0.6525 | 0.6284 | 0.6415 | 0.6151 | 0.6219 | 0.6172 | 0.6051 |
| standard_ari_var | 0.7347 | 0.6031 | 0.5226 | 0.5711 | 0.5942 | 0.6139 | 0.6341 | 0.6345 | 0.6525 | 0.6284 | 0.6415 | 0.6151 | 0.6219 | 0.6172 | 0.6051 |

## 4. Breakdown by Random Forest Configuration

| approach | DeepEnsemble300 | Default | Leaf1 | Leaf15 | Shallow |
| --- | --- | --- | --- | --- | --- |
| shaker_entropy | 0.7310 | 0.8639 | 0.0000 | 0.8840 | 0.6439 |
| shaker_geom_std | 0.7310 | 0.8639 | 0.0000 | 0.8840 | 0.6439 |
| shaker_geom_var | 0.7310 | 0.8639 | 0.0000 | 0.8840 | 0.6439 |
| standard_ari_std | 0.7615 | 0.8274 | 0.0000 | 0.8610 | 0.6467 |
| standard_ari_var | 0.7615 | 0.8274 | 0.0000 | 0.8610 | 0.6467 |

## 5. Statistical Significance & Win/Tie/Loss Matrix vs. Standard Arithmetic Variance

| approach | baseline | mean_diff_spearman | p_value | w_statistic | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shaker_entropy | standard_ari_var | nan | nan | nan | 0 | 2243 | 0 |
| shaker_geom_var | standard_ari_var | nan | nan | nan | 0 | 2243 | 0 |
| shaker_geom_std | standard_ari_var | nan | nan | nan | 0 | 2243 | 0 |
| standard_ari_std | standard_ari_var | nan | nan | nan | 0 | 2243 | 0 |

