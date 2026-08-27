# Pairwise Wilcoxon Signed-Rank Tests vs. Standard SMAC3 Baseline (N=18 Tasks)

Reference Baseline: **`SMAC3_HPOFacade_ei`** (Mean Normalized Regret: **0.0472**)

| Paradigm | UQ Extractor | Mean Regret | Mean Diff vs Base | Rel Reduction (%) | Win / Loss / Tie | Wilcoxon W | p_raw | Cliff's delta | Holm-Bonferroni adj p | Significance (p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Additive Hybrid (EI) | proximity_bc | 0.0338 | -0.0134 | -28.4% | 10 / 6 / 2 | 31.0 | 0.0557 | -0.031 | 0.3900 | Borderline |
| Additive Hybrid (EI) | likelihood_credal | 0.0261 | -0.0210 | -44.6% | 9 / 7 / 2 | 40.0 | 0.1477 | -0.142 | 0.8860 | Non-significant |
| Additive Hybrid (EI) | standard_proximity | 0.0355 | -0.0117 | -24.7% | 9 / 7 / 2 | 41.0 | 0.1627 | +0.046 | 0.8860 | Non-significant |
| Additive Hybrid (EI) | proximity_b | 0.0331 | -0.0140 | -29.7% | 9 / 7 / 2 | 46.0 | 0.2553 | -0.037 | 1.0000 | Non-significant |
| Additive Hybrid (EI) | standard_disagreement | 0.0360 | -0.0112 | -23.8% | 9 / 8 / 1 | 58.0 | 0.3812 | -0.031 | 1.0000 | Non-significant |
| Additive Hybrid (EI) | proximity_auto_lambda | 0.0402 | -0.0069 | -14.7% | 8 / 9 / 1 | 61.0 | 0.4631 | +0.006 | 1.0000 | Non-significant |
| Additive Hybrid (EI) | shaker_entropy | 0.0368 | -0.0104 | -22.0% | 7 / 9 / 2 | 56.0 | 0.5349 | -0.028 | 1.0000 | Non-significant |
| Direct Replacement (EI) | standard_disagreement | 0.0402 | -0.0069 | -14.7% | 12 / 3 / 3 | 20.0 | 0.0231 | -0.059 | 0.1617 | Significant (*) |
| Direct Replacement (EI) | standard_proximity | 0.0326 | -0.0146 | -31.0% | 13 / 3 / 2 | 27.0 | 0.0340 | -0.117 | 0.2040 | Significant (*) |
| Direct Replacement (EI) | proximity_auto_lambda | 0.0410 | -0.0061 | -13.0% | 11 / 4 / 3 | 27.0 | 0.0609 | -0.052 | 0.3045 | Borderline |
| Direct Replacement (EI) | shaker_entropy | 0.0366 | -0.0106 | -22.4% | 10 / 5 / 3 | 34.0 | 0.1398 | -0.120 | 0.5590 | Non-significant |
| Direct Replacement (EI) | proximity_b | 0.0444 | -0.0028 | -5.8% | 11 / 4 / 3 | 36.0 | 0.1728 | -0.034 | 0.5590 | Non-significant |
| Direct Replacement (EI) | likelihood_credal | 0.0442 | -0.0030 | -6.3% | 11 / 5 / 2 | 48.0 | 0.3011 | -0.043 | 0.6021 | Non-significant |
| Direct Replacement (EI) | proximity_bc | 0.0425 | -0.0047 | -10.0% | 8 / 7 / 3 | 51.0 | 0.6092 | -0.022 | 0.6092 | Non-significant |
