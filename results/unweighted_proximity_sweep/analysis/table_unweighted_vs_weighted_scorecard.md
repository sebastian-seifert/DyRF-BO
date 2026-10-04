# Unweighted vs Weighted Proximity Calibration Scorecard

## Executive Summary

Head-to-head empirical comparison of **Unweighted Proximity UQ** vs **Weighted RF-GAP (Legacy)**.
Evaluates Spearman rank correlation with Euclidean (d_norm) and Chebyshev (d_inf) projection distances,
difference in distance alignment Delta(rho), and normalized rank monotonicity ((1 + rho) / 2).

## Head-to-Head Calibration Scorecard Table

| Dimension Group | Comparison | N | Spearman d_norm (Unw) | Spearman d_norm (W) | Delta rho (Norm) | Spearman d_inf (Unw) | Spearman d_inf (W) | Delta rho (Inf) | Norm Rank Monotonicity (Unw) | Norm Rank Monotonicity (W) | W / T / L | Wilcoxon p | Cliff's Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Low-D (D <= 5)** | `prox_a_unweighted vs prox_a (weighted)` | 4800 | +0.1888 | +0.0514 | **+0.1373** | +0.1951 | +0.0611 | **+0.1340** | 0.5944 | 0.5257 | 3425/1/1374 | 3.8e-250 | +0.173 |
| **Low-D (D <= 5)** | `plcb_unweighted vs plcb (weighted)` | 4800 | +0.1888 | +0.0514 | **+0.1373** | +0.1951 | +0.0611 | **+0.1340** | 0.5944 | 0.5257 | 3425/1/1374 | 3.8e-250 | +0.173 |
| **High-D (D >= 16)** | `prox_a_unweighted vs prox_a (weighted)` | 3200 | +0.1887 | +0.1517 | **+0.0369** | +0.1927 | +0.1619 | **+0.0309** | 0.5943 | 0.5759 | 2009/0/1191 | 2.2e-41 | +0.091 |
| **High-D (D >= 16)** | `plcb_unweighted vs plcb (weighted)` | 3200 | +0.1887 | +0.1517 | **+0.0369** | +0.1927 | +0.1619 | **+0.0309** | 0.5943 | 0.5759 | 2009/0/1191 | 2.2e-41 | +0.091 |
| **All Dimensions** | `prox_a_unweighted vs prox_a (weighted)` | 9600 | +0.1845 | +0.0920 | **+0.0925** | +0.1902 | +0.1021 | **+0.0881** | 0.5923 | 0.5460 | 6529/1/3070 | 0.0e+00 | +0.137 |
| **All Dimensions** | `plcb_unweighted vs plcb (weighted)` | 9600 | +0.1845 | +0.0920 | **+0.0925** | +0.1902 | +0.1021 | **+0.0881** | 0.5923 | 0.5460 | 6529/1/3070 | 0.0e+00 | +0.137 |

## Statistical Methodology Notes
- **Normalized Rank Monotonicity**: Computed as `(1 + rho(d_norm, U)) / 2`, providing a normalized index in [0, 1] where 1.0 indicates perfect monotonic increase of uncertainty with distance.
- **Delta rho**: `rho(unweighted) - rho(weighted)`. Positive values indicate that unweighted proximity uncertainty exhibits superior monotonic distance alignment.
- **Paired Wilcoxon Signed-Rank Test**: Two-sided test testing whether the difference between paired unweighted and weighted correlations is significantly different from zero.
- **Cliff's Delta**: Non-parametric effect size in [-1, +1] where positive values favor the unweighted estimator.
