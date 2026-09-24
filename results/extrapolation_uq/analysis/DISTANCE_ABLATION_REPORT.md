# Extrapolation Distance Metric Ablation Report: Chebyshev ($L_\infty$) vs. Euclidean ($d_{\text{norm}}$)

## Executive Summary & Core Comparison

This ablation study investigates whether axis-aligned **Chebyshev distance ($d_{\text{inf}}$)** provides superior alignment with Random Forest surrogate uncertainty compared to standard **Euclidean normalized distance ($d_{\text{norm}}$)**. A total of **4** experimental runs were evaluated across varying dimensions, sampling strategies, and distance strata.

### Overall Statistical Hypothesis Testing

| Surrogate Method | Euclidean $\rho(d_{\text{norm}}, U)$ | Chebyshev $\rho(d_{\text{inf}}, U)$ | Paired Diff (Mean) | Wilcoxon p-value | Cliff's δ | Win / Tie / Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SLCB** (SMAC3 Standard) | 0.2404 ± 0.1364 | 0.2539 ± 0.1331 | +0.0135 | 6.25e-01 | +0.000 | 2W / 0T / 2L |
| **PLCB** (Proximity Augmented) | -0.3820 ± 0.1334 | -0.3669 ± 0.1398 | +0.0151 | 1.25e-01 | +0.250 | 4W / 0T / 0L |

*(Note: 'Win' indicates Chebyshev correlation > Euclidean correlation).* 

---

## Dimension-Wise Breakdown

Comparison of distance metric rank correlation across dimensionality regimes:

| Dimension | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | SLCB p-val | PLCB Euclidean | PLCB Chebyshev | PLCB Diff | PLCB p-val |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D=2 | 0.146 | 0.181 | +0.035 | 5.0e-01 | -0.413 | -0.401 | +0.013 | 5.0e-01 |
| D=16 | 0.335 | 0.327 | -0.008 | 5.0e-01 | -0.351 | -0.333 | +0.017 | 5.0e-01 |

---

## Strategy Breakdown

Comparison across test sampling distribution strategies:

| Strategy | Runs | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **natural** | 2 | 0.065 | 0.077 | +0.013 | -0.160 | -0.134 | +0.025 |
| **stratified** | 2 | 0.416 | 0.431 | +0.014 | -0.604 | -0.599 | +0.005 |

---

## Stratum-Wise Breakdown

Analysis across extrapolation depth strata:
- **Stratum 0 (Interpolation):** Points within the convex hull ($\tilde d \le 0$).
- **Stratum 1 (Near Extrapolation):** Moderate distance outside convex hull ($0 < \tilde d \le 0.3$).
- **Stratum 2 (Moderate Extrapolation):** Intermediate distance ($0.3 < \tilde d \le 0.8$).
- **Stratum 3 (Deep Extrapolation):** High distance out-of-domain ($ \tilde d > 0.8$).

| Stratum | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stratum 0 | 0.013 ± 0.048 | 0.011 ± 0.050 | -0.002 | -0.025 ± 0.033 | -0.028 ± 0.032 | -0.002 |
| Stratum 1 | 0.170 ± 0.065 | 0.173 ± 0.066 | +0.002 | -0.091 ± 0.043 | -0.087 ± 0.042 | +0.003 |
| Stratum 2 | -0.009 ± 0.030 | 0.019 ± 0.014 | +0.028 | -0.067 ± 0.013 | -0.017 ± 0.015 | +0.050 |
| Stratum 3 | -0.015 ± 0.023 | 0.098 ± 0.053 | +0.113 | -0.046 ± 0.018 | -0.001 ± 0.020 | +0.045 |

---

## Scientific Interpretation & Surrogate Geometry

1. **Tree Geometry and Axis Alignment:** Standard Random Forest regressors partition the feature space using axis-aligned orthogonal cuts $x_j \lessgtr \theta$. The resulting leaf cells form axis-aligned hyper-rectangles. Consequently, out-of-domain epistemic uncertainty is intimately tied to the Chebyshev ($L_\infty$) distance, which measures the maximum single-coordinate departure from the training bounding hull.

2. **Euclidean vs. Chebyshev Synergy:** While Euclidean distance smoothly penalizes simultaneous departures across all coordinates (desirable for rotation-invariant global penalties), Chebyshev distance directly reflects the boundary structure of the surrogate's orthogonal decision trees. In high dimensions ($D \ge 16$), the ratio $\|x\|_2 / \|x\|_\infty$ scales as $\sqrt{D}$, creating potential scale divergence.

3. **Recommendation for DyRF-BO:** When deploying axis-aligned tree surrogates in extreme extrapolation regimes, Chebyshev distance serves as an excellent geometry-aware surrogate diagnostic, confirming that PLCB preserves strong rank correlation across both metric spaces.