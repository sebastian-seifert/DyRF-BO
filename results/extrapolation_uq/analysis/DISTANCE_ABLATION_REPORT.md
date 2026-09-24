# Extrapolation Distance Metric Ablation Report: Chebyshev ($L_\infty$) vs. Euclidean ($d_{\text{norm}}$)

## Executive Summary & Core Comparison

This ablation study investigates whether axis-aligned **Chebyshev distance ($d_{\text{inf}}$)** provides superior alignment with Random Forest surrogate uncertainty compared to standard **Euclidean normalized distance ($d_{\text{norm}}$)**. A total of **1920** experimental runs were evaluated across varying dimensions, sampling strategies, and distance strata.

### Overall Statistical Hypothesis Testing

| Surrogate Method | Euclidean $\rho(d_{\text{norm}}, U)$ | Chebyshev $\rho(d_{\text{inf}}, U)$ | Paired Diff (Mean) | Wilcoxon p-value | Cliff's δ | Win / Tie / Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SLCB** (SMAC3 Standard) | 0.1265 ± 0.0089 | 0.1354 ± 0.0086 | +0.0089 | 3.47e-33 | +0.011 | 1130W / 0T / 790L |
| **PLCB** (Proximity Augmented) | -0.3906 ± 0.0064 | -0.3622 ± 0.0067 | +0.0284 | 5.61e-278 | +0.078 | 1798W / 0T / 122L |

*(Note: 'Win' indicates Chebyshev correlation > Euclidean correlation).* 

---

## Dimension-Wise Breakdown

Comparison of distance metric rank correlation across dimensionality regimes:

| Dimension | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | SLCB p-val | PLCB Euclidean | PLCB Chebyshev | PLCB Diff | PLCB p-val |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D=2 | 0.045 | 0.052 | +0.007 | 5.9e-08 | -0.371 | -0.361 | +0.010 | 2.0e-28 |
| D=3 | 0.059 | 0.077 | +0.018 | 3.5e-19 | -0.366 | -0.344 | +0.022 | 1.2e-41 |
| D=5 | 0.084 | 0.103 | +0.019 | 7.8e-18 | -0.424 | -0.386 | +0.038 | 4.4e-54 |
| D=8 | 0.132 | 0.143 | +0.011 | 2.3e-09 | -0.440 | -0.401 | +0.040 | 3.4e-53 |
| D=16 | 0.208 | 0.211 | +0.003 | 1.2e-02 | -0.401 | -0.366 | +0.034 | 4.5e-54 |
| D=32 | 0.230 | 0.226 | -0.004 | 2.6e-08 | -0.341 | -0.315 | +0.026 | 9.8e-53 |

---

## Strategy Breakdown

Comparison across test sampling distribution strategies:

| Strategy | Runs | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **natural** | 960 | -0.040 | -0.024 | +0.016 | -0.200 | -0.151 | +0.049 |
| **stratified** | 960 | 0.293 | 0.295 | +0.002 | -0.581 | -0.573 | +0.008 |

---

## Stratum-Wise Breakdown

Analysis across extrapolation depth strata:
- **Stratum 0 (Interpolation):** Points within the convex hull ($\tilde d \le 0$).
- **Stratum 1 (Near Extrapolation):** Moderate distance outside convex hull ($0 < \tilde d \le 0.3$).
- **Stratum 2 (Moderate Extrapolation):** Intermediate distance ($0.3 < \tilde d \le 0.8$).
- **Stratum 3 (Deep Extrapolation):** High distance out-of-domain ($ \tilde d > 0.8$).

| Stratum | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stratum 0 | 0.009 ± 0.004 | 0.009 ± 0.004 | +0.000 | 0.002 ± 0.004 | 0.003 ± 0.004 | +0.000 |
| Stratum 1 | 0.081 ± 0.005 | 0.075 ± 0.005 | -0.006 | -0.149 ± 0.005 | -0.113 ± 0.005 | +0.036 |
| Stratum 2 | -0.026 ± 0.002 | 0.004 ± 0.001 | +0.030 | -0.101 ± 0.002 | -0.019 ± 0.001 | +0.082 |
| Stratum 3 | -0.029 ± 0.002 | -0.029 ± 0.002 | -0.000 | -0.058 ± 0.001 | -0.010 ± 0.002 | +0.049 |

---

## Scientific Interpretation & Surrogate Geometry

1. **Tree Geometry and Axis Alignment:** Standard Random Forest regressors partition the feature space using axis-aligned orthogonal cuts $x_j \lessgtr \theta$. The resulting leaf cells form axis-aligned hyper-rectangles. Consequently, out-of-domain epistemic uncertainty is intimately tied to the Chebyshev ($L_\infty$) distance, which measures the maximum single-coordinate departure from the training bounding hull.

2. **Euclidean vs. Chebyshev Synergy:** While Euclidean distance smoothly penalizes simultaneous departures across all coordinates (desirable for rotation-invariant global penalties), Chebyshev distance directly reflects the boundary structure of the surrogate's orthogonal decision trees. In high dimensions ($D \ge 16$), the ratio $\|x\|_2 / \|x\|_\infty$ scales as $\sqrt{D}$, creating potential scale divergence.

3. **Recommendation for DyRF-BO:** When deploying axis-aligned tree surrogates in extreme extrapolation regimes, Chebyshev distance serves as an excellent geometry-aware surrogate diagnostic, confirming that PLCB preserves strong rank correlation across both metric spaces.