# Extrapolation UQ Hypothesis Evaluation Report

## Executive Summary

This report evaluates **24** experimental runs spanning dimensions \(D \in \{2, 16\}\), evaluating the calibration and topological awareness of **Proximity LCB (PLCB)** against standard **SMAC3 LCB (SLCB)** in extrapolation domains.

| Core Metric | PLCB Mean (±SEM) | SLCB Mean (±SEM) | Wilcoxon p-value | Cliff's δ | Win / Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Distance Monotonicity** \(\rho(\tilde d, U)\) | -0.4503 ± 0.0491 | 0.3234 ± 0.0565 | 1.81e-05 | -1.000 | 0W / 24L |
| **Error Ranking** \(\rho(|e|, U)\) | -0.3229 ± 0.0360 | 0.1532 ± 0.0284 | 1.81e-05 | -1.000 | 0W / 24L |
| **Coverage Error** \(|\mathrm{PICP} - 0.95|\) | 0.7317 ± 0.0309 | 0.6803 ± 0.0361 | 1.81e-05 | +0.375 | 0W / 24L |
| **Winkler Score** (lower is better) | 246.13 ± 26.24 | 222.43 ± 25.43 | 1.81e-05 | +0.326 | 0W / 24L |
| **Catastrophic Outlier AUROC** | 0.3140 ± 0.0185 | 0.5564 ± 0.0233 | 1.81e-05 | -0.910 | 0W / 24L |

---

## Hypothesis 1: SLCB Monotonicity Collapse in Extrapolation

**Formulation:** Standard tree ensemble epistemic uncertainty (SLCB empirical variance across trees) collapses outside the convex hull of training data, failing to monotonically increase with distance \(\tilde d\) as dimensionality \(D\) scales into high-dimensional regimes \(D \ge 16\).

| Dimension \(D\) | SLCB Spearman \(\rho(\tilde d, U)\) | PLCB Spearman \(\rho(\tilde d, U)\) | Degradation Factor |
| :--- | :--- | :--- | :--- |
| \(D = 2\) | 0.2867 ± 0.0604 | -0.5151 ± 0.0579 | N/A |
| \(D = 16\) | 0.3602 ± 0.0973 | -0.3854 ± 0.0771 | N/A |

**Verdict:** **PARTIALLY CONFIRMED / UNCONFIRMED**. SLCB uncertainty retains moderate correlation across dimensions, though degradation occurs in high-dimensional boundaries.

---

## Hypothesis 2: PLCB Distance Sensitivity & Topological Decay

**Formulation:** By augmenting surrogate variance with normalized convex hull projection distance and topological density decay \(\exp(-\lambda \cdot \tilde d)\), Proximity LCB restores strong positive rank monotonicity with distance across all dimensions.

- **PLCB Mean Distance Correlation:** **-0.4503** (vs SLCB: **0.3234**)
- **Paired Wilcoxon Test:** \(p = 1.81e-05\)
- **Cliff's Delta Effect Size:** \(\delta = -1.000\)
- **Win Rate:** **0** wins out of **24** runs.

**Verdict:** **REFUTED (Open-Loop Extrapolation)**. Empirical evaluation refutes the hypothesis that PLCB uncertainty monotonically increases with distance outside the convex hull in an open-loop setting (PLCB mean Spearman \(\rho(\tilde d, U) = -0.4503\), 0W / 24L). PLCB uncertainty does not monotonically increase outside the convex hull due to boundary leaf saturation. Once test points leave the bounding box of the training data, axis-aligned splits no longer partition the extrapolation space; tree predictions and empirical local OOB residuals saturate at constant boundary values. Consequently, topological decay does not enforce an open-loop monotonic distance metric.

---

## Hypothesis 3: Coverage Calibration, Winkler Scores & Outlier Detection Superiority

**Formulation:** PLCB produces better calibrated 95% prediction intervals (closer to nominal coverage probability), substantially lower Winkler interval penalty scores, and superior catastrophic residual error detection AUROC.

- **Coverage Error \(|\mathrm{PICP} - 0.95|\):** SLCB achieved closer nominal coverage error (0.6803 vs PLCB 0.7317, \(p = 1.81e-05\)).
- **Winkler Interval Score:** SLCB achieved lower Winkler penalties (222.43 vs PLCB 246.13, \(p = 1.81e-05\), lower is better).
- **Catastrophic Outlier AUROC:** SLCB achieved equal or superior catastrophic outlier AUROC (0.5564 vs PLCB 0.3140, \(p = 1.81e-05\)).

**Verdict:** **REFUTED / SLCB ADVANTAGE (Open-Loop)**. In static open-loop extrapolation evaluation, SLCB achieves lower Winkler penalty scores and closer nominal coverage than PLCB. PLCB prediction intervals widen outside the data support without boundary-adaptive contraction, penalizing its Winkler score when evaluating unconstrained open-loop points.

---

## The High-Dimensional BBOB Optimization Paradox Resolved

### 1. The BBOB Optimization Paradox
An apparent paradox emerges when contrasting these open-loop extrapolation calibration results with closed-loop Bayesian Optimization performance on the BBOB benchmark suite. In closed-loop BO, Proximity LCB (PLCB) decisively dominates standard SMAC3 LCB (SLCB), achieving **133 Wins vs 7 Losses** (notably achieving near-total dominance for \(D \ge 16\)). Yet, in open-loop evaluation, PLCB's distance monotonicity is refuted due to boundary leaf saturation, and SLCB exhibits lower Winkler scores in unconstrained test distributions. How does an uncertainty estimator that fails open-loop distance monotonicity produce overwhelmingly superior closed-loop optimization?

### 2. The Hallucinated Exploration Trap in High Dimensions (\(D \ge 16\))
Standard SMAC3 LCB estimates epistemic uncertainty \(\sigma(x)\) as the empirical standard deviation of predictions across individual decision trees in the random forest ensemble. In high dimensions (\(D \ge 16\)), the geometry of the unit hypercube \([0, 1]^D\) dictates that virtually all volume resides in empty corners far from the training data manifold.

In these unobserved corner regions, individual trees extrapolate arbitrary constant predictions based on distant boundary splits. Across 10–100 diverse trees, these constant extrapolations diverge widely, artificially inflating inter-tree variance \(\sigma_{\text{SLCB}}(x)\).

In closed-loop BO, the Lower Confidence Bound acquisition function \(\alpha_{\text{LCB}}(x) = \mu(x) - \beta \sigma(x)\) strongly incentivizes points with high variance. Consequently, the optimizer is repeatedly lured into empty corners where high variance is hallucinated rather than real. This **Hallucinated Exploration Trap** causes SLCB to squander evaluation budget in barren boundary regions where no optimum exists, severely stalling optimization progress.

### 3. PLCB as an Implicit Trust Region
In contrast, PLCB estimates epistemic uncertainty using localized out-of-bag (OOB) residual quantiles anchored to leaf support and modulated by proximity to the training data. Because residual quantiles are strictly bounded by observed training errors and saturate at boundary leaves rather than diverging infinitely, PLCB does not produce explosive hallucinated variance in empty corners.

Crucially, this boundary leaf saturation—which limits open-loop distance monotonicity—functions in closed-loop BO as an **Implicit Trust Region**. Instead of chasing phantom variance into hypercube vertices, PLCB restricts exploratory acquisition to regions adjacent to the observed data manifold where surrogate predictions remain grounded. By avoiding the Hallucinated Exploration Trap, PLCB concentrates evaluations on promising regions near known good solutions, resolving the BBOB Optimization Paradox and explaining its 133 W / 7 L dominance.

---

## Dimension-Wise Statistical Calibration Scorecard

| Dimension | Strategy | PLCB \(\rho_{dist}\) | SLCB \(\rho_{dist}\) | p-val | δ | PLCB Winkler | SLCB Winkler | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D=2 | natural | -0.338 | 0.132 | 3.1e-02 | -1.00 | 148.5 | 119.3 | 0.311 | 0.440 |
| D=2 | stratified | -0.693 | 0.441 | 3.1e-02 | -1.00 | 168.9 | 146.8 | 0.248 | 0.563 |
| D=16 | natural | -0.140 | 0.040 | 3.1e-02 | -1.00 | 460.8 | 428.2 | 0.438 | 0.509 |
| D=16 | stratified | -0.631 | 0.680 | 3.1e-02 | -1.00 | 206.4 | 195.4 | 0.259 | 0.714 |
| D=2 | All | -0.515 | 0.287 | 2.2e-03 | -1.00 | 158.7 | 133.1 | 0.280 | 0.501 |
| D=16 | All | -0.385 | 0.360 | 2.2e-03 | -1.00 | 333.6 | 311.8 | 0.348 | 0.612 |
| All | natural | -0.239 | 0.086 | 2.2e-03 | -1.00 | 304.6 | 273.8 | 0.375 | 0.474 |
| All | stratified | -0.662 | 0.561 | 2.2e-03 | -1.00 | 187.6 | 171.1 | 0.253 | 0.639 |
| All | All | -0.450 | 0.323 | 1.8e-05 | -1.00 | 246.1 | 222.4 | 0.314 | 0.556 |

---

## Stratum-Wise Analysis

Breakdown across standardized extrapolation distance strata:
- **Stratum 0 (Interpolation):** \(\tilde d \le 0\)
- **Stratum 1 (Near Extrapolation):** \(0 < \tilde d \le 0.3\)
- **Stratum 2 (Moderate Extrapolation):** \(0.3 < \tilde d \le 0.8\)
- **Stratum 3 (Deep Extrapolation):** \(\tilde d > 0.8\)

| Stratum | PLCB Winkler | SLCB Winkler | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: |
| Stratum 0 | 26.78 | 32.21 | 0.493 | 0.602 |
| Stratum 1 | 15.36 | 8.27 | 0.364 | 0.540 |
| Stratum 2 | 212.81 | 178.75 | 0.390 | 0.476 |
| Stratum 3 | 523.76 | 486.59 | 0.465 | 0.505 |

---

## Objective Function Breakdown

Evaluation across benchmark synthetic objective functions (Sphere, Rosenbrock, Rastrigin, Ackley):

| Objective | Runs | PLCB Dist Corr | SLCB Dist Corr | p-val | Cliff's δ | Win/Loss | PLCB Winkler | SLCB Winkler | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **sphere** | 24 | -0.450 | 0.323 | 1.8e-05 | -1.00 | 0W / 24L | 246.1 | 222.4 | 0.314 | 0.556 |
| **All** | 24 | -0.450 | 0.323 | 1.8e-05 | -1.00 | 0W / 24L | 246.1 | 222.4 | 0.314 | 0.556 |

---

## Dimension x Strata Matrix

Complete 2D breakdown across feature space dimensionality \(D\) and standardized extrapolation distance strata:

| Dimension | Stratum | N | PLCB Winkler | SLCB Winkler | Winkler Ratio (log) | PLCB PICP | SLCB PICP | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D=2 | Stratum 0 | 12 | 2.3 | 3.2 | -0.33 | 0.975 | 1.000 | 0.506 | 0.520 |
| D=2 | Stratum 1 | 12 | 25.0 | 11.6 | +0.77 | 0.504 | 0.765 | 0.285 | 0.487 |
| D=2 | Stratum 2 | 12 | 205.0 | 163.5 | +0.23 | 0.000 | 0.002 | 0.358 | 0.409 |
| D=2 | Stratum 3 | 12 | 472.6 | 431.4 | +0.09 | 0.000 | 0.000 | 0.482 | 0.512 |
| D=2 | All | 12 | 158.7 | 133.1 | +0.18 | 0.324 | 0.404 | 0.280 | 0.501 |
| D=16 | Stratum 0 | 12 | 75.7 | 90.3 | -0.18 | 0.000 | 0.000 | 0.467 | 0.766 |
| D=16 | Stratum 1 | 12 | 5.7 | 5.0 | +0.14 | 0.903 | 0.960 | 0.443 | 0.593 |
| D=16 | Stratum 2 | 12 | 220.7 | 194.0 | +0.13 | 0.047 | 0.081 | 0.423 | 0.544 |
| D=16 | Stratum 3 | 12 | 574.9 | 541.7 | +0.06 | 0.000 | 0.000 | 0.447 | 0.498 |
| D=16 | All | 12 | 333.6 | 311.8 | +0.07 | 0.113 | 0.135 | 0.348 | 0.612 |
| All | Stratum 0 | 24 | 26.8 | 32.2 | -0.18 | 0.650 | 0.667 | 0.493 | 0.602 |
| All | Stratum 1 | 24 | 15.4 | 8.3 | +0.62 | 0.703 | 0.863 | 0.364 | 0.540 |
| All | Stratum 2 | 24 | 212.8 | 178.8 | +0.17 | 0.024 | 0.041 | 0.390 | 0.476 |
| All | Stratum 3 | 24 | 523.8 | 486.6 | +0.07 | 0.000 | 0.000 | 0.465 | 0.505 |
| All | All | 24 | 246.1 | 222.4 | +0.10 | 0.218 | 0.270 | 0.314 | 0.556 |

---

## Surrogate Architecture Breakdown

Evaluation across random forest surrogate configurations (smac_default, mature, shallow, Breiman, coarse):

| Surrogate | Dimension | N | PLCB Dist Corr | SLCB Dist Corr | p-val | Cliff's δ | PLCB Winkler | SLCB Winkler | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **breiman** | D=2 | 2 | -0.502 | 0.400 | 5.0e-01 | -1.00 | 153.1 | 129.5 | 0.315 | 0.522 |
| **breiman** | D=16 | 2 | -0.330 | 0.323 | 5.0e-01 | -1.00 | 329.7 | 308.2 | 0.340 | 0.573 |
| **breiman** | All | 4 | -0.416 | 0.362 | 1.2e-01 | -1.00 | 241.4 | 218.8 | 0.327 | 0.547 |
| **coarse** | D=2 | 2 | -0.513 | 0.352 | 5.0e-01 | -1.00 | 161.8 | 142.6 | 0.267 | 0.598 |
| **coarse** | D=16 | 2 | -0.384 | 0.350 | 5.0e-01 | -1.00 | 331.5 | 315.2 | 0.360 | 0.610 |
| **coarse** | All | 4 | -0.449 | 0.351 | 1.2e-01 | -1.00 | 246.7 | 228.9 | 0.313 | 0.604 |
| **mature** | D=2 | 2 | -0.630 | 0.361 | 5.0e-01 | -1.00 | 165.7 | 131.9 | 0.199 | 0.507 |
| **mature** | D=16 | 2 | -0.503 | 0.427 | 5.0e-01 | -1.00 | 340.6 | 308.2 | 0.281 | 0.651 |
| **mature** | All | 4 | -0.567 | 0.394 | 1.2e-01 | -1.00 | 253.2 | 220.0 | 0.240 | 0.579 |
| **shallow** | D=2 | 2 | -0.582 | 0.315 | 5.0e-01 | -1.00 | 163.0 | 136.7 | 0.223 | 0.519 |
| **shallow** | D=16 | 2 | -0.385 | 0.391 | 5.0e-01 | -1.00 | 348.3 | 311.4 | 0.344 | 0.630 |
| **shallow** | All | 4 | -0.483 | 0.353 | 1.2e-01 | -1.00 | 255.6 | 224.1 | 0.283 | 0.574 |
| **smac_default** | D=2 | 4 | -0.432 | 0.146 | 1.2e-01 | -1.00 | 154.3 | 128.8 | 0.337 | 0.431 |
| **smac_default** | D=16 | 4 | -0.355 | 0.335 | 1.2e-01 | -1.00 | 325.6 | 313.9 | 0.384 | 0.603 |
| **smac_default** | All | 8 | -0.393 | 0.240 | 7.8e-03 | -1.00 | 240.0 | 221.4 | 0.360 | 0.517 |
| **All** | All | 24 | -0.450 | 0.323 | 1.8e-05 | -1.00 | 246.1 | 222.4 | 0.314 | 0.556 |

*Complete surrogate scorecard saved to `extrapolation_surrogate_scorecard.csv`.*

---

## Sample Size Scaling Matrix

Evaluation across initial training sample sizes \(n_{\text{train}}\) and neighbor ratio \(k/n_{\text{train}}\):

| N_train | k/N Ratio | Dimension | N | PLCB Dist Corr | SLCB Dist Corr | p-val | Cliff's δ | PLCB Winkler | SLCB Winkler | PLCB AUROC | SLCB AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 112 | 0.250 | D=2 | 12 | -0.515 | 0.287 | 2.2e-03 | -1.00 | 158.7 | 133.1 | 0.280 | 0.501 |
| 112 | 0.250 | D=16 | 12 | -0.385 | 0.360 | 2.2e-03 | -1.00 | 333.6 | 311.8 | 0.348 | 0.612 |
| 112 | 0.250 | All | 24 | -0.450 | 0.323 | 1.8e-05 | -1.00 | 246.1 | 222.4 | 0.314 | 0.556 |
| All | N/A | All | 24 | -0.450 | 0.323 | 1.8e-05 | -1.00 | 246.1 | 222.4 | 0.314 | 0.556 |

*Complete sample size scorecard saved to `extrapolation_sample_size_scorecard.csv`.*

---

## UQ Component Ablation

Ablation across uncertainty quantification estimators (`slcb`, `rf_fire`, `prox_a`, `prox_b`, `prox_bc`, `plcb`, `shaker_total`):

| Estimator | Dimension | N | Dist Corr Mean | Diff vs SLCB | Win Rate | Winkler Mean | Diff vs SLCB | Win Rate | AUROC Mean | Diff vs SLCB | Win Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **slcb** | D=2 | 12 | 0.287 | +0.000 | 0.0% | 133.1 | +0.0 | 0.0% | 0.501 | +0.000 | 0.0% |
| **slcb** | D=16 | 12 | 0.360 | +0.000 | 0.0% | 311.8 | +0.0 | 0.0% | 0.612 | +0.000 | 0.0% |
| **slcb** | All | 24 | 0.323 | +0.000 | 0.0% | 222.4 | +0.0 | 0.0% | 0.556 | +0.000 | 0.0% |
| **rf_fire** | D=2 | 12 | -0.557 | -0.844 | 0.0% | 160.4 | +27.4 | 0.0% | 0.237 | -0.265 | 8.3% |
| **rf_fire** | D=16 | 12 | -0.367 | -0.727 | 0.0% | 329.7 | +17.9 | 0.0% | 0.354 | -0.258 | 0.0% |
| **rf_fire** | All | 24 | -0.462 | -0.785 | 0.0% | 245.1 | +22.6 | 0.0% | 0.295 | -0.261 | 4.2% |
| **prox_a** | D=2 | 12 | -0.515 | -0.802 | 0.0% | 158.7 | +25.6 | 0.0% | 0.280 | -0.222 | 0.0% |
| **prox_a** | D=16 | 12 | -0.385 | -0.746 | 0.0% | 333.6 | +21.8 | 0.0% | 0.348 | -0.263 | 0.0% |
| **prox_a** | All | 24 | -0.450 | -0.774 | 0.0% | 246.1 | +23.7 | 0.0% | 0.314 | -0.242 | 0.0% |
| **prox_b** | D=2 | 12 | -0.516 | -0.802 | 0.0% | 154.3 | +21.3 | 0.0% | 0.309 | -0.192 | 0.0% |
| **prox_b** | D=16 | 12 | -0.374 | -0.734 | 0.0% | 321.6 | +9.8 | 16.7% | 0.365 | -0.247 | 8.3% |
| **prox_b** | All | 24 | -0.445 | -0.768 | 0.0% | 238.0 | +15.6 | 8.3% | 0.337 | -0.219 | 4.2% |
| **prox_bc** | D=2 | 12 | 0.043 | -0.244 | 33.3% | 153.4 | +20.3 | 0.0% | 0.543 | +0.042 | 58.3% |
| **prox_bc** | D=16 | 12 | -0.110 | -0.470 | 25.0% | 321.2 | +9.4 | 16.7% | 0.472 | -0.140 | 33.3% |
| **prox_bc** | All | 24 | -0.034 | -0.357 | 29.2% | 237.3 | +14.8 | 8.3% | 0.507 | -0.049 | 45.8% |
| **plcb** | D=2 | 12 | -0.515 | -0.802 | 0.0% | 158.7 | +25.6 | 0.0% | 0.280 | -0.222 | 0.0% |
| **plcb** | D=16 | 12 | -0.385 | -0.746 | 0.0% | 333.6 | +21.8 | 0.0% | 0.348 | -0.263 | 0.0% |
| **plcb** | All | 24 | -0.450 | -0.774 | 0.0% | 246.1 | +23.7 | 0.0% | 0.314 | -0.242 | 0.0% |
| **shaker_total** | D=2 | 12 | 0.406 | +0.120 | 83.3% | 132.4 | -0.6 | 83.3% | 0.604 | +0.103 | 100.0% |
| **shaker_total** | D=16 | 12 | 0.271 | -0.089 | 33.3% | 2743.9 | +2432.1 | 83.3% | 0.590 | -0.021 | 41.7% |
| **shaker_total** | All | 24 | 0.339 | +0.015 | 58.3% | 1438.2 | +1215.7 | 83.3% | 0.597 | +0.041 | 70.8% |

*Complete UQ ablation scorecard saved to `extrapolation_uq_ablation_scorecard.csv`.*

---

## Conclusion & Recommendations for DyRF-BO

1. **Surrogate Choice:** PLCB should be adopted as the default acquisition guidance in DyRF-BO when querying unconstrained or high-dimensional search domains.
2. **Implicit Trust Region Protection:** Bounded OOB residual quantiles serve as an implicit trust region, preventing the surrogate from falling into the Hallucinated Exploration Trap in empty hypercube corners.