# Epistemic Uncertainty Quantification in SMAC3: Benchmark & Statistical Evaluation Summary

## Executive Summary

This report presents the empirical benchmark evaluation and statistical hypothesis testing of the
**Distance-Aware Evidential Hybrid Random Forest (DA-EHRF)** pure epistemic uncertainty quantification
surrogate against the standard SMAC3 Random Forest baseline on local continuous optimization benchmarks,
fulfilling Mandates R0, R1, R2, and R3 of `ORIGINAL_REQUEST.md`.

## Benchmark Topologies & Evaluation Setup

| Topology | Dimension | Domain | Optimum | Noise Model | Challenge |
|---|---|---|---|---|---|
| **Ackley 2D** | 2 | $[-5, 5]^2$ | $f(0, 0) = 0.0$ | Gaussian $\sigma=0.05$ | Multimodal deceptive basins |
| **Rosenbrock 2D** | 2 | $[-2, 2]^2$ | $f(1, 1) = 0.0$ | Gaussian $\sigma=0.05$ | Curved parabolic valley |
| **Hartmann 6D** | 6 | $[0, 1]^6$ | $f(x^*) = -3.32237$ | Gaussian $\sigma=0.05$ | 6D multimodal landscape |
| **Synthetic Gap 2D** | 2 | $[-5, 5]^2$ | $f(0, 0) = 0.0$ | Gaussian $\sigma=0.05$ | Unobserved central exploration gap $[-1.5, 1.5]^2$ |

- **Paired Seeds**: $N = 35$ paired seeds per benchmark ($N \ge 30$).
- **Paired Seed Rigor**: For each seed $s$, initial design evaluations are 100% identical between Baseline and Proposed.
- **Reference Baseline**: Standard SMAC3 RF (`SMAC3_HPOFacade_ei`) with standard empirical tree variance.
- **Proposed Method**: DA-EHRF Decoupled Additive Acquisition (`CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential` with `WarmupCosineScheduler`).

## Formal Statistical Hypothesis Testing Results

### Statistical Power & Primary Evaluation Framework

- **Primary Evaluation**: Per-task paired-seed Wilcoxon signed-rank tests across $N = 35$ seeds ($N \ge 30$, fulfilling Mandate R3).
- **Cross-Task Aggregation Limitation**: When aggregating mean regrets across $N = 4$ tasks, the mathematical minimum two-sided Wilcoxon p-value is $p_{\min} = 2 \cdot (0.5)^4 = 0.125 > 0.05$. Therefore, per-task paired evaluations (below) are authoritative.

| task_id | optimizer_id | n_seeds | Baseline Mean Regret | Proposed Mean Regret | Mean Diff | Rel Reduction (%) | Win / Loss / Tie | Wilcoxon W | p_raw | Cliff's delta | Holm-Bonferroni adj p | Significance (adj p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ackley_2d | CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential | 35 | 1.1044 | 1.0689 | -0.0355 | -3.2% | 20 / 14 / 1 | 284.0 | 0.8175 | +0.042 | 1.0000 | Non-significant |
| hartmann_6d | CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential | 35 | 0.9609 | 1.1386 | +0.1777 | +18.5% | 13 / 21 / 1 | 204.0 | 0.1099 | +0.183 | 0.3298 | Non-significant |
| rosenbrock_2d | CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential | 35 | 3.3178 | 4.7766 | +1.4587 | +44.0% | 16 / 14 / 5 | 214.0 | 0.7036 | -0.095 | 1.0000 | Non-significant |
| synthetic_gap_2d | CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential | 35 | 3.1395 | 2.5946 | -0.5449 | -17.4% | 24 / 9 / 2 | 158.0 | 0.0286 | -0.210 | 0.1144 | Non-significant |

## Out-of-Distribution Epistemic Uncertainty Quantification (Gap Topology)

| Metric | Standard SMAC3 RF (Baseline) | DA-EHRF (Proposed) | Criterion | Status |
|---|---|---|---|---|
| **OOD-AUROC** | 0.7301 | **0.9997** | $\ge 0.80$ & $>$ Baseline | **PASSED (✓)** |
| **Error Correlation** | 0.3007 | **0.6261** | $>$ Baseline | **PASSED (✓)** |
| **Brier Score** | 0.1978 | **0.1169** | Lower is better | **PASSED (✓)** |

> **Theoretical Significance**: DA-EHRF achieves an OOD-AUROC of **0.9997** in the empty gap region,
isolating model ignorance where standard RF variance fails due to tree consensus on extrapolation boundaries.

## Local Runtime Overhead Profiling

- **Baseline RF Fit + Candidate Predict**: 100.09 ms
- **DA-EHRF Epistemic Extraction**: 11.21 ms
- **Relative Runtime Overhead**: **11.2%** (Target: $< 20.0\%$)
- **Status**: **PASSED (✓)** — well within local CPU budget constraints.

## Acceptance Criteria Verification Checklist

- [x] **Git Isolation (R0)**: All code, tests, and results reside strictly on `feat/epistemic-uncertainty-research`.
- [x] **Pure Epistemic Uncertainty Formulation (R1)**: DA-EHRF combines ensemble disagreement, leaf sample ignorance, and spatial distance.
- [x] **Acquisition Function Integration (R2)**: Decoupled Additive Acquisition with WarmupCosineScheduler.
- [x] **Local Benchmark & Statistical Rigor (R3)**: Standardized harness across 4 topologies with $N \ge 30$ paired seeds.
- [x] **OOD-AUROC & Error Correlation**: AUROC = 0.9997 $\ge 0.80$, Spearman = 0.6261 $>$ Baseline (0.3007) in gap topology.
- [x] **Runtime Overhead**: 11.2% overhead $< 20\%$ per BO iteration.
- [x] **Strict TDD & Reproducibility**: Unit tests and E2E test suites fully passing.
