# Project: DyRF-BO Epistemic Uncertainty Quantification in SMAC3

## Architecture
The project develops, integrates, and benchmarks pure epistemic uncertainty quantification methods for Random Forest surrogates in SMAC3 Bayesian Optimization.
The architecture comprises three core systems:
1. **Surrogate Epistemic Extractor Layer (`ep_extractors/`)**:
   - Extends `BaseEpistemicExtractor` to implement Distance-Aware Evidential Hybrid Random Forest (DA-EHRF):
     $$U_E(x) = \sqrt{V_{\text{ens}}(x) + V_{\text{leaf}}(x) + V_{\text{spatial}}(x)}$$
     where $V_{\text{ens}}(x) = \frac{1}{M}\sum (\mu_m(x) - \bar{\mu}(x))^2$, $V_{\text{leaf}}(x) = \sigma_0^2 \frac{1}{M}\sum \frac{1}{N_m(x)}$, and $V_{\text{spatial}}(x) = \sigma_0^2 (1 - \exp(-d_{\mathbf{W}}^2(x, \mathcal{D})/2\ell_0^2))$.
   - Outputs non-negative standard deviation in linear target units $[y]$.
2. **Acquisition Function & SMAC3 Model Integration Layer (`carps_integration/`)**:
   - `CustomUncertaintyRandomForest`: Subclasses SMAC3 `RandomForest` to intercept `_predict(X)` and inject epistemic variance into native SMAC3 acquisition functions (`EI`, `LCB`).
   - `AdditiveEpistemicAcquisition`: Decoupled additive acquisition combining max-relative normalized base acquisition with annealed epistemic exploration bonus $\beta_t \cdot \widetilde{U}_E(x)$ via `WarmupCosineScheduler`.
3. **Local Benchmarking & Statistical Evaluation Suite (`noisy_benchmarks/`, `scripts/`)**:
   - Evaluates across synthetic gap benchmarks (Ackley 2D/4D, Rosenbrock 2D/4D, Hartmann 6D) with $N \ge 30$ paired random seeds locally.
   - Computes Wilcoxon signed-rank tests ($p < 0.05$), Cliff's delta effect sizes, Holm-Bonferroni correction, OOD-AUROC, Spearman error-correlation, and runtime overhead profiling.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Git Isolation (R0) | Create and check out dedicated feature branch `feat/epistemic-uncertainty-research` | M0 | ORIGINAL_REQUEST §R0 |
| 2 | Pure Epistemic Formulation (DA-EHRF) | Implement `DistanceAwareEvidentialExtractor` combining ensemble disagreement, leaf sample ignorance, and spatial kernel distance | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Numerical Stability & Gap Invariants | Asymptotic bounds ($U_E \to 0$ in dense ID, $U_E \to \sigma_0$ in empty gaps), zero-noise robustness, NaN/Inf clipping | M1 | ORIGINAL_REQUEST §R1, AGENTS.md |
| 4 | TDD Extractor Test Suite | Comprehensive unit tests for DA-EHRF (`tests/test_distance_evidential_extractor.py`) testing stability, zero-variance, edge cases | M1 | ORIGINAL_REQUEST §Acceptance Criteria, AGENTS.md |
| 5 | SMAC3 RF Surrogate Binding | Integrate DA-EHRF into `CustomUncertaintyRandomForest` with direct variance injection | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Decoupled Additive Acquisition | Integrate DA-EHRF into `AdditiveEpistemicAcquisition` with `WarmupCosineScheduler` | M2 | ORIGINAL_REQUEST §R2 |
| 7 | TDD Acquisition Test Suite | Unit tests for acquisition integration (`tests/test_epistemic_acquisition_integration.py`) verifying bounds, annealing, and SMAC3 interface | M2 | ORIGINAL_REQUEST §Acceptance Criteria, AGENTS.md |
| 8 | Benchmark Topologies Configuration | Standardized harness for Ackley 2D/4D, Rosenbrock 2D/4D, and Hartmann 6D with synthetic exploration gaps | M3 | ORIGINAL_REQUEST §R3 |
| 9 | Local Paired Benchmarking ($N \ge 30$) | Execute local BO sweeps across $\ge 30$ seeds comparing DA-EHRF against default SMAC3 RF baseline | M3 | ORIGINAL_REQUEST §R3, §Acceptance Criteria |
| 10 | Paired Statistical Hypothesis Testing | Wilcoxon signed-rank test ($p < 0.05$), Cliff's delta, Holm-Bonferroni correction | M3 | ORIGINAL_REQUEST §Acceptance Criteria |
| 11 | OOD-AUROC & Runtime Profiling | Verify OOD-AUROC / error-correlation in gap regions and $< 20\%$ local runtime overhead per BO iteration | M3 | ORIGINAL_REQUEST §Acceptance Criteria |
| 12 | Automated Results Archival | Empirical results, p-values, and summary reports saved to repository on `feat/epistemic-uncertainty-research` | M3 | ORIGINAL_REQUEST §Acceptance Criteria |
| 13 | Requirement-Driven E2E Test Suite | Opaque-box 4-tier E2E tests covering features, boundary cases, pairwise interactions, and workflow acceptance criteria | E2E Track | Project Pattern Dual Track |
| 14 | Final Verification & Coverage Hardening | Pass 100% of E2E test suite (Tiers 1-4) and adversarial coverage hardening (Tier 5) | M4 | Project Pattern Final Milestone |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Git Branch Isolation (R0) | Create and check out `feat/epistemic-uncertainty-research`, verify clean working tree | none | DONE |
| M1 | Pure Epistemic Uncertainty Extractor (R1) | TDD implementation of `DistanceAwareEvidentialExtractor` (DA-EHRF) in `ep_extractors/distance_evidential.py` and `tests/test_distance_evidential_extractor.py` | M0 | DONE |
| M2 | Epistemic Acquisition Integration (R2) | Integration with `CustomUncertaintyRandomForest`, `AdditiveEpistemicAcquisition`, and unit tests in `tests/test_epistemic_acquisition_integration.py` | M1 | DONE |
| M3 | Local Benchmarking & Statistical Evaluation (R3) | Local $N \ge 30$ seed evaluation across Ackley 2D/4D, Rosenbrock 2D/4D, Hartmann 6D, Wilcoxon testing ($p < 0.05$), OOD-AUROC, overhead profiling, results archival | M2 | DONE |
| M4 | Final Milestone (E2E Pass & Adversarial Hardening) | Phase 1: 100% E2E test pass (Tiers 1-4). Phase 2: Adversarial coverage hardening (Tier 5) | M3, E2E Track | DONE |

## Interface Contracts
### `ep_extractors/distance_evidential.py` ↔ `carps_integration/custom_uncertainty_model.py`
- Class: `DistanceAwareEvidentialExtractor(BaseEpistemicExtractor)`
- Registration: `@UQExtractorRegistry.register("distance_evidential")`
- Methods:
  - `fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None`: Precomputes feature importances (MDI weights $\mathbf{W}$), global target prior std $\sigma_0 = \text{std}(y_{\text{train}})$, training points $\mathcal{D}_n$, and spatial bandwidth $\ell_0$.
  - `extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray`: Returns standard deviation array of shape `(n_samples,)` in linear target units $[y]$.
- Invariants:
  - Strictly non-negative: $U_E(x) \ge 0$.
  - Numerical stability: No NaNs, Infs, zero-division protected via epsilon clipping.
  - Scale: Expressed in target standard deviation $[y]$.

### `carps_integration/custom_uncertainty_model.py` ↔ `smac.acquisition.function`
- Method: `_predict(self, X: np.ndarray, covariance_type: str | None = "diagonal") -> tuple[np.ndarray, np.ndarray]`
- Contract: Returns `(mean, var)` where `mean` has shape `(N, 1)` and `var = (U_E(X)**2).reshape(-1, 1)`.

### `carps_integration/acquisitions.py` ↔ Optimization Loop
- Class: `AdditiveEpistemicAcquisition`
- Method: `compute_additive(self, preds, unc_tot, u_epistemic, y_best, beta_t=1.0) -> np.ndarray`
- Contract: Decoupled additive acquisition $\alpha_{\text{add}}(x) = \text{norm}(\alpha_{\text{base}}(x)) + \beta_t \cdot \text{norm}(U_E(x))$.

## Code Layout
- Extractor: `ep_extractors/distance_evidential.py`
- Extractor Registry: `ep_extractors/__init__.py`
- Unit Tests: `tests/test_distance_evidential_extractor.py`, `tests/test_epistemic_acquisition_integration.py`
- Benchmark Runner: `scripts/run_local_epistemic_benchmark.py`
- Statistical Analysis: `scripts/compute_pairwise_wilcoxon_suite.py`
- Results: `results/epistemic_research/`
- E2E Tests: `tests/e2e/`
