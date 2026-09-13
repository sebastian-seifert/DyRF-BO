# End-to-End Test Infrastructure: DyRF-BO Epistemic Uncertainty Quantification in SMAC3

## 1. Overview & Architecture

The DyRF-BO End-to-End (E2E) test infrastructure provides rigorous, opaque-box, requirement-driven verification across all core components of the Epistemic Uncertainty Quantification research pipeline. The suite directly verifies the requirements and acceptance criteria established in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

The test suite is organized into a 4-Tier testing architecture:

```
                      ┌─────────────────────────────────────────┐
                      │                 TIER 4                  │
                      │       Real-World BO Scenarios           │
                      │  (Ackley 2D, Rosenbrock 2D, Hartmann    │
                      │   6D, N>=30 Wilcoxon, AUROC, Overhead)  │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │                 TIER 3                  │
                      │       Cross-Feature Interactions        │
                      │ (Extractor + Surrogate, Surrogate + Acq,│
                      │    Harness + Wilcoxon Report Pipeline)  │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │                 TIER 2                  │
                      │       Boundary & Corner Cases           │
                      │   (Zero Noise, Far Extrapolation Gaps,  │
                      │    High D=20..50, Singleton Leaves, NaN)│
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │                 TIER 1                  │
                      │          Feature Coverage               │
                      │  (R0 Git Isolation, R1 Pure Extractor,  │
                      │   R2 Additive Acq, R3 Runner & Stats)   │
                      └─────────────────────────────────────────┘
```

---

## 2. Testing Philosophy & Principles

1. **Opaque-Box & Requirement-Driven**: Tests interact exclusively with public class and module APIs (`BaseEpistemicExtractor`, `UQExtractorRegistry`, `CustomUncertaintyRandomForest`, `AdditiveEpistemicAcquisition`, `NoisyBenchmarkProblem`, `NoisyBOHarness`, `compute_wilcoxon_suite`). Internal heuristics are validated via their external mathematical effects.
2. **Authoritative Expected Output Derivation**: Expected behaviors are derived from documented mathematical invariants:
   - Epistemic uncertainty is strictly non-negative: $U_E(x) \ge 0$.
   - Empty gaps exhibit higher uncertainty than dense training regions: $U_E(x_{\text{gap}}) > U_E(x_{\text{dense}})$.
   - Far out-of-distribution uncertainty is asymptotically bounded and numerically stable: $U_E(x) \in [0, \mathcal{O}(\sigma_0)]$.
   - Normalization maps arbitrary non-negative scores to $[0, 1]$.
   - Wilcoxon signed-rank tests must satisfy $p < 0.05$ across $N \ge 30$ seeds.
3. **Strict Test Integrity**:
   - Zero facade tests, zero mocked return values for core mathematical assertions.
   - Genuine surrogate model fitting, tree traversal, covariance transformation, and optimization iterations.
   - Independent verification by automated auditing subagents.
4. **Deterministic Isolation**:
   - Every test instantiates its own random number generator stream (`np.random.default_rng(seed)`).
   - Temporary file generation uses pytest's isolated `tmp_path` fixtures.
   - Tests have no cross-test state dependencies.

---

## 3. Tier Specifications & Feature Mapping

### Tier 1: Core Feature Coverage (`tests/e2e/test_tier1_features.py`)
Covers the 5 primary project requirements with >=5 tests per feature (25 tests total):

| Requirement | Description | Test Focus | Tests Count |
|---|---|---|:---:|
| **R0** | Version Control & Isolation | Branch name verification (`feat/epistemic-uncertainty-research`), main branch protection, clean root, git history validation | 5 |
| **R1** | Pure Epistemic Extractor Interface | Registry discovery, `fit(X, y)` contract, `(N,)` shape output, non-negativity $U_E(x) \ge 0$, KeyError on invalid names | 5 |
| **R2** | Epistemic Acquisition Enhancement | Decoupled additive acquisition, `WarmupCosineScheduler` warmup and cosine decay, `normalize_max_relative`, LCB min-shifting, `AcquisitionRegistry` | 5 |
| **R3** | Local Benchmark Runner Harness | `NoisyBenchmarkRegistry` discovery, synthetic functions (Ackley, Rosenbrock, Hartmann), `NoisyTelemetryLogger` tracking, deterministic seed reproducibility, SMAC3 target adapter | 5 |
| **R3** | Paired Statistical Evaluation | Cliff's delta non-parametric effect size, Holm-Bonferroni FWER step-down adjustment, Wilcoxon signed-rank testing, end-to-end report generation (.md, .csv, .tex), $N \ge 30$ seed constraint | 5 |

### Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundaries.py`)
Covers 5 boundary regimes with >=5 tests per category (27 tests total):

| Boundary Feature | Edge Case Description | Test Focus |
|---|---|---|
| **Zero Noise** | Deterministic limit | Constant flat surfaces ($y=c$), duplicate observations, near-zero variance $\sigma^2 \to 0$ in EI/LCB, terminal decay $\beta_t = 0$, noiseless quadratic function |
| **Extreme Extrapolation** | Gap topology | Far extrapolation at $10\times, 100\times, 1000\times$ distance, empty training gap $[3, 7]$ vs dense regions, asymptotic saturation, additive acquisition gap exploration bonus, disjoint multi-cluster domain |
| **High Dimensions** | Dimension scaling | Input dimensions $D \in [10, 25, 50]$, $N \ll D$ regime (5 samples in 20D), uninformative noise features, 500-candidate batch evaluation, 50D Euclidean distance numerical stability |
| **Degenerate Trees** | Minimal sample/tree count | Single sample dataset ($N=1$), identical leaf predictions, single tree forest ($M=1$), isolated singleton leaves ($N_m=1$), deep unconstrained trees (`max_depth=None`) |
| **NaN/Inf & Numerical Guards** | Precision bounds | `normalize_max_relative` handling of `NaN`, `+Inf`, `-Inf`, microscopic scale ($y \sim 10^{-8}$), macroscopic scale ($y \sim 10^8$), float32/float64 dtype compatibility, variance clipping guard |

### Tier 3: Pairwise Feature Combinations (`tests/e2e/test_tier3_combinations.py`)
Verifies cross-feature interactions across pipeline boundaries (18 tests total):

| Interaction Pair | Systems Combined | Verification Focus |
|---|---|---|
| **Extractor ↔ SMAC3 Surrogate** | `UQExtractorRegistry` + `CustomUncertaintyRandomForest` | Multi-extractor integration (`standard_disagreement`, `proximity_b`, `standard_proximity`, `proximity_bc`), variance equals $U_E(X)^2$, sequential dataset retraining, custom callable binding, inactive NaN imputation |
| **Surrogate ↔ Acquisition Scheduler** | `CustomUncertaintyRandomForest` + `AdditiveEpistemicAcquisition` + `WarmupCosineScheduler` | Pipeline integration with EI base, pipeline with LCB base, dynamic transition from exploration to exploitation across budget steps $t \in [1..T]$, candidate argmax selection shift, zero-epistemic degradation |
| **Harness ↔ Statistical Analyzer** | `NoisyBOHarness` / `NoisyTelemetryLogger` + `compute_wilcoxon_suite` | Telemetry records export to DataFrame, paired seed alignment across tasks and seeds, directionality concordance with Cliff's delta, multi-task Holm-Bonferroni correction, JSON $\to$ Parquet $\to$ Tables pipeline |

### Tier 4: Real-World Acceptance Scenarios (`tests/e2e/test_tier4_scenarios.py`)
Validates complete Bayesian Optimization flows against explicit acceptance criteria (6 tests total):

| Scenario | Benchmark / Metric | Acceptance Criterion Verified |
|---|---|---|
| **Ackley 2D BO** | Ackley 2D continuous ($[-5, 5]^2$) | Decoupled Additive Epistemic BO executes, logs incumbent regret, and achieves monotonic regret reduction toward global minimum |
| **Rosenbrock 2D BO** | BBOB Rosenbrock 2D ($[-5, 5]^2$) | Navigates curved parabolic valley, tracks true incumbent cost, and achieves convergence |
| **Hartmann 6D BO** | Hartmann 6D continuous ($[0, 1]^6$) | Explores multimodal 6-dimensional search space, tracks 6D vectors, and reduces regret over budget |
| **Paired Wilcoxon Suite** | Ackley 2D/4D, Rosenbrock 2D/4D, Hartmann 6D ($N=35$ seeds) | Novel method achieves statistically significant lower regret ($p < 0.05$ Wilcoxon signed-rank test across $\ge 30$ seeds) on at least 3 benchmark functions with Holm-Bonferroni correction |
| **OOD-AUROC in Gap** | Synthetic gap topology ($[3.5, 6.5]$ gap) | Novel epistemic metric achieves higher OOD-AUROC than standard ensemble variance ($\text{AUROC}_{\text{ep}} > \text{AUROC}_{\text{base}}$) and $\text{AUROC}_{\text{ep}} \ge 0.80$ |
| **Local Runtime Profiling** | $N=60, D=5$, 1000 candidates | Epistemic uncertainty extraction adds $< 20\%$ local runtime overhead per BO iteration relative to baseline Random Forest surrogate budget |

---

## 4. Formal Acceptance Thresholds

| Metric | Target / Threshold | Authoritative Source | Validation Test |
|---|---|---|---|
| **Git Isolation** | Branch == `feat/epistemic-uncertainty-research` | ORIGINAL_REQUEST §R0 | `test_r0_git_current_branch_is_feature_branch` |
| **Signal Positivity** | $U_E(x) \ge 0.0$ for all $x$ | PROJECT.md §Interface Contracts | `test_r1_extractor_non_negativity_and_finite_invariants` |
| **Statistical Significance** | Wilcoxon signed-rank test $p < 0.05$ | ORIGINAL_REQUEST §Acceptance Criteria | `test_t4_paired_wilcoxon_statistical_significance_validation` |
| **Evaluation Seeds** | $N \ge 30$ random seeds | ORIGINAL_REQUEST §R3, §Acceptance Criteria | `test_r3_sample_size_check_thirty_seeds` |
| **OOD-AUROC** | $\text{AUROC}_{\text{epistemic}} > \text{AUROC}_{\text{baseline}}$ | ORIGINAL_REQUEST §Acceptance Criteria | `test_t4_ood_auroc_metric_in_synthetic_gap` |
| **Runtime Overhead** | $\Delta T / T_{\text{baseline}} < 20\%$ | ORIGINAL_REQUEST §Acceptance Criteria | `test_t4_runtime_overhead_under_twenty_percent` |

---

## 5. Execution Guide

### Complete Test Suite Run
To execute all 76 E2E tests:
```bash
bash scripts/run_e2e_tests.sh
```
or directly via pytest:
```bash
.venv/bin/pytest tests/e2e/ -v
```

### Running Specific Tiers
```bash
# Tier 1 only (Core Features)
.venv/bin/pytest tests/e2e/test_tier1_features.py -v

# Tier 2 only (Boundaries & Corners)
.venv/bin/pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3 only (Cross-Feature Combinations)
.venv/bin/pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4 only (Real-World Scenarios)
.venv/bin/pytest tests/e2e/test_tier4_scenarios.py -v
```
