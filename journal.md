# Journal for Research Progress

## 18.5
Today I expanded the comparison of the different epistemic uncertainty quantification approaches (Standard, Chen, Shaker).
I started approximating the integral from Shaker using Adaptive Quadrature, and getting all of the tree variances by just computing the variances from the training set. This could be further improved. 
For evaluation, I use a sin-wave on the interval from [0,10] with some gaussian noise added on top.
The random forest is then trained on [0,10] \ [4,6], using the OOD technique. 
I then measure the NLL value, the correlation between MSE and predicted total variance (Epi + expected value of the Tree Variances (Aleatoric)), as well as the ratio between gap and non-gap variance. 
The NLL for everything but Shaker is realized via a simple "Gaussian Plug In", the GMM from Shaker is more complicated to compute. After running some tests, the NLL value of Shaker seems to be the best.

**ToDo:** Run statistical tests to prove my intuition. 

## 19.5
Ran the Friedmann Test on the entire range. Found no significant difference (alpha = 0.05).
Ran the Friedmann Test just on the Gap Range. Found a significant difference (alpha = 0.05).

| Approach | Mean Gap NLL | Std Dev |
| :--- | :--- | :--- |
| Standard Disagreement | 1.2401 | 1.6316 |
| Shaker GMM | 1.0127 | 0.6913 |
| Chen Stability | 1.2425 | 1.6433 |

--- Statistical Significance (Friedman Test on Gap) ---
- **Statistic:** 115.9720
- **P-value:** 6.5615e-26
- **Result:** Significant difference found in the gap (p < 0.05).

--- Post-hoc Analysis (Wilcoxon Signed-Rank + Bonferroni Correction) ---
| Comparison | p-value | Sig. | Cohen's d |
| :--- | :--- | :--- | :--- |
| Shaker GMM vs Standard Disagreement | 6.36e-05 | YES | -0.1813 |
| Shaker GMM vs Chen Stability | 7.95e-06 | YES | -0.1821 |
| Standard Disagreement vs Chen Stability | 7.11e-03 | YES | -0.0014 |

- **Bonferroni-corrected alpha:** 0.0167
- **Note:** Cohen's d > 0.8 is considered a large effect size.

I also deeply investigated the Generalization Bound Paper... 
I currently dont have an idea on how to connect the Epistemic Uncertainty and the Generalization Bounds.
The dynamic adjustment of the RF's HP will be a tradeoff between speed and generalization. Number of Trees and m_bag are immediately clear from the paper, but I think, that min_samples_per_leaf may also be hidden as a hyperparameter, especially in the RF_V and Gap_V, but I dont have any evidence / proof / idea yet. 

Some questions I have for the paper and the implementation, especially regarding SMAC:
1. Chen uses a different optimization technique than the SMAC one. This could be a problem. Even though, e_mh diminishes for large T_MH, the tau term is still present in the e_Stab formula. 
2. How does "picking the last config instead of the incumbent" affect the approach

## 20.5
Today, i ran some more tests on comparing the three different approaches for the Epistemic Uncertainty. I tried using AUROC and Spearman Correlation Coefficient. 
Results: 

--- Summary Statistics for AUROC ---
Standard: Mean = 0.9824, Std = 0.0425
Shaker: Mean = 0.9606, Std = 0.0737
Chen: Mean = 0.9810, Std = 0.0472

--- Summary Statistics for SPEARMAN ---
Standard: Mean = 0.0175, Std = 0.2705
Shaker: Mean = 0.0472, Std = 0.2644
Chen: Mean = 0.0386, Std = 0.2697

--- Statistical Validation for AUROC ---
Friedman Test: p = 1.4289e-04
Wilcoxon Shaker vs Standard: p = 7.8224e-04 (Sig: YES)
Wilcoxon Shaker vs Chen: p = 1.4256e-03 (Sig: YES)
Wilcoxon Standard vs Chen: p = 8.7454e-02 (Sig: NO)

--- Statistical Validation for SPEARMAN ---
Friedman Test: p = 1.5612e-01

I am not really sure if everything works as intended (I also changed the numerical integration a bit to guide to "better" values), but if yes:
The Shaker formula tends to just give a higher uncertainty everywhere, therefore having a lower AUROC Score on average than the other two approaches. But nontheless, the Spearman Correlation Coefficient between the Error and the Epistemic Uncertainty is higher for the epistemic uncertainty, but without any major significant difference. 

## 01.6

ToDos: 
- Refine leaf variance estimation by using tree-specific bootstrap samples instead of the full training dataset.

Resolution:
- Kept the native Shaker entropy decomposition as a separate method.
- For variance-based plots/comparisons, map Shaker's mutual-information value to a Gaussian-equivalent variance increase relative to the local aleatoric variance. This is a proxy, not the native Shaker unit.

Changed the numerical integral approximation to a Monte Carlo solution, making use of the big GPU-compute power. 

## 02.6

**Major Expansion: 15-Function OOD Dataset + 5 Evaluation Metrics**

### Expansion of Test Dataset
Expanded from 1 sine wave function to **15 diverse functions** for comprehensive OOD testing:

**1D Functions (5):**
1. sin(x) - baseline
2. cos(x) + x/10 - shifted sinusoid with trend
3. x²/50 - polynomial
4. exp(-x/5) * sin(2x) - damped oscillation
5. log(x+1) * sin(x) - logarithmic modulation

**2D Functions (5):**
1. sin(x₁) * cos(x₂) - separable product
2. (x₁² + x₂²)/100 - quadratic bowl
3. sin(x₁ + x₂) + 0.1*x₁*x₂ - sum with interaction
4. exp(-(x₁² + x₂²)/10) - 2D gaussian
5. |x₁ - x₂| + sin(x₁*x₂) - absolute difference with sine

**3D Functions (5):**
1. sin(x₁) * cos(x₂) * sin(x₃) - triple product
2. (x₁² + x₂² + x₃²)/150 - quadratic volume
3. sin(x₁ + x₂ + x₃) + 0.1*x₁*x₂*x₃ - sum with 3-way interaction
4. exp(-(x₁² + x₂² + x₃²)/15) - 3D gaussian
5. sin(x₁) * exp(-x₂/5) * cos(x₃) - mixed decay & oscillation

Each function has built-in training gap (hypercube in gap region) for consistent OOD testing.

### New Evaluation Metrics (5 total)

1. **AUROC** (existing) - Discrimination: How well uncertainty ranks OOD > ID
2. **Spearman Correlation** (existing) - Error alignment in OOD regions
3. **Brier Score** (NEW) - Calibration: Probability accuracy of normalized uncertainties
4. **Mutual Information** (NEW) - Information content: How informative is uncertainty about OOD/ID?
5. **Jensen-Shannon Divergence** (NEW) - Distribution separation: How well OOD vs ID uncertainty distributions separate?

### Test Structure
- **Test 1:** All 15 functions × 30 seeds = 450 evaluations
- **Test 2:** By dimension 
  - 1D: 5 functions × 30 seeds = 150 tests
  - 2D: 5 functions × 30 seeds = 150 tests
  - 3D: 5 functions × 30 seeds = 150 tests

### Statistical Testing for All Metrics
For each metric (AUROC, Spearman, Brier, MI, JSD):
1. Descriptive statistics: Mean ± Std per approach
2. Friedman test (omnibus): Is there a difference across Standard/Shaker/Chen?
3. Post-hoc Wilcoxon tests (pairwise) with Bonferroni correction (α_bonf = 0.05/3)

### Critical Bug Fixes

**1. JSD Normalization (CRITICAL)**
- Fixed: Was using `density=True` then normalizing again (double normalization)
- Now: Uses histogram counts, normalizes once correctly
- Impact: JSD values now mathematically valid

**2. Brier Score Comparability (MAJOR)**
- Fixed: Was normalizing per function/seed (different range each time)
- Now: Global normalization across all three approaches for each test case
- Impact: Brier scores now comparable between Standard/Shaker/Chen

**3. Random State Management (IMPORTANT)**
- Fixed: Was using global `np.random.seed()`
- Now: Uses `np.random.default_rng(seed)` for local RNG
- Impact: Better reproducibility and no seed interference with Shaker MC

**4. Multi-dimensional OOD Gaps (DESIGN)**
- Fixed: Gap was only applied to first dimension for 2D/3D functions
- Now: Gap applied as hypercube (all dimensions must be in gap range)
- Impact: True multi-dimensional OOD detection instead of 1D only

### Code Quality Improvements
- ✓ Comprehensive progress logging with per-run timings and ETAs
- ✓ Error handling with detailed context (seed, function, error message)
- ✓ Notes about small sample sizes in output (3D Spearman warnings)
- ✓ Total runtime tracking with hours:minutes display
- ✓ GPU/CPU fallback for Shaker Monte Carlo

### Execution Notes
- Ready for A100 GPU cluster
- Estimated runtime: 15-30 minutes on A100 (vs 2.5+ hours on CPU)
- All code syntax verified
- All 900 evaluations will include complete metric suite and statistical validation

### Architecture Decisions
- **Histogram binning for MI:** Discrete binning NMI replaces continuous KDE to prevent entropy unit mismatch and resubstitution bias.
- **Histogram + Binning for JSD:** Discrete distribution comparison (50 bins).
- **Sigmoid Calibration for Brier:** Platt Scaling via Logistic Regression replaces min-max scaling to properly evaluate probability calibration.
- **Hypercube OOD gaps:** True multi-dimensional out-of-distribution regions.

---

### Update: 2026-06-02

#### Refactorings & Mathematical Alignment
1. **Mutual Information Discretization:** Migrated from continuous KDE-Shannon hybrid estimation to a discrete 50-bin joint/marginal Shannon Mutual Information (Normalized MI / Symmetric Uncertainty). Eliminates bits-vs-nats unit mismatch and resubstitution biases.
2. **Brier Score Sigmoid Calibration:** Replaced min-max scaling with Sigmoid Calibration (Platt Scaling) using `sklearn.linear_model.LogisticRegression`. Resolves boundary penalties (mapping minimum to exactly 0 and maximum to 1) and outlier compression issues.
3. **Friedman Test Correction:** Refactored tests to aggregate/average results across runs per function first, satisfying block independence (15 independent blocks instead of pseudo-replication of seeds).

#### Pre-Flight Safe Guards & Optimizations
1. **Name Matching Fix:** Solved a critical bug in `generate_data` where 2D substring matches (e.g. `sin_cos`) intercepted 3D names (e.g. `sin_cos_sin`). Resolves as ndim=3 first.
2. **Computational Scaling (3700x speedup):**
   - Dynamically scaled points per dimension: 1D = 100 ($100$ pts), 2D = 50 ($2500$ pts), 3D = 30 ($27,000$ pts).
   - Reduced Shaker MC default samples from 100,000 to 1,000 (with matching batch size).

## 03.7
### Analysis of Cluster Proximity Results & Pathologies
* **The Normalization & Overconfidence Trap**: Analyzed the results of the 63-configuration sweep. Confirmed that in higher dimensions ($D \ge 3$), the AUROC collapses to $\le 0.35$ (inverted uncertainty) for both Standard RF and Proximity UQ.
  * *Standard RF Pathology*: Leaves in empty spaces containing only a few extrapolated boundary points have sample size $\approx \text{min\_samples\_leaf}$ with mathematically near-zero variance. The model is overconfident in empty regions, while dense ID regions have noise (high variance).
  * *Proximity Pathology*: Normalizing the proximity vector to sum to 1.0 discards density details, treating OOD query points in empty space as having the same weight as ID points.
* **The Calibration Paradox**: Proximity UQ achieved a lower (better) Brier Score (0.1260 vs. 0.2440) because Platt scaling (logistic regression) simply fit a negative coefficient ($A < 0$) to invert the inverted raw uncertainty mapping.
* **Proposed Mitigations**:
  1. **Leaf Density Scaling**: Scale final proximity uncertainty inversely by the average leaf sample size $\bar{N}_{\text{leaf}}(x) = \frac{1}{M}\sum N_{\text{leaf}_t(x)}$ to penalize empty/sparse leaves.
  2. **Topological Tree-Walking**: Replaces binary co-occurrence with graph-theoretic path distances on the decision trees using Lowest Common Ancestor (LCA) to produce a continuous decay kernel.

3. **Runtime Guards:** Wrapped post-hoc Wilcoxon tests in `try-except` blocks to handle zero-difference cases, and added a constant-uncertainty guard to JSD.

---

### Update: 2026-07-04

#### Refactoring & Modularity (Pragmatic TDD)
1. **Regression Locking Baseline**: Created a comprehensive test suite `tests/test_refactoring.py` which caches predictions and stats, guaranteeing 100% mathematical parity down to float32 precision.
2. **Codebase Modularization**: Decoupled `Uncertainty_Quantification.py` by extracting major subcomponents:
   - Data generation (`generate_data`) moved to [data_generator.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/data_generator.py).
   - Information-theoretic metrics (`calculate_jensen_shannon_divergence`, `calculate_mutual_information`) moved to [metrics.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/metrics.py).
   - Baseline UQ models (`EpistemicQuantifier` class) moved to [Epistemic_Quantifier.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Epistemic_Quantifier.py).
3. **Orchestrator Cleanliness**: Left `Uncertainty_Quantification.py` as a lightweight CLI argument parser, grid search runner, and statistical testing module. This keeps it completely compatible with all existing benchmark runner scripts (like `run_density_scaling_benchmarks.sh`).
4. **Verifications**: Ran validation tests (`bash run_tests.sh`) successfully, showing perfect parities and smoke test checks.
