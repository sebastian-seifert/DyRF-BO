# Detailed Implementation Plan: Uncertainty Calibration in the Extrapolation Regime
## Proximity LCB (k-NN OOB Quantiles) vs. Standard SMAC3 LCB (Hutter Law of Total Variance)

---

### 📍 Document Metadata
- **Repository**: `DyRF-BO`
- **Branch**: `feat/extrapolation-uq-calibration`
- **Target Implementation Directory**: `dyrf_bo/extrapolation_uq/`
- **Target Test Directory**: `tests/`
- **Governing Standard**: Strict Test-Driven Development (TDD)

---

## Section 1: Architecture & Package Layout

All functional modules will be isolated in `dyrf_bo/extrapolation_uq/` with 1-to-1 paired test suites in `tests/`:

```
DyRF-BO/
├── dyrf_bo/
│   └── extrapolation_uq/
│       ├── __init__.py              # Public exports
│       ├── qp_convex_hull.py        # QP solver for convex hull projection & Gram matrix precomputation
│       ├── test_objectives.py        # Vectorized synthetic benchmarks (Sphere, Rosenbrock, Rastrigin, Ackley)
│       ├── samplers.py               # Sub-domain training sampler + Dirichlet & ray-casting test samplers
│       ├── uq_evaluator.py           # Dual UQ extraction: Hutter LTV vs. Pure PLCB exploration term
│       ├── metrics_suite.py          # Monotonicity, PICP, MPIW, Winkler score, AUROC/AUPRC
│       └── runner.py                 # Multi-D sweep execution harness & streaming Parquet logger
├── scripts/
│   └── run_extrapolation_study.py    # CLI entry point for local execution & cluster sweeps
└── tests/
    ├── test_qp_convex_hull.py        # Unit tests for QP distance, projection & Gram caching
    ├── test_test_objectives.py       # Unit tests for synthetic objective shapes & normalization
    ├── test_extrapolation_samplers.py# Unit tests for sub-domain, Dirichlet & ray-casting samplers
    ├── test_uq_evaluator.py          # Unit tests for Hutter LTV extraction & PLCB exploration extraction
    ├── test_extrapolation_metrics.py # Unit tests for statistical & calibration metrics
    └── test_extrapolation_runner.py  # End-to-end integration test with mock configurations
```

---

## Section 2: High-Performance Convex Hull QP Engine (`qp_convex_hull.py`)

### 2.1 Mathematical Solver Formulation
For training points $X_{\text{train}} \in \mathbb{R}^{N \times D}$ and query point $x \in \mathbb{R}^D$:
$$\min_{w \in \mathbb{R}^N} \frac{1}{2} w^T P w + q^T w \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1 \; \forall i$$

* **Hessian Matrix**: $P = X_{\text{train}} X_{\text{train}}^T \in \mathbb{R}^{N \times N}$
  - Precomputed **once** per training set in $\mathcal{O}(N^2 D)$.
* **Linear Cost Vector**: $q = - X_{\text{train}} x \in \mathbb{R}^N$
  - Evaluated per query in $\mathcal{O}(ND)$.
* **Solver Backend**:
  - Primary: `scipy.optimize.minimize(method='SLSQP')` with equality constraint `{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}` and bounds `[(0.0, 1.0)] * N`.
  - Fallback/Accelerated: `osqp` sparse solver with cached factorization.
* **Warm-Start Optimization**: Initial guess $w_0 = \frac{1}{N} \mathbf{1}_N$ (uniform weights).

### 2.2 Numerical Distance Extraction
* **Projection**: $x_{\text{proj}} = X_{\text{train}}^T w^*$
* **Raw Euclidean Distance**: $d(x, \text{Conv}) = \|x - x_{\text{proj}}\|_2$
* **Interpolation Classification**:
  $$x \in \text{Conv}(X_{\text{train}}) \iff d(x, \text{Conv}) < 10^{-6}$$
* **Normalized Metrics**:
  1. $d_{\text{norm}}(x) = \frac{d(x, \text{Conv})}{\sqrt{D}}$ (RMS coordinate distance per axis).
  2. $d_{\text{rel}}(x) = \frac{d(x, \text{Conv})}{2\sqrt{D}}$ (Domain diameter percentage $\in [0, 1]$).
  3. $d_\infty(x) = \max_{1 \le j \le D} |x_j - x_{\text{proj}, j}|$ (Chebyshev axis-aligned distance).

---

## Section 3: Synthetic Objective Functions (`test_objectives.py`)

Accepts input array $X \in \mathbb{R}^{M \times D}$ on domain $[-1, 1]^D$:

1. **Sphere (Convex, Smooth Baseline)**:
   $$f(x) = \sum_{j=1}^D x_j^2$$
2. **Rosenbrock (Ill-Conditioned Valley)**:
   $$f(x) = \sum_{j=1}^{D-1} \left[ 100 (x_{j+1} - x_j^2)^2 + (1 - x_j)^2 \right]$$
3. **Rastrigin (Highly Multimodal Optima)**:
   $$f(x) = 10 D + \sum_{j=1}^D \left[ x_j^2 - 10 \cos(2\pi x_j) \right]$$
4. **Ackley (Outer Plateau with Central Basin)**:
   $$f(x) = -20 \exp\left( -0.2 \sqrt{\frac{1}{D}\sum_{j=1}^D x_j^2} \right) - \exp\left( \frac{1}{D}\sum_{j=1}^D \cos(2\pi x_j) \right) + 20 + e$$

* **Target Standardization**:
  Compute sample mean $\bar{y}_{\text{train}}$ and standard deviation $s_{\text{train}}$ on the training sample. Normalize both training and test outputs:
  $$\tilde{y} = \frac{y - \bar{y}_{\text{train}}}{s_{\text{train}} + 10^{-8}}$$

---

## Section 4: Training & Test Sampling Protocols (`samplers.py`)

### 4.1 Sub-Domain Training Sampler
* **Objective**: Avoid convex hull volume saturation in low dimensions ($D=2, 3$).
* **Sampling**: $X_{\text{train}} \sim \mathcal{U}([-0.5, 0.5]^D)$ using Latin Hypercube Sampling (LHS) via `scipy.stats.qmc.LatinHypercube`.

### 4.2 Test Sampling Engines ($M = 10,000$ points)

#### Strategy 1: Natural Uniform Sampling ($M = 10,000$)
* Draw $X_{\text{test}} \sim \mathcal{U}([-1, 1]^D)$.
* Compute $d_{\text{norm}}(x_i)$ via QP solver.
* Record empirical interpolation rate: $\hat{P}_D(\text{interp}) = \frac{1}{M} \sum_{i=1}^M \mathbb{I}(d(x_i) < 10^{-6})$.

#### Strategy 2: Stratified Sampling via Ray-Casting ($2,500$ points per stratum)
1. **Stratum 0 (Pure Interpolation, $d=0$)**:
   Sample $w \sim \text{Dirichlet}(\alpha = \mathbf{1}_N)$ and construct $x = X_{\text{train}}^T w$.
2. **Strata 1–3 (Extrapolation via Ray-Casting)**:
   - For each stratum target count ($2,500$ points):
     1. Randomly select anchor $x_0 \in X_{\text{train}}$.
     2. Sample isotropic unit direction $v = \frac{z}{\|z\|_2}$ with $z \sim \mathcal{N}(0, I_D)$.
     3. Sample target distance $d_{\text{target}}$ uniformly in stratum range:
        - *Stratum 1 (Near)*: $d_{\text{target}} \in (0.00, 0.15]$
        - *Stratum 2 (Medium)*: $d_{\text{target}} \in (0.15, 0.40]$
        - *Stratum 3 (Far)*: $d_{\text{target}} \in (0.40, 0.80]$
     4. Point candidate: $x_{\text{cand}} = \text{clip}\left( x_0 + d_{\text{target}} \cdot \sqrt{D} \cdot v, \; -1.0, \; 1.0 \right)$.
     5. Solve QP to get true $d_{\text{norm}}(x_{\text{cand}})$. If it falls inside the target stratum, accept; else re-sample.

---

## Section 5: Dual UQ Inference Engine (`uq_evaluator.py`)

### 5.1 Base Random Forest Configuration (Standard SMAC3 Parameters)
Both UQ quantifiers evaluate the **exact same underlying tree ensemble** fitted on $(X_{\text{train}}, \tilde{y}_{\text{train}})$:
* Surrogate model: `sklearn.ensemble.RandomForestRegressor`
  - `n_estimators`: 10 (SMAC HPO facade default; with 100 evaluated as mature forest setting)
  - `max_features`: $5/6 \approx 0.8333$
  - `min_samples_split`: 3
  - `min_samples_leaf`: 3
  - `splitter`: `'random'` (Extremely Randomized Trees)
  - `criterion`: `'squared_error'`
  - `bootstrapping`: `True`
  - `min_impurity_decrease`: `1e-8`
  - `oob_score`: `True`
  - `random_state`: seed

### 5.2 Standard SMAC3 LCB ($U_{\text{SLCB}}$ via Hutter Law of Total Variance)
For test points $X_{\text{test}} \in \mathbb{R}^{M \times D}$:
1. **Tree Predictions**: Query each tree $b \in \{1, \dots, B\}$: $T_b(X_{\text{test}}) \in \mathbb{R}^{M}$.
   - Mean prediction: $\hat{\mu}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$.
2. **Between-Tree Disagreement**:
   $$\sigma^2_{\text{between}}(x) = \frac{1}{B} \sum_{b=1}^B \left( T_b(x) - \hat{\mu}(x) \right)^2$$
3. **Within-Tree Leaf Variance (Leaf Impurity)**:
   For each tree $b$, determine the leaf node index for query $x$: $\text{leaf}_b = \text{tree}_b.\text{apply}(x)$.
   Extract leaf empirical variance directly from tree metadata:
   $$\sigma^2_{b, \text{within}}(x) = \text{tree}_b.\text{tree\_}.\text{impurity}[\text{leaf}_b]$$
   Mean within-tree variance:
   $$\sigma^2_{\text{within}}(x) = \frac{1}{B} \sum_{b=1}^B \sigma^2_{b, \text{within}}(x)$$
4. **Total Variance & Uncertainty**:
   $$\sigma^2_{\text{total}}(x) = \sigma^2_{\text{between}}(x) + \sigma^2_{\text{within}}(x)$$
   $$U_{\text{SLCB}}(x) = 1.96 \cdot \sqrt{\sigma^2_{\text{total}}(x)}$$

### 5.3 Proximity LCB ($U_{\text{PLCB}}$: Pure Extracted Exploration Term)
Using `GPUProximityRegressionUQ` / `ProximityBExtractor`:
* **Hyperparameters**:
  - $k = 28$
  - $\epsilon = 0.080791$
  - $\lambda = 0.20486$
  - $\text{level} = 0.95$ ($\kappa = 1.96$)
* **Interval Query**:
  Call `predict_with_intervals(X_{\text{test}}, n_neighbors=28, level=0.95, return_mae=True)`.
  Returns `y_pred_lwr`, `y_pred`, `y_pred_upr`, `local_mae`.
* **Lower Residual Quantile Extraction**:
  $$q_{0.025}^{(k)}(x) = y_{\text{pred\_lwr}}(x) - \hat{\mu}(x) \le 0$$
* **Point-Adaptive Exploration Floor**:
  $$\delta_{\text{floor}}(x) = \epsilon \cdot 1.96 \cdot \text{local\_mae}(x)$$
* **Extracted Exploration Uncertainty**:
  $$U_{\text{PLCB}}(x) = \max\left( \delta_{\text{floor}}(x), \; \left| q_{0.025}^{(k)}(x) \right| \right) = \max\left( \delta_{\text{floor}}(x), \; -q_{0.025}^{(k)}(x) \right)$$

---

## Section 6: Evaluation Metrics Engine (`metrics_suite.py`)

Metrics computed globally and partitioned by stratum ($d_{\text{norm}}$ bands):

1. **Distance Monotonicity**:
   $$\rho_{\text{dist}}(U) = \text{SpearmanRank}\left( d_{\text{norm}}(X_{\text{test}}), \; U(X_{\text{test}}) \right)$$
2. **Error Alignment**:
   $$\rho_{\text{err}}(U) = \text{SpearmanRank}\left( |\tilde{y} - \hat{\mu}|, \; U(X_{\text{test}}) \right)$$
3. **Empirical Coverage (Target = 0.95)**:
   $$\text{PICP} = \frac{1}{M}\sum_{i=1}^M \mathbb{I}\left( |\tilde{y}_i - \hat{\mu}_i| \le U(x_i) \right)$$
4. **Sharpness (Interval Width)**:
   $$\text{MPIW} = \frac{1}{M}\sum_{i=1}^M 2 \cdot U(x_i)$$
5. **Winkler Interval Score ($\alpha = 0.05$)**:
   With symmetric half-width $U(x)$, $y_{\text{lwr}} = \hat{\mu} - U$, $y_{\text{upr}} = \hat{\mu} + U$:
   $$S_\alpha(x) = 2U(x) + \frac{2}{0.05} \max\left(0, \; (\hat{\mu} - U) - \tilde{y}\right) + \frac{2}{0.05} \max\left(0, \; \tilde{y} - (\hat{\mu} + U)\right)$$
6. **Catastrophic Error Outlier Discrimination (AUROC / AUPRC)**:
   Binary target: $Y_{\text{outlier}} = \mathbb{I}\left( |\tilde{y} - \hat{\mu}| > \text{Quantile}_{0.90}(|\tilde{y} - \hat{\mu}|) \right)$.
   Compute Area Under ROC and PR curves using $U(x)$ as detector score.

---

## Section 7: Sweep Execution & Telemetry Streaming (`runner.py`)

### 7.1 Multi-Dimensional Grid
* **Dimensions**: $D \in \{2, 3, 5, 8, 16, 32\}$ (6 values)
* **Budgets**: $N \in \{112, 224, 448, 896\}$ (for $k=28$; 4 values)
* **Functions**: Sphere, Rosenbrock, Rastrigin, Ackley (4 functions)
* **Strategies**: Natural Uniform vs. Controlled Stratified (2 strategies)
* **Seeds**: 10 paired seeds
* **Total Runs**: $6 \times 4 \times 4 \times 2 \times 10 = \mathbf{1,920 \text{ runs}}$ ($19,200,000$ test evaluations).

### 7.2 Storage Hierarchy
* Raw per-run Parquet:
  `results/extrapolation_uq/raw/dim={D}/n={N}/func={func}/run_{seed}_{strategy}.parquet`
* Slice Summary Scorecard:
  `results/extrapolation_uq/analysis/extrapolation_calibration_scorecard.csv`

---

## Section 8: Step-by-Step TDD Phasing

### Phase 1: Convex Hull QP Engine
* **Test file**: `tests/test_qp_convex_hull.py`
  - Test 1: $d_{\text{norm}} = 0.0$ for exact training points and Dirichlet interior combinations.
  - Test 2: $d_{\text{norm}} > 0.0$ for points shifted outside bounding hull with exact analytic distance.
  - Test 3: Gram matrix caching produces identical projection to direct optimization.
  - Test 4: Projection solver handles $D=32$ within $< 2\text{ms}$ per query.
* **Module**: `dyrf_bo/extrapolation_uq/qp_convex_hull.py`

### Phase 2: Synthetic Functions & Samplers
* **Test files**: `tests/test_test_objectives.py`, `tests/test_extrapolation_samplers.py`
  - Test 1: Vectorized objectives match known analytical values across $D \in \{2, 16, 32\}$.
  - Test 2: Sub-domain training points strictly obey $[-0.5, 0.5]^D$.
  - Test 3: Ray-casting fills near, medium, and far strata with exact target quotas.
* **Modules**: `dyrf_bo/extrapolation_uq/test_objectives.py`, `dyrf_bo/extrapolation_uq/samplers.py`

### Phase 3: Dual UQ Inference Engine
* **Test file**: `tests/test_uq_evaluator.py`
  - Test 1: SMAC RF fits with native defaults (`splitter='random'`, `ratio_features=5/6`, `min_samples_leaf=3`).
  - Test 2: Hutter Law of Total Variance correctly combines between-tree and within-leaf impurity variance.
  - Test 3: Proximity LCB exploration term extracts $\max(\delta_{\text{floor}}, |q_{0.025}|) > 0$.
  - Test 4: Both UQ evaluators run on the identical underlying forest instance.
* **Module**: `dyrf_bo/extrapolation_uq/uq_evaluator.py`

### Phase 4: Metrics & Summary Runner
* **Test files**: `tests/test_extrapolation_metrics.py`, `tests/test_extrapolation_runner.py`
  - Test 1: PICP, MPIW, and Winkler score exactness on verified numerical mocks.
  - Test 2: Spearman $\rho$ monotonicity computation.
  - Test 3: End-to-end pilot run on $D=2$, $N=112$, Sphere function producing valid Parquet and summary CSV rows.
* **Modules**: `dyrf_bo/extrapolation_uq/metrics_suite.py`, `dyrf_bo/extrapolation_uq/runner.py`
