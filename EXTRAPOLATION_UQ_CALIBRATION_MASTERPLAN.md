# Masterplan: Uncertainty Calibration in the Extrapolation Regime
## Proximity LCB ($k$-NN OOB Residuals) vs. Standard SMAC3 LCB (Parametric RF Variance)

---

### 📍 Document Provenance
- **Repository**: `DyRF-BO`
- **Branch**: `feat/extrapolation-uq-calibration`
- **Related Modules**:
  - `Proximity_Regression_UQ.py` (`rfgap` / RF-FIRE empirical quantile prediction intervals)
  - `Epistemic_Quantifier.py`
  - `synthetic_functions.py`
  - `metrics.py`

---

## 1. Research Motivation & Core Hypothesis

### 1.1 The Theoretical Problem
In Bayesian Optimization (BO), acquisition functions such as Lower Confidence Bound (LCB) balance exploitation and exploration:
$$\alpha_{\text{LCB}}(x) = - \left( \hat{\mu}(x) - \beta^{1/2} \cdot U(x) \right)$$
Standard SMAC3 with a Random Forest surrogate estimates uncertainty $U_{\text{SLCB}}(x) = \sigma_{\text{RF}}(x)$ as the empirical standard deviation of predictions across the individual trees in the ensemble (or the law of total variance incorporating within-leaf noise). 

However, **axis-aligned decision trees partition space into orthogonal hypercubes with constant leaf predictions**:
1. Outside the bounding box / convex hull of training observations ($\text{Conv}(X_{\text{train}})$), splits no longer bisect the query region.
2. Every tree's leaf extends to infinity with a constant value.
3. As a query point $x$ moves arbitrarily far into the extrapolation regime, the ensemble variance $\sigma_{\text{RF}}^2(x)$ **plateaus or collapses** to an arbitrary constant, completely decoupled from the distance to the training data.

### 1.2 The Curse of Dimensionality Connection
In high dimensions ($D \ge 16$):
* The volume of the unit hypercube $[-1, 1]^D$ is concentrated in its outer corners and thin boundary shells.
* The convex hull of any polynomial number of training points $N$ covers an asymptotically negligible fraction of the domain volume:
  $$\lim_{D \to \infty} P_{x \sim \mathcal{U}}\left(x \in \text{Conv}(X_{\text{train}})\right) = 0$$
* Therefore, **high-dimensional Bayesian Optimization operates almost entirely in the extrapolation regime**.

### 1.3 The Core Hypothesis
> **Hypothesis**: Proximity LCB (PLCB) significantly outperforms Standard LCB (SLCB) in high dimensions ($D \ge 16$) because Standard LCB's uncertainty quantifier suffers catastrophic miscalibration and variance plateauing under extrapolation, whereas PLCB—which constructs confidence intervals from the Out-Of-Bag (OOB) residuals of the $k$-Nearest Neighbors in proximity space—retains monotonic error-tracking and robust coverage outside the convex hull.

---

## 2. Formal Uncertainty Quantification Definitions

To compare apples-to-apples at an exact **nominal 95% confidence level**, we formalize:

### 2.1 Standard SMAC3 LCB Uncertainty ($U_{\text{SLCB}}$)
Assuming Gaussian predictive distribution around ensemble mean $\hat{\mu}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)$:
* **Ensemble Variance**: $\sigma_{\text{RF}}^2(x) = \frac{1}{B}\sum_{b=1}^B \left( T_b(x) - \hat{\mu}(x) \right)^2$
* **95% Half-Width**:
  $$U_{\text{SLCB}}(x) = 1.96 \cdot \sigma_{\text{RF}}(x)$$
* **95% Prediction Interval**:
  $$\mathcal{I}_{\text{SLCB}}(x) = \left[ \hat{\mu}(x) - 1.96 \cdot \sigma_{\text{RF}}(x), \; \hat{\mu}(x) + 1.96 \cdot \sigma_{\text{RF}}(x) \right]$$

### 2.2 Proximity LCB Uncertainty ($U_{\text{PLCB}}$ via RF-FIRE / RFGAP)
For query point $x$, compute ensemble proximity weights $W(x, x_i)$ to training samples $x_i \in X_{\text{train}}$, extract the $k$-nearest neighbors $\mathcal{N}_k(x)$, and evaluate their Out-Of-Bag (OOB) residuals $r_i^{\text{OOB}} = y_i - \hat{\mu}_{-i}(x_i)$:
* **Empirical Quantiles**: Let $q_{\alpha/2}^{(k)}(x)$ and $q_{1 - \alpha/2}^{(k)}(x)$ be the empirical $\alpha/2 = 0.025$ and $1 - \alpha/2 = 0.975$ quantiles of $\{r_i^{\text{OOB}} \mid i \in \mathcal{N}_k(x)\}$.
* **95% Prediction Interval**:
  $$\mathcal{I}_{\text{PLCB}}(x) = \left[ \hat{\mu}(x) + q_{0.025}^{(k)}(x), \; \hat{\mu}(x) + q_{0.975}^{(k)}(x) \right] = \left[ \hat{y}_{\text{lwr}}(x), \; \hat{y}_{\text{upr}}(x) \right]$$
* **Symmetric Half-Width Metric**:
  $$U_{\text{PLCB}}(x) = \frac{\hat{y}_{\text{upr}}(x) - \hat{y}_{\text{lwr}}(x)}{2}$$

---

## 3. Geometric Formulation: High-Dimensional Convex Hull Distance

### 3.1 Definition of Interpolation vs. Extrapolation
A query point $x \in \mathbb{R}^D$ is defined as:
* **Interpolating** iff $x \in \text{Conv}(X_{\text{train}})$
* **Extrapolating** iff $x \notin \text{Conv}(X_{\text{train}})$

### 3.2 Quadratic Program (QP) Distance Solver & Standard Form
Rather than constructing geometric facet representations (which fails in $D > 8$ with $\mathcal{O}(N^{\lfloor D/2 \rfloor})$ complexity), the exact Euclidean projection onto $\text{Conv}(X_{\text{train}})$ is formulated as a convex Quadratic Program:

$$\min_{w \in \mathbb{R}^N} \frac{1}{2} \left\| X_{\text{train}}^T w - x \right\|_2^2 \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0 \; \forall i$$

Expanding the squared Euclidean objective:
$$\frac{1}{2} \left\| X_{\text{train}}^T w - x \right\|_2^2 = \frac{1}{2} w^T \left( X_{\text{train}} X_{\text{train}}^T \right) w - \left( X_{\text{train}} x \right)^T w + \frac{1}{2} \| x \|_2^2$$

Dropping the constant term $\frac{1}{2} \| x \|_2^2$ yields the canonical QP formulation ($\min_w \frac{1}{2} w^T P w + q^T w$):
* **Hessian Matrix**: $P = X_{\text{train}} X_{\text{train}}^T \in \mathbb{R}^{N \times N}$
* **Linear Cost Vector**: $q = - X_{\text{train}} x \in \mathbb{R}^N$
* **Constraints**: $\mathbf{1}^T w = 1, \quad 0 \le w_i \le 1 \; \forall i$

### 3.3 High-Throughput Gram Matrix Precomputation
In our experimental setup, $M = 10,000$ test queries must be evaluated per configuration:
* **One-Time Precomputation**: The Gram matrix $P = X_{\text{train}} X_{\text{train}}^T$ is symmetric positive semidefinite and **depends exclusively on the training points**. It is precomputed once per seed in $\mathcal{O}(N^2 D)$.
* **Per-Query Efficiency**: For each of the $10,000$ test points $x_j$, only the linear cost vector $q_j = - X_{\text{train}} x_j$ needs to be evaluated via a fast matrix-vector product ($\mathcal{O}(ND)$).
* **Solver Warm-Starts**: The constant Hessian structure across all $M$ queries allows solver factorization reuse (e.g., in OSQP or QP interior-point solvers), reducing per-point projection solve times to sub-millisecond scale.

### 3.4 Distance Extraction & Dimension Normalization

Once the optimal weights $w^* \in \Delta^{N-1}$ are obtained:
* **Projection**: $x_{\text{proj}} = X_{\text{train}}^T w^*$
* **Raw Euclidean Distance**: $d(x, \text{Conv}) = \| x - x_{\text{proj}} \|_2$
* **Interpolation Criterion**: $x \in \text{Conv}(X_{\text{train}}) \iff d(x, \text{Conv}) < 10^{-7}$

#### Why Dimension Normalization is Mandatory
In Euclidean space $\mathbb{R}^D$, distances scale as $\sqrt{D}$. For a domain $\mathcal{X} = [-1, 1]^D$:
* In $D = 2$, the maximum distance between corners is $\sqrt{2^2 + 2^2} = \sqrt{8} \approx 2.83$.
* In $D = 32$, the maximum distance between corners is $\sqrt{32 \times 2^2} = \sqrt{128} \approx 11.31$.
If raw Euclidean distance is used, a cutoff threshold (e.g. $d = 0.5$) represents significant extrapolation in $D=2$, but a negligible fraction of the space in $D=32$.

#### Dimension-Normalized Metrics Recorded in the Study
1. **Root-Mean-Square (RMS) Distance ($d_{\text{norm}}$ - Primary Metric)**:
   $$d_{\text{norm}}(x) = \frac{\| x - x_{\text{proj}} \|_2}{\sqrt{D}} = \sqrt{\frac{1}{D} \sum_{j=1}^D \left( x_j - x_{\text{proj}, j} \right)^2}$$
   - **Interpretation**: The average Euclidean displacement per coordinate axis.
   - **Domain Invariance**: Under this metric, the hypercube's maximum diagonal length is normalized to $\frac{2\sqrt{D}}{\sqrt{D}} = 2.0$ for all dimensions $D \in \{2, \dots, 32\}$.
   - **Stratification Stability**: Enables consistent stratification thresholds across all dimensions (e.g. near: $\le 0.15$, medium: $0.15 - 0.40$, far: $> 0.40$).

2. **Domain-Diameter Relative Metric ($d_{\text{rel}}$)**:
   $$d_{\text{rel}}(x) = \frac{d(x, \text{Conv})}{\text{diam}(\mathcal{X})} = \frac{\| x - x_{\text{proj}} \|_2}{2\sqrt{D}} \in [0, 1]$$
   - Directly expresses distance as a percentage of the maximum possible span of the space.

3. **Chebyshev ($L_\infty$) Coordinate Distance ($d_\infty$)**:
   $$d_\infty(x) = \| x - x_{\text{proj}} \|_\infty = \max_{1 \le j \le D} |x_j - x_{\text{proj}, j}|$$
   - **Relevance to Tree Models**: Because Random Forest splits are axis-aligned orthogonal cuts, $d_\infty$ captures the maximum extrapolation distance along any single decision feature.



---

## 4. Experimental Design Matrix

### 4.1 Factors & Parameter Grid
* **Dimensions ($D$)**: $\{2, 3, 5, 8, 16, 32\}$
* **Hyperparameter $k$**: $k = 10$ (primary default in `DyRF-BO`), with validation on $k = 25$.
* **Training Sample Multipliers ($N/k$)**: $N/k \in \{4, 8, 16, 32\}$
  * For $k = 10 \implies N \in \{40, 80, 160, 320\}$
  * For $k = 25 \implies N \in \{100, 200, 400, 800\}$
* **Replications**: $10$ independent pseudo-random seeds per configuration tuple.
* **Test Points Budget**: $M = 10,000$ points per seed.

### 4.2 Synthetic Test Objective Functions
All functions scalable to arbitrary dimension $D$ on domain $\mathcal{X} = [-1, 1]^D$:
1. **Sphere (Convex, Isotropic, Smooth Baseline)**:
   $$f_1(x) = \sum_{j=1}^D x_j^2$$
2. **Rosenbrock (Ill-Conditioned, Anisotropic Valley)**:
   $$f_2(x) = \sum_{j=1}^{D-1} \left[ 100 (x_{j+1} - x_j^2)^2 + (1 - x_j)^2 \right]$$
3. **Rastrigin (Highly Multimodal, Frequent Local Optima)**:
   $$f_3(x) = 10 D + \sum_{j=1}^D \left[ x_j^2 - 10 \cos(2 \pi x_j) \right]$$
4. **Ackley (Outer Plateau with Sharp Central Basin)**:
   $$f_4(x) = -20 \exp\left( -0.2 \sqrt{\frac{1}{D}\sum_{j=1}^D x_j^2} \right) - \exp\left( \frac{1}{D}\sum_{j=1}^D \cos(2\pi x_j) \right) + 20 + e$$

*Target Standardization*: Targets are normalized via training statistics: $\tilde{y} = \frac{y - \bar{y}_{\text{train}}}{s_{\text{train}}}$.

---

## 5. Sampling Protocols

### 5.1 Training Set Generation: Sub-Domain Inward Sampling
To prevent the convex hull from dominating the entire volume in low dimensions ($D = 2, 3$):
* **Training Region**: Sample $X_{\text{train}} \sim \mathcal{U}([-c, c]^D)$ with $c = 0.5$.
* **Evaluation Region**: Test points span the full bounding space $\mathcal{X}_{\text{test}} = [-1, 1]^D$.
* **Property**: Guarantees a defined extrapolation margin in all dimensions while retaining uniform density within the core training sub-domain.

### 5.2 Test Set Generation Strategies ($M = 10,000$ total)

#### Strategy 1: Natural Uniform Sampling ($M = 10,000$)
* Sample $x \sim \mathcal{U}([-1, 1]^D)$.
* Solve QP for each point to compute $d_{\text{norm}}(x)$.
* Compute empirical interpolation probability $\hat{P}_D(\text{interp}) = \frac{1}{M} \sum_{i=1}^M \mathbb{I}(d(x_i) = 0)$.

#### Strategy 2: Controlled / Distance-Stratified Sampling ($2,500$ points per stratum)
* **Stratum 0 (Pure Interpolation, $d = 0$)**:
  Generated directly via random convex combinations of training samples:
  $$x = \sum_{i=1}^N w_i x_i, \quad w \sim \text{Dirichlet}(\mathbf{1}_N)$$
* **Stratum 1 (Near Extrapolation)**: $0 < d_{\text{norm}}(x) \le 0.15$
* **Stratum 2 (Medium Extrapolation)**: $0.15 < d_{\text{norm}}(x) \le 0.40$
* **Stratum 3 (Far Extrapolation)**: $d_{\text{norm}}(x) > 0.40$ (points near the $[-1, 1]^D$ boundary).

---

## 6. Evaluation Metrics & Statistical Analysis

For each test point $i \in \{1, \dots, M\}$, we record:
$$\left( y_i, \hat{y}_i, e_i = |y_i - \hat{y}_i|, U_{\text{SLCB}}(x_i), U_{\text{PLCB}}(x_i), \mathcal{I}_{\text{SLCB}}(x_i), \mathcal{I}_{\text{PLCB}}(x_i), d_{\text{norm}}(x_i), \text{stratum} \right)$$

### 6.1 Epistemic Distance Monotonicity
Does the uncertainty quantifier scale with physical departure from the training data?
* **Metric**: Spearman rank correlation with normalized hull distance:
  $$\rho_{\text{dist}}(U) = \text{SpearmanRank}\left( d_{\text{norm}}(x), \; U(x) \right)$$
* **Hypothesis**: $\rho_{\text{dist}}(U_{\text{PLCB}}) \gg \rho_{\text{dist}}(U_{\text{SLCB}}) \approx 0$ in high dimensions.

### 6.2 Error-Uncertainty Calibration & Monotonicity
Does higher uncertainty correspond to higher true surrogate error?
* **Metric**: Spearman rank correlation with actual absolute residual:
  $$\rho_{\text{err}}(U) = \text{SpearmanRank}\left( |y - \hat{y}|, \; U(x) \right)$$
  (Computed globally and stratified by distance band).

### 6.3 Interval Coverage & Sharpness (Nominal 95% Confidence)
* **Prediction Interval Coverage Probability (PICP)**:
  $$\text{PICP} = \frac{1}{M} \sum_{i=1}^M \mathbb{I}\left( y_i \in \mathcal{I}(x_i) \right) \quad (\text{Target: } 0.95)$$
* **Mean Prediction Interval Width (MPIW)**:
  $$\text{MPIW} = \frac{1}{M} \sum_{i=1}^M \text{width}(\mathcal{I}(x_i))$$
* **Interval Score (Winkler Score)**: Penalizes both wide intervals and coverage violations:
  $$S_\alpha(\mathcal{I}, y) = \text{width} + \frac{2}{\alpha}(y_{\text{lwr}} - y)\mathbb{I}(y < y_{\text{lwr}}) + \frac{2}{\alpha}(y - y_{\text{upr}})\mathbb{I}(y > y_{\text{upr}})$$

### 6.4 Proper Scoring Rules
* **Continuous Ranked Probability Score (CRPS)**.

### 6.5 Catastrophic Error Discrimination
* **AUROC & AUPRC**:
  Binary classification task: identify test points with top 10% highest absolute residuals ($e_i > Q_{0.90}(e)$) using the uncertainty score $U(x)$ as the detection score.

---

## 7. Data Logging & Storage Architecture

### 7.1 Per-Point Raw Telemetry (`point_evaluations.parquet`)
```
- run_id: uuid
- seed: int
- function: str
- dimension: int
- n_train: int
- k: int
- sampling_strategy: str (natural vs. stratified)
- stratum: str (interp, near_extrap, med_extrap, far_extrap)
- d_norm: float
- y_true: float
- y_hat: float
- abs_error: float
- u_slcb: float
- u_plcb: float
- slcb_covered: bool
- plcb_covered: bool
- slcb_width: float
- plcb_width: float
```

### 7.2 Aggregated Summary Scorecard (`calibration_summary.csv`)
Grouped by `(dimension, n_train, function, sampling_strategy, stratum)`:
* Mean $\rho_{\text{dist}}$ (SLCB vs PLCB)
* Mean $\rho_{\text{err}}$ (SLCB vs PLCB)
* PICP (SLCB vs PLCB)
* MPIW (SLCB vs PLCB)
* Mean Winkler Score (SLCB vs PLCB)
* Error AUROC / AUPRC (SLCB vs PLCB)
* Wilcoxon signed-rank $p$-values across paired seeds.

---

## 8. Execution Phasing & Milestones

* **Phase 1: Architecture & Solvers (Test-First)**
  - Implement fast QP convex hull projection solver using vectorized quadratic programming (`scipy.optimize` / `osqp`).
  - Implement unit tests verifying:
    - Points generated via convex combination have $d(x) = 0$.
    - Points outside bounding box have strictly positive distance matching geometric projection.
* **Phase 2: Data Generator & Sampling Engine**
  - Implement sub-domain inward generator ($c=0.5$).
  - Implement Dirichlet interior sampler + stratified outward normal perturbations.
* **Phase 3: Experiment Harness**
  - Connect `ProximityRegressionUQ` (`rfgap` OOB residual quantiles) and baseline `RandomForestRegressor` ensemble variance.
  - Run pilot benchmark on $D \in \{2, 16\}$ with $N \in \{40, 160\}$ on Sphere & Rosenbrock.
* **Phase 4: Full Multi-D Cluster Sweep**
  - Execute full grid: 6 dimensions $\times$ 4 sample budgets $\times$ 4 functions $\times$ 2 strategies $\times$ 10 seeds = 3,840 runs.
* **Phase 5: Synthesis & Thesis Artifacts**
  - Generate calibration curves, distance-uncertainty trajectory plots, and thesis-ready summary tables.
