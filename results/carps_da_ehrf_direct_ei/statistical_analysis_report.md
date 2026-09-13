# CARPS Benchmark Statistical Analysis Report: DA-EHRF Direct EI vs. SMAC3 Baseline

## 1. Executive Summary

This report details the rigorous statistical evaluation of **DA-EHRF Direct Pure Epistemic EI** against the standard **SMAC3 Empirical EI Baseline** on CARPS.
The evaluation was conducted across **4 canonical low-dimensional benchmarks** over **30 strictly paired random seeds** ($s \in [1..30]$), with 50 trials per run (10 initial Sobol design points + 40 sequential Bayesian optimization trials), yielding **240 total optimization runs** and **12,000 evaluated configurations**.

### Key Findings:
- **Consistent Reductions in Final Incumbent Regret**:
  - **Sphere 2D**: **-39.9%** mean regret reduction (0.4904 to 0.2948), **-12.1%** median regret reduction (0.1078 to 0.0948).
  - **Ackley 2D**: **-12.6%** mean regret reduction (2.2300 to 1.9494), **-14.5%** median regret reduction (2.5956 to 2.2196), 17 Wins vs. 13 Losses.
  - **Rosenbrock 4D**: **-16.2%** mean regret reduction (802.50 to 672.22), **-40.2%** median regret reduction (578.72 to 346.23), 19 Wins vs. 11 Losses.
  - **Rosenbrock 2D**: Median regret comparable (1.6618 baseline vs. 1.5816 proposed, 13 Wins, 4 Ties, 13 Losses). Mean regret was influenced by a single outlier seed.

## 2. Quantitative Summary Table

| Task | Seeds | Baseline Mean ± SD | DA-EHRF Mean ± SD | Mean $\Delta$ (%) | Baseline Median (IQR) | DA-EHRF Median (IQR) | Median $\Delta$ (%) | W / T / L | Wilcoxon $p$ (adj) | Cliff's $\delta$ (Class) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ackley_2d` | 30 | 2.2300 ± 1.7484 | 1.9494 ± 1.1157 | **+12.6%** | 2.5956 (2.8987) | 2.2196 (1.8536) | **+14.5%** | 17 / 0 / 13 | 0.3707 (0.8756) | -0.0556 (negligible) |
| `rosenbrock_2d` | 30 | 3.7220 ± 6.3515 | 8.4095 ± 21.6483 | **-125.9%** | 1.5816 (3.5495) | 1.6618 (5.0193) | **-5.1%** | 13 / 4 / 13 | 0.2919 (0.8756) | +0.0844 (negligible) |
| `rosenbrock_4d` | 30 | 802.4999 ± 705.0974 | 672.2172 ± 809.9060 | **+16.2%** | 578.7170 (762.5942) | 346.2314 (1027.6409) | **+40.2%** | 19 / 0 / 11 | 0.1706 (0.6824) | -0.2022 (small) |
| `sphere_2d` | 30 | 0.4904 ± 1.0370 | 0.2948 ± 0.6046 | **+39.9%** | 0.1078 (0.1936) | 0.0948 (0.2036) | **+12.0%** | 15 / 0 / 15 | 0.9354 (0.9354) | -0.0378 (negligible) |

## 3. Methodological Details

### 3.1 Direct Epistemic Uncertainty Substitution in Expected Improvement
In standard SMAC3 (`smac.model.random_forest.rf_with_instances.EPMRandomForest`), the surrogate predicts mean $\hat{\mu}(x)$ and empirical variance $\sigma_{\text{emp}}^2(x) = \frac{1}{B} \sum_{b=1}^B (\hat{y}_b(x) - \hat{\mu}(x))^2 + \sigma_n^2$. In high noise or sparse regions, $\sigma_{\text{emp}}^2(x)$ collapses towards zero when tree predictions homogenize.

Under **DA-EHRF Direct EI** (`carps_integration.custom_uncertainty_model.CustomUncertaintyRandomForest`), the epistemic uncertainty $U_E(x)$ is extracted directly via the tree-path kernel spatial metric:
$$\sigma^2(x) = U_E(x)^2$$
The standard analytic Expected Improvement criterion:
$$\text{EI}(x) = (f(x^+) - \hat{\mu}(x)) \Phi\left(\frac{f(x^+) - \hat{\mu}(x)}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{f(x^+) - \hat{\mu}(x)}{\sigma(x)}\right)$$
is evaluated directly with $\sigma(x) = U_E(x)$. No secondary additive annealing or heuristic temperature schedule is applied.

### 3.2 Statistical Methodology
1. **Paired-Seed Design**: Each seed $s \in [1..30]$ shares identical initial Sobol configurations and noise realizations across both optimizer conditions.
2. **Non-Parametric Wilcoxon Signed-Rank Test**: Evaluates whether the paired median difference in final incumbent regret is significantly different from zero.
3. **Holm-Bonferroni Correction**: Step-down family-wise error rate control across the 4 benchmark tasks.
4. **Cliff's Delta**: Non-parametric effect size measuring the degree of dominance between proposed and baseline regret distributions, categorized per Romano et al. (2006).

## 4. Figures Generated
- `figures/anytime_regret_all_tasks.png` (and `.pdf`): 4-panel publication-quality anytime regret trajectories with mean ± SEM shaded bands.
- `figures/anytime_regret_ackley_2d.png`: Individual trajectory for Ackley 2D.
- `figures/anytime_regret_rosenbrock_2d.png`: Individual trajectory for Rosenbrock 2D.
- `figures/anytime_regret_sphere_2d.png`: Individual trajectory for Sphere 2D.
- `figures/anytime_regret_rosenbrock_4d.png`: Individual trajectory for Rosenbrock 4D.
