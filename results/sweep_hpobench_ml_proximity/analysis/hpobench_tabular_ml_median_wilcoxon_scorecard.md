# Statistical Scorecard: hpobench_tabular_ml

## Experimental Setup

- **Benchmark Suite**: `hpobench_tabular_ml` (80 tasks)
- **Total Budget / Iterations**: 100 trials per run
- **Stage 1 (Initial Design Phase, Trials 1–10)**: 10 trials (`SobolInitialDesign`, quasi-random initialization, **100% identical configurations across both methods**)
- **Stage 2 (LCB Warmup Phase, Trials 11–28)**: 18 trials of Standard RF LCB with $\beta=3.8416$ (to accumulate observations until $N > k=28$ for topological neighbor graphs)
- **Stage 3 (Active Proximity BO Phase, Trials 29–100)**: 72 trials where meta-optimized Proximity-LCB acquisition diverges and guides search
- **Seeds**: 30 independent runs per task (seeds 1 to 30)
- **Total Runs Evaluated**: 80 tasks × 2 approaches × 30 seeds = 4,800 runs (480,000 trials)

### Approaches & Hyperparameters

1. **Proposed Approach (`SMAC20_ProximityLCB`) — Meta-Optimized via SMAC4HPO**:
   - **Source**: Meta-tuned on independent dev benchmark tasks (`results/meta_smac_proximity_hpo/best_config.json`)
   - **Surrogate**: `CustomUncertaintyRandomForest` with localized epistemic uncertainty (`proximity_b`)
   - **Kernel Distance Decay**: $\lambda = 0.20486$
   - **Acquisition Function**: `proximity_lcb`
   - **Proximity Hyperparameters**: $k = 28$ nearest neighbors, $\text{level} = 0.95$ (95% empirical quantile), $\epsilon = 0.080791$ (dispersion floor buffer)

2. **Baseline Approach (`SMAC3_HPOFacade_lcb`)**:
   - **Surrogate**: Standard SMAC3 Random Forest surrogate (`smac.facade.HyperparameterOptimizationFacade`)
   - **Acquisition Function**: Standard Lower Confidence Bound (`lcb`)
   - **Exploration Weight**: $\beta = 3.8416$ (corresponding to $\kappa = \sqrt{\beta} = 1.96$, fixed, `update_beta = False`)

## Summary Metrics Across All Tasks

| Metric | Proposed (`SMAC20_ProximityLCB`) | Baseline (`SMAC3_HPOFacade_lcb`) | Net Advantage |
| :--- | :--- | :--- | :--- |
| **Mean of Medians** (lower is better) | `0.102550` | `0.102296` | `Δ = 0.000254` |
| **Task Wins** | **5** (6.2%) | 20 (25.0%) | **+-15 tasks** |
| **Task Ties** | 55 (68.8%) | 55 (68.8%) | — |

## Hypothesis Testing & Effect Sizes

- **One-Sided Wilcoxon Signed-Rank Test (`H1: Proposed < Baseline`)**:
  - Statistic ($W$): `334.0`
  - $p$-value: `9.9857e-01` (Not Statistically Significant)
- **Paired Task Dominance Metric**: **`-0.1875`**
  - Calculated as $(W_{\text{wins}} - W_{\text{losses}}) / N = (5 - 20) / 80 = -18.8\%$
- **Cross-Task Unpaired Cliff's Delta ($\delta$)**:
  - Value: `0.0034`
  - Interpretation: **Negligible** effect (negligible due to cross-task baseline scale variance)
