# BBOB High-D & Extreme Evaluation Scorecard: Proximity LCB vs SMAC3 EI

Evaluation comparing **SMAC20_ProximityLCB** ($k=25, \lambda=1.345, \epsilon=0.16$) vs **SMAC3_HPOFacade_ei** (native Expected Improvement $\xi=0.0$) across 30 seeds.

## 1. Aggregate Scorecard by Dimensionality

| Stratum | Tasks | Paired Runs | Win / Loss / Tie | Win Rate | Mean Regret (Prop vs Base) | Regret Reduction | Cliff's Delta | Demšar Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall (D=16 & D=32) | 144 | 4320 | 2297 / 1932 / 91 | 53.2% | 0.4486 vs 0.4733 | +5.2% | -0.0096 | 7.7269e-04 |
| High-D (D=16) | 75 | 2250 | 1285 / 915 / 50 | 57.1% | 0.4223 vs 0.4811 | +12.2% | -0.0173 | 1.6526e-08 |
| Extreme (D=32) | 69 | 2070 | 1012 / 1017 / 41 | 48.9% | 0.4772 vs 0.4649 | -2.6% | -0.0034 | 9.2361e-02 |

## 2. Independent Function-Aggregated Synthesis (Demšar, 2006)

- **Canonical BBOB Functions Evaluated**: 24
- **Functions Won by Proposed**: **16 / 24** (66.7%)
- **Wilcoxon Signed-Rank p-value (across 24 functions)**: `87.0, p = 7.3793e-02`
- **Exact Binomial Sign Test p-value**: `7.5795e-02`

## 3. Landscape Breakdown by BBOB Problem Class

| Problem Class | Tasks | Win Rate | Regret Proposed | Regret Baseline | Rel. Reduction | Mean Cliff's Delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Separable (F1-F5) | 30 | 45.1% | 0.4713 | 0.4373 | -7.8% | +0.099 |
| 2. Moderate Conditioning (F6-F9) | 24 | 52.1% | 0.4130 | 0.4329 | +4.6% | -0.061 |
| 3. Ill-Conditioned (F10-F14) | 30 | 55.3% | 0.3624 | 0.3912 | +7.4% | -0.116 |
| 4. Multi-modal w/ Structure (F15-F19) | 30 | 62.0% | 0.4251 | 0.5128 | +17.1% | -0.229 |
| 5. Multi-modal w/ Weak Structure (F20-F24) | 30 | 51.1% | 0.5643 | 0.5844 | +3.4% | -0.034 |
