# Statistical Scorecard: yahpo_rbv2_ranger

- **Evaluated Tasks**: 119
- **Proposed Approach**: `SMAC20_ProximityLCB`
- **Baseline Approach**: `SMAC3_HPOFacade_lcb`

## Summary Metrics

| Metric | Proposed (`SMAC20_ProximityLCB`) | Baseline (`SMAC3_HPOFacade_lcb`) |
| :--- | :--- | :--- |
| **Mean of Medians** | `-0.897284` | `-0.892880` |
| **Task Wins** | **108** (90.8%) | 2 (1.7%) |
| **Task Ties** | 9 (7.6%) | 9 (7.6%) |

## Hypothesis Testing & Effect Size

- **One-Sided Wilcoxon Signed-Rank Test (`H1: Proposed < Baseline`)**:
  - Statistic ($W$): `115.0`
  - $p$-value: `9.7182e-19` (Statistically Significant, p < 0.05)
- **Cliff's Delta Effect Size ($\delta$)**:
  - Value: `-0.0380`
  - Interpretation: **Negligible** effect in favor of Proposed
