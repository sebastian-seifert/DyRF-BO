# BBOB High-D & Extreme Evaluation Scorecard

Evaluation comparing **SMAC20_ProximityLCB** ($k=25, \lambda=1.345, \epsilon=0.16$) vs **SMAC3_HPOFacade_lcb** ($\beta=3.8416$) across 30 seeds.

## Overall (D=16 & D=32)
- **Tasks Evaluated**: 144
- **Paired Runs**: 4320
- **Win / Loss / Tie**: 3363 / 825 / 132 (Win Rate: 77.8%)
- **Cliff's Delta**: -0.0428
- **Demšar Task-Level Wilcoxon p-value**: 4.5275e-25
- **Mean Normalized Regret (Proposed)**: 0.3548
- **Mean Normalized Regret (Baseline)**: 0.5492

## High-D (D=16)
- **Tasks Evaluated**: 75
- **Paired Runs**: 2250
- **Win / Loss / Tie**: 1739 / 443 / 68 (Win Rate: 77.3%)
- **Cliff's Delta**: -0.0569
- **Demšar Task-Level Wilcoxon p-value**: 8.2226e-14
- **Mean Normalized Regret (Proposed)**: 0.3440
- **Mean Normalized Regret (Baseline)**: 0.5348

## Extreme (D=32)
- **Tasks Evaluated**: 69
- **Paired Runs**: 2070
- **Win / Loss / Tie**: 1624 / 382 / 64 (Win Rate: 78.5%)
- **Cliff's Delta**: -0.0499
- **Demšar Task-Level Wilcoxon p-value**: 6.7848e-13
- **Mean Normalized Regret (Proposed)**: 0.3665
- **Mean Normalized Regret (Baseline)**: 0.5649

