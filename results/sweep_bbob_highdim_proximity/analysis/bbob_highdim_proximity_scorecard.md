# BBOB High-D & Extreme Evaluation Scorecard

Evaluation comparing **SMAC20_ProximityLCB** ($k=25, \lambda=1.345, \epsilon=0.16$) vs **SMAC3_HPOFacade_lcb** ($\beta=3.8416$) across 30 seeds.

## Overall (D=16 & D=32)
- **Tasks Evaluated**: 144
- **Paired Runs**: 4320
- **Win / Loss / Tie**: 3015 / 1189 / 116 (Win Rate: 69.8%)
- **Cliff's Delta**: -0.0305
- **Demšar Task-Level Wilcoxon p-value**: 1.5326e-23
- **Mean Normalized Regret (Proposed)**: 0.3911
- **Mean Normalized Regret (Baseline)**: 0.5230

## High-D (D=16)
- **Tasks Evaluated**: 75
- **Paired Runs**: 2250
- **Win / Loss / Tie**: 1665 / 527 / 58 (Win Rate: 74.0%)
- **Cliff's Delta**: -0.0449
- **Demšar Task-Level Wilcoxon p-value**: 5.4981e-14
- **Mean Normalized Regret (Proposed)**: 0.3629
- **Mean Normalized Regret (Baseline)**: 0.5305

## Extreme (D=32)
- **Tasks Evaluated**: 69
- **Paired Runs**: 2070
- **Win / Loss / Tie**: 1350 / 662 / 58 (Win Rate: 65.2%)
- **Cliff's Delta**: -0.0285
- **Demšar Task-Level Wilcoxon p-value**: 1.7419e-10
- **Mean Normalized Regret (Proposed)**: 0.4218
- **Mean Normalized Regret (Baseline)**: 0.5148

