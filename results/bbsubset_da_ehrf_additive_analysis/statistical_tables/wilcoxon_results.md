# Wilcoxon Signed-Rank Test: DA-EHRF Additive EI vs. SMAC3 Baseline

- Reference Baseline: `SMAC3_HPOFacade_ei`
- Proposed Model: `CARPSDynamicRF_DAEHRF_AdditiveEI`
- Evaluated Tasks: 18 tasks

| Proposed Optimizer | Baseline Optimizer | N Tasks | Proposed Mean Regret | Baseline Mean Regret | Mean Diff vs Base | Rel Reduction (%) | Win / Loss / Tie | Wilcoxon W | p_val | Cliff's delta | Significance (p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CARPSDynamicRF_DAEHRF_AdditiveEI | SMAC3_HPOFacade_ei | 18 | 117079636.2324 | 13192.8113 | +117066443.4211 | +887350.2% | 9 / 9 / 0 | 69.0 | 0.4951 | -0.012 | Non-significant |
