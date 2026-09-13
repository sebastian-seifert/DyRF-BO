# Pairwise Wilcoxon Signed-Rank Tests vs. Standard SMAC3 Baseline (N=4 Tasks)

Reference Baseline: **`SMAC3_HPOFacade_ei`** (Mean Normalized Regret: **2.1307**)

> **Statistical Power Note**: With $N=4$ tasks, the mathematical minimum two-sided Wilcoxon signed-rank
> p-value is $p_{\min} = 2 \cdot (0.5)^4 = 0.125 > 0.05$ (even if one method wins on 100% of tasks).
> Consequently, cross-task aggregated testing cannot achieve statistical significance at $\alpha = 0.05$.
> The primary, rigorous statistical evaluation is therefore the **per-task paired seed testing ($N = 35 \ge 30$)**
> reported in `statistical_results.csv` and `BENCHMARK_SUMMARY.md`.

| Paradigm | UQ Extractor | Mean Regret | Mean Diff vs Base | Rel Reduction (%) | Win / Loss / Tie | Wilcoxon W | p_raw | Cliff's delta | Holm-Bonferroni adj p | Significance (adj p < 0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Additive Hybrid (EI) | distance_evidential | 2.3947 | +0.2640 | +12.4% | 2 / 2 / 0 | 4.0 | 0.8750 | +0.125 | 0.8750 | Non-significant |
