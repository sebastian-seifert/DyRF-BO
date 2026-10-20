# CARP-S BBOB Dimension 2 Benchmark Scorecard & Analysis

**Omnibus Friedman Test**: $\chi^2 = None$, $p = None$

## 2D Global Scorecard

| optimizer_id              |   n_tasks |   n_runs |   mean_norm_regret |   median_norm_regret |   mean_raw_cost |   ctrl_lcb_win_rate |   ctrl_lcb_cliffs_delta |   ctrl_lcb_rank_biserial |   ctrl_lcb_wilcoxon_p |   ctrl_ei_win_rate |   ctrl_ei_cliffs_delta |   ctrl_ei_rank_biserial |   ctrl_ei_wilcoxon_p |   ctrl_lcb_wilcoxon_adj_p |   ctrl_ei_wilcoxon_adj_p |
|:--------------------------|----------:|---------:|-------------------:|---------------------:|----------------:|--------------------:|------------------------:|-------------------------:|----------------------:|-------------------:|-----------------------:|------------------------:|---------------------:|--------------------------:|-------------------------:|
| SMAC20_ProximityLCB_tuned |        20 |      600 |           0.186398 |            0.0512864 |         10106.2 |                0.75 |                   -0.12 |                -0.691176 |            0.00754291 |                nan |                    nan |                     nan |                  nan |                0.00754291 |                      nan |
| SMAC3_HPOFacade_lcb       |        20 |      600 |           0.267571 |            0.132707  |         11517.6 |                0.5  |                    0    |                 0        |          nan          |                nan |                    nan |                     nan |                  nan |              nan          |                      nan |
