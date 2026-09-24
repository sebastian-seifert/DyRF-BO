# Table 1: Grand Total Calibration Scorecard across All Experiments

| Core Metric | Desirable | PLCB Mean (±SEM) | SLCB Mean (±SEM) | Paired Diff | Wilcoxon p-value | Cliff's δ | Win / Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Distance Monotonicity $\rho(\tilde d, U)$** | Higher | -0.3906 ± 0.0064 | 0.1265 ± 0.0089 | -0.5171 | 2.6e-295 | -0.714 | 115W / 1805L |
| **Error Alignment $\rho(|e|, U)$** | Higher | -0.0868 ± 0.0068 | 0.0949 ± 0.0068 | -0.1817 | 1.4e-76 | -0.350 | 522W / 1398L |
| **Coverage Error $|\mathrm{PICP} - 0.95|$** | Lower | 0.5988 ± 0.0052 | 0.4978 ± 0.0068 | +0.1010 | 2.8e-229 | +0.199 | 142W / 1699L |
| **Winkler Score (Interval Penalty)** | Lower | 161.32 ± 4.03 | 148.06 ± 3.90 | +13.26 | 5.4e-254 | +0.109 | 232W / 1688L |
| **Catastrophic Outlier AUROC** | Higher | 0.4630 ± 0.0050 | 0.5653 ± 0.0043 | -0.1022 | 1.9e-77 | -0.354 | 486W / 1434L |