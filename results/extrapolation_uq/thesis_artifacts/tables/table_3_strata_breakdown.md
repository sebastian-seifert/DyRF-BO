# Table 3: Performance Partitioned by Extrapolation Depth Strata

| Stratum | PLCB Winkler | SLCB Winkler | Winkler Ratio (log) | PLCB PICP | SLCB PICP | PLCB AUROC | SLCB AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stratum 0 (Interpolation: $\tilde d \le 0$)** | 37.0 ± 2.0 | 46.8 ± 2.3 | -0.23 | 0.625 | 0.591 | 0.519 | 0.626 |
| **Stratum 1 (Near Extrapolation: $0 < \tilde d \le 0.3$)** | 10.4 ± 0.3 | 6.4 ± 0.1 | +0.48 | 0.766 | 0.904 | 0.422 | 0.587 |
| **Stratum 2 (Moderate Extrapolation: $0.3 < \tilde d \le 0.8$)** | 129.0 ± 2.9 | 110.1 ± 2.7 | +0.16 | 0.255 | 0.397 | 0.451 | 0.529 |
| **Stratum 3 (Deep Extrapolation: $\tilde d > 0.8$)** | 336.0 ± 7.0 | 310.8 ± 6.8 | +0.08 | 0.134 | 0.237 | 0.467 | 0.543 |
| **Overall / Global** | 161.3 ± 4.0 | 148.1 ± 3.9 | +0.09 | 0.351 | 0.453 | 0.463 | 0.565 |