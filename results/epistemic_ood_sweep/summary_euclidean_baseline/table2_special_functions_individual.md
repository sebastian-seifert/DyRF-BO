# Table 2: Special Named Benchmark Functions Listed Individually

Averaged across seeds for each of the 10 special named benchmark functions.

| func_name     |   dim | gap_type   | approach              |   auroc |   fpr95 |   aupr |   spearman |   aurc |   oracle_aurc |    jsd |     mi |    nlpd |   brier |
|:--------------|------:|:-----------|:----------------------|--------:|--------:|-------:|-----------:|-------:|--------------:|-------:|-------:|--------:|--------:|
| ackley_2d     |     2 | empty      | distance_evidential   |  0.9998 |  0      | 0.9995 |     0.5985 | 0.1448 |        0.0955 | 0.9824 | 0.9829 |  0.7302 |  0.1162 |
| ackley_2d     |     2 | empty      | standard_disagreement |  0.7176 |  0.5794 | 0.4603 |     0.2724 | 0.1999 |        0.0955 | 0.4298 | 0.3997 |  0.6579 |  0.202  |
| ackley_2d     |     2 | sparse     | distance_evidential   |  0.9874 |  0.0571 | 0.979  |     0.2717 | 0.1281 |        0.0677 | 0.817  | 0.8239 |  0.5871 |  0.1793 |
| ackley_2d     |     2 | sparse     | standard_disagreement |  0.8995 |  0.3672 | 0.813  |     0.2009 | 0.1341 |        0.0677 | 0.5822 | 0.5635 |  0.0591 |  0.1395 |
| branin        |     2 | empty      | distance_evidential   |  0.9884 |  0.0718 | 0.9754 |     0.0224 | 0.0823 |        0.038  | 0.8217 | 0.8161 |  0.5777 |  0.1584 |
| branin        |     2 | empty      | standard_disagreement |  0.564  |  0.827  | 0.3114 |     0.0625 | 0.081  |        0.038  | 0.2076 | 0.1884 | -0.2172 |  0.2364 |
| branin        |     2 | sparse     | distance_evidential   |  0.862  |  0.6756 | 0.7777 |     0.0379 | 0.0825 |        0.0385 | 0.3785 | 0.3859 |  0.5306 |  0.2736 |
| branin        |     2 | sparse     | standard_disagreement |  0.3763 |  0.8844 | 0.2319 |     0.0797 | 0.0804 |        0.0385 | 0.231  | 0.2032 | -0.257  |  0.2693 |
| rosenbrock_2d |     2 | empty      | distance_evidential   |  0.9506 |  0.3295 | 0.929  |     0.149  | 0.0837 |        0.0414 | 0.6377 | 0.6563 |  0.5819 |  0.1625 |
| rosenbrock_2d |     2 | empty      | standard_disagreement |  0.405  |  0.8941 | 0.2416 |     0.1078 | 0.0862 |        0.0414 | 0.2335 | 0.2064 | -0.2508 |  0.2849 |
| rosenbrock_2d |     2 | sparse     | distance_evidential   |  0.843  |  0.6604 | 0.7545 |     0.1187 | 0.0824 |        0.0403 | 0.333  | 0.3403 |  0.539  |  0.2475 |
| rosenbrock_2d |     2 | sparse     | standard_disagreement |  0.4032 |  0.8766 | 0.2396 |     0.1436 | 0.0814 |        0.0403 | 0.2335 | 0.2043 | -0.2588 |  0.2862 |
| hartmann3     |     3 | empty      | distance_evidential   |  0.8499 |  0.5824 | 0.7274 |     0.231  | 0.1036 |        0.0533 | 0.3586 | 0.3534 |  0.651  |  0.2972 |
| hartmann3     |     3 | empty      | standard_disagreement |  0.8615 |  0.3359 | 0.6406 |     0.3133 | 0.0968 |        0.0533 | 0.4591 | 0.4216 |  0.0274 |  0.1521 |
| hartmann3     |     3 | sparse     | distance_evidential   |  0.5243 |  0.8916 | 0.312  |     0.1144 | 0.0982 |        0.0472 | 0.0583 | 0.0536 |  0.6181 |  0.3498 |
| hartmann3     |     3 | sparse     | standard_disagreement |  0.7906 |  0.5114 | 0.5335 |     0.2115 | 0.0912 |        0.0472 | 0.3092 | 0.2826 | -0.014  |  0.1752 |
| ackley_4d     |     4 | empty      | distance_evidential   |  0.9643 |  0.231  | 0.953  |     0.6114 | 0.2545 |        0.1672 | 0.7308 | 0.7465 |  1.005  |  0.2441 |
| ackley_4d     |     4 | empty      | standard_disagreement |  0.2082 |  0.9569 | 0.1963 |    -0.3877 | 0.6493 |        0.1672 | 0.363  | 0.3423 |  1.3569 |  0.3018 |
| ackley_4d     |     4 | sparse     | distance_evidential   |  0.5151 |  0.9637 | 0.3813 |     0.102  | 0.2194 |        0.1054 | 0.0713 | 0.0737 |  0.7582 |  0.3367 |
| ackley_4d     |     4 | sparse     | standard_disagreement |  0.5549 |  0.9606 | 0.4206 |     0.0516 | 0.2299 |        0.1054 | 0.1077 | 0.1116 |  0.4884 |  0.2353 |
| rosenbrock_4d |     4 | empty      | distance_evidential   |  0.4688 |  0.9187 | 0.2632 |     0.2686 | 0.1213 |        0.0639 | 0.0889 | 0.0769 |  0.6947 |  0.267  |
| rosenbrock_4d |     4 | empty      | standard_disagreement |  0.1356 |  0.9884 | 0.1822 |     0.3148 | 0.1156 |        0.0639 | 0.4158 | 0.3748 |  0.1538 |  0.3012 |
| rosenbrock_4d |     4 | sparse     | distance_evidential   |  0.1392 |  0.9956 | 0.1812 |     0.318  | 0.1075 |        0.058  | 0.3713 | 0.3413 |  0.669  |  0.2908 |
| rosenbrock_4d |     4 | sparse     | standard_disagreement |  0.1007 |  0.9931 | 0.1777 |     0.3626 | 0.1035 |        0.058  | 0.4923 | 0.4498 |  0.1266 |  0.3066 |
| friedman_6d   |     6 | empty      | distance_evidential   |  0.3925 |  0.9371 | 0.2376 |     0.0992 | 0.2001 |        0.1038 | 0.1118 | 0.0984 |  0.7073 |  0.3776 |
| friedman_6d   |     6 | empty      | standard_disagreement |  0.4463 |  0.9023 | 0.2555 |     0.0781 | 0.2041 |        0.1038 | 0.1003 | 0.0878 |  0.4376 |  0.2627 |
| friedman_6d   |     6 | sparse     | distance_evidential   |  0.0226 |  0.9999 | 0.1693 |     0.2961 | 0.1311 |        0.0721 | 0.7459 | 0.7323 |  0.6627 |  0.3992 |
| friedman_6d   |     6 | sparse     | standard_disagreement |  0.1136 |  0.9977 | 0.1774 |     0.2348 | 0.1394 |        0.0721 | 0.4239 | 0.4039 |  0.3487 |  0.3095 |
| hartmann6     |     6 | empty      | distance_evidential   |  0.7775 |  0.4759 | 0.4776 |     0.4607 | 0.1345 |        0.0798 | 0.3238 | 0.294  |  0.7617 |  0.2026 |
| hartmann6     |     6 | empty      | standard_disagreement |  0.8092 |  0.3547 | 0.5034 |     0.4875 | 0.1297 |        0.0798 | 0.4375 | 0.3919 |  0.3973 |  0.1817 |
| hartmann6     |     6 | sparse     | distance_evidential   |  0.4158 |  0.95   | 0.2515 |     0.3733 | 0.129  |        0.0711 | 0.0666 | 0.0593 |  0.7224 |  0.2341 |
| hartmann6     |     6 | sparse     | standard_disagreement |  0.6864 |  0.5001 | 0.3724 |     0.4396 | 0.1194 |        0.0711 | 0.3236 | 0.2858 |  0.3418 |  0.2193 |
| hartmann_6d   |     6 | empty      | distance_evidential   |  0.4656 |  0.916  | 0.2618 |     0.4642 | 0.1175 |        0.0696 | 0.0886 | 0.0772 |  0.7212 |  0.2333 |
| hartmann_6d   |     6 | empty      | standard_disagreement |  0.5099 |  0.8057 | 0.2756 |     0.5181 | 0.1112 |        0.0696 | 0.1381 | 0.1226 |  0.2729 |  0.2575 |
| hartmann_6d   |     6 | sparse     | distance_evidential   |  0.2259 |  0.9921 | 0.1943 |     0.4522 | 0.1092 |        0.0643 | 0.2113 | 0.1989 |  0.7023 |  0.2476 |
| hartmann_6d   |     6 | sparse     | standard_disagreement |  0.4574 |  0.842  | 0.2551 |     0.4994 | 0.1048 |        0.0643 | 0.1203 | 0.1064 |  0.2465 |  0.2668 |
| friedman_10d  |    10 | empty      | distance_evidential   |  0.3311 |  0.9649 | 0.2195 |     0.0089 | 0.2104 |        0.1042 | 0.1323 | 0.1185 |  0.7212 |  0.3927 |
| friedman_10d  |    10 | empty      | standard_disagreement |  0.4699 |  0.906  | 0.2701 |     0.0387 | 0.2081 |        0.1042 | 0.0872 | 0.0773 |  0.456  |  0.2603 |
| friedman_10d  |    10 | sparse     | distance_evidential   |  0.0029 |  1      | 0.1683 |     0.3095 | 0.1271 |        0.0709 | 0.922  | 0.9188 |  0.6708 |  0.4222 |
| friedman_10d  |    10 | sparse     | standard_disagreement |  0.0625 |  0.9999 | 0.1719 |     0.267  | 0.1328 |        0.0709 | 0.569  | 0.5571 |  0.3574 |  0.3227 |
