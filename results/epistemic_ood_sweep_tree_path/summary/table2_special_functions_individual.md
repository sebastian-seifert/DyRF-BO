# Table 2: Special Named Benchmark Functions Listed Individually

Averaged across seeds for each of the 10 special named benchmark functions.

| func_name     |   dim | gap_type   | approach              |   auroc |   fpr95 |   aupr |   spearman |   aurc |   oracle_aurc |    jsd |     mi |    nlpd |   brier |
|:--------------|------:|:-----------|:----------------------|--------:|--------:|-------:|-----------:|-------:|--------------:|-------:|-------:|--------:|--------:|
| ackley_2d     |     2 | empty      | distance_evidential   |  0.9306 |  0.3217 | 0.8971 |     0.535  | 0.1541 |        0.0955 | 0.6897 | 0.6921 |  0.7007 |  0.2711 |
| ackley_2d     |     2 | empty      | standard_disagreement |  0.7176 |  0.5794 | 0.4603 |     0.2724 | 0.1999 |        0.0955 | 0.4298 | 0.3997 |  0.6579 |  0.202  |
| ackley_2d     |     2 | sparse     | distance_evidential   |  0.9331 |  0.2632 | 0.827  |     0.214  | 0.1327 |        0.0677 | 0.6333 | 0.6179 |  0.5542 |  0.2628 |
| ackley_2d     |     2 | sparse     | standard_disagreement |  0.8995 |  0.3672 | 0.813  |     0.2009 | 0.1341 |        0.0677 | 0.5822 | 0.5635 |  0.0591 |  0.1395 |
| branin        |     2 | empty      | distance_evidential   |  0.9764 |  0.1448 | 0.9567 |     0.0261 | 0.0823 |        0.038  | 0.7719 | 0.7719 |  0.5411 |  0.2199 |
| branin        |     2 | empty      | standard_disagreement |  0.564  |  0.827  | 0.3114 |     0.0625 | 0.081  |        0.038  | 0.2076 | 0.1884 | -0.2172 |  0.2364 |
| branin        |     2 | sparse     | distance_evidential   |  0.9435 |  0.236  | 0.8787 |     0.0213 | 0.0827 |        0.0385 | 0.6112 | 0.6005 |  0.5301 |  0.2726 |
| branin        |     2 | sparse     | standard_disagreement |  0.3763 |  0.8844 | 0.2319 |     0.0797 | 0.0804 |        0.0385 | 0.231  | 0.2032 | -0.257  |  0.2693 |
| rosenbrock_2d |     2 | empty      | distance_evidential   |  0.8373 |  0.6221 | 0.7317 |     0.1488 | 0.0834 |        0.0414 | 0.3665 | 0.369  |  0.5194 |  0.3036 |
| rosenbrock_2d |     2 | empty      | standard_disagreement |  0.405  |  0.8941 | 0.2416 |     0.1078 | 0.0862 |        0.0414 | 0.2335 | 0.2064 | -0.2508 |  0.2849 |
| rosenbrock_2d |     2 | sparse     | distance_evidential   |  0.8541 |  0.453  | 0.7005 |     0.1044 | 0.083  |        0.0403 | 0.3859 | 0.3727 |  0.5216 |  0.3239 |
| rosenbrock_2d |     2 | sparse     | standard_disagreement |  0.4032 |  0.8766 | 0.2396 |     0.1436 | 0.0814 |        0.0403 | 0.2335 | 0.2043 | -0.2588 |  0.2862 |
| hartmann3     |     3 | empty      | distance_evidential   |  0.8193 |  0.6127 | 0.6522 |     0.2126 | 0.106  |        0.0533 | 0.3043 | 0.2944 |  0.6563 |  0.2962 |
| hartmann3     |     3 | empty      | standard_disagreement |  0.8615 |  0.3359 | 0.6406 |     0.3133 | 0.0968 |        0.0533 | 0.4591 | 0.4216 |  0.0274 |  0.1521 |
| hartmann3     |     3 | sparse     | distance_evidential   |  0.7405 |  0.7686 | 0.5495 |     0.1043 | 0.0999 |        0.0472 | 0.1836 | 0.1781 |  0.6427 |  0.312  |
| hartmann3     |     3 | sparse     | standard_disagreement |  0.7906 |  0.5114 | 0.5335 |     0.2115 | 0.0912 |        0.0472 | 0.3092 | 0.2826 | -0.014  |  0.1752 |
| ackley_4d     |     4 | empty      | distance_evidential   |  0.4825 |  0.7216 | 0.281  |    -0.0229 | 0.4393 |        0.1672 | 0.3474 | 0.3174 |  1.0747 |  0.3492 |
| ackley_4d     |     4 | empty      | standard_disagreement |  0.2082 |  0.9569 | 0.1963 |    -0.3877 | 0.6493 |        0.1672 | 0.363  | 0.3423 |  1.3569 |  0.3018 |
| ackley_4d     |     4 | sparse     | distance_evidential   |  0.3787 |  0.993  | 0.2602 |     0.0336 | 0.2312 |        0.1054 | 0.1062 | 0.1065 |  0.7584 |  0.3517 |
| ackley_4d     |     4 | sparse     | standard_disagreement |  0.5549 |  0.9606 | 0.4206 |     0.0516 | 0.2299 |        0.1054 | 0.1077 | 0.1116 |  0.4884 |  0.2353 |
| rosenbrock_4d |     4 | empty      | distance_evidential   |  0.3985 |  0.9533 | 0.2375 |     0.1925 | 0.1313 |        0.0639 | 0.0959 | 0.0825 |  0.6961 |  0.2708 |
| rosenbrock_4d |     4 | empty      | standard_disagreement |  0.1356 |  0.9884 | 0.1822 |     0.3148 | 0.1156 |        0.0639 | 0.4158 | 0.3748 |  0.1538 |  0.3012 |
| rosenbrock_4d |     4 | sparse     | distance_evidential   |  0.3259 |  0.967  | 0.2163 |     0.2073 | 0.1201 |        0.058  | 0.1299 | 0.1142 |  0.688  |  0.2765 |
| rosenbrock_4d |     4 | sparse     | standard_disagreement |  0.1007 |  0.9931 | 0.1777 |     0.3626 | 0.1035 |        0.058  | 0.4923 | 0.4498 |  0.1266 |  0.3066 |
| friedman_6d   |     6 | empty      | distance_evidential   |  0.6703 |  0.7739 | 0.4161 |     0.1222 | 0.2036 |        0.1038 | 0.1301 | 0.1197 |  0.8047 |  0.3837 |
| friedman_6d   |     6 | empty      | standard_disagreement |  0.4463 |  0.9023 | 0.2555 |     0.0781 | 0.2041 |        0.1038 | 0.1003 | 0.0878 |  0.4376 |  0.2627 |
| friedman_6d   |     6 | sparse     | distance_evidential   |  0.3929 |  0.9611 | 0.2418 |     0.0641 | 0.1621 |        0.0721 | 0.0804 | 0.0746 |  0.7728 |  0.3853 |
| friedman_6d   |     6 | sparse     | standard_disagreement |  0.1136 |  0.9977 | 0.1774 |     0.2348 | 0.1394 |        0.0721 | 0.4239 | 0.4039 |  0.3487 |  0.3095 |
| hartmann6     |     6 | empty      | distance_evidential   |  0.8284 |  0.424  | 0.5502 |     0.3932 | 0.1471 |        0.0798 | 0.3772 | 0.3493 |  0.8414 |  0.2078 |
| hartmann6     |     6 | empty      | standard_disagreement |  0.8092 |  0.3547 | 0.5034 |     0.4875 | 0.1297 |        0.0798 | 0.4375 | 0.3919 |  0.3973 |  0.1817 |
| hartmann6     |     6 | sparse     | distance_evidential   |  0.6466 |  0.808  | 0.372  |     0.316  | 0.1386 |        0.0711 | 0.1177 | 0.1081 |  0.8164 |  0.2265 |
| hartmann6     |     6 | sparse     | standard_disagreement |  0.6864 |  0.5001 | 0.3724 |     0.4396 | 0.1194 |        0.0711 | 0.3236 | 0.2858 |  0.3418 |  0.2193 |
| hartmann_6d   |     6 | empty      | distance_evidential   |  0.6309 |  0.7794 | 0.3479 |     0.3695 | 0.1314 |        0.0696 | 0.1433 | 0.1297 |  0.8131 |  0.2283 |
| hartmann_6d   |     6 | empty      | standard_disagreement |  0.5099 |  0.8057 | 0.2756 |     0.5181 | 0.1112 |        0.0696 | 0.1381 | 0.1226 |  0.2729 |  0.2575 |
| hartmann_6d   |     6 | sparse     | distance_evidential   |  0.5567 |  0.8679 | 0.306  |     0.3299 | 0.1265 |        0.0643 | 0.0848 | 0.0756 |  0.8056 |  0.2334 |
| hartmann_6d   |     6 | sparse     | standard_disagreement |  0.4574 |  0.842  | 0.2551 |     0.4994 | 0.1048 |        0.0643 | 0.1203 | 0.1064 |  0.2465 |  0.2668 |
| friedman_10d  |    10 | empty      | distance_evidential   |  0.7248 |  0.681  | 0.4733 |     0.1079 | 0.2048 |        0.1042 | 0.1855 | 0.1703 |  0.821  |  0.4013 |
| friedman_10d  |    10 | empty      | standard_disagreement |  0.4699 |  0.906  | 0.2701 |     0.0387 | 0.2081 |        0.1042 | 0.0872 | 0.0773 |  0.456  |  0.2603 |
| friedman_10d  |    10 | sparse     | distance_evidential   |  0.3679 |  0.9783 | 0.2345 |     0.0808 | 0.1589 |        0.0709 | 0.0871 | 0.0818 |  0.7877 |  0.4026 |
| friedman_10d  |    10 | sparse     | standard_disagreement |  0.0625 |  0.9999 | 0.1719 |     0.267  | 0.1328 |        0.0709 | 0.569  | 0.5571 |  0.3574 |  0.3227 |
