# CARP-S Optimization Benchmark Summary Report

This report presents the final cost comparison across different BO approaches, standard `smac3_bo` baseline, and Dynamic RF UQ extractors for CARP-S benchmarks.

## Benchmark Task: `cfg_lcbench_167168`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -84.470422 | 3.531822 | 1.579479 | 5/5 | -87.798317 | -79.116096 |
| `epistemic_proximity_bc` | -84.095531 | 4.197800 | 1.877313 | 5/5 | -86.795631 | -76.826591 |
| `epistemic_proximity_auto_lambda` | -83.934351 | 3.657427 | 1.635651 | 5/5 | -88.326736 | -79.980324 |
| `epistemic_shaker_entropy` | -83.661458 | 4.976773 | 2.225681 | 5/5 | -87.863319 | -75.980789 |
| `epistemic_proximity_b` | -83.284341 | 3.416134 | 1.527742 | 5/5 | -86.008881 | -77.534668 |
| `epistemic_likelihood_credal` | -82.283665 | 2.126054 | 0.950800 | 5/5 | -84.467033 | -79.565849 |
| `epistemic_standard_proximity` | -82.135902 | 5.027017 | 2.248150 | 5/5 | -87.541138 | -75.233940 |
| `epistemic_chen_variance` | -81.517543 | 2.833621 | 1.267234 | 5/5 | -84.726166 | -77.013649 |
| `epistemic_standard_disagreement` | -75.439822 | 4.210076 | 1.882803 | 5/5 | -79.149300 | -68.319260 |


## Benchmark Task: `cfg_lcbench_189873`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_shaker_entropy` | -89.494781 | 2.928912 | 1.309849 | 5/5 | -94.247757 | -86.814835 |
| `epistemic_proximity_auto_lambda` | -88.087842 | 4.912347 | 2.196869 | 5/5 | -93.172127 | -80.754730 |
| `epistemic_likelihood_credal` | -87.894626 | 4.200717 | 1.878618 | 5/5 | -93.704292 | -82.472908 |
| `epistemic_standard_disagreement` | -87.893307 | 1.418151 | 0.634216 | 5/5 | -90.215996 | -86.424271 |
| `epistemic_proximity_b` | -87.656239 | 4.350794 | 1.945734 | 5/5 | -92.613663 | -81.592331 |
| `smac3_bo (Baseline)` | -87.254099 | 8.559300 | 3.827835 | 5/5 | -96.933586 | -78.214500 |
| `epistemic_standard_proximity` | -86.865184 | 3.329114 | 1.488825 | 5/5 | -89.083694 | -81.003098 |
| `epistemic_chen_variance` | -86.814883 | 1.375467 | 0.615127 | 5/5 | -88.580475 | -84.982536 |
| `epistemic_proximity_bc` | -85.264317 | 5.743784 | 2.568698 | 5/5 | -94.348480 | -81.406105 |


## Benchmark Task: `cfg_lcbench_189906`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -90.451509 | 2.649959 | 1.185098 | 5/5 | -94.151451 | -86.782722 |
| `epistemic_shaker_entropy` | -89.921188 | 1.838222 | 0.822078 | 5/5 | -92.015709 | -87.618980 |
| `epistemic_proximity_bc` | -89.590239 | 1.187658 | 0.531137 | 5/5 | -91.228958 | -88.359428 |
| `epistemic_proximity_b` | -89.144925 | 1.705830 | 0.762870 | 5/5 | -91.389885 | -87.326012 |
| `epistemic_proximity_auto_lambda` | -88.713188 | 0.994822 | 0.444898 | 5/5 | -89.611649 | -87.221802 |
| `epistemic_standard_proximity` | -88.132889 | 1.331373 | 0.595408 | 5/5 | -90.138260 | -86.508385 |
| `epistemic_standard_disagreement` | -85.688387 | 1.330128 | 0.594851 | 5/5 | -87.601601 | -84.394241 |
| `epistemic_likelihood_credal` | -84.918518 | 4.442613 | 1.986797 | 5/5 | -91.539917 | -80.297951 |
| `epistemic_chen_variance` | -84.347197 | 2.925240 | 1.308207 | 5/5 | -88.605484 | -81.275475 |


## Benchmark Task: `cfg_ml_svm_12`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_b` | 0.032660 | 0.000000 | 0.000000 | 5/5 | 0.032660 | 0.032660 |
| `smac3_bo (Baseline)` | 0.032660 | 0.000000 | 0.000000 | 5/5 | 0.032660 | 0.032660 |
| `epistemic_standard_proximity` | 0.032727 | 0.000151 | 0.000067 | 5/5 | 0.032660 | 0.032997 |
| `epistemic_chen_variance` | 0.032795 | 0.000184 | 0.000082 | 5/5 | 0.032660 | 0.032997 |
| `epistemic_proximity_bc` | 0.032795 | 0.000301 | 0.000135 | 5/5 | 0.032660 | 0.033333 |
| `epistemic_standard_disagreement` | 0.032997 | 0.000412 | 0.000184 | 5/5 | 0.032660 | 0.033670 |
| `epistemic_proximity_auto_lambda` | 0.033872 | 0.002526 | 0.001130 | 5/5 | 0.032660 | 0.038384 |
| `epistemic_shaker_entropy` | 0.033872 | 0.002526 | 0.001130 | 5/5 | 0.032660 | 0.038384 |
| `epistemic_likelihood_credal` | 0.034074 | 0.002444 | 0.001093 | 5/5 | 0.032660 | 0.038384 |


## Benchmark Task: `cfg_ml_svm_3`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | 0.014442 | 0.000319 | 0.000143 | 5/5 | 0.014105 | 0.014737 |
| `epistemic_proximity_auto_lambda` | 0.014442 | 0.000352 | 0.000158 | 5/5 | 0.014105 | 0.014947 |
| `epistemic_proximity_bc` | 0.014526 | 0.000298 | 0.000133 | 5/5 | 0.014105 | 0.014947 |
| `epistemic_proximity_b` | 0.014526 | 0.000421 | 0.000188 | 5/5 | 0.014105 | 0.014947 |
| `epistemic_standard_disagreement` | 0.014568 | 0.000312 | 0.000140 | 5/5 | 0.014105 | 0.014947 |
| `epistemic_standard_proximity` | 0.014611 | 0.000319 | 0.000143 | 5/5 | 0.014105 | 0.014947 |
| `epistemic_chen_variance` | 0.014863 | 0.000188 | 0.000084 | 5/5 | 0.014526 | 0.014947 |
| `epistemic_likelihood_credal` | 0.014905 | 0.000231 | 0.000103 | 5/5 | 0.014526 | 0.015158 |
| `epistemic_shaker_entropy` | 0.015242 | 0.000659 | 0.000295 | 5/5 | 0.014947 | 0.016421 |


## Benchmark Task: `cfg_ml_svm_31`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_shaker_entropy` | 0.254276 | 0.000903 | 0.000404 | 5/5 | 0.253872 | 0.255892 |
| `epistemic_likelihood_credal` | 0.254949 | 0.001550 | 0.000693 | 5/5 | 0.253872 | 0.257239 |
| `smac3_bo (Baseline)` | 0.256700 | 0.002203 | 0.000985 | 5/5 | 0.253872 | 0.259933 |
| `epistemic_chen_variance` | 0.256970 | 0.001940 | 0.000868 | 5/5 | 0.253872 | 0.259259 |
| `epistemic_standard_disagreement` | 0.256970 | 0.001940 | 0.000868 | 5/5 | 0.253872 | 0.259259 |
| `epistemic_proximity_auto_lambda` | 0.257778 | 0.007005 | 0.003133 | 5/5 | 0.253872 | 0.270034 |
| `epistemic_proximity_b` | 0.259798 | 0.005722 | 0.002559 | 5/5 | 0.257239 | 0.270034 |
| `epistemic_standard_proximity` | 0.259798 | 0.005722 | 0.002559 | 5/5 | 0.257239 | 0.270034 |
| `epistemic_proximity_bc` | 0.260202 | 0.008671 | 0.003878 | 5/5 | 0.253872 | 0.270034 |


## Benchmark Task: `cfg_ml_xgboost_12`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_auto_lambda` | 0.008822 | 0.000151 | 0.000067 | 5/5 | 0.008754 | 0.009091 |
| `smac3_bo (Baseline)` | 0.008956 | 0.000184 | 0.000082 | 5/5 | 0.008754 | 0.009091 |
| `epistemic_standard_proximity` | 0.008956 | 0.000301 | 0.000135 | 5/5 | 0.008754 | 0.009428 |
| `epistemic_proximity_bc` | 0.009024 | 0.000369 | 0.000165 | 5/5 | 0.008754 | 0.009428 |
| `epistemic_proximity_b` | 0.009158 | 0.000369 | 0.000165 | 5/5 | 0.008754 | 0.009764 |
| `epistemic_chen_variance` | 0.009360 | 0.000151 | 0.000067 | 5/5 | 0.009091 | 0.009428 |
| `epistemic_likelihood_credal` | 0.009360 | 0.000151 | 0.000067 | 5/5 | 0.009091 | 0.009428 |
| `epistemic_standard_disagreement` | 0.009360 | 0.000151 | 0.000067 | 5/5 | 0.009091 | 0.009428 |
| `epistemic_shaker_entropy` | 0.009360 | 0.000151 | 0.000067 | 5/5 | 0.009091 | 0.009428 |


## Benchmark Task: `cfg_ml_xgboost_3`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_b` | 0.002232 | 0.000115 | 0.000052 | 5/5 | 0.002105 | 0.002316 |
| `epistemic_standard_proximity` | 0.002232 | 0.000115 | 0.000052 | 5/5 | 0.002105 | 0.002316 |
| `smac3_bo (Baseline)` | 0.002232 | 0.000115 | 0.000052 | 5/5 | 0.002105 | 0.002316 |
| `epistemic_likelihood_credal` | 0.002316 | 0.000000 | 0.000000 | 5/5 | 0.002316 | 0.002316 |
| `epistemic_proximity_auto_lambda` | 0.002316 | 0.000000 | 0.000000 | 5/5 | 0.002316 | 0.002316 |
| `epistemic_chen_variance` | 0.002316 | 0.000149 | 0.000067 | 5/5 | 0.002105 | 0.002526 |
| `epistemic_shaker_entropy` | 0.002316 | 0.000149 | 0.000067 | 5/5 | 0.002105 | 0.002526 |
| `epistemic_proximity_bc` | 0.002358 | 0.000094 | 0.000042 | 5/5 | 0.002316 | 0.002526 |
| `epistemic_standard_disagreement` | 0.002358 | 0.000094 | 0.000042 | 5/5 | 0.002316 | 0.002526 |


## Benchmark Task: `cfg_ml_xgboost_31`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_standard_disagreement` | 0.099125 | 0.000301 | 0.000135 | 5/5 | 0.098990 | 0.099663 |
| `smac3_bo (Baseline)` | 0.100337 | 0.001260 | 0.000563 | 5/5 | 0.098990 | 0.102357 |
| `epistemic_standard_proximity` | 0.100741 | 0.001819 | 0.000814 | 5/5 | 0.098990 | 0.103030 |
| `epistemic_chen_variance` | 0.100741 | 0.000768 | 0.000343 | 5/5 | 0.099663 | 0.101684 |
| `epistemic_proximity_bc` | 0.100875 | 0.001295 | 0.000579 | 5/5 | 0.099663 | 0.103030 |
| `epistemic_proximity_auto_lambda` | 0.101279 | 0.001622 | 0.000725 | 5/5 | 0.098990 | 0.103030 |
| `epistemic_proximity_b` | 0.102222 | 0.001608 | 0.000719 | 5/5 | 0.100337 | 0.104377 |
| `epistemic_shaker_entropy` | 0.102896 | 0.002151 | 0.000962 | 5/5 | 0.099663 | 0.105724 |
| `epistemic_likelihood_credal` | 0.103838 | 0.001807 | 0.000808 | 5/5 | 0.102357 | 0.106397 |


## Benchmark Task: `cfg_nb301_CIFAR10`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_b` | -94.151764 | 0.305320 | 0.136543 | 5/5 | -94.429329 | -93.762962 |
| `epistemic_standard_proximity` | -94.091536 | 0.272955 | 0.122069 | 5/5 | -94.341095 | -93.690247 |
| `epistemic_proximity_bc` | -94.068144 | 0.362620 | 0.162168 | 5/5 | -94.563675 | -93.600670 |
| `epistemic_shaker_entropy` | -94.067183 | 0.427856 | 0.191343 | 5/5 | -94.683014 | -93.582428 |
| `epistemic_proximity_auto_lambda` | -94.018683 | 0.175702 | 0.078576 | 5/5 | -94.255852 | -93.776527 |
| `epistemic_likelihood_credal` | -93.981978 | 0.348916 | 0.156040 | 5/5 | -94.452133 | -93.669830 |
| `epistemic_standard_disagreement` | -93.948198 | 0.227581 | 0.101777 | 5/5 | -94.255852 | -93.635201 |
| `epistemic_chen_variance` | -93.944743 | 0.233877 | 0.104593 | 5/5 | -94.255852 | -93.730881 |
| `smac3_bo (Baseline)` | -93.881616 | 0.210895 | 0.094315 | 5/5 | -94.255852 | -93.755547 |


## Benchmark Task: `cfg_rbv2_glmnet_375`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_standard_disagreement` | -0.960838 | 0.000034 | 0.000015 | 5/5 | -0.960877 | -0.960783 |
| `epistemic_chen_variance` | -0.960834 | 0.000120 | 0.000054 | 5/5 | -0.960910 | -0.960622 |
| `smac3_bo (Baseline)` | -0.960760 | 0.000185 | 0.000083 | 5/5 | -0.960908 | -0.960440 |
| `epistemic_proximity_auto_lambda` | -0.959646 | 0.001558 | 0.000697 | 5/5 | -0.960636 | -0.956924 |
| `epistemic_standard_proximity` | -0.959575 | 0.001545 | 0.000691 | 5/5 | -0.960785 | -0.957182 |
| `epistemic_proximity_bc` | -0.959471 | 0.001626 | 0.000727 | 5/5 | -0.960750 | -0.957035 |
| `epistemic_proximity_b` | -0.959401 | 0.002130 | 0.000953 | 5/5 | -0.960774 | -0.955821 |
| `epistemic_likelihood_credal` | -0.958592 | 0.002534 | 0.001133 | 5/5 | -0.960536 | -0.955445 |
| `epistemic_shaker_entropy` | -0.957244 | 0.001932 | 0.000864 | 5/5 | -0.960484 | -0.955445 |


## Benchmark Task: `cfg_rbv2_glmnet_458`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.997650 | 0.000498 | 0.000223 | 5/5 | -0.997994 | -0.996768 |
| `epistemic_proximity_bc` | -0.996415 | 0.000686 | 0.000307 | 5/5 | -0.997299 | -0.995484 |
| `epistemic_standard_proximity` | -0.996028 | 0.000502 | 0.000224 | 5/5 | -0.996808 | -0.995437 |
| `epistemic_chen_variance` | -0.995935 | 0.000354 | 0.000158 | 5/5 | -0.996360 | -0.995651 |
| `epistemic_standard_disagreement` | -0.995869 | 0.000927 | 0.000415 | 5/5 | -0.997466 | -0.995046 |
| `epistemic_proximity_auto_lambda` | -0.995739 | 0.000660 | 0.000295 | 5/5 | -0.996487 | -0.994943 |
| `epistemic_proximity_b` | -0.995612 | 0.000337 | 0.000151 | 5/5 | -0.995946 | -0.995199 |
| `epistemic_shaker_entropy` | -0.995469 | 0.000409 | 0.000183 | 5/5 | -0.996060 | -0.995046 |
| `epistemic_likelihood_credal` | -0.995095 | 0.000630 | 0.000282 | 5/5 | -0.995678 | -0.994080 |


## Benchmark Task: `cfg_rbv2_ranger_16`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.964565 | 0.005456 | 0.002440 | 5/5 | -0.971542 | -0.958659 |
| `epistemic_standard_proximity` | -0.964080 | 0.004666 | 0.002087 | 5/5 | -0.968408 | -0.956974 |
| `epistemic_proximity_auto_lambda` | -0.964056 | 0.003670 | 0.001641 | 5/5 | -0.966600 | -0.957612 |
| `epistemic_proximity_b` | -0.963426 | 0.004816 | 0.002154 | 5/5 | -0.969569 | -0.957458 |
| `epistemic_proximity_bc` | -0.960459 | 0.006910 | 0.003090 | 5/5 | -0.969674 | -0.952482 |
| `epistemic_likelihood_credal` | -0.960141 | 0.002675 | 0.001196 | 5/5 | -0.964634 | -0.957458 |
| `epistemic_chen_variance` | -0.959822 | 0.002783 | 0.001244 | 5/5 | -0.964198 | -0.957384 |
| `epistemic_standard_disagreement` | -0.959593 | 0.003056 | 0.001367 | 5/5 | -0.964198 | -0.956665 |
| `epistemic_shaker_entropy` | -0.957677 | 0.003830 | 0.001713 | 5/5 | -0.964198 | -0.954292 |


## Benchmark Task: `cfg_rbv2_ranger_42`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.992796 | 0.001636 | 0.000732 | 5/5 | -0.994247 | -0.990877 |
| `epistemic_standard_proximity` | -0.990673 | 0.003066 | 0.001371 | 5/5 | -0.992994 | -0.985556 |
| `epistemic_shaker_entropy` | -0.990280 | 0.001492 | 0.000667 | 5/5 | -0.992865 | -0.989067 |
| `epistemic_proximity_bc` | -0.989365 | 0.001172 | 0.000524 | 5/5 | -0.990621 | -0.987523 |
| `epistemic_proximity_b` | -0.989240 | 0.002841 | 0.001270 | 5/5 | -0.992027 | -0.984989 |
| `epistemic_standard_disagreement` | -0.988558 | 0.002302 | 0.001030 | 5/5 | -0.990470 | -0.985666 |
| `epistemic_chen_variance` | -0.988450 | 0.002168 | 0.000970 | 5/5 | -0.989907 | -0.984614 |
| `epistemic_proximity_auto_lambda` | -0.986422 | 0.002214 | 0.000990 | 5/5 | -0.989907 | -0.983905 |
| `epistemic_likelihood_credal` | -0.982256 | 0.006493 | 0.002904 | 5/5 | -0.989907 | -0.973058 |


## Benchmark Task: `cfg_rbv2_rpart_14`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.780665 | 0.011598 | 0.005187 | 5/5 | -0.800753 | -0.771493 |
| `epistemic_likelihood_credal` | -0.779082 | 0.013839 | 0.006189 | 5/5 | -0.796341 | -0.759588 |
| `epistemic_standard_disagreement` | -0.778068 | 0.015206 | 0.006800 | 5/5 | -0.795439 | -0.755103 |
| `epistemic_chen_variance` | -0.777253 | 0.012149 | 0.005433 | 5/5 | -0.796341 | -0.765800 |
| `epistemic_proximity_bc` | -0.772781 | 0.004010 | 0.001793 | 5/5 | -0.776086 | -0.766215 |
| `epistemic_proximity_b` | -0.771980 | 0.014321 | 0.006405 | 5/5 | -0.792770 | -0.754539 |
| `epistemic_proximity_auto_lambda` | -0.764590 | 0.015631 | 0.006990 | 5/5 | -0.783808 | -0.744649 |
| `epistemic_standard_proximity` | -0.761999 | 0.023509 | 0.010514 | 5/5 | -0.794029 | -0.732325 |
| `epistemic_shaker_entropy` | -0.741373 | 0.017202 | 0.007693 | 5/5 | -0.769095 | -0.727654 |


## Benchmark Task: `cfg_rbv2_rpart_40499`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.881203 | 0.003329 | 0.001489 | 5/5 | -0.885897 | -0.876677 |
| `epistemic_likelihood_credal` | -0.873513 | 0.003792 | 0.001696 | 5/5 | -0.879977 | -0.870288 |
| `epistemic_chen_variance` | -0.873433 | 0.004701 | 0.002102 | 5/5 | -0.879217 | -0.867237 |
| `epistemic_standard_disagreement` | -0.871136 | 0.009211 | 0.004119 | 5/5 | -0.881597 | -0.857036 |
| `epistemic_proximity_bc` | -0.868661 | 0.007133 | 0.003190 | 5/5 | -0.879628 | -0.860039 |
| `epistemic_proximity_b` | -0.864887 | 0.005181 | 0.002317 | 5/5 | -0.870916 | -0.858924 |
| `epistemic_proximity_auto_lambda` | -0.862043 | 0.012874 | 0.005758 | 5/5 | -0.875751 | -0.848194 |
| `epistemic_standard_proximity` | -0.861413 | 0.010201 | 0.004562 | 5/5 | -0.870276 | -0.849272 |
| `epistemic_shaker_entropy` | -0.854652 | 0.009835 | 0.004398 | 5/5 | -0.869679 | -0.845943 |


## Benchmark Task: `cfg_rbv2_super_1053`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_auto_lambda` | -0.888678 | 0.056087 | 0.025083 | 5/5 | -0.958932 | -0.835052 |
| `epistemic_likelihood_credal` | -0.887728 | 0.065892 | 0.029468 | 5/5 | -0.960928 | -0.808547 |
| `epistemic_standard_proximity` | -0.874770 | 0.052564 | 0.023507 | 5/5 | -0.958932 | -0.828751 |
| `epistemic_proximity_bc` | -0.873932 | 0.052485 | 0.023472 | 5/5 | -0.962362 | -0.837903 |
| `epistemic_proximity_b` | -0.867195 | 0.052841 | 0.023631 | 5/5 | -0.960589 | -0.837361 |
| `smac3_bo (Baseline)` | -0.865747 | 0.053521 | 0.023935 | 5/5 | -0.960138 | -0.834483 |
| `epistemic_shaker_entropy` | -0.849640 | 0.034250 | 0.015317 | 5/5 | -0.900388 | -0.808547 |
| `epistemic_standard_disagreement` | -0.845667 | 0.029034 | 0.012984 | 5/5 | -0.886101 | -0.806544 |
| `epistemic_chen_variance` | -0.840675 | 0.033458 | 0.014963 | 5/5 | -0.886101 | -0.806544 |


## Benchmark Task: `cfg_rbv2_super_1063`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.897574 | 0.043654 | 0.019523 | 5/5 | -0.958162 | -0.853447 |
| `epistemic_proximity_bc` | -0.897082 | 0.038142 | 0.017058 | 5/5 | -0.956514 | -0.853269 |
| `epistemic_proximity_auto_lambda` | -0.892230 | 0.012359 | 0.005527 | 5/5 | -0.900564 | -0.870349 |
| `epistemic_likelihood_credal` | -0.882481 | 0.041090 | 0.018376 | 5/5 | -0.951143 | -0.852217 |
| `epistemic_standard_disagreement` | -0.881035 | 0.024211 | 0.010827 | 5/5 | -0.899693 | -0.852329 |
| `epistemic_proximity_b` | -0.878313 | 0.017242 | 0.007711 | 5/5 | -0.900136 | -0.858714 |
| `epistemic_chen_variance` | -0.872012 | 0.024040 | 0.010751 | 5/5 | -0.899407 | -0.851691 |
| `epistemic_standard_proximity` | -0.870267 | 0.007473 | 0.003342 | 5/5 | -0.880390 | -0.861032 |
| `epistemic_shaker_entropy` | -0.858798 | 0.005934 | 0.002654 | 5/5 | -0.866072 | -0.851408 |


## Benchmark Task: `cfg_rbv2_super_1457`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_standard_proximity` | -0.826502 | 0.006995 | 0.003128 | 5/5 | -0.836262 | -0.819041 |
| `epistemic_proximity_bc` | -0.820755 | 0.005480 | 0.002451 | 5/5 | -0.826643 | -0.812835 |
| `epistemic_proximity_auto_lambda` | -0.820106 | 0.011555 | 0.005167 | 5/5 | -0.836025 | -0.807054 |
| `epistemic_chen_variance` | -0.813973 | 0.008807 | 0.003938 | 5/5 | -0.825980 | -0.802068 |
| `epistemic_standard_disagreement` | -0.811294 | 0.009800 | 0.004383 | 5/5 | -0.823805 | -0.796414 |
| `epistemic_proximity_b` | -0.811216 | 0.010023 | 0.004482 | 5/5 | -0.821320 | -0.796476 |
| `epistemic_likelihood_credal` | -0.805506 | 0.015669 | 0.007007 | 5/5 | -0.821287 | -0.785134 |
| `smac3_bo (Baseline)` | -0.803558 | 0.014808 | 0.006622 | 5/5 | -0.819624 | -0.782695 |
| `epistemic_shaker_entropy` | -0.779872 | 0.020311 | 0.009083 | 5/5 | -0.808215 | -0.760364 |


## Benchmark Task: `cfg_rbv2_super_1468`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.964602 | 0.014506 | 0.006487 | 5/5 | -0.976401 | -0.948339 |
| `epistemic_likelihood_credal` | -0.956035 | 0.012734 | 0.005695 | 5/5 | -0.976183 | -0.945732 |
| `epistemic_proximity_bc` | -0.953892 | 0.012772 | 0.005712 | 5/5 | -0.976133 | -0.945273 |
| `epistemic_chen_variance` | -0.953747 | 0.009243 | 0.004133 | 5/5 | -0.970151 | -0.947677 |
| `epistemic_proximity_b` | -0.950624 | 0.000736 | 0.000329 | 5/5 | -0.951878 | -0.950086 |
| `epistemic_standard_disagreement` | -0.950024 | 0.000301 | 0.000134 | 5/5 | -0.950443 | -0.949631 |
| `epistemic_proximity_auto_lambda` | -0.949323 | 0.001609 | 0.000719 | 5/5 | -0.950827 | -0.947233 |
| `epistemic_standard_proximity` | -0.947508 | 0.002567 | 0.001148 | 5/5 | -0.950066 | -0.943595 |
| `epistemic_shaker_entropy` | -0.945542 | 0.002051 | 0.000917 | 5/5 | -0.948107 | -0.943595 |


## Benchmark Task: `cfg_rbv2_super_1479`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_likelihood_credal` | -0.974806 | 0.041785 | 0.018687 | 5/5 | -0.994072 | -0.900064 |
| `epistemic_chen_variance` | -0.973766 | 0.043622 | 0.019508 | 5/5 | -0.993771 | -0.895734 |
| `smac3_bo (Baseline)` | -0.964182 | 0.063121 | 0.028229 | 5/5 | -0.993657 | -0.851294 |
| `epistemic_standard_disagreement` | -0.956640 | 0.082993 | 0.037115 | 5/5 | -0.994103 | -0.808180 |
| `epistemic_proximity_bc` | -0.945406 | 0.061409 | 0.027463 | 5/5 | -0.992937 | -0.841322 |
| `epistemic_proximity_auto_lambda` | -0.879130 | 0.150570 | 0.067337 | 5/5 | -0.983581 | -0.627943 |
| `epistemic_standard_proximity` | -0.864286 | 0.146137 | 0.065354 | 5/5 | -0.968125 | -0.618594 |
| `epistemic_proximity_b` | -0.847814 | 0.166078 | 0.074272 | 5/5 | -0.983740 | -0.616375 |
| `epistemic_shaker_entropy` | -0.813764 | 0.202816 | 0.090702 | 5/5 | -0.962862 | -0.579442 |


## Benchmark Task: `cfg_rbv2_super_15`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.986022 | 0.001256 | 0.000561 | 5/5 | -0.988231 | -0.985092 |
| `epistemic_proximity_b` | -0.983396 | 0.001481 | 0.000662 | 5/5 | -0.985440 | -0.981517 |
| `epistemic_standard_proximity` | -0.982929 | 0.004358 | 0.001949 | 5/5 | -0.987334 | -0.975708 |
| `epistemic_proximity_bc` | -0.981168 | 0.002391 | 0.001069 | 5/5 | -0.983504 | -0.977532 |
| `epistemic_standard_disagreement` | -0.980884 | 0.008296 | 0.003710 | 5/5 | -0.986261 | -0.966169 |
| `epistemic_proximity_auto_lambda` | -0.980247 | 0.002207 | 0.000987 | 5/5 | -0.983241 | -0.977532 |
| `epistemic_chen_variance` | -0.980119 | 0.008046 | 0.003598 | 5/5 | -0.985547 | -0.966169 |
| `epistemic_likelihood_credal` | -0.977051 | 0.003053 | 0.001365 | 5/5 | -0.980647 | -0.972641 |
| `epistemic_shaker_entropy` | -0.975492 | 0.005688 | 0.002544 | 5/5 | -0.980647 | -0.966169 |


## Benchmark Task: `cfg_rbv2_xgboost_12`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `smac3_bo (Baseline)` | -0.983456 | 0.000891 | 0.000398 | 5/5 | -0.984788 | -0.982371 |
| `epistemic_chen_variance` | -0.982627 | 0.001054 | 0.000471 | 5/5 | -0.984440 | -0.981886 |
| `epistemic_standard_disagreement` | -0.982542 | 0.001056 | 0.000472 | 5/5 | -0.983567 | -0.980961 |
| `epistemic_proximity_auto_lambda` | -0.981478 | 0.000930 | 0.000416 | 5/5 | -0.982291 | -0.979968 |
| `epistemic_standard_proximity` | -0.981223 | 0.002688 | 0.001202 | 5/5 | -0.984481 | -0.977750 |
| `epistemic_proximity_bc` | -0.980580 | 0.001785 | 0.000798 | 5/5 | -0.983299 | -0.978954 |
| `epistemic_proximity_b` | -0.980325 | 0.004030 | 0.001802 | 5/5 | -0.984378 | -0.973669 |
| `epistemic_likelihood_credal` | -0.972788 | 0.007894 | 0.003530 | 5/5 | -0.982238 | -0.961305 |
| `epistemic_shaker_entropy` | -0.968149 | 0.017846 | 0.007981 | 5/5 | -0.979324 | -0.937399 |


## Benchmark Task: `cfg_rbv2_xgboost_1501`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_standard_proximity` | -0.942055 | 0.005594 | 0.002502 | 5/5 | -0.950699 | -0.935622 |
| `smac3_bo (Baseline)` | -0.937371 | 0.008824 | 0.003946 | 5/5 | -0.948057 | -0.929743 |
| `epistemic_proximity_auto_lambda` | -0.936174 | 0.007829 | 0.003501 | 5/5 | -0.945218 | -0.927888 |
| `epistemic_standard_disagreement` | -0.934979 | 0.014901 | 0.006664 | 5/5 | -0.949716 | -0.909955 |
| `epistemic_chen_variance` | -0.933592 | 0.020931 | 0.009361 | 5/5 | -0.949691 | -0.897066 |
| `epistemic_proximity_b` | -0.930677 | 0.011787 | 0.005271 | 5/5 | -0.947860 | -0.914840 |
| `epistemic_proximity_bc` | -0.928587 | 0.009340 | 0.004177 | 5/5 | -0.942712 | -0.919877 |
| `epistemic_likelihood_credal` | -0.908220 | 0.012171 | 0.005443 | 5/5 | -0.925738 | -0.896461 |
| `epistemic_shaker_entropy` | -0.878297 | 0.033508 | 0.014985 | 5/5 | -0.915819 | -0.825857 |


## Benchmark Task: `cfg_rbv2_xgboost_16`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_standard_proximity` | -0.978009 | 0.012540 | 0.005608 | 5/5 | -0.998812 | -0.965699 |
| `epistemic_proximity_b` | -0.973370 | 0.005265 | 0.002355 | 5/5 | -0.979294 | -0.967065 |
| `epistemic_standard_disagreement` | -0.972936 | 0.006541 | 0.002925 | 5/5 | -0.980153 | -0.964715 |
| `epistemic_proximity_auto_lambda` | -0.972110 | 0.008182 | 0.003659 | 5/5 | -0.979488 | -0.959452 |
| `epistemic_chen_variance` | -0.970826 | 0.010102 | 0.004518 | 5/5 | -0.978252 | -0.953555 |
| `smac3_bo (Baseline)` | -0.970210 | 0.005580 | 0.002495 | 5/5 | -0.974773 | -0.960939 |
| `epistemic_proximity_bc` | -0.968173 | 0.007684 | 0.003437 | 5/5 | -0.977489 | -0.958641 |
| `epistemic_likelihood_credal` | -0.957671 | 0.011492 | 0.005139 | 5/5 | -0.974052 | -0.942984 |
| `epistemic_shaker_entropy` | -0.935107 | 0.025987 | 0.011622 | 5/5 | -0.953555 | -0.889630 |


## Benchmark Task: `cfg_rbv2_xgboost_40499`

| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `epistemic_proximity_b` | -0.998243 | 0.002675 | 0.001196 | 5/5 | -1.000000 | -0.993915 |
| `epistemic_likelihood_credal` | -0.997802 | 0.003646 | 0.001630 | 5/5 | -1.000000 | -0.991596 |
| `smac3_bo (Baseline)` | -0.997214 | 0.006003 | 0.002685 | 5/5 | -1.000000 | -0.986479 |
| `epistemic_proximity_auto_lambda` | -0.996783 | 0.007190 | 0.003216 | 5/5 | -1.000000 | -0.983920 |
| `epistemic_standard_proximity` | -0.996374 | 0.005001 | 0.002237 | 5/5 | -1.000000 | -0.990087 |
| `epistemic_standard_disagreement` | -0.995196 | 0.006879 | 0.003076 | 5/5 | -1.000000 | -0.985144 |
| `epistemic_proximity_bc` | -0.993508 | 0.006026 | 0.002695 | 5/5 | -1.000000 | -0.987811 |
| `epistemic_chen_variance` | -0.991202 | 0.005014 | 0.002242 | 5/5 | -0.998270 | -0.984780 |
| `epistemic_shaker_entropy` | -0.976512 | 0.023298 | 0.010419 | 5/5 | -0.999969 | -0.939390 |

