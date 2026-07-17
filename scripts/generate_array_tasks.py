#!/usr/bin/env python3
import os

# Define sweep variables
seeds = [1, 2, 3, 4, 5]
trials = 50
tasks = [
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_12",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_31",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_3",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_12",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_31",
    "+task/YAHPO/SO=cfg_lcbench_167168",
    "+task/YAHPO/SO=cfg_lcbench_189873",
    "+task/YAHPO/SO=cfg_lcbench_189906",
    "+task/YAHPO/SO=cfg_nb301_CIFAR10",
    "+task/YAHPO/SO=cfg_rbv2_glmnet_375",
    "+task/YAHPO/SO=cfg_rbv2_glmnet_458",
    "+task/YAHPO/SO=cfg_rbv2_ranger_16",
    "+task/YAHPO/SO=cfg_rbv2_ranger_42",
    "+task/YAHPO/SO=cfg_rbv2_rpart_14",
    "+task/YAHPO/SO=cfg_rbv2_rpart_40499",
    "+task/YAHPO/SO=cfg_rbv2_super_1053",
    "+task/YAHPO/SO=cfg_rbv2_super_1063",
    "+task/YAHPO/SO=cfg_rbv2_super_1457",
    "+task/YAHPO/SO=cfg_rbv2_super_1468",
    "+task/YAHPO/SO=cfg_rbv2_super_1479",
    "+task/YAHPO/SO=cfg_rbv2_super_15",
    "+task/YAHPO/SO=cfg_rbv2_xgboost_12",
    "+task/YAHPO/SO=cfg_rbv2_xgboost_1501",
    "+task/YAHPO/SO=cfg_rbv2_xgboost_16",
    "+task/YAHPO/SO=cfg_rbv2_xgboost_40499"
]


approaches = [
    "standard_disagreement",
    "chen_variance",
    "shaker_entropy",
    "likelihood_credal",
    "standard_proximity",
    "proximity_b",
    "proximity_bc"
]

os.makedirs("results", exist_ok=True)
task_file_path = "results/array_tasks.txt"

with open(task_file_path, "w") as f:
    # 1. Write the custom Dynamic UQ runs
    for task in tasks:
        task_name = task.split("=")[-1]
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/telemetry_{approach}_{task_name}_seed{seed}.json"
                f.write(f"+optimizer=dyrf_epistemic_hpobench optimizer.extractor_name={approach} {task} task.optimization_resources.n_trials={trials} seed={seed} optimizer.telemetry_path={telemetry}\n")
                
    # 2. Write the standard SMAC3 baseline runs
    for task in tasks:
        for seed in seeds:
            f.write(f"+optimizer/smac20=hpo {task} task.optimization_resources.n_trials={trials} seed={seed}\n")

print(f"Generated {sum(1 for _ in open(task_file_path))} configuration lines in {task_file_path}")
