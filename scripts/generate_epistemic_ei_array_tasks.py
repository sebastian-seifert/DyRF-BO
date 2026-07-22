#!/usr/bin/env python3
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

def generate_array_tasks(output_path: str = "results/epistemic_ei_array_tasks.txt") -> list:
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

    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Epistemic EI runs across all 8 approaches (1,040 tasks)
    for task in tasks:
        task_name = task.split("=")[-1]
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/telemetry_epistemic_{approach}_{task_name}_seed{seed}.json"
                line = (
                    f"+optimizer=dyrf_epistemic_ei "
                    f"++optimizer.extractor_name={approach} "
                    f"++optimizer.acq_uncertainty_type=epistemic "
                    f"++optimizer.enable_adaptation=false "
                    f"{task} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=CARPSDynamicRF_epistemic optimizer_container_id=CARPSDynamicRF"
                )
                lines.append(line)

    # 2. Standard SMAC3 BO baseline runs (130 tasks)
    for task in tasks:
        for seed in seeds:
            line = (
                f"+optimizer/smac20=hpo {task} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} optimizer_id=SMAC3-HPOFacade optimizer_container_id=SMAC3"
            )
            lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} array tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_array_tasks()
