#!/usr/bin/env python3
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

def generate_array_tasks(output_dir: str = "results/carps_epistemic_ei_scaled") -> list:
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

    # Exclude proximity_auto_lambda explicitly
    approaches = [app for app in UQExtractorRegistry.list_registered() if app != "proximity_auto_lambda"]

    os.makedirs(output_dir, exist_ok=True)
    tasks_file = os.path.join(output_dir, "tasks.txt")
    metadata_file = os.path.join(output_dir, "metadata.json")

    lines = []

    # 1. DyRF-BO Epistemic EI runs across all 7 extractors (excluding auto-lambda) (910 tasks)
    for task in tasks:
        task_name = task.split("=")[-1]
        for approach in approaches:
            for seed in seeds:
                telemetry = f"{output_dir}/raw/telemetry_epistemic_{approach}_{task_name}_seed{seed}.json"
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

    with open(tasks_file, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    metadata = {
        "sweep_name": "carps_epistemic_ei_scaled",
        "description": "CARP-S EU-guided EI Re-run with correctly scaled signals across 26 tasks, 5 seeds, 7 extractors + SMAC3 baseline",
        "tasks_count": len(tasks),
        "seeds": seeds,
        "approaches": approaches + ["smac3_bo"],
        "total_tasks": len(lines)
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(lines)} array task lines in {tasks_file}")
    return lines

if __name__ == "__main__":
    generate_array_tasks()
