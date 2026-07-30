#!/usr/bin/env python3
"""Task generator for smoke testing Epistemic BO Sweep across EI, PI, and LCB acquisitions."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

def generate_smoke_epistemic_full_acq_tasks(output_path: str = "results/smoke_epistemic_full_acq_tasks.txt") -> list:
    acquisitions = ["ei", "pi", "lcb"]
    smoke_task = "+task/YAHPO/SO=cfg_rbv2_glmnet_375"
    smoke_approach = "standard_disagreement"
    seed = 1
    trials = 2

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Epistemic runs (EI, PI, LCB)
    for acq in acquisitions:
        telemetry = f"results/epistemic_acq/smoke/{acq}/telemetry_epistemic_{acq}_{smoke_approach}_branin_seed{seed}.json"
        line = (
            f"+optimizer=smac20_custom_uncertainty "
            f"++optimizer.acq_func_name={acq} "
            f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={smoke_approach} "
            f"{smoke_task} task.optimization_resources.n_trials={trials} "
            f"seed={seed} ++optimizer.telemetry_path={telemetry} "
            f"optimizer_id=SMAC20_CustomUncertainty_{acq}_{smoke_approach} optimizer_container_id=SMAC20_CustomUncertainty"
        )
        lines.append(line)

    # 2. SMAC3 BO baseline runs (EI, PI, LCB)
    for acq in acquisitions:
        telemetry = f"results/epistemic_acq/smoke/baseline/{acq}/telemetry_smac3_{acq}_branin_seed{seed}.json"
        line = (
            f"+optimizer/smac20=hpo "
            f"++optimizer.acq_func_name={acq} "
            f"{smoke_task} "
            f"task.optimization_resources.n_trials={trials} "
            f"seed={seed} ++optimizer.telemetry_path={telemetry} "
            f"optimizer_id=SMAC3_HPOFacade_{acq} optimizer_container_id=SMAC3"
        )
        lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} smoke test tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_smoke_epistemic_full_acq_tasks()
