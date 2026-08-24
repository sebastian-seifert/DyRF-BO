#!/usr/bin/env python3
"""Task Generator for Aleatoric OOD Masterplan Sweep.

Generates task commands for 15 benchmark functions x 7 noise regimes x 5 RF configs x 5 seeds = 2625 tasks.
Outputs commands targeting results/OOD_Aleatoric_Sweep/json/.
"""

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ep_extractors.aleatoric_ood_masterplan import get_benchmark_functions, get_noise_regimes, get_rf_configs

def generate_aleatoric_ood_tasks(output_file: Optional[str] = None) -> List[str]:
    if output_file is None:
        os.makedirs("results/OOD_Aleatoric_Sweep", exist_ok=True)
        output_file = "results/OOD_Aleatoric_Sweep/aleatoric_ood_tasks.txt"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    funcs = list(get_benchmark_functions().keys())
    noises = list(get_noise_regimes().keys())
    rf_configs = list(get_rf_configs().keys())
    seeds = [1, 2, 3, 4, 5]

    tasks = []
    for f in funcs:
        for n in noises:
            for rf_cfg in rf_configs:
                for s in seeds:
                    cmd = (
                        f"python -c 'from ep_extractors.aleatoric_ood_masterplan import run_single_aleatoric_ood_experiment; "
                        f"import json, os; "
                        f"res = run_single_aleatoric_ood_experiment(\"{f}\", \"{n}\", \"{rf_cfg}\", seed={s}); "
                        f"os.makedirs(\"results/OOD_Aleatoric_Sweep/json\", exist_ok=True); "
                        f"open(\"results/OOD_Aleatoric_Sweep/json/res_{f}_{n}_{rf_cfg}_seed{s}.json\", \"w\").write(json.dumps(res, indent=2))'"
                    )
                    tasks.append(cmd)

    with open(output_file, "w") as tf:
        for t in tasks:
            tf.write(t + "\n")

    print(f"Generated {len(tasks)} tasks in '{output_file}'.")
    return tasks

if __name__ == "__main__":
    generate_aleatoric_ood_tasks()
