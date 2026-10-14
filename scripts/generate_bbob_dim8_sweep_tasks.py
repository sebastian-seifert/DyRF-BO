#!/usr/bin/env python3
"""Task Generator for Dedicated CARP-S BBOB Dimension 8 Benchmark Sweep.

Generates CLI task strings for all 11 evaluated optimizers on 8D BBOB tasks:
- Baselines (2):
  1. SMAC3_HPOFacade_lcb
  2. SMAC3_HPOFacade_ei
- Entropy (1):
  3. SMAC20_Entropy_LCB
- Proximity Family (8):
  4. SMAC20_ProximityA_LCB
  5. SMAC20_ProximityA_LCB_CV
  6. SMAC20_ProximityAC_LCB
  7. SMAC20_ProximityAC_LCB_CV
  8. SMAC20_ProximityB_LCB
  9. SMAC20_ProximityB_LCB_CV
  10. SMAC20_ProximityBC_LCB
  11. SMAC20_ProximityBC_LCB_CV

Across:
- 72 CARP-S BBOB Tasks (Dim 8 * 24 functions * 3 instances)
- 20 strictly paired seeds (seeds 1 to 20 by default)
- 100 trials budget per run (T=100)

Total: 72 tasks * 11 optimizers * 20 seeds = 15,840 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbob_full_sweep_tasks import (
    ALL_OPTIMIZERS,
    generate_bbob_full_sweep_tasks,
)

BBOB_DIM8_OUTPUT_DIR = "results/sweep_bbob_dim8"
BBOB_DIM8_RUNS_DIR = "runs/sweep_bbob_dim8"


def generate_bbob_dim8_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    output_dir: str = BBOB_DIM8_OUTPUT_DIR,
    runs_dir: str = BBOB_DIM8_OUTPUT_DIR,
    baserundir: str = BBOB_DIM8_RUNS_DIR,
) -> List[str]:
    """Generates task command lines strictly for dimension 8 BBOB benchmarks."""
    commands = generate_bbob_full_sweep_tasks(
        seeds=seeds,
        trials=trials,
        dimensions=[8],
        output_dir=output_dir,
        runs_dir=runs_dir,
        baserundir=baserundir,
    )

    if output_dir:
        standard_manifest = os.path.join(output_dir, "tasks.txt")
        with open(standard_manifest, "w", encoding="utf-8") as f:
            for cmd in commands:
                f.write(f"{cmd}\n")

    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dedicated BBOB Dimension 8 sweep tasks.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per run (default: 100)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BBOB_DIM8_OUTPUT_DIR,
        help="Output directory for task manifests",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=BBOB_DIM8_OUTPUT_DIR,
        help="Directory where telemetry files will be saved",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default=BBOB_DIM8_RUNS_DIR,
        help="Baserundir for CARP-S raw trial logs",
    )

    args = parser.parse_args()
    cmds = generate_bbob_dim8_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
    )
    print(f"[SUCCESS] Generated {len(cmds)} BBOB Dimension 8 tasks in '{args.output_dir}'")
