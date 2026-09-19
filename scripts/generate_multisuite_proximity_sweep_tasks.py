#!/usr/bin/env python3
"""Unified Multi-Suite Generator for Large-Scale Realworld Proximity LCB Benchmark Sweeps.

Generates paired Hydra task commands and SLURM array submit scripts for:
1. Suite 1: YAHPO rbv2_ranger (119 tasks * 2 optimizers * 30 seeds = 7,140 runs)
2. Suite 2: YAHPO rbv2_super (103 tasks * 2 optimizers * 30 seeds = 6,180 runs)
3. Suite 3: HPOBench Tabular ML (88 tasks * 2 optimizers * 30 seeds = 5,280 runs)

Grand Total: 310 tasks, 18,600 runs across all 3 suites.

To strictly adhere to SLURM cluster array limits (<= 5,000 tasks per job array),
each suite is automatically partitioned into manageable chunks (default <= 2,500 tasks),
producing dedicated `tasks_part*.txt`, `submit_array_part*.sbatch`, and a master
`submit_all_parts.sh` per suite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry


SUITE_DISPLAY_NAMES = {
    "yahpo_rbv2_ranger": "YAHPO rbv2_ranger (119 tasks)",
    "yahpo_rbv2_super": "YAHPO rbv2_super (103 tasks)",
    "hpobench_ml": "HPOBench Tabular ML (88 tasks)",
}

SUITE_JOB_PREFIXES = {
    "yahpo_rbv2_ranger": "rngr",
    "yahpo_rbv2_super": "supr",
    "hpobench_ml": "hpob",
}


def load_proximity_params(
    meta_json_path: str | None = None,
    k: int = 25,
    decay_lambda: float = 1.345,
    eps: float = 0.16,
    level: float = 0.95,
    uncertainty_func: str = "proximity_b",
) -> Dict[str, Any]:
    """Loads proximity parameters from SMAC4HPO meta-tuning json or falls back to defaults."""
    params = {
        "k": k,
        "decay_lambda": decay_lambda,
        "eps": eps,
        "level": level,
        "uncertainty_func": uncertainty_func,
    }
    if meta_json_path and os.path.isfile(meta_json_path):
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both direct flat keys and nested config structures
            cfg = data.get("best_config", data.get("config", data))
            if "k" in cfg:
                params["k"] = int(cfg["k"])
            if "decay_lambda" in cfg:
                params["decay_lambda"] = float(cfg["decay_lambda"])
            if "eps" in cfg:
                params["eps"] = float(cfg["eps"])
            if "level" in cfg:
                params["level"] = float(cfg["level"])
            if "uncertainty_func" in cfg:
                params["uncertainty_func"] = str(cfg["uncertainty_func"])
            print(f"[INFO] Successfully loaded tuned hyperparameters from {meta_json_path}: {params}")
        except Exception as e:
            print(f"[WARN] Failed loading from {meta_json_path} ({e}). Using provided defaults.")
    return params


def generate_suite_tasks(
    suite: str,
    task_list: List[str],
    seeds: Union[int, Sequence[int]] = 30,
    trials: int = 100,
    proximity_params: Dict[str, Any] | None = None,
    kappa_baseline: float = 1.96,
) -> List[str]:
    """Generates all paired Hydra command lines for a given suite."""
    if proximity_params is None:
        proximity_params = load_proximity_params()

    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    beta_baseline = round(float(kappa_baseline) ** 2, 4)  # 1.96^2 = 3.8416
    runs_dir = f"results/sweep_{suite}_proximity"
    baserundir = f"runs/sweep_{suite}_proximity"

    k_val = proximity_params["k"]
    level_val = proximity_params["level"]
    eps_val = proximity_params["eps"]
    decay_val = proximity_params["decay_lambda"]
    unc_func = proximity_params["uncertainty_func"]

    lines: List[str] = []

    # 1. Proposed Method: SMAC20_ProximityLCB
    for task_arg in task_list:
        task_id = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC20_ProximityLCB_{task_id}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac20_proximity_lcb "
                f"++optimizer.acq_func_kwargs.k={k_val} "
                f"++optimizer.acq_func_kwargs.level={level_val} "
                f"++optimizer.acq_func_kwargs.eps={eps_val} "
                f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={unc_func} "
                f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={decay_val} "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC20_ProximityLCB optimizer_container_id=SMAC20_ProximityLCB"
            )
            lines.append(cmd)

    # 2. Baseline Method: SMAC3_HPOFacade_lcb (kappa=1.96, beta=3.8416)
    for task_arg in task_list:
        task_id = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC3_HPOFacade_lcb_{task_id}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer/smac20=hpo "
                f"++optimizer.acq_func_name=lcb "
                f"++optimizer.acq_func_kwargs.beta={beta_baseline} "
                f"++optimizer.acq_func_kwargs.update_beta=false "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC3_HPOFacade_lcb optimizer_container_id=SMAC3_HPOFacade"
            )
            lines.append(cmd)

    return lines


def chunk_tasks(tasks: List[str], max_chunk_size: int = 2500) -> List[List[str]]:
    """Splits a list of tasks into chunks strictly bounded by max_chunk_size (<= 5,000)."""
    if not tasks:
        return []
    n_chunks = math.ceil(len(tasks) / max_chunk_size)
    chunk_size = math.ceil(len(tasks) / n_chunks)
    chunks = []
    for i in range(0, len(tasks), chunk_size):
        chunks.append(tasks[i : i + chunk_size])
    return chunks


def generate_sbatch_content(
    suite: str,
    task_file: str,
    log_dir: str,
    partition: str = "ai",
) -> str:
    """Generates the SLURM sbatch script content without hardcoded array limits."""
    prefix = SUITE_JOB_PREFIXES.get(suite, suite[:4])
    job_name = f"{prefix}_prox"
    
    return f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%A_%a.log
#SBATCH --error={log_dir}/{job_name}_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Initialize conda / venv
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p {log_dir}

TASK_FILE="{task_file}"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Suite: {suite}"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/run_carps_patched.py $TASK_ARGS
else
    python3 scripts/run_carps_patched.py $TASK_ARGS
fi

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
"""


def generate_launcher_sh_content(
    suite: str,
    task_file: str,
    sbatch_file: str,
    chunk_size: int = 200,
    concurrency: int = 25,
) -> str:
    """Generates the master chunked submission shell script adhering to LUIS MaxArraySize <= 300."""
    return f"""#!/bin/bash
set -e

TASK_FILE="{task_file}"
SBATCH_FILE="{sbatch_file}"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
CHUNK_SIZE={chunk_size}
CONCURRENCY={concurrency}

echo "=================================================="
echo "Submitting $TOTAL_TASKS tasks for {suite}"
echo "Chunk Size: $CHUNK_SIZE (LUIS MaxArraySize <= 300, %$CONCURRENCY concurrency)"
echo "=================================================="

# Optional range overrides from CLI: e.g. ./script.sh [START_TASK] [END_TASK]
REQ_START=${{1:-1}}
REQ_END=${{2:-$TOTAL_TASKS}}

if [ "$REQ_START" -lt 1 ]; then REQ_START=1; fi
if [ "$REQ_END" -gt "$TOTAL_TASKS" ]; then REQ_END=$TOTAL_TASKS; fi

for (( start=REQ_START; start<=REQ_END; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $REQ_END ]; then
        end=$REQ_END
    fi
    JOB_ID=$(sbatch --parsable --array=${{start}}-${{end}}%${{CONCURRENCY}} "$SBATCH_FILE")
    echo "Submitted Chunk (${{start}}-${{end}} / ${{TOTAL_TASKS}}) -> Job ID: ${{JOB_ID}}"
done

echo "=================================================="
echo "{suite} successfully scheduled on LUIS cluster!"
echo "=================================================="
"""


def generate_batch_sh_content(
    suite: str,
    batch_num: int,
    start_task: int,
    end_task: int,
    total_tasks: int,
    task_file: str,
    sbatch_file: str,
    chunk_size: int = 200,
    concurrency: int = 25,
) -> str:
    """Generates a standalone batch runner script bounded below the 5,000 QOS limit."""
    return f"""#!/bin/bash
set -e

TASK_FILE="{task_file}"
SBATCH_FILE="{sbatch_file}"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
START_TASK={start_task}
END_TASK={end_task}
CHUNK_SIZE={chunk_size}
CONCURRENCY={concurrency}

echo "=================================================="
echo "Submitting {suite} - BATCH {batch_num} (Tasks $START_TASK to $END_TASK of $TOTAL_TASKS)"
echo "Chunk Size: $CHUNK_SIZE (LUIS MaxArraySize <= 300, %$CONCURRENCY concurrency)"
echo "=================================================="

for (( start=START_TASK; start<=END_TASK; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $END_TASK ]; then
        end=$END_TASK
    fi
    JOB_ID=$(sbatch --parsable --array=${{start}}-${{end}}%${{CONCURRENCY}} "$SBATCH_FILE")
    echo "Submitted Chunk (${{start}}-${{end}} / ${{TOTAL_TASKS}}) -> Job ID: ${{JOB_ID}}"
done

echo "=================================================="
echo "Batch {batch_num} for {suite} successfully submitted!"
echo "=================================================="
"""


def generate_orchestrator_sh_content(all_batch_scripts: List[str]) -> str:
    """Generates the master orchestrator shell script that polls the SLURM queue."""
    script_lines = "\n".join(f'    "{s}"' for s in all_batch_scripts)
    return f"""#!/bin/bash
# Master Multi-Suite Proximity LCB Sweep Orchestrator
# Automatically executes sweep batches sequentially, waiting for queue to empty between batches.
set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR/.."

BATCHES=(
{script_lines}
)

wait_for_queue_empty() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring SLURM queue for user $USER..."
    while true; do
        if [ -n "$SLURM_JOB_ID" ]; then
            PENDING_OR_RUNNING=$(squeue -u "$USER" -h -t R,PD | grep -v "^ *$SLURM_JOB_ID " | wc -l)
        else
            PENDING_OR_RUNNING=$(squeue -u "$USER" -h -t R,PD | wc -l)
        fi
        if [ "$PENDING_OR_RUNNING" -le 1 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue is clear ($PENDING_OR_RUNNING active job(s) remaining, proceeding to next batch)."
            break
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Active jobs in queue: $PENDING_OR_RUNNING. Checking again in 60s..."
        sleep 60
    done
}}

# Allow starting from a specific batch (default: 2, skipping already completed batch 1)
START_BATCH="${{1:-2}}"

echo "=================================================="
echo "Starting Automated Multi-Suite Sweep Orchestrator"
echo "Total Batches available: ${{#BATCHES[@]}}"
echo "Starting from Batch: $START_BATCH"
echo "=================================================="

for idx in "${{!BATCHES[@]}}"; do
    batch_script="${{BATCHES[$idx]}}"
    batch_num=$(( idx + 1 ))
    if [ "$batch_num" -lt "$START_BATCH" ]; then
        echo "Skipping Batch $batch_num (already completed): $batch_script"
        continue
    fi
    echo ""
    echo "=================================================="
    echo "Executing Batch $batch_num / ${{#BATCHES[@]}}: $batch_script"
    echo "=================================================="
    bash "$batch_script"

    echo "Batch $batch_num submitted. Waiting 15s for SLURM scheduler to update..."
    sleep 15

    wait_for_queue_empty
    echo "Batch $batch_num completed!"
done

echo ""
echo "=================================================="
echo "ALL MULTI-SUITE SWEEPS COMPLETED SUCCESSFULLY!"
echo "=================================================="
"""


def generate_all_suite_artifacts(
    suites: Sequence[str] = ("yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml"),
    seeds: int = 30,
    trials: int = 100,
    chunk_size: int = 200,
    max_batch_size: int = 4000,
    meta_json_path: str | None = None,
    k: int = 25,
    decay_lambda: float = 1.345,
    eps: float = 0.16,
    level: float = 0.95,
    uncertainty_func: str = "proximity_b",
    partition: str = "ai",
    concurrency: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Generates tasks, LUIS-compliant sbatch scripts, batches, and orchestrator."""
    params = load_proximity_params(
        meta_json_path=meta_json_path,
        k=k,
        decay_lambda=decay_lambda,
        eps=eps,
        level=level,
        uncertainty_func=uncertainty_func,
    )

    results_summary = {}
    all_batch_scripts = []
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        task_list = CarpsRealworldRegistry.get_tasks_for_suite(suite)
        cmds = generate_suite_tasks(
            suite=suite,
            task_list=task_list,
            seeds=seeds,
            trials=trials,
            proximity_params=params,
        )

        suite_dir = Path(f"results/sweep_{suite}_proximity")
        log_dir = suite_dir / "slurm_logs"
        runs_dir = Path(f"runs/sweep_{suite}_proximity")

        suite_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Master task file
        master_task_file = suite_dir / "tasks.txt"
        with open(master_task_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cmds) + "\n")

        # Clean up any legacy chunk files and scripts
        for part_file in suite_dir.glob("tasks_part*.txt"):
            part_file.unlink()
        for legacy_file in suite_dir.glob("*.sbatch"):
            legacy_file.unlink()
        for legacy_file in suite_dir.glob("*.sh"):
            legacy_file.unlink()
        for obsolete_p in scripts_dir.glob(f"submit_sweep_{suite}_proximity_p*.sbatch"):
            obsolete_p.unlink()

        # 2. Single clean array sbatch script
        sbatch_path = scripts_dir / f"submit_sweep_{suite}_proximity_array.sbatch"
        sbatch_content = generate_sbatch_content(
            suite=suite,
            task_file=str(master_task_file),
            log_dir=str(log_dir),
            partition=partition,
        )
        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(sbatch_content)

        # 3. Master full sweep launcher
        submit_sh = scripts_dir / f"submit_sweep_{suite}_proximity.sh"
        sh_content = generate_launcher_sh_content(
            suite=suite,
            task_file=str(master_task_file),
            sbatch_file=str(sbatch_path),
            chunk_size=chunk_size,
            concurrency=concurrency,
        )
        with open(submit_sh, "w", encoding="utf-8") as f:
            f.write(sh_content)
        os.chmod(submit_sh, 0o755)

        # 4. Standalone bounded batch scripts (strictly <= max_batch_size to never exceed QOS limit)
        suite_batch_scripts = []
        total_runs = len(cmds)
        if total_runs <= max_batch_size:
            batch_ranges = [(1, 1, total_runs)]
        else:
            mid = ((total_runs // 2) // chunk_size) * chunk_size
            if mid == 0:
                mid = total_runs // 2
            batch_ranges = [
                (1, 1, mid),
                (2, mid + 1, total_runs),
            ]

        for b_num, b_start, b_end in batch_ranges:
            b_script_path = scripts_dir / f"submit_sweep_{suite}_proximity_batch{b_num}.sh"
            b_content = generate_batch_sh_content(
                suite=suite,
                batch_num=b_num,
                start_task=b_start,
                end_task=b_end,
                total_tasks=total_runs,
                task_file=str(master_task_file),
                sbatch_file=str(sbatch_path),
                chunk_size=chunk_size,
                concurrency=concurrency,
            )
            with open(b_script_path, "w", encoding="utf-8") as f:
                f.write(b_content)
            os.chmod(b_script_path, 0o755)
            suite_batch_scripts.append(str(b_script_path))
            all_batch_scripts.append(str(b_script_path))

        results_summary[suite] = {
            "tasks_count": len(task_list),
            "total_runs": len(cmds),
            "chunk_size": chunk_size,
            "master_file": str(master_task_file),
            "sbatch_file": str(sbatch_path),
            "submit_sh": str(submit_sh),
            "batch_scripts": suite_batch_scripts,
        }

    # 5. Master Orchestrator script for all batches
    orchestrator_path = scripts_dir / "orchestrate_all_sweeps.sh"
    orchestrator_content = generate_orchestrator_sh_content(all_batch_scripts)
    with open(orchestrator_path, "w", encoding="utf-8") as f:
        f.write(orchestrator_content)
    os.chmod(orchestrator_path, 0o755)

    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Multi-Suite Generator for Large-Scale Realworld Proximity LCB Benchmark Sweeps."
    )
    parser.add_argument(
        "--suite",
        choices=["yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml", "all"],
        default="all",
        help="Which benchmark suite to generate (default: all 3 suites).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of seeds to run per task (default: 30).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials per run (default: 100).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Chunk size per SLURM array submission to strictly obey LUIS MaxArraySize <= 300 (default: 200).",
    )
    parser.add_argument(
        "--from-meta-json",
        type=str,
        default="results/meta_smac_proximity_hpo/best_config.json",
        help="Path to best_config.json from meta-tuning (if exists).",
    )
    parser.add_argument("--k", type=int, default=25, help="Proximity k-neighbors.")
    parser.add_argument("--decay-lambda", type=float, default=1.345, help="Proximity decay lambda.")
    parser.add_argument("--eps", type=float, default=0.16, help="Proximity epsilon floor.")
    parser.add_argument("--level", type=float, default=0.95, help="Proximity confidence level.")
    parser.add_argument("--uncertainty-func", type=str, default="proximity_b", help="Uncertainty func.")
    parser.add_argument("--partition", type=str, default="ai", help="SLURM partition.")
    parser.add_argument("--concurrency", type=int, default=25, help="SLURM array concurrency limit %%N (default: 25).")

    args = parser.parse_args()

    suites = (
        ["yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml"]
        if args.suite == "all"
        else [args.suite]
    )

    summary = generate_all_suite_artifacts(
        suites=suites,
        seeds=args.seeds,
        trials=args.trials,
        chunk_size=args.chunk_size,
        meta_json_path=args.from_meta_json,
        k=args.k,
        decay_lambda=args.decay_lambda,
        eps=args.eps,
        level=args.level,
        uncertainty_func=args.uncertainty_func,
        partition=args.partition,
        concurrency=args.concurrency,
    )

    print("\n=======================================================")
    print("Multi-Suite Proximity LCB Benchmark Generation Summary:")
    print("=======================================================")
    for s, info in summary.items():
        print(f"[{s}]")
        print(f"  Distinct Tasks: {info['tasks_count']}")
        print(f"  Total Runs (30 seeds * 2 algs): {info['total_runs']}")
        print(f"  Array Sbatch Script: {info['sbatch_file']}")
        print(f"  Launcher Script (Chunk Size: {info['chunk_size']}): {info['submit_sh']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
