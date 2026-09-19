#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=hpob_p2
#SBATCH --output=results/sweep_hpobench_ml_proximity/slurm_logs/hpob_p2_%A_%a.log
#SBATCH --error=results/sweep_hpobench_ml_proximity/slurm_logs/hpob_p2_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-1760%64

# Initialize conda / venv
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/sweep_hpobench_ml_proximity/slurm_logs

TASK_FILE="results/sweep_hpobench_ml_proximity/tasks_part2.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Suite: hpobench_ml | Part: 2"
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
