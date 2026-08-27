#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=1v1_sweep
#SBATCH --output=results/slurm_logs/%x_%A_%a.log
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Initialize conda / venv
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/slurm_logs

if [ -z "$SWEEP_TASK_FILE" ]; then
    echo "ERROR: SWEEP_TASK_FILE environment variable is not set!" >&2
    exit 1
fi

if [ ! -f "$SWEEP_TASK_FILE" ] || [ ! -s "$SWEEP_TASK_FILE" ]; then
    echo "ERROR: Task file $SWEEP_TASK_FILE does not exist or is empty!" >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SWEEP_TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Sweep Task File: $SWEEP_TASK_FILE"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

python3 scripts/run_carps_patched.py $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
