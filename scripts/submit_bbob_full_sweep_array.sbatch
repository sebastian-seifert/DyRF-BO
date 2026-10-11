#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=bbob_full
#SBATCH --output=results/sweep_bbob_full_comparison/logs/bbob_full_%A_%a.log
#SBATCH --error=results/sweep_bbob_full_comparison/logs/bbob_full_%A_%a.err
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

mkdir -p results/sweep_bbob_full_comparison/logs

# Allow overriding TASK_FILE via environment variable, defaulting to tasks_all.txt
TASK_FILE="${TASK_FILE:-results/sweep_bbob_full_comparison/tasks_all.txt}"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty. Run scripts/generate_bbob_full_sweep_tasks.py first." >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Task File: $TASK_FILE"
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
