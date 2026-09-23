#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=extrap_uq
#SBATCH --output=results/extrapolation_uq/logs/array_%A_%a.log
#SBATCH --error=results/extrapolation_uq/logs/array_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Environment / Module / Conda Setup
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Constrain single-thread execution per worker to prevent CPU oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_FILE="results/extrapolation_sweep_tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing. Generating tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_extrapolation_sweep_tasks.py
    else
        python3 scripts/generate_extrapolation_sweep_tasks.py
    fi
fi

mkdir -p results/extrapolation_uq/logs

TASK_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running Command: $TASK_CMD"
echo "=================================================="

eval "$TASK_CMD"

EXIT_CODE=$?
echo "=================================================="
echo "Task Index $SLURM_ARRAY_TASK_ID Finished with Exit Code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
