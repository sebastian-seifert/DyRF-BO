#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=epistemic_ood_sweep
#SBATCH --output=results/epistemic_ood_sweep/logs/array_%A_%a.log
#SBATCH --error=results/epistemic_ood_sweep/logs/array_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Environment / Module / Conda Setup
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Constrain single-thread execution per worker to prevent CPU oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_FILE="results/epistemic_ood_sweep_tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing. Generating tasks..."
    python scripts/generate_epistemic_ood_sweep_tasks.py
fi

mkdir -p results/epistemic_ood_sweep/logs

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
