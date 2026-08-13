#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=ood_sweep_task
#SBATCH --output=results/ood_sweep/logs/array_%A_%a.log
#SBATCH --error=results/ood_sweep/logs/array_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


TASK_FILE="results/ood_sweep_tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: ${TASK_FILE} missing." >&2
    exit 1
fi

TASK_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running Command: $TASK_CMD"
echo "=================================================="

eval "$TASK_CMD"

echo "=================================================="
echo "Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
