#!/bin/bash
#SBATCH --job-name=aleatoric_ood
#SBATCH --output=results/OOD_Aleatoric_Sweep/logs/ood_%A_%a.out
#SBATCH --error=results/OOD_Aleatoric_Sweep/logs/ood_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu

mkdir -p results/OOD_Aleatoric_Sweep/logs results/OOD_Aleatoric_Sweep/json

TASK_FILE="results/OOD_Aleatoric_Sweep/aleatoric_ood_tasks.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Task file missing or empty. Generating tasks..."
    python scripts/generate_aleatoric_ood_tasks.py
fi

TASK_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK_CMD" ]; then
    echo "ERROR: Task ID ${SLURM_ARRAY_TASK_ID} has no command in $TASK_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting OOD Aleatoric Task ${SLURM_ARRAY_TASK_ID}"
echo "Command: ${TASK_CMD}"
echo "=========================================="

export PYTHONPATH=.
eval "$TASK_CMD"

EXIT_CODE=$?
echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code ${EXIT_CODE}"
exit $EXIT_CODE
