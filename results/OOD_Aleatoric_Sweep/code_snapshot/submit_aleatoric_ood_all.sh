#!/bin/bash
# Master runner for Aleatoric OOD Sweep on SLURM (splits 2625 tasks into 250-task array chunks)

mkdir -p results/OOD_Aleatoric_Sweep/logs results/OOD_Aleatoric_Sweep/json

TASK_FILE="results/OOD_Aleatoric_Sweep/aleatoric_ood_tasks.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Generating OOD task file..."
    python scripts/generate_aleatoric_ood_tasks.py
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE")
CHUNK_SIZE=250

echo "Submitting ${TOTAL_TASKS} total tasks in chunks of ${CHUNK_SIZE}..."

for ((START=1; START<=TOTAL_TASKS; START+=CHUNK_SIZE)); do
    END=$((START + CHUNK_SIZE - 1))
    if [ $END -gt $TOTAL_TASKS ]; then
        END=$TOTAL_TASKS
    fi
    echo "Submitting array range: ${START}-${END}"
    sbatch --array=${START}-${END} scripts/submit_aleatoric_ood_array.sbatch
done

echo "All OOD chunks submitted successfully!"
