#!/bin/bash
set -e

echo "=================================================="
echo "Preparing Aleatoric UQ Masterplan Array Job"
echo "=================================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/generate_aleatoric_masterplan_tasks.py
else
    python3 scripts/generate_aleatoric_masterplan_tasks.py
fi

TASK_FILE="results/aleatoric_masterplan_tasks.txt"
TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/aleatoric_masterplan/logs

CHUNK_SIZE=200
echo "Submitting ${TOTAL_TASKS} tasks for Aleatoric Masterplan Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_aleatoric_masterplan_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Aleatoric Masterplan Sweep successfully scheduled on LUIS cluster!"
echo "=================================================="
