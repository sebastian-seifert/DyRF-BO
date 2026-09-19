#!/bin/bash
set -e

TASK_FILE="results/sweep_yahpo_rbv2_ranger_proximity/tasks.txt"
SBATCH_FILE="scripts/submit_sweep_yahpo_rbv2_ranger_proximity_array.sbatch"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
START_TASK=3401
END_TASK=7140
CHUNK_SIZE=200
CONCURRENCY=25

echo "=================================================="
echo "Submitting yahpo_rbv2_ranger - BATCH 2 (Tasks $START_TASK to $END_TASK of $TOTAL_TASKS)"
echo "Chunk Size: $CHUNK_SIZE (LUIS MaxArraySize <= 300, %$CONCURRENCY concurrency)"
echo "=================================================="

for (( start=START_TASK; start<=END_TASK; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $END_TASK ]; then
        end=$END_TASK
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} "$SBATCH_FILE")
    echo "Submitted Chunk (${start}-${end} / ${TOTAL_TASKS}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Batch 2 for yahpo_rbv2_ranger successfully submitted!"
echo "=================================================="
