#!/bin/bash
set -e

echo "=================================================="
echo "Submitting BBOB High-D & Extreme Sweep - BATCH 2 (Tasks 5001 to 8640)"
echo "=================================================="

TASK_FILE="results/sweep_bbob_highdim_proximity/tasks.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Generating task file..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbob_highdim_proximity_tasks.py
    else
        python3 scripts/generate_bbob_highdim_proximity_tasks.py
    fi
else
    echo "Using existing task file: ${TASK_FILE}"
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -lt 5001 ]; then
    echo "ERROR: Task file has fewer than 5001 tasks!" >&2
    exit 1
fi

mkdir -p results/sweep_bbob_highdim_proximity/logs

START_TASK=5001
END_TASK=8640
if [ $END_TASK -gt $TOTAL_TASKS ]; then
    END_TASK=$TOTAL_TASKS
fi
CHUNK_SIZE=200

echo "Submitting tasks ${START_TASK} to ${END_TASK} in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize <= 200, %25 concurrency)..."

for (( start=START_TASK; start<=END_TASK; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $END_TASK ]; then
        end=$END_TASK
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_bbob_highdim_proximity_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Batch 2 (Tasks 5001-8640) successfully scheduled on LUIS cluster!"
echo "=================================================="
