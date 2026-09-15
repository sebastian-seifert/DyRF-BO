#!/bin/bash
set -e

echo "=================================================="
echo "Submitting BBOB High-D & Extreme Sweep - BATCH 1 (Tasks 1 to 5000)"
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

mkdir -p results/sweep_bbob_highdim_proximity/logs

START_TASK=1
END_TASK=5000
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
echo "Batch 1 (Tasks 1-5000) successfully scheduled on LUIS cluster!"
echo "=================================================="
