#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S BBsubset Additive Hybrid Sweep"
echo "=================================================="

TASK_FILE="results/bbsubset_additive_hybrid_tasks.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Generating task file..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_additive_hybrid_tasks.py
    else
        python3 scripts/generate_bbsubset_additive_hybrid_tasks.py
    fi
else
    echo "Using existing task file: ${TASK_FILE}"
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/bbsubset_additive_hybrid/logs

CHUNK_SIZE=200
echo "Submitting ${TOTAL_TASKS} tasks for BBsubset Additive Hybrid Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_bbsubset_additive_hybrid_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "CARP-S BBsubset Additive Hybrid Sweep successfully scheduled on LUIS cluster!"
echo "=================================================="
