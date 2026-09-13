#!/bin/bash
set -e

echo "=================================================="
echo "Preparing Epistemic OOD Detection Benchmark Sweep"
echo "=================================================="

# 1. Generate Task File if missing
TASK_FILE="results/epistemic_ood_sweep_tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_ood_sweep_tasks.py
    else
        python3 scripts/generate_epistemic_ood_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total tasks in task file: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/epistemic_ood_sweep/logs

# LUIS MaxArraySize limit is typically 200 or 1000. Chunking into 200 tasks per sub-array:
CHUNK_SIZE=200
CONCURRENCY=25

echo "Submitting ${TOTAL_TASKS} tasks in chunks of ${CHUNK_SIZE} (concurrency %${CONCURRENCY})..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} scripts/submit_epistemic_ood_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "All ${TOTAL_TASKS} tasks successfully scheduled on cluster!"
echo "Parse results when finished: python scripts/parse_epistemic_ood_sweep_results.py"
echo "=================================================="
